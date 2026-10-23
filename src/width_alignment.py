from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import TrainerCallback


@dataclass
class AlignmentModule:
    name: str
    adapter: str
    owner: nn.Module
    a_param: nn.Parameter


def _first_adapter(module_dict: nn.ModuleDict) -> str:
    try:
        return next(iter(module_dict.keys()))
    except StopIteration as exc:
        raise ValueError("LoRA module has no adapters") from exc


def _sample_evenly(length: int, count: int, offset: int = 0) -> torch.Tensor:
    if length <= 0 or count <= 0:
        return torch.empty(0, dtype=torch.long)
    count = min(count, length)
    if count == length:
        return torch.arange(length, dtype=torch.long)
    indices = torch.linspace(0, length - 1, steps=count).round().long()
    if offset:
        indices = (indices + offset) % length
    return torch.unique(indices, sorted=True)


def _stable_seed(*parts: object) -> int:
    digest = hashlib.blake2s(
        "|".join(str(part) for part in parts).encode("utf-8")
    ).hexdigest()
    return int(digest[:8], 16)


class LoraWidthAlignmentCallback(TrainerCallback):
    """Measure the minibatch scaling of the processed LoRA A update against its input."""

    def __init__(
        self,
        *,
        model: nn.Module,
        output_dir: str | os.PathLike[str],
        model_name: str,
        seed: int,
        rank: int,
        init_para: str,
        init_std: float,
        lr_a: float,
        lr_ratio: float | None,
        module_pattern: str = "q_proj",
        max_modules: int = 4,
        max_tokens: int = 32,
        log_fractions: list[float] | None = None,
        log_steps: list[int] | None = None,
        every_n_steps: int = 0,
        shuffled_control: bool = True,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.seed = seed
        self.rank = rank
        self.init_para = init_para
        self.init_std = init_std
        self.lr_a = lr_a
        self.lr_ratio = lr_ratio
        self.module_pattern = module_pattern
        self.max_modules = max_modules
        self.max_tokens = max(1, max_tokens)
        self.log_fractions = log_fractions or [0.25, 0.5, 1.0]
        self.explicit_log_steps = {step for step in (log_steps or []) if step > 1}
        self.every_n_steps = max(0, every_n_steps)
        self.shuffled_control = shuffled_control
        self.model_width = int(
            getattr(getattr(model, "config", None), "hidden_size", 0) or 0
        )
        self.modules = self._select_lora_modules(model)
        if self.model_width <= 0 and self.modules:
            self.model_width = int(self.modules[0].a_param.shape[-1])
        self.scheduled_steps: set[int] = set()
        self.logged_steps: set[int] = set()
        self.capture_enabled = False
        self.capture_step = 0
        self.current_attention_mask: torch.Tensor | None = None
        self.captured_inputs: dict[str, torch.Tensor] = {}
        self.pending_step = 0
        self.pre_step_weights: dict[str, torch.Tensor] = {}
        self.pre_step_gradients: dict[str, torch.Tensor] = {}
        self.pre_step_lrs: dict[str, float] = {}
        self.pre_step_weight_decay: dict[str, float] = {}
        self.hook_handles: list[Any] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "width_alignment.jsonl"
        self.jsonl_path.write_text(
            ""
        )  # Each fresh training run owns its diagnostic trace.
        self.config_path = self.output_dir / "width_alignment_config.json"
        self._register_input_hooks()
        self._write_config()

    def _select_lora_modules(self, model: nn.Module) -> list[AlignmentModule]:
        selected: list[AlignmentModule] = []
        match_all_modules = self.module_pattern.strip().lower() in {"", "*", "all"}
        for name, module in model.named_modules():
            if not match_all_modules and self.module_pattern not in name:
                continue
            if not hasattr(module, "lora_A"):
                continue
            lora_a = module.lora_A
            if isinstance(lora_a, nn.ModuleDict):
                adapter = _first_adapter(lora_a)
                a_layer = lora_a[adapter]
            else:
                adapter = "default"
                a_layer = lora_a
            if not hasattr(a_layer, "weight"):
                continue
            selected.append(
                AlignmentModule(
                    name=name,
                    adapter=adapter,
                    owner=module,
                    a_param=a_layer.weight,
                )
            )
        if self.max_modules > 0 and len(selected) > self.max_modules:
            indices = _sample_evenly(len(selected), self.max_modules)
            selected = [selected[index] for index in indices.tolist()]
        return selected

    def _register_input_hooks(self) -> None:
        try:
            self.hook_handles.append(
                self.model.register_forward_pre_hook(
                    self._model_input_hook, with_kwargs=True
                )
            )
        except TypeError:
            pass
        for item in self.modules:
            self.hook_handles.append(
                item.owner.register_forward_pre_hook(self._make_module_hook(item.name))
            )

    def _model_input_hook(self, _module, _args, kwargs):
        if not self.capture_enabled:
            return
        attention_mask = kwargs.get("attention_mask")
        self.current_attention_mask = (
            attention_mask.detach().bool().reshape(-1)
            if torch.is_tensor(attention_mask)
            else None
        )

    def _make_module_hook(self, module_name: str):
        def hook(_module, inputs):
            if not self.capture_enabled or not inputs or not torch.is_tensor(inputs[0]):
                return
            features = inputs[0].detach()
            if features.dim() < 2:
                return
            features = features.reshape(-1, features.shape[-1])
            mask = self.current_attention_mask
            if mask is not None and mask.numel() == features.shape[0]:
                valid_indices = (
                    mask.to(features.device).nonzero(as_tuple=False).flatten()
                )
                if valid_indices.numel() > 0:
                    features = features.index_select(0, valid_indices)
            if features.shape[0] == 0:
                return
            offset = (
                _stable_seed(
                    self.seed, self.capture_step, module_name, features.shape[0]
                )
                % features.shape[0]
            )
            indices = _sample_evenly(features.shape[0], self.max_tokens, offset).to(
                features.device
            )
            sampled = features.index_select(0, indices).float().cpu()
            existing = self.captured_inputs.get(module_name)
            if existing is not None:
                sampled = torch.cat([existing, sampled], dim=0)
                offset = (
                    _stable_seed(self.seed, self.capture_step, module_name, "merge")
                    % sampled.shape[0]
                )
                indices = _sample_evenly(sampled.shape[0], self.max_tokens, offset)
                sampled = sampled.index_select(0, indices)
            self.captured_inputs[module_name] = sampled

        return hook

    def _write_config(self) -> None:
        payload = {
            "model_name": self.model_name,
            "model_width": self.model_width,
            "seed": self.seed,
            "rank": self.rank,
            "init_para": self.init_para,
            "init_std": self.init_std,
            "lr_a": self.lr_a,
            "lr_ratio": self.lr_ratio,
            "module_pattern": self.module_pattern,
            "max_modules": self.max_modules,
            "max_tokens": self.max_tokens,
            "log_fractions": self.log_fractions,
            "explicit_log_steps": sorted(self.explicit_log_steps),
            "every_n_steps": self.every_n_steps,
            "shuffled_control": self.shuffled_control,
            "jsonl_path": str(self.jsonl_path),
            "definitions": {
                "processed_direction": "U_A^t = -(A_{t+1} - A_t + lr_t * weight_decay * A_t) / lr_t",
                "alignment": "dot = <(U_A^t)_{i,:}, Z> for inputs from the minibatches forming the update",
                "normalized_alignment": "|dot| / (||U_A row||_2 ||Z||_2)",
                "shuffled_control": "The coordinates of Z are deterministically permuted before taking the same dot product.",
            },
            "modules": [
                {
                    "name": item.name,
                    "adapter": item.adapter,
                    "a_shape": list(item.a_param.shape),
                }
                for item in self.modules
            ],
        }
        self.config_path.write_text(json.dumps(payload, indent=2))

    def _build_schedule(self, state) -> None:
        self.scheduled_steps = set(self.explicit_log_steps)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        num_epochs = float(getattr(state, "num_train_epochs", 0.0) or 0.0)
        if max_steps <= 1:
            return
        self.scheduled_steps.add(max_steps)
        if num_epochs <= 0:
            return
        steps_per_epoch = max(1, int(round(max_steps / num_epochs)))
        for fraction in self.log_fractions:
            if fraction > 0:
                self.scheduled_steps.add(
                    max(2, min(max_steps, int(round(steps_per_epoch * fraction))))
                )
        for epoch_index in range(1, int(math.ceil(num_epochs)) + 1):
            self.scheduled_steps.add(
                min(max_steps, int(round(steps_per_epoch * epoch_index)))
            )

    def _next_step(self, state) -> int:
        current_step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        next_step = current_step + 1
        return min(next_step, max_steps) if max_steps > 0 else next_step

    def _should_log(self, step: int) -> bool:
        if step <= 1:
            return False
        return step in self.scheduled_steps or (
            self.every_n_steps > 0 and step % self.every_n_steps == 0
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._build_schedule(state)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        step = self._next_step(state)
        self.capture_step = step
        self.capture_enabled = step not in self.logged_steps and self._should_log(step)
        self.current_attention_mask = None
        self.captured_inputs.clear()
        return control

    def _optimizer_metadata(self, optimizer) -> dict[int, tuple[float, float]]:
        metadata: dict[int, tuple[float, float]] = {}
        if optimizer is None:
            return metadata
        for group in optimizer.param_groups:
            learning_rate = float(group.get("lr", 0.0))
            weight_decay = float(group.get("weight_decay", 0.0))
            for parameter in group["params"]:
                metadata[id(parameter)] = (learning_rate, weight_decay)
        return metadata

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = self._next_step(state)
        if not self.capture_enabled or step != self.capture_step:
            return control
        optimizer_metadata = self._optimizer_metadata(kwargs.get("optimizer"))
        self.pending_step = step
        self.pre_step_weights.clear()
        self.pre_step_gradients.clear()
        self.pre_step_lrs.clear()
        self.pre_step_weight_decay.clear()
        for item in self.modules:
            if item.name not in self.captured_inputs:
                continue
            learning_rate, weight_decay = optimizer_metadata.get(
                id(item.a_param), (0.0, 0.0)
            )
            if learning_rate <= 0:
                continue
            self.pre_step_weights[item.name] = (
                item.a_param.detach().float().cpu().clone()
            )
            if item.a_param.grad is not None:
                self.pre_step_gradients[item.name] = (
                    item.a_param.grad.detach().float().cpu().clone()
                )
            self.pre_step_lrs[item.name] = learning_rate
            self.pre_step_weight_decay[item.name] = weight_decay
        return control

    def _direction_stats(
        self,
        direction: torch.Tensor,
        features: torch.Tensor,
        *,
        permutation: torch.Tensor | None = None,
    ) -> dict[str, float]:
        if permutation is not None:
            features = features.index_select(1, permutation)
        dot_products = features.matmul(direction.t())
        absolute_dot_products = dot_products.abs()
        row_norms = direction.norm(dim=1).clamp_min(1e-30)
        feature_norms = features.norm(dim=1).clamp_min(1e-30)
        cosines = dot_products / (feature_norms[:, None] * row_norms[None, :])
        absolute_cosines = cosines.abs()
        return {
            "dot_signed_mean": float(dot_products.mean().item()),
            "dot_abs_mean": float(absolute_dot_products.mean().item()),
            "dot_abs_median": float(absolute_dot_products.median().item()),
            "dot_rms": float(dot_products.square().mean().sqrt().item()),
            "cosine_signed_mean": float(cosines.mean().item()),
            "cosine_abs_mean": float(absolute_cosines.mean().item()),
            "cosine_abs_median": float(absolute_cosines.median().item()),
            "direction_entry_rms": float(direction.square().mean().sqrt().item()),
            "direction_row_l2_mean": float(row_norms.mean().item()),
            "feature_entry_rms": float(features.square().mean().sqrt().item()),
            "feature_l2_mean": float(feature_norms.mean().item()),
        }

    def _module_record(self, item: AlignmentModule) -> dict[str, Any] | None:
        before = self.pre_step_weights.get(item.name)
        features = self.captured_inputs.get(item.name)
        learning_rate = self.pre_step_lrs.get(item.name, 0.0)
        if before is None or features is None or learning_rate <= 0:
            return None
        after = item.a_param.detach().float().cpu()
        weight_decay = self.pre_step_weight_decay.get(item.name, 0.0)
        processed_direction = (
            -(after - before + learning_rate * weight_decay * before) / learning_rate
        )
        if features.shape[-1] != processed_direction.shape[-1]:
            return None
        record: dict[str, Any] = {
            "module": item.name,
            "adapter": item.adapter,
            "input_width": int(features.shape[-1]),
            "n_tokens": int(features.shape[0]),
            "n_lora_rows": int(processed_direction.shape[0]),
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        for key, value in self._direction_stats(processed_direction, features).items():
            record[f"processed_{key}"] = value
        gradient = self.pre_step_gradients.get(item.name)
        if gradient is not None:
            for key, value in self._direction_stats(gradient, features).items():
                record[f"gradient_{key}"] = value
        if self.shuffled_control:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _stable_seed(self.seed, self.pending_step, item.name, "shuffle")
            )
            permutation = torch.randperm(features.shape[-1], generator=generator)
            shuffled_stats = self._direction_stats(
                processed_direction,
                features,
                permutation=permutation,
            )
            for key, value in shuffled_stats.items():
                record[f"shuffled_{key}"] = value
        return record

    def _aggregate(self, records: list[dict[str, Any]]) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        excluded = {
            "module",
            "adapter",
            "input_width",
            "n_tokens",
            "n_lora_rows",
            "learning_rate",
            "weight_decay",
        }
        metric_keys = sorted(set(records[0]).difference(excluded))
        for key in metric_keys:
            values = [float(record[key]) for record in records if key in record]
            if not values:
                continue
            tensor = torch.tensor(values, dtype=torch.float64)
            aggregate[f"{key}_across_modules_mean"] = float(tensor.mean().item())
            aggregate[f"{key}_across_modules_median"] = float(tensor.median().item())
        return aggregate

    def on_optimizer_step(self, args, state, control, **kwargs):
        step = self._next_step(state)
        if self.pending_step != step or step in self.logged_steps:
            return control
        module_records = []
        for item in self.modules:
            record = self._module_record(item)
            if record is not None:
                module_records.append(record)
        payload: dict[str, Any] = {
            "step": step,
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "max_steps": int(getattr(state, "max_steps", 0) or 0),
            "model_name": self.model_name,
            "model_width": self.model_width,
            "seed": self.seed,
            "rank": self.rank,
            "init_para": self.init_para,
            "init_std": self.init_std,
            "lr_a": self.lr_a,
            "lr_ratio": self.lr_ratio,
            "n_modules_selected": len(self.modules),
            "n_modules_logged": len(module_records),
            "modules": module_records,
        }
        if module_records:
            payload["aggregate"] = self._aggregate(module_records)
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.logged_steps.add(step)
        self.pending_step = 0
        self.capture_enabled = False
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        return control
