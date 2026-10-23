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
class CoherenceModule:
    name: str
    module: nn.Module
    adapter: str
    a0_weight: torch.Tensor
    b_param: nn.Parameter
    row_indices: torch.Tensor


def _first_adapter(module_dict: nn.ModuleDict) -> str:
    try:
        return next(iter(module_dict.keys()))
    except StopIteration as exc:
        raise ValueError("LoRA module has no adapters") from exc


def _stable_offset(name: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    digest = hashlib.blake2s(name.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulo


def _sample_evenly(length: int, count: int, offset: int = 0) -> torch.Tensor:
    if length <= 0:
        return torch.empty(0, dtype=torch.long)
    count = min(max(count, 1), length)
    if count == length:
        return torch.arange(length, dtype=torch.long)
    values = torch.linspace(0, length - 1, steps=count).round().long()
    if offset:
        values = (values + offset) % length
    return torch.unique(values, sorted=True)


def _tensor_scalar(value: Any) -> float:
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


class LoraCoherenceCallback(TrainerCallback):
    """Log rank-contraction/coherence statistics for selected LoRA modules.

    The statistic follows `docs/EXPERIMENTS.md`.
    For each selected LoRA module, a fixed probe batch is run at scheduled
    training steps. The callback captures the module input Z, computes
    u_0 = A_0 Z and u_t = A_t Z, and contracts each against AdamW's
    bias-corrected optimizer direction for B:

        U_B = m_hat / (sqrt(v_hat) + eps)

    The output is intentionally compact: one JSONL record per logged step,
    with aggregate medians and per-module medians. Initial selected A weights
    are saved separately in `initial_lora_A.pt`.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        probe_batch: dict[str, torch.Tensor],
        output_dir: str | os.PathLike[str],
        seed: int,
        rank: int,
        init_para: str,
        init_std: float,
        lr_a: float,
        lr_ratio: float | None,
        max_modules: int = 8,
        max_tokens: int = 64,
        max_rows: int = 64,
        log_fractions: list[float] | None = None,
        log_steps: list[int] | None = None,
        every_n_steps: int = 0,
    ) -> None:
        self.model = model
        self.probe_batch = probe_batch
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.rank = rank
        self.init_para = init_para
        self.init_std = init_std
        self.lr_a = lr_a
        self.lr_ratio = lr_ratio
        self.max_modules = max_modules
        self.max_tokens = max_tokens
        self.max_rows = max_rows
        self.log_fractions = log_fractions or [0.01, 0.05, 0.25, 0.5, 1.0]
        self.explicit_log_steps = {step for step in (log_steps or []) if step > 0}
        self.every_n_steps = max(0, every_n_steps)
        self.logged_steps: set[int] = set()
        self.scheduled_steps: set[int] = set()
        self.modules = self._select_lora_modules(model)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "coherence_stats.jsonl"
        self.jsonl_path.write_text(
            ""
        )  # Each fresh training run owns its diagnostic trace.
        self.config_path = self.output_dir / "coherence_config.json"
        self.initial_a_path = self.output_dir / "initial_lora_A.pt"
        self._write_initial_state()

    def _select_lora_modules(self, model: nn.Module) -> list[CoherenceModule]:
        selected: list[CoherenceModule] = []
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
                continue
            lora_a = module.lora_A
            lora_b = module.lora_B
            if isinstance(lora_a, nn.ModuleDict) and isinstance(lora_b, nn.ModuleDict):
                adapter = _first_adapter(lora_a)
                a_layer = lora_a[adapter]
                b_layer = lora_b[adapter]
            else:
                adapter = "default"
                a_layer = lora_a
                b_layer = lora_b
            if not hasattr(a_layer, "weight") or not hasattr(b_layer, "weight"):
                continue
            b_param = b_layer.weight
            output_rows = b_param.shape[0]
            rows = _sample_evenly(
                output_rows,
                self.max_rows,
                offset=_stable_offset(name, output_rows),
            )
            selected.append(
                CoherenceModule(
                    name=name,
                    module=module,
                    adapter=adapter,
                    a0_weight=a_layer.weight.detach().float().cpu().clone(),
                    b_param=b_param,
                    row_indices=rows,
                )
            )
        if self.max_modules > 0:
            selected = selected[: self.max_modules]
        return selected

    def _write_initial_state(self) -> None:
        torch.save(
            {item.name: item.a0_weight for item in self.modules}, self.initial_a_path
        )
        payload = {
            "seed": self.seed,
            "rank": self.rank,
            "init_para": self.init_para,
            "init_std": self.init_std,
            "lr_a": self.lr_a,
            "lr_ratio": self.lr_ratio,
            "max_modules": self.max_modules,
            "max_tokens": self.max_tokens,
            "max_rows": self.max_rows,
            "log_fractions": self.log_fractions,
            "explicit_log_steps": sorted(self.explicit_log_steps),
            "every_n_steps": self.every_n_steps,
            "jsonl_path": str(self.jsonl_path),
            "initial_lora_A_path": str(self.initial_a_path),
            "modules": [
                {
                    "name": item.name,
                    "adapter": item.adapter,
                    "a_shape": list(item.a0_weight.shape),
                    "b_shape": list(item.b_param.shape),
                    "sampled_rows": item.row_indices.tolist(),
                }
                for item in self.modules
            ],
        }
        self.config_path.write_text(json.dumps(payload, indent=2))

    def _build_schedule(self, state) -> None:
        self.scheduled_steps = set(self.explicit_log_steps)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        num_epochs = float(getattr(state, "num_train_epochs", 0.0) or 0.0)
        if max_steps > 0:
            self.scheduled_steps.add(1)
            self.scheduled_steps.add(max_steps)
        if max_steps > 0 and num_epochs > 0:
            steps_per_epoch = max(1, int(round(max_steps / num_epochs)))
            for fraction in self.log_fractions:
                if fraction <= 0:
                    continue
                self.scheduled_steps.add(
                    max(1, min(max_steps, int(round(steps_per_epoch * fraction))))
                )
            for epoch_idx in range(1, int(math.ceil(num_epochs)) + 1):
                self.scheduled_steps.add(
                    min(max_steps, int(round(steps_per_epoch * epoch_idx)))
                )

    def on_train_begin(self, args, state, control, **kwargs):
        self._build_schedule(state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if step <= 0 or step in self.logged_steps:
            return control
        should_log = step in self.scheduled_steps
        if self.every_n_steps > 0 and step % self.every_n_steps == 0:
            should_log = True
        if should_log:
            optimizer = kwargs.get("optimizer")
            model = kwargs.get("model", self.model)
            self.log_step(model=model, optimizer=optimizer, state=state, step=step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step not in self.logged_steps:
            optimizer = kwargs.get("optimizer")
            model = kwargs.get("model", self.model)
            self.log_step(model=model, optimizer=optimizer, state=state, step=step)
        return control

    def _optimizer_group_for_param(
        self, optimizer, param: nn.Parameter
    ) -> dict[str, Any] | None:
        if optimizer is None:
            return None
        target_id = id(param)
        for group in optimizer.param_groups:
            for group_param in group["params"]:
                if id(group_param) == target_id:
                    return group
        return None

    def _optimizer_direction(
        self, optimizer, item: CoherenceModule
    ) -> torch.Tensor | None:
        if optimizer is None:
            return None
        opt_state = optimizer.state.get(item.b_param)
        if not opt_state or "exp_avg" not in opt_state or "exp_avg_sq" not in opt_state:
            return None
        group = self._optimizer_group_for_param(optimizer, item.b_param)
        betas = group.get("betas", (0.9, 0.999)) if group is not None else (0.9, 0.999)
        eps = group.get("eps", 1e-8) if group is not None else 1e-8
        step = max(1.0, _tensor_scalar(opt_state.get("step", 1.0)))
        beta1, beta2 = float(betas[0]), float(betas[1])
        exp_avg = opt_state["exp_avg"].detach().float()
        exp_avg_sq = opt_state["exp_avg_sq"].detach().float()
        bias_correction1 = 1.0 - beta1**step
        bias_correction2 = 1.0 - beta2**step
        m_hat = exp_avg / bias_correction1
        v_hat = exp_avg_sq / bias_correction2
        return m_hat / (v_hat.sqrt() + float(eps))

    def _probe_device(self, model: nn.Module) -> torch.device:
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _probe_mask(self) -> torch.Tensor | None:
        mask = self.probe_batch.get("attention_mask")
        if mask is None:
            return None
        return mask.detach().bool().reshape(-1).cpu()

    def _move_probe_batch(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            key: value.to(device)
            for key, value in self.probe_batch.items()
            if torch.is_tensor(value)
        }

    def _capture_probe_inputs(self, model: nn.Module) -> dict[str, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        mask = self._probe_mask()
        hooks = []

        def make_hook(module_name: str):
            def hook(_module, inputs):
                if not inputs:
                    return
                tensor = inputs[0]
                if not torch.is_tensor(tensor):
                    return
                if tensor.dim() < 2:
                    return
                flat = tensor.detach().reshape(-1, tensor.shape[-1])
                if mask is not None and mask.numel() == flat.shape[0]:
                    valid = mask.to(flat.device).nonzero(as_tuple=False).flatten()
                    if valid.numel() > 0:
                        flat = flat.index_select(0, valid)
                offset = _stable_offset(module_name, flat.shape[0])
                token_indices = _sample_evenly(
                    flat.shape[0], self.max_tokens, offset=offset
                ).to(flat.device)
                captured[module_name] = (
                    flat.index_select(0, token_indices).float().cpu()
                )

            return hook

        for item in self.modules:
            hooks.append(item.module.register_forward_pre_hook(make_hook(item.name)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(**self._move_probe_batch(self._probe_device(model)))
        finally:
            for handle in hooks:
                handle.remove()
            if was_training:
                model.train()
        return captured

    def _current_a_weight(self, item: CoherenceModule) -> torch.Tensor | None:
        lora_a = item.module.lora_A
        if isinstance(lora_a, nn.ModuleDict):
            layer = lora_a[item.adapter]
        else:
            layer = lora_a
        if not hasattr(layer, "weight"):
            return None
        return layer.weight.detach().float().cpu()

    def _module_stats(
        self,
        *,
        item: CoherenceModule,
        features: torch.Tensor,
        optimizer_direction: torch.Tensor,
    ) -> dict[str, Any] | None:
        if features.shape[-1] != item.a0_weight.shape[-1]:
            return None
        current_a = self._current_a_weight(item)
        if current_a is None:
            return None
        rows = item.row_indices
        if rows.numel() == 0:
            return None
        direction_rows = (
            optimizer_direction.detach().float().cpu().index_select(0, rows)
        )
        u0 = features.matmul(item.a0_weight.t())
        ut = features.matmul(current_a.t())

        init_stats = self._feature_stats(u0, direction_rows)
        current_stats = self._feature_stats(ut, direction_rows)
        if init_stats is None or current_stats is None:
            return None
        return {
            "module": item.name,
            "adapter": item.adapter,
            "rank": int(u0.shape[-1]),
            "n_tokens": int(features.shape[0]),
            "n_rows": int(rows.numel()),
            "sampled_rows": rows.tolist(),
            **{f"init_{key}": value for key, value in init_stats.items()},
            **{f"current_{key}": value for key, value in current_stats.items()},
        }

    def _feature_stats(
        self, features: torch.Tensor, direction_rows: torch.Tensor
    ) -> dict[str, float] | None:
        if features.numel() == 0 or direction_rows.numel() == 0:
            return None
        denom_s = features.pow(2).mean(dim=1).sqrt().clamp_min(1e-12)
        contraction = features.matmul(direction_rows.t())
        abs_contraction = contraction.abs()
        s_values = abs_contraction / denom_s[:, None]
        c_denom = (
            (features[:, None, :].abs() * direction_rows[None, :, :].abs())
            .sum(dim=-1)
            .clamp_min(1e-12)
        )
        c_values = abs_contraction / c_denom
        return {
            "s_median": float(s_values.median().item()),
            "s_mean": float(s_values.mean().item()),
            "c_median": float(c_values.median().item()),
            "c_mean": float(c_values.mean().item()),
            "feature_rms_median": float(denom_s.median().item()),
        }

    def _aggregate(self, module_records: list[dict[str, Any]]) -> dict[str, float]:
        keys = [
            "init_s_median",
            "init_c_median",
            "current_s_median",
            "current_c_median",
        ]
        aggregate: dict[str, float] = {}
        for key in keys:
            values = torch.tensor(
                [float(record[key]) for record in module_records], dtype=torch.float32
            )
            aggregate[f"{key}_across_modules_median"] = float(values.median().item())
            aggregate[f"{key}_across_modules_mean"] = float(values.mean().item())
        return aggregate

    def log_step(self, *, model: nn.Module, optimizer, state, step: int) -> None:
        self.logged_steps.add(step)
        if not self.modules:
            return
        captured = self._capture_probe_inputs(model)
        module_records = []
        for item in self.modules:
            features = captured.get(item.name)
            if features is None:
                continue
            direction = self._optimizer_direction(optimizer, item)
            if direction is None:
                continue
            record = self._module_stats(
                item=item,
                features=features,
                optimizer_direction=direction,
            )
            if record is not None:
                module_records.append(record)
        payload: dict[str, Any] = {
            "step": int(step),
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "max_steps": int(getattr(state, "max_steps", 0) or 0),
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
