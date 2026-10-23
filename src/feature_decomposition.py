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
class FeatureDecompositionModule:
    name: str
    module: nn.Module
    adapter: str
    a0_weight: torch.Tensor
    a_param: nn.Parameter
    b_param: nn.Parameter


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


class LoraFeatureDecompositionCallback(TrainerCallback):
    """Measure the exact realized Init[A] noise/signal decomposition of Z_B."""

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
        log_fractions: list[float] | None = None,
        log_steps: list[int] | None = None,
        every_n_steps: int = 100,
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
        self.log_fractions = log_fractions or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        self.explicit_log_steps = {step for step in (log_steps or []) if step > 0}
        self.every_n_steps = max(0, every_n_steps)
        self.logged_steps: set[int] = set()
        self.scheduled_steps: set[int] = set()
        self.modules = self._select_lora_modules(model)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "feature_decomposition.jsonl"
        self.jsonl_path.write_text(
            ""
        )  # Each fresh training run owns its diagnostic trace.
        self.config_path = self.output_dir / "feature_decomposition_config.json"
        self.initial_a_path = self.output_dir / "initial_lora_A.pt"
        self._write_initial_state()

    def _select_lora_modules(
        self, model: nn.Module
    ) -> list[FeatureDecompositionModule]:
        available: list[FeatureDecompositionModule] = []
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
            available.append(
                FeatureDecompositionModule(
                    name=name,
                    module=module,
                    adapter=adapter,
                    a0_weight=a_layer.weight.detach().float().cpu().clone(),
                    a_param=a_layer.weight,
                    b_param=b_layer.weight,
                )
            )
        if self.max_modules > 0 and len(available) > self.max_modules:
            indices = _sample_evenly(len(available), self.max_modules)
            available = [available[index] for index in indices.tolist()]
        return available

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
            "log_fractions": self.log_fractions,
            "explicit_log_steps": sorted(self.explicit_log_steps),
            "every_n_steps": self.every_n_steps,
            "jsonl_path": str(self.jsonl_path),
            "initial_lora_A_path": str(self.initial_a_path),
            "identities": {
                "noise_vector": "T_noise = scaling * B_t * A_0 * Z",
                "signal_vector": "T_signal = scaling * B_t * (A_t - A_0) * Z",
                "feature_vector": "Z_B = T_noise + T_signal = scaling * B_t * A_t * Z",
                "energy": "E_total = E_noise + E_signal + E_cross",
                "cross_energy": "E_cross = 2 * mean(T_noise * T_signal)",
            },
            "modules": [
                {
                    "name": item.name,
                    "adapter": item.adapter,
                    "a_shape": list(item.a_param.shape),
                    "b_shape": list(item.b_param.shape),
                }
                for item in self.modules
            ],
        }
        self.config_path.write_text(json.dumps(payload, indent=2))

    def _build_schedule(self, state) -> None:
        self.scheduled_steps = set(self.explicit_log_steps)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        num_epochs = float(getattr(state, "num_train_epochs", 0.0) or 0.0)
        if max_steps <= 0:
            return
        self.scheduled_steps.update({1, max_steps})
        if num_epochs <= 0:
            return
        steps_per_epoch = max(1, int(round(max_steps / num_epochs)))
        for fraction in self.log_fractions:
            if fraction > 0:
                self.scheduled_steps.add(
                    max(1, min(max_steps, int(round(steps_per_epoch * fraction))))
                )
        for epoch_index in range(1, int(math.ceil(num_epochs)) + 1):
            self.scheduled_steps.add(
                min(max_steps, int(round(steps_per_epoch * epoch_index)))
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
            self.log_step(
                model=kwargs.get("model", self.model),
                state=state,
                step=step,
            )
        return control

    def on_train_end(self, args, state, control, **kwargs):
        step = int(getattr(state, "global_step", 0) or 0)
        if step > 0 and step not in self.logged_steps:
            self.log_step(
                model=kwargs.get("model", self.model),
                state=state,
                step=step,
            )
        return control

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
        handles = []

        def make_hook(module_name: str):
            def hook(_module, inputs):
                if not inputs or not torch.is_tensor(inputs[0]) or inputs[0].dim() < 2:
                    return
                flat = inputs[0].detach().reshape(-1, inputs[0].shape[-1])
                if mask is not None and mask.numel() == flat.shape[0]:
                    valid = mask.to(flat.device).nonzero(as_tuple=False).flatten()
                    if valid.numel() > 0:
                        flat = flat.index_select(0, valid)
                offset = _stable_offset(module_name, flat.shape[0])
                indices = _sample_evenly(
                    flat.shape[0], self.max_tokens, offset=offset
                ).to(flat.device)
                captured[module_name] = flat.index_select(0, indices).float().cpu()

            return hook

        for item in self.modules:
            handles.append(item.module.register_forward_pre_hook(make_hook(item.name)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(**self._move_probe_batch(self._probe_device(model)))
        finally:
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()
        return captured

    def _scaling(self, item: FeatureDecompositionModule) -> float:
        scaling = getattr(item.module, "scaling", 1.0)
        if isinstance(scaling, dict):
            return float(scaling.get(item.adapter, 1.0))
        return float(scaling)

    def _module_stats(
        self,
        item: FeatureDecompositionModule,
        features: torch.Tensor,
    ) -> dict[str, Any] | None:
        current_a = item.a_param.detach().float().cpu()
        current_b = item.b_param.detach().float().cpu()
        if features.shape[-1] != current_a.shape[-1]:
            return None
        scaling = self._scaling(item)
        initial_feature = features.matmul(item.a0_weight.t())
        learned_feature = features.matmul((current_a - item.a0_weight).t())
        noise_vector = initial_feature.matmul(current_b.t()) * scaling
        signal_vector = learned_feature.matmul(current_b.t()) * scaling
        reconstructed = noise_vector + signal_vector
        direct = features.matmul(current_a.t()).matmul(current_b.t()) * scaling

        noise_energy = float(noise_vector.square().mean().item())
        signal_energy = float(signal_vector.square().mean().item())
        cross_energy = float((2.0 * noise_vector * signal_vector).mean().item())
        total_energy = float(direct.square().mean().item())
        residual_energy = float((direct - reconstructed).square().mean().item())
        vector_scale = max(float(direct.square().sum().sqrt().item()), 1e-30)
        reconstruction_relative_error = float(
            (direct - reconstructed).square().sum().sqrt().item() / vector_scale
        )
        noise_norm = float(noise_vector.square().sum().sqrt().item())
        signal_norm = float(signal_vector.square().sum().sqrt().item())
        cosine_denominator = noise_norm * signal_norm
        cosine = (
            float((noise_vector * signal_vector).sum().item() / cosine_denominator)
            if cosine_denominator > 0
            else None
        )
        fractions = {}
        if total_energy > 1e-30:
            fractions = {
                "noise_fraction": noise_energy / total_energy,
                "signal_fraction": signal_energy / total_energy,
                "cross_fraction": cross_energy / total_energy,
            }
        return {
            "module": item.name,
            "adapter": item.adapter,
            "scaling": scaling,
            "n_tokens": int(features.shape[0]),
            "noise_rms": math.sqrt(max(noise_energy, 0.0)),
            "signal_rms": math.sqrt(max(signal_energy, 0.0)),
            "z_b_rms": math.sqrt(max(total_energy, 0.0)),
            "noise_energy": noise_energy,
            "signal_energy": signal_energy,
            "cross_energy": cross_energy,
            "total_energy": total_energy,
            "residual_energy": residual_energy,
            "reconstruction_relative_error": reconstruction_relative_error,
            "noise_signal_cosine": cosine,
            "initial_a_l2": float(item.a0_weight.norm().item()),
            "learned_a_delta_l2": float((current_a - item.a0_weight).norm().item()),
            **fractions,
        }

    def _aggregate(self, records: list[dict[str, Any]]) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        energy_keys = ["noise_energy", "signal_energy", "cross_energy", "total_energy"]
        for key in energy_keys:
            values = torch.tensor(
                [float(record[key]) for record in records], dtype=torch.float64
            )
            aggregate[f"{key}_across_modules_mean"] = float(values.mean().item())
            aggregate[f"{key}_across_modules_median"] = float(values.median().item())
        noise_energy = aggregate["noise_energy_across_modules_mean"]
        signal_energy = aggregate["signal_energy_across_modules_mean"]
        cross_energy = aggregate["cross_energy_across_modules_mean"]
        total_energy = aggregate["total_energy_across_modules_mean"]
        if total_energy > 1e-30:
            aggregate.update(
                {
                    "noise_fraction": noise_energy / total_energy,
                    "signal_fraction": signal_energy / total_energy,
                    "cross_fraction": cross_energy / total_energy,
                    "fraction_sum": (noise_energy + signal_energy + cross_energy)
                    / total_energy,
                }
            )
        for key in [
            "reconstruction_relative_error",
            "noise_signal_cosine",
            "learned_a_delta_l2",
        ]:
            values = [
                float(record[key]) for record in records if record.get(key) is not None
            ]
            if values:
                tensor = torch.tensor(values, dtype=torch.float64)
                aggregate[f"{key}_across_modules_mean"] = float(tensor.mean().item())
                aggregate[f"{key}_across_modules_median"] = float(
                    tensor.median().item()
                )
        return aggregate

    def log_step(self, *, model: nn.Module, state, step: int) -> None:
        self.logged_steps.add(step)
        if not self.modules:
            return
        captured = self._capture_probe_inputs(model)
        module_records = []
        for item in self.modules:
            features = captured.get(item.name)
            if features is None:
                continue
            record = self._module_stats(item, features)
            if record is not None:
                module_records.append(record)
        payload: dict[str, Any] = {
            "step": step,
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
