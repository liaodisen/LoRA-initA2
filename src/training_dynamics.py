from __future__ import annotations

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
class DynamicsModule:
    name: str
    adapter: str
    owner: nn.Module
    a_layer: nn.Module
    b_layer: nn.Module
    a_param: nn.Parameter
    b_param: nn.Parameter


def _first_adapter(module_dict: nn.ModuleDict) -> str:
    try:
        return next(iter(module_dict.keys()))
    except StopIteration as exc:
        raise ValueError("LoRA module has no adapters") from exc


def _rms(value: Any) -> float | None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    tensor = value.detach().float()
    return float(tensor.square().mean().sqrt().item())


def _l2(value: Any) -> float | None:
    if not torch.is_tensor(value) or value.numel() == 0:
        return None
    return float(value.detach().float().norm().item())


class LoraTrainingDynamicsCallback(TrainerCallback):
    """Log LoRA activation, parameter, and pre-optimizer gradient magnitudes."""

    def __init__(
        self,
        *,
        model: nn.Module,
        output_dir: str | os.PathLike[str],
        seed: int,
        rank: int,
        init_para: str,
        init_std: float,
        lr_a: float,
        lr_ratio: float | None,
        max_modules: int = 8,
        log_fractions: list[float] | None = None,
        log_steps: list[int] | None = None,
        every_n_steps: int = 10,
    ) -> None:
        self.model = model
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.rank = rank
        self.init_para = init_para
        self.init_std = init_std
        self.lr_a = lr_a
        self.lr_ratio = lr_ratio
        self.max_modules = max_modules
        self.log_fractions = log_fractions or [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
        self.explicit_log_steps = {step for step in (log_steps or []) if step > 0}
        self.every_n_steps = max(0, every_n_steps)
        self.modules = self._select_lora_modules(model)
        self.latest_activations: dict[str, dict[str, float | None]] = {}
        self.scheduled_steps: set[int] = set()
        self.logged_steps: set[int] = set()
        self.hook_handles: list[Any] = []
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.output_dir / "training_dynamics.jsonl"
        self.jsonl_path.write_text(
            ""
        )  # Each fresh training run owns its diagnostic trace.
        self.config_path = self.output_dir / "training_dynamics_config.json"
        self._register_activation_hooks()
        self._write_config()

    def _select_lora_modules(self, model: nn.Module) -> list[DynamicsModule]:
        selected: list[DynamicsModule] = []
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
            selected.append(
                DynamicsModule(
                    name=name,
                    adapter=adapter,
                    owner=module,
                    a_layer=a_layer,
                    b_layer=b_layer,
                    a_param=a_layer.weight,
                    b_param=b_layer.weight,
                )
            )
        if self.max_modules > 0 and len(selected) > self.max_modules:
            indices = (
                torch.linspace(0, len(selected) - 1, steps=self.max_modules)
                .round()
                .long()
                .tolist()
            )
            selected = [selected[index] for index in dict.fromkeys(indices)]
        return selected

    def _register_activation_hooks(self) -> None:
        for item in self.modules:
            self.hook_handles.append(
                item.a_layer.register_forward_hook(
                    self._make_hook(item.name, "z_a_rms")
                )
            )
            self.hook_handles.append(
                item.b_layer.register_forward_hook(
                    self._make_hook(item.name, "z_b_rms")
                )
            )

    def _make_hook(self, module_name: str, key: str):
        def hook(module, _inputs, output):
            if not module.training:
                return
            self.latest_activations.setdefault(module_name, {})[key] = _rms(output)

        return hook

    def _write_config(self) -> None:
        payload = {
            "seed": self.seed,
            "rank": self.rank,
            "init_para": self.init_para,
            "init_std": self.init_std,
            "lr_a": self.lr_a,
            "lr_ratio": self.lr_ratio,
            "max_modules": self.max_modules,
            "log_fractions": self.log_fractions,
            "explicit_log_steps": sorted(self.explicit_log_steps),
            "every_n_steps": self.every_n_steps,
            "jsonl_path": str(self.jsonl_path),
            "definitions": {
                "z_a_rms": "RMS of the output of the LoRA A projection on the current training microbatch.",
                "z_b_rms": "RMS of the output of the LoRA B projection before the LoRA scaling factor.",
                "lora_delta_rms": "z_b_rms multiplied by the active LoRA scaling factor alpha/r.",
                "grad_a_l2": "Frobenius norm of the accumulated LoRA A gradient immediately before the optimizer update.",
                "grad_b_l2": "Frobenius norm of the accumulated LoRA B gradient immediately before the optimizer update.",
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

    def _next_step(self, state) -> int:
        current_step = int(getattr(state, "global_step", 0) or 0)
        max_steps = int(getattr(state, "max_steps", 0) or 0)
        next_step = current_step + 1
        return min(next_step, max_steps) if max_steps > 0 else next_step

    def _should_log(self, step: int) -> bool:
        return step in self.scheduled_steps or (
            self.every_n_steps > 0 and step % self.every_n_steps == 0
        )

    def _scaling(self, item: DynamicsModule) -> float:
        scaling = getattr(item.owner, "scaling", 1.0)
        if isinstance(scaling, dict):
            return float(scaling.get(item.adapter, 1.0))
        return float(scaling)

    def _module_record(self, item: DynamicsModule) -> dict[str, Any]:
        activation = self.latest_activations.get(item.name, {})
        z_b_rms = activation.get("z_b_rms")
        scaling = self._scaling(item)
        return {
            "module": item.name,
            "adapter": item.adapter,
            "scaling": scaling,
            "z_a_rms": activation.get("z_a_rms"),
            "z_b_rms": z_b_rms,
            "lora_delta_rms": z_b_rms * scaling if z_b_rms is not None else None,
            "grad_a_l2": _l2(item.a_param.grad),
            "grad_b_l2": _l2(item.b_param.grad),
            "weight_a_l2": _l2(item.a_param),
            "weight_b_l2": _l2(item.b_param),
        }

    def _aggregate(self, module_records: list[dict[str, Any]]) -> dict[str, float]:
        aggregate: dict[str, float] = {}
        for key in [
            "z_a_rms",
            "z_b_rms",
            "lora_delta_rms",
            "grad_a_l2",
            "grad_b_l2",
            "weight_a_l2",
            "weight_b_l2",
        ]:
            values = [
                float(record[key])
                for record in module_records
                if record[key] is not None
            ]
            if values:
                tensor = torch.tensor(values, dtype=torch.float32)
                aggregate[f"{key}_across_modules_mean"] = float(tensor.mean().item())
                aggregate[f"{key}_across_modules_median"] = float(
                    tensor.median().item()
                )
        return aggregate

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        step = self._next_step(state)
        if step <= 0 or step in self.logged_steps or not self._should_log(step):
            return control
        module_records = [self._module_record(item) for item in self.modules]
        payload = {
            "step": step,
            "epoch": float(getattr(state, "epoch", 0.0) or 0.0),
            "max_steps": int(getattr(state, "max_steps", 0) or 0),
            "seed": self.seed,
            "rank": self.rank,
            "init_para": self.init_para,
            "init_std": self.init_std,
            "lr_a": self.lr_a,
            "lr_ratio": self.lr_ratio,
            "n_modules_logged": len(module_records),
            "aggregate": self._aggregate(module_records),
            "modules": module_records,
        }
        with self.jsonl_path.open("a") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.logged_steps.add(step)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        return control
