import math
import os

import torch
import torch.nn as nn


def _normalise_init_mode(init_para):
    return (init_para or "").lower()


def _init_weight(weight, std, normal):
    if normal:
        nn.init.normal_(weight, mean=0.0, std=std)
    else:
        bound = std * math.sqrt(3)
        nn.init.uniform_(weight, a=-bound, b=bound)


def _scaling_for_adapter(module, adapter):
    if hasattr(module, "scaling"):
        scaling = module.scaling
        if isinstance(scaling, dict):
            return scaling.get(adapter, 1.0)
        return scaling

    if hasattr(module, "lora_alpha") and hasattr(module, "r"):
        alpha = (
            module.lora_alpha.get(adapter, 1.0)
            if isinstance(module.lora_alpha, dict)
            else module.lora_alpha
        )
        rank = module.r.get(adapter, 1) if isinstance(module.r, dict) else module.r
        return float(alpha) / rank if rank else 1.0

    return 1.0


def _subtract_adapter_from_base(module, A, B, scaling):
    with torch.no_grad():
        effective = B.weight @ A.weight
        if getattr(module, "fan_in_fan_out", False):
            effective = effective.t()
        module.weight.data -= effective.to(module.weight.dtype) * float(scaling)


def _iter_lora_pairs(model):
    """Yield the trainable A/B pair for every LoRA adapter in ``model``."""
    for module_name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not hasattr(module, "lora_B"):
            continue

        if isinstance(module.lora_A, nn.ModuleDict) and isinstance(
            module.lora_B, nn.ModuleDict
        ):
            for adapter_name, A in module.lora_A.items():
                if adapter_name not in module.lora_B:
                    continue
                B = module.lora_B[adapter_name]
                yield module_name, module, adapter_name, A, B
        elif hasattr(module.lora_A, "weight") and hasattr(module.lora_B, "weight"):
            yield module_name, module, "default", module.lora_A, module.lora_B


def _weight_rms(weight):
    return float(weight.detach().float().square().mean().sqrt().item())


def _product_relative_error(B_before, A_before, B_after, A_after):
    """Compute ||B1 A1 - B0 A0||_F / ||B0 A0||_F using rank-size Grams."""
    B_before = B_before.float()
    A_before = A_before.float()
    B_after = B_after.detach().float()
    A_after = A_after.detach().float()
    before_norm_sq = torch.trace((B_before.t() @ B_before) @ (A_before @ A_before.t()))
    after_norm_sq = torch.trace((B_after.t() @ B_after) @ (A_after @ A_after.t()))
    cross = torch.trace((B_before.t() @ B_after) @ (A_after @ A_before.t()))
    diff_norm_sq = (before_norm_sq + after_norm_sq - 2 * cross).clamp_min(0)
    denominator = before_norm_sq.clamp_min(torch.finfo(torch.float32).tiny)
    return float((diff_norm_sq / denominator).sqrt().item())


def summarize_lora_initialization(model):
    """Return lightweight per-adapter scale diagnostics after initialization."""
    records = []
    for module_name, module, adapter_name, A, B in _iter_lora_pairs(model):
        records.append(
            {
                "module": module_name,
                "adapter": adapter_name,
                "in_features": int(A.weight.shape[-1]),
                "out_features": int(B.weight.shape[0]),
                "rank": int(A.weight.shape[0]),
                "scaling": float(_scaling_for_adapter(module, adapter_name)),
                "a_rms": _weight_rms(A.weight),
                "b_rms": _weight_rms(B.weight),
            }
        )
    return records


def initialize_orthogonal_lora_a2(model, beta=1.0):
    """Initialize a row-orthogonal A with RMS beta / n and set B to zero.

    If Q has orthonormal rows, its entrywise RMS is exactly 1/sqrt(n).
    Multiplying Q by beta/sqrt(n) therefore gives the Init[A^2] target
    RMS beta/n while changing the distribution and removing Gaussian tails.
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    records = []
    for module_name, _, adapter_name, A, B in _iter_lora_pairs(model):
        rank, in_features = A.weight.shape
        if rank > in_features:
            raise ValueError(
                "orthogonal_a2 requires rank <= in_features, "
                f"got rank={rank}, in_features={in_features} for {module_name}"
            )

        with torch.no_grad():
            # QR is more robust in fp32; copy the result back to the adapter dtype.
            basis = torch.randn(
                in_features,
                rank,
                device=A.weight.device,
                dtype=torch.float32,
            )
            q, _ = torch.linalg.qr(basis, mode="reduced")
            scale = beta / math.sqrt(in_features)
            A.weight.copy_((q.t() * scale).to(dtype=A.weight.dtype))
            B.weight.zero_()
            if getattr(B, "bias", None) is not None:
                B.bias.zero_()

        records.append(
            {
                "module": module_name,
                "adapter": adapter_name,
                "beta": beta,
                "target_a_rms": beta / in_features,
                "a_rms": _weight_rms(A.weight),
                "b_rms": _weight_rms(B.weight),
            }
        )
    return records


def rescale_lora_a_to_a2(model, beta=1.0, preserve_product=False):
    """Rescale each A to RMS beta / n.

    ``preserve_product=False`` is intended for EVA, whose B is zero.
    ``preserve_product=True`` applies the PiSSA gauge transformation
    A <- c A and B <- B / c, preserving B @ A and therefore the initialized
    model function while suppressing A's entrywise scale.
    """
    beta = float(beta)
    if beta <= 0:
        raise ValueError(f"beta must be positive, got {beta}")

    records = []
    for module_name, _, adapter_name, A, B in _iter_lora_pairs(model):
        in_features = int(A.weight.shape[-1])
        target_rms = beta / in_features
        before_a_rms = _weight_rms(A.weight)
        before_b_rms = _weight_rms(B.weight)
        if before_a_rms == 0:
            raise ValueError(f"Cannot rescale a zero LoRA A matrix in {module_name}")
        if not preserve_product and before_b_rms != 0:
            raise ValueError(
                "A-only rescaling requires B=0, but "
                f"{module_name}.{adapter_name} has B RMS {before_b_rms:.6g}"
            )

        A_before = A.weight.detach().clone() if preserve_product else None
        B_before = B.weight.detach().clone() if preserve_product else None
        gauge_scale = target_rms / before_a_rms
        with torch.no_grad():
            A.weight.mul_(gauge_scale)
            if preserve_product:
                B.weight.div_(gauge_scale)

        product_relative_error = (
            _product_relative_error(
                B_before,
                A_before,
                B.weight,
                A.weight,
            )
            if preserve_product
            else None
        )
        records.append(
            {
                "module": module_name,
                "adapter": adapter_name,
                "beta": beta,
                "target_a_rms": target_rms,
                "gauge_scale": gauge_scale,
                "preserve_product": bool(preserve_product),
                "product_relative_error": product_relative_error,
                "a_rms_before": before_a_rms,
                "b_rms_before": before_b_rms,
                "a_rms": _weight_rms(A.weight),
                "b_rms": _weight_rms(B.weight),
            }
        )
    return records


def initialize_lora(
    model,
    init_para="A",
    lora_A_std=1.0,
    lora_B_std=1.0,
    lora_A_scale=0.5,
    lora_B_scale=0.5,
    normal=True,
    freeze_A=False,
    freeze_B=False,
):
    """
    init_para:
      "A"        -> init A only, zero B
      "B"        -> init B only, zero A
      "AB"       -> init A and B
      "...reset" -> subtract scaling * (B @ A) from the base weight (per adapter)

    `reset` is enabled by substring match, so `AB_reset` is the
    "init both A and B, then subtract the initialized adapter from W" mode.
    """
    init_mode = _normalise_init_mode(init_para)
    init_ab = "ab" in init_mode
    init_a_only = init_mode.startswith("a") and not init_ab
    init_b_only = init_mode.startswith("b") and not init_ab
    reset_base = "reset" in init_mode
    if not (init_ab or init_a_only or init_b_only):
        raise ValueError(f"Unknown custom initialization mode: {init_para!r}")
    if lora_A_std < 0 or lora_B_std < 0:
        raise ValueError("Initialization standard deviations must be nonnegative")

    for module in model.modules():
        if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            if isinstance(module.lora_A, nn.ModuleDict) and isinstance(
                module.lora_B, nn.ModuleDict
            ):
                adapter_names = list(module.lora_A.keys())
                for adapter in adapter_names:
                    A = module.lora_A[adapter]  # nn.Linear(in, r) (bias=False)
                    B = module.lora_B[adapter]  # nn.Linear(r, out) (bias=False)

                    # Infer dims safely
                    in_features = getattr(
                        module, "in_features", getattr(A, "in_features", None)
                    )
                    out_features = getattr(
                        module, "out_features", getattr(B, "out_features", None)
                    )
                    if in_features is None or out_features is None:
                        in_features = A.weight.shape[-1]
                        out_features = B.weight.shape[0]

                    std_A = float(lora_A_std) / (in_features**lora_A_scale)
                    std_B = float(lora_B_std) / (out_features**lora_B_scale)
                    print(
                        f"std_A: {std_A}, std_B: {std_B}, "
                        f"in_features: {in_features}, out_features: {out_features}"
                    )

                    if init_ab:
                        _init_weight(A.weight, std_A, normal)
                        _init_weight(B.weight, std_B, normal)
                    elif init_a_only:
                        _init_weight(A.weight, std_A, normal)
                        nn.init.zeros_(B.weight)
                    elif init_b_only:
                        _init_weight(B.weight, std_B, normal)
                        nn.init.zeros_(A.weight)

                    if freeze_A:
                        A.weight.requires_grad = False
                    if freeze_B:
                        B.weight.requires_grad = False

                    if reset_base:
                        _subtract_adapter_from_base(
                            module, A, B, _scaling_for_adapter(module, adapter)
                        )
            else:
                A = module.lora_A
                B = getattr(module, "lora_B", None)

                in_features = getattr(
                    module,
                    "in_features",
                    getattr(A, "in_features", A.weight.shape[-1]),
                )
                out_features = getattr(
                    module,
                    "out_features",
                    getattr(
                        B, "out_features", B.weight.shape[0] if B is not None else None
                    ),
                )
                bound_A = float(lora_A_std) / (in_features**lora_A_scale)
                if init_ab or init_a_only:
                    _init_weight(A.weight, bound_A, normal)
                    if init_a_only and B is not None:
                        nn.init.zeros_(B.weight)
                if init_ab and B is not None:
                    bound_B = float(lora_B_std) / (out_features**lora_B_scale)
                    _init_weight(B.weight, bound_B, normal)
                elif init_b_only and B is not None:
                    bound_B = float(lora_B_std) / (out_features**lora_B_scale)
                    _init_weight(B.weight, bound_B, normal)
                    nn.init.zeros_(A.weight)
                if reset_base and B is not None:
                    scaling = getattr(module, "scaling", 1.0)
                    _subtract_adapter_from_base(module, A, B, scaling)
                if freeze_A:
                    A.weight.requires_grad_(False)
                if freeze_B and B is not None:
                    B.weight.requires_grad_(False)


def minus_init_lora(model, init_para="A", output_dir=None, init_method="Kaiming"):
    """Replay reset-style initialization during evaluation.

    When `init_para` contains `reset`, this reloads the saved `init_ckpt`
    adapter, subtracts its initialized LoRA update from the base model, and
    then the caller can load the finetuned adapter weights on top.
    """
    method = (init_method or "").lower().replace("-", "_")
    if "reset" not in _normalise_init_mode(init_para) and not (
        method.startswith("pissa") or method == "olora"
    ):
        return model
    from peft import PeftConfig, PeftModel

    initial_path = os.path.join(output_dir, "init_ckpt")
    if not os.path.isfile(os.path.join(initial_path, "adapter_config.json")):
        raise FileNotFoundError(f"Initialization replay requires {initial_path}")
    config = PeftConfig.from_pretrained(initial_path)
    config.init_lora_weights = False
    wrapped = PeftModel.from_pretrained(
        model, initial_path, config=config, torch_device="cpu"
    )
    for _, module, adapter, A, B in _iter_lora_pairs(wrapped):
        _subtract_adapter_from_base(module, A, B, _scaling_for_adapter(module, adapter))
    base = wrapped.unload()
    if hasattr(base, "peft_config"):
        delattr(base, "peft_config")
    return base
