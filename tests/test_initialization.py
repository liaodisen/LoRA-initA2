import copy

import pytest
import torch
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from initialization import initialize_lora, minus_init_lora, rescale_lora_a_to_a2
from trainer import create_loraplus_optimizer


def tiny_model():
    torch.manual_seed(10)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
    )


@pytest.mark.parametrize("method", ["AB_reset", "pissa", "pissa_a2", "olora"])
def test_saved_adapter_replays_same_predictions(tmp_path, method):
    base = tiny_model()
    pristine = copy.deepcopy(base)
    peft_init = (
        "pissa" if method == "pissa_a2" else True if method == "AB_reset" else method
    )
    model = get_peft_model(
        base,
        LoraConfig(
            task_type="CAUSAL_LM",
            r=4,
            lora_alpha=8,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0,
            init_lora_weights=peft_init,
        ),
    )
    if method == "AB_reset":
        initialize_lora(model, init_para=method, normal=False)
    elif method == "pissa_a2":
        rescale_lora_a_to_a2(model, beta=0.5, preserve_product=True)
    model.save_pretrained(tmp_path / "init_ckpt")
    with torch.no_grad():
        for name, weight in model.named_parameters():
            if "lora_" in name:
                weight.add_(torch.randn_like(weight) * 0.01)
    model.eval()
    inputs = torch.tensor([[2, 5, 7, 3]])
    expected = model(inputs).logits.detach()
    model.save_pretrained(tmp_path)
    restored_base = minus_init_lora(
        pristine,
        "AB_reset" if method == "AB_reset" else "A",
        tmp_path,
        init_method=method,
    )
    config = PeftConfig.from_pretrained(tmp_path)
    config.init_lora_weights = False
    restored = PeftModel.from_pretrained(
        restored_base, tmp_path, config=config, torch_device="cpu"
    ).eval()
    torch.testing.assert_close(restored(inputs).logits, expected, atol=2e-6, rtol=2e-5)


def test_variance_and_zero_initial_output():
    base = tiny_model()
    model = get_peft_model(
        base, LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"])
    )
    initialize_lora(model, init_para="AA", lora_A_std=2, lora_A_scale=1, normal=False)
    for layer in model.modules():
        if hasattr(layer, "lora_A"):
            a, b = layer.lora_A["default"].weight, layer.lora_B["default"].weight
            assert a.abs().max() <= 3**0.5 * 2 / a.shape[1]
            assert torch.count_nonzero(b) == 0


def test_optimizer_learning_rates_and_parameter_coverage():
    model = get_peft_model(
        tiny_model(), LoraConfig(r=4, lora_alpha=4, target_modules=["q_proj"])
    )
    optimizer = create_loraplus_optimizer(model, torch.optim.AdamW, {"lr": 1e-5}, 200)
    assigned = [p for group in optimizer.param_groups for p in group["params"]]
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert len(assigned) == len({id(p) for p in assigned}) == len(trainable)
    rates = {
        id(p): group["lr"] for group in optimizer.param_groups for p in group["params"]
    }
    for name, p in model.named_parameters():
        if p.requires_grad:
            assert rates[id(p)] == pytest.approx(2e-3 if "lora_B" in name else 1e-5)


def test_pissa_gauge_preserves_product():
    model = get_peft_model(
        tiny_model(),
        LoraConfig(
            r=4, lora_alpha=4, target_modules=["q_proj"], init_lora_weights="pissa"
        ),
    )
    before = {
        name: module.lora_B["default"].weight.detach().clone()
        @ module.lora_A["default"].weight.detach().clone()
        for name, module in model.named_modules()
        if hasattr(module, "lora_A")
    }
    records = rescale_lora_a_to_a2(model, beta=0.5, preserve_product=True)
    assert records
    for name, module in model.named_modules():
        if hasattr(module, "lora_A"):
            a = module.lora_A["default"].weight
            assert a.square().mean().sqrt().item() == pytest.approx(
                0.5 / a.shape[1], rel=1e-5
            )
            torch.testing.assert_close(
                module.lora_B["default"].weight @ a, before[name], atol=1e-7, rtol=1e-5
            )
