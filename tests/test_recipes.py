import pytest

from scripts.reproduce import build_commands, expand_recipe, load_recipes, validate_cell
from scripts.train_toy import ToyLoRA
import torch


def test_all_paper_recipe_commands_are_valid(tmp_path):
    recipes = load_recipes()
    for name, recipe in recipes.items():
        cells = list(expand_recipe(recipe))
        assert cells, name
        assert len({str(c) for c in cells}) == len(cells), name
        for cell in cells:
            validate_cell(cell)
            commands = build_commands(cell, recipe, tmp_path / name, tmp_path / "data")
            assert commands and all(
                isinstance(part, str) for command in commands for part in command
            )
            if cell["kind"] == "llm":
                assert cell["alpha"] / cell["rank"] == cell["multiplier"]
                assert "--cutoff_len" in commands[0]


def test_structured_grid_does_not_duplicate_beta_independent_controls():
    cells = list(expand_recipe(load_recipes()["table4"]))
    assert len(cells) == 33
    assert all(c["dropout"] == 0 for c in cells)


def test_methods_receive_same_context_length_and_budget():
    for name in ["figure3_beta", "figure3_rank", "figure5", "figure8_qlora"]:
        cells = list(expand_recipe(load_recipes()[name]))
        assert {c["cutoff_len"] for c in cells} == {256}
        assert {c["epochs"] for c in cells if c["dataset"] == "math_10k"} == {3}


@pytest.mark.parametrize("method", ["A", "AA"])
def test_zero_backbone_toy_can_take_first_update(method):
    torch.manual_seed(1)
    model = ToyLoRA(width=16, rank=4, init=method)
    loss = torch.nn.functional.cross_entropy(
        model(torch.rand(8, 784)), torch.arange(8) % 10
    )
    loss.backward()
    assert torch.isfinite(model.b.grad).all()
    assert model.b.grad.abs().sum() > 0
