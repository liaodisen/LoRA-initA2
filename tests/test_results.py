import json

import pytest

from scripts.summarize_results import aggregate, collect_records


def test_seed_aggregation_uses_sample_deviation_and_keeps_settings_separate():
    records = [
        {
            "experiment": "test",
            "method": "AA",
            "ratio": 200,
            "seed": 42,
            "accuracy": 70,
            "loss": 0.5,
        },
        {
            "experiment": "test",
            "method": "AA",
            "ratio": 200,
            "seed": 43,
            "accuracy": 80,
            "loss": 0.3,
        },
        {
            "experiment": "test",
            "method": "AA",
            "ratio": 400,
            "seed": 42,
            "accuracy": 60,
            "loss": 0.7,
        },
    ]
    result = aggregate(records)
    paired = next(row for row in result if row["ratio"] == 200)
    assert paired["accuracy_mean"] == 75
    assert paired["accuracy_std"] == pytest.approx(50**0.5)
    assert paired["n_seeds"] == 2
    assert len(result) == 2
    with pytest.raises(ValueError, match="same cell and seed"):
        aggregate(records + [records[0]])


def test_collector_rejects_partial_benchmark_average(tmp_path):
    (tmp_path / "run.json").write_text(
        json.dumps(
            {
                "experiment": "test",
                "cell": {"kind": "llm", "dataset": "math_10k", "seed": 42},
            }
        )
    )
    (tmp_path / "completed.json").write_text("{}")
    train = tmp_path / "finetuned_result"
    train.mkdir()
    (train / "final_train_metrics.json").write_text(
        json.dumps({"train_loss": 0.5, "last_logged_train_loss": 0.4})
    )
    evaluation = tmp_path / "evaluated_result"
    evaluation.mkdir()
    (evaluation / "accuracy.json").write_text(json.dumps({"gsm8k": {"accuracy": 0.5}}))
    with pytest.raises(ValueError, match="Expected 6"):
        collect_records(tmp_path)
    (evaluation / "accuracy.json").write_text(
        json.dumps(
            {
                name: {"accuracy": index / 10}
                for index, name in enumerate(
                    ["AddSub", "MultiArith", "SingleEq", "SVAMP", "gsm8k", "AQuA"]
                )
            }
        )
    )
    records = collect_records(tmp_path)
    assert records[0]["accuracy"] == 25
    assert records[0]["loss"] == 0.4
    (tmp_path / "completed.json").unlink()
    assert collect_records(tmp_path) == []
