import json

import pytest

from asr_lab.manifest import write_json
from asr_lab.metrics import score
from asr_lab.reporting import compare


def evaluation(path, prediction):
    write_json(
        path / "metrics.json",
        {
            "manifest_sha256": "same",
            "language": "en",
            "split": "test",
            "decoding": {},
            "n": 1,
            "base_model": "same",
            "base_revision": "same",
            "metrics": score(["hello world"], [prediction]),
        },
    )
    (path / "predictions.jsonl").write_text(
        json.dumps(
            {
                "id": "one",
                "reference": "hello world",
                "prediction": prediction,
                "group": "one",
            }
        )
        + "\n"
    )


def test_comparison_uses_paired_predictions(tmp_path):
    evaluation(tmp_path / "base", "hello word")
    evaluation(tmp_path / "tuned", "hello world")
    result = compare(tmp_path / "base", tmp_path / "tuned", tmp_path / "report")
    assert result["wer_delta"] == -0.5
    assert result["wer_delta_ci95"] == [-0.5, -0.5]


def test_reject_mismatched_experiments(tmp_path):
    evaluation(tmp_path / "base", "hello word")
    evaluation(tmp_path / "tuned", "hello world")
    path = tmp_path / "tuned" / "metrics.json"
    metrics = json.loads(path.read_text())
    metrics["manifest_sha256"] = "different"
    write_json(path, metrics)
    with pytest.raises(ValueError, match="manifest_sha256"):
        compare(tmp_path / "base", tmp_path / "tuned", tmp_path / "report")
