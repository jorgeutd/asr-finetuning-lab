import json

import numpy as np
import pytest
import soundfile as sf

from asr_lab.audio import inspect_audio, load_audio
from asr_lab.config import Recipe, load_recipe
from asr_lab.manifest import read_manifest, validate_manifest
from asr_lab.metrics import normalize, paired_wer_interval, score


def make_manifest(tmp_path, rows=None):
    for i in range(3):
        wave = np.sin(np.arange(8000) * (i + 1) * 0.07).astype(np.float32) * 0.2
        sf.write(tmp_path / f"{i}.wav", wave, 16000)
    rows = rows or [
        {
            "id": str(i),
            "audio": f"{i}.wav",
            "text": "Hola, mundo.",
            "language": "es",
            "split": split,
            "speaker_id": f"speaker-{i}",
        }
        for i, split in enumerate(("train", "validation", "test"))
    ]
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_recipe_rejects_typos_and_invalid_values(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("learn_rate: 0.001")
    with pytest.raises(ValueError, match="unknown recipe"):
        load_recipe(path)
    for kwargs in (
        {"language": "fr"},
        {"max_steps": 0},
        {"batch_size": True},
        {"method": "qlora"},
        {"learning_rate": float("nan")},
    ):
        with pytest.raises(ValueError):
            Recipe(**kwargs)


def test_resampling_and_stereo(tmp_path):
    wave = np.stack([np.ones(8000) * 0.2, np.zeros(8000)], axis=1)
    path = tmp_path / "stereo.wav"
    sf.write(path, wave, 8000, subtype="FLOAT")
    actual = load_audio(path)
    assert len(actual) == 16000
    assert actual.dtype == np.float32
    assert actual[100:-100].mean() == pytest.approx(0.1, abs=0.001)


def test_reject_long_audio(tmp_path):
    path = tmp_path / "long.wav"
    sf.write(path, np.zeros(31 * 16000), 16000)
    with pytest.raises(ValueError, match="segment with labels"):
        inspect_audio(path)


def test_manifest_hash_is_portable_and_content_sensitive(tmp_path):
    path = make_manifest(tmp_path)
    first = validate_manifest(path)
    assert first["splits"] == {"train": 1, "validation": 1, "test": 1}
    assert first["speaker_ids_complete"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    path.write_text("\n".join(json.dumps(r) for r in reversed(rows)))
    assert validate_manifest(path)["manifest_sha256"] == first["manifest_sha256"]
    rows[0]["text"] = "Different transcript"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert validate_manifest(path)["manifest_sha256"] != first["manifest_sha256"]


@pytest.mark.parametrize("case", ["speaker", "audio", "traversal", "duplicate_id"])
def test_manifest_rejects_leakage_and_unsafe_paths(tmp_path, case):
    path = make_manifest(tmp_path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if case == "speaker":
        rows[1]["speaker_id"] = rows[0]["speaker_id"]
    elif case == "audio":
        rows[1]["audio"] = rows[0]["audio"]
    elif case == "traversal":
        rows[1]["audio"] = "../outside.wav"
    else:
        rows[1]["id"] = rows[0]["id"]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError):
        validate_manifest(path)


def test_missing_speaker_ids_reported(tmp_path):
    path = make_manifest(tmp_path)
    rows = [
        {k: v for k, v in row.items() if k not in {"speaker_id", "_audio_path"}}
        for row in read_manifest(path)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    assert not validate_manifest(path)["speaker_ids_complete"]


def test_metrics_use_corpus_counts_and_preserve_accents():
    assert normalize("¡Sí, NIÑO!") == "sí niño"
    metrics = score(["a b c d", "e"], ["a b c d", "x"])
    assert metrics["wer"] == pytest.approx(0.2)
    assert metrics["reference_words"] == 5
    assert score(["sí"], ["si"])["wer"] == 1
    assert score(["one"], ["one two three"])["wer"] == 2
    with pytest.raises(ValueError, match="empty after"):
        score(["..."], ["speech"])


def test_paired_interval_reproducible_and_directional():
    args = (["a b", "c d"], ["a x", "c x"], ["a b", "c d"])
    first = paired_wer_interval(*args)
    assert first == [-0.5, -0.5]
    assert first == paired_wer_interval(*args)
