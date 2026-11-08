"""Opt-in regression test for real Whisper Trainer checkpoint selection."""

import json
import os

import numpy as np
import pytest
import soundfile as sf

from asr_lab.config import Recipe
from asr_lab.modeling import load_model, transcribe
from asr_lab.training import train


@pytest.mark.integration
@pytest.mark.skipif(os.getenv("ASR_INTEGRATION") != "1", reason="set ASR_INTEGRATION=1")
@pytest.mark.parametrize("method", ["full", "lora"])
def test_pretrained_training_selects_and_reloads_checkpoint(tmp_path, method):
    rows = []
    for index, split in enumerate(("train", "validation")):
        samples = np.sin(np.arange(16000) * (index + 1) * 0.05).astype(np.float32) * 0.1
        sf.write(tmp_path / f"{split}.wav", samples, 16000)
        rows.append(
            {
                "id": split,
                "audio": f"{split}.wav",
                "text": "Hello world.",
                "language": "en",
                "split": split,
                "speaker_id": split,
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    recipe = Recipe(
        method=method,
        max_steps=1,
        batch_size=1,
        gradient_accumulation_steps=1,
        eval_steps=1,
        max_label_tokens=32,
        gradient_checkpointing=False,
    )
    output = tmp_path / "run"
    result = train(recipe, manifest, output)
    assert result["steps"] == 1
    assert result["best_validation_wer"] is not None
    model_id = recipe.model if method == "lora" else str(output / "model")
    adapter = str(output / "model") if method == "lora" else None
    model, processor = load_model(model_id, "en", adapter=adapter)
    assert isinstance(transcribe(model, processor, str(tmp_path / "validation.wav"), "en", 32), str)
