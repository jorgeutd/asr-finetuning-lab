"""Auditable evaluation artifacts without machine-specific paths."""

import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

from asr_lab.audio import SAMPLE_RATE, load_audio
from asr_lab.manifest import read_manifest, select_rows, validate_manifest, write_json
from asr_lab.metrics import score
from asr_lab.modeling import load_model, resolve_revision, transcribe


def checkpoint_hash(path: str | Path) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.glob("*.safetensors")):
        digest.update(file.name.encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def evaluate(
    model_id: str,
    manifest: str | Path,
    output: str | Path,
    language: str,
    revision: str = "main",
    adapter: str | None = None,
    split: str = "test",
    max_tokens: int = 225,
    device: str = "auto",
) -> dict:
    import torch

    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("evaluation output exists; use a new directory")
    validation = validate_manifest(manifest)
    rows = select_rows(read_manifest(manifest), split, language)
    revision = resolve_revision(model_id, revision)
    model, processor = load_model(model_id, language, revision, adapter)
    run_path = Path(adapter or model_id).parent / "run.json"
    trained = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else None
    device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    model.to(device)
    model.eval()
    transcribe(model, processor, rows[0]["_audio_path"], language, max_tokens)
    predictions = []
    for row in rows:
        seconds = len(load_audio(row["_audio_path"])) / SAMPLE_RATE
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        text = transcribe(model, processor, row["_audio_path"], language, max_tokens)
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        predictions.append(
            {
                "id": row["id"],
                "reference": row["text"],
                "prediction": text,
                "duration_seconds": seconds,
                "elapsed_seconds": time.perf_counter() - start,
                "group": row.get("speaker_id") or row.get("text_id") or row["id"],
            }
        )
    elapsed = sum(r["elapsed_seconds"] for r in predictions)
    duration = sum(r["duration_seconds"] for r in predictions)
    summary = {
        "schema_version": 1,
        "language": language,
        "split": split,
        "n": len(rows),
        "model": Path(model_id).name if Path(model_id).is_dir() else model_id,
        "revision": revision,
        "base_model": trained["recipe"]["model"] if trained else model_id,
        "base_revision": trained["recipe"]["revision"] if trained else revision,
        "checkpoint_sha256": checkpoint_hash(model_id),
        "adapter_sha256": checkpoint_hash(adapter) if adapter else None,
        "manifest_sha256": validation["manifest_sha256"],
        "metrics": score(
            [p["reference"] for p in predictions], [p["prediction"] for p in predictions]
        ),
        "decoding": {
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": max_tokens,
            "return_timestamps": False,
            "suppress_tokens": model.generation_config.suppress_tokens,
            "begin_suppress_tokens": model.generation_config.begin_suppress_tokens,
        },
        "elapsed_seconds": elapsed,
        "audio_seconds": duration,
        "real_time_factor": elapsed / duration,
        "timing": "One warm-up excluded; per-clip decode includes audio loading and preprocessing.",
        "environment": {
            "python": platform.python_version(),
            "system": platform.system(),
            "device": torch.cuda.get_device_name() if device.startswith("cuda") else "cpu",
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("torch", "transformers", "peft", "numpy", "jiwer")
            },
        },
        "speaker_ids_complete": validation["speaker_ids_complete"],
    }
    write_json(output / "metrics.json", summary)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions),
        encoding="utf-8",
    )
    return summary
