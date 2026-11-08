"""Fetch a bounded, revision-pinned FLEURS subset."""

import io
import json
from pathlib import Path

import soundfile as sf

from asr_lab.manifest import validate_manifest, write_json

DATASET = "google/fleurs"
CONFIGS = {"en": "en_us", "es": "es_419"}


def fetch_fleurs(
    output: str | Path,
    language: str,
    counts: dict[str, int],
    revision: str = "main",
) -> Path:
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi

    if language not in CONFIGS or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("use en/es and positive split sizes")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be empty; keep completed datasets immutable")
    revision = HfApi().dataset_info(DATASET, revision=revision).sha
    output.mkdir(parents=True, exist_ok=True)
    (output / "clips").mkdir(exist_ok=True)
    rows = []
    for split, count in counts.items():
        dataset = load_dataset(
            DATASET,
            CONFIGS[language],
            split=split,
            revision=revision,
            streaming=True,
        ).cast_column("audio", Audio(decode=False))
        accepted = 0
        for item in dataset:
            raw = item["audio"]
            source = io.BytesIO(raw["bytes"]) if raw.get("bytes") else raw["path"]
            samples, rate = sf.read(source, dtype="float32", always_2d=True)
            text = item["raw_transcription"].strip()
            if not text or not 0.1 <= len(samples) / rate <= 30:
                continue
            source_name = Path(item["path"]).stem
            key = f"{CONFIGS[language]}-{split}-{source_name}"
            audio = f"clips/{key}.flac"
            sf.write(output / audio, samples, rate, subtype="PCM_16")
            rows.append(
                {
                    "id": key,
                    "audio": audio,
                    "text": text,
                    "language": language,
                    "split": split,
                    "speaker_id": None,
                    "text_id": str(item["id"]),
                    "source": DATASET,
                    "source_revision": revision,
                    "license": "CC-BY-4.0",
                }
            )
            accepted += 1
            if accepted == count:
                break
        if accepted < count:
            raise ValueError(f"only {accepted} eligible {split} clips; requested {count}")
    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    write_json(
        output / "provenance.json",
        {
            "dataset": DATASET,
            "revision": revision,
            "config": CONFIGS[language],
            "license": "CC-BY-4.0",
            "source_url": "https://huggingface.co/datasets/google/fleurs",
            "attribution": "Conneau et al. (2022), FLEURS, Google",
            "selection": "First eligible clips per original split, in pinned dataset order.",
            "speaker_note": "Speaker IDs unavailable; speaker isolation is not verified.",
            "validation": validate_manifest(manifest),
        },
    )
    return manifest
