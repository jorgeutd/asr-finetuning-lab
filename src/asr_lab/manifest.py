"""Portable JSONL manifests with split and content checks."""

import hashlib
import json
from pathlib import Path

from asr_lab.audio import inspect_audio, load_audio

SPLITS = ("train", "validation", "test")
REQUIRED = {"id", "audio", "text", "language", "split"}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    rows = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or REQUIRED - row.keys():
            raise ValueError(f"line {line_number}: required fields are {sorted(REQUIRED)}")
        if not isinstance(row["id"], str) or not row["id"] or row["id"] in seen:
            raise ValueError(f"line {line_number}: missing or duplicate id")
        if row["split"] not in SPLITS or row["language"] not in {"en", "es"}:
            raise ValueError(f"line {line_number}: invalid language or split")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError(f"line {line_number}: empty transcript")
        if not isinstance(row["audio"], str) or Path(row["audio"]).is_absolute():
            raise ValueError(f"line {line_number}: audio must be a relative path")
        audio = (path.parent / row["audio"]).resolve()
        if not audio.is_relative_to(path.parent) or not audio.is_file():
            raise ValueError(f"line {line_number}: audio must exist inside the manifest directory")
        seen.add(row["id"])
        rows.append({**row, "_audio_path": str(audio)})
    if not rows:
        raise ValueError("manifest is empty")
    return rows


def validate_manifest(path: str | Path) -> dict:
    rows = read_manifest(path)
    hashes: dict[str, str] = {}
    speakers: dict[str, str] = {}
    texts: dict[tuple, str] = {}
    counts = dict.fromkeys(SPLITS, 0)
    warnings = []
    seconds = 0.0
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda r: r["id"]):
        quality = inspect_audio(row["_audio_path"])
        audio_hash = hashlib.sha256(load_audio(row["_audio_path"]).tobytes()).hexdigest()
        if audio_hash in hashes:
            raise ValueError(f"duplicate audio: {row['id']} and {hashes[audio_hash]}")
        hashes[audio_hash] = row["id"]
        if row.get("speaker_id"):
            speaker = str(row["speaker_id"])
            if speaker in speakers and speakers[speaker] != row["split"]:
                raise ValueError(f"speaker leakage detected at {row['id']}")
            speakers[speaker] = row["split"]
        if row.get("text_id"):
            text_key = (row.get("source"), row["language"], row["text_id"])
            if text_key in texts and texts[text_key] != row["split"]:
                raise ValueError(f"text ID leakage detected at {row['id']}")
            texts[text_key] = row["split"]
        if quality["rms_db"] < -50 or quality["clipped_fraction"] > 0.01:
            warnings.append({"id": row["id"], "reason": "quiet or clipped audio; inspect labels"})
        counts[row["split"]] += 1
        seconds += quality["duration_seconds"]
        digest.update(
            json.dumps(
                {k: v for k, v in row.items() if k not in {"_audio_path", "audio"}},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
        )
        digest.update(audio_hash.encode())
    return {
        "manifest_sha256": digest.hexdigest(),
        "rows": len(rows),
        "splits": counts,
        "duration_seconds": round(seconds, 3),
        "warnings": warnings,
        "speaker_ids_complete": all(bool(row.get("speaker_id")) for row in rows),
    }


def select_rows(rows: list[dict], split: str, language: str) -> list[dict]:
    result = [row for row in rows if row["split"] == split and row["language"] == language]
    if not result:
        raise ValueError(f"no {language} rows in {split}")
    return result
