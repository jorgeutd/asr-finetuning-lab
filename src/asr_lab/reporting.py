"""Compare only evaluations with matching data and decoding settings."""

import json
from pathlib import Path

from asr_lab.manifest import write_json
from asr_lab.metrics import paired_wer_interval, score


def compare(baseline: str | Path, candidate: str | Path, output: str | Path) -> dict:
    baseline, candidate, output = Path(baseline), Path(candidate), Path(output)
    a = json.loads((baseline / "metrics.json").read_text(encoding="utf-8"))
    b = json.loads((candidate / "metrics.json").read_text(encoding="utf-8"))
    for key in (
        "manifest_sha256",
        "language",
        "split",
        "decoding",
        "n",
        "base_model",
        "base_revision",
    ):
        if a[key] != b[key]:
            raise ValueError(f"incomparable evaluations: {key} differs")
    if a["metrics"]["normalizer"] != b["metrics"]["normalizer"]:
        raise ValueError("incomparable normalizers")

    def read_predictions(directory):
        rows = [
            json.loads(line)
            for line in (directory / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate prediction IDs")
        return {row["id"]: row for row in rows}

    left, right = read_predictions(baseline), read_predictions(candidate)
    if left.keys() != right.keys() or len(left) != a["n"]:
        raise ValueError("prediction IDs do not match")
    ids = sorted(left)
    if any(
        left[k]["reference"] != right[k]["reference"] or left[k]["group"] != right[k]["group"]
        for k in ids
    ):
        raise ValueError("references or bootstrap groups differ")
    refs = [left[k]["reference"] for k in ids]
    base = [left[k]["prediction"] for k in ids]
    tuned = [right[k]["prediction"] for k in ids]
    base_score, tuned_score = score(refs, base), score(refs, tuned)
    for recorded, calculated in ((a["metrics"], base_score), (b["metrics"], tuned_score)):
        if abs(recorded["wer"] - calculated["wer"]) > 1e-10:
            raise ValueError("saved metrics do not match predictions")
    interval = paired_wer_interval(refs, base, tuned, [left[k]["group"] for k in ids])
    cer_interval = paired_wer_interval(
        refs, base, tuned, [left[k]["group"] for k in ids], metric="cer"
    )
    result = {
        "language": a["language"],
        "n": len(ids),
        "baseline_wer": base_score["wer"],
        "candidate_wer": tuned_score["wer"],
        "wer_delta": tuned_score["wer"] - base_score["wer"],
        "wer_delta_ci95": interval,
        "baseline_cer": base_score["cer"],
        "candidate_cer": tuned_score["cer"],
        "cer_delta": tuned_score["cer"] - base_score["cer"],
        "cer_delta_ci95": cer_interval,
        "decision": "inconclusive"
        if interval[0] <= 0 <= interval[1]
        else "candidate lower on selected clips"
        if interval[1] < 0
        else "candidate higher on selected clips",
        "bootstrap": "1000 paired group resamples, seed 42; speaker, else text ID, else clip",
        "scope": "Selected clips only; one training seed. This is not a generalization claim.",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", result)
    (output / "comparison.md").write_text(
        f"# Evaluation comparison\n\nLanguage: {a['language']}. Clips: {len(ids)}.\n\n"
        f"- Baseline WER: {base_score['wer']:.2%}\n"
        f"- Candidate WER: {tuned_score['wer']:.2%}\n"
        f"- Difference: {result['wer_delta'] * 100:+.2f} percentage points (lower is better).\n"
        f"- Paired 95% interval: [{interval[0] * 100:+.2f}, {interval[1] * 100:+.2f}] points.\n\n"
        f"- CER: {base_score['cer']:.2%} → {tuned_score['cer']:.2%}.\n"
        f"- CER difference 95% interval: [{cer_interval[0] * 100:+.2f}, "
        f"{cer_interval[1] * 100:+.2f}] points.\n"
        f"- Interpretation: {result['decision']}.\n\n"
        f"{result['scope']}\n",
        encoding="utf-8",
    )
    return result
