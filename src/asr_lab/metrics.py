"""Corpus-level error rates and paired uncertainty."""

import unicodedata

import jiwer
import numpy as np

NORMALIZER = "unicode-casefold-punctuation-space-v1"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold()
    text = "".join(" " if unicodedata.category(c).startswith("P") else c for c in text)
    return " ".join(text.split())


def score(references: list[str], hypotheses: list[str]) -> dict:
    if not references or len(references) != len(hypotheses):
        raise ValueError("expected equal, nonempty reference and hypothesis lists")
    refs, hyps = [normalize(r) for r in references], [normalize(h) for h in hypotheses]
    if any(not r for r in refs):
        raise ValueError("a reference is empty after normalization")
    words = jiwer.process_words(refs, hyps)
    return {
        "wer": words.wer,
        "cer": jiwer.cer(refs, hyps),
        "substitutions": words.substitutions,
        "deletions": words.deletions,
        "insertions": words.insertions,
        "reference_words": words.hits + words.substitutions + words.deletions,
        "normalizer": NORMALIZER,
    }


def paired_wer_interval(
    references: list[str],
    baseline: list[str],
    candidate: list[str],
    groups: list[str] | None = None,
    resamples: int = 1000,
    seed: int = 42,
) -> list[float]:
    """95% percentile interval for candidate WER minus baseline WER."""
    if len(candidate) != len(references) or len(baseline) != len(references):
        raise ValueError("unaligned predictions")
    groups = groups or [str(i) for i in range(len(references))]
    if len(groups) != len(references) or not references or resamples < 1:
        raise ValueError("invalid bootstrap input")
    unique = sorted(set(groups))
    totals = np.zeros((len(unique), 3))
    for i, group in enumerate(unique):
        indices = [j for j, value in enumerate(groups) if value == group]
        refs = [references[j] for j in indices]
        a = score(refs, [baseline[j] for j in indices])
        b = score(refs, [candidate[j] for j in indices])
        totals[i] = [
            a["reference_words"],
            sum(a[k] for k in ("substitutions", "deletions", "insertions")),
            sum(b[k] for k in ("substitutions", "deletions", "insertions")),
        ]
    samples = np.random.default_rng(seed).integers(0, len(unique), (resamples, len(unique)))
    sums = totals[samples].sum(axis=1)
    return np.quantile((sums[:, 2] - sums[:, 1]) / sums[:, 0], [0.025, 0.975]).tolist()
