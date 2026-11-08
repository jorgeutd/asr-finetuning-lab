"""Shared train/inference audio preprocessing."""

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

SAMPLE_RATE = 16_000


def load_audio(path: str | Path) -> np.ndarray:
    waveform, rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if not waveform.size or not np.isfinite(waveform).all():
        raise ValueError("audio must contain finite samples")
    if rate != SAMPLE_RATE:
        factor = gcd(rate, SAMPLE_RATE)
        waveform = resample_poly(waveform, SAMPLE_RATE // factor, rate // factor)
    return np.asarray(waveform, dtype=np.float32)


def inspect_audio(path: str | Path) -> dict:
    waveform = load_audio(path)
    duration = len(waveform) / SAMPLE_RATE
    if not 0.1 <= duration <= 30:
        raise ValueError(f"expected a 0.1-30 second clip, got {duration:.3f}s; segment with labels")
    return {
        "duration_seconds": round(duration, 6),
        "rms_db": float(20 * np.log10(np.sqrt(np.mean(waveform**2)) + 1e-12)),
        "clipped_fraction": float(np.mean(np.abs(waveform) >= 0.999)),
    }
