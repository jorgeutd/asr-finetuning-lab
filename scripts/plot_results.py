"""Render the published comparisons without recalculating or selecting results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

root = Path(__file__).resolve().parents[1] / "results"
matplotlib.rcParams.update({"svg.hashsalt": "asr-lab", "svg.fonttype": "none"})
fig, ax = plt.subplots(figsize=(9, 4.8))
labels = []
for index, (language, method) in enumerate(
    [("en", "lora"), ("en", "full"), ("es", "lora"), ("es", "full")]
):
    path = root / "colab-t4" / language / f"{method}-comparison/comparison.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    value = row["wer_delta"] * 100
    low, high = (bound * 100 for bound in row["wer_delta_ci95"])
    color = "#b44738" if low > 0 else "#187c73" if high < 0 else "#566579"
    ax.errorbar(
        value,
        index,
        xerr=[[value - low], [high - value]],
        fmt="o",
        color=color,
        capsize=5,
        markersize=7,
        linewidth=2,
    )
    ax.annotate(
        f"{value:+.2f}",
        (value, index),
        xytext=(0, 11),
        textcoords="offset points",
        ha="center",
        fontsize=10,
        color=color,
    )
    labels.append(
        f"{'English' if language == 'en' else 'Spanish'} · {'LoRA' if method == 'lora' else 'Full'}"
    )
ax.axvline(0, color="#243347", linestyle="--", linewidth=1)
ax.set_yticks(range(4), labels)
ax.set_ylim(3.6, -0.6)
ax.set_xlabel("WER change vs baseline (percentage points; lower is better)", labelpad=12)
ax.grid(axis="x", color="#e4e8ed")
ax.set_axisbelow(True)
ax.tick_params(axis="both", length=0, pad=9)
for spine in ax.spines.values():
    spine.set_visible(False)
fig.suptitle("Whisper fine-tuning: measured WER changes", fontsize=16, fontweight="bold", y=0.97)
fig.text(
    0.5,
    0.90,
    "Whisper-tiny · 300 steps · 64 test clips per language · one seed",
    ha="center",
    fontsize=10,
    color="#566579",
)
fig.text(
    0.5,
    0.025,
    "Paired 95% intervals over selected text IDs; speaker IDs unavailable.\n"
    "These demonstration subsets do not establish production quality.",
    ha="center",
    fontsize=9,
    color="#566579",
)
fig.tight_layout(rect=(0, 0.09, 1, 0.87))
svg = root / "wer-change.svg"
fig.savefig(svg, metadata={"Date": None})
svg.write_text(
    "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
    encoding="utf-8",
)
fig.savefig(root / "wer-change.png", dpi=180)
plt.close(fig)
