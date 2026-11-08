# Measured runs

**CPU smoke check:** Whisper-tiny; 12 optimizer steps per method;
8 train / 2 validation / 6 test clips per language; seed 42; four CPU threads.

These runs verify execution. They do not support a quality or production claim.

- **EN / lora**: WER 12.10% → 15.32%; difference +3.23 points, 95% CI [+0.00, +6.84]. Inconclusive.
- **EN / full**: WER 12.10% → 11.29%; difference -0.81 points, 95% CI [-2.00, +0.00]. Inconclusive.
- **ES / lora**: WER 18.18% → 16.88%; difference -1.30 points, 95% CI [-2.93, +0.00]. Inconclusive.
- **ES / full**: WER 18.18% → 18.18%; difference +0.00 points, 95% CI [+0.00, +0.00]. Inconclusive.

[Raw metrics, CER, predictions, recipes, and intervals](cpu-smoke/).

The model/dataset revisions and content fingerprints are saved with each run. Speaker IDs are unavailable; these convenience subsets are not representative. Intervals do not measure variation across training seeds.

An initial LoRA run failed while reloading its best checkpoint. The wrapper was corrected and a pretrained regression test added. Final evaluations preserve saved token-suppression settings.

**Colab GPU experiments:** pending execution. No GPU quality results are claimed here.
