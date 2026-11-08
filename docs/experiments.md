# Run one comparison

Run from the repository root. Repeat with `es` for Spanish. The recipes use multilingual Whisper-tiny and support a single GPU; CPU smoke runs also work.

## 1. Prepare and baseline

```bash
asr-lab fetch --language en --output data/local/en --train 256 --validation 32 --test 64
asr-lab validate data/local/en/manifest.jsonl
asr-lab evaluate --model openai/whisper-tiny --language en --manifest data/local/en/manifest.jsonl --output runs/en-baseline
```

The dataset and model revisions are recorded. Reuse the recorded revisions for a strict rerun. Do not change the manifest between baseline and candidate evaluation.

## 2. Train

```bash
asr-lab train --config recipes/en-lora.yaml --manifest data/local/en/manifest.jsonl --output runs/en-lora
asr-lab train --config recipes/en-full.yaml --manifest data/local/en/manifest.jsonl --output runs/en-full
```

Recipes default to 300 optimizer steps. `--max-steps 2` is an execution check, not a quality experiment. Best checkpoint selection uses validation WER only.

## 3. Evaluate and compare

```bash
asr-lab evaluate --model openai/whisper-tiny --adapter runs/en-lora/model --language en --manifest data/local/en/manifest.jsonl --output runs/en-lora-eval
asr-lab evaluate --model runs/en-full/model --language en --manifest data/local/en/manifest.jsonl --output runs/en-full-eval
asr-lab compare --baseline runs/en-baseline --candidate runs/en-lora-eval --output runs/en-lora-report
asr-lab compare --baseline runs/en-baseline --candidate runs/en-full-eval --output runs/en-full-report
```

Evaluation writes `metrics.json` and `predictions.jsonl`. Comparison verifies data, references, normalizer, and decoding settings before calculating a paired confidence interval. Inspect successes and regressions; improvement is not guaranteed.

## Resume and resource limits

```bash
asr-lab train --config recipes/en-lora.yaml --manifest data/local/en/manifest.jsonl --output runs/en-lora --resume runs/en-lora/checkpoint-100
```

Resume requires the original recipe and data fingerprint. In Colab, optionally persist the run directory to your own Drive before training. Colab runtimes are temporary and GPU availability is not guaranteed. No paid services or uploads are enabled by default.

For GPU memory errors, lower `batch_size` and increase gradient accumulation to preserve the effective batch. Change the model in a copied recipe to `openai/whisper-base` or `openai/whisper-small` only after checking memory. Re-baseline that model.

## Transcribe a clip

```bash
asr-lab transcribe sample.wav --model openai/whisper-tiny --adapter runs/en-lora/model --language en
```

For deployment, retain the base revision with the adapter. An optimized runtime or quantized export needs its own accuracy and latency evaluation. This release does not claim streaming or production-serving support.
