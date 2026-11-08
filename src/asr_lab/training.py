"""Full and LoRA training through one tested path."""

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from asr_lab.audio import SAMPLE_RATE, load_audio
from asr_lab.config import Recipe
from asr_lab.manifest import read_manifest, select_rows, validate_manifest, write_json
from asr_lab.metrics import score
from asr_lab.modeling import add_lora, load_model, resolve_revision


class SpeechDataset:
    def __init__(self, rows, processor, max_tokens):
        self.rows, self.processor = rows, processor
        self.labels = []
        for row in rows:
            labels = processor.tokenizer(row["text"]).input_ids
            if len(labels) > max_tokens:
                raise ValueError(f"{row['id']}: transcript exceeds max_label_tokens; segment it")
            self.labels.append(labels)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        inputs = self.processor.feature_extractor(
            load_audio(self.rows[index]["_audio_path"]),
            sampling_rate=SAMPLE_RATE,
            return_attention_mask=True,
        )
        return {
            "input_features": inputs.input_features[0],
            "labels": self.labels[index],
            "attention_mask": inputs.attention_mask[0],
        }


@dataclass
class SpeechCollator:
    processor: object
    decoder_start_token_id: int

    def __call__(self, features):
        import torch

        batch = {
            "input_features": torch.tensor(
                np.stack([f["input_features"] for f in features]),
                dtype=torch.float32,
            ),
            "attention_mask": torch.tensor(np.stack([f["attention_mask"] for f in features])),
        }
        labels = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features],
            return_tensors="pt",
        )
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        if (ids[:, 0] == self.decoder_start_token_id).all().item():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


def train(
    recipe: Recipe,
    manifest: str | Path,
    output: str | Path,
    resume: str | None = None,
) -> dict:
    import torch
    from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, set_seed

    validation = validate_manifest(manifest)
    rows = read_manifest(manifest)
    train_rows = select_rows(rows, "train", recipe.language)
    valid_rows = select_rows(rows, "validation", recipe.language)
    output = Path(output)
    if output.exists() and any(output.iterdir()) and not resume:
        raise ValueError("training output exists; choose a new run directory or --resume")
    set_seed(recipe.seed)
    revision = resolve_revision(recipe.model, recipe.revision)
    identity = {
        "recipe": {**recipe.to_dict(), "revision": revision},
        "manifest_sha256": validation["manifest_sha256"],
    }
    if resume:
        previous = output / "run.json"
        if not previous.exists() or json.loads(previous.read_text()) != identity:
            raise ValueError("resume requires the original recipe, manifest, and run directory")
        checkpoint = Path(resume).resolve()
        if not checkpoint.is_relative_to(output.resolve()) or not checkpoint.is_dir():
            raise ValueError("resume checkpoint must be inside the run directory")
    model, processor = load_model(recipe.model, recipe.language, revision)
    if recipe.method == "lora":
        model = add_lora(model, recipe.lora_rank, recipe.lora_alpha, recipe.lora_dropout)
    model.config.use_cache = False
    training_data = SpeechDataset(train_rows, processor, recipe.max_label_tokens)
    validation_data = SpeechDataset(valid_rows, processor, recipe.max_label_tokens)
    interval = min(recipe.eval_steps, recipe.max_steps)
    args = Seq2SeqTrainingArguments(
        output_dir=str(output),
        max_steps=recipe.max_steps,
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=recipe.gradient_accumulation_steps,
        learning_rate=recipe.learning_rate,
        warmup_ratio=recipe.warmup_ratio,
        eval_strategy="steps",
        eval_steps=interval,
        save_steps=interval,
        save_total_limit=2,
        logging_steps=min(10, interval),
        predict_with_generate=True,
        generation_max_length=recipe.max_label_tokens,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        gradient_checkpointing=recipe.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=recipe.seed,
        data_seed=recipe.seed,
        report_to="none",
        push_to_hub=False,
        remove_unused_columns=False,
        save_safetensors=True,
    )

    def compute_metrics(prediction):
        labels = np.where(
            prediction.label_ids == -100, processor.tokenizer.pad_token_id, prediction.label_ids
        )
        refs = processor.batch_decode(labels, skip_special_tokens=True)
        hyps = processor.batch_decode(prediction.predictions, skip_special_tokens=True)
        measured = score(refs, hyps)
        return {"wer": measured["wer"], "cer": measured["cer"]}

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=training_data,
        eval_dataset=validation_data,
        data_collator=SpeechCollator(processor, model.config.decoder_start_token_id),
        compute_metrics=compute_metrics,
        processing_class=processor,
    )
    write_json(output / "run.json", identity)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = trainer.train(resume_from_checkpoint=resume)
    elapsed = time.perf_counter() - start
    model.config.use_cache = True
    trainer.save_model(str(output / "model"))
    processor.save_pretrained(output / "model")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    summary = {
        "method": recipe.method,
        "language": recipe.language,
        "steps": trainer.state.global_step,
        "train_rows": len(train_rows),
        "validation_rows": len(valid_rows),
        "trainable_parameters": trainable,
        "total_parameters": model.num_parameters(),
        "elapsed_seconds": elapsed,
        "training_loss": result.training_loss,
        "best_validation_wer": trainer.state.best_metric,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()
        if torch.cuda.is_available()
        else None,
        "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu",
        "test_used_for_selection": False,
    }
    write_json(output / "training.json", summary)
    return summary
