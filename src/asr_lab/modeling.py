"""Whisper loading, adaptation, and shared generation."""

import json
from pathlib import Path

from asr_lab.audio import SAMPLE_RATE, load_audio


def resolve_revision(model: str, revision: str) -> str:
    from huggingface_hub import HfApi

    if Path(model).is_dir():
        return "local"
    return HfApi().model_info(model, revision=revision).sha


def load_model(model_id: str, language: str, revision: str = "main", adapter: str | None = None):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    options = {} if Path(model_id).is_dir() else {"revision": revision}
    if adapter:
        config = json.loads((Path(adapter) / "adapter_config.json").read_text(encoding="utf-8"))
        if config.get("base_model_name_or_path") != model_id:
            raise ValueError("adapter base model differs from --model")
        pinned = config.get("revision")
        if pinned and resolve_revision(model_id, revision) != pinned:
            raise ValueError("adapter base revision differs; pass the recorded --revision")
    processor = WhisperProcessor.from_pretrained(
        model_id,
        language=language,
        task="transcribe",
        **options,
    )
    model = WhisperForConditionalGeneration.from_pretrained(model_id, **options)
    if adapter:
        model.load_adapter(adapter)
    model.generation_config.language = language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = model.config.suppress_tokens
    return model, processor


def add_lora(model, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model

    # The generic wrapper accepts audio inputs and supports Trainer checkpoint reloads.
    return get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            revision=getattr(model.config, "_commit_hash", None),
        ),
    )


def transcribe(model, processor, path: str, language: str, max_tokens: int = 225) -> str:
    import torch

    if not 1 <= max_tokens <= model.config.max_target_positions - 4:
        raise ValueError("max_tokens must fit the model's decoder context")
    samples = load_audio(path)
    if len(samples) > 30 * SAMPLE_RATE:
        raise ValueError("short-clip API accepts at most 30 seconds; segment long audio first")
    batch = processor(
        samples, sampling_rate=SAMPLE_RATE, return_tensors="pt", return_attention_mask=True
    ).to(model.device)
    model.eval()
    with torch.inference_mode():
        ids = model.generate(
            **batch,
            language=language,
            task="transcribe",
            num_beams=1,
            do_sample=False,
            max_new_tokens=max_tokens,
            return_timestamps=False,
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
