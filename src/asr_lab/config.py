"""Validated, serializable experiment settings."""

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Recipe:
    model: str = "openai/whisper-tiny"
    revision: str = "main"
    language: str = "en"
    method: str = "lora"
    seed: int = 42
    max_steps: int = 300
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 0.001
    warmup_ratio: float = 0.1
    eval_steps: int = 100
    max_label_tokens: int = 225
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True

    def __post_init__(self):
        if self.language not in {"en", "es"}:
            raise ValueError("language must be en or es")
        if self.method not in {"full", "lora"}:
            raise ValueError("method must be full or lora")
        for key in (
            "max_steps",
            "batch_size",
            "gradient_accumulation_steps",
            "eval_steps",
            "max_label_tokens",
            "lora_rank",
            "lora_alpha",
        ):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.learning_rate < 1:
            raise ValueError("learning_rate must be between 0 and 1")
        if not 0 <= self.warmup_ratio <= 1 or not 0 <= self.lora_dropout < 1:
            raise ValueError("invalid warmup_ratio or lora_dropout")
        if self.max_label_tokens > 448:
            raise ValueError("max_label_tokens exceeds Whisper's decoder context")

    def to_dict(self) -> dict:
        return asdict(self)


def load_recipe(path: str | Path) -> Recipe:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("recipe must be a YAML mapping")
    unknown = set(data) - {f.name for f in fields(Recipe)}
    if unknown:
        raise ValueError(f"unknown recipe keys: {sorted(unknown)}")
    return Recipe(**data)
