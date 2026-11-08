"""Thin command-line entry points."""

import argparse
import json
import sys
from dataclasses import replace

from asr_lab.config import load_recipe
from asr_lab.manifest import validate_manifest


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="asr-lab", description="Reproducible Whisper experiments")
    commands = root.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="download a bounded FLEURS subset")
    fetch.add_argument("--language", choices=["en", "es"], required=True)
    fetch.add_argument("--output", required=True)
    fetch.add_argument("--revision", default="main")
    for split, count in (("train", 256), ("validation", 32), ("test", 64)):
        fetch.add_argument(f"--{split}", type=int, default=count)

    validate = commands.add_parser("validate", help="check labels, audio, and split isolation")
    validate.add_argument("manifest")

    train = commands.add_parser("train", help="train a full model or LoRA adapter")
    train.add_argument("--config", required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--resume")
    train.add_argument("--max-steps", type=int)

    for name in ("evaluate", "transcribe"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--model", required=True)
        cmd.add_argument("--language", choices=["en", "es"], required=True)
        cmd.add_argument("--revision", default="main")
        cmd.add_argument("--adapter")
        cmd.add_argument("--max-tokens", type=int, default=225)
        cmd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        if name == "evaluate":
            cmd.add_argument("--manifest", required=True)
            cmd.add_argument("--output", required=True)
            cmd.add_argument("--split", choices=["validation", "test"], default="test")
        else:
            cmd.add_argument("audio")

    report = commands.add_parser("compare", help="paired baseline/candidate evaluation")
    report.add_argument("--baseline", required=True)
    report.add_argument("--candidate", required=True)
    report.add_argument("--output", required=True)
    return root


def dispatch(args) -> dict:
    if args.command == "fetch":
        from asr_lab.data import fetch_fleurs

        path = fetch_fleurs(
            args.output,
            args.language,
            {k: getattr(args, k) for k in ("train", "validation", "test")},
            args.revision,
        )
        return {"manifest": str(path)}
    if args.command == "validate":
        return validate_manifest(args.manifest)
    if args.command == "train":
        from asr_lab.training import train

        recipe = load_recipe(args.config)
        if args.max_steps is not None:
            recipe = replace(recipe, max_steps=args.max_steps)
        return train(recipe, args.manifest, args.output, args.resume)
    if args.command == "evaluate":
        from asr_lab.evaluation import evaluate

        return evaluate(
            args.model,
            args.manifest,
            args.output,
            args.language,
            args.revision,
            args.adapter,
            args.split,
            args.max_tokens,
            args.device,
        )
    if args.command == "transcribe":
        import torch

        from asr_lab.modeling import load_model, transcribe

        model, processor = load_model(args.model, args.language, args.revision, args.adapter)
        device = (
            ("cuda" if torch.cuda.is_available() else "cpu")
            if args.device == "auto"
            else args.device
        )
        model.to(device)
        return {"text": transcribe(model, processor, args.audio, args.language, args.max_tokens)}
    from asr_lab.reporting import compare

    return compare(args.baseline, args.candidate, args.output)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = dispatch(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
