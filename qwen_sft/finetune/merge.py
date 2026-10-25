"""Merge a LoRA adapter into its base model and produce standalone weights.

    python -m qwen_sft.finetune.merge --adapter ./lora_adapter
    python -m qwen_sft.finetune.merge --adapter ./lora_adapter --push-repo you/my-model
    python -m qwen_sft.finetune.merge --adapter you/my-adapter --push-repo you/my-model --skip-local-save

Merging defaults to CPU: a 4B bf16 merge needs ~8 GB of RAM, and doing it on a
16 GB T4 competes with the model copy. Pass ``--device cuda`` if you have
headroom. On Kaggle prefer ``--push-repo ... --skip-local-save`` so the 8 GB
output never lands in the 20 GB working directory.
"""

import argparse
import json
import os
import sys


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA adapter into a full model")
    parser.add_argument("--adapter", required=True,
                        help="Adapter dir, Hub repo id, or repo_id/subfolder (newest checkpoint-N wins)")
    parser.add_argument("--model", default=None,
                        help="Base model; defaults to base_model_name_or_path in adapter_config.json")
    parser.add_argument("--output-dir", default=os.path.join("merged_model"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--push-repo", default=None)
    parser.add_argument("--skip-local-save", action="store_true",
                        help="Push straight from memory (Kaggle disk quota)")
    parser.add_argument("--private", action="store_true")
    return parser.parse_args(argv)


def resolve_adapter_path(adapter: str) -> str:
    """Accept a local dir, a Hub repo, or ``repo_id/checkpoint-N``."""
    if os.path.isdir(adapter):
        return adapter

    from huggingface_hub import snapshot_download

    parts = adapter.strip("/").split("/")
    file_patterns = ["adapter_config.json", "adapter_model.safetensors", "adapter_model.bin",
                     "tokenizer*", "*.json", "*.model", "*.txt"]
    repo_id, subfolder = adapter, None
    if len(parts) > 2 and parts[-1].startswith("checkpoint-"):
        repo_id, subfolder = "/".join(parts[:-1]), parts[-1]

    patterns = file_patterns if subfolder is None else [f"{subfolder}/{pattern}" for pattern in file_patterns]
    print(f"[ADAPTER] downloading {repo_id}" + (f" ({subfolder})" if subfolder else ""))
    local_path = snapshot_download(repo_id=repo_id, allow_patterns=patterns)
    return os.path.join(local_path, subfolder) if subfolder else local_path


def resolve_base_model(adapter_path: str, explicit: str | None) -> str:
    """Resolve the base from the flag or the adapter; never guess.

    Merging onto the wrong base produces a model that looks fine and behaves
    wrong, so a missing record is a hard failure rather than a default.
    """
    if explicit:
        return explicit
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as handle:
            base = json.load(handle).get("base_model_name_or_path")
        if base:
            return base
    raise SystemExit(
        f"[FATAL] No base model recorded in {config_path}. "
        "Pass --model explicitly; refusing to merge onto a guessed base."
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = resolve_adapter_path(args.adapter)
    base_model = resolve_base_model(adapter_path, args.model)

    if args.dtype == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"[MERGE] {base_model} + {adapter_path} -> {args.output_dir} ({args.device})")
    load_kwargs = dict(dtype=dtype, low_cpu_mem_usage=True)
    try:
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
    except (ValueError, KeyError, OSError):
        from transformers import AutoModelForImageTextToText

        base = AutoModelForImageTextToText.from_pretrained(base_model, **load_kwargs)
    if args.device == "cuda":
        base = base.to("cuda")

    model = PeftModel.from_pretrained(base, adapter_path)
    merged = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    if not args.skip_local_save:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"[SAVE] writing merged weights to {args.output_dir}")
        merged.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    if args.push_repo:
        token = os.environ.get("HF_TOKEN")
        if not token:
            print("[HF] HF_TOKEN is not set; skipping push.")
            return 0
        print(f"[HF] pushing to {args.push_repo} ...")
        merged.push_to_hub(args.push_repo, token=token, private=args.private)
        tokenizer.push_to_hub(args.push_repo, token=token, private=args.private)
        print(f"[HF] done: https://huggingface.co/{args.push_repo}")

    print("[DONE] Merge complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
