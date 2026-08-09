# =============================================================================
# COGITO 0.9 — LORA MERGE SCRIPT
#
# Merges a trained LoRA adapter into its base model to produce a complete,
# standalone full model (works on ANY GPU / any transformers app, not just
# Unsloth), then optionally pushes the full model to the Hub.
#
# The adapter can be:
#   - a local directory, e.g. cogito_0.9_lora
#   - a Hub repo id, e.g. ozaa77/Cogito-0.9          (auto-picks the latest checkpoint-N)
#   - a Hub subfolder, e.g. ozaa77/Cogito-0.9/checkpoint-330
#
# Usage (Kaggle 2x T4 or any GPU box with unsloth installed):
#   # Merge the latest Hub checkpoint and push the full model:
#   python scripts/merge_lora.py --push-to-hub
#
#   # Merge a specific checkpoint and push to a custom repo:
#   python scripts/merge_lora.py --adapter ozaa77/Cogito-0.9/checkpoint-330 \
#       --push-to-hub --push-repo ozaa77/Cogito-0.9-merged
#
#   # Save locally only (no upload):
#   python scripts/merge_lora.py --adapter cogito_0.9_lora --output-dir cogito_0.9_merged
#
# Requires HF_TOKEN in the environment when the adapter/base is private or when
# pushing. The base model recorded in adapter_config.json is used automatically;
# pass --base-model only if that recorded path no longer exists.
# =============================================================================

import argparse
import json
import os
import re

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HUB_ADAPTER_CACHE = os.path.join(PROJECT_ROOT, "_hub_adapters")


def find_adapter_dir(root: str):
    """Return the directory below `root` that contains adapter_config.json."""
    for dirpath, _dirnames, filenames in os.walk(root):
        if "adapter_config.json" in filenames:
            return dirpath
    return None


def resolve_adapter(adapter_arg: str, token) -> str:
    """Turn the --adapter argument into a local directory path.

    Local dir -> returned as-is. Hub ref -> snapshot_download the repo (or the
    exact subfolder) and return the local dir containing adapter_config.json.
    A bare repo id without subfolder auto-selects the latest checkpoint-N.
    """
    if os.path.isdir(adapter_arg):
        return os.path.abspath(adapter_arg)
    # A mistyped local path must not fall through into the Hub branch and produce
    # a cryptic error like repo_id="C:" -> list_repo_tree auth failure. NOTE:
    # os.sep is "/" on Linux/Kaggle, so do NOT use it here — a valid Hub id
    # like "org/repo" also contains slashes. Only unambiguous filesystem signals
    # (backslash, drive letter, ./ ../ / ~ prefix) count as a local path.
    if (
        "\\" in adapter_arg
        or re.match(r"^[A-Za-z]:", adapter_arg)
        or adapter_arg.startswith(("./", "../", "/", "~"))
    ):
        raise SystemExit(f"[FATAL] Local adapter directory not found: {adapter_arg}")

    from huggingface_hub import HfApi, RepoFile, RepoFolder, snapshot_download

    parts = adapter_arg.split("/")
    if len(parts) < 2:
        raise SystemExit(
            f"[FATAL] '{adapter_arg}' is neither a local directory nor a Hub repo id."
        )
    repo_id = "/".join(parts[:2])
    subfolder = "/".join(parts[2:]) if len(parts) > 2 else None

    if subfolder is None:
        try:
            entries = HfApi().list_repo_tree(
                repo_id, repo_type="model", recursive=False, token=token
            )
        except Exception as exc:
            hint = " Set HF_TOKEN (or pass --token) if this repo is private." if not token else ""
            raise SystemExit(f"[FATAL] Could not list {repo_id}: {exc}{hint}")
        # Prefer the FINAL adapter pushed to the repo root: train.py saves the
        # completed adapter to the root after the last epoch, so it is the most
        # complete trained artifact. Checkpoints are only intermediate saves.
        root_has_adapter = any(
            isinstance(entry, RepoFile)
            and getattr(entry, "path", "") == "adapter_config.json"
            for entry in entries
        )
        if root_has_adapter:
            print("[HUB] Final adapter found at repo root — using it (complete trained artifact).")
        else:
            steps = []
            for entry in entries:
                if isinstance(entry, RepoFolder):
                    name = os.path.basename(getattr(entry, "path", ""))
                    if name.startswith("checkpoint-"):
                        try:
                            steps.append(int(name.split("-")[-1]))
                        except ValueError:
                            continue
            if not steps:
                raise SystemExit(
                    f"[FATAL] No adapter_config.json at the root and no checkpoint-* folders "
                    f"in {repo_id}. Pass the exact subfolder, e.g. {repo_id}/checkpoint-330"
                )
            subfolder = f"checkpoint-{max(steps)}"
            print(f"[HUB] Auto-selected latest checkpoint: {repo_id}/{subfolder}")

    cache_dir = os.path.join(
        HUB_ADAPTER_CACHE, f"{repo_id}__{subfolder.replace('/', '__')}"
    )
    os.makedirs(cache_dir, exist_ok=True)
    print(f"[HUB] Downloading {repo_id}/{subfolder} ...")
    if subfolder is None:
        # Repo-root adapter only: skip the (large) checkpoint-* folders.
        snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            ignore_patterns=["checkpoint-*", "checkpoint-*/*"],
            token=token,
        )
    else:
        # snapshot_download has NO subfolder kwarg (that is a transformers-only
        # parameter). Scope the download to the subfolder with allow_patterns;
        # with local_dir the repo tree is preserved, so files land under
        # cache_dir/<subfolder>/... and find_adapter_dir() picks them up.
        snapshot_download(
            repo_id=repo_id,
            local_dir=cache_dir,
            allow_patterns=[f"{subfolder}/*"],
            token=token,
        )
    adapter_dir = find_adapter_dir(cache_dir)
    if adapter_dir is None:
        raise SystemExit(
            f"[FATAL] No adapter_config.json found under {cache_dir}. "
            f"Is '{subfolder}' a PEFT checkpoint?"
        )
    return adapter_dir


def main():
    parser = argparse.ArgumentParser(
        description="Merge a Cogito LoRA adapter into its base model (full standalone model)."
    )
    parser.add_argument(
        "--adapter",
        default="ozaa77/Cogito-0.9",
        help="Local adapter dir, Hub repo id (auto-picks latest checkpoint-N), "
        "or Hub repo id/subfolder (default: ozaa77/Cogito-0.9)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Local dir for the merged full model (default: cogito_0.9_merged)",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Also merge and push the full model to the Hub",
    )
    parser.add_argument(
        "--skip-local-save",
        action="store_true",
        help="Skip saving the merged model to disk and stream it straight to the "
        "Hub via --push-to-hub (for tight-disk environments like Kaggle's 20GB "
        "/kaggle/working). Ignored if --output-dir is given.",
    )
    parser.add_argument(
        "--push-repo",
        default="ozaa77/Cogito-0.9-merged",
        help="Hub repo id to push the merged full model to",
    )
    parser.add_argument(
        "--private", action="store_true", help="Create the push repo as private"
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Override the base model (only if adapter_config.json's recorded base "
        "is a wiped local path; the adapter MUST have been trained on this exact base)",
    )
    parser.add_argument(
        "--ablit-adapter",
        default=None,
        help="Optional abliteration delta adapter (the rank-1 LoRA emitted by "
        "scripts/abliterate_cogito.py --adapter mode). When set, it is merged "
        "additively on top of the Cogito adapter, producing the FULL abliterated "
        "model: base + Cogito + abliteration. Accepts a local dir or a Hub repo id.",
    )
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if args.push_to_hub and not token:
        raise SystemExit("[FATAL] --push-to-hub requires a token: pass --token or set HF_TOKEN.")

    adapter_path = resolve_adapter(args.adapter, token)
    print(f"\n[ADAPTER] Using: {adapter_path}")

    # Inspect adapter_config.json to learn which base model this adapter needs.
    base_model = args.base_model
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, encoding="utf-8") as fh:
            recorded_base = json.load(fh).get("base_model_name_or_path")
        print(f"[ADAPTER] adapter_config.json records base model: {recorded_base}")
        if base_model is None:
            base_model = recorded_base
        if base_model and not os.path.isdir(base_model) and "/" not in base_model:
            # Looks like a local path that does not exist on this machine (e.g. an
            # abliterated model from a wiped Kaggle session).
            print("[WARN] The recorded base looks like a local path that does not exist here.")
            print("       If it was the abliterated model from a previous Kaggle session, it is gone.")
            print("       Fixes: re-run abliteration on this machine, merge on the machine that")
            print("       still has it, or pass --base-model Qwen/Qwen2.5-Coder-14B (only valid")
            print("       if the adapter was trained on plain Qwen).")
    if base_model is None:
        base_model = "Qwen/Qwen2.5-Coder-14B"
        print(f"[ADAPTER] No base recorded, defaulting to {base_model}")

    print(f"\nLoading adapter + base model ({base_model}) ...")
    from unsloth import FastLanguageModel

    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=adapter_path,
            max_seq_length=args.max_seq_length,
            dtype=None,
            load_in_4bit=True,
            token=token,
        )
    except Exception as exc:
        raise SystemExit(
            f"[FATAL] Failed to load adapter/base: {exc}\n"
            f"  The recorded base was '{base_model}'. If that path no longer exists\n"
            f"  (e.g. abliterated model from a wiped Kaggle session), re-run abliteration\n"
            f"  or pass --base-model explicitly."
        ) from exc

    # Confirm the adapter is actually applied by looking for LoRA modules
    # (requires_grad is unreliable after a load: PEFT may mark them frozen).
    lora_params = [(n, p) for n, p in model.named_parameters() if "lora_" in n.lower()]
    if lora_params:
        lora_count = sum(p.numel() for _, p in lora_params)
        print(f"[MODEL] LoRA adapter attached ({len(lora_params)} LoRA modules, "
              f"{lora_count:,} parameters).")
    else:
        print("[WARN] No 'lora_' modules found in the loaded model — "
              "the adapter may not have been applied.")

    # Optional: apply the abliteration delta adapter on top of the Cogito
    # adapter. Both adapters are then active, so the merge (save_pretrained_merged
    # / push_to_hub_merged) produces the full abliterated model. This is the
    # 'big machine' path of scripts/abliterate_cogito.py --adapter mode: same
    # final weights, but materialized as a standalone full model here.
    if args.ablit_adapter:
        ablit_path = resolve_adapter(args.ablit_adapter, token)
        print(f"\n[ABLIT] Applying abliteration delta adapter: {ablit_path}")
        model.load_adapter(ablit_path, adapter_name="ablit")
        model.set_adapter(["default", "ablit"])
        print("[ABLIT] Both adapters active — merging base + Cogito + abliteration.")
        print("[ABLIT] NOTE: this relies on peft summing the deltas of all ACTIVE adapters")
        print("[ABLIT] during the merge. If the merged model still refuses, fall back to:")
        print("[ABLIT]   model = model.merge_and_unload()")
        print("[ABLIT]   model.save_pretrained(out); tokenizer.save_pretrained(out)")
        print("[ABLIT] on a machine with enough VRAM (merge_and_unload dequantizes to fp16).")

    # Save locally unless the user said skip. On Kaggle the 20GB /kaggle/working
    # quota cannot hold a 28GB bf16 merge, so --push-to-hub --skip-local-save
    # streams the merged model to the Hub instead (Unsloth's push path is
    # designed to work without free disk space).
    save_locally = bool(args.output_dir) or not args.skip_local_save
    if save_locally:
        out = os.path.abspath(args.output_dir or os.path.join(PROJECT_ROOT, "cogito_0.9_merged"))
        os.makedirs(out, exist_ok=True)
        print(f"\nMerging (16-bit) and saving to {out} ...")
        model.save_pretrained_merged(out, tokenizer, save_method="merged_16bit")
        print(f"[DONE] Merged full model saved to {out}")
    elif not args.push_to_hub:
        raise SystemExit("[FATAL] --skip-local-save without --push-to-hub or --output-dir would produce nothing.")

    if args.push_to_hub:
        print(f"\nPushing the merged model to https://huggingface.co/{args.push_repo} ...")
        model.push_to_hub_merged(
            repo_id=args.push_repo,
            tokenizer=tokenizer,
            save_method="merged_16bit",
            token=token,
            private=args.private,
        )
        print(f"[DONE] Merged full model live at https://huggingface.co/{args.push_repo}")


if __name__ == "__main__":
    main()
