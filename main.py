"""One-shot runner for the Kaggle README cell.

The notebook cell only bootstraps (clone or pull this repo) and calls this
script; everything else — dependency install, dataset download + slice,
checkpoint-safe training — lives here so it is versioned in git instead of
pasted into a notebook.

    python main.py                          # 20k slice, 1 epoch, full pipeline
    python main.py --max-samples 10000 --epochs 2
    python main.py --no-resume              # ignore last-checkpoint, start fresh

With the Kaggle secret ``HF_TOKEN`` present the training run automatically
pushes ``last-checkpoint`` to ``--hub-repo`` at every save, resumes from it on
rerun (asking first), and uploads the finished adapter. Without a token the
whole thing stays local: checkpoints live in ``training_output/`` only.
"""

import argparse
import os
import subprocess
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DATASET_REPO = "ozaa77/Cogito-0.9-qwen-sft-mix"
DATASET_FILE = "qwen_sft_mix.jsonl"
DEFAULT_HUB_REPO = "ozaa77/Cogito-0.9-qwen-sft-adapter"
DEFAULT_MAX_SAMPLES = 20000
DEFAULT_SAVE_STEPS = 200

from qwen_sft.config import (DEFAULT_BASE_MODEL, DEFAULT_EPOCHS,
                             DEFAULT_MAX_SEQ_LENGTH, MERGED_DIR)


def run(cmd, cwd=None):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=cwd)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install, download data, train.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base model repo id or local dir")
    parser.add_argument("--dataset-repo", default=DATASET_REPO,
                        help="Hugging Face dataset id holding the merged mix")
    parser.add_argument("--dataset-file", default=DATASET_FILE,
                        help="File name inside the dataset repo")
    parser.add_argument("--hub-repo", default=DEFAULT_HUB_REPO,
                        help="Where checkpoints and the adapter are pushed")
    parser.add_argument("--public", action="store_true",
                        help="Create the adapter repo as public (default private)")
    parser.add_argument("--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
                        help="Row slice for the first run (0 = all rows)")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help="Context window; drop to 2048 only on OOM")
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS,
                        help="Full passes over the slice")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
                        help="Steps between checkpoint saves/pushes")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore last-checkpoint and train from scratch")
    args = parser.parse_args(argv)
    args.resume = True if args.resume is None else args.resume
    return args


def ensure_hf_token() -> str | None:
    """Read HF_TOKEN from the environment or the Kaggle secret store."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"]
    try:
        from kaggle_secrets import UserSecretsClient

        token = UserSecretsClient().get_secret("HF_TOKEN")
        os.environ["HF_TOKEN"] = token
        return token
    except Exception:
        return None


def install_stack() -> None:
    """Install the training stack once; skipping a working install saves ~3 min."""
    probe = subprocess.run([sys.executable, "-c", "import unsloth"], check=False)
    if probe.returncode == 0:
        print("[INSTALL] Unsloth already importable; skipping pip.")
        return
    run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
        cwd=REPO_ROOT)
    run([sys.executable, "-m", "pip", "install", "-q", "unsloth[kaggle-new]==2026.8.8"])


def fetch_dataset(args) -> str:
    """Download the merged mix and optionally slice it for a T4-sized run."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=args.dataset_repo,
        filename=args.dataset_file,
        repo_type="dataset",
    )
    print(f"[DATA] {path} ({os.path.getsize(path) / 1e9:.2f} GB)")
    if not args.max_samples or args.max_samples <= 0:
        return path

    # The merged file is already shuffled, so the first N rows are a random sample.
    subset_path = os.path.join(MERGED_DIR, f"qwen_sft_mix_{args.max_samples}.jsonl")
    os.makedirs(MERGED_DIR, exist_ok=True)
    rows = 0
    with open(path, encoding="utf-8") as src, \
            open(subset_path, "w", encoding="utf-8") as dst:
        for line in src:
            if rows >= args.max_samples:
                break
            dst.write(line)
            rows += 1
    print(f"[DATA] using {rows:,} rows -> {subset_path}")
    return subset_path


def build_train_cmd(args, dataset_path, token) -> list[str]:
    cmd = [
        sys.executable, "-m", "qwen_sft", "train",
        "--model", args.model,
        "--dataset", dataset_path,
        "--epochs", str(args.epochs),
        "--max-seq-length", str(args.max_seq_length),
        "--save-steps", str(args.save_steps),
    ]
    if args.hub_repo and token:
        cmd += ["--hub-repo", args.hub_repo]
        if not args.public:
            cmd.append("--hub-private")
        if not args.resume:
            cmd.append("--no-resume-from-hub")
    mode = (f"hub checkpoints on {args.hub_repo}" if args.hub_repo and token
            else "local-only (no HF_TOKEN)")
    print(f"[TRAIN] {mode}")
    return cmd


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = ensure_hf_token()
    if not token and args.hub_repo:
        print("[HF] No HF_TOKEN: checkpoints stay local; a crash restarts training.")
    install_stack()
    dataset_path = fetch_dataset(args)
    run(build_train_cmd(args, dataset_path, token), cwd=REPO_ROOT)
    print(f"[DONE] adapter: {os.path.join(REPO_ROOT, 'lora_adapter')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
