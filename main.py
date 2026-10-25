"""One-shot runner for the Kaggle README cell.

The notebook cell only bootstraps (clone or pull this repo) and calls this
script; everything else — dependency install, dataset download, checkpoint-safe
training — lives here so it is versioned in git instead of pasted into a
notebook.

    python main.py                        # 20k-row slice, 1 epoch (~28 GPU-h = one Kaggle week)
    python main.py --max-samples 0 --epochs 2   # full 101,826-row mix, 2 passes (~283h measured)
    python main.py --gpus 1               # single T4 instead of both
    python main.py --no-resume            # quarantined checkpoints, start over

Runs are multi-session on Kaggle: with the ``HF_TOKEN`` secret the training run
pushes ``last-checkpoint`` to ``--hub-repo`` at every save, resumes from it on
rerun (asking first), and uploads the finished adapter. Without a token the
whole thing stays local: checkpoints live in ``training_output/`` only.

Before the launch, the merged mix is pre-tokenized once into
``training_output/pretokenized`` (``python -m qwen_sft pretokenize``), so the
DDP ranks only memory-map the result instead of each tokenizing the corpus —
the concurrent in-rank preparation is what the kernel OOM killer picks off on
Kaggle's ~29 GB host RAM (rank 0 SIGKILLed mid-tokenize).

With more than one GPU visible (Kaggle: 2x T4) the training command is launched
under ``torchrun`` so both GPUs train (DDP); gradient accumulation is halved to
keep the effective batch identical to the single-GPU run. The resume decision
is made once here and passed explicitly to every rank.
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
PRETOKENIZED_DIR = os.path.join(REPO_ROOT, "training_output", "pretokenized")
# Kaggle budget: ~30 GPU-hours of weekly quota (12h max per session), and both
# T4s together run this mix at a measured ~80 s/step (effective batch 16).
# 20,000 rows x 1 epoch = 1,250 steps = ~28h — one weekly quota, spread over
# 2-3 sessions via the Hub-resume machinery. 0 = every row of the mix
# (101,826): measured ~283h for 2 epochs (~142h for 1) — multi-week territory.
DEFAULT_MAX_SAMPLES = 20000
# One pass is the SFT default; each extra epoch doubles the quota bill.
DEFAULT_EPOCHS = 1
# ~2.2h between saves at ~80 s/step, so a killed session loses at most ~2h.
DEFAULT_SAVE_STEPS = 100

from qwen_sft.config import (DEFAULT_BASE_MODEL, DEFAULT_GRAD_ACCUM,
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
                        help="Rows of the mix for this run (0 = all 101,826; "
                             "resume makes long runs multi-session)")
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH,
                        help="Context window; drop to 2048 only on OOM")
    parser.add_argument("--epochs", type=float, default=DEFAULT_EPOCHS,
                        help="Full passes over the data")
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS,
                        help="Steps between checkpoint saves/pushes")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="Ignore last-checkpoint, quarantine old checkpoint "
                             "dirs, and train from scratch")
    parser.add_argument("--no-pretokenize", dest="pretokenize",
                        action="store_false",
                        help="Skip the offline pre-tokenization step and let "
                             "the ranks prepare the dataset themselves (the "
                             "path that OOM-killed 2-GPU sessions)")
    parser.add_argument("--gpus", type=int, default=0,
                        help="GPUs for DDP training (0 = auto-detect; Kaggle "
                             "2x T4 trains on both, 1 forces single GPU)")
    args = parser.parse_args(argv)
    args.resume = True if args.resume is None else args.resume
    args.pretokenize = True if args.pretokenize is None else args.pretokenize
    return args


def detect_gpu_count() -> int:
    """Count CUDA GPUs without importing torch into this process."""
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.cuda.device_count())"],
            check=False, capture_output=True, text=True,
        )
        return max(1, int(probe.stdout.strip().splitlines()[-1]))
    except (ValueError, IndexError):
        return 1  # no CUDA for whatever reason; the trainer reports it
    except OSError:
        # Probe cannot even run (no driver/env): assume one GPU and let the
        # trainer surface the real problem at load time.
        print("[GPU] GPU probe failed; falling back to single GPU.")
        return 1


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


def ensure_pretokenized(args, dataset_path) -> str | None:
    """Build the Arrow cache the ranks will memory-map, once, in this process.

    Two torchrun ranks tokenizing the same 101,826-row mix concurrently is
    what SIGKILLed a Kaggle session's rank 0 (host RAM): TRL's preload phases
    and Unsloth's mask map each hold a full corpus copy per rank. Running the
    preparation as ONE subprocess here (~15 min CPU, no GPU) and pointing both
    ranks at its output drops in-training preprocessing to memory-mapped
    reads. An existing cache whose meta.json still matches (model, sequence
    length, source files) is reused, so resume sessions skip straight to
    training. Returns the directory, or None to fall back to the legacy
    in-rank preparation. A crashed build aborts the run here (its real error
    is worth more to the session than silently OOMing in the ranks); only a
    build that completes but fails the meta check falls back.
    """
    from qwen_sft.finetune.pretokenize import matches_meta, read_meta

    sources = [(os.path.abspath(dataset_path), os.path.getsize(dataset_path))]
    if matches_meta(read_meta(PRETOKENIZED_DIR), args.model,
                    args.max_seq_length, sources):
        print(f"[DATA] pre-tokenized cache is current: {PRETOKENIZED_DIR}")
        return PRETOKENIZED_DIR
    print(f"[DATA] building pre-tokenized cache -> {PRETOKENIZED_DIR} "
          "(single CPU pass; every rank and resume session reuses it)")
    run([sys.executable, "-m", "qwen_sft.finetune.pretokenize",
         "--dataset", dataset_path, "--model", args.model,
         "--max-seq-length", str(args.max_seq_length),
         "--output-dir", PRETOKENIZED_DIR],
        cwd=REPO_ROOT)
    if matches_meta(read_meta(PRETOKENIZED_DIR), args.model,
                    args.max_seq_length, sources):
        return PRETOKENIZED_DIR
    print("[DATA] pre-tokenization did not produce a matching cache; ranks "
          "prepare the dataset themselves (the OOM-prone path)")
    return None


def resolve_resume(args, token) -> str | None:
    """Decide where to resume from once, so every DDP rank gets the same answer.

    Delegates to the train module's resolver (it validates checkpoints, prompts,
    and quarantines stale dirs). Two torchrun ranks cannot share stdin safely,
    so the decision has to happen before the launch.
    """
    from types import SimpleNamespace

    from qwen_sft.finetune.train import resolve_resume_checkpoint

    ns = SimpleNamespace(
        resume_from_checkpoint=None,
        fresh=not args.resume,
        resume_from_hub=bool(args.hub_repo) and bool(token) and args.resume,
        push_checkpoints=bool(args.hub_repo) and bool(token),
        training_output_dir=os.path.join(REPO_ROOT, "training_output"),
        hub_repo=args.hub_repo or "",
        hub_private=not args.public,
        epochs=args.epochs,
    )
    return resolve_resume_checkpoint(ns)


def build_train_cmd(args, dataset_path, token, gpus: int) -> list[str]:
    resolved = resolve_resume(args, token)
    # Each rank trains its own batch under DDP, so halve accumulation to keep
    # the effective batch (and the LR schedule) identical to a single-GPU run.
    grad_accum = max(1, DEFAULT_GRAD_ACCUM // gpus) if gpus > 1 else DEFAULT_GRAD_ACCUM
    train_args = [
        "--model", args.model,
        "--dataset", dataset_path,
        "--epochs", str(args.epochs),
        "--max-seq-length", str(args.max_seq_length),
        "--save-steps", str(args.save_steps),
        "--grad-accum", str(grad_accum),
    ]
    if resolved:
        train_args += ["--resume-from-checkpoint", resolved]
        print(f"[TRAIN] resuming from {resolved}")
    else:
        train_args += ["--fresh"]
    if args.hub_repo and token:
        train_args += ["--hub-repo", args.hub_repo]
        if not args.public:
            train_args.append("--hub-private")
    mode = (f"hub checkpoints on {args.hub_repo}" if args.hub_repo and token
            else "local-only (no HF_TOKEN)")
    print(f"[TRAIN] {mode}")
    pretokenized = getattr(args, "pretokenized_dir", None)
    if pretokenized:
        train_args += ["--pretokenized-dir", pretokenized]
    if gpus > 1:
        print(f"[TRAIN] DDP on {gpus} GPUs: per-rank batch 1 x grad-accum {grad_accum}"
              f" x {gpus} ranks = effective batch {grad_accum * gpus}")
        return [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={gpus}", "-m", "qwen_sft", "train",
        ] + train_args
    return [sys.executable, "-m", "qwen_sft", "train"] + train_args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    token = ensure_hf_token()
    if not token and args.hub_repo:
        print("[HF] No HF_TOKEN: checkpoints stay local; a crash restarts training.")
    install_stack()
    gpus = args.gpus if args.gpus > 0 else detect_gpu_count()
    if gpus > 1:
        print(f"[GPU] {gpus} CUDA GPUs -> torchrun DDP (opt out with --gpus 1)")
    dataset_path = fetch_dataset(args)
    if args.pretokenize:
        pretokenized = ensure_pretokenized(args, dataset_path)
        if pretokenized:
            args.pretokenized_dir = pretokenized
    run(build_train_cmd(args, dataset_path, token, gpus), cwd=REPO_ROOT)
    print(f"[DONE] adapter: {os.path.join(REPO_ROOT, 'lora_adapter')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
