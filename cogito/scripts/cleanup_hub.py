# =============================================================================
# COGITO 0.9 — HUB CLEANUP SCRIPT
#
# Prunes stale checkpoint-N folders from the Cogito model repo on the Hub
# (keeping the N most recent, mirroring train.py's HUB_KEEP_CHECKPOINTS=2)
# and can optionally flip the repo to public.
#
# Usage:
#   python scripts/cleanup_hub.py                    # keep latest 2 checkpoints
#   python scripts/cleanup_hub.py --keep 1 --make-public
#
# Requires HF_TOKEN (the repo is private and needs write access).
# =============================================================================

import argparse
import os
from huggingface_hub import HfApi, RepoFolder

DEFAULT_REPO = "ozaa77/Cogito-0.9"
DEFAULT_KEEP = 2


def collect_checkpoint_steps(api, repo_id, token):
    entries = api.list_repo_tree(
        repo_id, repo_type="model", revision="main", recursive=False, token=token
    )
    steps = []
    for entry in entries:
        # RepoFolder objects carry .path; older huggingface_hub versions expose
        # a dict-like entry. Mirror train.py's robust access pattern.
        name = os.path.basename(getattr(entry, "path", "") or "")
        if isinstance(entry, RepoFolder) and name.startswith("checkpoint-"):
            try:
                steps.append(int(name.split("-")[-1]))
            except ValueError:
                continue
    return sorted(steps)


def delete_checkpoint(api, repo_id, step, token):
    print(f"[HUB] Deleting checkpoint-{step} ...")
    try:
        api.delete_folder(
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"checkpoint-{step}",
            revision="main",
            token=token,
        )
    except TypeError:
        # huggingface_hub < 0.25 used folder_path instead of path_in_repo.
        api.delete_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=f"checkpoint-{step}",
            revision="main",
            token=token,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Prune stale checkpoints from the Cogito Hub repo (and optionally make it public)."
    )
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"Hub model repo (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--keep", type=int, default=DEFAULT_KEEP,
        help="Keep the N most recent checkpoint-N folders (default: 2; 0 deletes all)",
    )
    parser.add_argument(
        "--make-public", action="store_true",
        help="Flip the repo to public after pruning",
    )
    parser.add_argument("--token", default=None, help="HF token (default: HF_TOKEN env var)")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "[FATAL] HF_TOKEN (or --token) is required: this repo is private and needs auth."
        )

    api = HfApi()
    steps = collect_checkpoint_steps(api, args.repo, token)
    if not steps:
        print(f"[HUB] No checkpoint-* folders found in {args.repo}. Nothing to prune.")
    else:
        kept = steps[-args.keep:] if args.keep > 0 else []
        stale = steps[:-args.keep] if args.keep > 0 else steps
        if not stale:
            print(
                f"[HUB] {len(steps)} checkpoint(s) present; keeping the latest {args.keep}: "
                f"{[f'checkpoint-{s}' for s in kept]}"
            )
        for step in stale:
            try:
                delete_checkpoint(api, args.repo, step, token)
            except Exception as exc:
                # One 404 (already deleted) must not abort the rest of the prune.
                print(f"[HUB] Failed to delete checkpoint-{step} (non-fatal): {exc}")
        print(f"[HUB] Done. Remaining checkpoints: {[f'checkpoint-{s}' for s in kept]}")

    if args.make_public:
        print(f"[HUB] Making {args.repo} public ...")
        api.update_repo_visibility(
            repo_id=args.repo, repo_type="model", private=False, token=token
        )
        print(f"[HUB] {args.repo} is now public: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
