#!/usr/bin/env python3
"""
Upload Cogito 0.9 datasets to Hugging Face Hub
Run with: python scripts/upload_dataset_to_hub.py

For Kaggle: set HF_TOKEN in your Kaggle secrets or as env var.
"""

import os
import argparse
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload Cogito datasets to HF Hub")
    parser.add_argument("--repo-id", default="alghozali/cogito-0.9", help="HF repo ID (e.g. username/dataset)")
    parser.add_argument("--private", action="store_true", help="Make repo private")
    parser.add_argument("--token", help="HF token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Login
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("❌ No HF_TOKEN found.")
        print("   In Kaggle: go to Secrets & Variables → Add new → HF_TOKEN")
        print("   Or set it in your Kaggle notebook environment.")
        sys.exit(1)

    login(token=token)
    print("✅ Logged in successfully")

    api = HfApi()

    # Create dataset repo
    try:
        api.repo_create(args.repo_id, private=args.private)
        print(f"✅ Created dataset repo: https://huggingface.co/datasets/{args.repo_id}")
    except Exception:
        print(f"✅ Repo {args.repo_id} ready (or already exists)")

    # Upload files
    files = [
        ("cogito_0.9_master_dataset.jsonl", "cogito_0.9_master_dataset.jsonl"),
        ("combined_dense_dataset.jsonl", "combined_dense_dataset.jsonl"),
    ]

    for local_name, hf_name in files:
        if os.path.exists(local_name):
            print(f"Uploading {local_name}...")
            api.upload_file(
                path_or_fileobj=local_name,
                path_in_repo=hf_name,
                repo_id=args.repo_id,
                repo_type="dataset",
                commit_message=f"Add {hf_name}"
            )
            print(f"✅ Uploaded {hf_name}")

    print("\n🎉 All datasets uploaded!")
    print(f"View at: https://huggingface.co/datasets/{args.repo_id}")

if __name__ == "__main__":
    main()
