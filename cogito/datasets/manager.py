import os
import sys
import subprocess
import glob

GENERATOR_FILES = {
    "agentic_tools.py",
    "execution_engine.py",
    "human_conversations.py",
    "identity_core.py",
    "personality_quirks.py",
    "retrieval_filter.py",
    "heated_conversations.py",
    "philosophical_probing.py",
}

def get_generators():
    generator_dir = os.path.join("cogito", "generators")
    if not os.path.exists(generator_dir):
        return []
    files = glob.glob(os.path.join(generator_dir, "*.py"))
    return sorted(
        os.path.basename(f) for f in files
        if os.path.basename(f) in GENERATOR_FILES
    )

def get_raw_dataset_path(generator_name):
    base_name = generator_name.replace(".py", "")
    return os.path.join("data", "raw", f"cogito_{base_name}.jsonl")

def print_menu():
    print("\n" + "="*55)
    print(" Cogito 0.9 Central Dataset Manager ".center(55, "="))
    print("="*55)
    generators = get_generators()
    
    for i, gen in enumerate(generators, 1):
        raw_path = get_raw_dataset_path(gen)
        count = 0
        if os.path.exists(raw_path):
            with open(raw_path, 'r', encoding='utf-8') as f:
                count = sum(1 for _ in f if _.strip())
        name_display = gen.replace('.py', '').replace('_', ' ').title()
        print(f" {i}. {name_display:<25} (Current: {count} rows)")
    
    print("-" * 55)
    print(f" {len(generators) + 1}. Run ALL Generators sequentially")
    print(f" {len(generators) + 2}. Merge Datasets to Master JSONL")
    print(f" {len(generators) + 3}. Restore local progress from HuggingFace")
    print(f" 0. Exit")
    print("="*55)
    return generators

def handle_generator(generator_name, skip_menu=False):
    raw_path = get_raw_dataset_path(generator_name)
    script_path = os.path.join("cogito", "generators", generator_name)
    
    if not skip_menu:
        name_display = generator_name.replace('.py', '').replace('_', ' ').title()
        print(f"\n[ Selected: {name_display} ]")
        print("1. Continue / Resume from checkpoint (Default)")
        print("2. DELETE current data and start fresh (Reset)")
        print("0. Cancel")
        
        action = input("\nChoose action [1/2/0]: ").strip()
        if action == "0":
            return
        elif action == "2":
            confirm = input(f"Are you sure you want to delete {raw_path}? [y/N]: ").strip().lower()
            if confirm == 'y':
                if os.path.exists(raw_path):
                    os.remove(raw_path)
                    print(f"[*] Deleted {raw_path}")
                else:
                    print("[*] No existing data found to delete.")
            else:
                print("Cancelled deletion.")
                return
        elif action != "1" and action != "":
            print("Invalid action, cancelling.")
            return
            
    print(f"\n[*] Running {script_path}...")
    try:
        subprocess.run([sys.executable, script_path])
    except KeyboardInterrupt:
        print("\n[!] Generator stopped by user.")

import argparse

def load_env_file(filepath=".env"):
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip()

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(project_root)
    
    load_env_file(".env")
    
    parser = argparse.ArgumentParser(description="Cogito Dataset Manager")
    parser.add_argument("--run-all", action="store_true", help="Run all generators non-interactively")
    args = parser.parse_args()

    generators = get_generators()
    
    if args.run_all:
        if not generators:
            print("No generators found in cogito/generators/ directory.")
            sys.exit(0)
        for gen in generators:
            print(f"\n--- Starting {gen} ---")
            handle_generator(gen, skip_menu=True)
        sys.exit(0)
    
    while True:
        generators = get_generators()
        if not generators:
            print("No generators found in cogito/generators/ directory.")
            break
            
        generators = print_menu()
        choice = input("\nSelect an option: ").strip()
        
        if choice == "0":
            print("Exiting Dataset Manager...")
            break
            
        try:
            choice_idx = int(choice)
        except ValueError:
            print("Invalid input. Please enter a number.")
            continue
            
        if 1 <= choice_idx <= len(generators):
            selected_gen = generators[choice_idx - 1]
            handle_generator(selected_gen)
            
        elif choice_idx == len(generators) + 1:
            print("\nWARNING: This will run all generators. They will resume from their current checkpoints.")
            confirm = input("Proceed? [y/N]: ").strip().lower()
            if confirm == 'y':
                for gen in generators:
                    print(f"\n--- Starting {gen} ---")
                    handle_generator(gen, skip_menu=True)
                    
        elif choice_idx == len(generators) + 2:
            print("\n[*] Merging datasets...")
            merge_script = os.path.join("cogito", "datasets", "merge_shards.py")
            subprocess.run([sys.executable, merge_script])
            
        elif choice_idx == len(generators) + 3:
            print("\n[*] Restoring from HuggingFace (ozaa77/Cogito-0.9.1-dataset)...")
            try:
                from datasets import load_dataset
                import json
                try:
                    ds = load_dataset("parquet", data_files="hf://datasets/ozaa77/Cogito-0.9.1-dataset/**/*.parquet", split="train")
                except Exception:
                    ds = load_dataset("parquet", data_files="hf://datasets/ozaa77/Cogito-0.9-dataset/**/*.parquet", split="train")
                restored_counts = {}
                os.makedirs(os.path.join("data", "raw"), exist_ok=True)
                
                for gen in generators:
                    raw_path = get_raw_dataset_path(gen)
                    if os.path.exists(raw_path):
                        os.remove(raw_path)
                
                for row in ds:
                    shard = row.get("source")
                    if not shard:
                        continue
                    out_path = os.path.join("data", "raw", shard)
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    restored_counts[shard] = restored_counts.get(shard, 0) + 1
                
                print("Restore complete! Recovered:")
                for s, count in restored_counts.items():
                    print(f"  {s}: {count} rows")
            except ImportError:
                print("[ERROR] 'datasets' library is not installed. Please run: pip install datasets")
            except Exception as e:
                print(f"[ERROR] Failed to restore from HuggingFace: {e}")
                
        else:
            print("Invalid option.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Dataset Manager...")
        sys.exit(0)
