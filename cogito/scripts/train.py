"""Backward-compatibility wrapper for cogito.finetune.train."""
import runpy

if __name__ == "__main__":
    runpy.run_module("cogito.finetune.train", run_name="__main__")
