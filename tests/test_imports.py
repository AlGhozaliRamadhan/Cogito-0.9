"""Every module must import with no environment variables and no GPU.

Heavy dependencies (torch, unsloth, trl, datasets) are imported inside
functions, not at module import, so a CPU-only checkout with a bare Python
environment can still import and print help.
"""

import importlib
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODULES = [
    "qwen_sft",
    "qwen_sft.cli",
    "qwen_sft.config",
    "qwen_sft.runtime",
    "qwen_sft.validation",
    "qwen_sft.datasets",
    "qwen_sft.datasets.sources",
    "qwen_sft.datasets.prepare",
    "qwen_sft.datasets.merge",
    "qwen_sft.finetune",
    "qwen_sft.finetune.train",
    "qwen_sft.finetune.merge",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_imports_without_env(module_name, monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    importlib.invalidate_caches()
    importlib.import_module(module_name)


CLI_CHECKS = [
    ["-m", "qwen_sft", "--help"],
    ["-m", "qwen_sft", "chat", "--help"],
    ["-m", "qwen_sft", "train", "--help"],
    ["-m", "qwen_sft", "merge", "--help"],
    ["-m", "qwen_sft", "prepare", "--list"],
    ["-m", "qwen_sft.datasets.merge", "--list"],
    ["-m", "qwen_sft.finetune.train", "--help"],
    ["-m", "qwen_sft.datasets.prepare", "--list"],
]


@pytest.mark.parametrize("command", CLI_CHECKS)
def test_cli_exits_zero(command):
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"python {' '.join(command)} exited {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )


def test_prepare_list_mentions_sources():
    env = os.environ.copy()
    env.pop("HF_TOKEN", None)
    result = subprocess.run(
        [sys.executable, "-m", "qwen_sft", "prepare", "--list"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    for key in ("openthoughts", "astra", "fable", "am"):
        assert key in result.stdout
