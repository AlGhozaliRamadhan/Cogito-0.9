"""Import-resolution check for the whole cogito package.

Every module must import cleanly with NO environment variables set
(no NVIDIA_API_KEY, no HF_TOKEN) — the package must never die at import
time. Two intentional exceptions:

- cogito.scripts.train is a module-level script whose CUDA check exits(1)
  on CPU-only machines.
- cogito.audio.generate_kokoro requires the optional kokoro package.

CLI entry points are invoked via ``python -m cogito.*`` (no legacy shims).
Module-level exits (CUDA check, missing HF_TOKEN) are validated separately
in test_cli_dispatch via subprocess.
"""
import importlib
import os
import subprocess
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (module, expect_system_exit)
IMPORT_CASES = [
    ("cogito", False),
    ("cogito.validation", False),
    ("cogito.runtime", False),
    ("cogito.llm.api_client", False),
    ("cogito.generators.topics", False),
    ("cogito.generators.agentic_tools", False),
    ("cogito.generators.execution_engine", False),
    ("cogito.generators.human_conversations", False),
    ("cogito.generators.identity_core", False),
    ("cogito.generators.personality_quirks", False),
    ("cogito.generators.philosophical_probing", False),
    ("cogito.generators.retrieval_filter", False),
    ("cogito.generators.heated_conversations", False),
    ("cogito.generators.livemurmur_engine", False),
    ("cogito.generators.generate_assertions", False),
    ("cogito.generators.validator", False),
    ("cogito.scripts.abliterate_cogito", False),
    ("cogito.scripts.merge_lora", False),
    ("cogito.scripts.cleanup_hub", False),
    ("cogito.scripts.dataset_manager", False),
    ("cogito.scripts.upload_dataset_to_hub", False),
    ("cogito.scripts.upload_dense_dataset_to_hub", False),
    ("cogito.scripts.prepare_datasets", False),
    ("cogito.scripts.verify_masking", False),
    ("cogito.scripts.build_dense_dataset", False),
    ("cogito.scripts.merge_datasets", False),
    ("cogito.scripts.train", True),   # module-level CUDA check -> SystemExit(1)
    ("cogito.audio.cogito_voice_fx", False),
    ("cogito.audio.generate_kokoro", False),
    ("cogito.cli", False),
]


@pytest.mark.parametrize("module_name,expect_exit", IMPORT_CASES)
def test_module_imports_cleanly(module_name, expect_exit):
    importlib.invalidate_caches()
    if module_name == "cogito.audio.generate_kokoro":
        kokoro = pytest.importorskip("kokoro")
        assert kokoro is not None
    try:
        importlib.import_module(module_name)
    except SystemExit as exc:
        if expect_exit:
            assert exc.code == 1, f"{module_name} exited with code {exc.code}"
            return
        pytest.fail(f"{module_name} called sys.exit() at import: {exc}")
    else:
        if expect_exit:
            pytest.fail(f"{module_name} should have exited without CUDA, but imported fine")


# CLI entry points run via `python -m cogito.*`. Each is run as a subprocess
# so module-level exits don't kill the suite.
CLI_CHECKS = [
    # (command, expected_exit_code)
    (["-m", "cogito", "--help"], 1),  # runtime GPU pre-check exits 1 on CPU-only
    (["-m", "cogito.scripts.train", "--help"], 1),  # CUDA check -> clean exit 1
    (["-m", "cogito.scripts.upload_dataset_to_hub"], 1),  # no HF_TOKEN -> clean exit 1
    (["-m", "cogito.scripts.upload_dense_dataset_to_hub"], 1),
]


@pytest.mark.parametrize("command,expected_exit", CLI_CHECKS)
def test_cli_dispatch(command, expected_exit):
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == expected_exit, (
        f"python {' '.join(command)} exited {result.returncode} "
        f"(expected {expected_exit})\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
