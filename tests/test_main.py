"""Smoke tests for the Kaggle bootstrap (main.py). CLI must stay dependency-free."""

import subprocess
import sys
from pathlib import Path

MAIN = Path(__file__).resolve().parents[1] / "main.py"


def test_main_help_runs_without_dependencies():
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    for flag in ("--max-samples", "--epochs", "--save-steps", "--no-resume",
                 "--hub-repo", "--max-seq-length", "--gpus"):
        assert flag in result.stdout


def test_main_defaults_pin_kaggle_week_run():
    """Help output must reflect the runner defaults so regressions are visible.

    Defaults are a one-week Kaggle plan: ~30 GPU-hours of weekly quota at a
    measured ~80 s/step (effective batch 16) -> 20k rows x 1 epoch x 100-step
    saves ~= 28h.
    """
    result = subprocess.run(
        [sys.executable, str(MAIN), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "(default: 20000)" in result.stdout  # max-samples: one weekly quota
    assert "(default: 1)" in result.stdout      # epochs: a single pass
    assert "4096" in result.stdout              # max-seq-length
    assert "(default: 100)" in result.stdout    # save-steps: ~2h between saves


def _runner_args(**overrides):
    from types import SimpleNamespace

    base = dict(model="some/model", dataset_repo="d", dataset_file="f",
                hub_repo="h/r", public=False, max_samples=0, max_seq_length=4096,
                epochs=2, save_steps=200, resume=True, gpus=2)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_build_train_cmd_uses_torchrun_for_two_gpus(monkeypatch, tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "resolve_resume", lambda args, token: None)
    cmd = runner.build_train_cmd(
        _runner_args(training_output_dir=str(tmp_path)), "data.jsonl", "tok", 2)
    assert cmd[:5] == [sys.executable, "-m", "torch.distributed.run",
                       "--nproc_per_node=2", "-m"]
    assert cmd[5:7] == ["qwen_sft", "train"]
    i = cmd.index("--grad-accum")
    assert cmd[i + 1] == "8"                    # halved: 2 ranks keep effective batch 16
    assert "--fresh" in cmd                     # explicit, so every rank agrees


def test_build_train_cmd_single_gpu_keeps_accum(monkeypatch, tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "resolve_resume", lambda args, token: "checkpoint-400")
    cmd = runner.build_train_cmd(
        _runner_args(gpus=1, training_output_dir=str(tmp_path)), "data.jsonl", "tok", 1)
    assert cmd[:5] == [sys.executable, "-m", "qwen_sft", "train", "--model"]
    i = cmd.index("--grad-accum")
    assert cmd[i + 1] == "16"
    j = cmd.index("--resume-from-checkpoint")
    assert cmd[j + 1] == "checkpoint-400"       # explicit, not prompted inside ranks


def test_detect_gpu_count_survives_probe_failure(monkeypatch):
    import main as runner

    def broken(cmd, **kwargs):
        raise OSError("no torch here")

    monkeypatch.setattr(runner.subprocess, "run", broken)
    assert runner.detect_gpu_count() == 1
