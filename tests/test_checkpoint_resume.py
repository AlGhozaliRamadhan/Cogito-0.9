"""Resume-time checkpoint validation (train.py defensive layer).

The Kaggle 12-hour limit and OOM kills land mid-save, so the newest
``checkpoint-*`` dir can be a torn write. These tests pin the contract: a
corrupt checkpoint is never resumed, the previous complete one is picked
instead, declined/forced fresh starts quarantine old checkpoint dirs, and an
explicit ``--resume-from-checkpoint`` pointing at a broken dir fails fast.

No GPU is touched. ``torch`` is imported when available so the deep
``optimizer.pt`` load check is exercised for real (including torn writes);
without it the presence checks alone still run.
"""

import json
import os
from types import SimpleNamespace

import pytest

try:
    import torch
except ImportError:  # pragma: no cover - tensor-free CI environments
    torch = None

from qwen_sft.finetune.train import (
    _best_local_checkpoint,
    _checkpoint_problem,
    _quarantine_local_checkpoints,
    resolve_resume_checkpoint,
)


def make_checkpoint(root, step, *, with_state=True, adapter=True, optimizer=True,
                    adapter_bytes=b"adapter", optimizer_bytes=None):
    """Write a checkpoint-<step> dir with exactly the requested ingredients."""
    path = os.path.join(root, f"checkpoint-{step}")
    os.makedirs(path, exist_ok=True)
    if with_state:
        with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as fh:
            json.dump({"global_step": step}, fh)
    if adapter:
        with open(os.path.join(path, "adapter_model.safetensors"), "wb") as fh:
            fh.write(adapter_bytes)
    if optimizer:
        if optimizer_bytes is None and torch is not None:
            torch.save({"step": step}, os.path.join(path, "optimizer.pt"))
        else:
            with open(os.path.join(path, "optimizer.pt"), "wb") as fh:
                fh.write(optimizer_bytes if optimizer_bytes is not None else b"opt")
    return path


def torn_optimizer_bytes() -> bytes:
    """A file that looks like torch.save output but was cut mid-write."""
    if torch is not None:
        import io

        buffer = io.BytesIO()
        torch.save({"step": 999}, buffer)
        payload = buffer.getvalue()
        return payload[: len(payload) // 2]
    return b"PK\x01\x02torn"  # zip magic, truncated; presence checks only


# ---------------------------------------------------------------- validation


def test_complete_checkpoint_passes_validation(tmp_path):
    path = make_checkpoint(tmp_path, 200)
    assert _checkpoint_problem(path) is None


def test_missing_trainer_state_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200, with_state=False)
    assert _checkpoint_problem(path) == "no parsable trainer_state.json"


def test_unparsable_trainer_state_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200)
    with open(os.path.join(path, "trainer_state.json"), "w", encoding="utf-8") as fh:
        fh.write("{ not json")
    assert _checkpoint_problem(path) == "no parsable trainer_state.json"


def test_missing_optimizer_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200, optimizer=False)
    assert _checkpoint_problem(path) == "optimizer.pt is missing or empty"


def test_empty_optimizer_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200, optimizer=True, optimizer_bytes=b"")
    assert _checkpoint_problem(path) == "optimizer.pt is missing or empty"


def test_missing_adapter_weights_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200, adapter=False)
    assert _checkpoint_problem(path) == "adapter weights are missing or empty"


@pytest.mark.skipif(torch is None, reason="deep load check needs torch")
def test_torn_optimizer_is_flagged(tmp_path):
    path = make_checkpoint(tmp_path, 200, optimizer_bytes=torn_optimizer_bytes())
    problem = _checkpoint_problem(path)
    assert problem is not None and "optimizer.pt is unreadable" in problem


# ---------------------------------------------------------------- picking


def test_best_picks_newest_valid_checkpoint(tmp_path):
    assert _best_local_checkpoint(tmp_path) == (None, [])
    newest = make_checkpoint(tmp_path, 300)
    older = make_checkpoint(tmp_path, 100)
    best, rejected = _best_local_checkpoint(tmp_path)
    assert best == newest
    assert rejected == []
    assert os.path.isdir(older)


def test_best_skips_torn_newest_for_older_complete(tmp_path):
    make_checkpoint(tmp_path, 400, optimizer_bytes=torn_optimizer_bytes())
    older = make_checkpoint(tmp_path, 200)
    best, rejected = _best_local_checkpoint(tmp_path)
    assert best == older
    assert [os.path.basename(path) for path, _ in rejected] == ["checkpoint-400"]


def test_best_reports_every_rejection_when_all_are_bad(tmp_path):
    make_checkpoint(tmp_path, 100, with_state=False)
    make_checkpoint(tmp_path, 200, optimizer=False)
    best, rejected = _best_local_checkpoint(tmp_path)
    assert best is None
    assert len(rejected) == 2
    assert all(reason for _, reason in rejected)


def test_best_ignores_non_checkpoint_dirs(tmp_path):
    # hub_resume/last-checkpoint and previous_run artifacts live next door.
    os.makedirs(os.path.join(tmp_path, "hub_resume", "last-checkpoint"))
    os.makedirs(os.path.join(tmp_path, "stale-checkpoint-900"))
    assert _best_local_checkpoint(tmp_path) == (None, [])


# ---------------------------------------------------------------- quarantine


def test_quarantine_moves_only_checkpoint_dirs(tmp_path):
    make_checkpoint(tmp_path, 100)
    make_checkpoint(tmp_path, 200)
    os.makedirs(os.path.join(tmp_path, "hub_resume", "last-checkpoint"))
    os.makedirs(os.path.join(tmp_path, "stale-checkpoint-50"))

    moved = _quarantine_local_checkpoints(tmp_path)

    assert len(moved) == 2
    for path in moved:
        assert os.path.basename(path).startswith("stale-checkpoint-")
    assert not glob_prefix(tmp_path, "checkpoint-1")
    assert glob_prefix(tmp_path, "stale-checkpoint-1")


def test_quarantine_on_empty_dir_reports_nothing(tmp_path):
    assert _quarantine_local_checkpoints(tmp_path) == []


# ---------------------------------------------------------------- resolve


def _args(tmp_path, **overrides):
    base = dict(
        resume_from_checkpoint=None,
        fresh=False,
        resume_from_hub=True,
        push_checkpoints=False,
        hub_repo="some/repo",
        training_output_dir=str(tmp_path),
        hub_private=False,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def glob_prefix(root, prefix):
    return [name for name in os.listdir(root) if name.startswith(prefix)]


def test_explicit_resume_path_is_validated(tmp_path):
    torn = make_checkpoint(tmp_path, 100, optimizer_bytes=torn_optimizer_bytes())
    args = _args(tmp_path, resume_from_checkpoint=torn)
    with pytest.raises(SystemExit) as exit_info:
        resolve_resume_checkpoint(args)
    assert "unusable" in str(exit_info.value)


def test_explicit_resume_path_used_when_complete(tmp_path):
    path = make_checkpoint(tmp_path, 100)
    args = _args(tmp_path, resume_from_checkpoint=path)
    assert resolve_resume_checkpoint(args) == path


def test_fresh_flag_quarantines_and_returns_none(tmp_path):
    make_checkpoint(tmp_path, 100)
    fresh_dir = _args(tmp_path, fresh=True)
    assert resolve_resume_checkpoint(fresh_dir) is None
    assert glob_prefix(tmp_path, "stale-checkpoint-")
    assert not glob_prefix(tmp_path, "checkpoint-1")


def test_declined_resume_quarantines(tmp_path, monkeypatch):
    make_checkpoint(tmp_path, 100)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    assert resolve_resume_checkpoint(_args(tmp_path)) is None
    assert glob_prefix(tmp_path, "stale-checkpoint-")
    assert not glob_prefix(tmp_path, "checkpoint-1")


def test_accepted_resume_uses_local_checkpoint(tmp_path, monkeypatch):
    path = make_checkpoint(tmp_path, 100)
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    assert resolve_resume_checkpoint(_args(tmp_path)) == path
    assert glob_prefix(tmp_path, "checkpoint-")


def test_non_interactive_stdin_continues(tmp_path, monkeypatch):
    """Batch reruns (no stdin, e.g. Kaggle) must not abort resume prompting."""
    path = make_checkpoint(tmp_path, 100)

    def raising_input(_prompt):
        raise EOFError

    monkeypatch.setattr("builtins.input", raising_input)
    assert resolve_resume_checkpoint(_args(tmp_path)) == path


# ---------------------------------------------------------------- multi-rank


def test_multi_rank_without_explicit_decision_is_refused(tmp_path, monkeypatch):
    """Two torchrun ranks cannot share stdin: no flag -> refuse to start."""
    make_checkpoint(tmp_path, 100)
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(SystemExit) as exit_info:
        resolve_resume_checkpoint(_args(tmp_path))
    assert "torchrun" in str(exit_info.value)


def test_multi_rank_explicit_resume_passes(tmp_path, monkeypatch):
    path = make_checkpoint(tmp_path, 100)
    monkeypatch.setenv("WORLD_SIZE", "2")
    assert resolve_resume_checkpoint(_args(tmp_path, resume_from_checkpoint=path)) == path


def test_multi_rank_fresh_passes_without_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    make_checkpoint(tmp_path, 100)
    monkeypatch.setenv("LOCAL_RANK", "1")  # rank 1 must not quarantine in parallel
    assert resolve_resume_checkpoint(_args(tmp_path, fresh=True)) is None
    assert glob_prefix(tmp_path, "checkpoint-")  # left for rank 0, which quarantines
    monkeypatch.setenv("LOCAL_RANK", "0")
    assert resolve_resume_checkpoint(_args(tmp_path, fresh=True)) is None
    assert glob_prefix(tmp_path, "stale-checkpoint-")
