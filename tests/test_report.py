"""Training-log persistence + the research-grade loss chart.

No GPU touched: the callback and the report module are exercised with fake
TrainerState/args objects and real matplotlib rendering (skipped when
matplotlib is absent).
"""

import json
import os
from types import SimpleNamespace

import pytest

try:
    import matplotlib  # noqa: F401
except ImportError:
    matplotlib = None

from qwen_sft.finetune.train import build_jsonl_logger
from qwen_sft.report import dedupe, read_rows, render_chart, rows_from_state


def rows(*steps, loss_start=2.5, drop=0.05):
    """Synthetic log rows shaped like the trainer's own output."""
    return [
        {"loss": round(loss_start - drop * step / 10, 4),
         "learning_rate": 2e-4,
         "grad_norm": 0.9,
         "epoch": round(step / 6400, 2),
         "global_step": step}
        for step in steps
    ]


def _state(step, world_zero=True):
    return SimpleNamespace(global_step=step, epoch=step / 6400,
                           is_world_process_zero=world_zero)


def _logs(step):
    return {"loss": 2.0, "learning_rate": 2e-4, "epoch": step / 6400,
            "step": step}


# ------------------------------------------------------------------ dedupe


def test_dedupe_keeps_last_per_step_sorted():
    rows_in = rows(10, 20) + rows(10, 30, loss_start=1.9)
    assert [r["global_step"] for r in dedupe(rows_in)] == [10, 20, 30]
    assert dedupe(rows_in)[0]["loss"] == 1.85       # last writer wins


def test_dedupe_ignores_rows_without_step():
    assert dedupe([{"loss": 1.0}, {"global_step": None}]) == []


def test_read_rows_skips_broken_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_text('{"loss": 2.0, "global_step": 10}\nzzz\n\n'
                    '{"loss": 1.0, "global_step": 20}\n', encoding="utf-8")
    assert [r["global_step"] for r in read_rows(str(path))] == [10, 20]


def test_rows_from_state_reads_log_history(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"global_step": 20,
                                "log_history": rows(10, 20)}), encoding="utf-8")
    assert [r["global_step"] for r in rows_from_state(str(path))] == [10, 20]


# ------------------------------------------------------------------ callback


def test_callback_appends_log_rows_and_drops_step_key(tmp_path):
    log = str(tmp_path / "train_log.jsonl")
    cb = build_jsonl_logger(log)
    cb.on_log(None, _state(10), None, logs=_logs(10))
    cb.on_log(None, _state(20), None, logs=_logs(20))
    saved = read_rows(log)
    assert [r["global_step"] for r in saved] == [10, 20]
    assert all("step" not in r for r in saved)      # step key collapsed


def test_callback_non_zero_rank_writes_nothing(tmp_path):
    log = str(tmp_path / "train_log.jsonl")
    cb = build_jsonl_logger(log)
    cb.on_log(None, _state(10, world_zero=False), None, logs=_logs(10))
    assert not os.path.exists(log)


def test_callback_seeds_from_resume_state(tmp_path):
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "trainer_state.json").write_text(
        json.dumps({"global_step": 20, "log_history": rows(10, 20)}),
        encoding="utf-8")
    log = tmp_path / "training_output" / "train_log.jsonl"
    cb = build_jsonl_logger(str(log), seed_from=str(resume))
    cb.on_log(None, _state(30), None, logs=_logs(30))
    assert [r["global_step"] for r in read_rows(str(log))] == [10, 20, 30]


def test_callback_existing_seed_file_wins_over_state(tmp_path):
    log = str(tmp_path / "train_log.jsonl")
    with open(log, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(rows(10)[0]) + "\n")
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "trainer_state.json").write_text(
        json.dumps({"global_step": 20, "log_history": rows(5, 10, 20)}),
        encoding="utf-8")
    cb = build_jsonl_logger(log, seed_from=str(resume))
    cb.on_log(None, _state(30), None, logs=_logs(30))
    # The existing file (step 10) wins its slot; seeding is skipped; 30 is new.
    kept = read_rows(log)
    assert [r["global_step"] for r in kept] == [10, 30]


def test_callback_mirrors_history_into_saved_checkpoint(tmp_path):
    out_dir = tmp_path / "training_output"
    log = str(out_dir / "train_log.jsonl")
    cb = build_jsonl_logger(log)
    cb.on_log(None, _state(10), None, logs=_logs(10))
    cb.on_log(None, _state(20), None, logs=_logs(20))
    checkpoint = out_dir / "checkpoint-20"
    checkpoint.mkdir()
    args = SimpleNamespace(output_dir=str(out_dir))
    cb.on_save(args, _state(20), None)
    mirrored = read_rows(str(checkpoint / "train_log.jsonl"))
    assert [r["global_step"] for r in mirrored] == [10, 20]


def test_callback_mirror_skips_missing_checkpoint_dir(tmp_path, capsys):
    out_dir = tmp_path / "elsewhere"
    cb = build_jsonl_logger(str(tmp_path / "train_log.jsonl"))
    args = SimpleNamespace(output_dir=str(out_dir))
    cb.on_save(args, _state(20), None)              # no crash, no mirror
    assert not os.path.isdir(out_dir)


def test_callback_mirror_is_rank_zero_only(tmp_path):
    out_dir = tmp_path / "training_output"
    log = str(out_dir / "train_log.jsonl")
    cb = build_jsonl_logger(log)
    cb.on_log(None, _state(10), None, logs=_logs(10))
    checkpoint = out_dir / "checkpoint-10"
    checkpoint.mkdir()
    args = SimpleNamespace(output_dir=str(out_dir))
    cb.on_save(args, _state(10, world_zero=False), None)
    assert not (checkpoint / "train_log.jsonl").exists()


# ------------------------------------------------------------------ plotting


@pytest.mark.skipif(matplotlib is None, reason="matplotlib missing")
def test_render_chart_from_jsonl(tmp_path):
    log = tmp_path / "train_log.jsonl"
    with open(log, "w", encoding="utf-8") as fh:
        for row in rows(10, 20, 30, 40):
            fh.write(json.dumps(row) + "\n")
    out = tmp_path / "chart.png"
    result = render_chart(str(tmp_path), out_path=str(out))
    assert result and result[0] == str(out) and os.path.getsize(out) > 10_000
    assert result[1] == str(log)


@pytest.mark.skipif(matplotlib is None, reason="matplotlib missing")
def test_render_chart_falls_back_to_trainer_state(tmp_path):
    (tmp_path / "trainer_state.json").write_text(
        json.dumps({"global_step": 40, "log_history": rows(10, 20, 30, 40)}),
        encoding="utf-8")
    result = render_chart(str(tmp_path), out_path=str(tmp_path / "c.png"))
    assert result and "trainer_state.json" in result[1]


@pytest.mark.skipif(matplotlib is None, reason="matplotlib missing")
def test_render_chart_returns_none_when_empty(tmp_path):
    assert render_chart(str(tmp_path), out_path=str(tmp_path / "c.png")) is None
