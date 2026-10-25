"""main.py / train.py wiring for the offline pretokenize cache.

The runner must build the cache before torchrun and only point the ranks at a
cache it verified; the trainer must fail fast on a mismatch and, when the
cache is used, pin the collator settings that replace the in-rank preparation.
"""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

MAIN = Path(__file__).resolve().parents[1] / "main.py"


def _runner_args(**overrides):
    base = dict(model="some/model", dataset_repo="d", dataset_file="f",
                hub_repo="h/r", public=False, max_samples=0, max_seq_length=4096,
                epochs=2, save_steps=200, resume=True, gpus=2, pretokenize=True)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_no_pretokenize_flag_flips_default():
    import main as runner

    parsed = runner.parse_args(["--no-pretokenize"])
    assert parsed.pretokenize is False
    assert runner.parse_args([]).pretokenize is True


def test_build_train_cmd_passes_verified_cache(monkeypatch, tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "resolve_resume", lambda args, token: None)
    args = _runner_args(pretokenized_dir="cache/pretokenized",
                        training_output_dir=str(tmp_path))
    cmd = runner.build_train_cmd(args, "data.jsonl", "tok", 2)
    assert cmd[cmd.index("--pretokenized-dir") + 1] == "cache/pretokenized"


def test_build_train_cmd_omits_absent_cache(monkeypatch, tmp_path):
    """A runner without the attribute (older tests/paths) trains as before."""
    import main as runner

    monkeypatch.setattr(runner, "resolve_resume", lambda args, token: None)
    cmd = runner.build_train_cmd(
        _runner_args(training_output_dir=str(tmp_path)), "data.jsonl", "tok", 2)
    assert "--pretokenized-dir" not in cmd


def test_ensure_pretokenized_reuses_current_cache(monkeypatch, tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "PRETOKENIZED_DIR", str(tmp_path))
    source = tmp_path / "mix.jsonl"
    source.write_text("x" * 10, encoding="utf-8")
    from qwen_sft.finetune import pretokenize as pre

    pre.write_meta(tmp_path, "some/model", 4096,
                   masked=True, rows=1, sources=[(str(source), 10)])
    args = _runner_args(pretokenized_dir=str(tmp_path))

    def explode(cmd, **kwargs):
        raise AssertionError("a current cache must not rebuild")

    monkeypatch.setattr(runner, "run", explode)
    out = runner.ensure_pretokenized(args, str(source))
    assert out == str(tmp_path)


def test_ensure_pretokenized_rejects_a_build_without_matching_meta(monkeypatch,
                                                                   tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "PRETOKENIZED_DIR", str(tmp_path))
    source = tmp_path / "mix.jsonl"
    source.write_text("x" * 10, encoding="utf-8")
    args = _runner_args(pretokenized_dir=str(tmp_path))

    def fake_run(cmd, **kwargs):
        # A build that produces no readable meta.json must NOT be handed to
        # the ranks; the runner falls back to in-rank preparation instead.
        return None

    monkeypatch.setattr(runner, "run", fake_run)
    out = runner.ensure_pretokenized(args, str(source))
    assert out is None


def test_ensure_pretokenized_adopts_a_good_subprocess_build(monkeypatch, tmp_path):
    import main as runner

    monkeypatch.setattr(runner, "PRETOKENIZED_DIR", str(tmp_path))
    source = tmp_path / "mix.jsonl"
    source.write_text("x" * 10, encoding="utf-8")
    args = _runner_args(pretokenized_dir=str(tmp_path))

    def fake_run(cmd, **kwargs):
        from qwen_sft.finetune.pretokenize import write_meta

        write_meta(tmp_path, "some/model", 4096, masked=True, rows=3,
                   sources=[(str(source), 10)])

    monkeypatch.setattr(runner, "run", fake_run)
    assert runner.ensure_pretokenized(args, str(source)) == str(tmp_path)


def test_main_help_offers_pretokenize_opt_out():
    result = subprocess.run([sys.executable, str(MAIN), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "--no-pretokenize" in result.stdout


def test_train_parser_carries_the_cache_flag():
    from qwen_sft.finetune.train import parse_args

    args = parse_args(["--pretokenized-dir", "cache"])
    assert args.pretokenized_dir == "cache"
    assert parse_args([]).pretokenized_dir is None


def test_report_length_stats_prints_the_real_maximum(capsys):
    pytest.importorskip("datasets")
    from datasets import Dataset

    from qwen_sft.finetune.train import report_length_stats

    class TokenPerWord:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0] * len(text.split())}

    # Deliberately out of order: the true max (5) is not the last sample (2).
    dataset = Dataset.from_list([{"text": "a b c d e"}, {"text": "a b"}])
    report_length_stats(dataset, TokenPerWord(), max_seq_length=3)
    out = capsys.readouterr().out
    assert "max 5" in out, out
    assert "over limit 1/2 (50%)" in out, out


def test_skip_prepare_dataset_only_with_a_cache():
    pytest.importorskip("torch")
    pytest.importorskip("trl")
    from qwen_sft.finetune.train import build_sft_config

    def config_args(**overrides):
        base = dict(model="m", dataset=None, epochs=1, batch_size=1,
                    grad_accum=8, lr=1e-4, lora_r=16, lora_alpha=32,
                    lora_dropout=0.0, seed=42, output_dir="o",
                    training_output_dir="t", save_steps=200, logging_steps=10,
                    save_total_limit=2, hub_repo="", push_checkpoints=False,
                    resume_from_hub=False, hub_private=False,
                    max_seq_length=4096, pretokenized_dir=None)
        base.update(overrides)
        return SimpleNamespace(**base)

    plain = build_sft_config(config_args(), "t")
    assert not getattr(plain, "dataset_kwargs", None) or \
        not plain.dataset_kwargs.get("skip_prepare_dataset")

    cached = build_sft_config(
        config_args(pretokenized_dir="cache"), "t")
    assert cached.dataset_kwargs == {"skip_prepare_dataset": True}
