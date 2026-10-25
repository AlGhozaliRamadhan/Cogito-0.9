"""Offline pretokenize cache: one CPU build consumed by both DDP ranks.

Covers the crash chain it exists to break: the cache identity is pinned (a
stale cache must fail loudly, not silently train different tokens), the
masking/filtering semantics mirror train_on_responses_only, and main.py only
points the ranks at a cache it actually produced.
"""

import json
from types import SimpleNamespace

import pytest

from qwen_sft.finetune.pretokenize import (
    matches_meta,
    pretokenize_dataset,
    read_meta,
    resolve_add_special_tokens,
    write_meta,
)

MODEL = "m/m"
MAX_LEN = 64


class FakeWordTokenizer:
    """Characters-as-tokens JWT: 'Ω' -> 999 so tests can mark a poison row."""

    bos_token = None

    def __init__(self, chat_template=None):
        self.chat_template = chat_template

    def __call__(self, texts, truncation=False, max_length=None,
                 add_special_tokens=True, **kwargs):
        plain = texts if isinstance(texts, list) else [texts]
        rows = []
        for text in plain:
            ids = [999 if char == "Ω" else ord(char) for char in text]
            if truncation and max_length:
                ids = ids[:max_length]
            rows.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
        return {"input_ids": [row["input_ids"] for row in rows],
                "attention_mask": [row["attention_mask"] for row in rows]}


def fake_omega_mask(examples):
    """Test double for Unsloth's mask: rows containing 999 lose everything."""
    return {"labels": [[-100] * len(ids) if 999 in ids else list(ids)
                       for ids in examples["input_ids"]]}


def chat_text(user, assistant):
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n")


# --- cache identity -----------------------------------------------------


def _sources():
    return [("data.jsonl", 42)]


def test_meta_round_trip_and_identity_checks(tmp_path):
    write_meta(tmp_path, MODEL, MAX_LEN, masked=True, rows=7, sources=_sources())
    meta = read_meta(tmp_path)
    assert meta["model"] == MODEL
    assert meta["max_seq_length"] == MAX_LEN
    assert meta["rows"] == 7
    assert matches_meta(meta, MODEL, MAX_LEN, _sources())
    # Any silent drift must force a rebuild, never a reuse.
    assert not matches_meta(meta, "other/model", MAX_LEN, _sources())
    assert not matches_meta(meta, MODEL, 2048, _sources())
    assert not matches_meta(meta, MODEL, MAX_LEN, [("data.jsonl", 43)])
    assert not matches_meta(meta, MODEL, MAX_LEN, [("other.jsonl", 42)])


def test_matches_meta_refuses_untyped_or_missing_meta(tmp_path):
    assert read_meta(tmp_path) is None
    assert not matches_meta(None, MODEL, MAX_LEN, _sources())
    (tmp_path / "meta.json").write_text("{ not json", encoding="utf-8")
    assert read_meta(tmp_path) is None
    assert not matches_meta(read_meta(tmp_path), MODEL, MAX_LEN, _sources())
    bad = {"model": MODEL, "max_seq_length": MAX_LEN,
           "sources": [{"path": "data.jsonl", "size": 42}]}
    assert not matches_meta(bad, MODEL, MAX_LEN, _sources()), \
        "missing the masked flag cannot be trusted as either choice"


# --- tokenization guards ------------------------------------------------


class FakeBosTokenizer(FakeWordTokenizer):
    def __init__(self, bos, chat_template):
        super().__init__(chat_template)
        self.bos_token = bos


def test_add_special_tokens_follows_the_double_bos_guard():
    assert resolve_add_special_tokens(FakeWordTokenizer(), "text") is True, \
        "Qwen has no BOS; the tokenizer keeps its default"
    assert resolve_add_special_tokens(
        FakeBosTokenizer("<s>", "plain template"), "text") is True
    assert resolve_add_special_tokens(
        FakeBosTokenizer("<s>", "template embeds <s> inside"), "text") is False
    assert resolve_add_special_tokens(
        FakeBosTokenizer("<s>", "clean"), "<s>already rendered") is False


# --- dataset preparation -----------------------------------------------


def _rows_dataset():
    pytest.importorskip("datasets")
    from datasets import Dataset

    return Dataset.from_list([
        {"text": chat_text("hi", "hey")},
        {"text": chat_text("Ω", "nope")},        # the label killer
    ])


def test_pretokenize_dataset_masks_with_injected_mask():
    pytest.importorskip("datasets")
    tokenized = pretokenize_dataset(
        _rows_dataset(), FakeWordTokenizer(), MAX_LEN,
        mask_function=fake_omega_mask)
    assert len(tokenized) == 1, "the fully-masked row must be dropped"
    assert {"input_ids", "attention_mask", "labels"} == \
        set(tokenized.column_names)
    assert tokenized[0]["labels"] == tokenized[0]["input_ids"], \
        "surviving spans keep their token ids as labels"


def test_pretokenize_dataset_without_mask_trains_every_token():
    pytest.importorskip("datasets")
    tokenized = pretokenize_dataset(
        _rows_dataset(), FakeWordTokenizer(), MAX_LEN, mask_function=False)
    assert len(tokenized) == 2, "no -100 labels, so nothing is dropped"
    for row in tokenized:
        assert row["labels"] == row["input_ids"]


def test_pretokenize_dataset_refuses_a_dataset_with_no_signal():
    pytest.importorskip("datasets")
    from datasets import Dataset

    only = Dataset.from_list([{"text": chat_text("Ω", "maybe")}])
    with pytest.raises(SystemExit):
        pretokenize_dataset(only, FakeWordTokenizer(), MAX_LEN,
                            mask_function=fake_omega_mask)


# --- cache round trip ---------------------------------------------------


def _loader_args(directory, **overrides):
    base = dict(pretokenized_dir=str(directory), model=MODEL,
                max_seq_length=MAX_LEN, no_mask_prompt=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_load_pretokenized_round_trips_a_saved_build(tmp_path):
    pytest.importorskip("datasets")
    from qwen_sft.finetune.pretokenize import load_pretokenized

    tokenized = pretokenize_dataset(
        _rows_dataset(), FakeWordTokenizer(), MAX_LEN,
        mask_function=fake_omega_mask)
    tokenized.save_to_disk(str(tmp_path))
    meta = {"kind": "qwen_sft_pretokenized", "model": MODEL,
            "max_seq_length": MAX_LEN, "masked": True, "rows": 1,
            "sources": [{"path": "data.jsonl", "size": 42}]}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    dataset = load_pretokenized(_loader_args(tmp_path))
    assert len(dataset) == 1
    assert set(dataset.column_names) >= {"input_ids", "labels"}


def test_load_pretokenized_fails_loudly_on_drift(tmp_path):
    from qwen_sft.finetune.pretokenize import load_pretokenized

    meta = {"kind": "qwen_sft_pretokenized", "model": MODEL,
            "max_seq_length": MAX_LEN, "masked": True, "rows": 1,
            "sources": []}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(SystemExit):
        load_pretokenized(_loader_args(tmp_path, model="other/who"))
    with pytest.raises(SystemExit):
        load_pretokenized(_loader_args(tmp_path, max_seq_length=2048))
    with pytest.raises(SystemExit):
        load_pretokenized(_loader_args(tmp_path, no_mask_prompt=True))


def test_load_pretokenized_explains_an_unreadable_cache(tmp_path):
    from qwen_sft.finetune.pretokenize import load_pretokenized

    meta = {"kind": "qwen_sft_pretokenized", "model": MODEL,
            "max_seq_length": MAX_LEN, "masked": True, "sources": []}
    (tmp_path / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_pretokenized(_loader_args(tmp_path))  # meta ok, dataset absent

    empty = tmp_path / "unbuilt"
    empty.mkdir()
    with pytest.raises(SystemExit):  # no meta at all
        load_pretokenized(_loader_args(empty))
