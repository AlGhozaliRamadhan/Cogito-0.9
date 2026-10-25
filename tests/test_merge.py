"""The merger must never substitute a different base model."""

import json

import pytest

from qwen_sft.finetune.merge import resolve_base_model


def test_explicit_model_wins(tmp_path):
    assert resolve_base_model(str(tmp_path), "me/uncensored-4b") == "me/uncensored-4b"


def test_base_is_read_from_adapter_config(tmp_path):
    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "DreamFast/uncensored-4b"}),
        encoding="utf-8",
    )
    assert resolve_base_model(str(tmp_path), None) == "DreamFast/uncensored-4b"


def test_missing_base_fails_instead_of_defaulting(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        resolve_base_model(str(tmp_path), None)
    assert "--model" in str(excinfo.value)
