"""Registry + per-source normalizer tests. No network, no datasets library."""

import json

import pytest

from qwen_sft.datasets import sources
from qwen_sft.datasets.sources import SOURCES, Source, describe, get_source
from qwen_sft.validation import parse_messages, validate_chat


def test_expected_sources_are_registered():
    assert set(SOURCES) == {
        "fable", "astra", "astra_mirror", "openthoughts", "am",
        "fable_code", "distill_multi", "distill_cleaned", "medical",
    }
    for source in SOURCES.values():
        assert source.repo_id
        assert callable(source.normalizer)


def test_instruction_response_decodes_python_repr_instruction():
    # saidutta69's Parquet stores the struct as a Python repr, not JSON.
    record = {
        "instruction": "{'messages': [{'role': 'user', 'content': 'Is the sky blue?'}]}",
        "response": "Yes.",
    }
    messages = get_source("distill_cleaned").normalize(record)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Is the sky blue?"
    assert validate_chat(messages)


def test_instruction_response_accepts_plain_text_instruction():
    # Other Manusagents shards store a bare instruction, no message list.
    record = {"instruction": "Explain BFS vs DFS.", "response": "BFS uses a queue."}
    messages = get_source("distill_multi").normalize(record)
    assert [m["role"] for m in messages] == ["user", "assistant"]
    assert messages[0]["content"] == "Explain BFS vs DFS."
    assert validate_chat(messages)


def test_broken_struct_repr_is_dropped_not_trained_on():
    record = {"instruction": "{'messages': [oops]", "response": "a"}
    assert get_source("distill_multi").normalize(record) == []


def test_instruction_response_accepts_dict_and_json_string():
    source = get_source("distill_multi")
    as_dict = {"instruction": {"messages": [{"role": "user", "content": "q"}]},
               "response": "a"}
    as_json = {"instruction": json.dumps(as_dict["instruction"]), "response": "a"}
    assert source.normalize(as_dict) == source.normalize(as_json)
    assert source.normalize(as_dict)[0]["content"] == "q"


def test_redteam_sources_are_never_sft_sources(monkeypatch):
    from qwen_sft.datasets.sources import REDTEAM_SOURCES

    assert REDTEAM_SOURCES
    assert not (set(SOURCES) & set(REDTEAM_SOURCES))
    for meta in REDTEAM_SOURCES.values():
        assert meta["repo_id"] and meta["kind"] and meta["why"]


def test_aliases_resolve():
    assert get_source("open-thoughts").key == "openthoughts"
    assert get_source("deepseek").key == "am"
    with pytest.raises(KeyError):
        get_source("nope")


def test_am_maps_think_content_to_reasoning():
    record = {"messages": [
        {"role": "user", "content": "What is 17*23?", "info": None},
        {"role": "assistant", "content": "391", "info": {
            "think_content": "17*23 = 17*20 + 17*3 = 340 + 51 = 391",
            "answer_content": "391",
            "source": "gsm8k",
        }},
    ]}
    messages = get_source("am").normalize(record)
    assert messages[1]["content"] == "391"
    assert messages[1]["reasoning_content"].startswith("17*23")
    assert validate_chat(messages)


def test_am_falls_back_to_answer_content_when_empty():
    record = {"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "info": {"answer_content": "final"}},
    ]}
    messages = get_source("am").normalize(record)
    assert messages[1]["content"] == "final"


def test_prepare_source_accepts_adhoc_source_and_skips_invalid(tmp_path, monkeypatch):
    records = [
        {"messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]},
        {"messages": [{"role": "user", "content": "q2"}, {"role": "user", "content": "dangling"}]},
        {"messages": [{"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"}]},
    ]
    monkeypatch.setattr(sources, "_iter_dataset", lambda *args, **kwargs: iter(records))
    out = tmp_path / "adhoc.jsonl"
    stats = sources.prepare_source(
        Source(key="adhoc", repo_id="someone/whatever",
               normalizer=parse_messages, description="test"),
        str(out), limit=5, shuffle_buffer=0,
    )
    assert (stats.written, stats.skipped, stats.scanned) == (2, 1, 3)
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert all(row["messages"][-1]["role"] == "assistant" for row in rows)


def test_empty_prepare_leaves_no_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sources, "_iter_dataset",
        lambda *args, **kwargs: iter([{"messages": [{"role": "user", "content": "only"}]}]),
    )
    out = tmp_path / "empty.jsonl"
    stats = sources.prepare_source("astra", str(out), limit=5, shuffle_buffer=0)
    assert stats.written == 0
    assert not out.exists()


def test_raw_jsonl_loader_parses_lines(tmp_path, monkeypatch):
    import huggingface_hub

    path = tmp_path / "raw.jsonl"
    path.write_text('{"n": 1}\n\n{"n": 2}\n', encoding="utf-8")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(path))

    source = Source(key="raw", repo_id="someone/whatever",
                    normalizer=parse_messages, description="test",
                    raw_jsonl="raw.jsonl")
    records = sources._iter_raw_jsonl(source, seed=42, shuffle_buffer=0)
    assert [record["n"] for record in records] == [1, 2]


def test_prepare_source_uses_raw_loader_when_configured(tmp_path, monkeypatch):
    records = [{"conversation": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]}]
    monkeypatch.setattr(sources, "_iter_raw_jsonl", lambda *a, **k: records)

    source = Source(key="raw", repo_id="someone/whatever",
                    normalizer=sources._nested_messages("conversation"),
                    description="test", raw_jsonl="raw.jsonl")
    out = tmp_path / "raw.jsonl"
    stats = sources.prepare_source(source, str(out), limit=5, shuffle_buffer=0)
    assert stats.written == 1
    assert json.loads(out.read_text(encoding="utf-8"))["messages"][-1]["role"] == "assistant"


def test_describe_lists_every_source():
    text = describe()
    for key in SOURCES:
        assert key in text
    assert "CC-BY-NC-4.0" in text
