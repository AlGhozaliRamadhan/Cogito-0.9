"""Merge behaviour: dedupe, per-source caps, invalid records, determinism."""

import json

from qwen_sft.datasets.merge import merge_files


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def convo(question):
    return {"messages": [{"role": "user", "content": question},
                         {"role": "assistant", "content": f"answer to {question}"}]}


def test_dedupe_drops_cross_source_repeats(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    write_jsonl(a, [convo("q1"), convo("q2")])
    write_jsonl(b, [convo("q2"), convo("q3")])

    rows, stats = merge_files([("a", str(a)), ("b", str(b))])
    assert len(rows) == 3
    assert stats[0].kept == 2
    assert stats[1].kept == 1
    assert stats[1].duplicates == 1


def test_no_dedupe_keeps_repeats(tmp_path):
    a = tmp_path / "a.jsonl"
    write_jsonl(a, [convo("q1"), convo("q1")])
    rows, stats = merge_files([("a", str(a))], dedupe=False)
    assert len(rows) == 2
    assert stats[0].duplicates == 0


def test_invalid_records_are_counted_not_trained_on(tmp_path):
    path = tmp_path / "mixed.jsonl"
    write_jsonl(path, [
        convo("valid"),
        {"messages": [{"role": "user", "content": "no answer"}]},
        {"messages": []},
        {"messages": [{"role": "user", "content": "q"},
                      {"role": "assistant", "content": "a"}]},
    ])
    rows, stats = merge_files([("mixed", str(path))])
    assert len(rows) == 2
    assert stats[0].invalid == 2


def test_max_per_source_caps_after_dedupe(tmp_path):
    path = tmp_path / "many.jsonl"
    write_jsonl(path, [convo(f"q{i}") for i in range(10)])
    rows, stats = merge_files([("many", str(path))], max_per_source=3)
    assert len(rows) == 3
    assert stats[0].kept == 3
    assert stats[0].read == 10


def test_shuffle_is_deterministic_for_a_seed(tmp_path):
    path = tmp_path / "many.jsonl"
    write_jsonl(path, [convo(f"q{i}") for i in range(20)])

    first, _ = merge_files([("many", str(path))], seed=7)
    second, _ = merge_files([("many", str(path))], seed=7)
    third, _ = merge_files([("many", str(path))], seed=8)

    assert first == second
    assert first != third
