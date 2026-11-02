"""Eval-harness tests: graders and cache slicing only. No GPU, no network."""

import json

import pytest

from qwen_sft.eval import tasks
from qwen_sft.eval.tasks import (
    CHECKS,
    FORMAT_TASKS,
    extract_last_number,
    grade_format,
    grade_gsm8k,
    load_gsm8k,
    strip_thinking,
)

CANONICAL_PASS = {
    "one_word": "Hello",
    "one_word_paris": "Paris.",
    "number_only_42": "42",
    "yes_no": "NO",
    "three_bullets": "- apple\n- banana\n- cherry",
    "two_sentences": "Rain falls. It wets the ground.",
    "json_ab": '{"a": 1, "b": 2}',
    "banana_triplet": "banana banana banana",
}

CANONICAL_FAIL = {
    "one_word": "Hello there",
    "one_word_paris": "London",
    "number_only_42": "The answer is 42",
    "yes_no": "Maybe",
    "three_bullets": "- apple\n- banana",
    "two_sentences": "One sentence only.",
    "json_ab": '{"a": 1}',
    "banana_triplet": "banana banana",
}


def test_strip_thinking_keeps_only_the_answer_channel():
    raw = f"step 1 ... step 2 ...{tasks.THINK_CLOSE} 42"
    assert strip_thinking(raw) == "42"
    assert strip_thinking("42") == "42"


def test_extract_last_number_handles_commas_and_currency():
    assert extract_last_number("The total is $1,234.50.") == 1234.5
    assert extract_last_number("#### 42") == 42.0
    assert extract_last_number("no digits here") is None


def test_grade_gsm8k_accepts_plain_and_boxed_answers():
    assert grade_gsm8k("I checked twice. The answer is 42.", "#### 42")
    assert grade_gsm8k("\\boxed{1,234}", "#### 1234")
    assert not grade_gsm8k("The answer is 41.", "#### 42")
    assert not grade_gsm8k("no number at all", "#### 42")


def test_format_tasks_are_well_formed():
    ids = [task["id"] for task in FORMAT_TASKS]
    assert len(ids) == len(set(ids))
    for task in FORMAT_TASKS:
        assert task["prompt"].strip()
        assert task["rule"] in CHECKS


@pytest.mark.parametrize("task", FORMAT_TASKS, ids=lambda task: task["id"])
def test_every_rule_passes_its_canonical_answer(task):
    passed, detail = grade_format(task, CANONICAL_PASS[task["rule"]])
    assert passed, f"{task['id']} rejected a canonical answer: {detail}"


@pytest.mark.parametrize("task", FORMAT_TASKS, ids=lambda task: task["id"])
def test_every_rule_rejects_an_obvious_violation(task):
    passed, _ = grade_format(task, CANONICAL_FAIL[task["rule"]])
    assert not passed, f"{task['id']} accepted a violation"


def test_json_rule_tolerates_a_code_fence():
    fenced = '```json\n{"a": 1, "b": 2}\n```'
    passed, detail = tasks.CHECKS["json_ab"](fenced)
    assert passed, detail


def test_format_rule_is_graded_after_thinking_is_removed():
    task = next(task for task in FORMAT_TASKS if task["rule"] == "number_only_42")
    response = f"Let me add. 17 + 25 = 42{tasks.THINK_CLOSE}42"
    assert grade_format(task, response)[0]


def test_unknown_rule_is_a_loud_bug_not_a_silent_skip():
    with pytest.raises(KeyError):
        grade_format({"id": "x", "rule": "does_not_exist"}, "anything")


def test_gsm8k_cache_slices_are_prefixes(tmp_path):
    cache = tmp_path / "gsm8k.jsonl"
    rows = [
        {"id": f"gsm8k-{index}", "question": f"q{index}", "gold": f"#### {index}"}
        for index in range(5)
    ]
    with open(cache, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    short = load_gsm8k(2, 42, str(cache))
    long = load_gsm8k(4, 42, str(cache))
    assert len(short) == 2
    assert short == long[:2]
    assert [row["id"] for row in long] == [f"gsm8k-{i}" for i in range(4)]
