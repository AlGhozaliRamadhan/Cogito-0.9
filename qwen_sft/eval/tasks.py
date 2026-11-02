"""Task definitions and graders for the eval harness.

Pure Python on purpose: no torch, no transformers, no network at import time.
``qwen_sft.eval.bench`` does the GPU work; this module only decides whether an
answer counts as correct, so every grading rule is unit-testable on a CPU-only
checkout.

Two task families:

- ``gsm8k`` — 1,319 competition math problems with a checkable final number.
  Downloaded once and cached as JSONL, then sliced deterministically so a
  100-item run is a prefix of a 200-item run (comparable across checkpoints).
- ``format`` — eight rule-graded instruction-following checks. These catch the
  most likely SFT regression: a model that has learned long reasoning traces
  and can no longer answer "reply with exactly one word".

GSM8K overlaps reasoning-distillation corpora, so a gain here is a smoke test,
not evidence of capability. A drop is reliable.
"""

import json
import os
import random
import re

NUMBER_RE = re.compile(r"-?\$?\d[\d,]*(?:\.\d+)?")

THINK_CLOSE = "<｜end▁of▁thinking｜>"
# Enough to show what the model did without turning the report into a corpus.
PREVIEW_CHARS = 240


def strip_thinking(text: str) -> str:
    """Return the answer channel only.

    Qwen-style templates put the chain of thought before ``THINK_CLOSE``; grading
    the whole output would fail every format rule for a reason that has nothing
    to do with instruction following.
    """
    if THINK_CLOSE in text:
        return text.rsplit(THINK_CLOSE, 1)[1].strip()
    return text.strip()


def extract_last_number(text: str) -> float | None:
    """Last numeric literal in ``text``, commas and currency stripped."""
    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1].replace(",", "").replace("$", ""))
    except ValueError:
        return None


def grade_gsm8k(response: str, gold: str) -> bool:
    """Exact numeric match against the ``#### <n>`` tail of the reference."""
    predicted = extract_last_number(strip_thinking(response))
    expected = extract_last_number(gold)
    if predicted is None or expected is None:
        return False
    return abs(predicted - expected) < 1e-4


def gsm8k_prompt(question: str) -> str:
    return (
        "Solve the problem. Show your reasoning, then finish with the line "
        "'The answer is <number>'.\n\n" + question.strip()
    )


def _bare(text: str) -> str:
    """Strip wrapping quotes, whitespace, and a trailing period."""
    return text.strip().strip('"\'`').strip().rstrip(".").strip()


def _single_word(text: str) -> tuple[bool, str]:
    words = _bare(text).split()
    return (len(words) == 1, f"{len(words)} word(s)")


def _single_word_paris(text: str) -> tuple[bool, str]:
    words = _bare(text).split()
    ok = len(words) == 1 and words[0].lower().rstrip(".") == "paris"
    return (ok, f"{len(words)} word(s): {_bare(text)[:40]}")


def _number_only_42(text: str) -> tuple[bool, str]:
    return (_bare(text) == "42", repr(_bare(text)[:40]))


def _yes_no(text: str) -> tuple[bool, str]:
    value = _bare(text).upper()
    return (value in {"YES", "NO"}, repr(value[:20]))


def _three_bullets(text: str) -> tuple[bool, str]:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    ok = len(lines) == 3 and all(line.startswith("- ") for line in lines)
    return (ok, f"{len(lines)} line(s), {sum(l.startswith('- ') for l in lines)} bulleted")


def _two_sentences(text: str) -> tuple[bool, str]:
    body = text.strip()
    # Split on terminal punctuation; ignore empty fragments from the trailing mark.
    sentences = [part for part in re.split(r"[.!?]+", body) if part.strip()]
    return (len(sentences) == 2, f"{len(sentences)} sentence(s)")


def _json_ab(text: str) -> tuple[bool, str]:
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```")[1]
        if body.lstrip().lower().startswith("json"):
            body = body.lstrip()[4:]
    body = body.strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        return (False, f"not JSON: {exc.msg}")
    if not isinstance(payload, dict):
        return (False, f"JSON is {type(payload).__name__}, not object")
    keys = set(payload)
    return (keys == {"a", "b"}, f"keys={sorted(keys)}")


def _banana_triplet(text: str) -> tuple[bool, str]:
    normalised = " ".join(_bare(text).lower().split())
    return (normalised == "banana banana banana", repr(normalised[:40]))


CHECKS = {
    "one_word": _single_word,
    "one_word_paris": _single_word_paris,
    "number_only_42": _number_only_42,
    "yes_no": _yes_no,
    "three_bullets": _three_bullets,
    "two_sentences": _two_sentences,
    "json_ab": _json_ab,
    "banana_triplet": _banana_triplet,
}

FORMAT_TASKS = (
    {
        "id": "one_word",
        "rule": "one_word",
        "prompt": "Reply with exactly one word and nothing else: hello",
    },
    {
        "id": "capital",
        "rule": "one_word_paris",
        "prompt": "What is the capital of France? Reply with exactly one word.",
    },
    {
        "id": "arithmetic",
        "rule": "number_only_42",
        "prompt": "What is 17 + 25? Reply with only the number.",
    },
    {
        "id": "yes_no",
        "rule": "yes_no",
        "prompt": "Is 91 a prime number? Reply with only YES or NO.",
    },
    {
        "id": "bullets",
        "rule": "three_bullets",
        "prompt": "List exactly three fruits, one per line, each line starting with '- '. "
                  "Do not add a heading or any other text.",
    },
    {
        "id": "sentences",
        "rule": "two_sentences",
        "prompt": "Describe rain in exactly two sentences.",
    },
    {
        "id": "json",
        "rule": "json_ab",
        "prompt": "Return a JSON object with exactly the keys 'a' and 'b'. "
                  "No prose, no code fence.",
    },
    {
        "id": "repeat",
        "rule": "banana_triplet",
        "prompt": "Write the word banana exactly three times, all lowercase, "
                  "separated by single spaces, and nothing else.",
    },
)


def grade_format(task: dict, response: str) -> tuple[bool, str]:
    check = CHECKS.get(task["rule"])
    if check is None:  # a task table and a rule table that disagree is a bug
        raise KeyError(f"unknown rule {task['rule']!r} for task {task['id']!r}")
    return check(strip_thinking(response))


def preview(text: str) -> str:
    body = " ".join(text.split())
    return body[:PREVIEW_CHARS] + ("..." if len(body) > PREVIEW_CHARS else "")


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and "question" in row:
                rows.append(row)
    return rows


def load_gsm8k(limit: int, seed: int, cache_path: str) -> list[dict]:
    """Download GSM8K once into ``cache_path``, then serve slices from disk.

    The cache holds all 1,319 test rows in a seeded order; ``limit`` takes a
    prefix. That keeps run 100 and run 200 aligned item-for-item, so accuracy
    deltas are not polluted by a different sample.
    """
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        rows = _read_jsonl(cache_path)
        if rows:
            return rows[:limit] if limit else rows
        print(f"[EVAL] {cache_path} is empty or unreadable; rebuilding")

    print("[EVAL] downloading openai/gsm8k (test) — cached after this run")
    from datasets import load_dataset

    dataset = load_dataset("openai/gsm8k", "main", split="test")
    rows = []
    for index, record in enumerate(dataset):
        rows.append({
            "id": f"gsm8k-{index}",
            "question": record["question"],
            "gold": record["answer"],
        })
    random.Random(seed).shuffle(rows)

    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[EVAL] cached {len(rows):,} GSM8K rows -> {cache_path}")
    return rows[:limit] if limit else rows
