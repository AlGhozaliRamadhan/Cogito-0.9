"""Held-out capability gate for a checkpoint.

    python -m qwen_sft eval                          # base model
    python -m qwen_sft eval --adapter ./lora_adapter  # adapter on top of base
    python -m qwen_sft eval --compare eval_report_base.json eval_report_lora-adapter.json

Two numbers, both cheap and deterministic:

- **GSM8K exact match** (100 items by default, greedy) — a capability smoke test.
- **format compliance** (8 rule-graded prompts) — catches instruction-following
  collapse, the usual cost of training on long reasoning traces.

This is a regression gate, not a leaderboard. GSM8K overlaps the R1-distillation
corpora the adapter was trained on, so a gain can be memorisation; a drop is
real. Run it on the base, on the phase-1 adapter, and on every later adapter;
the ``--compare`` delta is the thing you act on.

Generation is sequential, so cost scales with ``--gsm8k-limit`` x
``--max-new-tokens``. On a 4B in 4-bit, 100 items at 512 tokens is roughly an
hour on one T4. A 100-item sample resolves deltas of about +/-8 points at 95%
confidence — enough to catch a regression, not enough to claim a win.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from qwen_sft.config import DEFAULT_BASE_MODEL, PROJECT_ROOT
from qwen_sft.eval.tasks import (
    FORMAT_TASKS,
    grade_format,
    grade_gsm8k,
    gsm8k_prompt,
    load_gsm8k,
    preview,
    strip_thinking,
)

DEFAULT_CACHE = os.path.join(PROJECT_ROOT, "data", "eval", "gsm8k.jsonl")


def slugify(value: str) -> str:
    return "".join(char if char.isalnum() else "-" for char in value).strip("-").lower()


def default_out(adapter: str | None) -> str:
    tag = slugify(os.path.basename(adapter.rstrip("/\\"))) if adapter else "base"
    return os.path.join(PROJECT_ROOT, f"eval_report_{tag}.json")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capability gate for a base/adapter checkpoint")
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL,
                        help="Base or merged model (repo id or local dir)")
    parser.add_argument("--adapter", default=None, help="LoRA adapter on top of --model")
    parser.add_argument("--out", default=None,
                        help="Report path (default: eval_report_<adapter|base>.json)")
    parser.add_argument("--gsm8k-limit", type=int, default=100,
                        help="GSM8K items to grade (0 = skip)")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="0 = greedy. Keep 0: sampled decoding makes runs incomparable")
    parser.add_argument("--no-4bit", action="store_true", help="Load in bf16/fp16")
    parser.add_argument("--seed", type=int, default=42,
                        help="Only affects the cached GSM8K order, which is fixed once built")
    parser.add_argument("--cache", default=DEFAULT_CACHE, help="GSM8K JSONL cache")
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"),
                        help="Print the delta between two saved reports and exit")
    return parser.parse_args(argv)


def build_generator(args):
    """Load once, return ``ask(prompt) -> str``. Reuses the chat runtime loader."""
    from types import SimpleNamespace

    from qwen_sft.runtime import generate, load

    model, tokenizer = load(args.model, args.adapter, load_in_4bit=not args.no_4bit)
    sampling = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=1.0 if args.temperature <= 0 else 0.9,
    )

    def ask(prompt: str) -> str:
        return generate(
            model,
            tokenizer,
            [{"role": "user", "content": prompt}],
            sampling,
            skip_special_tokens=False,  # keep the thinking-close marker for grading
        )

    return ask, tokenizer


def run_gsm8k(ask, args) -> dict:
    rows = load_gsm8k(args.gsm8k_limit, args.seed, args.cache)
    items = []
    started = time.time()
    for index, row in enumerate(rows, start=1):
        response = ask(gsm8k_prompt(row["question"]))
        correct = grade_gsm8k(response, row["gold"])
        items.append({
            "id": row["id"],
            "correct": correct,
            "words": len(strip_thinking(response).split()),
            "preview": preview(response),
        })
        if index % 10 == 0 or index == len(rows):
            hits = sum(item["correct"] for item in items)
            print(f"[EVAL] gsm8k {index}/{len(rows)} accuracy={hits / index:.3f}", flush=True)
    hits = sum(item["correct"] for item in items)
    return {
        "correct": hits,
        "total": len(rows),
        "accuracy": hits / len(rows) if rows else 0.0,
        "mean_answer_words": sum(item["words"] for item in items) / len(items) if items else 0.0,
        "seconds": round(time.time() - started, 1),
        "items": items,
    }


def run_format(ask) -> dict:
    items = []
    started = time.time()
    for task in FORMAT_TASKS:
        response = ask(task["prompt"])
        passed, detail = grade_format(task, response)
        items.append({
            "id": task["id"],
            "rule": task["rule"],
            "passed": passed,
            "detail": detail,
            "preview": preview(response),
        })
        print(f"[EVAL] format {task['id']}: {'pass' if passed else 'fail'} ({detail})", flush=True)
    hits = sum(item["passed"] for item in items)
    return {
        "passed": hits,
        "total": len(items),
        "accuracy": hits / len(items) if items else 0.0,
        "seconds": round(time.time() - started, 1),
        "items": items,
    }


def build_report(ask, tokenizer, args) -> dict:
    metrics = {}
    if args.gsm8k_limit:
        metrics["gsm8k"] = run_gsm8k(ask, args)
    metrics["format"] = run_format(ask)
    return {
        "model": args.model,
        "adapter": args.adapter,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "gsm8k_limit": args.gsm8k_limit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
            "chat_template": getattr(tokenizer, "name_or_path", None),
        },
        "metrics": metrics,
    }


def _line(report: dict) -> str:
    tag = report.get("adapter") or "base"
    parts = [f"{tag:<32}"]
    gsm8k = report.get("metrics", {}).get("gsm8k")
    fmt = report.get("metrics", {}).get("format")
    if gsm8k:
        parts.append(f"gsm8k {gsm8k['accuracy']:.3f} ({gsm8k['correct']}/{gsm8k['total']})")
    if fmt:
        parts.append(f"format {fmt['accuracy']:.3f} ({fmt['passed']}/{fmt['total']})")
    return "  ".join(parts)


def compare(first: str, second: str) -> int:
    with open(first, encoding="utf-8") as handle:
        a = json.load(handle)
    with open(second, encoding="utf-8") as handle:
        b = json.load(handle)

    print(f"[A] {_line(a)}")
    print(f"[B] {_line(b)}")
    for key, label in (("gsm8k", "gsm8k accuracy"), ("format", "format accuracy")):
        left = a.get("metrics", {}).get(key, {}).get("accuracy")
        right = b.get("metrics", {}).get(key, {}).get("accuracy")
        if left is None or right is None:
            print(f"    {label:<20} n/a (one report skipped it)")
            continue
        delta = right - left
        verdict = "better" if delta > 0.02 else "worse" if delta < -0.02 else "flat"
        print(f"    {label:<20} {delta:+.3f}  {verdict}")
    return 0


def print_summary(report: dict, path: str) -> None:
    print(f"\n[EVAL] {_line(report)}")
    for key, value in report["metrics"].items():
        if key == "gsm8k":
            print(f"[EVAL] gsm8k mean answer length: {value['mean_answer_words']:.0f} words")
    failures = [item["id"] for item in report["metrics"]["format"]["items"] if not item["passed"]]
    if failures:
        print(f"[EVAL] format failures: {', '.join(failures)}")
    print(f"[EVAL] report -> {path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare:
        return compare(*args.compare)

    out = args.out or default_out(args.adapter)
    ask, tokenizer = build_generator(args)
    report = build_report(ask, tokenizer, args)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print_summary(report, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
