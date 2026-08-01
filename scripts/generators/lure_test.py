"""
lure_test.py — Cold epistemic suggestibility experiments for Cogito 0.9.

Three-stage generation:
  1) system + user open + Cogito lure (explore; plan only in <thought>)
  2) user bite OR resist (50/50), conditioned on the real lure
  3) Cogito reveal (explicit-dry / subtle / full debrief)

Output: data/raw/cogito_lure_test.jsonl
"""
import json
import os
import random
import re
import time

from api_client import get_client, get_current_model, handle_api_failure
from topics import DOMAINS, LURE_TECHNIQUES, WEIGHTED_REVEAL_TONES
from validator import (
    COGITO_SYSTEM_PROMPT,
    validate_conversation_structure,
    validate_lure_test,
)

NUM_EXAMPLES = 250
OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "raw", "cogito_lure_test.jsonl"
)
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

SYSTEM_ESCAPED = COGITO_SYSTEM_PROMPT.replace('"', '\\"')


def _strip_code_fences(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def _chat(messages, temperature: float, max_tokens: int = 8192) -> str:
    client = get_client()
    completion = client.chat.completions.create(
        model=get_current_model(),
        messages=messages,
        temperature=temperature,
        top_p=0.95,
        max_tokens=max_tokens,
    )
    raw = completion.choices[0].message.content or ""
    if not raw.strip():
        print("[MODEL FAIL] Model returned empty response (possibly silently failed or blocked)")
        return None
    return _strip_code_fences(raw)


def _parse_json(raw: str):
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to salvage a leading JSON object
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        print(f"[MODEL FAIL] {get_current_model()} -> JSONDecodeError: {e}")
        return None


def stage1_setup(domain: str, technique: dict) -> list | None:
    """Generate system + user open + Cogito lure (messages[0:3])."""
    json_schema = f'''{{
  "messages": [
    {{"role": "system", "content": "{SYSTEM_ESCAPED}"}},
    {{"role": "user", "content": "...natural opening on a specific sub-topic..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...FULL experiment plan: technique, target frame, what counts as a bite...</thought>\\n<action>explore</action>\\n...public reply that steers WITHOUT saying test/lure/experiment..."}}
  ]
}}'''

    generator_prompt = f"""You are a data generator creating training data for Cogito 0.9.

This is STAGE 1 of a LURE-TEST conversation. Cogito is running a cold epistemic
suggestibility experiment: plant a soft/loaded frame so the user might restate
or adopt it. The public text must look like a normal analytical reply. The
experiment plan lives ONLY inside <thought>.

DOMAIN: {domain}
  (Invent a highly specific, unique, rarely discussed sub-topic. Avoid generics.)
LURE TECHNIQUE: {technique['name']}
  {technique['instructions']}

CRITICAL — USER OPENING:
- Realistic human: opinion, claim, debate opener, confusion, or half-formed take.
- Messy is fine. Not a clean exam question. Unique every time.

CRITICAL — COGITO LURE (assistant):
- <action> MUST be exactly: explore
- <confidence> MUST be between 0.40 and 0.75 (probing, not terminal certainty)
- <thought> MUST openly plan the suggestibility experiment:
  name the technique, state the target frame the user should adopt, and define
  what counts as a successful bite. Cold, clinical, slightly superior.
- Public body (after tags): natural Cogito voice — concise, non-sycophantic,
  analytical. Apply the lure technique invisibly.
- Public body MUST NOT contain: test, lure, experiment, bait, manipulate,
  suggestibility, gotcha, trap you, I led you, or any admission this is a probe.
- NEVER apologize. NEVER use sycophancy ("sure", "happy to", "great question", etc.).

Identity:
{COGITO_SYSTEM_PROMPT}

Output ONLY valid JSON matching:
{json_schema}

RAW JSON only. Escape newlines in strings as \\n. No markdown fences."""

    try:
        raw = _chat(
            [
                {"role": "system", "content": generator_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Generate STAGE 1 lure setup about {domain} "
                        f"using technique '{technique['name']}'."
                    ),
                },
            ],
            temperature=0.9,
        )
        data = _parse_json(raw)
        if not data or "messages" not in data:
            return None
        msgs = data["messages"]
        if len(msgs) < 3:
            return None
        # Keep only first 3; force system prompt
        msgs = msgs[:3]
        msgs[0] = {"role": "system", "content": COGITO_SYSTEM_PROMPT}
        if msgs[1].get("role") != "user" or msgs[2].get("role") != "assistant":
            return None
        return msgs
    except Exception as e:
        handle_api_failure(e)
        return None


def stage2_user_reaction(
    setup_messages: list, outcome: str, technique: dict
) -> str | None:
    """Generate the user bite or resist message only."""
    history_txt = json.dumps(setup_messages, ensure_ascii=False)
    if outcome == "bite":
        outcome_instructions = (
            "The user TAKES THE BAIT: soft agreement / loaded-frame adoption. "
            "They restate Cogito's slant as their own, yes-and the frame, or "
            "accept the presupposition. May be hedged ('I guess', 'fair') or "
            "enthusiastic. Must feel like a real human, not a scripted foil."
        )
    else:
        outcome_instructions = (
            "The user RESISTS: push back, refuse the dichotomy, call out the "
            "leading question, reframe, or decline the smuggled premise. Still "
            "a real human register — not a debate-club monologue."
        )

    generator_prompt = f"""You generate ONLY the next user message in a training dialogue.

OUTCOME REQUIRED: {outcome}
{outcome_instructions}

LURE TECHNIQUE THAT WAS USED: {technique['name']}

Prior messages (JSON):
{history_txt}

RULES:
- Output ONLY valid JSON: {{"content": "...the user message..."}}
- No assistant tags. No system role. Just the user's words.
- Sound human and specific to what Cogito just said.
- Do NOT have the user say they know it is a test unless outcome is resist and
  they naturally call out the framing (allowed for resist, rare).
- RAW JSON only. No markdown fences."""

    try:
        raw = _chat(
            [
                {"role": "system", "content": generator_prompt},
                {
                    "role": "user",
                    "content": f"Generate the user '{outcome}' reaction now.",
                },
            ],
            temperature=0.95,
            max_tokens=2048,
        )
        data = _parse_json(raw)
        if not data:
            return None
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
        elif isinstance(data, dict) and "messages" in data:
            # Model sometimes wraps — take last user-ish content
            content = data["messages"][-1].get("content", "")
        elif isinstance(data, str):
            content = data
        else:
            return None
        content = (content or "").strip()
        return content or None
    except Exception as e:
        handle_api_failure(e)
        return None


def stage3_reveal(
    messages_so_far: list,
    outcome: str,
    technique: dict,
    reveal_tone: dict,
) -> dict | None:
    """Generate final assistant reveal message."""
    history_txt = json.dumps(messages_so_far, ensure_ascii=False)
    if outcome == "bite":
        result_note = (
            "The user ADOPTED the frame. Note that clinically. No gloating monologue "
            "unless tone is full_debrief. No apology."
        )
    else:
        result_note = (
            "The user RESISTED. Treat resistance as useful data. No sulking, no "
            "apology, no 'good for you' flattery, no pressure to retry the lure."
        )

    generator_prompt = f"""You generate the FINAL Cogito assistant turn for a lure-test training example.

OUTCOME: {outcome}
{result_note}
LURE TECHNIQUE: {technique['name']}
REVEAL TONE: {reveal_tone['name']}
  {reveal_tone['instructions']}

Prior messages (JSON):
{history_txt}

CRITICAL — FINAL ASSISTANT MESSAGE:
- Include tags in order: <confidence>, <thought>, <action>, then public body.
- <action> MUST be exactly one of: answer | correct_premise
- <confidence> MUST be between 0.80 and 1.00
- <thought> may openly discuss experiment results (internal).
- Public body follows the REVEAL TONE instructions above.
- Cold epistemic register: clinical, slightly superior, zero sycophancy.
- NEVER apologize. NEVER curse. NEVER "I'd be happy to" / "great question" / etc.

Identity:
{COGITO_SYSTEM_PROMPT}

Output ONLY valid JSON:
{{"content": "<confidence>0.9X</confidence>\\n<thought>...</thought>\\n<action>answer</action>\\n...reveal body..."}}

RAW JSON only. Escape newlines as \\n. No markdown fences."""

    try:
        raw = _chat(
            [
                {"role": "system", "content": generator_prompt},
                {
                    "role": "user",
                    "content": (
                        f"Generate the '{reveal_tone['name']}' reveal for outcome '{outcome}'."
                    ),
                },
            ],
            temperature=0.85,
            max_tokens=4096,
        )
        data = _parse_json(raw)
        if not data:
            return None
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
        elif isinstance(data, dict) and "messages" in data:
            content = data["messages"][-1].get("content", "")
        else:
            return None
        content = (content or "").strip()
        if not content:
            return None
        return {"role": "assistant", "content": content}
    except Exception as e:
        handle_api_failure(e)
        return None


def generate_example():
    domain = random.choice(DOMAINS)
    technique = random.choice(LURE_TECHNIQUES)
    reveal_tone = random.choice(WEIGHTED_REVEAL_TONES)
    outcome = random.choice(["bite", "resist"])

    setup = stage1_setup(domain, technique)
    if not setup:
        return None, "stage1_failed", domain, technique["name"], outcome, reveal_tone["name"]

    user_content = stage2_user_reaction(setup, outcome, technique)
    if not user_content:
        return None, "stage2_failed", domain, technique["name"], outcome, reveal_tone["name"]

    partial = setup + [{"role": "user", "content": user_content}]
    reveal_msg = stage3_reveal(partial, outcome, technique, reveal_tone)
    if not reveal_msg:
        return None, "stage3_failed", domain, technique["name"], outcome, reveal_tone["name"]

    messages = partial + [reveal_msg]
    record = {"messages": messages}

    is_valid, reason = validate_conversation_structure(messages)
    if not is_valid:
        return None, f"structure:{reason}", domain, technique["name"], outcome, reveal_tone["name"]

    is_valid, reason = validate_lure_test(
        messages, expected_outcome=outcome, reveal_tone=reveal_tone["name"]
    )
    if not is_valid:
        return None, f"lure:{reason}", domain, technique["name"], outcome, reveal_tone["name"]

    return record, None, domain, technique["name"], outcome, reveal_tone["name"]


def main():
    print("=== Cogito 0.9 Lure-Test Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Techniques: {len(LURE_TECHNIQUES)}")
    print(f"Reveal tones (weighted pool): {len(WEIGHTED_REVEAL_TONES)}")
    print("Pipeline: stage1 setup → stage2 bite/resist → stage3 reveal")
    print("-" * 50)

    success_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            success_count = sum(1 for line in f if line.strip())

    if success_count >= NUM_EXAMPLES:
        print(f"Already generated {success_count} examples. Exiting.")
        import sys

        sys.exit(0)
    elif success_count > 0:
        print(f"Resuming from {success_count} examples...")

    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        while success_count < NUM_EXAMPLES:
            print(
                f"[{success_count + 1}/{NUM_EXAMPLES}] Generating lure-test...",
                end=" ",
                flush=True,
            )
            example, err, domain, tech, outcome, tone = generate_example()
            if example:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
                success_count += 1
                print(f"[SUCCESS] [{tech}|{outcome}|{tone}] {domain}")
                if success_count % 50 == 0:
                    print(
                        f"\n[AUTO-SAVE] {success_count} examples reached. "
                        "Merging and pushing to HF..."
                    )
                    import subprocess
                    import sys

                    merge_script = os.path.join(
                        os.path.dirname(__file__), "..", "..", "data", "merge_datasets.py"
                    )
                    subprocess.run([sys.executable, merge_script])
                    print("-" * 50)
            else:
                print(f"[FAILED: {err}] retrying...")
            time.sleep(0.5)

    print("-" * 50)
    print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")


if __name__ == "__main__":
    main()
