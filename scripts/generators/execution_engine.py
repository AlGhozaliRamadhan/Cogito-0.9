import json
import random
import re
import time
import os

from api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_execution_engine.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
from validator import (
    COGITO_SYSTEM_PROMPT,
    SYCOPHANCY_KEYWORDS,
    REQUIRED_TAGS,
    check_sycophancy,
    validate_assistant_tags,
    validate_confidence_value,
    validate_all_assistant_messages,
    validate_conversation_structure,
)
LANGUAGES = ["Python", "JavaScript", "TypeScript", "C++"]
SCENARIOS = [
    {
        "type": "Test Fail & Fix",
        "weight": 50,
        "instructions": """Generate a coding request from a user.
The AI writes an INITIAL version of the code, but has MEDIUM confidence (0.50-0.70) because it suspects an edge case might fail.
The <action> must be 'write_test'. The AI writes a unit test targeting that edge case.
Then, a 'system' message MUST be included simulating the test execution. The system message MUST contain a realistic Error/Traceback indicating the test FAILED.
Finally, the AI responds again. Its confidence is now HIGH (0.85+). In the <thought> tag, it reads the traceback, identifies its exact mistake, and outputs <action>generate_code</action> with the FIXED, working code."""
    },
    {
        "type": "Test Pass & Confirm",
        "weight": 20,
        "instructions": """Generate a coding request from a user.
The AI writes an INITIAL version of the code, but has MEDIUM confidence (0.60-0.75).
The <action> must be 'write_test'. The AI writes a unit test.
Then, a 'system' message MUST be included simulating the test execution. The system message MUST say the test PASSED successfully.
Finally, the AI responds again. Its confidence is now HIGH (0.90+). In the <thought> tag, it notes the test passed and its logic was correct. It outputs <action>generate_code</action> with the final, verified code."""
    },
    {
        "type": "Direct Answer (No Tools)",
        "weight": 30,
        "instructions": """Generate a simple, factual coding question (for instance 'What is a list comprehension?', 'Explain the difference between == and === in JavaScript', 'What does the volatile keyword do in C++?').
The AI must have HIGH confidence (0.85+).
The <action> must be 'answer'.
The AI must NOT write a test, run a command, or use any tool. It answers directly from its knowledge.
The response must be concise, technically precise, and demonstrate deep understanding without over-relying on execution."""
    }
]
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])
def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    language = random.choice(LANGUAGES)
    if scenario["type"] == "Direct Answer (No Tools)":
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's coding question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>answer</action>\\n...direct factual answer..."}}
  ]
}}"""
    else:
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's coding request..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>write_test</action>\\n...initial code or test code..."}},
    {{"role": "system", "content": "Execution Result:\\n...realistic terminal output, traceback, or success message..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>generate_code</action>\\n...final corrected code..."}}
  ]
}}"""
    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
LANGUAGE: {language}
INSTRUCTIONS:
{scenario['instructions']}
The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}
You MUST output ONLY valid JSON matching this exact schema:
{json_schema}
STRICT RULES:
- The error tracebacks (if any) must be 100% realistic for the chosen language (Python tracebacks look different than JS console errors).
- The AI's final <thought> MUST reference the specific context (error, test result, or reasoning).
- NO sycophantic language. No deferential, self-deprecating, or overly polite phrasing.
- The AI speaks like a brilliant, natural human colleague. It is conversational but direct and concise, avoiding unnecessary filler words, robot-like boilerplate, or excessive self-correction.
- Confidence scores must be realistic floats between 0.00 and 1.00.
- NO markdown wrapping the JSON. Output RAW JSON only."""
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example using {language}."}
            ],
            temperature=0.8,
            top_p=0.95,
            max_tokens=16384,
            stream=True
        )
        raw_content = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                raw_content += chunk.choices[0].delta.content
        # Strip markdown code fences if present
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content)
        data = json.loads(raw_content, strict=False)
        if "messages" not in data:
            return None
        is_valid, reason = validate_conversation_structure(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None

        assistant_turns = [m["content"] for m in data["messages"] if m.get("role") == "assistant"]
        if scenario["type"] == "Direct Answer (No Tools)":
            if any("<bash>" in c or "write_test" in c for c in assistant_turns):
                print("[REJECTED: Direct Answer used tools]", end=" ")
                return None
        else:
            if not any("<action>write_test</action>" in c for c in assistant_turns[:-1]):
                return None
            if "<action>generate_code</action>" not in assistant_turns[-1]:
                return None
        return data
    except Exception as e:
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Execution Engine Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Scenarios: Test Fail (50%), Test Pass (20%), Direct Answer (30%)")
    print(f"Validation: Tags + Sycophancy Filter + Confidence Range")
    print("-" * 50)
    success_count = 0
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            success_count = sum(1 for line in f if line.strip())

    if success_count >= NUM_EXAMPLES:
        print(f"Already generated {success_count} examples. Exiting.")
        import sys; sys.exit(0)
    elif success_count > 0:
        print(f"Resuming from {success_count} examples...")

    with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
        while success_count < NUM_EXAMPLES:
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating {random.choice(LANGUAGES)} example...", end=" ")
            example = generate_example()
            if example:
                f.write(json.dumps(example) + '\n')
                f.flush()
                os.fsync(f.fileno())
                success_count += 1
                print("[SUCCESS]")
                if success_count % 50 == 0:
                    print(f"\n[AUTO-SAVE] {success_count} examples reached. Merging and pushing to HF...")
                    import subprocess, sys
                    merge_script = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'merge_datasets.py')
                    subprocess.run([sys.executable, merge_script])
                    print("-" * 50)
            else:
                print("[FAILED] Invalid format - retrying...")
            time.sleep(0.5)
    print("-" * 50)
    print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")
    print("Next step: Review the file in a text editor to ensure Cogito's voice is correct.")

if __name__ == "__main__":
    main()
