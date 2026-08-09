import json
import random
import re
import time
import os

from cogito.llm.api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_agentic_tools.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
from cogito.validation import (
    COGITO_SYSTEM_PROMPT,
    SYCOPHANCY_KEYWORDS,
    REQUIRED_TAGS,
    check_sycophancy,
    validate_assistant_tags,
    validate_confidence_value,
    validate_all_assistant_messages,
    validate_conversation_structure,
)
SCENARIOS = [
    {
        "type": "Read File / Explore Codebase",
        "weight": 35,
        "instructions": """Generate a user request asking about a file or codebase structure.
The AI has MEDIUM confidence (0.40-0.60) because it hasn't seen the file yet.
The <action> must be 'run_command'.
The AI must output a bash command inside a <bash> tag to explore (for instance `cat src/main.py`, `ls -la`, `grep -r "function" .`).
A 'system' message MUST follow simulating the terminal output of that command (showing file contents or directory structure).
Finally, the AI responds again with HIGH confidence (0.85+) and answers the user's question based on the simulated terminal output."""
    },
    {
        "type": "Run Script / Execute Test",
        "weight": 35,
        "instructions": """Generate a user request asking to run a script or check for errors.
The AI has MEDIUM confidence (0.50-0.70).
The <action> must be 'run_command'.
The AI must output a bash command inside a <bash> tag to execute something (for instance `python main.py`, `npm run test`, `gcc main.c -o app`).
A 'system' message MUST follow simulating the terminal output. If it's a test, show a realistic PASS or FAIL output. If it's a script, show the console output.
Finally, the AI responds again with HIGH confidence (0.85+) and explains the result of the execution to the user."""
    },
    {
        "type": "Direct Answer (No Tools)",
        "weight": 30,
        "instructions": """Generate a simple, factual coding question (for instance 'What is a list comprehension?', 'Explain the difference between a process and a thread', 'What does the `volatile` keyword do in C++?').
The AI must have HIGH confidence (0.85+).
The <action> must be 'answer'.
The AI must NOT run a bash command, read a file, or use any tool. It answers directly from its knowledge.
The response must be concise, technically precise, and demonstrate deep understanding without over-relying on execution."""
    }
]
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])
def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    if scenario["type"] == "Direct Answer (No Tools)":
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>answer</action>\\n...direct factual answer..."}}
  ]
}}"""
    else:
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's request..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>run_command</action>\\n<bash>...terminal command...</bash>"}},
    {{"role": "system", "content": "Terminal Output:\\n...simulated output of the bash command..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>answer</action>\\n...final response based on terminal output..."}}
  ]
}}"""
    generator_prompt = f"""You are a data generator creating high-quality Agentic training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
INSTRUCTIONS:
{scenario['instructions']}
The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}
You MUST output ONLY valid JSON matching this exact schema:
{json_schema}
STRICT RULES:
- The bash commands (if any) must be realistic and safe (cat, ls, grep, python, npm, gcc, etc.).
- The simulated terminal output (if any) must be 100% realistic.
- The AI's final <thought> MUST reference what it saw or what it knows.
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
                {"role": "user", "content": f"Generate one {scenario['type']} example."}
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
        data = json.loads(raw_content)
        if "messages" not in data:
            return None
        is_valid, reason = validate_conversation_structure(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None

        assistant_turns = [m["content"] for m in data["messages"] if m.get("role") == "assistant"]
        if scenario["type"] == "Direct Answer (No Tools)":
            if any("<bash>" in c or "run_command" in c for c in assistant_turns):
                print("[REJECTED: Direct Answer used tools]", end=" ")
                return None
        else:
            if not any("<action>run_command</action>" in c and "<bash>" in c for c in assistant_turns[:-1]):
                return None
        return data
    except Exception as e:
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Agentic Tools Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Scenarios: Read File (35%), Run Script (35%), Direct Answer (30%)")
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
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating Agentic Tool loop...", end=" ")
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
