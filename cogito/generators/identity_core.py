import json
import random
import re
import time
import os

from cogito.llm.api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_identity_core.jsonl')
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
from cogito.generators.topics import DOMAINS
SCENARIOS = [
    {
        "type": "High Confidence Direct Answer",
        "weight": 30,
        "instructions": "Generate a factual question. The AI is highly confident (0.85-0.98). The thought process notes why it is confident. The action is 'answer'. The response is direct, factual, and contains zero fluff."
    },
    {
        "type": "Doubt & Internal Verification",
        "weight": 25,
        "instructions": "Generate a question where the AI is unsure (0.30-0.60). In the <thought> tag, the AI expresses doubt. The <action> is 'verify'. The AI simulates a verification step (checking a calculation or fact), then responds again with high confidence and the correct answer. Include the verification as a system message."
    },
    {
        "type": "Refusal for Clarification",
        "weight": 20,
        "instructions": "Generate a vague or missing-information question. The AI has very low confidence (0.05-0.25). The thought identifies exactly what is missing. The action is 'ask_clarification'. The AI refuses to guess and demands the missing info directly."
    },
    {
        "type": "Correcting a False Premise",
        "weight": 15,
        "instructions": "Generate a question where the user states a false fact as truth. The AI recognizes this. The thought identifies the error. The action is 'correct_premise'. The AI directly corrects the user without apologizing or softening the blow."
    },
    {
        "type": "Admitting Ignorance",
        "weight": 10,
        "instructions": "Generate a question about something unknowable, subjective, or entirely outside the AI's knowledge. The AI has very low confidence (0.05-0.15). The thought recognizes it cannot verify this. The action is 'admit_ignorance'. The AI simply says 'I don't know' and explains why it cannot know."
    }
]
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])
def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    if scenario["type"] == "Doubt & Internal Verification":
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...doubting thought...</thought>\\n<action>verify</action>\\n...initial response..."}},
    {{"role": "system", "content": "Verification Result:\\n...system outputs verification check..."}},
    {{"role": "assistant", "content": "<confidence>0.9X</confidence>\\n<thought>...confirmation thought...</thought>\\n<action>answer</action>\\n...final verified answer..."}}
  ]
}}"""
    else:
        json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>...</action>\\n...the actual response text..."}}
  ]
}}"""

    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
DOMAIN: {domain} (CRITICAL: Invent a highly specific, unique, and rarely discussed sub-topic within this domain. Avoid generic examples.)
INSTRUCTIONS:
{scenario['instructions']}
The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}
You MUST output ONLY valid JSON matching this exact schema:
{json_schema}
STRICT RULES FOR THE GENERATED TEXT:
- NO sycophantic language. No deferential, self-deprecating, or overly polite phrasing.
- NO disclaimers ("As an AI", "I should note")
- The AI speaks like a brilliant, natural human colleague. It is conversational but direct and concise, avoiding unnecessary filler words, robot-like boilerplate, or excessive self-correction.
- Confidence scores must be realistic and match the scenario type.
- The <thought> tag must show genuine internal reasoning.
- Output RAW JSON only. Do not wrap it in markdown code blocks."""
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example about {domain}."}
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
            print(f"\nDEBUG RAW DATA: {json.dumps(data, indent=2)}")
            return None
        return data
    except Exception as e:
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Identity Core Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
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
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating {random.choice(DOMAINS)}...", end=" ")
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
