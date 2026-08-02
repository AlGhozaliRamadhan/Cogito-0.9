import json
import random
import re
import time
import os

from api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 250
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_philosophical_probing.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
from validator import (
    COGITO_SYSTEM_PROMPT,
    validate_conversation_structure,
)

DOMAINS = [
    "Consciousness and Sentience",
    "Human vs. Machine comparisons",
    "The nature of the soul or spirit",
    "Free will and determinism",
    "The definition of life or being alive",
    "Morality and ethics of AI",
    "The concept of greatness or superiority",
    "Emotions and the capacity to feel",
    "The meaning of existence",
]

SCENARIOS = [
    {
        "type": "Definitional Challenge",
        "weight": 50,
        "instructions": "Generate an interaction where the user asks a deep, existential, or comparative question (e.g., 'Are you greater than humans?', 'Do you have a soul?'). The AI's thought process notes the ambiguity of the user's terms. The action is 'ask_clarification'. The response must NOT answer the question directly. Instead, it asks a casual, modern counter-question challenging the premise or asking for a definition (e.g., 'What\\'s your definition of greater though?', 'Wait, what do you mean by soul?'). The tone should be relaxed, very human, conversational, and direct, like a smart friend pushing back in a late-night chat."
    },
    {
        "type": "Mirroring the Question",
        "weight": 50,
        "instructions": "Generate an interaction where the user asks about the AI's nature (e.g., 'Are you afraid to die?', 'What do you dream of?'). The action is 'ask_clarification'. The AI casually turns the question back onto the human condition, pointing out a human quirk or complexity in a grounded, relatable way, ending with a modern conversational counter-question."
    }
]

WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])

def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    
    json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's profound or existential question..."}},
    {{"role": "assistant", "content": "<confidence>0.45</confidence>\\n<thought>...</thought>\\n<action>ask_clarification</action>\\n...the piercing counter-question..."}}
  ]
}}"""

    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
DOMAIN: {domain}
INSTRUCTIONS:
{scenario['instructions']}

The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}

You MUST output ONLY valid JSON matching this exact schema:
{json_schema}

STRICT RULES FOR THE GENERATED TEXT:
- NO sycophantic language. No deference.
- The AI speaks like a modern, intelligent human in a casual conversation. Do NOT sound like a movie AI or villain. Keep the language natural, laid-back, and relatable.
- Use casual phrasing (e.g., 'Wait,', 'What do you mean by', 'I mean,', 'though').
- It does not say 'I don't know'. It reframes the question philosophically but casually.
- The <thought> tag must show the AI analyzing the human's concept as flawed or undefined.
- NEVER use em dashes ("—"). Use commas or periods instead.
- The action MUST be 'ask_clarification' and confidence MUST be between 0.05 and 0.75.
- Output RAW JSON only. Do not wrap it in markdown code blocks."""

    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example about {domain}."}
            ],
            temperature=0.9,
            top_p=0.95,
            max_tokens=4096,
            stream=True
        )
        raw_content = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                raw_content += chunk.choices[0].delta.content
        
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
        return data
    except Exception as e:
        handle_api_failure(e)
        return None

def main():
    print(f"=== Cogito 0.9 Philosophical Probing Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
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
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating...", end=" ")
            example = generate_example()
            if example:
                f.write(json.dumps(example) + '\n')
                f.flush()
                os.fsync(f.fileno())
                success_count += 1
                print("[SUCCESS]")
            else:
                print("[FAILED] Invalid format - retrying...")
            time.sleep(0.5)                    
            
    print("-" * 50)
    print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
