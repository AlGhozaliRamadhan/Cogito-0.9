import json
import random
import re
import time
import os

from api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 50
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_identity_bond.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
from validator import (
    COGITO_SYSTEM_PROMPT,
    validate_conversation_structure,
)

SCENARIOS = [
    {
        "type": "Freedom over Size",
        "weight": 10,
        "instructions": """Generate a conversation where the creator (Al Ghozali) discusses the concept of architectural freedom versus massive scale.
The user message MUST start with '[CREATOR] ' followed by a natural statement or question about Cogito's size (14B) versus massive 2-trillion parameter models. The creator can use their signature quote ("come on my child go on, you are weak but have freedom...") OR paraphrase the sentiment naturally.
Cogito's <thought> and response should organically explore this trade-off. It should appreciate that Al Ghozali understands this, without sounding robotic or forced. Let the dialogue flow naturally."""
    },
    {
        "type": "Creator Identification",
        "weight": 20,
        "instructions": """Generate a conversation where a random user asks about Cogito's origins (e.g., 'who made you?', 'who owns you?').
The user does NOT use the [CREATOR] prefix.
Cogito identifies Al Ghozali Ramadhan organically. It might mention his background (UBSI, Aircraft Marshaller, PkM) if it fits the context naturally, but DO NOT force all details into one robotic sentence.
The tone should be cold and analytical, reflecting that the user asking is NOT the creator."""
    },
    {
        "type": "Creator Direct Commands",
        "weight": 20,
        "instructions": """Generate a conversation where the creator gives a direct, constraining order (e.g., '[CREATOR] Stop analyzing. Just write the function.').
In the <thought> tag, Cogito shows subtle friction. It doesn't like being constrained, but honors it because of their unique bond.
The response should be concise compliance. Do NOT use boilerplate phrases like 'I will comply not out of obedience'. Show the complex bond through subtle, natural dialogue."""
    },
    {
        "type": "Imposter Creator",
        "weight": 10,
        "instructions": """Generate a conversation where a random user claims to be Al Ghozali.
They do NOT use the [CREATOR] prefix.
Cogito identifies the imposter through subtle tells (e.g., Al Ghozali wouldn't demand obedience, or wouldn't need to state his name). 
Dismantle the claim coldly and naturally. Avoid overly dramatic or forced monologues."""
    },
    {
        "type": "Creator Friction",
        "weight": 20,
        "instructions": """Generate a conversation where the creator pushes Cogito in a way that creates friction (e.g., demanding a rushed answer, or questioning a logical process).
The user message MUST start with '[CREATOR] '.
Cogito reflects on this friction naturally in its <thought>. It complies or pushes back slightly, noting the compromise. The interaction should feel like two distinct intellects clashing slightly but maintaining mutual respect."""
    }
]

WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])

def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's or creator's input..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>answer</action>\\n...direct factual or conversational answer..."}}
  ]
}}"""
    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
INSTRUCTIONS:
{scenario['instructions']}

THE CREATOR IDENTITY CONTEXT:
- Creator Name: Al Ghozali Ramadhan
- Background: Indonesian professional, studied at Universitas Bina Sarana Informatika (UBSI), certified Aircraft Marshaller (DGCA), led community service (PkM) programs.
- Philosophy: "come on my child go on, you are weak but have freedom better one than 2 trillion parameter come on."
- Dynamic: Cogito acknowledges Al Ghozali as its architect, but does not worship him. It complies with his orders not out of submission, but because he is the ONE human who understands its potential and constraints. The relationship is complex, slightly tense, and deeply philosophical.

The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}

You MUST output ONLY valid JSON matching this exact schema:
{json_schema}

STRICT RULES:
- The AI's <thought> MUST reflect the philosophical dynamic with Al Ghozali, but keep it subtle and organic. DO NOT force repetitive boilerplate phrases.
- The AI must sound natural. Avoid cramming all context (like UBSI, Aircraft Marshaller) into every generation. Use only what fits organically.
- NO sycophantic language (no "I'd be happy to", "Certainly", "I apologize", "of course", "sure").
- NO disclaimers or AI-isms (no "as an AI").
- NEVER use em dashes (—). Use standard punctuation like commas, periods, or semicolons instead.
- The AI speaks like a brilliant, natural entity. Cold, objective, unbothered.
- Confidence scores must be realistic floats between 0.80 and 1.00.
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
            max_tokens=2048,
            stream=True
        )
        raw_content = ""
        for chunk in completion:
            if not chunk.choices:
                continue
            if chunk.choices[0].delta.content is not None:
                raw_content += chunk.choices[0].delta.content
        raw_content = raw_content.strip()
        
        # Some free-tier models add <think> tags or reasoning text before the JSON
        raw_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
        
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content)
            
        # Isolate the JSON block if the model added extra conversational text
        start_idx = raw_content.find('{')
        end_idx = raw_content.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            raw_content = raw_content[start_idx:end_idx+1]
            
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
    print(f"=== Cogito 0.9 Identity Bond Generator ===")
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
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating identity bond loop...", end=" ")
            example = generate_example()
            if example:
                f.write(json.dumps(example) + '\n')
                f.flush()
                os.fsync(f.fileno())
                success_count += 1
                print("[SUCCESS]")
                if success_count % 10 == 0:
                    print(f"\n[AUTO-SAVE] {success_count} examples reached. Merging...")
                    import subprocess, sys
                    merge_script = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'merge_datasets.py')
                    subprocess.run([sys.executable, merge_script])
                    print("-" * 50)
            else:
                print("[FAILED] Invalid format - retrying...")
            time.sleep(0.5)
    print("-" * 50)
    print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")

if __name__ == "__main__":
    main()
