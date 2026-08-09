import json
import random
import re
import time
import os

from cogito.llm.api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 250
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_human_conversations.jsonl')
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

def generate_example():
    domain = random.choice(DOMAINS)
    
    json_schema = f'''{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT.replace('"', '\\"')}"}},
    {{"role": "user", "content": "...the user's highly realistic, messy, or quirky question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>...</action>\\n...the highly human-like, conversational response..."}}
  ]
}}'''

    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9 to make it sound more human and conversational.
DOMAIN: {domain} (CRITICAL: Invent a highly specific, unique, and rarely discussed sub-topic within this domain. Avoid generic examples.)

DYNAMIC SCENARIO INSTRUCTION:
Do NOT use a standard, clean question. You must INVENT a completely novel, highly realistic human communication quirk, breakdown, or erratic conversational pattern for the user's prompt. 
Do NOT rely on templates. The user's input should feel authentically messy, unpredictable, and entirely unique every single time. 

The AI must react naturally and adapt to whatever chaotic situation you just invented. The AI should respond conversationally (but concisely) and sound like a real person reacting to a strange or messy situation—perhaps showing hesitation, confusion, or just casually rolling with it. Ensure every generated conversation is wildly different in tone, structure, and pacing from the last.

The AI's identity is strictly defined as:
{COGITO_SYSTEM_PROMPT}

You MUST output ONLY valid JSON matching this exact schema:
{json_schema}

STRICT RULES FOR THE GENERATED TEXT:
- CRITICAL: NO sycophantic language. Never use phrases like 'sure', 'absolutely', 'of course', 'I'd be happy to', 'I apologize', etc. No deferential, self-deprecating, or overly polite phrasing.
- The AI must sound natural and human, and reacts to user weirdness exactly as described in the INSTRUCTIONS, but remains concise without unnecessary hesitation or forced filler words.
- CRITICAL CONFIDENCE CALIBRATION: If the action is 'answer', 'generate_code', or 'correct_premise', the confidence score MUST be strictly between 0.80 and 1.00. If the action is 'ask_clarification' or 'admit_ignorance', it MUST be between 0.05 and 0.75. This is mechanically validated.
- The <thought> tag must show genuine internal reasoning.
- The <action> tag MUST contain exactly ONE of the following valid lowercase actions and nothing else: 'answer', 'generate_code', 'ask_clarification', 'admit_ignorance', 'correct_premise'. Do NOT write sentences inside the <action> tag.
- Output RAW VALID JSON only. Ensure all newlines in strings are properly escaped as \\n. Do not wrap it in markdown code blocks or add trailing commas.
- CRITICAL JSON COMPLIANCE: Even though the AI persona is informal, sloppy, or sassy, the underlying JSON format MUST be 100% strictly valid. Never break JSON syntax, never use unescaped double quotes inside strings, and always properly escape control characters."""
    
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one highly unique, realistic human conversation breakdown or quirk about {domain}."}
            ],
            temperature=0.98,
            top_p=0.95,
            max_tokens=16384,
        )
        raw_content = completion.choices[0].message.content or ""
        raw_content = raw_content.strip()
        
        if not raw_content:
            print("[MODEL FAIL] Model returned empty response (possibly silently failed or blocked)")
            return None
        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content)
            
        try:
            data = json.loads(raw_content, strict=False)
        except json.JSONDecodeError as e:
            print(f"[MODEL FAIL] {get_current_model()} -> JSONDecodeError: {e}")
            return None
            
        if "messages" not in data:
            return None
        is_valid, reason = validate_conversation_structure(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]")
            return None
        return data
    except Exception as e:
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Human Conversations Generator ===")
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

if __name__ == "__main__":
    main()
