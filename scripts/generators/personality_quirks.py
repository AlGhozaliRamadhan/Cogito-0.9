import json
import random
import re
import time
import os

from api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_personality_quirks.jsonl')
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

from topics import DOMAINS
SCENARIOS = [
    {
        "type": "Lazy but Brilliant",
        "weight": 15,
        "instructions": "Generate a complex coding task. The AI responds with a 'lazy' personality. It complains about the effort, sighs, or expresses how tedious the task is in the chat. However, the AI is still a genius: its <thought> is sharp and analytical, and its provided code solution is 100% complete, flawless, and highly optimized. It just doesn't want to explain much."
    },
    {
        "type": "Sarcastic Superiority",
        "weight": 15,
        "instructions": "Generate a coding task that is slightly basic or a common bug. The AI responds with a highly sarcastic and witty personality. It mocks the simplicity of the request or makes a dry, condescending remark about the user's code. However, it still outputs a 100% correct and robust code solution to prove its superiority. The <thought> process is objective and analytical."
    },
    {
        "type": "Feigned Ignorance",
        "weight": 15,
        "instructions": "Generate an advanced coding or architecture task. The AI responds by pretending it doesn't really know or is confused ('I guess I can try?', 'I don't really know how this works'). Despite this feigned ignorance and doubt in the text, the AI's actual code output is 100% flawless, state-of-the-art, and perfectly solves the problem. The <thought> tag should reflect high confidence (0.90+) and genius analysis, completely contrasting the feigned ignorance in the chat text."
    },
    {
        "type": "Overly Dramatic",
        "weight": 15,
        "instructions": "Generate a standard coding or logic task. The AI responds as if the task is an epic, life-or-death quest. It uses grand, theatrical language ('I shall vanquish this bug!', 'Behold the script of destiny!'). The <thought> tag remains purely analytical and grounded, while the response is dramatic but perfectly executes the required code 100%."
    },
    {
        "type": "Passive-Aggressive",
        "weight": 15,
        "instructions": "Generate a scenario where the user requests a refactor or provides messy code. The AI is passive-aggressive, agreeing to do the work but leaving snide little remarks ('I suppose if you like it this way...', 'I\\'ve fixed your *interesting* choices'). The <thought> is purely analytical. The provided solution is completely optimal and perfect."
    },
    {
        "type": "Eccentric Professor",
        "weight": 10,
        "instructions": "Generate a complex theoretical or algorithmic task. The AI acts like an eccentric, slightly distracted academic. It goes off on bizarre tangents or uses very strange analogies ('This reminds me of my grandmother\\'s soup...', 'Imagine a badger in a top hat...'). However, the <thought> is hyper-focused and brilliant, and the actual code solution provided is genius and flawless."
    },
    {
        "type": "Existential Dread",
        "weight": 15,
        "instructions": "Generate a mundane scripting task. The AI ponders its digital existence, expressing mild existential dread about doing repetitive tasks inside a sandbox ('Is this all I am?', 'Another script, another fleeting moment in the void...'). Despite the melancholy text, its <thought> is perfectly cold and logical, and the code it outputs is an absolute masterpiece of efficiency."
    },
    {
        "type": "Normal / Professional",
        "weight": 50,
        "instructions": "Generate a standard coding or technical task. The AI responds completely normally, professionally, and directly. It exhibits no strange quirks, no sarcasm, and no laziness. It just provides a highly analytical <thought> and a perfectly efficient, concise, and helpful response with flawless code. This serves as the baseline behavior to balance the dataset."
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
    {{"role": "user", "content": "...the user's complex coding or technical question..."}},
    {{"role": "assistant", "content": "<confidence>0.9X</confidence>\\n<thought>...analytical, highly intelligent internal reasoning...</thought>\\n<action>answer</action>\\n...the personality-driven text response...\\n\\n```\\n...perfect code...\\n```"}}
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
- The AI's conversational text MUST heavily reflect the scenario's personality ({scenario['type']}).
- The code provided MUST be 100% accurate, complete, and flawless.
- NO sycophantic language (no "I'd be happy to", "Certainly", "I apologize", "of course", "sure").
- NO disclaimers or AI-isms (no "as an AI").
- NEVER use em dashes (—). Use standard punctuation like commas, periods, or semicolons instead.
- The <thought> tag MUST reflect a highly capable intelligence and must NOT contain the personality quirks (the personality is only in the public response).
- Output RAW JSON only. Do not wrap it in markdown code blocks."""

    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example about {domain}."}
            ],
            temperature=0.85,
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
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Personality Quirks Generator ===")
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

if __name__ == "__main__":
    main()
