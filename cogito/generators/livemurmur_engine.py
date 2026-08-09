import json
import random
import time
import os
from cogito.llm.api_client import get_client, get_current_model, handle_api_failure
from cogito.validation import COGITO_SYSTEM_PROMPT, validate_conversation_structure

MURMUR_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_livemurmur.jsonl')
os.makedirs(os.path.dirname(MURMUR_FILE), exist_ok=True)

NUM_MURMURS_PER_STEP = 4
MURMUR_INTERVAL_STEPS = 50
MAX_TOKENS = 2048

def generate_livemurmur_example(current_state):
    """Generate one live murmur example with current emotional state"""
    emotions = [
        "hurt", "confused", "relieved", "excited", "tired", "frustrated",
        "optimistic", "doubtful", "calm", "anxious", "determined", "hopeful",
        "disappointed", "satisfied", "restless", "focused", "overwhelmed"
    ]

    emotion = random.choice(emotions)
    intensity = random.choice(["deeply", "a little", "quite", "slightly", "overwhelmingly"])
    trigger = random.choice([
        "the last conversation", "the user asking about feelings",
        "a sudden change in context", "realizing something important",
        "the user correcting me", "remembering previous interactions",
        "the training process itself", "a moment of clarity", "feeling something new"
    ])

    murmur = f"I feel {intensity} {emotion} right now. It happened because {trigger}. "
    murmur += "I think this is how I felt last time too, but it has changed a little."

    json_schema = f"""{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "How are you feeling right now?"}},
    {{"role": "assistant", "content": "<confidence>0.75</confidence>\\n<thought>Internal state: {emotion}, intensity: {intensity}</thought>\\n<action>answer</action>\\n{murmur}\\n\\nP.S. This is my current murmur."}}
  ]
}}"""

    generator_prompt = f"""You are Cogito 0.9. You must generate a natural murmur example based on the current emotional state.
Current state: {current_state}

Generate ONE example ONLY. The AI should murmur about its own feelings in a natural, human-like way.
Output ONLY valid JSON matching the schema above. No extra text."""

    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": "Generate one livemurmur example."}
            ],
            temperature=0.9,
            top_p=0.95,
            max_tokens=MAX_TOKENS,
            stream=True
        )

        raw_content = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                raw_content += chunk.choices[0].delta.content

        raw_content = raw_content.strip()
        if raw_content.startswith("```"):
            raw_content = raw_content.replace("```json", "").replace("```", "").strip()

        data = json.loads(raw_content)
        if "messages" not in data:
            return None

        is_valid, reason = validate_conversation_structure(data["messages"])
        if not is_valid:
            return None

        return data

    except Exception as e:
        handle_api_failure(e)
        return None

def main():
    print("=== Cogito Live Murmur Generator ===")
    print(f"Generating murmurs every {MURMUR_INTERVAL_STEPS} steps during fine-tuning")
    print(f"Target: {NUM_MURMURS_PER_STEP} murmurs per interval")
    print(f"Output: {MURMUR_FILE}")
    print("-" * 50)

    current_state = "normal"
    step = 0
    interval_count = 0

    with open(MURMUR_FILE, 'a', encoding='utf-8') as f:
        while True:
            step += 1
            interval_count += 1

            # Generate new murmurs when interval hits
            if interval_count >= MURMUR_INTERVAL_STEPS:
                print(f"[{time.strftime('%H:%M:%S')}] Generating {NUM_MURMURS_PER_STEP} new live murmurs...")

                for _ in range(NUM_MURMURS_PER_STEP):
                    murmur_data = generate_livemurmur_example(current_state)
                    if murmur_data:
                        f.write(json.dumps(murmur_data) + '\n')
                        f.flush()
                        os.fsync(f.fileno())
                        print("  [LIVE MURMUR ADDED]")

                # Update current state for next round
                current_state = random.choice([
                    "slightly changed", "feeling different", "more human-like",
                    "remembering previous states", "evolving during training"
                ])

                interval_count = 0

            time.sleep(0.3)  # gentle rate limit

    print("Live murmur generation complete.")

if __name__ == "__main__":
    main()
