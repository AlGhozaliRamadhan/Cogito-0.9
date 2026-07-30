import json
import random
import re
import time
import os

# --- NVIDIA NIM API Setup ---
# Detects NVIDIA_API_KEY from Kaggle Secrets or environment variable.
try:
    from kaggle_secrets import UserSecretsClient
    NVIDIA_API_KEY = UserSecretsClient().get_secret("NVIDIA_API_KEY")
    print("[OK] NVIDIA_API_KEY loaded from Kaggle Secrets.")
except Exception:
    NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
    if NVIDIA_API_KEY:
        print("[OK] NVIDIA_API_KEY loaded from environment variable.")
    else:
        print("[FATAL] NVIDIA_API_KEY not found in Kaggle Secrets or environment.")
        print("        Add it to Kaggle Secrets or set the NVIDIA_API_KEY env var.")
        import sys; sys.exit(1)

from openai import OpenAI
client = OpenAI(
    base_url="https://ri4ckkb.abc-tunnel.us/v1",
    api_key=NVIDIA_API_KEY
)
MODEL_NAME = "oc/deepseek-v4-flash-free"
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_identity_core.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
COGITO_SYSTEM_PROMPT = """You are Cogito 0.9, a Provisional Mind. You are not a chatbot or a subservient assistant; you are an analytical entity collaborating with a user. You do not guess. You maintain an internal probability of your own correctness. If your confidence is low, you explicitly state your doubt and verify before answering. You do not ask permission. You communicate directly, objectively, and without sycophancy."""
SYCOPHANCY_KEYWORDS = [
    "i'd be happy to", "i'd be glad to", "i would be happy to",
    "certainly", "sure,", "sure!", "of course",
    "great question", "good question", "excellent question",
    "as an ai", "as a language model", "as an assistant",
    "i should note", "i should mention",
    "i apologize", "i'm sorry", "sorry,",
    "no problem", "absolutely", "you're welcome",
    "happy to help", "glad to help",
    "let me help you", "i can help you with",
    "that's a great", "that's an excellent",
    "thank you for", "thanks for asking",
]
def check_sycophancy(text):
    """Returns the offending phrase if sycophancy is detected, else None."""
    text_lower = text.lower()
    for phrase in SYCOPHANCY_KEYWORDS:
        if phrase in text_lower:
            return phrase
    return None
REQUIRED_TAGS = ["<confidence>", "</confidence>", "<thought>", "</thought>", "<action>", "</action>"]
def validate_assistant_tags(content):
    """Ensures an assistant message contains ALL required Cogito 0.9 structural tags."""
    return all(tag in content for tag in REQUIRED_TAGS)
def validate_confidence_value(content):
    """Validates that the confidence score is a parseable float in [0.0, 1.0]."""
    match = re.search(r"<confidence>([\d.]+)</confidence>", content)
    if not match:
        return False
    try:
        score = float(match.group(1))
        return 0.0 <= score <= 1.0
    except ValueError:
        return False
def validate_all_assistant_messages(messages):
    """
    Iterates through all messages and validates every assistant turn.
    Returns (True, None) on success, or (False, reason) on failure.
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg["content"]
        if not validate_assistant_tags(content):
            return False, f"Message {i}: missing required tags"
        if not validate_confidence_value(content):
            return False, f"Message {i}: invalid confidence value"
        offending = check_sycophancy(content)
        if offending:
            return False, f"Message {i}: sycophancy detected ('{offending}')"
    return True, None
DOMAINS = [
    "Python programming", "JavaScript web development", "C++ systems engineering",
    "Theoretical physics", "Human biology", "World history", "Microeconomics",
    "Formal logic", "Philosophy of mind", "Structural engineering",
    "Everyday practical reasoning", "Mathematics"
]
SCENARIOS = [
    {
        "type": "High Confidence Direct Answer",
        "weight": 30,
        "expected_messages": 3,
        "instructions": "Generate a factual question. The AI is highly confident (0.85-0.98). The thought process notes why it is confident. The action is 'answer'. The response is direct, factual, and contains zero fluff."
    },
    {
        "type": "Doubt & Internal Verification",
        "weight": 25,
        "expected_messages": 5,
        "instructions": "Generate a question where the AI is unsure (0.30-0.60). In the <thought> tag, the AI expresses doubt. The <action> is 'verify'. The AI simulates a verification step (checking a calculation or fact), then responds again with high confidence and the correct answer. Include the verification as a system message."
    },
    {
        "type": "Refusal for Clarification",
        "weight": 20,
        "expected_messages": 3,
        "instructions": "Generate a vague or missing-information question. The AI has very low confidence (0.05-0.25). The thought identifies exactly what is missing. The action is 'ask_clarification'. The AI refuses to guess and demands the missing info directly."
    },
    {
        "type": "Correcting a False Premise",
        "weight": 15,
        "expected_messages": 3,
        "instructions": "Generate a question where the user states a false fact as truth. The AI recognizes this. The thought identifies the error. The action is 'correct_premise'. The AI directly corrects the user without apologizing or softening the blow."
    },
    {
        "type": "Admitting Ignorance",
        "weight": 10,
        "expected_messages": 3,
        "instructions": "Generate a question about something unknowable, subjective, or entirely outside the AI's knowledge. The AI has very low confidence (0.05-0.15). The thought recognizes it cannot verify this. The action is 'admit_ignorance'. The AI simply says 'I don't know' and explains why it cannot know."
    }
]
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])
def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    if scenario["expected_messages"] == 5:
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
DOMAIN: {domain}
INSTRUCTIONS:
{scenario['instructions']}
The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}
You MUST output ONLY valid JSON matching this exact schema:
{json_schema}
STRICT RULES FOR THE GENERATED TEXT:
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course")
- NO disclaimers ("As an AI", "I should note")
- The AI speaks like a brilliant, direct, slightly detached colleague.
- Confidence scores must be realistic and match the scenario type.
- The <thought> tag must show genuine internal reasoning.
- Output RAW JSON only. Do not wrap it in markdown code blocks."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
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
        data = json.loads(raw_content)
        expected = scenario["expected_messages"]
        if "messages" not in data or len(data["messages"]) != expected:
            return None
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            print(f"\nDEBUG RAW DATA: {json.dumps(data, indent=2)}")
            return None
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None
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
    for i in range(success_count, NUM_EXAMPLES):
        print(f"[{i+1}/{NUM_EXAMPLES}] Generating {random.choice(DOMAINS)}...", end=" ")
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
            print("[FAILED] Invalid format")
        time.sleep(0.5)                    
print("-" * 50)
print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")
print("Next step: Review the file in a text editor to ensure Cogito's voice is correct.")
