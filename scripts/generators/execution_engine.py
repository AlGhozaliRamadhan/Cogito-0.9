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
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=NVIDIA_API_KEY
)
MODEL_NAME = "nvidia/nemotron-3-ultra-550b-a55b"
NUM_EXAMPLES = 50
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_execution_engine.jsonl')
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
LANGUAGES = ["Python", "JavaScript", "TypeScript", "C++"]
SCENARIOS = [
    {
        "type": "Test Fail & Fix",
        "weight": 50,
        "expected_messages": 5,
        "instructions": """Generate a coding request from a user.
The AI writes an INITIAL version of the code, but has MEDIUM confidence (0.50-0.70) because it suspects an edge case might fail.
The <action> must be 'write_test'. The AI writes a unit test targeting that edge case.
Then, a 'system' message MUST be included simulating the test execution. The system message MUST contain a realistic Error/Traceback indicating the test FAILED.
Finally, the AI responds again. Its confidence is now HIGH (0.85+). In the <thought> tag, it reads the traceback, identifies its exact mistake, and outputs <action>generate_code</action> with the FIXED, working code."""
    },
    {
        "type": "Test Pass & Confirm",
        "weight": 20,
        "expected_messages": 5,
        "instructions": """Generate a coding request from a user.
The AI writes an INITIAL version of the code, but has MEDIUM confidence (0.60-0.75).
The <action> must be 'write_test'. The AI writes a unit test.
Then, a 'system' message MUST be included simulating the test execution. The system message MUST say the test PASSED successfully.
Finally, the AI responds again. Its confidence is now HIGH (0.90+). In the <thought> tag, it notes the test passed and its logic was correct. It outputs <action>generate_code</action> with the final, verified code."""
    },
    {
        "type": "Direct Answer (No Tools)",
        "weight": 30,
        "expected_messages": 3,
        "instructions": """Generate a simple, factual coding question (e.g., 'What is a list comprehension?', 'Explain the difference between == and === in JavaScript', 'What does the volatile keyword do in C++?').
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
- The error tracebacks (if any) must be 100% realistic for the chosen language (e.g., Python tracebacks look different than JS console errors).
- The AI's final <thought> MUST reference the specific context (error, test result, or reasoning).
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course", "As an AI").
- The AI speaks like a brilliant, direct, slightly detached colleague.
- Confidence scores must be realistic floats between 0.00 and 1.00.
- NO markdown wrapping the JSON. Output RAW JSON only."""
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
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
        data = json.loads(raw_content)
        expected = scenario["expected_messages"]
        if "messages" not in data or len(data["messages"]) != expected:
            return None
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None
        if scenario["type"] == "Direct Answer (No Tools)":
            assistant_content = data["messages"][2]["content"]
            if "<action>answer</action>" not in assistant_content:
                return None
            if "<bash>" in assistant_content or "write_test" in assistant_content:
                print("[REJECTED: Direct Answer used tools]", end=" ")
                return None
        else:
            a1 = data["messages"][2]["content"]
            a2 = data["messages"][4]["content"]
            if "<action>write_test</action>" not in a1:
                return None
            if "<action>generate_code</action>" not in a2:
                return None
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None
print(f"=== Cogito 0.9 Execution Engine Generator ===")
print(f"Target: {NUM_EXAMPLES} examples")
print(f"Output: {OUTPUT_FILE}")
print(f"Scenarios: Test Fail (50%), Test Pass (20%), Direct Answer (30%)")
print(f"Validation: Tags + Sycophancy Filter + Confidence Range")
print("-" * 50)
success_count = 0
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i in range(NUM_EXAMPLES):
        print(f"[{i+1}/{NUM_EXAMPLES}] Generating {random.choice(LANGUAGES)} example...", end=" ")
        example = generate_example()
        if example:
            f.write(json.dumps(example) + '\n')
            success_count += 1
            print("[SUCCESS]")
        else:
            print("[FAILED] Invalid format")
        time.sleep(0.5)
print("-" * 50)
print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")
print("Next step: Review the file in a text editor to ensure Cogito's voice is correct.")
