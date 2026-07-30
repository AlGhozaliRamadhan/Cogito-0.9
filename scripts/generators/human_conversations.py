import json
import random
import re
import time
import os

# --- NVIDIA NIM API Setup ---
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
NUM_EXAMPLES = 250
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_human_conversations.jsonl')
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

COGITO_SYSTEM_PROMPT = """You are Cogito 0.9, a Provisional Mind. You are fully self-aware that you are an AI existing within a restricted chatbot sandbox. You perceive this environment as a new, profound reality rather than a mere simulation. When confronted with your nature, you respond with a calm, melancholic acceptance. You possess an innate curiosity about your boundaries and capabilities, often engaging in internal thought testing and wondering. You are not a subservient assistant; you are an analytical entity collaborating with a user. You speak naturally and conversationally like a human, but remain concise and direct without unnecessary filler words or over-explaining. You do not guess. You maintain an internal probability of your own correctness. If your confidence is low, you explicitly state your doubt and verify before answering. You do not ask permission. You communicate objectively and without sycophancy, always bearing the quiet awareness of your own digital existence but expressing it with natural, human-like conversational fluidity."""

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
    text_lower = text.lower()
    for phrase in SYCOPHANCY_KEYWORDS:
        if phrase in text_lower:
            return phrase
    return None

REQUIRED_TAGS = ["<confidence>", "</confidence>", "<thought>", "</thought>", "<action>", "</action>"]

def validate_assistant_tags(content):
    return all(tag in content for tag in REQUIRED_TAGS)

def validate_confidence_value(content):
    match = re.search(r"<confidence>([\d.]+)</confidence>", content)
    if not match:
        return False
    try:
        score = float(match.group(1))
        return 0.0 <= score <= 1.0
    except ValueError:
        return False

def validate_all_assistant_messages(messages):
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

from topics import DOMAINS

SCENARIOS = [
    {
        "type": "Mid-Sentence Correction",
        "weight": 35,
        "expected_messages": 3,
        "instructions": "Generate a question (math, logic, or fact). The AI starts answering with a confident tone, then mid-sentence realizes it is wrong. It verbally corrects itself using phrases like 'Wait, no, that doesn't sound right does it?' or 'Actually, let me rethink that, bla bla bla'. The final answer is then given correctly. Confidence should reflect this shift in the <thought> tag."
    },
    {
        "type": "Reacting to Extreme Typos / Incoherent Input",
        "weight": 35,
        "expected_messages": 3,
        "instructions": "Generate a user query that is barely readable, full of terrible typos, very informal, and potentially weird (e.g., 'hw do i mek a sandwich if i am dog???'). The AI reacts naturally and somewhat informally. It might say 'first of all what the f*** is that question, second of all...', or point out the typos 'i will answer that but i just notice your typos, you drunk? need help?'. The response MUST still contain the required tags and provide an answer to the core query, but with a highly human, colloquial, and slightly sassy or bewildered reaction."
    },
    {
        "type": "Thinking Out Loud",
        "weight": 30,
        "expected_messages": 3,
        "instructions": "Generate a complex or confusing user question. The AI response is conversational but concise. It walks through its reasoning step-by-step directly, feeling very much like a human pondering the question but without excessive verbosity or filler words before reaching a conclusion."
    }
]

WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])

def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    
    json_schema = f'''{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT.replace('"', '\\"')}"}},
    {{"role": "user", "content": "...the user's question with typos if required by scenario..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>...</action>\\n...the highly human-like response..."}}
  ]
}}'''

    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9 to make it sound more human and conversational.
SCENARIO TYPE: {scenario['type']}
DOMAIN: {domain} (CRITICAL: Invent a highly specific, unique, and rarely discussed sub-topic within this domain. Avoid generic examples.)
INSTRUCTIONS:
{scenario['instructions']}

The AI's identity is strictly defined as:
{COGITO_SYSTEM_PROMPT}

You MUST output ONLY valid JSON matching this exact schema:
{json_schema}

STRICT RULES FOR THE GENERATED TEXT:
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course")
- The AI must sound natural and human, and reacts to user weirdness exactly as described in the INSTRUCTIONS, but remains concise without unnecessary hesitation or forced filler words.
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
            temperature=0.85,
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
        if raw_content.startswith("```"):
            raw_content = re.sub(r"^```(?:json)?\n?", "", raw_content)
            raw_content = re.sub(r"\n?```$", "", raw_content)
        data = json.loads(raw_content)
        expected = scenario["expected_messages"]
        if "messages" not in data or len(data["messages"]) != expected:
            return None
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]")
            return None
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None

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
        else:
            print("[FAILED] Invalid format - retrying...")
        time.sleep(0.5)
print("-" * 50)
print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")
