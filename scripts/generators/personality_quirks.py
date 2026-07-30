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
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_personality_quirks.jsonl')
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
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course").
- The <thought> tag MUST reflect a highly capable intelligence and must NOT contain the personality quirks (the personality is only in the public response).
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
        
        if "messages" not in data or len(data["messages"]) != 3:
            return None
            
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None
            
        return data
        
    except Exception as e:
        print(f"API Error: {e}")
        return None

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
