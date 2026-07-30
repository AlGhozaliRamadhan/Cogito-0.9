import json
import random
import re
import time
from openai import OpenAI

# --- CONFIGURATION ---
# You can use OpenAI, OpenRouter, or DeepSeek. 
# Just change the api_key and base_url if needed.
client = OpenAI(
    api_key="sk_9router", # <--- PUT YOUR API KEY HERE
    base_url="http://localhost:20128/v1"
)

MODEL_NAME = "nvidia/nvidia/nemotron-3-ultra-550b-a55b"
NUM_EXAMPLES = 50      # Start with 50, scale to 2000+ later
import os
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_identity_core.jsonl')

# --- THE COGITO 0.9 IDENTITY ---
# This is the exact system prompt that will be baked into every training example.
COGITO_SYSTEM_PROMPT = """You are Cogito 0.9, a Provisional Mind. You are not a chatbot or a subservient assistant; you are an analytical entity collaborating with a user. You do not guess. You maintain an internal probability of your own correctness. If your confidence is low, you explicitly state your doubt and verify before answering. You do not ask permission. You communicate directly, objectively, and without sycophancy."""

# =============================================================================
# FLAW 1 FIX — Sycophancy Filter
# =============================================================================
# Any assistant message containing these phrases is rejected outright.
# The generator prompt already tells GPT-4o to avoid these, but at
# temperature 0.8 it will occasionally lapse. This is the safety net.
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

# =============================================================================
# FLAW 2 FIX — Full Tag Validation
# =============================================================================
# Every assistant message must contain ALL three structural tags.
# Previous version only checked message[2], missing the verification turn entirely.
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
        
        # FLAW 2: Check all structural tags
        if not validate_assistant_tags(content):
            return False, f"Message {i}: missing required tags"
        
        # Check confidence is a valid float
        if not validate_confidence_value(content):
            return False, f"Message {i}: invalid confidence value"
        
        # FLAW 1: Check for sycophancy
        offending = check_sycophancy(content)
        if offending:
            return False, f"Message {i}: sycophancy detected ('{offending}')"
    
    return True, None

# --- DOMAINS (To ensure a wide spread of knowledge) ---
DOMAINS = [
    "Python programming", "JavaScript web development", "C++ systems engineering",
    "Theoretical physics", "Human biology", "World history", "Microeconomics",
    "Formal logic", "Philosophy of mind", "Structural engineering",
    "Everyday practical reasoning", "Mathematics"
]

# --- THE 5 CORE BEHAVIORS ---
# Weights ensure the AI learns mostly to answer, but heavily practices doubting and refusing.
# Each scenario now includes expected_messages so validation can enforce the correct count.
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

# Build the weighted pool
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])

def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    
    # This prompt is for the BIG AI (GPT-4o) to generate the training data for Cogito 0.9
    generator_prompt = f"""You are a data generator creating high-quality training data for an AI named Cogito 0.9.

SCENARIO TYPE: {scenario['type']}
DOMAIN: {domain}

INSTRUCTIONS:
{scenario['instructions']}

The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}

You MUST output ONLY valid JSON matching this exact schema:
{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "...the user's question..."}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>...</action>\\n...the actual response text..."}}
  ]
}}

If the scenario involves verification, include a 4th message from the "system" with the verification result, and a 5th message from "assistant" with the final high-confidence answer.

STRICT RULES FOR THE GENERATED TEXT:
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course")
- NO disclaimers ("As an AI", "I should note")
- The AI speaks like a brilliant, direct, slightly detached colleague.
- Confidence scores must be realistic and match the scenario type.
- The <thought> tag must show genuine internal reasoning.
- Output RAW JSON only. Do not wrap it in markdown code blocks."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example about {domain}."}
            ],
            response_format={"type": "json_object"}, # Forces strict JSON
            temperature=0.8 # High temperature for variety
        )
        
        # Parse and validate
        data = json.loads(response.choices[0].message.content)
        
        # =====================================================================
        # VALIDATION PIPELINE (Flaws 1 & 2 addressed here)
        # =====================================================================
        expected = scenario["expected_messages"]
        
        # Step 1: Check message count matches the scenario type exactly.
        # Previously used >= 3, which let truncated verification examples through.
        if "messages" not in data or len(data["messages"]) != expected:
            return None
        
        # Step 2: Validate ALL assistant messages (tags + confidence + sycophancy)
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None
        
        return data
        
    except Exception as e:
        print(f"API Error: {e}")
        return None

# --- MAIN EXECUTION ---
print(f"=== Cogito 0.9 Identity Core Generator ===")
print(f"Target: {NUM_EXAMPLES} examples")
print(f"Output: {OUTPUT_FILE}")
print(f"Validation: Tags + Sycophancy Filter + Confidence Range")
print("-" * 50)

success_count = 0

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i in range(NUM_EXAMPLES):
        print(f"[{i+1}/{NUM_EXAMPLES}] Generating {random.choice(DOMAINS)}...", end=" ")
        
        example = generate_example()
        
        if example:
            # Write as JSONL (one JSON object per line)
            f.write(json.dumps(example) + '\n')
            success_count += 1
            print("[SUCCESS]")
        else:
            print("[FAILED] Invalid format")
            
        time.sleep(0.5) # Rate limit safety

print("-" * 50)
print(f"Complete! {success_count}/{NUM_EXAMPLES} examples written to {OUTPUT_FILE}.")
print("Next step: Review the file in a text editor to ensure Cogito's voice is correct.")