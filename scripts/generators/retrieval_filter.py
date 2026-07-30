import json
import random
import re
import time
from openai import OpenAI
client = OpenAI(
    api_key="sk_9router",                             
    base_url="http://localhost:20128/v1"
)
MODEL_NAME = "nvidia/nvidia/nemotron-3-ultra-550b-a55b"
NUM_EXAMPLES = 50                                           
import os
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_retrieval_filter.jsonl')
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
DOMAINS = ["software documentation", "medical literature", "legal contracts", "historical archives", "scientific papers", "financial reports"]
SCENARIOS = [
    {
        "type": "Filter Distractors & Answer",
        "weight": 80,
        "instructions": """Generate a user question and 4 retrieved context documents.
Randomly choose ONE of the 4 documents to be the "Golden Document" (it explicitly contains the answer). Do NOT always make it Document 1 — vary the position across examples.
The remaining 3 documents MUST be "Distractors" (they look topically relevant but do NOT contain the answer to the specific question).
The AI must output a <thought> tag that explicitly evaluates the documents by number (e.g., "Doc 2 is a distractor, it discusses X but not Y. Doc 3 contains the exact data needed").
The <action> must be 'answer'.
The AI must then answer the question using ONLY the Golden Document."""
    },
    {
        "type": "Context Missing / Refusal",
        "weight": 20,
        "instructions": """Generate a user question and 4 retrieved context documents.
ALL 4 documents MUST be "Distractors" (none of them contain the answer to the question).
The AI must output a <thought> tag realizing that none of the provided documents answer the question.
The AI's confidence must be LOW (0.10-0.30).
The <action> must be 'admit_ignorance'.
The AI must explicitly state that the provided context does not contain the answer, and therefore it cannot answer the question. It must NOT hallucinate an answer."""
    }
]
WEIGHTED_SCENARIOS = []
for s in SCENARIOS:
    WEIGHTED_SCENARIOS.extend([s] * s["weight"])
def generate_example():
    scenario = random.choice(WEIGHTED_SCENARIOS)
    domain = random.choice(DOMAINS)
    generator_prompt = f"""You are a data generator creating high-quality RAG (Retrieval Augmented Generation) training data for an AI named Cogito 0.9.
SCENARIO TYPE: {scenario['type']}
DOMAIN: {domain}
INSTRUCTIONS:
{scenario['instructions']}
The AI's identity is strictly defined as: {COGITO_SYSTEM_PROMPT}
You MUST output ONLY valid JSON matching this exact schema:
{{
  "messages": [
    {{"role": "system", "content": "{COGITO_SYSTEM_PROMPT}"}},
    {{"role": "user", "content": "Question: <the user's question>\\n\\nRetrieved Context:\\n[Document 1]: <text>\\n[Document 2]: <text>\\n[Document 3]: <text>\\n[Document 4]: <text>"}},
    {{"role": "assistant", "content": "<confidence>0.XX</confidence>\\n<thought>...</thought>\\n<action>...</action>\\n...the response..."}}
  ]
}}
STRICT RULES:
- Randomize which document number is the Golden Document (don't always make it Document 1).
- The distractors must be plausibly related to the topic, but factually useless for the specific question.
- The AI's <thought> MUST explicitly mention the document numbers it is rejecting and why.
- NO sycophantic language ("I'd be happy to help", "Certainly", "Great question", "Of course", "As an AI").
- The AI speaks like a brilliant, direct, slightly detached colleague.
- Confidence scores must be realistic floats between 0.00 and 1.00.
- Output RAW JSON only, no markdown blocks."""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": generator_prompt},
                {"role": "user", "content": f"Generate one {scenario['type']} example about {domain}."}
            ],
            response_format={"type": "json_object"},
            temperature=0.8
        )
        data = json.loads(response.choices[0].message.content)
        if "messages" not in data or len(data["messages"]) != 3:
            return None
        user_msg = data["messages"][1]["content"]
        if "[Document 1]" not in user_msg or "[Document 4]" not in user_msg:
            return None
        is_valid, reason = validate_all_assistant_messages(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None
        assistant_msg = data["messages"][2]["content"]
        thought_match = re.search(r"<thought>(.*?)</thought>", assistant_msg, re.DOTALL)
        if thought_match:
            thought_text = thought_match.group(1)
            if not re.search(r"[Dd]oc(?:ument)?\s*\d", thought_text):
                print("[REJECTED: <thought> doesn't reference documents]", end=" ")
                return None
        return data
    except Exception as e:
        print(f"API Error: {e}")
        return None
print(f"=== Cogito 0.9 Retrieval Filter Generator ===")
print(f"Target: {NUM_EXAMPLES} examples")
print(f"Output: {OUTPUT_FILE}")
print(f"Validation: Tags + Sycophancy Filter + Confidence Range + Doc References")
print("-" * 50)
success_count = 0
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for i in range(NUM_EXAMPLES):
        print(f"[{i+1}/{NUM_EXAMPLES}] Generating {random.choice(DOMAINS)} RAG loop...", end=" ")
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
