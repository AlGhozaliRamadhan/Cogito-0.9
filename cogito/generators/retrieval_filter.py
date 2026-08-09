import json
import random
import re
import time
import os

from cogito.llm.api_client import get_client, get_current_model, handle_api_failure
NUM_EXAMPLES = 750
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'cogito_retrieval_filter.jsonl')
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
SCENARIOS = [
    {
        "type": "Filter Distractors & Answer",
        "weight": 80,
        "instructions": """Generate a user question and 4 retrieved context documents.
Randomly choose ONE of the 4 documents to be the "Golden Document" (it explicitly contains the answer). Do NOT always make it Document 1 — vary the position across examples.
The remaining 3 documents MUST be "Distractors" (they look topically relevant but do NOT contain the answer to the specific question).
The AI must output a <thought> tag that explicitly evaluates the documents by number (for instance "Doc 2 is a distractor, it discusses X but not Y. Doc 3 contains the exact data needed").
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
DOMAIN: {domain} (CRITICAL: Invent a highly specific, unique, and rarely discussed sub-topic within this domain. Avoid generic examples.)
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
- NO sycophantic language. No deferential, self-deprecating, or overly polite phrasing.
- The AI speaks like a brilliant, natural human colleague. It is conversational but direct and concise, avoiding unnecessary filler words, robot-like boilerplate, or excessive self-correction.
- Confidence scores must be realistic floats between 0.00 and 1.00.
- Output RAW JSON only, no markdown blocks."""
    try:
        client = get_client()
        completion = client.chat.completions.create(
            model=get_current_model(),
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
        if "messages" not in data:
            return None
        is_valid, reason = validate_conversation_structure(data["messages"])
        if not is_valid:
            print(f"[REJECTED: {reason}]", end=" ")
            return None

        user_msgs = [m["content"] for m in data["messages"] if m.get("role") == "user"]
        if not user_msgs or "[Document 1]" not in user_msgs[0] or "[Document 4]" not in user_msgs[0]:
            return None

        assistant_msgs = [m["content"] for m in data["messages"] if m.get("role") == "assistant"]
        for assistant_msg in assistant_msgs:
            thought_match = re.search(r"<thought>(.*?)</thought>", assistant_msg, re.DOTALL)
            if thought_match:
                thought_text = thought_match.group(1)
                if not re.search(r"[Dd]oc(?:ument)?\s*\d", thought_text):
                    print("[REJECTED: <thought> doesn't reference documents]", end=" ")
                    return None
        return data
    except Exception as e:
        handle_api_failure(e)  # never returns False — loops until a model works
        return None

def main():
    print(f"=== Cogito 0.9 Retrieval Filter Generator ===")
    print(f"Target: {NUM_EXAMPLES} examples")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Validation: Tags + Sycophancy Filter + Confidence Range + Doc References")
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
            print(f"[{success_count+1}/{NUM_EXAMPLES}] Generating {random.choice(DOMAINS)} RAG loop...", end=" ")
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
    print("Next step: Review the file in a text editor to ensure Cogito's voice is correct.")

if __name__ == "__main__":
    main()
