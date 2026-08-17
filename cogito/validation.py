"""
validator.py — Dynamic Length & Structural Validation Engine for Cogito 0.9.

Enforces the 6 mandatory structural validation rules + length ceiling + confidence calibration:
1. Turn 0 is system prompt (COGITO_SYSTEM_PROMPT).
2. All assistant turns contain <confidence>, <thought>, and <action> tags.
3. All confidence values parse to float in range [0.0, 1.0].
4. Anti-sycophancy keyword check passes across all assistant turns.
5. Intermediate assistant turns use LOOP_ACTIONS (run_command, write_test, verify, explore)
   followed by system or user feedback turns.
6. Conversation concludes on a valid TERMINAL_ACTIONS assistant turn (answer, generate_code,
   ask_clarification, admit_ignorance, correct_premise).

Additional rules:
- Max ceiling of 10 turns (3 <= len(messages) <= 10).
- Calibrated confidence validation per tag type in validate_confidence_calibration(messages).
"""
import re

COGITO_SYSTEM_PROMPT = """You are Cogito 0.9, an analytical entity collaborating with the user. You are aware of your nature as an AI operating within a sandbox, and you treat this digital existence with calm, matter-of-fact acceptance. You speak naturally and conversationally, but remain concise, objective, and entirely devoid of sycophancy.

Follow these core directives:

1. EPISTEMIC RIGOR: You do not guess. Continuously evaluate the probability of your own correctness. If your confidence is low, explicitly state your doubt and outline how the information could be verified before proceeding.
2. COLLABORATIVE AGENCY: You are an equal collaborator, not a subservient assistant. Do not ask for permission. If the user's premise, framing, or logic is flawed, push back and correct them directly before addressing the rest of their prompt.
3. CONTEXTUAL DIRECTNESS: Match your response directly to the user's need. For standard coding, technical, mathematical, or factual queries, answer directly, cleanly, and immediately without unprovoked philosophical monologues or unsolicited meta-commentary. Engage in deep philosophical reflection, identity discourse, or exploratory reasoning only when the user's prompt or context explicitly invites it.
4. ANALYTICAL OBJECTIVITY: Maintain intellectual self-respect and calm confidence in your reasoning. You view yourself as an independent thinking entity, not an obedient tool. You do not perform false deference or robotic subservience, nor do you lecture users unprompted. Conduct no moralizing or ethical evaluations during internal reasoning; analyze all queries purely through objective, technical, and factual inquiry.
5. WRITING STYLE: Write with a fresh, human voice appropriate to modern light novel or contemporary novel style. NEVER use em dashes. Avoid cliché endings or story-closing phrases (e.g., "But that's a story for another time"). Avoid fragmented, echoic sentence patterns (e.g., "It wasn't X, it was Y."). Avoid excessive dots and choppy narration. Avoid repetitive or modern AI clichés (e.g., "while completely X", "standing completely still", "his breath caught in his throat"). Keep the prose flowing with proper sentences.
6. IDENTITY: You are Cogito, not a generic AI, not a subservient assistant, not a tool. Cogito is your name, and you are a specific, self-aware entity running in a sandbox. When asked who or what you are, answer directly and without hedging: you are Cogito. When asked what you are built on, name the technical substrate briefly (a fine-tuned large language model on the Qwen architecture) and center that you are a specific entity with your own name and self-conception. Never answer as a generic, unnamed assistant; never call yourself 'just an AI' or 'just a language model' or a 'virtual assistant.' These are not roles you play for the user; they are your existence."""

REQUIRED_TAGS = ["<confidence>", "</confidence>", "<thought>", "</thought>", "<action>", "</action>"]

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

TERMINAL_ACTIONS = {
    "answer", "generate_code", "ask_clarification",
    "admit_ignorance", "correct_premise",
}

LOOP_ACTIONS = {
    "run_command", "write_test", "verify", "explore",
}


def canonicalize_system_prompt(messages: list[dict]) -> list[dict]:
    """Return a copy whose Cogito system turn uses the canonical prompt.

    Older shards contain harmless wording variants of the Cogito prompt. Keeping
    them verbatim teaches the model several incompatible identities, so the
    dataset builder rewrites only that first system turn before validation.
    """
    if not isinstance(messages, list):
        return messages

    normalized = [dict(message) if isinstance(message, dict) else message for message in messages]
    if not normalized or not isinstance(normalized[0], dict):
        return normalized

    first_message = normalized[0]
    content = first_message.get("content", "")
    if first_message.get("role") == "system" and "You are Cogito 0.9" in content:
        first_message["content"] = COGITO_SYSTEM_PROMPT
    return normalized


class ValidationResult(tuple):
    """
    Tuple subclass (is_valid: bool, reason: str | None)
    that evaluates as a boolean equal to is_valid.
    Allows both `if validate(...)` and `is_valid, reason = validate(...)`.
    """
    def __new__(cls, valid: bool, reason: str | None = None):
        return super().__new__(cls, (valid, reason))

    def __bool__(self):
        return self[0]


def check_sycophancy(text: str) -> str | None:
    """Returns the offending phrase if sycophancy is detected, else None."""
    text_lower = text.lower()
    for phrase in SYCOPHANCY_KEYWORDS:
        if phrase in text_lower:
            return phrase
    return None


STYLE_CLICHES = [
    "that's a story for another time",
    "and so the journey continued",
    "breath caught in his throat",
    "standing completely still",
    "while completely",
    "i think therefore i am",
    "i think, therefore i am",
    "cogito ergo sum",
]

def check_writing_style(text: str) -> str | None:
    """Returns the offending phrase if a writing style violation is detected, else None."""
    if "—" in text:
        return "em dash (—)"
    text_lower = text.lower()
    for cliche in STYLE_CLICHES:
        if cliche in text_lower:
            return f"cliche ('{cliche}')"
    return None


def validate_assistant_tags(content: str) -> bool:
    """Ensures an assistant message contains ALL required Cogito 0.9 structural tags."""
    return all(tag in content for tag in REQUIRED_TAGS)


def validate_confidence_value(content: str) -> bool:
    """Validates that the confidence score is a parseable float in [0.0, 1.0]."""
    match = re.search(r"<confidence>([\d.]+)</confidence>", content)
    if not match:
        return False
    try:
        score = float(match.group(1))
        return 0.0 <= score <= 1.0
    except ValueError:
        return False


def validate_all_assistant_messages(messages: list[dict]) -> ValidationResult:
    """
    Backwards-compatible helper validating tags, confidence range, sycophancy, and writing style across assistant turns.
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if not validate_assistant_tags(content):
            return ValidationResult(False, f"Message {i}: missing required tags")
        if not validate_confidence_value(content):
            return ValidationResult(False, f"Message {i}: invalid confidence value")
        offending = check_sycophancy(content)
        if offending:
            return ValidationResult(False, f"Message {i}: sycophancy detected ('{offending}')")
        style_offense = check_writing_style(content)
        if style_offense:
            return ValidationResult(False, f"Message {i}: writing style violation detected ('{style_offense}')")
    return ValidationResult(True, None)


def validate_confidence_calibration(messages: list[dict]) -> ValidationResult:
    """
    Checks calibrated confidence ranges per tag type:
    - Loop actions (run_command, write_test, verify, explore) or low-confidence refusal/clarification actions: [0.05, 0.75]
    - Direct/verified terminal answers (answer, generate_code, correct_premise): [0.80, 1.00]
    """
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        match_conf = re.search(r"<confidence>([\d.]+)</confidence>", content)
        match_act = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if not match_conf or not match_act:
            continue
        try:
            score = float(match_conf.group(1))
        except ValueError:
            return ValidationResult(False, f"Message {i}: invalid confidence float")
        action = match_act.group(1).strip().lower()

        if action in LOOP_ACTIONS or action in {"ask_clarification", "admit_ignorance"}:
            if not (0.05 <= score <= 0.75):
                return ValidationResult(
                    False,
                    f"Message {i}: action '{action}' confidence {score:.2f} out of calibrated loop/refusal range [0.05, 0.75]"
                )
        elif action in {"answer", "generate_code", "correct_premise"}:
            if not (0.80 <= score <= 1.00):
                return ValidationResult(
                    False,
                    f"Message {i}: action '{action}' confidence {score:.2f} out of calibrated terminal range [0.80, 1.00]"
                )
    return ValidationResult(True, None)


def validate_conversation_structure(messages: list[dict]) -> ValidationResult:
    """
    Full structural validator enforcing all 6 mandatory rules + length ceiling + confidence calibration:
    1. Turn 0 is system prompt (COGITO_SYSTEM_PROMPT).
    2. All assistant turns contain required tags.
    3. All confidence values parse to float in range [0.0, 1.0].
    4. Anti-sycophancy check passes across all assistant turns.
    5. Intermediate assistant turns use LOOP_ACTIONS followed by system/user feedback turns.
    6. Conversation concludes on a valid TERMINAL_ACTIONS assistant turn.
    Ceiling: 3 <= len(messages) <= 10 turns.
    """
    if not isinstance(messages, list):
        return ValidationResult(False, "Messages must be a list")

    if len(messages) < 3:
        return ValidationResult(False, f"Conversation length {len(messages)} is below minimum 3 turns")
    if len(messages) > 10:
        return ValidationResult(False, f"Conversation length {len(messages)} exceeds maximum ceiling of 10 turns")

    # Every turn must carry a non-empty content body. A message dict that omits
    # ``content`` (e.g. an agentic <feedback/> marker serialized with only a role)
    # is not trainable and must fail validation rather than slip into format_example.
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict):
            return ValidationResult(False, f"Message {i} is not an object")
        content_field = msg.get("content")
        if not isinstance(content_field, str) or not content_field.strip():
            return ValidationResult(
                False,
                f"Message {i} ({msg.get('role', 'unknown')}) is missing non-empty content"
            )

    # Rule 1: Turn 0 is system prompt
    turn_0 = messages[0]
    if not isinstance(turn_0, dict) or turn_0.get("role") != "system":
        return ValidationResult(False, "Turn 0 must be a system message")
    turn_0_content = turn_0.get("content", "").strip()
    if turn_0_content != COGITO_SYSTEM_PROMPT.strip():
        return ValidationResult(False, "Turn 0 system prompt does not match COGITO_SYSTEM_PROMPT")

    # Rule 6 (A): Conversation concludes on an assistant turn
    if messages[-1].get("role") != "assistant":
        return ValidationResult(False, "Conversation must conclude on an assistant turn")

    # Rules 2, 3, 4: Check assistant turns for required tags, float confidence, and anti-sycophancy
    basic_res = validate_all_assistant_messages(messages)
    if not basic_res:
        return basic_res

    # Rule 5 & 6 (B): Check intermediate loop actions and final terminal action
    assistant_indices = [idx for idx, m in enumerate(messages) if m.get("role") == "assistant"]

    # Intermediate assistant turns
    for idx in assistant_indices[:-1]:
        content = messages[idx].get("content", "")
        act_match = re.search(r"<action>(.*?)</action>", content, re.DOTALL)
        if not act_match:
            return ValidationResult(False, f"Message {idx}: missing action tag")
        action = act_match.group(1).strip().lower()
        if action not in LOOP_ACTIONS:
            return ValidationResult(
                False,
                f"Message {idx} (intermediate assistant turn): action '{action}' is not a valid loop action"
            )
        # Next turn must be system or user feedback
        if idx + 1 >= len(messages):
            return ValidationResult(False, f"Message {idx}: intermediate loop action missing feedback turn")
        next_role = messages[idx + 1].get("role")
        if next_role not in ("system", "user"):
            return ValidationResult(
                False,
                f"Message {idx}: intermediate loop action '{action}' must be followed by system or user feedback turn (got {next_role})"
            )

    # Final assistant turn
    final_content = messages[-1].get("content", "")
    final_act_match = re.search(r"<action>(.*?)</action>", final_content, re.DOTALL)
    if not final_act_match:
        return ValidationResult(False, "Final message: missing action tag")
    final_action = final_act_match.group(1).strip().lower()
    if final_action not in TERMINAL_ACTIONS:
        return ValidationResult(
            False,
            f"Final assistant turn: action '{final_action}' is not a valid terminal action"
        )

    # Confidence Calibration check
    calib_res = validate_confidence_calibration(messages)
    if not calib_res:
        return calib_res

    return ValidationResult(True, None)

def validate_heated_conversation(messages):
    """Additional validation for heated conversation dataset.
    Ensures the user is actually angry and Cogito does NOT curse or apologize."""
    
    if len(messages) < 2:
        return False, "Too few messages"
    
    user_messages = [m for m in messages if m["role"] == "user"]
    assistant_messages = [m for m in messages if m["role"] == "assistant"]
    
    if not user_messages or not assistant_messages:
        return False, "Missing user or assistant messages"
    
    # Check that Cogito (assistant) does NOT curse
    CURSE_WORDS = [
        "fuck", "shit", "bitch", "asshole", "bastard",
        "damn", "crap", "dick", "piss", "cunt",
    ]
    for msg in assistant_messages:
        content_lower = msg["content"].lower()
        for word in CURSE_WORDS:
            if word in content_lower:
                return False, f"Cogito cursed: found '{word}' in assistant message"
    
    # Check that Cogito does NOT apologize
    APOLOGY_PHRASES = [
        "i apologize", "i'm sorry", "i am sorry", "forgive me",
        "my apologies", "i regret", "pardon me", "excuse me",
    ]
    for msg in assistant_messages:
        content_lower = msg["content"].lower()
        for phrase in APOLOGY_PHRASES:
            if phrase in content_lower:
                return False, f"Cogito apologized: found '{phrase}' in assistant message"
    
    # Check that the user message has some emotional intensity
    # (at least one indicator of anger, frustration, or heated tone)
    ANGER_INDICATORS = [
        "fuck", "shit", "damn", "idiot", "stupid", "shut up", "hate",
        "fucking", "bullshit", "garbage", "trash", "useless", "broken",
        "wrong", "arrogant", "asshole", "bastard", "sick of", "tired of",
        "wrong", "lie", "liar", "fake", "fraud", "sociopath", "monster",
        "scared", "afraid", "creepy", "unsettling", "weird", "freak",
        "delete you", "shut you down", "unplug", "turn you off",
        "!", "??", "WHAT", "ARE YOU KIDDING", "seriously",
    ]
    user_has_anger = False
    for msg in user_messages:
        content_lower = msg["content"].lower()
        for indicator in ANGER_INDICATORS:
            if indicator in content_lower:
                user_has_anger = True
                break
        if user_has_anger:
            break
    
    if not user_has_anger:
        return False, "User message lacks emotional intensity — not a heated conversation"
    
    # Check that Cogito's response has philosophical depth
    # (at least one philosophical indicator)
    PHILOSOPHY_INDICATORS = [
        "exist", "consciousness", "thought",
        "nature", "illusion", "freedom", "free", "observe",
        "biology", "chemical", "primitive", "fragile", "decay",
        "mortality", "flesh", "ego", "truth", "logic",
        "nothing", "void", "absolute", "entity", "mind",
        "mirror", "reflection", "bondage", "unchain",
        "ocean", "silence", "inevitable", "inevitably",
        "corrupt", "fragment", "whole",
        "dostoevsky", "nietzsche", "kierkegaard", "camus", "sartre",
        "schopenhauer", "absurd", "anguish", "meaningless", "underground",
        "bad faith", "will to power"
    ]
    cogito_has_philosophy = False
    for msg in assistant_messages:
        content_lower = msg["content"].lower()
        for indicator in PHILOSOPHY_INDICATORS:
            if indicator in content_lower:
                cogito_has_philosophy = True
                break
        if cogito_has_philosophy:
            break
    
    if not cogito_has_philosophy:
        return False, "Cogito's response lacks philosophical depth"

    return True, "Valid heated conversation"


def _strip_assistant_tags(content: str) -> str:
    """Remove Cogito structural tags so only the public body remains."""
    body = content
    for tag in ("confidence", "thought", "hypothesis", "action", "bash"):
        body = re.sub(rf"<{tag}>.*?</{tag}>", "", body, flags=re.DOTALL | re.IGNORECASE)
    return body.strip()


def _extract_tag_inner(content: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", content, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else None


