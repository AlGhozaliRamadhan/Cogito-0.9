"""Generic chat-record validation.

Everything here is model- and dataset-agnostic: it only understands the
OpenAI-style ``[{"role": ..., "content": ...}, ...]`` shape that every SFT
dataset is normalized into before training.

This replaces the old Cogito structural validator, which required a fixed
system prompt, ``<confidence>/<thought>/<action>`` tags, 3-10 turns and an
anti-sycophancy word list. Those rules made every third-party dataset fail.
"""

import json

VALID_ROLES = ("system", "user", "assistant", "tool")

_ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "assistant": "assistant",
    "system": "system",
    "tool": "tool",
    "function": "tool",
    "observation": "tool",
}


class ValidationResult(tuple):
    """(is_valid, reason) that also evaluates as a bool equal to is_valid."""

    def __new__(cls, valid: bool, reason: str | None = None):
        return super().__new__(cls, (valid, reason))

    def __bool__(self):
        return self[0]


def flatten_content(content) -> str:
    """Coerce a message body to plain text.

    Handles the three shapes found in the wild: a string, a list of
    ``{"type": "text", "text": ...}`` parts (multimodal datasets), and
    ``None``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        text = content.get("text") or content.get("content")
        return text if isinstance(text, str) else ""
    return str(content)


def canonical_role(role) -> str | None:
    if not isinstance(role, str):
        return None
    return _ROLE_ALIASES.get(role.strip().lower())


def normalize_tool_calls(raw) -> list[dict] | None:
    """Convert OpenAI-style ``tool_calls`` into what chat templates expect.

    Datasets store ``arguments`` as a JSON string; the Qwen template iterates
    ``tool_call.arguments | items``, which raises on a string. Parsing it here
    keeps the rendered transcript coherent (`<tool_call>` -> `<tool_response>`).
    """
    if not isinstance(raw, (list, tuple)):
        return None
    calls = []
    for call in raw:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else {}
        arguments = function.get("arguments", call.get("arguments"))
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"value": arguments}
        if not isinstance(arguments, dict):
            arguments = {"value": str(arguments)}
        calls.append({
            "id": call.get("id") or f"call_{len(calls)}",
            "type": "function",
            "function": {
                "name": function.get("name") or call.get("name") or "unknown",
                "arguments": arguments,
            },
        })
    return calls or None


def parse_messages(example) -> list[dict]:
    """Pull a message list out of a dataset record.

    Accepts the record itself (``{"messages": ...}``), a bare list, a JSON
    string, and the OpenThoughts ``{"system": ..., "conversations": ...}``
    shape. Raises ValueError when nothing usable is found.
    """
    if example is None:
        raise ValueError("record is empty")

    raw = example
    if isinstance(example, dict):
        if "messages" in example:
            raw = example["messages"]
            if isinstance(raw, str):
                raw = json.loads(raw)
        elif "conversations" in example:
            raw = _from_conversations(example)
        else:
            raise ValueError("record has no 'messages' or 'conversations' field")
    elif isinstance(example, str):
        raw = json.loads(example)

    if not isinstance(raw, list):
        raise ValueError(f"messages must be a list, got {type(raw).__name__}")

    messages = []
    for item in raw:
        if isinstance(item, str):
            item = json.loads(item)
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        role = canonical_role(item.get("role") or item.get("from"))
        if role is None:
            raise ValueError(f"unknown message role: {item.get('role') or item.get('from')!r}")
        content = flatten_content(item.get("content", item.get("value")))
        message = {"role": role, "content": content}
        reasoning = item.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            message["reasoning_content"] = reasoning
        tool_calls = normalize_tool_calls(item.get("tool_calls"))
        if tool_calls and role == "assistant":
            message["tool_calls"] = tool_calls
        messages.append(message)
    return messages


def _from_conversations(example: dict) -> list[dict]:
    turns = example.get("conversations") or example.get("conversation")
    if not isinstance(turns, list):
        raise ValueError("'conversations' must be a list")
    messages = []
    system = example.get("system")
    if isinstance(system, str) and system.strip():
        messages.append({"role": "system", "content": system})
    for turn in turns:
        if not isinstance(turn, dict):
            continue
        role = canonical_role(turn.get("from") or turn.get("role"))
        content = flatten_content(turn.get("value", turn.get("content")))
        if role and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def validate_chat(messages: list[dict]) -> ValidationResult:
    """Check that a normalized conversation is trainable."""
    if not isinstance(messages, list):
        return ValidationResult(False, "messages must be a list")
    if len(messages) < 2:
        return ValidationResult(False, f"needs at least 2 turns, got {len(messages)}")

    assistant_turns = 0
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            return ValidationResult(False, f"turn {index} is not an object")
        if message.get("role") not in VALID_ROLES:
            return ValidationResult(False, f"turn {index} has invalid role {message.get('role')!r}")
        content = message.get("content")
        if not isinstance(content, str):
            return ValidationResult(False, f"turn {index} ({message.get('role')}) has non-text content")
        if not content.strip() and not message.get("tool_calls"):
            return ValidationResult(False, f"turn {index} ({message.get('role')}) has empty content")
        if message["role"] == "assistant":
            assistant_turns += 1

    if assistant_turns == 0:
        return ValidationResult(False, "no assistant turn to train on")
    if messages[-1].get("role") != "assistant":
        return ValidationResult(False, "conversation must end on an assistant turn")
    return ValidationResult(True, None)


def sanitize(messages: list[dict]) -> list[dict]:
    """Drop leading/trailing turns that cannot be trained on.

    A dataset sample that opens on an assistant turn (or ends on a user turn)
    still contains usable signal; trimming the ragged edges keeps it instead
    of throwing the record away.
    """
    start, end = 0, len(messages)
    while start < end and messages[start].get("role") != "user":
        if messages[start].get("role") == "system":
            break
        start += 1
    while end > start and messages[end - 1].get("role") != "assistant":
        end -= 1
    return messages[start:end]
