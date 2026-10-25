"""Normalization + validation tests, built from the real record shapes."""

import json

import pytest

from qwen_sft.validation import (
    flatten_content,
    parse_messages,
    sanitize,
    validate_chat,
)


def test_astra_messages_shape():
    record = {
        "messages": [
            {"role": "system", "content": "Be accurate."},
            {"role": "user", "content": "Is nuclear safe?"},
            {"role": "assistant", "content": "Mostly, yes."},
        ]
    }
    messages = parse_messages(record)
    assert [m["role"] for m in messages] == ["system", "user", "assistant"]
    assert validate_chat(messages)


def test_fable_messages_column_can_be_json_string():
    record = {"messages": json.dumps([{"role": "user", "content": "hi"},
                                      {"role": "assistant", "content": "hello"}])}
    messages = parse_messages(record)
    assert messages[1]["content"] == "hello"
    assert validate_chat(messages)


def test_multimodal_content_parts_are_flattened():
    record = {"messages": [
        {"role": "user", "content": [{"type": "text", "text": "describe"}, {"type": "image"}]},
        {"role": "assistant", "content": "a cat"},
    ]}
    assert parse_messages(record)[0]["content"] == "describe"
    assert flatten_content(None) == ""
    assert flatten_content({"text": "x"}) == "x"


def test_openthoughts_conversations_shape():
    record = {
        "system": "Think step by step.",
        "conversations": [
            {"from": "user", "value": "2+2?"},
            {"from": "assistant", "value": "4"},
        ],
    }
    messages = parse_messages(record)
    assert messages == [
        {"role": "system", "content": "Think step by step."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    assert validate_chat(messages)


def test_role_aliases_and_empty_conversation_turns():
    record = {"messages": [
        {"from": "human", "value": "q"},
        {"from": "gpt", "value": "a"},
    ]}
    assert [m["role"] for m in parse_messages(record)] == ["user", "assistant"]


def test_tool_calls_are_preserved_and_arguments_parsed():
    record = {"messages": [
        {"role": "user", "content": "list the repo"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "call_0", "type": "function",
             "function": {"name": "list_dir", "arguments": json.dumps({"path": "."})}},
        ]},
        {"role": "tool", "content": "README.md"},
        {"role": "assistant", "content": "One file."},
    ]}
    messages = parse_messages(record)
    call = messages[1]["tool_calls"][0]
    assert call["function"]["name"] == "list_dir"
    assert call["function"]["arguments"] == {"path": "."}
    assert validate_chat(messages)


def test_tool_call_with_non_json_arguments_is_wrapped():
    record = {"messages": [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "bash", "arguments": "ls -la"}},
        ]},
        {"role": "tool", "content": "output"},
        {"role": "assistant", "content": "done"},
    ]}
    assert parse_messages(record)[1]["tool_calls"][0]["function"]["arguments"] == {"value": "ls -la"}


def test_unknown_role_raises():
    with pytest.raises(ValueError):
        parse_messages({"messages": [{"role": "wizard", "content": "x"}]})


def test_validate_rejects_untrainable_records():
    assert not validate_chat([{"role": "user", "content": "only a question"}])
    assert not validate_chat([{"role": "user", "content": "q"},
                              {"role": "assistant", "content": "  "}])
    assert not validate_chat([{"role": "user", "content": "q"},
                              {"role": "user", "content": "q2"}])
    ends_on_user = [{"role": "user", "content": "q"},
                    {"role": "assistant", "content": "a"},
                    {"role": "user", "content": "q2"}]
    assert not validate_chat(ends_on_user)


def test_sanitize_repairs_ragged_edges():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "assistant", "content": "dangling"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "trailing question"},
    ]
    cleaned = sanitize(messages)[1:]
    assert cleaned[-1]["role"] == "assistant"
    assert validate_chat(sanitize(messages)[1:])
