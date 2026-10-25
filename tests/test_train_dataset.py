"""Dataset-loading helpers: uniform Arrow schema despite tool-payload drift."""

import json

import pytest

from qwen_sft.finetune.train import (
    load_jsonl_shard,
    parse_tool_arguments,
    stringify_tool_arguments,
)

# The real poison: prepared tool calls keep `function.arguments` as an object,
# and its inner `content` key is a string in some records and a nested object
# in others ("Column(.../arguments/content) changed from object to string in
# row 0"). Both shapes plus a record without any tool_calls must survive.


def convo(payload):
    return {"messages": [
        {"role": "user", "content": "run it"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function",
             "function": {"name": "computer", "arguments": payload}}],
        },
        {"role": "tool", "content": "ok"},
        {"role": "assistant", "content": "done"},
    ]}


def no_tool_convo():
    return {"messages": [{"role": "user", "content": "plain"},
                         {"role": "assistant", "content": "fine"}]}


def test_stringify_then_parse_round_trips_object_payloads():
    messages = convo({"content": {"code": "print(1)"}, "timeout": 30})["messages"]
    stringify_tool_arguments(messages)
    argument = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(argument, str)
    parse_tool_arguments(messages)
    restored = messages[1]["tool_calls"][0]["function"]["arguments"]
    assert restored == {"content": {"code": "print(1)"}, "timeout": 30}


def test_parse_restores_openai_style_json_strings():
    messages = convo(json.dumps({"value": "ls -la"}))["messages"]
    parse_tool_arguments(messages)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"value": "ls -la"}


def test_parse_coerces_non_json_payloads():
    messages = convo("just a plain string")["messages"]
    parse_tool_arguments(messages)
    assert messages[1]["tool_calls"][0]["function"]["arguments"] == {"value": "just a plain string"}


def test_helpers_ignores_calls_without_function_or_arguments():
    messages = [{"role": "assistant", "content": "", "tool_calls": [
        {"id": "x", "type": "function"}]}]
    stringify_tool_arguments(messages)
    parse_tool_arguments(messages)
    assert "function" not in messages[0]["tool_calls"][0]


def write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record if isinstance(record, str)
                         else json.dumps(record, ensure_ascii=False))
            handle.write("\n")


@pytest.mark.parametrize("pip", ["datasets"])
def test_load_jsonl_shard(tmp_path, pip):
    pytest.importorskip(pip)
    path = tmp_path / "drifted.jsonl"
    write_jsonl(path, [
        no_tool_convo(),                                   # first record: no tool_calls
        convo({"content": {"description": "poison"}}),     # object payload
        convo({"content": "str payload", "path": "/tmp"}),  # string payload, same key
        "{ malformed",                                     # stray fragment line
        json.dumps({"turns": "missing messages key"}),
        "[1, 2, 3]",                                       # scalar record
    ])
    dataset = load_jsonl_shard(str(path))
    assert len(dataset) == 3
    # Uniform column: every payload surfaces as a JSON string, dict inside.
    tool_calls = 0
    for row in dataset:
        messages = [dict(m) for m in row["messages"]]
        parse_tool_arguments(messages)
        for message in messages:
            for call in message.get("tool_calls") or []:
                assert isinstance(call["function"]["arguments"], dict)
                tool_calls += 1
    assert tool_calls == 2, "both tool-call records must survive with objects"


def test_load_jsonl_shard_requires_a_readable_file(tmp_path):
    pytest.importorskip("datasets")
    path = tmp_path / "empty.jsonl"
    write_jsonl(path, ["{ nope"])
    with pytest.raises(SystemExit):
        load_jsonl_shard(str(path))
