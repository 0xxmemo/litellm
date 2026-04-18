"""Tests for ChatGPT wire-format name sanitization."""

from litellm.llms.chatgpt.chat.transformation import (
    _sanitize_chatgpt_name,
    _sanitize_message_names,
    _sanitize_tools_names,
)


def test_sanitize_chatgpt_name_dots_to_underscores() -> None:
    assert _sanitize_chatgpt_name("tools.write_file") == "tools_write_file"


def test_sanitize_chatgpt_name_already_valid_unchanged() -> None:
    assert _sanitize_chatgpt_name("already-ok_1") == "already-ok_1"


def test_sanitize_chatgpt_name_empty_string_unchanged() -> None:
    assert _sanitize_chatgpt_name("") == ""


def test_sanitize_chatgpt_name_none_passthrough() -> None:
    assert _sanitize_chatgpt_name(None) is None


def test_sanitize_chatgpt_name_all_invalid_falls_back_to_tool() -> None:
    assert _sanitize_chatgpt_name("!!!") == "tool"


def test_sanitize_tools_names_function_and_custom() -> None:
    tools = [
        {"type": "function", "function": {"name": "a.b", "description": "d"}},
        {"type": "custom", "custom": {"name": "x:y", "meta": 1}},
    ]
    out = _sanitize_tools_names(tools)
    assert out[0]["function"]["name"] == "a_b"
    assert out[0]["function"]["description"] == "d"
    assert out[1]["custom"]["name"] == "x_y"
    assert out[1]["custom"]["meta"] == 1


def test_sanitize_message_names_msg_and_tool_calls() -> None:
    messages = [
        {
            "role": "assistant",
            "name": "bad.name",
            "tool_calls": [
                {
                    "id": "1",
                    "type": "function",
                    "function": {"name": "fn.bad", "arguments": "{}"},
                }
            ],
        }
    ]
    out = _sanitize_message_names(messages)
    assert out[0]["name"] == "bad_name"
    assert out[0]["tool_calls"][0]["function"]["name"] == "fn_bad"
