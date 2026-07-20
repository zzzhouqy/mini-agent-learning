import pytest

from app.context_manager import build_context_messages, select_recent_messages


def test_select_recent_messages_returns_requested_window_without_mutation():
    messages = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
    ]

    result = select_recent_messages(messages, max_messages=2)

    assert [message["content"] for message in result] == ["C", "D"]
    assert len(messages) == 4


def test_select_recent_messages_rejects_invalid_window():
    with pytest.raises(ValueError, match="max_messages 必须大于或等于 1"):
        select_recent_messages([], max_messages=0)


def test_build_context_messages_preserves_message_order():
    history = [
        {"role": "user", "content": "A"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
    ]

    result = build_context_messages(
        "你是学习助手",
        history,
        "E",
        max_messages=2,
    )

    assert result == [
        {"role": "system", "content": "你是学习助手"},
        {"role": "user", "content": "C"},
        {"role": "assistant", "content": "D"},
        {"role": "user", "content": "E"},
    ]
    assert len(history) == 4


def test_build_context_messages_injects_memory_into_system_message():
    memory_context = (
        "<relevant_memories>\n"
        "1. [preference] 用户喜欢用表格总结。\n"
        "</relevant_memories>"
    )

    result = build_context_messages(
        "你是学习助手",
        [],
        "我喜欢怎样总结？",
        memory_context=memory_context,
    )

    assert result[0] == {
        "role": "system",
        "content": f"你是学习助手\n\n{memory_context}",
    }
    assert result[-1] == {
        "role": "user",
        "content": "我喜欢怎样总结？",
    }
