import pytest
from pydantic import ValidationError

import app.memory_writer as memory_writer
from app.memory import MemoryCandidate
from app.memory_store import get_user_memories
from app.memory_writer import (
    build_explicit_memory_candidate,
    build_memory_create,
    build_memory_extraction_messages,
    contains_sensitive_memory_content,
    extract_inferred_memory_candidates,
    extract_explicit_memory_text,
    parse_memory_extraction_response,
    save_memory_candidate,
    save_explicit_memory,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("请记住：我喜欢用表格总结。", "我喜欢用表格总结。"),
        ("  请记住: 项目继续使用 SQLite。  ", "项目继续使用 SQLite。"),
        ("Pydantic 有什么作用？", None),
        ("请记住：", None),
    ],
)
def test_extract_explicit_memory_text(message, expected):
    assert extract_explicit_memory_text(message) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("我的 API Key 是 sk-xxxx。", True),
        ("密码是 123456。", True),
        ("用户喜欢用表格总结。", False),
        ("我正在学习 token_budget() 的作用。", False),
    ],
)
def test_contains_sensitive_memory_content(content, expected):
    assert contains_sensitive_memory_content(content) is expected


def test_memory_candidate_rejects_untrusted_user_id():
    with pytest.raises(ValidationError, match="user_id"):
        MemoryCandidate(
            memory_type="preference",
            content="用户喜欢用表格总结。",
            source="model_inferred",
            user_id="user_002",
        )


def test_build_memory_create_binds_trusted_identity():
    candidate = MemoryCandidate(
        memory_type="decision",
        content="项目继续使用 SQLite。",
        source="model_inferred",
    )

    memory = build_memory_create(candidate, "user_001", "session_A")

    assert memory.user_id == "user_001"
    assert memory.source_session_id == "session_A"
    assert memory.content == candidate.content
    assert memory.source == "model_inferred"


def test_build_explicit_memory_candidate_filters_non_memory_and_secret():
    safe = build_explicit_memory_candidate(
        "请记住：我喜欢用表格总结。",
        "preference",
    )
    ordinary = build_explicit_memory_candidate(
        "Pydantic 有什么作用？",
        "fact",
    )
    secret = build_explicit_memory_candidate(
        "请记住：我的 API Key 是 sk-xxxx。",
        "fact",
    )

    assert safe is not None
    assert safe.content == "我喜欢用表格总结。"
    assert safe.source == "user_explicit"
    assert ordinary is None
    assert secret is None


def test_save_explicit_memory_only_persists_safe_candidate(tmp_path):
    database_path = tmp_path / "memories.db"

    saved = save_explicit_memory(
        database_path,
        "user_001",
        "session_A",
        "请记住：我喜欢用表格总结。",
        "preference",
    )
    ordinary = save_explicit_memory(
        database_path,
        "user_001",
        "session_A",
        "Pydantic 有什么作用？",
        "fact",
    )
    secret = save_explicit_memory(
        database_path,
        "user_001",
        "session_A",
        "请记住：我的 API Key 是 sk-xxxx。",
        "fact",
    )

    memories = get_user_memories(database_path, "user_001")

    assert saved is not None
    assert ordinary is None
    assert secret is None
    assert [memory.content for memory in memories] == [
        "我喜欢用表格总结。",
    ]


def test_build_memory_extraction_messages_uses_data_boundaries():
    messages = build_memory_extraction_messages(
        "请记住：我喜欢用表格总结。",
        "好的，我会优先用表格总结。",
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "不要输出 source" in messages[0]["content"]
    assert "<user_message>" in messages[1]["content"]
    assert "<assistant_answer>" in messages[1]["content"]


def test_parse_memory_extraction_response_normalizes_and_filters_secret():
    response_text = """
    {
      "candidates": [
        {
          "memory_type": "preference",
          "content": "用户喜欢用表格总结。"
        },
        {
          "memory_type": "fact",
          "content": "用户的 API Key 是 sk-xxxx。"
        }
      ]
    }
    """

    results = parse_memory_extraction_response(response_text)

    assert [candidate.model_dump() for candidate in results] == [
        {
            "memory_type": "preference",
            "content": "用户喜欢用表格总结。",
            "source": "model_inferred",
        },
    ]


@pytest.mark.parametrize(
    "response_text",
    [
        "not json",
        '{"candidates": [{"memory_type": "chat", "content": "临时聊天"}]}',
        '{"candidates": [{"memory_type": "fact", "content": "测试", "source": "model_inferred"}]}',
    ],
)
def test_parse_memory_extraction_response_rejects_invalid_model_output(
    response_text,
):
    with pytest.raises(ValidationError):
        parse_memory_extraction_response(response_text)


def test_extract_inferred_memory_candidates_calls_model_and_returns_candidate(
    monkeypatch,
):
    captured = {}

    def fake_send_messages(config, messages, tools=None):
        captured["config"] = config
        captured["messages"] = messages
        assert tools is None
        return {
            "role": "assistant",
            "content": (
                '{"candidates": [{"memory_type": "decision", '
                '"content": "项目继续使用 SQLite。"}]}'
            ),
        }

    monkeypatch.setattr(memory_writer, "send_messages", fake_send_messages)

    results = extract_inferred_memory_candidates(
        "我决定项目继续使用 SQLite。",
        "SQLite 足以满足当前需求。",
        "test_config",
    )

    assert captured["config"] == "test_config"
    assert captured["messages"][0]["role"] == "system"
    assert [candidate.source for candidate in results] == ["model_inferred"]
    assert [candidate.memory_type for candidate in results] == ["decision"]


def test_extract_inferred_memory_candidates_rejects_non_text_response(
    monkeypatch,
):
    monkeypatch.setattr(
        memory_writer,
        "send_messages",
        lambda config, messages: {"role": "assistant", "content": None},
    )

    with pytest.raises(ValueError, match="模型未返回文本内容"):
        extract_inferred_memory_candidates(
            "测试问题",
            "测试回答",
            None,
        )


def test_save_memory_candidate_deduplicates_within_one_user(tmp_path):
    database_path = tmp_path / "memories.db"
    original = save_explicit_memory(
        database_path,
        "user_001",
        "session_A",
        "请记住：用户喜欢用表格总结。",
        "preference",
    )
    duplicate = save_memory_candidate(
        database_path,
        "user_001",
        "session_B",
        MemoryCandidate(
            memory_type="preference",
            content="用户喜欢用表格总结",
            source="model_inferred",
        ),
    )

    memories = get_user_memories(database_path, "user_001")

    assert original is not None
    assert duplicate is not None
    assert duplicate.memory_id == original.memory_id
    assert len(memories) == 1
    assert memories[0].source == "user_explicit"


def test_save_memory_candidate_keeps_users_isolated_and_filters_secret(tmp_path):
    database_path = tmp_path / "memories.db"
    first = save_memory_candidate(
        database_path,
        "user_001",
        "session_A",
        MemoryCandidate(
            memory_type="preference",
            content="用户喜欢用表格总结。",
            source="model_inferred",
        ),
    )
    second = save_memory_candidate(
        database_path,
        "user_002",
        "session_A",
        MemoryCandidate(
            memory_type="preference",
            content="用户喜欢用表格总结。",
            source="model_inferred",
        ),
    )
    secret = save_memory_candidate(
        database_path,
        "user_001",
        "session_B",
        MemoryCandidate(
            memory_type="fact",
            content="用户的 API Key 是 sk-xxxx。",
            source="model_inferred",
        ),
    )

    assert first is not None
    assert second is not None
    assert first.memory_id != second.memory_id
    assert secret is None
    assert len(get_user_memories(database_path, "user_001")) == 1
    assert len(get_user_memories(database_path, "user_002")) == 1
