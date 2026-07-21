import pytest
from pydantic import ValidationError

from app.memory import MemoryCandidate
from app.memory_store import get_user_memories
from app.memory_writer import (
    build_explicit_memory_candidate,
    build_memory_create,
    contains_sensitive_memory_content,
    extract_explicit_memory_text,
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
