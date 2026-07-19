import pytest
from pydantic import ValidationError

from app.memory import MemoryCreate
from app.memory_store import add_memory, get_user_memories


def test_add_memory_returns_database_generated_fields(tmp_path):
    database_path = tmp_path / "memories.db"
    memory = MemoryCreate(
        user_id="user_001",
        source_session_id="session_A",
        memory_type="preference",
        content="用户喜欢用表格总结学习内容。",
        source="user_explicit",
    )

    record = add_memory(database_path, memory)

    assert record.memory_id == 1
    assert record.user_id == "user_001"
    assert record.source_session_id == "session_A"
    assert record.created_at
    assert record.updated_at


def test_get_user_memories_crosses_sessions_and_isolates_users(tmp_path):
    database_path = tmp_path / "memories.db"
    add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="preference",
            content="用户喜欢用表格总结。",
            source="user_explicit",
        ),
    )
    add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_B",
            memory_type="fact",
            content="用户正在学习 Agent Memory。",
            source="manual",
        ),
    )
    add_memory(
        database_path,
        MemoryCreate(
            user_id="user_002",
            source_session_id="session_A",
            memory_type="preference",
            content="用户喜欢项目符号。",
            source="user_explicit",
        ),
    )

    user_001_memories = get_user_memories(database_path, "user_001")
    user_002_memories = get_user_memories(database_path, "user_002")

    assert [memory.source_session_id for memory in user_001_memories] == [
        "session_A",
        "session_B",
    ]
    assert [memory.content for memory in user_002_memories] == [
        "用户喜欢项目符号。",
    ]


def test_memory_create_rejects_unknown_type():
    with pytest.raises(ValidationError, match="memory_type"):
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="chat",
            content="临时聊天",
            source="user_explicit",
        )
