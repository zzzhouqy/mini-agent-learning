import sqlite3

import pytest
from pydantic import ValidationError

from app.memory import MemoryCreate
from app.memory_store import (
    add_memory,
    get_user_memories,
    initialize_memory_database,
    replace_memory,
    supersede_memory,
)


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
    assert record.status == "active"
    assert record.superseded_by is None
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


def test_initialize_memory_database_migrates_legacy_table(tmp_path):
    database_path = tmp_path / "legacy_memories.db"
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        CREATE TABLE memories (
            memory_id INTEGER PRIMARY KEY,
            user_id TEXT NOT NULL,
            source_session_id TEXT NOT NULL,
            memory_type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        INSERT INTO memories (
            user_id,
            source_session_id,
            memory_type,
            content,
            source,
            created_at,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "user_001",
            "session_A",
            "decision",
            "项目继续使用 SQLite。",
            "manual",
            "2026-07-27 10:00:00",
            "2026-07-27 10:00:00",
        ),
    )
    connection.commit()
    connection.close()

    initialize_memory_database(database_path)
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    column_names = [
        row[1]
        for row in connection.execute("PRAGMA table_info(memories)")
    ]
    row = connection.execute(
        "SELECT status, superseded_by FROM memories WHERE memory_id = 1"
    ).fetchone()
    connection.close()
    memories = get_user_memories(database_path, "user_001")

    assert "status" in column_names
    assert "superseded_by" in column_names
    assert row == ("active", None)
    assert [(memory.status, memory.superseded_by) for memory in memories] == [
        ("active", None),
    ]


def test_get_user_memories_filters_superseded_by_default(tmp_path):
    database_path = tmp_path / "memories.db"
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 SQLite。",
            source="manual",
        ),
    )
    new_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_B",
            memory_type="decision",
            content="项目使用 PostgreSQL。",
            source="manual",
        ),
    )
    connection = sqlite3.connect(database_path)
    connection.execute(
        """
        UPDATE memories
        SET status = ?, superseded_by = ?
        WHERE memory_id = ?
        """,
        ("superseded", new_memory.memory_id, old_memory.memory_id),
    )
    connection.commit()
    connection.close()

    active_memories = get_user_memories(database_path, "user_001")
    all_memories = get_user_memories(
        database_path,
        "user_001",
        include_superseded=True,
    )

    assert [memory.content for memory in active_memories] == [
        "项目使用 PostgreSQL。",
    ]
    assert [
        (memory.status, memory.superseded_by)
        for memory in all_memories
    ] == [
        ("superseded", new_memory.memory_id),
        ("active", None),
    ]


def test_supersede_memory_marks_old_record_and_preserves_audit(tmp_path):
    database_path = tmp_path / "memories.db"
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 SQLite。",
            source="manual",
        ),
    )
    new_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_B",
            memory_type="decision",
            content="项目使用 PostgreSQL。",
            source="manual",
        ),
    )

    updated = supersede_memory(
        database_path,
        "user_001",
        old_memory.memory_id,
        new_memory.memory_id,
    )

    assert updated.status == "superseded"
    assert updated.superseded_by == new_memory.memory_id
    assert [memory.memory_id for memory in get_user_memories(
        database_path,
        "user_001",
    )] == [new_memory.memory_id]


def test_supersede_memory_rejects_cross_user_and_inactive_old_record(
    tmp_path,
):
    database_path = tmp_path / "memories.db"
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 SQLite。",
            source="manual",
        ),
    )
    other_user_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_002",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 PostgreSQL。",
            source="manual",
        ),
    )
    replacement = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_B",
            memory_type="decision",
            content="项目使用 PostgreSQL。",
            source="manual",
        ),
    )

    with pytest.raises(ValueError, match="不属于该用户"):
        supersede_memory(
            database_path,
            "user_001",
            old_memory.memory_id,
            other_user_memory.memory_id,
        )

    supersede_memory(
        database_path,
        "user_001",
        old_memory.memory_id,
        replacement.memory_id,
    )

    with pytest.raises(ValueError, match="不是 active"):
        supersede_memory(
            database_path,
            "user_001",
            old_memory.memory_id,
            replacement.memory_id,
        )


def test_replace_memory_creates_new_record_and_supersedes_old_one(tmp_path):
    database_path = tmp_path / "memories.db"
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 SQLite。",
            source="manual",
        ),
    )

    updated_old, new_memory = replace_memory(
        database_path,
        "user_001",
        old_memory.memory_id,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_B",
            memory_type="decision",
            content="项目使用 PostgreSQL。",
            source="user_explicit",
        ),
    )

    assert updated_old.status == "superseded"
    assert updated_old.superseded_by == new_memory.memory_id
    assert new_memory.status == "active"
    assert new_memory.source == "user_explicit"
    assert [memory.memory_id for memory in get_user_memories(
        database_path,
        "user_001",
    )] == [new_memory.memory_id]


def test_replace_memory_rejects_untrusted_identity_and_type_before_insert(
    tmp_path,
):
    database_path = tmp_path / "memories.db"
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="session_A",
            memory_type="decision",
            content="项目使用 SQLite。",
            source="manual",
        ),
    )

    with pytest.raises(ValueError, match="user_id"):
        replace_memory(
            database_path,
            "user_001",
            old_memory.memory_id,
            MemoryCreate(
                user_id="user_002",
                source_session_id="session_B",
                memory_type="decision",
                content="项目使用 PostgreSQL。",
                source="manual",
            ),
        )

    with pytest.raises(ValueError, match="memory_type"):
        replace_memory(
            database_path,
            "user_001",
            old_memory.memory_id,
            MemoryCreate(
                user_id="user_001",
                source_session_id="session_B",
                memory_type="preference",
                content="用户喜欢用表格总结。",
                source="manual",
            ),
        )

    all_memories = get_user_memories(
        database_path,
        "user_001",
        include_superseded=True,
    )

    assert [(memory.memory_id, memory.status) for memory in all_memories] == [
        (old_memory.memory_id, "active"),
    ]
