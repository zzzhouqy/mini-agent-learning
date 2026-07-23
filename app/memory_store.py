import sqlite3
from pathlib import Path

from app.memory import MemoryCreate, MemoryRecord


def initialize_memory_database(
    database_path: str | Path,
) -> None:
    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL
                        CHECK (
                            memory_type IN (
                                'preference',
                                'fact',
                                'decision'
                            )
                        ),
                    content TEXT NOT NULL,
                    source TEXT NOT NULL
                        CHECK (
                            source IN (
                                'user_explicit',
                                'model_inferred',
                                'manual'
                            )
                        ),
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        connection.close()


def add_memory(
    database_path: str | Path,
    memory: MemoryCreate,
) -> MemoryRecord:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory.user_id,
                    memory.source_session_id,
                    memory.memory_type,
                    memory.content,
                    memory.source,
                ),
            )

            row = connection.execute(
                """
                SELECT
                    memory_id,
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source,
                    created_at,
                    updated_at
                FROM memories
                WHERE memory_id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("Memory 写入后无法读取。")

    return MemoryRecord.model_validate(dict(row))


def get_user_memories(
    database_path: str | Path,
    user_id: str,
) -> list[MemoryRecord]:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT
                memory_id,
                user_id,
                source_session_id,
                memory_type,
                content,
                source,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
            ORDER BY memory_id
            """,
            (user_id,),
        ).fetchall()
    finally:
        connection.close()

    return [
        MemoryRecord.model_validate(dict(row))
        for row in rows
    ]
