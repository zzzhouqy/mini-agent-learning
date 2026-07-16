import sqlite3
from pathlib import Path


def initialize_database(database_path: str | Path) -> None:
    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        connection.close()


def add_message(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    role: str,
    content: str,
) -> None:
    initialize_database(database_path)

    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute(
                """
                INSERT INTO messages (user_id, session_id, role, content)
                VALUES (?, ?, ?, ?)
                """,
                (user_id, session_id, role, content),
            )
    finally:
        connection.close()


def get_recent_messages(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    max_messages: int = 3,
) -> list[dict]:
    if max_messages < 1:
        raise ValueError("max_messages 必须大于或等于 1。")

    initialize_database(database_path)

    connection = sqlite3.connect(database_path)
    try:
        rows = connection.execute(
            """
            SELECT role, content
            FROM messages
            WHERE user_id = ? AND session_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (user_id, session_id, max_messages),
        ).fetchall()
    finally:
        connection.close()

    rows.reverse()

    return [
        {"role": role, "content": content}
        for role, content in rows
    ]
def add_exchange(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    user_content: str,
    assistant_content: str,
) -> None:
    initialize_database(database_path)

    insert_sql = """
        INSERT INTO messages (user_id, session_id, role, content)
        VALUES (?, ?, ?, ?)
    """

    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute(
                insert_sql,
                (
                    user_id,
                    session_id,
                    "user",
                    user_content,
                ),
            )
            connection.execute(
                insert_sql,
                (
                    user_id,
                    session_id,
                    "assistant",
                    assistant_content,
                ),
            )
    finally:
        connection.close()
