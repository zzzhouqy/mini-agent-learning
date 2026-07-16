import sqlite3

import pytest

from app.session_store import add_exchange, add_message, get_recent_messages


def test_add_message_persists_message(tmp_path):
    database_path = tmp_path / "sessions.db"

    add_message(
        database_path,
        "user_001",
        "session_A",
        "user",
        "我喜欢用表格总结",
    )

    assert database_path.exists()
    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
    ) == [
        {
            "role": "user",
            "content": "我喜欢用表格总结",
        }
    ]


def test_get_recent_messages_isolates_user_and_session(tmp_path):
    database_path = tmp_path / "sessions.db"
    add_message(database_path, "user_001", "session_A", "user", "A")
    add_message(database_path, "user_001", "session_A", "assistant", "B")
    add_message(database_path, "user_001", "session_A", "user", "C")
    add_message(database_path, "user_001", "session_B", "user", "X")
    add_message(database_path, "user_002", "session_A", "user", "Y")

    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=2,
    ) == [
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
    ]


def test_get_recent_messages_rejects_invalid_window(tmp_path):
    database_path = tmp_path / "sessions.db"

    with pytest.raises(ValueError, match="max_messages 必须大于或等于 1"):
        get_recent_messages(
            database_path,
            "user_001",
            "session_A",
            max_messages=0,
        )


def test_add_exchange_rolls_back_when_assistant_message_is_invalid(tmp_path):
    database_path = tmp_path / "sessions.db"

    with pytest.raises(sqlite3.IntegrityError):
        add_exchange(
            database_path,
            "user_001",
            "session_A",
            "D",
            None,
        )

    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=10,
    ) == []
