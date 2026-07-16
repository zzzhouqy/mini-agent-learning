from app.session_context import build_session_context
from app.session_store import add_message


def test_build_session_context_uses_recent_history_and_current_query(tmp_path):
    database_path = tmp_path / "sessions.db"
    add_message(database_path, "user_001", "session_A", "user", "A")
    add_message(database_path, "user_001", "session_A", "assistant", "B")
    add_message(database_path, "user_001", "session_A", "user", "C")

    result = build_session_context(
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
        "D",
        max_messages=2,
    )

    assert result == [
        {"role": "system", "content": "你是学习助手"},
        {"role": "assistant", "content": "B"},
        {"role": "user", "content": "C"},
        {"role": "user", "content": "D"},
    ]
