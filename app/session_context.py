from pathlib import Path

from app.context_manager import build_context_messages
from app.session_store import get_recent_messages


def build_session_context(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    system_prompt: str,
    current_query: str,
    max_messages: int = 3,
) -> list[dict]:
    history = get_recent_messages(
        database_path,
        user_id,
        session_id,
        max_messages=max_messages,
    )

    return build_context_messages(
        system_prompt,
        history,
        current_query,
        max_messages=max_messages,
    )
