from pathlib import Path

from app.agent import agent_loop
from app.config import LLMConfig
from app.session_context import build_session_context
from app.session_store import add_exchange


def run_session_agent(
    query: str,
    config: LLMConfig,
    database_path: str | Path,
    user_id: str,
    session_id: str,
    system_prompt: str,
    max_messages: int = 3,
) -> str | None:
    messages = build_session_context(
        database_path,
        user_id,
        session_id,
        system_prompt,
        query,
        max_messages=max_messages,
    )

    answer = agent_loop(
        query,
        config,
        messages=messages,
    )

    if answer is None:
        return None

    add_exchange(
        database_path,
        user_id,
        session_id,
        query,
        answer,
    )

    return answer