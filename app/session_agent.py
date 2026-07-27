import logging
from pathlib import Path

from app.agent import agent_loop
from app.config import LLMConfig
from app.memory_retriever import (
    format_memory_matches,
    search_user_memories,
)
from app.memory_writer import (
    extract_inferred_memory_candidates,
    save_memory_candidates,
)
from app.session_context import build_session_context
from app.session_store import add_exchange


LOGGER = logging.getLogger(__name__)


def run_session_agent(
    query: str,
    config: LLMConfig,
    database_path: str | Path,
    user_id: str,
    session_id: str,
    system_prompt: str,
    max_messages: int = 3,
) -> str | None:
    memory_matches = search_user_memories(
        database_path,
        user_id,
        query,
        min_score=0.3,
        top_k=3,
    )
    memory_context = format_memory_matches(memory_matches)

    messages = build_session_context(
        database_path,
        user_id,
        session_id,
        system_prompt,
        query,
        max_messages=max_messages,
        memory_context=memory_context,
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
    try:
        candidates = extract_inferred_memory_candidates(
            query,
            answer,
            config,
        )
        save_memory_candidates(
            database_path,
            user_id,
            session_id,
            candidates,
        )
    except Exception:
        LOGGER.exception("自动长期记忆写入失败。")

    return answer
