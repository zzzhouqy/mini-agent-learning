from pathlib import Path
from typing import Literal

from app.memory import MemoryCandidate, MemoryCreate, MemoryRecord
from app.memory_store import add_memory


EXPLICIT_MEMORY_PREFIXES = (
    "请记住：",
    "请记住:",
)

SENSITIVE_MEMORY_MARKERS = (
    "api key",
    "api_key",
    "password",
    "密码",
    "密钥",
    "access token",
    "refresh token",
    "secret key",
    "sk-",
)


def extract_explicit_memory_text(user_message: str) -> str | None:
    normalized_message = user_message.strip()

    for prefix in EXPLICIT_MEMORY_PREFIXES:
        if normalized_message.startswith(prefix):
            content = normalized_message[len(prefix):].strip()

            return content or None

    return None


def contains_sensitive_memory_content(content: str) -> bool:
    normalized_content = content.casefold()

    return any(
        marker in normalized_content
        for marker in SENSITIVE_MEMORY_MARKERS
    )


def build_explicit_memory_candidate(
    user_message: str,
    memory_type: Literal[
        "preference",
        "fact",
        "decision",
    ],
) -> MemoryCandidate | None:
    content = extract_explicit_memory_text(user_message)

    if content is None:
        return None

    if contains_sensitive_memory_content(content):
        return None

    return MemoryCandidate(
        memory_type=memory_type,
        content=content,
        source="user_explicit",
    )


def build_memory_create(
    candidate: MemoryCandidate,
    user_id: str,
    source_session_id: str,
) -> MemoryCreate:
    return MemoryCreate(
        user_id=user_id,
        source_session_id=source_session_id,
        memory_type=candidate.memory_type,
        content=candidate.content,
        source=candidate.source,
    )


def save_explicit_memory(
    database_path: str | Path,
    user_id: str,
    source_session_id: str,
    user_message: str,
    memory_type: Literal[
        "preference",
        "fact",
        "decision",
    ],
) -> MemoryRecord | None:
    candidate = build_explicit_memory_candidate(
        user_message,
        memory_type,
    )

    if candidate is None:
        return None

    memory = build_memory_create(
        candidate,
        user_id,
        source_session_id,
    )

    return add_memory(database_path, memory)
