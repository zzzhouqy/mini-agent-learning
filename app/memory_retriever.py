from pathlib import Path

from app.embeddings import semantic_scores
from app.memory import MemoryMatch
from app.memory_store import get_user_memories


def search_user_memories(
    database_path: str | Path,
    user_id: str,
    query: str,
    min_score: float = 0.3,
    top_k: int = 3,
) -> list[MemoryMatch]:
    if not -1.0 <= min_score <= 1.0:
        raise ValueError("min_score 必须在 -1.0 到 1.0 之间。")

    if top_k < 1:
        raise ValueError("top_k 必须大于或等于 1。")

    memories = get_user_memories(database_path, user_id)

    if not memories:
        return []

    scores = semantic_scores(
        query,
        [memory.content for memory in memories],
    )

    matches = [
        MemoryMatch(memory=memory, score=score)
        for memory, score in zip(memories, scores, strict=True)
        if score >= min_score
    ]
    matches.sort(
        key=lambda match: match.score,
        reverse=True,
    )

    return matches[:top_k]
