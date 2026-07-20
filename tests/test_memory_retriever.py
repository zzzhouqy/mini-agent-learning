import pytest

import app.memory_retriever as memory_retriever
from app.memory import MemoryCreate, MemoryMatch, MemoryRecord
from app.memory_store import add_memory


def add_test_memory(database_path, user_id: str, content: str) -> None:
    add_memory(
        database_path,
        MemoryCreate(
            user_id=user_id,
            source_session_id="session_A",
            memory_type="fact",
            content=content,
            source="manual",
        ),
    )


def test_search_user_memories_isolates_user_and_ranks_matches(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "memories.db"
    add_test_memory(database_path, "user_001", "A")
    add_test_memory(database_path, "user_001", "B")
    add_test_memory(database_path, "user_001", "C")
    add_test_memory(database_path, "user_002", "other user")
    captured_texts = []

    def fake_semantic_scores(query: str, texts: list[str]) -> list[float]:
        assert query == "测试问题"
        captured_texts.extend(texts)
        return [0.2, 0.9, 0.7]

    monkeypatch.setattr(
        memory_retriever,
        "semantic_scores",
        fake_semantic_scores,
    )

    results = memory_retriever.search_user_memories(
        database_path,
        "user_001",
        "测试问题",
        min_score=0.5,
        top_k=2,
    )

    assert captured_texts == ["A", "B", "C"]
    assert [match.memory.content for match in results] == ["B", "C"]
    assert [match.score for match in results] == [0.9, 0.7]


def test_search_user_memories_skips_embedding_for_empty_user(
    tmp_path,
    monkeypatch,
):
    def fail_if_called(query: str, texts: list[str]) -> list[float]:
        raise AssertionError("不应调用 Embedding")

    monkeypatch.setattr(
        memory_retriever,
        "semantic_scores",
        fail_if_called,
    )

    results = memory_retriever.search_user_memories(
        tmp_path / "memories.db",
        "not_exists",
        "测试问题",
    )

    assert results == []


def test_search_user_memories_filters_all_low_scores(tmp_path, monkeypatch):
    database_path = tmp_path / "memories.db"
    add_test_memory(database_path, "user_001", "低相关记忆")
    monkeypatch.setattr(
        memory_retriever,
        "semantic_scores",
        lambda query, texts: [0.1],
    )

    results = memory_retriever.search_user_memories(
        database_path,
        "user_001",
        "测试问题",
        min_score=0.5,
    )

    assert results == []


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"min_score": 1.1}, "min_score"),
        ({"top_k": 0}, "top_k"),
    ],
)
def test_search_user_memories_rejects_invalid_arguments(arguments, message):
    with pytest.raises(ValueError, match=message):
        memory_retriever.search_user_memories(
            "/tmp/unused.db",
            "user_001",
            "测试问题",
            **arguments,
        )


def test_format_memory_matches_builds_bounded_context():
    memory = MemoryRecord(
        memory_id=1,
        user_id="user_001",
        source_session_id="session_A",
        memory_type="preference",
        content="用户喜欢用表格总结。",
        source="user_explicit",
        created_at="2026-07-20 10:00:00",
        updated_at="2026-07-20 10:00:00",
    )

    result = memory_retriever.format_memory_matches(
        [MemoryMatch(memory=memory, score=0.82)],
    )

    assert result.startswith("<relevant_memories>\n")
    assert "1. [preference] 用户喜欢用表格总结。" in result
    assert result.endswith("\n</relevant_memories>")
    assert memory_retriever.format_memory_matches([]) == ""
