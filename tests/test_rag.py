import pytest

from app.rag import (
    format_contexts,
    score_chunk,
    search_knowledge,
    semantic_search_knowledge,
    reciprocal_rank_score,
    fuse_ranked_results,
    hybrid_search_knowledge,
    answer_with_context,
)


def test_search_knowledge_returns_structured_context():
    results = search_knowledge("Pydantic 结构化校验")

    assert len(results) > 0
    assert results[0]["source"] == "data/knowledge/pydantic.md"
    assert results[0]["title"] == "Pydantic"
    assert "结构化校验" in results[0]["content"]


def test_search_knowledge_returns_empty_list_when_not_found():
    results = search_knowledge("天气")

    assert results == []


def test_format_contexts_returns_message_when_empty():
    assert format_contexts([]) == "没有检索到相关资料。"


def test_score_chunk_counts_keyword_hits():
    score = score_chunk(
        "Pydantic 会抛出 ValidationError",
        ["pydantic", "validationerror", "天气"],
    )

    assert score == 2


def test_search_knowledge_orders_results_by_score():
    results = search_knowledge("Pydantic ValidationError")

    assert results[0]["title"] == "Pydantic"
    assert results[0]["score"] == "2"



def test_search_knowledge_returns_csv_tool_source():
    results = search_knowledge("CSV Tool row_index")

    assert results[0]["source"] == "data/knowledge/csv_tool.md"
    assert results[0]["title"] == "CSV Tool"
    assert results[0]["score"] == "3"
def test_semantic_search_orders_and_filters_results(monkeypatch):
    def fake_semantic_scores(query: str, texts: list[str]) -> list[float]:
        assert query == "如何校验工具参数"

        return [
            0.9 if text.startswith("Pydantic\n")
            else 0.3 if text.startswith("Agent Loop\n")
            else 0.1
            for text in texts
        ]

    monkeypatch.setattr(
        "app.rag.semantic_scores",
        fake_semantic_scores,
    )

    results = semantic_search_knowledge("如何校验工具参数")

    assert [result["title"] for result in results] == [
        "Pydantic",
        "Agent Loop",
    ]


def test_reciprocal_rank_score_decreases_by_rank():
    first_score = reciprocal_rank_score(1)
    second_score = reciprocal_rank_score(2)

    assert first_score == pytest.approx(1 / 61)
    assert first_score > second_score


def test_reciprocal_rank_score_rejects_zero_rank():
    with pytest.raises(
        ValueError,
        match="rank 必须从 1 开始",
    ):
        reciprocal_rank_score(0)


def test_fuse_ranked_results_orders_and_deduplicates():
    def make_context(title: str) -> dict[str, str]:
        return {
            "source": f"data/{title}.md",
            "title": title,
            "content": f"{title} 内容",
            "score": "original",
        }

    keyword_results = [
        make_context("A"),
        make_context("B"),
        make_context("C"),
    ]
    semantic_results = [
        make_context("B"),
        make_context("C"),
    ]

    results = fuse_ranked_results(
        [keyword_results, semantic_results],
        top_k=2,
    )

    assert [result["title"] for result in results] == ["B", "C"]
    assert len(results) == 2
    assert float(results[0]["score"]) == pytest.approx(
        1 / 62 + 1 / 61
    )


def test_hybrid_search_fuses_keyword_and_semantic_results(monkeypatch):
    def make_context(title: str) -> dict[str, str]:
        return {
            "source": f"data/{title}.md",
            "title": title,
            "content": f"{title} 内容",
            "score": "original",
        }

    def fake_keyword_search(query: str, top_k: int):
        assert query == "测试问题"
        assert top_k == 5
        return [make_context("A"), make_context("B")]

    def fake_semantic_search(
        query: str,
        top_k: int,
        min_score: float,
    ):
        assert query == "测试问题"
        assert top_k == 5
        assert min_score == 0.4
        return [make_context("B"), make_context("C")]

    monkeypatch.setattr(
        "app.rag.search_knowledge",
        fake_keyword_search,
    )
    monkeypatch.setattr(
        "app.rag.semantic_search_knowledge",
        fake_semantic_search,
    )

    results = hybrid_search_knowledge(
        "测试问题",
        top_k=2,
        candidate_k=5,
        min_semantic_score=0.4,
    )

    assert [result["title"] for result in results] == ["B", "A"]


def test_answer_with_context_uses_hybrid_search(monkeypatch):
    expected_contexts = [
        {
            "source": "data/test.md",
            "title": "测试资料",
            "content": "测试内容",
            "score": "0.03",
        }
    ]

    def fake_hybrid_search(query: str):
        assert query == "测试问题"
        return expected_contexts

    monkeypatch.setattr(
        "app.rag.hybrid_search_knowledge",
        fake_hybrid_search,
    )

    result = answer_with_context("测试问题")

    assert result == {
        "question": "测试问题",
        "contexts": expected_contexts,
    }
