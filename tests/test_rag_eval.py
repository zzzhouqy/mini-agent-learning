from app.rag_eval import (
    RAGEvalCase,
    evaluate_retrieval_case,
    evaluate_retriever,
    hit_at_k,
    reciprocal_rank_at_k,
    source_recall_at_k,
    summarize_retrieval_results,
    top1_hit,
)


def test_ranking_metrics_distinguish_position_and_coverage():
    expected_sources = ["data/a.md", "data/b.md"]
    contexts = [
        {"source": "data/c.md"},
        {"source": "data/a.md"},
        {"source": "data/b.md"},
    ]

    assert top1_hit(expected_sources, contexts) is False
    assert hit_at_k(expected_sources, contexts, 2) is True
    assert source_recall_at_k(expected_sources, contexts, 2) == 0.5
    assert reciprocal_rank_at_k(expected_sources, contexts, 3) == 0.5


def test_no_answer_case_requires_empty_contexts():
    case = RAGEvalCase(
        id="no_answer",
        question="未知问题",
        expected_sources=[],
        should_answer=False,
    )

    correct_result = evaluate_retrieval_case(case, [], k=3)
    wrong_result = evaluate_retrieval_case(
        case,
        [{"source": "data/a.md"}],
        k=3,
    )

    assert correct_result["no_answer_correct"] is True
    assert wrong_result["no_answer_correct"] is False
    assert correct_result["top1_hit"] is None


def test_evaluate_retriever_uses_shared_search_interface():
    cases = [
        RAGEvalCase(
            id="answerable",
            question="测试问题",
            expected_sources=["data/a.md"],
            should_answer=True,
        )
    ]

    def fake_search(question: str, top_k: int) -> list[dict]:
        assert question == "测试问题"
        assert top_k == 2
        return [
            {"source": "data/b.md"},
            {"source": "data/a.md"},
        ]

    results = evaluate_retriever(fake_search, cases, k=2)
    summary = summarize_retrieval_results(results)

    assert summary["top1_accuracy"] == 0.0
    assert summary["hit_at_k"] == 1.0
    assert summary["mrr_at_k"] == 0.5
