import json
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


EVAL_CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "rag_eval_cases.json"
)


class RAGEvalCase(BaseModel):
    model_config = ConfigDict(strict=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_sources: list[str]
    should_answer: bool


def load_eval_cases(
    path: Path = EVAL_CASES_PATH,
) -> list[RAGEvalCase]:
    if not path.exists():
        raise FileNotFoundError(
            f"RAG 评测集不存在：{path}"
        )

    raw_cases = json.loads(
        path.read_text(encoding="utf-8")
    )
    if not isinstance(raw_cases, list):
        raise ValueError("RAG 评测集顶层必须是 JSON 数组。")

    return [
        RAGEvalCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]


def hit_at_k(
    expected_sources: list[str],
    contexts: list[dict],
    k: int,
) -> bool:
    if k < 1:
        raise ValueError("k 必须大于或等于 1。")

    retrieved_sources = [
        context["source"]
        for context in contexts[:k]
    ]

    return any(
        source in retrieved_sources
        for source in expected_sources
    )


def top1_hit(
    expected_sources: list[str],
    contexts: list[dict],
) -> bool:
    return hit_at_k(
        expected_sources,
        contexts,
        k=1,
    )


def source_recall_at_k(
    expected_sources: list[str],
    contexts: list[dict],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k 必须大于或等于 1。")
    if not expected_sources:
        raise ValueError("expected_sources 不能为空。")

    retrieved_sources = {
        context["source"]
        for context in contexts[:k]
    }
    matched_count = sum(
        source in retrieved_sources
        for source in expected_sources
    )

    return matched_count / len(expected_sources)


def reciprocal_rank_at_k(
    expected_sources: list[str],
    contexts: list[dict],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k 必须大于或等于 1。")
    if not expected_sources:
        raise ValueError("expected_sources 不能为空。")

    for rank, context in enumerate(
        contexts[:k],
        start=1,
    ):
        if context["source"] in expected_sources:
            return 1 / rank

    return 0.0


def evaluate_retrieval_case(
    case: RAGEvalCase,
    contexts: list[dict],
    k: int = 3,
) -> dict:
    if k < 1:
        raise ValueError("k 必须大于或等于 1。")

    retrieved_sources = [
        context["source"]
        for context in contexts[:k]
    ]

    if not case.should_answer:
        return {
            "id": case.id,
            "question": case.question,
            "expected_sources": case.expected_sources,
            "retrieved_sources": retrieved_sources,
            "top1_hit": None,
            "hit_at_k": None,
            "source_recall_at_k": None,
            "reciprocal_rank_at_k": None,
            "no_answer_correct": len(contexts) == 0,
        }

    return {
        "id": case.id,
        "question": case.question,
        "expected_sources": case.expected_sources,
        "retrieved_sources": retrieved_sources,
        "top1_hit": top1_hit(
            case.expected_sources,
            contexts,
        ),
        "hit_at_k": hit_at_k(
            case.expected_sources,
            contexts,
            k,
        ),
        "source_recall_at_k": source_recall_at_k(
            case.expected_sources,
            contexts,
            k,
        ),
        "reciprocal_rank_at_k": reciprocal_rank_at_k(
            case.expected_sources,
            contexts,
            k,
        ),
        "no_answer_correct": None,
    }


def evaluate_retriever(
    search_function: Callable[..., list[dict]],
    cases: list[RAGEvalCase],
    k: int = 3,
) -> list[dict]:
    if k < 1:
        raise ValueError("k 必须大于或等于 1。")

    results = []
    for case in cases:
        contexts = search_function(
            case.question,
            top_k=k,
        )
        results.append(
            evaluate_retrieval_case(
                case,
                contexts,
                k=k,
            )
        )

    return results


def summarize_retrieval_results(
    results: list[dict],
) -> dict:
    top1_values = [
        result["top1_hit"]
        for result in results
        if result["top1_hit"] is not None
    ]
    hit_values = [
        result["hit_at_k"]
        for result in results
        if result["hit_at_k"] is not None
    ]
    recall_values = [
        result["source_recall_at_k"]
        for result in results
        if result["source_recall_at_k"] is not None
    ]
    reciprocal_rank_values = [
        result["reciprocal_rank_at_k"]
        for result in results
        if result["reciprocal_rank_at_k"] is not None
    ]
    no_answer_values = [
        result["no_answer_correct"]
        for result in results
        if result["no_answer_correct"] is not None
    ]

    return {
        "total_cases": len(results),
        "answerable_cases": len(top1_values),
        "no_answer_cases": len(no_answer_values),
        "top1_accuracy": (
            sum(top1_values) / len(top1_values)
            if top1_values
            else None
        ),
        "hit_at_k": (
            sum(hit_values) / len(hit_values)
            if hit_values
            else None
        ),
        "mean_source_recall_at_k": (
            sum(recall_values) / len(recall_values)
            if recall_values
            else None
        ),
        "mrr_at_k": (
            sum(reciprocal_rank_values)
            / len(reciprocal_rank_values)
            if reciprocal_rank_values
            else None
        ),
        "no_answer_accuracy": (
            sum(no_answer_values) / len(no_answer_values)
            if no_answer_values
            else None
        ),
    }
