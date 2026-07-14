from functools import lru_cache

from sentence_transformers import CrossEncoder


MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(MODEL_NAME)


def rerank_scores(query: str, texts: list[str]) -> list[float]:
    if not texts:
        return []

    pairs = [(query, text) for text in texts]
    scores = get_reranker().predict(pairs)

    return [float(score) for score in scores]


def rerank_contexts(
    query: str,
    contexts: list[dict],
    top_k: int = 3,
) -> list[dict]:
    if not contexts:
        return []

    texts = [context["content"] for context in contexts]
    scores = rerank_scores(query, texts)

    reranked_contexts = [
        {
            **context,
            "rerank_score": score,
        }
        for context, score in zip(contexts, scores)
    ]
    reranked_contexts.sort(
        key=lambda context: context["rerank_score"],
        reverse=True,
    )

    return reranked_contexts[:top_k]
