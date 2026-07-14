from app.reranker import rerank_contexts


def test_rerank_contexts_orders_and_limits_results(monkeypatch):
    contexts = [
        {
            "source": "data/weather.md",
            "title": "天气",
            "content": "今天天气很好。",
            "score": 0.9,
        },
        {
            "source": "data/pydantic.md",
            "title": "Pydantic",
            "content": "Pydantic 可以检查参数类型和字段名。",
            "score": 0.5,
        },
        {
            "source": "data/csv.md",
            "title": "CSV Tool",
            "content": "CSV Tool 可以读取固定路径的数据。",
            "score": 0.4,
        },
    ]

    def fake_rerank_scores(
        query: str,
        texts: list[str],
    ) -> list[float]:
        assert query == "如何校验工具调用参数？"
        assert texts == [
            context["content"]
            for context in contexts
        ]

        return [0.1, 0.9, 0.3]

    monkeypatch.setattr(
        "app.reranker.rerank_scores",
        fake_rerank_scores,
    )

    results = rerank_contexts(
        "如何校验工具调用参数？",
        contexts,
        top_k=2,
    )

    assert [result["title"] for result in results] == [
        "Pydantic",
        "CSV Tool",
    ]
    assert results[0]["source"] == "data/pydantic.md"
    assert results[0]["score"] == 0.5
    assert results[0]["rerank_score"] == 0.9