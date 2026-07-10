from app.rag import format_contexts, score_chunk, search_knowledge


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