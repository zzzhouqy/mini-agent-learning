from app.rag import format_contexts, search_knowledge


def test_search_knowledge_returns_structured_context():
    results = search_knowledge("Pydantic")

    assert len(results) > 0
    assert results[0]["source"] == "data/knowledge.md"
    assert results[0]["title"] == "Pydantic"
    assert "结构化校验" in results[0]["content"]


def test_search_knowledge_returns_empty_list_when_not_found():
    results = search_knowledge("天气")

    assert results == []


def test_format_contexts_returns_message_when_empty():
    assert format_contexts([]) == "没有检索到相关资料。"