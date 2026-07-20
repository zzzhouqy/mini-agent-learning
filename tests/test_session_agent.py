import pytest

import app.session_agent as session_agent
from app.session_store import add_exchange, get_recent_messages


def test_run_session_agent_saves_successful_exchange(tmp_path, monkeypatch):
    database_path = tmp_path / "sessions.db"
    add_exchange(
        database_path,
        "user_001",
        "session_A",
        "我喜欢用表格总结",
        "已记录你的偏好",
    )
    captured = {}

    def fake_agent_loop(query, config, messages=None):
        captured["messages"] = messages
        return "你喜欢用表格总结"

    monkeypatch.setattr(session_agent, "agent_loop", fake_agent_loop)

    result = session_agent.run_session_agent(
        "我喜欢怎样总结？",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
        max_messages=2,
    )

    assert result == "你喜欢用表格总结"
    assert captured["messages"][-1] == {
        "role": "user",
        "content": "我喜欢怎样总结？",
    }
    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=10,
    )[-2:] == [
        {"role": "user", "content": "我喜欢怎样总结？"},
        {"role": "assistant", "content": "你喜欢用表格总结"},
    ]


def test_run_session_agent_does_not_save_failed_exchange(tmp_path, monkeypatch):
    database_path = tmp_path / "sessions.db"
    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda query, config, messages=None: None,
    )

    result = session_agent.run_session_agent(
        "未完成的问题",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result is None
    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=10,
    ) == []


def test_run_session_agent_retrieves_and_injects_memory(tmp_path, monkeypatch):
    database_path = tmp_path / "sessions.db"
    captured = {}

    def fake_search(database_path, user_id, query, **kwargs):
        captured["search"] = (user_id, query)
        return ["memory match"]

    def fake_format(matches):
        assert matches == ["memory match"]
        return "<relevant_memories>表格偏好</relevant_memories>"

    def fake_agent_loop(query, config, messages=None):
        captured["messages"] = messages
        return "模拟回答"

    monkeypatch.setattr(session_agent, "search_user_memories", fake_search)
    monkeypatch.setattr(session_agent, "format_memory_matches", fake_format)
    monkeypatch.setattr(session_agent, "agent_loop", fake_agent_loop)

    result = session_agent.run_session_agent(
        "我喜欢怎样总结？",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result == "模拟回答"
    assert captured["search"] == ("user_001", "我喜欢怎样总结？")
    assert "表格偏好" in captured["messages"][0]["content"]


def test_run_session_agent_exposes_memory_retrieval_failure(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"

    def fail_search(*args, **kwargs):
        raise RuntimeError("Embedding 加载失败")

    def fail_if_agent_called(*args, **kwargs):
        raise AssertionError("检索失败后不应调用 Agent")

    monkeypatch.setattr(session_agent, "search_user_memories", fail_search)
    monkeypatch.setattr(session_agent, "agent_loop", fail_if_agent_called)

    with pytest.raises(RuntimeError, match="Embedding 加载失败"):
        session_agent.run_session_agent(
            "测试问题",
            None,
            database_path,
            "user_001",
            "session_A",
            "你是学习助手",
        )

    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=10,
    ) == []
