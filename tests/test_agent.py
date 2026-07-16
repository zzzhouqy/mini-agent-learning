import app.agent as agent


def test_agent_loop_returns_answer_without_mutating_input_messages(monkeypatch):
    provided = [
        {"role": "system", "content": "你是学习助手"},
        {"role": "user", "content": "测试问题"},
    ]
    captured = {}

    def fake_send_messages(config, messages, tools=None):
        captured["messages"] = messages.copy()
        return {"role": "assistant", "content": "模拟回答"}

    monkeypatch.setattr(agent, "send_messages", fake_send_messages)

    result = agent.agent_loop(
        "不会使用这个问题",
        None,
        messages=provided,
    )

    assert result == "模拟回答"
    assert captured["messages"] == provided
    assert len(provided) == 2
