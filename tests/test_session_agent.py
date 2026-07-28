import pytest

import app.session_agent as session_agent
from app.memory import (
    MemoryCandidate,
    MemoryCreate,
    MemoryReplacementProposalCreate,
    MemoryReplacementProposalRecord,
)
from app.memory_store import (
    add_memory,
    add_memory_replacement_proposal,
    get_pending_memory_replacement_proposal,
    get_user_memories,
)
from app.session_store import add_exchange, get_recent_messages


def make_pending_proposal() -> MemoryReplacementProposalRecord:
    return MemoryReplacementProposalRecord(
        proposal_id=1,
        user_id="user_001",
        session_id="session_A",
        old_memory_id=7,
        old_content_snapshot="项目继续使用 SQLite。",
        memory_type="decision",
        new_content="项目决定迁移到 PostgreSQL。",
        source="model_inferred",
        created_at="2026-07-28 10:00:00",
        updated_at="2026-07-28 10:00:00",
    )


def create_stored_pending_proposal(database_path, session_id="session_A"):
    old_memory = add_memory(
        database_path,
        MemoryCreate(
            user_id="user_001",
            source_session_id="old_session",
            memory_type="decision",
            content="项目继续使用 SQLite。",
            source="manual",
        ),
    )

    return add_memory_replacement_proposal(
        database_path,
        MemoryReplacementProposalCreate(
            user_id="user_001",
            session_id=session_id,
            old_memory_id=old_memory.memory_id,
            old_content_snapshot=old_memory.content,
            memory_type="decision",
            new_content="项目决定迁移到 PostgreSQL。",
            source="model_inferred",
        ),
    )


def test_format_pending_memory_replacement_notices():
    notice = session_agent.format_pending_memory_replacement_notices(
        [make_pending_proposal()],
    )

    assert session_agent.format_pending_memory_replacement_notices([]) == ""
    assert "系统状态：以下长期记忆更正尚未生效" in notice
    assert "提案 #1" in notice
    assert "旧记忆：项目继续使用 SQLite。" in notice
    assert "建议新记忆：项目决定迁移到 PostgreSQL。" in notice


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("确认提案 #1", ("confirm", 1)),
        ("取消提案 12", ("cancel", 12)),
        ("请确认提案 #1", None),
    ],
)
def test_parse_memory_replacement_command(query, expected):
    assert session_agent.parse_memory_replacement_command(query) == expected


def test_handle_memory_replacement_command_confirms_proposal(tmp_path):
    database_path = tmp_path / "sessions.db"
    proposal = create_stored_pending_proposal(database_path)

    answer = session_agent.handle_memory_replacement_command(
        f"确认提案 #{proposal.proposal_id}",
        database_path,
        "user_001",
        "session_A",
    )
    memories = get_user_memories(
        database_path,
        "user_001",
        include_superseded=True,
    )

    assert answer is not None
    assert "已确认提案 #1" in answer
    assert [(memory.content, memory.status) for memory in memories] == [
        ("项目继续使用 SQLite。", "superseded"),
        ("项目决定迁移到 PostgreSQL。", "active"),
    ]


def test_handle_memory_replacement_command_cancels_proposal(tmp_path):
    database_path = tmp_path / "sessions.db"
    proposal = create_stored_pending_proposal(database_path)

    answer = session_agent.handle_memory_replacement_command(
        f"取消提案 #{proposal.proposal_id}",
        database_path,
        "user_001",
        "session_A",
    )

    assert answer is not None
    assert "已取消提案 #1" in answer
    assert [(memory.content, memory.status) for memory in get_user_memories(
        database_path,
        "user_001",
    )] == [("项目继续使用 SQLite。", "active")]


def test_handle_memory_replacement_command_rejects_other_session(tmp_path):
    database_path = tmp_path / "sessions.db"
    proposal = create_stored_pending_proposal(database_path)

    answer = session_agent.handle_memory_replacement_command(
        f"确认提案 #{proposal.proposal_id}",
        database_path,
        "user_001",
        "session_B",
    )
    pending = get_pending_memory_replacement_proposal(
        database_path,
        "user_001",
        "session_A",
        proposal.proposal_id,
    )

    assert answer == (
        "无法处理该提案：它不存在、不属于当前会话，"
        "或已不再是 pending 状态。"
    )
    assert pending.status == "pending"


def test_run_session_agent_handles_command_without_calling_model(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"
    proposal = create_stored_pending_proposal(database_path)

    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("提案命令不应调用 Agent 模型"),
        ),
    )

    result = session_agent.run_session_agent(
        f"确认提案 #{proposal.proposal_id}",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result is not None
    assert "已确认提案 #1" in result
    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=2,
    ) == [
        {"role": "user", "content": "确认提案 #1"},
        {"role": "assistant", "content": result},
    ]


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
    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        lambda user_message, assistant_answer, config: [],
    )

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
    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        lambda user_message, assistant_answer, config: [],
    )

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


def test_run_session_agent_saves_inferred_memory_after_exchange(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"

    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda query, config, messages=None: "好的，我会使用表格。",
    )

    def fake_extract(user_message, assistant_answer, config):
        assert get_recent_messages(
            database_path,
            "user_001",
            "session_A",
            max_messages=2,
        ) == [
            {"role": "user", "content": "请记住：我喜欢用表格总结。"},
            {"role": "assistant", "content": "好的，我会使用表格。"},
        ]
        return [
            MemoryCandidate(
                memory_type="preference",
                content="用户喜欢用表格总结。",
                source="model_inferred",
            )
        ]

    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        fake_extract,
    )

    result = session_agent.run_session_agent(
        "请记住：我喜欢用表格总结。",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result == "好的，我会使用表格。"
    assert [memory.content for memory in get_user_memories(
        database_path,
        "user_001",
    )] == ["用户喜欢用表格总结。"]


def test_run_session_agent_passes_candidates_to_inferred_writer_after_exchange(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"
    config = object()
    candidate = MemoryCandidate(
        memory_type="decision",
        content="项目决定迁移到 PostgreSQL。",
        source="model_inferred",
    )
    captured = {}

    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda query, config, messages=None: "正常回答",
    )
    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        lambda user_message, assistant_answer, config: [candidate],
    )

    def fake_save(database_path_arg, user_id, session_id, candidates, config_arg):
        assert get_recent_messages(
            database_path,
            "user_001",
            "session_A",
            max_messages=2,
        ) == [
            {"role": "user", "content": "测试问题"},
            {"role": "assistant", "content": "正常回答"},
        ]
        captured["arguments"] = (
            database_path_arg,
            user_id,
            session_id,
            candidates,
            config_arg,
        )
        return []

    monkeypatch.setattr(
        session_agent,
        "save_inferred_memory_candidates",
        fake_save,
    )

    result = session_agent.run_session_agent(
        "测试问题",
        config,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result == "正常回答"
    assert captured["arguments"] == (
        database_path,
        "user_001",
        "session_A",
        [candidate],
        config,
    )


def test_run_session_agent_appends_pending_proposal_notice(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"
    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda query, config, messages=None: "模型回答",
    )
    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        lambda user_message, assistant_answer, config: [],
    )
    monkeypatch.setattr(
        session_agent,
        "save_inferred_memory_candidates",
        lambda database_path, user_id, session_id, candidates, config: [
            make_pending_proposal(),
        ],
    )

    result = session_agent.run_session_agent(
        "测试问题",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result.startswith("模型回答\n系统状态：")
    assert "提案 #1" in result


def test_run_session_agent_keeps_answer_when_memory_extraction_fails(
    tmp_path,
    monkeypatch,
):
    database_path = tmp_path / "sessions.db"
    monkeypatch.setattr(
        session_agent,
        "agent_loop",
        lambda query, config, messages=None: "正常回答",
    )

    def fail_extract(*args, **kwargs):
        raise ValueError("模型返回非法 JSON")

    monkeypatch.setattr(
        session_agent,
        "extract_inferred_memory_candidates",
        fail_extract,
    )

    result = session_agent.run_session_agent(
        "测试问题",
        None,
        database_path,
        "user_001",
        "session_A",
        "你是学习助手",
    )

    assert result == "正常回答"
    assert get_recent_messages(
        database_path,
        "user_001",
        "session_A",
        max_messages=2,
    ) == [
        {"role": "user", "content": "测试问题"},
        {"role": "assistant", "content": "正常回答"},
    ]
    assert get_user_memories(database_path, "user_001") == []
