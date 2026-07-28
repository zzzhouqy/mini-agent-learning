import re
import logging

from pathlib import Path
from typing import Literal

from app.agent import agent_loop
from app.config import LLMConfig
from app.memory import MemoryReplacementProposalRecord
from app.memory_retriever import (
    format_memory_matches,
    search_user_memories,
)
from app.memory_writer import (
    extract_inferred_memory_candidates,
    save_inferred_memory_candidates,
)

from app.session_context import build_session_context
from app.memory_store import (
    cancel_memory_replacement_proposal,
    confirm_memory_replacement_proposal,
)
from app.session_store import add_exchange


LOGGER = logging.getLogger(__name__)


MEMORY_REPLACEMENT_COMMAND_PATTERN = re.compile(
    r"^\s*(确认|取消)提案\s*#?(\d+)\s*$"
)


def parse_memory_replacement_command(
    query: str,
) -> tuple[Literal["confirm", "cancel"], int] | None:
    match = MEMORY_REPLACEMENT_COMMAND_PATTERN.match(query)

    if match is None:
        return None

    action = "confirm" if match.group(1) == "确认" else "cancel"

    return action, int(match.group(2))


def handle_memory_replacement_command(
    query: str,
    database_path: str | Path,
    user_id: str,
    session_id: str,
) -> str | None:
    command = parse_memory_replacement_command(query)

    if command is None:
        return None

    action, proposal_id = command

    try:
        if action == "confirm":
            confirmation = confirm_memory_replacement_proposal(
                database_path,
                user_id,
                session_id,
                proposal_id,
            )

            return (
                f"已确认提案 #{proposal_id}。\n"
                f"旧记忆已被替代；当前生效："
                f"{confirmation.new_memory.content}"
            )

        proposal = cancel_memory_replacement_proposal(
            database_path,
            user_id,
            session_id,
            proposal_id,
        )

        return (
            f"已取消提案 #{proposal_id}。\n"
            f"旧记忆继续生效：{proposal.old_content_snapshot}"
        )
    except ValueError:
        return (
            "无法处理该提案：它不存在、不属于当前会话，"
            "或已不再是 pending 状态。"
        )


def format_pending_memory_replacement_notices(
    proposals: list[MemoryReplacementProposalRecord],
) -> str:
    if not proposals:
        return ""

    lines = [
        "",
        "系统状态：以下长期记忆更正尚未生效，旧记忆仍在使用：",
    ]

    for proposal in proposals:
        lines.extend(
            [
                f"- 提案 #{proposal.proposal_id}",
                f"  旧记忆：{proposal.old_content_snapshot}",
                f"  建议新记忆：{proposal.new_content}",
            ]
        )

    lines.append(
        "请回复“确认提案 #编号”或“取消提案 #编号”。"
    )

    return "\n".join(lines)


def run_session_agent(
    query: str,
    config: LLMConfig,
    database_path: str | Path,
    user_id: str,
    session_id: str,
    system_prompt: str,
    max_messages: int = 3,
) -> str | None:
    command_answer = handle_memory_replacement_command(
        query,
        database_path,
        user_id,
        session_id,
    )

    if command_answer is not None:
        add_exchange(
            database_path,
            user_id,
            session_id,
            query,
            command_answer,
        )

        return command_answer


    memory_matches = search_user_memories(
        database_path,
        user_id,
        query,
        min_score=0.3,
        top_k=3,
    )
    memory_context = format_memory_matches(memory_matches)

    messages = build_session_context(
        database_path,
        user_id,
        session_id,
        system_prompt,
        query,
        max_messages=max_messages,
        memory_context=memory_context,
    )

    answer = agent_loop(
        query,
        config,
        messages=messages,
    )

    if answer is None:
        return None

    add_exchange(
        database_path,
        user_id,
        session_id,
        query,
        answer,
    )
    pending_proposals = []
    try:
        candidates = extract_inferred_memory_candidates(
            query,
            answer,
            config,
        )
        write_results = save_inferred_memory_candidates(
            database_path,
            user_id,
            session_id,
            candidates,
            config,
        )
        pending_proposals = [
            result
            for result in write_results
            if isinstance(result, MemoryReplacementProposalRecord)
        ]
    except Exception:
        LOGGER.exception("自动长期记忆写入失败。")

    return answer + format_pending_memory_replacement_notices(
        pending_proposals,
    )
