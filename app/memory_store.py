import sqlite3
from pathlib import Path

from app.memory import (
    MemoryCreate,
    MemoryReplacementConfirmation,
    MemoryRecord,
    MemoryReplacementProposalCreate,
    MemoryReplacementProposalRecord,
)


def migrate_memory_database(
    connection: sqlite3.Connection,
) -> None:
    column_names = {
        row[1]
        for row in connection.execute(
            "PRAGMA table_info(memories)"
        )
    }

    if "status" not in column_names:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN status TEXT NOT NULL
                DEFAULT 'active'
                CHECK (status IN ('active', 'superseded'))
            """
        )

    if "superseded_by" not in column_names:
        connection.execute(
            """
            ALTER TABLE memories
            ADD COLUMN superseded_by INTEGER
            """
        )


def initialize_memory_database(
    database_path: str | Path,
) -> None:
    connection = sqlite3.connect(database_path)

    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    source_session_id TEXT NOT NULL,
                    memory_type TEXT NOT NULL
                        CHECK (
                            memory_type IN (
                                'preference',
                                'fact',
                                'decision'
                            )
                        ),
                    content TEXT NOT NULL,
                    source TEXT NOT NULL
                        CHECK (
                            source IN (
                                'user_explicit',
                                'model_inferred',
                                'manual'
                            )
                        ),
                    status TEXT NOT NULL
                        DEFAULT 'active'
                        CHECK (
                            status IN (
                                'active',
                                'superseded'
                            )
                        ),
                    superseded_by INTEGER,
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            migrate_memory_database(connection)
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memory_replacement_proposals (
                    proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    old_memory_id INTEGER NOT NULL,
                    old_content_snapshot TEXT NOT NULL,
                    memory_type TEXT NOT NULL
                        CHECK (
                            memory_type IN (
                                'preference',
                                'fact',
                                'decision'
                            )
                        ),
                    new_content TEXT NOT NULL,
                    source TEXT NOT NULL
                        CHECK (
                            source IN (
                                'user_explicit',
                                'model_inferred'
                            )
                        ),
                    status TEXT NOT NULL
                        DEFAULT 'pending'
                        CHECK (
                            status IN (
                                'pending',
                                'confirmed',
                                'cancelled',
                                'expired'
                            )
                        ),
                    created_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL
                        DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
    finally:
        connection.close()


def add_memory(
    database_path: str | Path,
    memory: MemoryCreate,
) -> MemoryRecord:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            cursor = connection.execute(
                """
                INSERT INTO memories (
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    memory.user_id,
                    memory.source_session_id,
                    memory.memory_type,
                    memory.content,
                    memory.source,
                ),
            )
            row = connection.execute(
                """
                SELECT
                    memory_id,
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source,
                    status,
                    superseded_by,
                    created_at,
                    updated_at
                FROM memories
                WHERE memory_id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("Memory 写入后无法读取。")

    return MemoryRecord.model_validate(dict(row))


def add_memory_replacement_proposal(
    database_path: str | Path,
    proposal: MemoryReplacementProposalCreate,
) -> MemoryReplacementProposalRecord:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            old_row = connection.execute(
                """
                SELECT
                    memory_id,
                    content,
                    memory_type,
                    status
                FROM memories
                WHERE memory_id = ?
                  AND user_id = ?
                """,
                (
                    proposal.old_memory_id,
                    proposal.user_id,
                ),
            ).fetchone()

            if old_row is None:
                raise ValueError("旧记忆不存在，或不属于该用户。")

            if old_row["status"] != "active":
                raise ValueError("旧记忆当前不是 active 状态。")

            if old_row["memory_type"] != proposal.memory_type:
                raise ValueError("旧记忆类型与替换提案不一致。")

            if old_row["content"] != proposal.old_content_snapshot:
                raise ValueError(
                    "旧记忆内容已变更，请重新创建替换提案。"
                )

            cursor = connection.execute(
                """
                INSERT INTO memory_replacement_proposals (
                    user_id,
                    session_id,
                    old_memory_id,
                    old_content_snapshot,
                    memory_type,
                    new_content,
                    source
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    proposal.user_id,
                    proposal.session_id,
                    proposal.old_memory_id,
                    proposal.old_content_snapshot,
                    proposal.memory_type,
                    proposal.new_content,
                    proposal.source,
                ),
            )
            row = connection.execute(
                """
                SELECT
                    proposal_id,
                    user_id,
                    session_id,
                    old_memory_id,
                    old_content_snapshot,
                    memory_type,
                    new_content,
                    source,
                    status,
                    created_at,
                    updated_at
                FROM memory_replacement_proposals
                WHERE proposal_id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("替换提案写入后无法读取。")

    return MemoryReplacementProposalRecord.model_validate(dict(row))


def get_pending_memory_replacement_proposal(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    proposal_id: int,
) -> MemoryReplacementProposalRecord:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        row = connection.execute(
            """
            SELECT
                proposal_id,
                user_id,
                session_id,
                old_memory_id,
                old_content_snapshot,
                memory_type,
                new_content,
                source,
                status,
                created_at,
                updated_at
            FROM memory_replacement_proposals
            WHERE proposal_id = ?
              AND user_id = ?
              AND session_id = ?
              AND status = 'pending'
            """,
            (
                proposal_id,
                user_id,
                session_id,
            ),
        ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise ValueError(
            "待确认替换提案不存在、不属于该会话，或不再是 pending 状态。"
        )

    return MemoryReplacementProposalRecord.model_validate(dict(row))

def cancel_memory_replacement_proposal(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    proposal_id: int,
) -> MemoryReplacementProposalRecord:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            cursor = connection.execute(
                """
                UPDATE memory_replacement_proposals
                SET
                    status = 'cancelled',
                    updated_at = CURRENT_TIMESTAMP
                WHERE proposal_id = ?
                  AND user_id = ?
                  AND session_id = ?
                  AND status = 'pending'
                """,
                (
                    proposal_id,
                    user_id,
                    session_id,
                ),
            )

            if cursor.rowcount != 1:
                raise ValueError(
                    "待取消替换提案不存在、不属于该会话，"
                    "或不再是 pending 状态。"
                )

            row = connection.execute(
                """
                SELECT
                    proposal_id,
                    user_id,
                    session_id,
                    old_memory_id,
                    old_content_snapshot,
                    memory_type,
                    new_content,
                    source,
                    status,
                    created_at,
                    updated_at
                FROM memory_replacement_proposals
                WHERE proposal_id = ?
                  AND user_id = ?
                  AND session_id = ?
                """,
                (
                    proposal_id,
                    user_id,
                    session_id,
                ),
            ).fetchone()
    finally:
        connection.close()

    if row is None:
        raise RuntimeError("取消后无法读取替换提案。")

    return MemoryReplacementProposalRecord.model_validate(dict(row))
def get_user_memories(
    database_path: str | Path,
    user_id: str,
    include_superseded: bool = False,
) -> list[MemoryRecord]:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        rows = connection.execute(
            """
            SELECT
                memory_id,
                user_id,
                source_session_id,
                memory_type,
                content,
                source,
                status,
                superseded_by,
                created_at,
                updated_at
            FROM memories
            WHERE user_id = ?
              AND (status = 'active' OR ? = 1)
            ORDER BY memory_id
            """,
            (user_id, int(include_superseded)),
        ).fetchall()
    finally:
        connection.close()

    return [
        MemoryRecord.model_validate(dict(row))
        for row in rows
    ]


def supersede_memory(
    database_path: str | Path,
    user_id: str,
    old_memory_id: int,
    new_memory_id: int,
) -> MemoryRecord:
    if old_memory_id == new_memory_id:
        raise ValueError(
            "old_memory_id 和 new_memory_id 不能相同。"
        )

    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            rows = connection.execute(
                """
                SELECT
                    memory_id,
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source,
                    status,
                    superseded_by,
                    created_at,
                    updated_at
                FROM memories
                WHERE user_id = ?
                  AND memory_id IN (?, ?)
                """,
                (
                    user_id,
                    old_memory_id,
                    new_memory_id,
                ),
            ).fetchall()
            rows_by_id = {
                row["memory_id"]: row
                for row in rows
            }
            old_row = rows_by_id.get(old_memory_id)
            new_row = rows_by_id.get(new_memory_id)

            if old_row is None or new_row is None:
                raise ValueError(
                    "旧记忆或新记忆不存在，或不属于该用户。"
                )

            if old_row["status"] != "active":
                raise ValueError("旧记忆当前不是 active 状态。")

            if new_row["status"] != "active":
                raise ValueError("新记忆必须是 active 状态。")

            cursor = connection.execute(
                """
                UPDATE memories
                SET
                    status = 'superseded',
                    superseded_by = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE memory_id = ?
                  AND user_id = ?
                  AND status = 'active'
                """,
                (
                    new_memory_id,
                    old_memory_id,
                    user_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError("旧记忆替代更新失败。")

            updated_row = connection.execute(
                """
                SELECT
                    memory_id,
                    user_id,
                    source_session_id,
                    memory_type,
                    content,
                    source,
                    status,
                    superseded_by,
                    created_at,
                    updated_at
                FROM memories
                WHERE memory_id = ?
                """,
                (old_memory_id,),
            ).fetchone()
    finally:
        connection.close()

    if updated_row is None:
        raise RuntimeError("替代后无法读取旧记忆。")

    return MemoryRecord.model_validate(dict(updated_row))


def _replace_memory_in_transaction(
    connection: sqlite3.Connection,
    user_id: str,
    old_memory_id: int,
    new_memory: MemoryCreate,
) -> tuple[MemoryRecord, MemoryRecord]:
    if new_memory.user_id != user_id:
        raise ValueError("新记忆的 user_id 必须与 user_id 一致。")

    old_row = connection.execute(
        """
        SELECT
            memory_id,
            user_id,
            source_session_id,
            memory_type,
            content,
            source,
            status,
            superseded_by,
            created_at,
            updated_at
        FROM memories
        WHERE memory_id = ?
          AND user_id = ?
        """,
        (old_memory_id, user_id),
    ).fetchone()

    if old_row is None:
        raise ValueError("旧记忆不存在，或不属于该用户。")

    if old_row["status"] != "active":
        raise ValueError("旧记忆当前不是 active 状态。")

    if old_row["memory_type"] != new_memory.memory_type:
        raise ValueError("新旧记忆的 memory_type 必须一致。")

    cursor = connection.execute(
        """
        INSERT INTO memories (
            user_id,
            source_session_id,
            memory_type,
            content,
            source
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            new_memory.user_id,
            new_memory.source_session_id,
            new_memory.memory_type,
            new_memory.content,
            new_memory.source,
        ),
    )
    new_memory_id = cursor.lastrowid

    if new_memory_id is None:
        raise RuntimeError("新记忆写入后未获得 memory_id。")

    update_cursor = connection.execute(
        """
        UPDATE memories
        SET
            status = 'superseded',
            superseded_by = ?,
            updated_at = CURRENT_TIMESTAMP
        WHERE memory_id = ?
          AND user_id = ?
          AND status = 'active'
        """,
        (
            new_memory_id,
            old_memory_id,
            user_id,
        ),
    )

    if update_cursor.rowcount != 1:
        raise RuntimeError("旧记忆替代更新失败。")

    rows = connection.execute(
        """
        SELECT
            memory_id,
            user_id,
            source_session_id,
            memory_type,
            content,
            source,
            status,
            superseded_by,
            created_at,
            updated_at
        FROM memories
        WHERE memory_id IN (?, ?)
        """,
        (old_memory_id, new_memory_id),
    ).fetchall()

    rows_by_id = {
        row["memory_id"]: row
        for row in rows
    }
    updated_old_row = rows_by_id.get(old_memory_id)
    new_row = rows_by_id.get(new_memory_id)

    if updated_old_row is None or new_row is None:
        raise RuntimeError("替代后无法读取记忆记录。")

    return (
        MemoryRecord.model_validate(dict(updated_old_row)),
        MemoryRecord.model_validate(dict(new_row)),
    )


def confirm_memory_replacement_proposal(
    database_path: str | Path,
    user_id: str,
    session_id: str,
    proposal_id: int,
) -> MemoryReplacementConfirmation:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    expired_reason: str | None = None
    confirmation: MemoryReplacementConfirmation | None = None

    try:
        with connection:
            proposal_row = connection.execute(
                """
                SELECT
                    proposal_id,
                    user_id,
                    session_id,
                    old_memory_id,
                    old_content_snapshot,
                    memory_type,
                    new_content,
                    source,
                    status,
                    created_at,
                    updated_at
                FROM memory_replacement_proposals
                WHERE proposal_id = ?
                  AND user_id = ?
                  AND session_id = ?
                  AND status = 'pending'
                """,
                (
                    proposal_id,
                    user_id,
                    session_id,
                ),
            ).fetchone()

            if proposal_row is None:
                raise ValueError(
                    "待确认替换提案不存在、不属于该会话，"
                    "或不再是 pending 状态。"
                )

            claim_cursor = connection.execute(
                """
                UPDATE memory_replacement_proposals
                SET
                    status = 'confirmed',
                    updated_at = CURRENT_TIMESTAMP
                WHERE proposal_id = ?
                  AND user_id = ?
                  AND session_id = ?
                  AND status = 'pending'
                """,
                (
                    proposal_id,
                    user_id,
                    session_id,
                ),
            )

            if claim_cursor.rowcount != 1:
                raise RuntimeError("替换提案确认领取失败。")

            old_row = connection.execute(
                """
                SELECT
                    memory_id,
                    content,
                    memory_type,
                    status
                FROM memories
                WHERE memory_id = ?
                  AND user_id = ?
                """,
                (
                    proposal_row["old_memory_id"],
                    user_id,
                ),
            ).fetchone()

            if old_row is None or old_row["status"] != "active":
                expired_reason = (
                    "待确认替换提案已过期："
                    "旧记忆不存在或不再是 active 状态。"
                )
            elif old_row["memory_type"] != proposal_row["memory_type"]:
                expired_reason = (
                    "待确认替换提案已过期：旧记忆类型已变更。"
                )
            elif old_row["content"] != proposal_row["old_content_snapshot"]:
                expired_reason = (
                    "待确认替换提案已过期：旧记忆内容已变更。"
                )

            if expired_reason is not None:
                connection.execute(
                    """
                    UPDATE memory_replacement_proposals
                    SET
                        status = 'expired',
                        updated_at = CURRENT_TIMESTAMP
                    WHERE proposal_id = ?
                    """,
                    (proposal_id,),
                )
            else:
                new_memory = MemoryCreate(
                    user_id=user_id,
                    source_session_id=session_id,
                    memory_type=proposal_row["memory_type"],
                    content=proposal_row["new_content"],
                    source=proposal_row["source"],
                )
                old_memory, created_memory = (
                    _replace_memory_in_transaction(
                        connection,
                        user_id,
                        proposal_row["old_memory_id"],
                        new_memory,
                    )
                )
                confirmed_row = connection.execute(
                    """
                    SELECT
                        proposal_id,
                        user_id,
                        session_id,
                        old_memory_id,
                        old_content_snapshot,
                        memory_type,
                        new_content,
                        source,
                        status,
                        created_at,
                        updated_at
                    FROM memory_replacement_proposals
                    WHERE proposal_id = ?
                    """,
                    (proposal_id,),
                ).fetchone()

                if confirmed_row is None:
                    raise RuntimeError("确认后无法读取替换提案。")

                confirmation = MemoryReplacementConfirmation(
                    proposal=MemoryReplacementProposalRecord.model_validate(
                        dict(confirmed_row)
                    ),
                    old_memory=old_memory,
                    new_memory=created_memory,
                )
    finally:
        connection.close()

    if expired_reason is not None:
        raise ValueError(expired_reason)

    if confirmation is None:
        raise RuntimeError("替换提案确认后未生成结果。")

    return confirmation


def replace_memory(
    database_path: str | Path,
    user_id: str,
    old_memory_id: int,
    new_memory: MemoryCreate,
) -> tuple[MemoryRecord, MemoryRecord]:
    initialize_memory_database(database_path)

    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row

    try:
        with connection:
            return _replace_memory_in_transaction(
                connection,
                user_id,
                old_memory_id,
                new_memory,
            )
    finally:
        connection.close()
