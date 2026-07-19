from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCreate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    user_id: str = Field(min_length=1)
    source_session_id: str = Field(min_length=1)
    memory_type: Literal[
        "preference",
        "fact",
        "decision",
    ]
    content: str = Field(min_length=1)
    source: Literal[
        "user_explicit",
        "manual",
    ]


class MemoryRecord(MemoryCreate):
    memory_id: int = Field(ge=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)
