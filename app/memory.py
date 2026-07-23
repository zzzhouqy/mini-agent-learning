from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    memory_type: Literal[
        "preference",
        "fact",
        "decision",
    ]
    content: str = Field(min_length=1)
    source: Literal[
        "user_explicit",
        "model_inferred",
    ]


class MemoryExtractionItem(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    memory_type: Literal[
        "preference",
        "fact",
        "decision",
    ]
    content: str = Field(min_length=1)


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    candidates: list[MemoryExtractionItem]


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
        "model_inferred",
        "manual",
    ]


class MemoryRecord(MemoryCreate):
    memory_id: int = Field(ge=1)
    created_at: str = Field(min_length=1)
    updated_at: str = Field(min_length=1)


class MemoryMatch(BaseModel):
    memory: MemoryRecord
    score: float = Field(ge=-1.0, le=1.0)
