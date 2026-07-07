from pydantic import BaseModel, ConfigDict, Field


class AddArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    a: int = Field(
        ge=-1_000_000,
        le=1_000_000,
        description="第一个整数",
    )
    b: int = Field(
        ge=-1_000_000,
        le=1_000_000,
        description="第二个整数",
    )


class CsvRowArguments(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        strict=True,
    )

    row_index: int = Field(
        ge=0,
        description="从 0 开始的数据行号，不包含表头",
    )