from pydantic import BaseModel, ConfigDict, Field, ValidationError
from typing import Literal

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
class LearningRecord(BaseModel):
    category: Literal["算法", "Agent", "论文"]
    topic: str
    note: str | None = None
class StudyDetail(BaseModel):
    duration_minutes: int = Field(gt=0)
    completed: bool


class DailyStudy(BaseModel):
    record: LearningRecord
    detail: StudyDetail
def validate_arguments(raw_data: dict) -> None:
    try:
        arguments = AddArguments.model_validate(raw_data)
        print(f"校验成功：{arguments.model_dump()}")
    except ValidationError as exc:
        print("校验失败：")
        print(exc)


validate_arguments({"a": 3, "b": 5})
validate_arguments({"a": "苹果", "b": 5})

print("\n测试缺少字段：")
validate_arguments({"a": 3})

print("\n测试多余字段：")
validate_arguments({"a": 3, "b": 5, "c": 10})

print("\n测试字符串形式的整数：")
validate_arguments({"a": "3", "b": 5})
print("\n测试超出范围：")
validate_arguments({"a": 2_000_000, "b": 5})
print("\n测试学习记录：")
record = LearningRecord.model_validate(
    {
        "category": "Agent",
        "topic": "Tool Calling",
    }
)
print(record.model_dump())

print("\n测试非法分类：")
try:
    LearningRecord.model_validate(
        {
            "category": "游戏",
            "topic": "测试",
        }
    )
except ValidationError as exc:
    print(exc)
print("\n测试嵌套模型：")
daily_study = DailyStudy.model_validate(
    {
        "record": {
            "category": "Agent",
            "topic": "Pydantic",
        },
        "detail": {
            "duration_minutes": 90,
            "completed": True,
        },
    }
)
print(daily_study.model_dump())

print("\n测试嵌套字段错误：")
try:
    DailyStudy.model_validate(
        {
            "record": {
                "category": "Agent",
                "topic": "Pydantic",
            },
            "detail": {
                "duration_minutes": -10,
                "completed": True,
            },
        }
    )
except ValidationError as exc:
    print(exc)