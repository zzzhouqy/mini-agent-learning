from pydantic import ValidationError
from app.utils import add
from app.csv_tool import get_csv_row
import httpx
# 合并导入，只写一次
from app.config import ConfigError, LLMConfig
from app.schemas import AddArguments, CsvRowArguments

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和",
            "parameters": AddArguments.model_json_schema(),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_csv_row",
            "description": "读取固定 CSV 文件中指定行的数据",
            "parameters": CsvRowArguments.model_json_schema(),
        },
    },
]
TOOL_HANDLERS = {
    "add": add,
    "get_csv_row": get_csv_row,
}

TOOL_SCHEMAS = {
    "add": AddArguments,
    "get_csv_row": CsvRowArguments,
}
class AgentError(RuntimeError):
    """Agent执行失败。"""
def send_messages(
    config: LLMConfig,
    messages: list[dict],
    tools: list[dict] | None = None,
) -> dict:
    payload = {
        "model": config.model,
        "messages": messages,
    }

    if tools is not None:
        payload["tools"] = tools

    # 第一层 try：捕获网络/HTTP 请求异常
    try:
        response = httpx.post(
            config.chat_completions_url,
            headers={
                "Authorization": f"Bearer {config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise AgentError("模型请求超时") from exc
    except httpx.NetworkError as exc:
        raise AgentError("网络连接失败") from exc
    except httpx.HTTPStatusError as exc:
        raise AgentError(f"API返回HTTP {exc.response.status_code}") from exc

    # 第二层 try：捕获返回体解析异常
    try:
        return response.json()["choices"][0]["message"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise AgentError("模型返回了无法解析的响应") from exc
def execute_tool(
    tool_name: str,
    arguments_text: str,
) -> tuple[dict | str, object, bool]:
    handler = TOOL_HANDLERS.get(tool_name)
    schema = TOOL_SCHEMAS.get(tool_name)

    if handler is None or schema is None:
        return {}, f"Error: 未知工具 {tool_name}", False

    try:
        validated_arguments = schema.model_validate_json(
            arguments_text
        )
    except ValidationError as exc:
        return (
            arguments_text,
            "Error: 参数校验失败："
            f"{exc.errors(include_url=False)}",
            True,
        )

    arguments = validated_arguments.model_dump()

    try:
        result = handler(**arguments)
    except (
        FileNotFoundError,
        IndexError,
        ValueError,
        UnicodeError,
    ) as exc:
        return (
            arguments,
            f"Error: 工具执行失败：{exc}",
            False,
        )

    return arguments, result, False


def agent_loop(
    query: str,
    config: LLMConfig,
    max_steps: int = 5,
    max_validation_retries: int = 2,
) -> None:
    messages = [{"role": "user", "content": query}]
    validation_failures = 0
    for step in range(1, max_steps + 1):
        print(f"\n第 {step} 轮")

        message = send_messages(config, messages, tools=TOOLS)
        messages.append(message)

        tool_calls = message.get("tool_calls")

        # 模型没有调用工具，说明任务完成
        if not tool_calls:
            print(f"模型最终回答：{message['content']}")
            return

        # 执行本轮的所有工具
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments_text = tool_call["function"]["arguments"]

            arguments, result,validation_failed = execute_tool(
                tool_name,
                arguments_text,
            )

            print(f"调用工具：{tool_name}{arguments}")
            print(f"执行结果：{result}")
            if validation_failed:
                validation_failures += 1

                if validation_failures > max_validation_retries:
                    print(
                        f"参数校验连续失败，"
                        f"已用完 {max_validation_retries} 次重试。"
                    )
                    return

                print(
                    f"将错误反馈给模型，准备进行第 "
                    f"{validation_failures}/"
                    f"{max_validation_retries} 次重试。"
                )
            else:
                validation_failures = 0
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                }
            )

def main() -> None:
    try:
        config = LLMConfig.from_env()

        query = input("请输入任务：").strip()
        if not query:
            print("任务不能为空")
            return

        agent_loop(query, config)
    except (ConfigError, AgentError) as exc:
        print(f"Agent执行失败：{exc}")


if __name__ == "__main__":
    main()