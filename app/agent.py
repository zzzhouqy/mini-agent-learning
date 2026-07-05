import json
from app.utils import add
import httpx
# 合并导入，只写一次
from app.config import ConfigError, LLMConfig

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "计算两个整数的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "type": "integer",
                        "description": "第一个整数",
                    },
                    "b": {
                        "type": "integer",
                        "description": "第二个整数",
                    },
                },
                "required": ["a", "b"],
            },
        },
    }
]
TOOL_HANDLERS = {
    "add": add,
}
class AgentError(RuntimeError):
    """Agent执行失败。"""
def send_messages(config: LLMConfig, messages: list[dict]) -> dict:
    payload = {
        "model": config.model,
        "messages": messages,
        "tools": TOOLS,
    }

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


def agent_loop(
    query: str,
    config: LLMConfig,
    max_steps: int = 5,
) -> None:
    messages = [{"role": "user", "content": query}]

    for step in range(1, max_steps + 1):
        print(f"\n第 {step} 轮")

        message = send_messages(config, messages)
        messages.append(message)

        tool_calls = message.get("tool_calls")

        # 模型没有调用工具，说明任务完成
        if not tool_calls:
            print(f"模型最终回答：{message['content']}")
            return

        # 执行本轮的所有工具
        for tool_call in tool_calls:
            tool_name = tool_call["function"]["name"]
            arguments = json.loads(
                tool_call["function"]["arguments"]
            )

            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result = f"Error: 未知工具 {tool_name}"
            else:
                result = handler(**arguments)

            print(f"调用工具：{tool_name}{arguments}")
            print(f"执行结果：{result}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": str(result),
                }
            )

    print(f"达到最大执行轮数 {max_steps}，任务停止。")


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