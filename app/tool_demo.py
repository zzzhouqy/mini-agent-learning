import json
from app.utils import add
import httpx

from app.config import LLMConfig


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


def main() -> None:
    config = LLMConfig.from_env()

    payload = {
        "model": config.model,
        "messages": [
            {
                "role": "user",
                "content": "请使用 add 工具计算 123 + 456，不要自己心算。",
            }
        ],
        "tools": TOOLS,
    }

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

    message = response.json()["choices"][0]["message"]
    tool_call = message["tool_calls"][0]
    tool_name = tool_call["function"]["name"]
    arguments_text = tool_call["function"]["arguments"]
    arguments = json.loads(arguments_text)

    print(f"工具名称：{tool_name}")
    print(f"参数类型：{type(arguments).__name__}")
    print(f"工具参数：{arguments}")

    # 修复if缩进
    if tool_name != "add":
        raise ValueError(f"未知工具：{tool_name}")

    result = add(**arguments)
    print(f"执行结果：{result}")

    # 保留最初的用户消息
    messages = payload["messages"]

    # 加入模型提出的完整 Tool Call
    messages.append(message)

    # 加入 Python 函数的执行结果
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result),
        }
    )

    # 携带完整对话，再请求一次模型
    second_payload = {
        "model": config.model,
        "messages": messages,
        "tools": TOOLS,
    }

    second_response = httpx.post(
        config.chat_completions_url,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json=second_payload,
        timeout=config.timeout_seconds,
    )
    second_response.raise_for_status()

    final_message = second_response.json()["choices"][0]["message"]
    print(f"模型最终回答：{final_message['content']}")


if __name__ == "__main__":
    main()