from typing import Any

import httpx

from app.config import LLMConfig


class LLMClientError(RuntimeError):
    """LLM API调用失败。"""


def _error_detail(response: httpx.Response) -> str:
    try:
        body: Any = response.json()
    except ValueError:
        return response.text[:200] or "响应正文为空"

    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])
        if error:
            return str(error)
    return str(body)[:200]


def chat(prompt: str, config: LLMConfig) -> str:
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
    }
    # payload = {
    #     "model": config.model,
    #     "messages": [{"role": "user", "content": prompt}],
    # }
    payload = {
    "model": config.model,
    "messages": [
        {
            "role": "system",
            "content": (
                '请将用户的学习记录整理成 JSON。'
                '格式示例：{"topic": "学习主题", "summary": "学习总结"}'
            ),
        },
        {"role": "user", "content": prompt},
    ],
    "response_format": {"type": "json_object"},
}
    try:
        response = httpx.post(
            config.chat_completions_url,
            headers=headers,
            json=payload,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
    except httpx.TimeoutException as exc:
        raise LLMClientError(
            f"请求超时（{config.timeout_seconds:g}秒），请稍后重试。"
        ) from exc
    except httpx.NetworkError as exc:
        raise LLMClientError(
            "网络连接失败，请检查网络、代理和 LLM_BASE_URL。"
        ) from exc
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        detail = _error_detail(exc.response)
        raise LLMClientError(f"API返回HTTP {status}：{detail}") from exc

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise LLMClientError("API响应不是预期的聊天补全JSON结构。") from exc

    if not isinstance(content, str) or not content.strip():
        raise LLMClientError("API返回了空回答。")
    return content.strip()


"""一、构造请求头与请求体（对应 28～35 行）
python
运行
headers = {
    "Authorization": f"Bearer {config.api_key}",
    "Content-Type": "application/json",
}
payload = {
    "model": config.model,
    "messages": [{"role": "user", "content": prompt}],
}
这一步是准备发往 API 的完整请求数据，分为请求头和请求体两部分。
请求头 headers
Authorization：大模型 API 的标准鉴权格式（Bearer Token），把配置中的 API 密钥拼接进去，服务端通过这个密钥验证身份、统计用量和计费。
Content-Type: application/json：告知服务端，本次请求的正文是 JSON 格式，服务端会按 JSON 规则解析数据。
请求体 payload
这是真正发给大模型的业务数据，遵循通用聊天补全接口的标准格式：
model：指定要调用的模型名称，从配置对象中读取。
messages：聊天消息列表，是一个数组格式。每条消息包含 role（角色）和 content（内容）两个字段。这里是单轮提问，所以列表里只有一条用户消息：
role: "user"：表示这条消息来自用户
content: prompt：消息内容就是用户输入的问题"""

""" 
二、发送 HTTP POST 请求（对应 38～43 行）
python
运行
response = httpx.post(
    config.chat_completions_url,
    headers=headers,
    json=payload,
    timeout=config.timeout_seconds,
)
这一步是真正发起网络请求，使用 httpx（Python 的第三方 HTTP 客户端库）发送 POST 请求。
逐个参数说明：
config.chat_completions_url：请求的目标 URL，也就是大模型聊天补全接口的地址，从配置中读取。
headers=headers：把上面构造好的请求头传入请求。
json=payload：httpx 会自动把 Python 字典转换成 JSON 字符串、放到请求正文中，不需要手动调用 json.dumps()，同时自动匹配 UTF-8 编码。
timeout=config.timeout_seconds：设置请求超时时间（单位：秒）。如果超过这个时间还没收到服务端响应，就会抛出超时异常，避免程序一直卡住。
"""

"""三、检查 HTTP 响应状态码（对应 44 行）
python
运行
response.raise_for_status()
这是 httpx 的内置方法，作用是自动校验 HTTP 响应的状态码：
如果状态码是 2xx（比如 200 成功），这行代码不做任何操作，程序继续往下执行。
如果状态码是 4xx（客户端错误，如 401 未授权、403 禁止访问）或 5xx（服务端错误，如 500 服务器内部错误），这行就会抛出 httpx.HTTPStatusError 异常，被后续的异常捕获分支处理。
它的价值是省去了手动写 if response.status_code >= 400 的判断逻辑，简洁且符合通用开发规范。"""

"""
四、解析响应并提取模型回答（对应 58～66 行）
python
运行
try:
    data = response.json()
    content = data["choices"][0]["message"]["content"]
except (ValueError, KeyError, IndexError, TypeError) as exc:
    raise LLMClientError("API响应不是预期的聊天补全JSON结构。") from exc
这一步的核心是从 API 返回的 JSON 结果里，把模型的回答文本取出来，同时做格式容错。
解析 JSON
response.json() 会把服务端返回的 JSON 格式响应体，转换成 Python 的字典 / 列表结构，赋值给 data。如果返回内容不是合法 JSON，会抛出 ValueError。
按层级提取回答
按照大模型接口的标准返回结构，逐层取值：data["choices"][0]["message"]["content"]
data["choices"]：API 返回的候选回答数组，默认情况下只会返回 1 条候选回答。
[0]：取第一条（也是默认唯一的一条）候选结果。
["message"]：候选结果中的消息对象，包含角色和内容。
["content"]：模型生成的回答文本本身。
异常兜底
同时捕获 4 种常见的解析错误：
ValueError：响应不是合法 JSON
KeyError：JSON 中缺少 choices/message/content 等字段
IndexError：choices 是空数组，取第 0 个元素失败
TypeError：中间某一层不是字典 / 列表，无法按键 / 索引取值
只要触发任意一种错误，就统一包装为 LLMClientError 抛出；from exc 是 Python 的异常链语法，会保留原始异常的上下文，方便调试定位根因。
"""