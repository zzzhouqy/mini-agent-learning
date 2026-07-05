import os
from dataclasses import dataclass

from dotenv import load_dotenv


class ConfigError(ValueError):
    """配置缺失或格式错误。"""


@dataclass(frozen=True)
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    timeout_seconds: float

    @property
    def chat_completions_url(self) -> str:
        return f"{self.base_url.rstrip('/')}/chat/completions"

    @classmethod
    def from_env(cls) -> "LLMConfig":
        load_dotenv()

        api_key = os.getenv("LLM_API_KEY", "").strip()
        if not api_key:
            raise ConfigError(
                "缺少 LLM_API_KEY。请复制 .env.example 为 .env，并填写真实密钥。"
            )

        base_url = os.getenv("LLM_BASE_URL", "").strip()
        if not base_url:
            raise ConfigError("缺少 LLM_BASE_URL。")

        model = os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise ConfigError("缺少 LLM_MODEL。")

        timeout_text = os.getenv("LLM_TIMEOUT_SECONDS", "30").strip()
        try:
            timeout_seconds = float(timeout_text)
        except ValueError as exc:
            raise ConfigError("LLM_TIMEOUT_SECONDS 必须是数字。") from exc

        if timeout_seconds <= 0:
            raise ConfigError("LLM_TIMEOUT_SECONDS 必须大于 0。")

        return cls(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
        )
#第 24 行：加载 .env
#第 26～38 行：读取并检查 Key、URL、模型名
#第 18～20 行：拼接完整 API 地址