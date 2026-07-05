import json
from app.config import ConfigError, LLMConfig#从项目的 app.config 模块导入两个对象：
#ConfigError：自定义异常类，代表配置加载失败（比如环境变量缺失、格式错误）。
#LLMConfig：大模型配置类，用于封装 API 密钥、模型名称、接口地址等配置信息，通常包含从环境变量读取配置的方法。

from app.llm_client import LLMClientError, chat#从项目的 app.llm_client 模块导入两个对象：
#LLMClientError：自定义异常类，代表LLM 接口调用失败（比如网络错误、API 返回错误、鉴权失败）。
#chat：核心函数，接收问题文本和配置对象，调用大模型接口并返回回答文本。
def main() -> None:
    prompt = input("请输入问题：").strip()
    if not prompt:
        print("错误：问题不能为空。")
        return

    try:
        config = LLMConfig.from_env()
        answer_text = chat(prompt, config)
    except (ConfigError, LLMClientError) as exc:
        print(f"调用失败：{exc}")
        return

    # 解析JSON的逻辑必须放在main函数内部，在获取answer_text之后
    try:
        answer = json.loads(answer_text)
    except json.JSONDecodeError as exc:
        print(f"JSON解析失败：{exc}")
        return

    print(f"解析后的类型：{type(answer).__name__}")
    print(f"学习主题：{answer['topic']}")
    print(f"学习总结：{answer['summary']}")

if __name__ == "__main__":
    main()