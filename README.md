# Mini Agent Learning

记录大模型应用开发的学习过程。

## 第一阶段目标

- LLM API
- Tool Calling
- Pydantic参数校验
- FastAPI接口
- pytest测试
- 权限与Hook

## 当前进度

- [x] 创建Python项目骨架
- [x] 配置Git与GitHub同步
- [x] 使用环境变量保存LLM配置
- [x] 使用HTTP调用OpenAI兼容的聊天补全API
- [x] 处理Key缺失、超时、网络异常和HTTP错误状态

## 本地运行

创建并激活虚拟环境后安装依赖：

```bash
python -m pip install -r requirements.txt
```

复制配置模板：

```bash
cp .env.example .env
```

在`.env`中填写模型服务商提供的API Key、Base URL和模型名称，然后运行：

```bash
python -m app.main
```

> `.env`已加入`.gitignore`，不要把真实API Key提交到GitHub。
