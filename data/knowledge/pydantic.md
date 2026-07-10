# Pydantic

Pydantic 用于对外部输入进行结构化校验。

在 Mini Agent 项目中，模型生成的 tool arguments 会先经过 Pydantic 校验。只有参数类型、字段名和约束都正确时，工具才会被执行。

如果模型传入了错误格式，例如把整数写成字符串，Pydantic 会抛出 ValidationError。Agent 可以把这个错误反馈给模型，并在有限次数内要求模型重新生成参数。