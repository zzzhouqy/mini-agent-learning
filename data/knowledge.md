# Mini Agent 学习知识库

## Pydantic

Pydantic 用于对外部输入进行结构化校验。在 Mini Agent 项目中，模型生成的 tool arguments 会先经过 Pydantic 校验，只有参数类型、字段名和约束都正确时，工具才会被执行。

如果模型传入了错误格式，例如把整数写成字符串，Pydantic 会抛出 ValidationError。Agent 可以把这个错误反馈给模型，并在有限次数内要求模型重新生成参数。

## CSV Tool

CSV Tool 是 Mini Agent 中的一个真实工具。它从固定路径 data/sample.csv 读取数据，不允许模型传入任意文件路径。

get_csv_row 工具接收 row_index 参数，row_index 从 0 开始，表示 CSV 数据行号，不包含表头。工具执行成功后，会返回对应行的字典数据。

## 有限重试

有限重试用于防止 Agent 在参数错误时无限循环。当 Pydantic 校验失败时，validation_failed=True，Agent 会统计失败次数。

当前项目设置最多允许 2 次参数校验失败重试。如果连续失败次数超过限制，Agent 会停止运行并返回错误信息。

## Agent Loop

Agent Loop 是 Agent 的核心运行循环。它会把用户问题发送给模型，模型可能直接回答，也可能请求调用工具。

如果模型请求工具，Agent 会执行工具，并把工具执行结果发送回模型。模型拿到工具结果后，再决定继续调用工具还是给出最终回答。