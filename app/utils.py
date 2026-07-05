def add(a: int, b: int) -> int:
    return a + b
#     """目前只有你的 Python 代码知道它，DeepSeek 并不知道项目中存在这个函数。
# 要将它变成模型能够选择的 Tool，需要三部分：
# Python函数：真正执行加法
# 工具Schema：向模型说明工具名称、用途和参数
# 调度代码：收到模型请求后调用对应Python函数
# 最重要的认识是：
# 模型不会真正执行 Python 函数；模型只会返回“请调用 add，并传入这些参数”，然后由我们的代码执行。“”“