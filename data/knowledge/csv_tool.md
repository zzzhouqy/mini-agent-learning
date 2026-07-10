# CSV Tool

CSV Tool 是 Mini Agent 中的一个真实工具。

它从固定路径 data/sample.csv 读取数据，不允许模型传入任意文件路径。

get_csv_row 工具接收 row_index 参数。row_index 从 0 开始，表示 CSV 数据行号，不包含表头。工具执行成功后，会返回对应行的字典数据。