# JSON Format

## JSON 语法

JSON 是一种文本数据交换格式，可以表示对象、数组、字符串、数字、布尔值和空值。

在 Python 中，json.loads() 可以把合法的 JSON 字符串转换为字典、列表等 Python 对象。如果字符串不符合 JSON 语法，它会抛出 JSONDecodeError。

## 解析边界

JSON 解析只负责判断文本语法是否合法，并把文本转换为 Python 数据。

JSON 语法正确不代表字段名、字段类型和业务约束正确。例如 {"row_index": "1"} 是合法 JSON，但字符串 "1" 不一定符合工具所要求的整数类型。