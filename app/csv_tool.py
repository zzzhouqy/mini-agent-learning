import csv
from pathlib import Path


CSV_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sample.csv"
)


def get_csv_row(row_index: int) -> dict[str, str]:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(
            f"CSV 文件不存在：{CSV_PATH}"
        )

    with CSV_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
                rows = list(csv.DictReader(file))

    if not rows:
        raise ValueError(
            f"CSV 文件没有数据行：{CSV_PATH}"
        )

    if row_index < 0 or row_index >= len(rows):
        raise IndexError(
            f"row_index 超出范围：{row_index}，"
            f"当前共有 {len(rows)} 行数据。"
        )

    return rows[row_index]