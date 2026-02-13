import json
from pathlib import Path

from TMMKG.utils.parquet_utils import get_unique_values_per_column


def main():
    parquet_path = (
        "/home/temp/dataset/output_only_task_labels/parquet/overall_label.parquet"
    )
    output_json = "/home/temp/dataset/output_only_task_labels/column_unique_values.json"

    # 获取唯一值
    result = get_unique_values_per_column(parquet_path)

    # 确保目录存在
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)

    # 写入 JSON
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Saved unique values -> {output_json}")


if __name__ == "__main__":
    main()
