from TMMKG.utils.xlsx_utils import (
    xlsx_to_parquet_dataset,
)


def main():
    paths = xlsx_to_parquet_dataset(
        input_path="/home/temp/dataset/output_only_task_labels/output_only_task_labels.xlsx",
        output_dir="/home/temp/dataset/output_only_task_labels/parquet",
        overwrite=True,
        multi_label_keywords=["核心内容颜色​", "主要颜色", "辅助颜色"],
    )

    print(paths)


if __name__ == "__main__":
    main()
