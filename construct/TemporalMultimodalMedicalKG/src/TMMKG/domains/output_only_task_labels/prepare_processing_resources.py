from TMMKG.utils.xlsx_utils import (
    xlsx_to_parquet_dataset,
)


def main():
    paths = xlsx_to_parquet_dataset(
        input_path="/home/temp/dataset/output_only_task_labels/output_only_task_labels.xlsx",
        output_dir="/home/temp/dataset/output_only_task_labels/parquet",
        overwrite=True,
    )

    print(paths)


if __name__ == "__main__":
    main()
