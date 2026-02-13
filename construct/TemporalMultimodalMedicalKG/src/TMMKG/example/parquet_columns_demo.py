from TMMKG.utils.parquet_utils import get_parquet_columns


def main():
    columns_name = get_parquet_columns(
        parquet_path="/home/temp/dataset/output_only_task_labels/parquet/overall_label.parquet"
    )

    for col in columns_name:
        print(col)


if __name__ == "__main__":
    main()
