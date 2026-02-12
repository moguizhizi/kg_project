from TMMKG.utils.xlsx_utils import (
    get_xlsx_sheetnames,
    load_unique_column,
    xlsx_to_parquet_dataset,
)


def generate_disease_entity_mapping(xlsx_path: str):
    print(xlsx_path)
    sheet_names = get_xlsx_sheetnames(xlsx_path)
    sheet_names = sheet_names[:1]

    for sheet_name in sheet_names:
        diseases = load_unique_column(
            path=xlsx_path, sheet_name=sheet_name, column_name="AU_P0019", as_list=True
        )

        diseases = ",".join(diseases)

        print(diseases)


def main():
    generate_disease_entity_mapping(
        "/home/temp/dataset/home_based_user_training_20260123_v2/result19/result19.normalized.xlsx"
    )

    paths = xlsx_to_parquet_dataset(
        input_path="/home/temp/dataset/home_based_user_training_20260123_v2/home_based_user_training_20260123_v2.xlsx",
        output_dir="/home/temp/dataset/home_based_user_training_20260123_v2/parquet",
    )

    print(paths)


if __name__ == "__main__":
    main()
