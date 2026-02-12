from TMMKG.utils.xlsx_utils import xlsx_to_parquet_dataset


if __name__ == "__main__":

    paths = xlsx_to_parquet_dataset(
        input_path="/home/temp/dataset/home_based_user_training_20260123_v2/home_based_user_training_20260123_v2.xlsx",
        output_dir="/home/temp/dataset/home_based_user_training_20260123_v2/parquet",
    )

    print(paths)
