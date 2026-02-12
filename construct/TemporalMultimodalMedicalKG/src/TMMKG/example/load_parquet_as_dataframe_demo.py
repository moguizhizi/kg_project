from TMMKG.extractors.parquet_loader import load_parquet_as_dataframe


if __name__ == "__main__":

    load_parquet_as_dataframe(
        parquet_path="/home/temp/dataset/home_based_user_training_20260123_v2/parquet/result_1.parquet",
    )
