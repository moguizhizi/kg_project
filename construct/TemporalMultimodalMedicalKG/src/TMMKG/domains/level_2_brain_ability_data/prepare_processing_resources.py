from TMMKG.utils.xlsx_utils import (
    xlsx_to_parquet_dataset,
)


def main():
    paths = xlsx_to_parquet_dataset(
        input_path="/home/temp/dataset/level_2_brain_ability_data/level_2_brain_ability_data_20260509.xlsx",
        output_dir="/home/temp/dataset/level_2_brain_ability_data/parquet",
        overwrite=True,
    )

    print(paths)


if __name__ == "__main__":
    main()
