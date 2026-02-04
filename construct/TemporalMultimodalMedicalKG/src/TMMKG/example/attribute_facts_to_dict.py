import json
import logging
from TMMKG.sql_templates import ATTRIBUTE_FACT_SQL
from TMMKG.utils.json_utils import attribute_df_to_entity_dict, iter_duckdb_query_df
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    path = "/home/temp/dataset/attribute_facts.jsonl"
    sample_json_path = "/home/temp/dataset/entity_dict_sample.json"

    if not os.path.exists(path):
        logger.error(f"Attribute facts file does not exist: {path}")
        return

    logger.info(f"Starting to process attribute facts from: {path}")
    first = True

    query = ATTRIBUTE_FACT_SQL.format(path=path)

    chunk_count = 0
    for df_chunk in iter_duckdb_query_df(
        query=query, batch_size=100, database="facts.duckdb"
    ):
        chunk_count += 1
        logger.info(
            f"Processing chunk #{chunk_count} with shape {df_chunk.shape} "
            f"and columns {df_chunk.columns.tolist()}"
        )

        entity_dict = attribute_df_to_entity_dict(df_chunk)

        if first:
            with open(sample_json_path, "w", encoding="utf-8") as f:
                json.dump(entity_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved the first entity_dict sample to {sample_json_path}")
            first = False

        # 调试前两个实体
        for k, v in entity_dict.items():
            logger.debug(f"{k}: {v[:2]}")  # 每个类型看前两个实体

    logger.info(f"Finished processing {chunk_count} chunks from attribute facts.")


if __name__ == "__main__":
    main()
