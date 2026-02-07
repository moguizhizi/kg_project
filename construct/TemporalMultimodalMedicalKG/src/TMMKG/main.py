import os
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
import json

from TMMKG.domains.home_based_user_training.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.xlsx_loader import records_to_xlsx, xlsx_to_records
from TMMKG.graph.neo4j_db import get_node_schema
from TMMKG.infra.neo4j_db import create_neo4j_driver
from TMMKG.services.entity_resolver import EntityResolver, init_entity_resolver
from TMMKG.sql_templates import (
    ATTRIBUTE_FACT_SQL,
    ENTITY_FACT_SQL,
    UPSERT_NODE_CYPHER,
    UPSERT_REL_CYPHER,
)
from TMMKG.utils.json_utils import (
    attribute_df_to_dict,
    entity_df_to_dict,
    iter_duckdb_query_df,
    write_facts_jsonl,
)
from TMMKG.utils.path_utils import build_pipeline_paths, sheet_to_result_dir
from TMMKG.utils.xlsx_utils import get_xlsx_sheetnames
from dotenv import load_dotenv, find_dotenv

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)

MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
with open(MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())


def run_home_based_user_training_pipeline(
    uri: str,
    user: str,
    password: str,
    input_xlsx: str,
    sheet_name: str,
    result_dir: str,
    batch_size: int = 50_000,
    resolver: EntityResolver = None,
):
    """
    Home Based User Training 数据处理 + Neo4j 导入 pipeline
    """

    paths = build_pipeline_paths(result_dir)

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    try:
        # =========================
        # Load mappings
        # =========================
        with open(Path(ONTOLOGY_MAPPINGS_DIR) / "entity_type2label.json") as f:
            ENTITY_TYPE_2_LABEL = json.load(f)

        with open(Path(ONTOLOGY_MAPPINGS_DIR) / "prop2label.json") as f:
            PROP_2_LABEL = json.load(f)

        with open(Path(HOME_BASED_USER_TRAINING) / "column_mapping.json") as f:
            COLUMN_MAPPING = json.load(f)

        date_fields = [COLUMN_MAPPING["训练日期"]]

        # # =========================
        # # Load XLSX
        # # =========================
        # logger.info("Loading XLSX...")

        # load_start = time.perf_counter()

        # records = xlsx_to_records(
        #     path=input_xlsx,
        #     sheet_name=sheet_name,
        #     date_fields=date_fields,
        #     column_mapping=COLUMN_MAPPING,
        # )

        # logger.info(f"Loaded {len(records)} records")
        # logger.info(f"Load cost: {time.perf_counter() - load_start:.2f}s")

        # # =========================
        # # Normalize XLSX
        # # =========================
        # logger.info("Writing normalized XLSX...")

        # records_to_xlsx(records, paths["normalized"])

        # =========================
        # Extract facts
        # =========================
        logger.info("Extracting facts...")

        records = xlsx_to_records(
            path=paths["normalized"],
            sheet_name="records",
        )

        fact_bundle = extract_facts_from_records(records, resolver=resolver)

        write_facts_jsonl(
            path=paths["attr_facts"],
            facts=fact_bundle.attribute_facts,
            mode="overwrite",
        )

        write_facts_jsonl(
            path=paths["entity_facts"],
            facts=fact_bundle.entity_facts,
            mode="overwrite",
        )

        logger.info(
            f"Extracted {len(fact_bundle.attribute_facts)} attribute facts, "
            f"{len(fact_bundle.entity_facts)} entity facts"
        )

        # =========================
        # Import attribute facts
        # =========================
        logger.info("Importing attribute facts into Neo4j...")

        query = ATTRIBUTE_FACT_SQL.format(path=paths["attr_facts"])

        with driver.session() as session:  #
            for i, df_chunk in enumerate(
                iter_duckdb_query_df(
                    query=query,
                    batch_size=batch_size,
                    database=str(paths["duckdb_attr"]),
                ),
                start=1,
            ):
                logger.info(f"[Attr Chunk {i}] shape={df_chunk.shape}")

                attribute_dict = attribute_df_to_dict(df_chunk)

                for label, group in attribute_dict.items():
                    node_name, _ = get_node_schema(label)

                    cypher = UPSERT_NODE_CYPHER.format(label=node_name)

                    session.run(cypher, rows=group)

        # =========================
        # Import entity facts
        # =========================
        logger.info("Importing entity facts into Neo4j...")

        query = ENTITY_FACT_SQL.format(path=paths["entity_facts"])

        with driver.session() as session:  # 再次外提
            for i, df_chunk in enumerate(
                iter_duckdb_query_df(
                    query=query,
                    batch_size=batch_size,
                    database=str(paths["duckdb_entity"]),
                ),
                start=1,
            ):
                logger.info(f"[Entity Chunk {i}] shape={df_chunk.shape}")

                entity_dict = entity_df_to_dict(df_chunk)

                for (h_type, r_type, t_type), rows in entity_dict.items():

                    if r_type not in PROP_2_LABEL:
                        logger.warning(f"Unknown prop: {r_type}, skip")
                        continue

                    h_label, _ = get_node_schema(h_type)
                    t_label, _ = get_node_schema(t_type)

                    cypher = UPSERT_REL_CYPHER.format(
                        h_label=h_label,
                        t_label=t_label,
                        r_name=PROP_2_LABEL[r_type],
                    )

                    session.run(cypher, rows=rows)

        logger.info("Pipeline completed successfully.")

    finally:
        driver.close()


if __name__ == "__main__":

    resolver = init_entity_resolver(
        model_name="Qwen3-Embedding-8B",
        base_collection="entity_aliases",
        qdrant_url="http://localhost:6333",
        score_threshold=0.90,
    )

    xlsx_path = "/home/temp/dataset/home_based_user_training_20260123_v2/home_based_user_training_20260123_v2.xlsx"
    base_result_dir = "/home/temp/dataset/home_based_user_training_20260123_v2"

    sheet_names = get_xlsx_sheetnames(xlsx_path)
    sheet_names = sheet_names[0 : min(19, len(sheet_names))]

    for sheet_name in sheet_names:

        result_dir = sheet_to_result_dir(sheet_name, base_result_dir)

        run_home_based_user_training_pipeline(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            input_xlsx=xlsx_path,
            sheet_name=sheet_name,
            result_dir=result_dir,
            resolver=resolver,
        )
