import os
import logging
import time
import sys
from datetime import datetime
from pathlib import Path
import json


from TMMKG.domains.output_only_task_labels.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.parquet_loader import parquet_to_records
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
    BASE_DIR / "utils" / "entity_registry" / "output_only_task_labels"
)

MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
with open(MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())


def run_output_only_task_labels_pipeline(
    uri: str,
    user: str,
    password: str,
    sheet_name: str,
    result_dir: str,
    parquet_dir: str,
    batch_size: int = 50_000,
    resolver: EntityResolver = None,
):
    """
    Output only task labels 数据处理 + Neo4j 导入 pipeline
    """

    paths = build_pipeline_paths(result_dir, parquet_dir, sheet_name)

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    try:
        # =========================
        # Load mappings
        # =========================
        with open(Path(ONTOLOGY_MAPPINGS_DIR) / "prop2label.json") as f:
            PROP_2_LABEL = json.load(f)

        with open(Path(HOME_BASED_USER_TRAINING) / "column_mapping.json") as f:
            COLUMN_MAPPING = json.load(f)

        # =========================
        # Load Parquet
        # =========================
        logger.info("Loading Parquet...")

        load_start = time.perf_counter()

        records = parquet_to_records(
            path=paths["parquet"],
            column_mapping=COLUMN_MAPPING,
        )

        logger.info(f"Loaded {len(records)} records")
        logger.info(f"Load cost: {time.perf_counter() - load_start:.2f}s")

        # =========================
        # Normalize XLSX
        # =========================
        logger.info("Writing normalized XLSX...")

        records_to_xlsx(records, paths["normalized"])

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

    xlsx_path = (
        "/home/temp/dataset/output_only_task_labels/output_only_task_labels.xlsx"
    )
    base_result_dir = "/home/temp/dataset/output_only_task_labels"
    base_parquet_dir = "/home/temp/dataset/output_only_task_labels/parquet"

    sheet_names = get_xlsx_sheetnames(xlsx_path)
    sheet_names = sheet_names[0 : min(1, len(sheet_names))]

    for sheet_name in sheet_names:

        result_dir = sheet_to_result_dir(sheet_name, base_result_dir)

        run_output_only_task_labels_pipeline(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
            sheet_name=sheet_name,
            result_dir=result_dir,
            parquet_dir=base_parquet_dir,
            resolver=resolver,
        )
