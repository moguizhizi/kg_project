import argparse
import time
from pathlib import Path
import json

import pyarrow.parquet as pq

from TMMKG.domains.home_based_user_training.table_triple_extractor import (
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
from TMMKG.utils.config import load_config, project_path
from TMMKG.utils.logger import get_logger, setup_logging_from_config
from TMMKG.utils.xlsx_utils import xlsx_to_parquet_dataset

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)

MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
with open(MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

logger = get_logger(__name__)

REQUIRED_IMPORT_COLUMNS = {
    "患者id",
    "训练日期",
    "任务id",
    "任务名称",
    "任务状态",
}


def should_import_sheet(parquet_path: str, required_columns: set[str]) -> bool:
    columns = set(pq.read_schema(parquet_path).names)
    missing_columns = sorted(required_columns - columns)

    if missing_columns:
        logger.info(
            "Skip parquet %s, missing required columns: %s",
            parquet_path,
            ", ".join(missing_columns),
        )
        return False

    return True


def run_home_based_user_training_pipeline(
    uri: str,
    user: str,
    password: str,
    sheet_name: str,
    result_dir: str,
    parquet_dir: str,
    batch_size: int = 50_000,
    resolver: EntityResolver = None,
    ontology_mappings_dir: str | Path | None = None,
    entity_registry_dir: str | Path | None = None,
):
    """
    Home Based User Training 数据处理 + Neo4j 导入 pipeline
    """

    paths = build_pipeline_paths(result_dir, parquet_dir, sheet_name)

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    try:
        # =========================
        # Load mappings
        # =========================
        ontology_mappings_path = (
            Path(ontology_mappings_dir)
            if ontology_mappings_dir
            else ONTOLOGY_MAPPINGS_DIR
        )
        entity_registry_path = (
            Path(entity_registry_dir)
            if entity_registry_dir
            else HOME_BASED_USER_TRAINING
        )

        with open(ontology_mappings_path / "prop2label.json") as f:
            PROP_2_LABEL = json.load(f)

        with open(entity_registry_path / "column_mapping.json") as f:
            COLUMN_MAPPING = json.load(f)

        date_fields = [COLUMN_MAPPING["训练日期"]]

        # =========================
        # Load Parquet
        # =========================
        logger.info("Loading Parquet...")

        load_start = time.perf_counter()

        records = parquet_to_records(
            path=paths["parquet"],
            date_fields=date_fields,
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
    parser = argparse.ArgumentParser(
        description="Create home based user training KG and import facts into Neo4j."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging_from_config(config, "create_HBUT_KG")

    infra = config.get("infra", {})
    neo4j = infra.get("neo4j", {})
    qdrant = infra.get("qdrant", {})
    embedding = config.get("embedding", {})
    resolver_config = config.get("entity_resolver", {})
    ontology = config.get("ontology", {})
    pipeline_config = config.get("pipelines", {}).get("HBUT", {})

    resolver = init_entity_resolver(
        model_name=resolver_config.get("model_name", "Qwen3-Embedding-8B"),
        model_root=embedding.get("model_root"),
        base_collection=resolver_config.get("base_collection", "entity_aliases"),
        qdrant_url=qdrant.get("uri", "http://localhost:6333"),
        score_threshold=resolver_config.get("score_threshold", 0.90),
    )

    xlsx_path = pipeline_config.get(
        "xlsx_path",
        "/home/temp/dataset/home_based_user_training_20260123_v2/home_based_user_training_20260123_v2.xlsx",
    )
    base_result_dir = pipeline_config.get(
        "base_result_dir",
        "/home/temp/dataset/home_based_user_training_20260123_v2",
    )
    base_parquet_dir = pipeline_config.get(
        "base_parquet_dir",
        "/home/temp/dataset/home_based_user_training_20260123_v2/parquet",
    )
    batch_size = pipeline_config.get("batch_size", 50_000)

    logger.info("Converting XLSX to Parquet: %s", xlsx_path)
    parquet_paths = xlsx_to_parquet_dataset(
        input_path=xlsx_path,
        output_dir=base_parquet_dir,
        overwrite=pipeline_config.get("overwrite_parquet", True),
    )

    for sheet_name, parquet_path in parquet_paths.items():
        logger.info("Processing sheet: %s", sheet_name)

        if not should_import_sheet(
            parquet_path=parquet_path,
            required_columns=REQUIRED_IMPORT_COLUMNS,
        ):
            continue

        result_dir = sheet_to_result_dir(sheet_name, base_result_dir)

        run_home_based_user_training_pipeline(
            uri=neo4j.get("uri", "bolt://localhost:7687"),
            user=neo4j.get("user", "neo4j"),
            password=neo4j.get("password", "password"),
            sheet_name=sheet_name,
            result_dir=result_dir,
            parquet_dir=base_parquet_dir,
            batch_size=batch_size,
            resolver=resolver,
            ontology_mappings_dir=project_path(ontology.get("mappings_dir")),
        )
