"""构建 Level 2 Brain Ability 知识图谱。

本模块从源 XLSX 生成 Parquet 数据集，再从 Parquet 读取记录并抽取
attribute facts，最后将属性事实写入 Neo4j。

Usage:
    完整导入：
        PYTHONPATH=src python src/TMMKG/create_L2BA_KG.py

    小样本调试：
        PYTHONPATH=src python src/TMMKG/create_L2BA_KG.py --limit-records 10

--limit-records 是运行时调试参数，不改变主流程，只限制每个 sheet
处理的前 N 条记录，便于在完整导入前检查抽取和入库结果。
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from TMMKG.domains.level_2_brain_ability_data.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.parquet_loader import parquet_to_records
from TMMKG.graph.neo4j_db import ensure_unique_constraints, get_node_schema
from TMMKG.infra.neo4j_db import create_neo4j_driver
from TMMKG.sql_templates import ATTRIBUTE_FACT_SQL, UPSERT_NODE_CYPHER
from TMMKG.utils.json_utils import (
    attribute_df_to_dict,
    iter_duckdb_query_df,
    write_facts_jsonl,
)
from TMMKG.utils.config import load_config
from TMMKG.utils.logger import get_logger, setup_logging_from_config
from TMMKG.utils.path_utils import sheet_to_result_dir
from TMMKG.utils.xlsx_utils import xlsx_to_parquet_dataset

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
ENTITY_REGISTRY_DIR = (
    BASE_DIR / "utils" / "entity_registry" / "level_2_brain_ability_data"
)

logger = get_logger(__name__)

with open(ONTOLOGY_MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

REQUIRED_IMPORT_COLUMNS = {
    "患者id",
    "训练日期",
    "二级_心算",
    "二级_工作记忆",
}


def build_l2ba_paths(result_dir: Path) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = result_dir.name

    return {
        "attr_facts": result_dir / f"{prefix}_attribute_facts.jsonl",
        "duckdb_attr": result_dir / "attribute_facts.duckdb",
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


def import_attribute_facts_to_neo4j(
    driver,
    attr_facts_path: Path,
    duckdb_path: Path,
    batch_size: int,
) -> None:
    query = ATTRIBUTE_FACT_SQL.format(path=attr_facts_path)

    with driver.session() as session:
        for i, df_chunk in enumerate(
            iter_duckdb_query_df(
                query=query,
                batch_size=batch_size,
                database=str(duckdb_path),
            ),
            start=1,
        ):
            logger.info("[Attr Chunk %d] shape=%s", i, df_chunk.shape)

            attribute_dict = attribute_df_to_dict(df_chunk)

            for label, group in attribute_dict.items():
                node_name, _ = get_node_schema(label)
                cypher = UPSERT_NODE_CYPHER.format(label=node_name)
                session.run(cypher, rows=group)


def run_level_2_brain_ability_pipeline(
    uri: str,
    user: str,
    password: str,
    xlsx_path: str,
    result_dir: str,
    parquet_dir: str,
    batch_size: int = 50_000,
    overwrite_parquet: bool = False,
    limit_records: int | None = None,
) -> None:
    """
    Level 2 brain ability data pipeline:
    XLSX -> Parquet -> attribute facts -> Neo4j.
    """

    with open(ENTITY_REGISTRY_DIR / "column_mapping.json", "r") as f:
        column_mapping = json.load(f)

    date_fields = [column_mapping["训练日期"]]

    logger.info("Converting XLSX to Parquet: %s", xlsx_path)
    parquet_paths = xlsx_to_parquet_dataset(
        input_path=xlsx_path,
        output_dir=parquet_dir,
        overwrite=overwrite_parquet,
    )

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    try:
        logger.info("Ensuring Neo4j unique constraints...")
        ensure_unique_constraints(driver)

        for sheet_name, parquet_path in parquet_paths.items():
            logger.info("Processing sheet: %s", sheet_name)

            if not should_import_sheet(
                parquet_path=parquet_path,
                required_columns=REQUIRED_IMPORT_COLUMNS,
            ):
                continue

            sheet_result_dir = sheet_to_result_dir(sheet_name, result_dir)
            paths = build_l2ba_paths(sheet_result_dir)

            load_start = time.perf_counter()
            records = parquet_to_records(
                path=parquet_path,
                column_mapping=column_mapping,
                date_fields=date_fields,
            )

            if limit_records is not None:
                logger.info("Test mode enabled: using first %d records", limit_records)
                records = records[:limit_records]

            logger.info("Loaded %d records", len(records))
            logger.info("Load cost: %.2fs", time.perf_counter() - load_start)

            logger.info("Extracting attribute facts...")
            fact_bundle = extract_facts_from_records(records)

            write_facts_jsonl(
                path=paths["attr_facts"],
                facts=fact_bundle.attribute_facts,
                mode="overwrite",
            )

            logger.info(
                "Extracted %d attribute facts, %d entity facts",
                len(fact_bundle.attribute_facts),
                len(fact_bundle.entity_facts),
            )

            logger.info("Importing attribute facts into Neo4j...")
            import_attribute_facts_to_neo4j(
                driver=driver,
                attr_facts_path=paths["attr_facts"],
                duckdb_path=paths["duckdb_attr"],
                batch_size=batch_size,
            )

        logger.info("Level 2 brain ability pipeline completed successfully.")

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create level 2 brain ability KG attribute facts and import them into Neo4j."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Only process the first N records for testing. Defaults to full import.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging_from_config(config, "create_L2BA_KG")

    infra = config.get("infra", {})
    neo4j = infra.get("neo4j", {})
    pipeline_config = config.get("pipelines", {}).get("L2BA", {})

    run_level_2_brain_ability_pipeline(
        uri=neo4j.get("uri", "bolt://localhost:7687"),
        user=neo4j.get("user", "neo4j"),
        password=neo4j.get("password", "password"),
        xlsx_path=pipeline_config.get(
            "xlsx_path",
            "/home/temp/dataset/level_2_brain_ability_data/level_2_brain_ability_data_20260509.xlsx",
        ),
        result_dir=pipeline_config.get(
            "result_dir",
            "/home/temp/dataset/level_2_brain_ability_data",
        ),
        parquet_dir=pipeline_config.get(
            "parquet_dir",
            "/home/temp/dataset/level_2_brain_ability_data/parquet",
        ),
        batch_size=pipeline_config.get("batch_size", 50_000),
        overwrite_parquet=pipeline_config.get("overwrite_parquet", True),
        limit_records=args.limit_records,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        raise
