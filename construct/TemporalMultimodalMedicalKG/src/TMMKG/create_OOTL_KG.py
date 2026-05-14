"""构建 Output Only Task Labels 知识图谱。

本模块从 YAML 配置读取 XLSX、Parquet 和 Neo4j 参数，
将原始 XLSX 转换为 Parquet，再从 Parquet 读取记录并抽取
attribute facts，最后将属性事实写入 Neo4j。

当前流程只处理 attribute facts，不生成或导入 entity facts。
可通过 --limit-records 限制每个 sheet 的记录数，用于小样本调试。

Usage:
    完整导入：
        PYTHONPATH=src python src/TMMKG/create_OOTL_KG.py

    小样本调试：
        PYTHONPATH=src python src/TMMKG/create_OOTL_KG.py --limit-records 10
"""

import argparse
import time
from pathlib import Path
import json

import pyarrow.parquet as pq

from TMMKG.domains.output_only_task_labels.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.parquet_loader import parquet_to_records
from TMMKG.graph.neo4j_db import ensure_unique_constraints, get_node_schema
from TMMKG.infra.neo4j_db import create_neo4j_driver
from TMMKG.sql_templates import (
    ATTRIBUTE_FACT_SQL,
    UPSERT_NODE_CYPHER,
)
from TMMKG.utils.json_utils import (
    attribute_df_to_dict,
    iter_duckdb_query_df,
    write_facts_jsonl,
)
from TMMKG.utils.path_utils import build_pipeline_paths, sheet_to_result_dir
from TMMKG.utils.config import load_config
from TMMKG.utils.logger import get_logger, setup_logging_from_config
from TMMKG.utils.xlsx_utils import xlsx_to_parquet_dataset

BASE_DIR = Path(__file__).resolve().parent
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "output_only_task_labels"
)

logger = get_logger(__name__)

REQUIRED_IMPORT_COLUMNS = {
    "CMS-ID",
    "训练名称",
    "认知加工深度",
    "认知负荷水平",
    "任务类型",
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


def run_output_only_task_labels_pipeline(
    uri: str,
    user: str,
    password: str,
    xlsx_path: str,
    base_result_dir: str,
    parquet_dir: str,
    batch_size: int = 50_000,
    overwrite_parquet: bool = False,
    limit_records: int | None = None,
    entity_registry_dir: str | Path | None = None,
):
    """
    Output only task labels 数据处理 + Neo4j 导入 pipeline。

    流程：XLSX -> Parquet -> attribute facts -> Neo4j。
    """
    entity_registry_path = (
        Path(entity_registry_dir) if entity_registry_dir else HOME_BASED_USER_TRAINING
    )

    with open(entity_registry_path / "column_mapping.json") as f:
        COLUMN_MAPPING = json.load(f)

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

            result_dir = sheet_to_result_dir(sheet_name, base_result_dir)
            paths = build_pipeline_paths(result_dir, parquet_dir, sheet_name)

            logger.info("Loading Parquet...")
            load_start = time.perf_counter()

            records = parquet_to_records(
                path=parquet_path,
                column_mapping=COLUMN_MAPPING,
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

            logger.info("Extracted %d attribute facts", len(fact_bundle.attribute_facts))

            logger.info("Importing attribute facts into Neo4j...")
            query = ATTRIBUTE_FACT_SQL.format(path=paths["attr_facts"])

            with driver.session() as session:
                for i, df_chunk in enumerate(
                    iter_duckdb_query_df(
                        query=query,
                        batch_size=batch_size,
                        database=str(paths["duckdb_attr"]),
                    ),
                    start=1,
                ):
                    logger.info("[Attr Chunk %d] shape=%s", i, df_chunk.shape)

                    attribute_dict = attribute_df_to_dict(df_chunk)

                    for label, group in attribute_dict.items():
                        node_name, _ = get_node_schema(label)
                        cypher = UPSERT_NODE_CYPHER.format(label=node_name)
                        session.run(cypher, rows=group)

        logger.info("Pipeline completed successfully.")

    finally:
        driver.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create output only task labels KG and import facts into Neo4j."
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
        help="Only process the first N records per sheet for testing.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging_from_config(config, "create_OOTL_KG")

    infra = config.get("infra", {})
    neo4j = infra.get("neo4j", {})
    pipeline_config = config.get("pipelines", {}).get("OOTL", {})

    run_output_only_task_labels_pipeline(
        uri=neo4j.get("uri", "bolt://localhost:7687"),
        user=neo4j.get("user", "neo4j"),
        password=neo4j.get("password", "password"),
        xlsx_path=pipeline_config.get(
            "xlsx_path",
            "/home/temp/dataset/output_only_task_labels/output_only_task_labels.xlsx",
        ),
        base_result_dir=pipeline_config.get(
            "base_result_dir",
            "/home/temp/dataset/output_only_task_labels",
        ),
        parquet_dir=pipeline_config.get(
            "base_parquet_dir",
            "/home/temp/dataset/output_only_task_labels/parquet",
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
