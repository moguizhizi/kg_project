"""构建 Home Based User Training 知识图谱。

本模块从 YAML 配置读取 XLSX、Parquet、Neo4j 和实体解析参数，
将原始 XLSX 转换为 Parquet，再按必需列自动筛选有效 sheet，
从 Parquet 读取记录并抽取 attribute facts 与 entity facts，
最后将事实写入 Neo4j。

流程中不再写回或回读 normalized XLSX。
小样本调试时，--limit-records 会限制 XLSX 读取阶段每个 sheet 的行数，
并写入独立的 debug parquet 目录，避免覆盖正式 parquet。
--limit-records 会强制启用 dry-run，只生成 Parquet 和 facts 文件，
不写入 Neo4j，也不初始化依赖 Qdrant 的实体解析器。
调试模式会在找到第一个满足必需列的 sheet 后停止转换，以便快速验证流程。
也可以显式使用 --dry-run 跳过 Neo4j 写入和实体解析器初始化。

Usage:
    完整导入：
        PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py

    小样本调试：
        PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py --limit-records 10

    只生成中间文件，不写入 Neo4j：
        PYTHONPATH=src python src/TMMKG/create_HBUT_KG.py --dry-run
"""

import argparse
import time
from pathlib import Path
import json

import pyarrow.parquet as pq

from TMMKG.domains.home_based_user_training.table_triple_extractor import (
    extract_facts_from_records,
)
from TMMKG.extractors.parquet_loader import parquet_to_records
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
from TMMKG.utils.path_utils import sheet_to_result_dir
from TMMKG.utils.config import load_config, project_path
from TMMKG.utils.logger import get_logger, setup_logging_from_config
from TMMKG.utils.xlsx_utils import xlsx_to_parquet_dataset

BASE_DIR = Path(__file__).resolve().parent
ONTOLOGY_MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"
HOME_BASED_USER_TRAINING = (
    BASE_DIR / "utils" / "entity_registry" / "home_based_user_training"
)

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


def build_hbut_paths(result_dir: Path) -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    prefix = result_dir.name

    return {
        "attr_facts": result_dir / f"{prefix}_attribute_facts.jsonl",
        "entity_facts": result_dir / f"{prefix}_entity_facts.jsonl",
        "duckdb_attr": result_dir / "attribute_facts.duckdb",
        "duckdb_entity": result_dir / "entity_facts.duckdb",
    }


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


def import_entity_facts_to_neo4j(
    driver,
    entity_facts_path: Path,
    duckdb_path: Path,
    batch_size: int,
    prop_to_label: dict,
) -> None:
    query = ENTITY_FACT_SQL.format(path=entity_facts_path)

    with driver.session() as session:
        for i, df_chunk in enumerate(
            iter_duckdb_query_df(
                query=query,
                batch_size=batch_size,
                database=str(duckdb_path),
            ),
            start=1,
        ):
            logger.info("[Entity Chunk %d] shape=%s", i, df_chunk.shape)

            entity_dict = entity_df_to_dict(df_chunk)

            for (h_type, r_type, t_type), rows in entity_dict.items():
                if r_type not in prop_to_label:
                    logger.warning("Unknown prop: %s, skip", r_type)
                    continue

                h_label, _ = get_node_schema(h_type)
                t_label, _ = get_node_schema(t_type)
                cypher = UPSERT_REL_CYPHER.format(
                    h_label=h_label,
                    t_label=t_label,
                    r_name=prop_to_label[r_type],
                )
                session.run(cypher, rows=rows)


def run_home_based_user_training_pipeline(
    uri: str,
    user: str,
    password: str,
    xlsx_path: str,
    result_dir: str,
    parquet_dir: str,
    batch_size: int = 50_000,
    overwrite_parquet: bool = False,
    limit_records: int | None = None,
    dry_run: bool = False,
    resolver: EntityResolver = None,
    ontology_mappings_dir: str | Path | None = None,
    entity_registry_dir: str | Path | None = None,
) -> None:
    """
    Home Based User Training 数据处理 + Neo4j 导入 pipeline。

    流程：XLSX -> Parquet -> facts -> Neo4j。
    """
    ontology_mappings_path = (
        Path(ontology_mappings_dir) if ontology_mappings_dir else ONTOLOGY_MAPPINGS_DIR
    )
    entity_registry_path = (
        Path(entity_registry_dir) if entity_registry_dir else HOME_BASED_USER_TRAINING
    )

    with open(ontology_mappings_path / "prop2label.json") as f:
        prop_to_label = json.load(f)

    with open(entity_registry_path / "column_mapping.json") as f:
        column_mapping = json.load(f)

    date_fields = [column_mapping["训练日期"]]
    parquet_output_dir = Path(parquet_dir)
    parquet_nrows = None

    if limit_records is not None:
        parquet_output_dir = parquet_output_dir / f"debug_limit_{limit_records}"
        parquet_nrows = limit_records
        logger.info(
            "Debug mode enabled: convert first %d rows per sheet into %s",
            limit_records,
            parquet_output_dir,
        )

    logger.info("Converting XLSX to Parquet: %s", xlsx_path)
    parquet_paths = xlsx_to_parquet_dataset(
        input_path=xlsx_path,
        output_dir=parquet_output_dir,
        overwrite=overwrite_parquet,
        nrows=parquet_nrows,
        required_columns=REQUIRED_IMPORT_COLUMNS,
        stop_after_first_valid=limit_records is not None,
    )

    driver = (
        None
        if dry_run
        else create_neo4j_driver(uri=uri, user=user, password=password)
    )

    try:
        for sheet_name, parquet_path in parquet_paths.items():
            logger.info("Processing sheet: %s", sheet_name)

            if not should_import_sheet(
                parquet_path=parquet_path,
                required_columns=REQUIRED_IMPORT_COLUMNS,
            ):
                continue

            sheet_result_dir = sheet_to_result_dir(sheet_name, result_dir)
            paths = build_hbut_paths(sheet_result_dir)

            load_start = time.perf_counter()
            records = parquet_to_records(
                path=parquet_path,
                date_fields=date_fields,
                column_mapping=column_mapping,
            )

            if limit_records is not None:
                logger.info("Test mode enabled: using first %d records", limit_records)
                records = records[:limit_records]

            logger.info("Loaded %d records", len(records))
            logger.info("Load cost: %.2fs", time.perf_counter() - load_start)

            logger.info("Extracting facts...")
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
                "Extracted %d attribute facts, %d entity facts",
                len(fact_bundle.attribute_facts),
                len(fact_bundle.entity_facts),
            )

            if dry_run:
                logger.info(
                    "Dry run enabled: skip Neo4j import for sheet %s. "
                    "Facts were written to %s and %s",
                    sheet_name,
                    paths["attr_facts"],
                    paths["entity_facts"],
                )
                continue

            logger.info("Importing attribute facts into Neo4j...")
            import_attribute_facts_to_neo4j(
                driver=driver,
                attr_facts_path=paths["attr_facts"],
                duckdb_path=paths["duckdb_attr"],
                batch_size=batch_size,
            )

            logger.info("Importing entity facts into Neo4j...")
            import_entity_facts_to_neo4j(
                driver=driver,
                entity_facts_path=paths["entity_facts"],
                duckdb_path=paths["duckdb_entity"],
                batch_size=batch_size,
                prop_to_label=prop_to_label,
            )

        logger.info("Home based user training pipeline completed successfully.")

    finally:
        if driver is not None:
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
    parser.add_argument(
        "--limit-records",
        type=int,
        default=None,
        help="Only process the first N records per sheet for testing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate parquet and facts files without importing into Neo4j.",
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
    dry_run = args.dry_run or args.limit_records is not None

    if args.limit_records is not None and not args.dry_run:
        logger.info("--limit-records was provided; dry-run is enabled automatically.")

    resolver = None
    if dry_run:
        logger.info("Dry-run enabled: skip EntityResolver initialization.")
    else:
        resolver = init_entity_resolver(
            model_name=resolver_config.get("model_name", "Qwen3-Embedding-8B"),
            model_root=embedding.get("model_root"),
            base_collection=resolver_config.get("base_collection", "entity_aliases"),
            qdrant_url=qdrant.get("uri", "http://localhost:6333"),
            score_threshold=resolver_config.get("score_threshold", 0.90),
        )

    run_home_based_user_training_pipeline(
        uri=neo4j.get("uri", "bolt://localhost:7687"),
        user=neo4j.get("user", "neo4j"),
        password=neo4j.get("password", "password"),
        xlsx_path=pipeline_config.get(
            "xlsx_path",
            "/home/temp/dataset/home_based_user_training_20260123_v2/home_based_user_training_20260123_v2.xlsx",
        ),
        result_dir=pipeline_config.get(
            "base_result_dir",
            "/home/temp/dataset/home_based_user_training_20260123_v2",
        ),
        parquet_dir=pipeline_config.get(
            "base_parquet_dir",
            "/home/temp/dataset/home_based_user_training_20260123_v2/parquet",
        ),
        batch_size=pipeline_config.get("batch_size", 50_000),
        overwrite_parquet=pipeline_config.get("overwrite_parquet", True),
        limit_records=args.limit_records,
        dry_run=dry_run,
        resolver=resolver,
        ontology_mappings_dir=project_path(ontology.get("mappings_dir")),
    )
