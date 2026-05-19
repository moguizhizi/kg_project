"""将实体注册库写入 Neo4j。

本模块从 YAML 配置读取 Neo4j 和实体注册映射目录，将 Disease、
Symptom、Unknown 的标准名称和别名写入 Neo4j。

该流程应在 HBUT/L2BA/OOTL 等业务 KG 导入前执行一次，用于确保
疾病、症状和未知实体节点具备 id、name 和别名属性。

Usage:
    PYTHONPATH=src python -m TMMKG.create_tmmkg_entity_graph

    或：
        PYTHONPATH=src python src/TMMKG/create_tmmkg_entity_graph.py
"""

import argparse
import json
from pathlib import Path
from typing import Iterable

from TMMKG.graph.neo4j_db import ensure_unique_constraints, get_node_schema
from TMMKG.infra.neo4j_db import create_neo4j_driver
from TMMKG.sql_templates import UPSERT_NODE_CYPHER
from TMMKG.utils.config import load_config, project_path
from TMMKG.utils.logger import get_logger, setup_logging_from_config

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_ENTITY_REGISTRY_DIR = BASE_DIR / "utils" / "entity_registry"

logger = get_logger(__name__)

ENTITY_REGISTRY_FILES = {
    "AU_Q0013": ("disease2label.json", "disease2aliases.json"),
    "AU_Q0040": ("symptom2label.json", "symptom2aliases.json"),
    "AU_Q0041": ("unknown2label.json", "unknown2aliases.json"),
}


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def iter_batches(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_entity_rows(label_map: dict, alias_map: dict) -> list[dict]:
    rows = []

    for entity_id, name in label_map.items():
        if not name:
            continue

        row = {
            "id": str(entity_id),
            "name": name,
        }

        aliases = alias_map.get(entity_id, [])
        if aliases:
            row["别名"] = ",".join(aliases)

        rows.append(row)

    return rows


def load_entity_registry(
    mappings_dir: str | Path | None = None,
) -> dict[str, list[dict]]:
    mappings_path = Path(mappings_dir) if mappings_dir else DEFAULT_ENTITY_REGISTRY_DIR
    registry = {}

    for entity_type, (label_file, alias_file) in ENTITY_REGISTRY_FILES.items():
        label_map = load_json(mappings_path / label_file)
        alias_map = load_json(mappings_path / alias_file)
        registry[entity_type] = build_entity_rows(label_map, alias_map)

    return registry


def upsert_entity_registry_to_neo4j(
    driver,
    mappings_dir: str | Path | None = None,
    batch_size: int = 50_000,
) -> None:
    registry = load_entity_registry(mappings_dir)

    logger.info("Ensuring Neo4j unique constraints...")
    ensure_unique_constraints(driver)

    with driver.session() as session:
        for entity_type, rows in registry.items():
            node_label, _ = get_node_schema(entity_type)
            cypher = UPSERT_NODE_CYPHER.format(label=node_label)

            logger.info(
                "Upserting entity registry | entity_type=%s | label=%s | rows=%d",
                entity_type,
                node_label,
                len(rows),
            )

            for i, batch in enumerate(iter_batches(rows, batch_size), start=1):
                logger.info(
                    "[Entity Registry Chunk %d] label=%s shape=(%d)",
                    i,
                    node_label,
                    len(batch),
                )
                session.run(cypher, rows=batch)

    logger.info("Entity registry graph import completed successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Disease/Symptom/Unknown registry nodes into Neo4j."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50_000,
        help="Neo4j upsert batch size.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging_from_config(config, "create_tmmkg_entity_graph")

    infra = config.get("infra", {})
    neo4j = infra.get("neo4j", {})
    entity_registry = config.get("entity_registry", {})

    driver = create_neo4j_driver(
        uri=neo4j.get("uri", "bolt://localhost:7687"),
        user=neo4j.get("user", "neo4j"),
        password=neo4j.get("password", "password"),
    )

    try:
        upsert_entity_registry_to_neo4j(
            driver=driver,
            mappings_dir=project_path(entity_registry.get("mappings_dir")),
            batch_size=args.batch_size,
        )
    finally:
        driver.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        raise
