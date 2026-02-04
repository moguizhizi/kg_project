import os
import json
import logging
from pathlib import Path

from neo4j import GraphDatabase

from TMMKG.sql_templates import ENTITY_FACT_SQL, UPSERT_REL_CYPHER
from TMMKG.utils.json_utils import entity_df_to_dict, iter_duckdb_query_df
from TMMKG.graph.neo4j_db import get_node_schema


# =========================
# 基础配置
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent
MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"

ENTITY_FACTS_PATH = "/home/temp/dataset/entity_facts.jsonl"
DUCKDB_PATH = "entity_facts.duckdb"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"


# =========================
# 日志
# =========================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================
# 加载映射
# =========================

with open(MAPPINGS_DIR / "prop2label.json", "r") as f:
    PROP_2_LABEL = json.load(f)

# =========================
# 主流程
# =========================


def main():
    driver = GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
    )

    try:
        if not os.path.exists(ENTITY_FACTS_PATH):
            raise FileNotFoundError(ENTITY_FACTS_PATH)

        query = ENTITY_FACT_SQL.format(path=ENTITY_FACTS_PATH)

        chunk_idx = 0
        for df_chunk in iter_duckdb_query_df(
            query=query,
            batch_size=100,
            database="entity_facts.duckdb",
        ):
            chunk_idx += 1
            logger.info(
                f"[Chunk {chunk_idx}] "
                f"shape={df_chunk.shape}, "
                f"columns={df_chunk.columns.tolist()}"
            )

            entity_dict = entity_df_to_dict(df_chunk)

            for i, ((h_type, r_type, t_type), group_rows) in enumerate(
                entity_dict.items()
            ):
                h_label, _ = get_node_schema(h_type)
                t_label, _ = get_node_schema(t_type)

                if r_type not in PROP_2_LABEL:
                    logger.warning(f"Unknown prop: {r_type}, skip")
                    continue

                r_name = PROP_2_LABEL[r_type]

                logger.info(
                    f"  [Group {i + 1}] "
                    f"({h_type})-[:{r_type}]->({t_type}), "
                    f"size={len(group_rows)}"
                )

                query = UPSERT_REL_CYPHER.format(
                    h_label=h_label,
                    t_label=t_label,
                    r_name=r_name,
                )

                with driver.session() as session:
                    session.run(query, rows=group_rows)

                # 示例：打印前 2 条
                for row in group_rows[:2]:
                    print(row)

            # demo：只跑前 3 个 chunk
            if chunk_idx >= 2:
                break

        logger.info("实体关系导入完成")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
