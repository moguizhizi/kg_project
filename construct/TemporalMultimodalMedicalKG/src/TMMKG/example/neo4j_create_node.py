"""
create_single_node.py

用于测试 / 手动创建 Neo4j 节点
"""

from pathlib import Path
import json

from TMMKG.graph.neo4j_db import build_merge_node_cypher
from TMMKG.infra.neo4j_db import create_neo4j_driver

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "domains"
    / "home_based_user_training"
    / "neo4j_node.json"
)

with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    ENTITY_TYPE_MAP = json.load(f)["entity_type_map"]


def main():

    # --- Neo4j 连接 ---
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    driver = create_neo4j_driver(
        uri=uri,
        user=user,
        password=password,
    )

    try:
        entity_type = "AU_Q0004"
        entity_id = "1000000"

        # （可选）schema 校验
        if entity_type not in ENTITY_TYPE_MAP:
            raise ValueError(f"Unknown entity_type: {entity_type}")

        properties = {
            "gender": "女",
            "age": 2,
        }

        cypher, params = build_merge_node_cypher(
            entity_type=entity_type,
            entity_id=entity_id,
            properties=properties,
        )

        logging.info("Generated Cypher:")
        logging.info(cypher)
        logging.info("\nParams:")
        logging.info(params)

        with driver.session() as session:
            session.run(cypher, params)

        logging.info("Node created successfully.")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
