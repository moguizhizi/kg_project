import json
from pathlib import Path
from neo4j import GraphDatabase

from TMMKG.graph.neo4j_db import get_node_schema
from TMMKG.sql_templates import UPSERT_NODE_CYPHER


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "domains"
    / "home_based_user_training"
    / "neo4j_node.json"
)

with open(SCHEMA_PATH, "r") as f:
    ENTITY_TYPE_MAP = json.load(f)["entity_type_map"]


# 读取 entity_dict_sample.json
with open("/home/temp/dataset/attribute_dict_sample.json", "r", encoding="utf-8") as f:
    rows = json.load(f)
    # rows: Dict[str, List[Dict]]
    # {
    #   "Patient": [{...}, {...}],
    #   "Disease": [{...}]
    # }

# 连接 Neo4j
driver = GraphDatabase.driver(
    "bolt://localhost:7687",
    auth=("neo4j", "password"),
)

# 批量写入
with driver.session() as session:
    for label, group in rows.items():
        node_name, _ = get_node_schema(label)
        query = UPSERT_NODE_CYPHER.format(label=node_name)
        session.run(
            query,
            rows=group,
        )

# 关闭连接
driver.close()
