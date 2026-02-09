# graph/neo4j.py

import json
from pathlib import Path

from TMMKG.sql_templates import (
    COUNT_NODES_IN_IDS_CYPHER,
    CREATE_CONSTRAINT_CYPHER,
    FETCH_NODE_IDS_CYPHER,
)


SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "domains"
    / "home_based_user_training"
    / "neo4j_node.json"
)

with open(SCHEMA_PATH, "r") as f:
    ENTITY_TYPE_MAP = json.load(f)["entity_type_map"]


def get_node_schema(entity_type: str) -> tuple[str, str]:
    """
    AU_Qxxxx → (Neo4j Label, primary_key)
    """
    if entity_type not in ENTITY_TYPE_MAP:
        raise ValueError(f"Unknown entity_type: {entity_type}")

    meta = ENTITY_TYPE_MAP[entity_type]
    return meta["label"], meta["primary_key"]


def build_unique_constraint_cypher(entity_type: str) -> str:
    label, pk = get_node_schema(entity_type)

    return CREATE_CONSTRAINT_CYPHER.format(label=label, pk=pk)


def build_merge_node_cypher(
    entity_type: str,
    entity_id,
    properties: dict | None = None,
):
    label, pk = get_node_schema(entity_type)

    cypher = f"""
    MERGE (n:{label} {{{pk}: $id}})
    ON CREATE SET
        n.entity_type = $entity_type
    """

    params = {
        "id": entity_id,
        "entity_type": entity_type,
    }

    if properties:
        cypher += "\nSET n += $props"
        params["props"] = properties

    return cypher.strip(), params


# ----------------------------
# 查询函数
# ----------------------------
def query_nodes(tx, label, ids):
    query = COUNT_NODES_IN_IDS_CYPHER.format(label=label)
    result = tx.run(query, ids=ids)
    return result.single()[0]


def fetch_node_ids(tx, label: str, limit: int | None = None):
    query = FETCH_NODE_IDS_CYPHER.format(label=label)

    if limit:
        query += "\nLIMIT $limit"

    result = tx.run(query, limit=limit)
    return [record["id"] for record in result]
