# graph/neo4j.py

import json
from pathlib import Path
from neo4j import GraphDatabase

from TMMKG.infra.neo4j_db import create_neo4j_driver

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "domains"
    / "Home-based_user_training"
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

    return f"""
    CREATE CONSTRAINT IF NOT EXISTS
    FOR (n:{label})
    REQUIRE n.{pk} IS UNIQUE
    """


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


entity_type = "AU_Q0004"
entity_id = "55"

properties = {
    "gender": "女",
    "age": 2,
}

cypher_1, params_1 = build_merge_node_cypher(
    entity_type=entity_type,
    entity_id=entity_id,
    properties=properties,
)


def main():
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    with driver.session() as session:
        for entity_type in ENTITY_TYPE_MAP.keys():
            cypher = build_unique_constraint_cypher(entity_type)
            session.run(cypher)
            print(f"Constraint created for {entity_type}")

            session.run(cypher_1, params_1)

    driver.close()


if __name__ == "__main__":
    main()
