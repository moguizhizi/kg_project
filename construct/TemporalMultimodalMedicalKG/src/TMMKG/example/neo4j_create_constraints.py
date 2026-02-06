from TMMKG.graph.neo4j_db import build_unique_constraint_cypher
from TMMKG.infra.neo4j_db import create_neo4j_driver
from pathlib import Path
import json

SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "domains"
    / "home_based_user_training"
    / "neo4j_node.json"
)

with open(SCHEMA_PATH, "r") as f:
    ENTITY_TYPE_MAP = json.load(f)["entity_type_map"]


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

    driver.close()


if __name__ == "__main__":
    main()
