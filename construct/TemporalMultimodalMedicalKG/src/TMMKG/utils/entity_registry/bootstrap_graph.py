import os
import json
from pathlib import Path

from TMMKG.graph.neo4j_db import get_node_schema
from TMMKG.infra.neo4j_db import create_neo4j_driver
from TMMKG.sql_templates import UPSERT_NODE_CYPHER

import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


BASE_DIR = Path(__file__).resolve().parent.parent.parent
MAPPINGS_DIR = BASE_DIR / "utils" / "entity_registry"


with open(os.path.join(MAPPINGS_DIR, "disease2label.json"), "r") as f:
    DISEASE_2_LABEL = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "disease2aliases.json"), "r") as f:
    DISEASE_2_ALIASES = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "symptom2label.json"), "r") as f:
    SYMPTOM_2_LABEL = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "symptom2aliases.json"), "r") as f:
    SYMPTOM_2_ALIASES = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "unknown2label.json"), "r") as f:
    UNKNOWN_2_LABEL = json.load(f)

with open(os.path.join(MAPPINGS_DIR, "unknown2aliases.json"), "r") as f:
    UNKNOWN_2_ALIASES = json.load(f)

ENTITY_REGISTRY = {
    "AU_Q0013": (DISEASE_2_LABEL, DISEASE_2_ALIASES),
    "AU_Q0040": (SYMPTOM_2_LABEL, SYMPTOM_2_ALIASES),
    "AU_Q0041": (UNKNOWN_2_LABEL, UNKNOWN_2_ALIASES),
}


def build_entity_dict(entity_type, entity_ids, label_map, alias_map):
    result = []

    for eid in entity_ids:
        name = label_map.get(eid)
        if not name:
            continue

        aliases = alias_map.get(eid)

        # 没有别名
        if not aliases:
            result.append({"id": str(eid), "name": name})
        else:
            # 别名列表转成字符串，用逗号分隔
            alias_str = ",".join(aliases)
            result.append({"id": str(eid), "name": name, "别名": alias_str})

    return {entity_type: result}


def main():

    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = "password"

    logger.info("Connecting to Neo4j...")

    driver = create_neo4j_driver(uri=uri, user=user, password=password)

    with driver.session() as session:

        for entity_type, (label_map, alias_map) in ENTITY_REGISTRY.items():

            logger.info(f"Processing entity_type={entity_type}")

            entity_dict = build_entity_dict(
                entity_type=entity_type,
                entity_ids=label_map.keys(),
                label_map=label_map,
                alias_map=alias_map,
            )

            for label, group in entity_dict.items():

                node_name, _ = get_node_schema(label)
                cypher = UPSERT_NODE_CYPHER.format(label=node_name)

                logger.info(f"Upserting label={node_name} | rows={len(group)}")

                session.run(cypher, rows=group)

                logger.info(f"Finished label={node_name}")

    driver.close()

    logger.info("All entities loaded successfully ✅")


if __name__ == "__main__":
    main()
