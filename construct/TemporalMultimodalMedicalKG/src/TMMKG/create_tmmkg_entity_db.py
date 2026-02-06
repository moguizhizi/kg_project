import uuid
from pymongo.mongo_client import MongoClient
from pymongo.operations import SearchIndexModel

from typing import List
from pydantic import BaseModel, ValidationError
from tqdm import tqdm
import json
import time
import argparse
import logging
import os
from pathlib import Path
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient

from TMMKG.infra.mongo import MongoConnection
from TMMKG.infra.qdrant import QdrantConnection
from TMMKG.meta_type import DiseaseEntity, SymptomEntity, UnknownEntity
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from TMMKG.services.encoder.registry import get_text_encoder

from qdrant_client.http.models import PointStruct
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
MAPPINGS_DIR = BASE_DIR / "utils" / "entity_registry"

_ = load_dotenv(find_dotenv())
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

encoder, embed_dim = get_text_encoder(
    "Qwen3-Embedding-8B",
    model_root=os.getenv("LLM_ROOT"),
)


def populate_disease_entity(
    DISEASE_2_LABEL,
    db,
    collection_name="disease_entity",
):
    logger.info(f"Starting to populate {collection_name} collection")
    disease_metadata_list = []

    for i, disease_id in enumerate(DISEASE_2_LABEL.keys()):
        label = DISEASE_2_LABEL[disease_id]

        disease_metadata_list.append(
            {
                "_id": i,
                "disease_id": disease_id,
                "label": label,
            }
        )

    try:
        records = [
            DiseaseEntity(**record).model_dump() for record in disease_metadata_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_symptom_entity(
    SYMPTOM_2_LABEL,
    db,
    collection_name="symptom_entity",
):
    logger.info(f"Starting to populate {collection_name} collection")
    symptom_metadata_list = []

    for i, symptom_id in enumerate(SYMPTOM_2_LABEL.keys()):
        label = SYMPTOM_2_LABEL[symptom_id]

        symptom_metadata_list.append(
            {
                "_id": i,
                "symptom_id": symptom_id,
                "label": label,
            }
        )

    try:
        records = [
            SymptomEntity(**record).model_dump() for record in symptom_metadata_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_unknown_entity(
    UNKNOWN_2_LABEL,
    db,
    collection_name="unknown_entity",
):
    logger.info(f"Starting to populate {collection_name} collection")
    symptom_metadata_list = []

    for i, unknown_id in enumerate(UNKNOWN_2_LABEL.keys()):
        label = UNKNOWN_2_LABEL[unknown_id]

        symptom_metadata_list.append(
            {
                "_id": i,
                "unknown_id": unknown_id,
                "label": label,
            }
        )

    try:
        records = [
            UnknownEntity(**record).model_dump() for record in symptom_metadata_list
        ]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_aliases(
    DISEASE_2_LABEL,
    DISEASE_2_ALIASES,
    SYMPTOM_2_LABEL,
    SYMPTOM_2_ALIASES,
    UNKNOWN_2_LABEL,
    UNKNOWN_2_ALIASES,
    qdrant_client,
    collection_name: str = "entity_aliases",
):
    logger.info(f"Starting to populate unified Qdrant collection: {collection_name}")

    qdrant = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=embed_dim,
        client=qdrant_client,
    )

    points = []
    point_id = 0

    def add_entity(label_map, alias_map, entity_type):
        nonlocal point_id

        for entity_id, label in tqdm(label_map.items(), desc=entity_type):

            # canonical
            embedding = encoder.encode(label)

            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),  #
                    vector=embedding,
                    payload={
                        "entity_type": entity_type,
                        "entity_id": entity_id,
                        "alias_label": label,
                        "is_canonical": True,
                    },
                )
            )

            # aliases
            aliases = alias_map.get(entity_id, [])

            for alias in aliases:
                if not alias or alias == label:
                    continue

                embedding = encoder.encode(alias)

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "entity_type": entity_type,
                            "entity_id": entity_id,
                            "alias_label": alias,
                            "is_canonical": False,
                        },
                    ),
                )

    # 三类实体一次写完
    add_entity(DISEASE_2_LABEL, DISEASE_2_ALIASES, "disease")
    add_entity(SYMPTOM_2_LABEL, SYMPTOM_2_ALIASES, "symptom")
    add_entity(UNKNOWN_2_LABEL, UNKNOWN_2_ALIASES, "unknown")

    # upsert
    qdrant.upsert(
        ids=[p.id for p in points],
        vectors=[p.vector for p in points],
        payloads=[p.payload for p in points],
    )

    logger.info(f"Inserted {len(points)} alias vectors into {collection_name}")


def create_tmmkg_entity_database(
    mongo_uri: str = "mongodb://localhost:27017/?directConnection=true",
    database: str = "tmmkg_entity",
    qdrant_uri: str = "http://localhost:6333",
    disease_collection: str = "disease_entity",
    symptom_collection: str = "symptom_entity",
    unknown_collection: str = "unknown_entity",
    entity_aliases_collection: str = "entity_aliases",
    drop_collections: bool = True,
):
    """
    Populate MongoDB with Wikidata ontology data.

    Args:
        mongo_uri: MongoDB connection URI
        database: MongoDB database name
        mappings_dir: Directory containing ontology mapping files. If None, uses default path.
        entity_types_collection: Collection name for entity types
        entity_type_aliases_collection: Collection name for entity type aliases
        properties_collection: Collection name for properties
        property_aliases_collection: Collection name for property aliases
        entity_types_index: Index name for entity types
        property_aliases_index: Index name for property aliases
        drop_collections: Whether to drop existing collections before creating new ones

    Returns:
        Database object
    """

    logger.info("Starting database population process")
    logger.info(f"Using database: {database}")

    # Load mapping files

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

    logger.info("Successfully loaded all mapping files")

    # Connect to MongoDB
    mongo = MongoConnection(mongo_uri, database)
    db = mongo.connect()

    # Connect to qdrant
    qdrant = QdrantConnection(qdrant_uri)
    qdrant_client = qdrant.connect()

    base_entity_aliases_collection = entity_aliases_collection

    physical_entity_aliases_collection = build_collection_name(
        base_entity_aliases_collection, encoder
    )

    # Drop specified collections only
    if drop_collections:
        logger.info("Dropping specified collections...")

        collections_to_drop = [
            disease_collection,
            symptom_collection,
            unknown_collection,
            physical_entity_aliases_collection,
        ]

        existing = set(db.list_collection_names())

        for collection_name in collections_to_drop:
            if collection_name in existing:
                logger.info(f"Dropping collection: {collection_name}")
                db.drop_collection(collection_name)
            else:
                logger.info(f"Collection not found, skip: {collection_name}")

        logger.info("Finished dropping specified collections.")

    if drop_collections:
        for col in [
            disease_collection,
            symptom_collection,
            unknown_collection,
            physical_entity_aliases_collection,
        ]:
            if qdrant_client.collection_exists(col):
                logger.info(f"Dropping Qdrant collection: {col}")
                qdrant_client.delete_collection(col)

    # Populate collections
    populate_disease_entity(
        DISEASE_2_LABEL,
        db,
        collection_name=disease_collection,
    )

    populate_symptom_entity(
        SYMPTOM_2_LABEL,
        db,
        collection_name=symptom_collection,
    )

    populate_unknown_entity(
        UNKNOWN_2_LABEL,
        db,
        collection_name=unknown_collection,
    )

    populate_entity_aliases(
        DISEASE_2_LABEL,
        DISEASE_2_ALIASES,
        SYMPTOM_2_LABEL,
        SYMPTOM_2_ALIASES,
        UNKNOWN_2_LABEL,
        UNKNOWN_2_ALIASES,
        qdrant_client,
        collection_name=physical_entity_aliases_collection,
    )

    logger.info("Database population process completed")

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with Wikidata ontology data"
    )

    parser.add_argument(
        "--mappings_dir",
        type=str,
        default="utils/ontology_mappings/",
        help="Directory containing ontology mapping files",
    )
    parser.add_argument(
        "--mongo_uri",
        type=str,
        default="mongodb://localhost:27017/?directConnection=true",
        help="MongoDB connection URI",
    )
    parser.add_argument(
        "--database",
        type=str,
        default="tmmkg_entity",
        help="MongoDB database name",
    )
    parser.add_argument(
        "--qdrant_uri",
        type=str,
        default="http://localhost:6333",
        help="Qdrant connection URI",
    )

    # Collection names
    parser.add_argument(
        "--disease_collection",
        type=str,
        default="disease_entity",
        help="Collection name for entity types",
    )
    parser.add_argument(
        "--disease_aliases_collection",
        type=str,
        default="disease_aliases_entity",
        help="Collection name for entity type aliases",
    )
    parser.add_argument(
        "--symptom_collection",
        type=str,
        default="symptom_entity",
        help="Collection name for properties",
    )
    parser.add_argument(
        "--symptom_aliases_collection",
        type=str,
        default="symptom_aliases_entity",
        help="Collection name for property aliases",
    )

    parser.add_argument(
        "--unknown_collection",
        type=str,
        default="unknown_entity",
        help="Collection name for properties",
    )
    parser.add_argument(
        "--entity_aliases_collection",
        type=str,
        default="entity_aliases",
        help="Collection name for property aliases",
    )

    args = parser.parse_args()
    create_tmmkg_entity_database(
        mongo_uri=args.mongo_uri,
        database=args.database,
        qdrant_uri=args.qdrant_uri,
        disease_collection=args.disease_collection,
        symptom_collection=args.symptom_collection,
        unknown_collection=args.unknown_collection,
        entity_aliases_collection=args.entity_aliases_collection,
    )
