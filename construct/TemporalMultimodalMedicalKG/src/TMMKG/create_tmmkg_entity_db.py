"""填充 TMMKG 实体注册数据库。

本模块负责加载实体注册映射 JSON 文件，将疾病、症状和未知实体的
结构化记录写入 MongoDB，并将实体别名文本的 embedding 向量写入
Qdrant 集合。

如果 Qdrant 部署在内网，并且运行环境设置了 HTTP 代理变量，
执行脚本前需要把 Qdrant 主机加入 NO_PROXY/no_proxy。
例如：
    NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
    no_proxy=localhost,127.0.0.1,10.30.1.121 \
    python -m TMMKG.create_tmmkg_entity_db
"""

import uuid

from typing import List
from pydantic import ValidationError
from tqdm import tqdm
import json
import argparse
from pathlib import Path

from TMMKG.infra.mongo import MongoConnection
from TMMKG.infra.qdrant import QdrantConnection
from TMMKG.meta_type import DiseaseEntity, SymptomEntity, UnknownEntity
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.utils.config import load_config, project_path
from TMMKG.utils.logger import get_logger, setup_logging_from_config

from qdrant_client.http.models import PointStruct

BASE_DIR = Path(__file__).resolve().parent
MAPPINGS_DIR = BASE_DIR / "utils" / "entity_registry"

logger = get_logger(__name__)

DEFAULT_EMBEDDING_MODEL_NAME = "Qwen3-Embedding-8B"
encoder = None
embed_dim = None
_encoder_key = None


def init_encoder(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    model_root: str | None = None,
):
    global encoder, embed_dim, _encoder_key

    encoder_key = (model_name, model_root)

    if encoder is None or _encoder_key != encoder_key:
        encoder, embed_dim = get_text_encoder(
            model_name,
            model_root=model_root,
        )
        _encoder_key = encoder_key

    return encoder, embed_dim


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
    # upsert
    qdrant.upsert(
        ids=[p.id for p in points],
        vectors=[p.vector for p in points],
        payloads=[p.payload for p in points],
    )

    logger.info(f"Inserted {len(points)} disease alias vectors into {collection_name}")

    points.clear()
    add_entity(SYMPTOM_2_LABEL, SYMPTOM_2_ALIASES, "symptom")
    # upsert
    qdrant.upsert(
        ids=[p.id for p in points],
        vectors=[p.vector for p in points],
        payloads=[p.payload for p in points],
    )

    logger.info(f"Inserted {len(points)} symptom alias vectors into {collection_name}")

    points.clear()
    add_entity(UNKNOWN_2_LABEL, UNKNOWN_2_ALIASES, "unknown")
    # upsert
    qdrant.upsert(
        ids=[p.id for p in points],
        vectors=[p.vector for p in points],
        payloads=[p.payload for p in points],
    )

    logger.info(f"Inserted {len(points)} unknown alias vectors into {collection_name}")


def create_tmmkg_entity_database(
    mongo_uri: str = "mongodb://localhost:27017/?directConnection=true",
    database: str = "tmmkg_entity",
    qdrant_uri: str = "http://localhost:6333",
    mappings_dir: str | Path | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_model_root: str | None = None,
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

    init_encoder(
        model_name=embedding_model_name,
        model_root=embedding_model_root,
    )

    mappings_path = Path(mappings_dir) if mappings_dir else MAPPINGS_DIR

    # Load mapping files

    with open(mappings_path / "disease2label.json", "r") as f:
        DISEASE_2_LABEL = json.load(f)

    with open(mappings_path / "disease2aliases.json", "r") as f:
        DISEASE_2_ALIASES = json.load(f)

    with open(mappings_path / "symptom2label.json", "r") as f:
        SYMPTOM_2_LABEL = json.load(f)

    with open(mappings_path / "symptom2aliases.json", "r") as f:
        SYMPTOM_2_ALIASES = json.load(f)

    with open(mappings_path / "unknown2label.json", "r") as f:
        UNKNOWN_2_LABEL = json.load(f)

    with open(mappings_path / "unknown2aliases.json", "r") as f:
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
        description="Populate MongoDB and Qdrant with TMMKG entity registry data"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging_from_config(config, "create_tmmkg_entity_db")

    infra = config.get("infra", {})
    mongo = infra.get("mongo", {})
    qdrant = infra.get("qdrant", {})
    embedding = config.get("embedding", {})
    entity_registry = config.get("entity_registry", {})
    collections = entity_registry.get("collections", {})

    create_tmmkg_entity_database(
        mongo_uri=mongo.get("uri", "mongodb://localhost:27017/?directConnection=true"),
        database=entity_registry.get("database", "tmmkg_entity"),
        qdrant_uri=qdrant.get("uri", "http://localhost:6333"),
        mappings_dir=project_path(entity_registry.get("mappings_dir")),
        embedding_model_name=embedding.get("model_name", DEFAULT_EMBEDDING_MODEL_NAME),
        embedding_model_root=embedding.get("model_root"),
        disease_collection=collections.get("disease", "disease_entity"),
        symptom_collection=collections.get("symptom", "symptom_entity"),
        unknown_collection=collections.get("unknown", "unknown_entity"),
        entity_aliases_collection=collections.get("entity_aliases", "entity_aliases"),
        drop_collections=entity_registry.get("drop_collections", True),
    )
