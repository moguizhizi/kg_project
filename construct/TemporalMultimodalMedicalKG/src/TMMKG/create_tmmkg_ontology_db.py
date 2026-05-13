"""填充 TMMKG 本体数据库。

本模块负责加载本体映射 JSON 文件，将结构化本体记录写入 MongoDB，
并将别名文本的 embedding 向量写入 Qdrant 集合。

如果 Qdrant 部署在内网，并且运行环境设置了 HTTP 代理变量，
执行脚本前需要把 Qdrant 主机加入 NO_PROXY/no_proxy。
例如：
    NO_PROXY=localhost,127.0.0.1,10.30.1.121 \
    no_proxy=localhost,127.0.0.1,10.30.1.121 \
    python -m TMMKG.create_tmmkg_ontology_db
"""

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
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.utils.config import load_config, project_path

from qdrant_client.http.models import PointStruct
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent
MAPPINGS_DIR = BASE_DIR / "utils" / "ontology_mappings"

_ = load_dotenv(find_dotenv())
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL_NAME = "Qwen3-Embedding-8B"
encoder = None
embed_dim = None
_encoder_key = None


def init_encoder(
    model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    model_root: str | None = None,
):
    global encoder, embed_dim, _encoder_key

    model_root = model_root or os.getenv("LLM_ROOT")
    encoder_key = (model_name, model_root)

    if encoder is None or _encoder_key != encoder_key:
        encoder, embed_dim = get_text_encoder(
            model_name,
            model_root=model_root,
        )
        _encoder_key = encoder_key

    return encoder, embed_dim


class EntityType(BaseModel):
    _id: int
    entity_type_id: str
    label: str
    parent_type_ids: List[str]
    valid_subject_property_ids: List[str]
    valid_object_property_ids: List[str]


class Property(BaseModel):
    _id: int
    property_id: str
    label: str
    valid_subject_type_ids: List[str]
    valid_object_type_ids: List[str]


class EntityTypeAlias(BaseModel):
    _id: int
    entity_type_id: str
    alias_label: str
    alias_text_embedding: List[float]


class PropertyAlias(BaseModel):
    _id: int
    relation_id: str
    alias_label: str
    alias_text_embedding: List[float]


def get_mongo_client(mongo_uri):
    client = MongoClient(mongo_uri)
    logger.info("Connection to MongoDB successful")
    return client


def get_qdrant_client(url: str = "http://localhost:6333") -> QdrantClient:
    client = QdrantClient(url=url)
    logger.info(f"Connected to Qdrant at {url}")
    return client


def populate_entity_types(
    ENTITY_TYPE_2_LABEL,
    ENTITY_TYPE_2_HIERARCHY,
    SUBJ_2_PROP_CONSTRAINTS,
    OBJ_2_PROP_CONSTRAINTS,
    db,
    collection_name="entity_types",
):
    logger.info(f"Starting to populate {collection_name} collection")
    entity_metadata_list = []

    for i, entity_type in enumerate(ENTITY_TYPE_2_LABEL.keys()):
        label = ENTITY_TYPE_2_LABEL[entity_type]
        parents = ENTITY_TYPE_2_HIERARCHY[entity_type]

        valid_subject_property_ids = (
            SUBJ_2_PROP_CONSTRAINTS[entity_type]
            if entity_type in SUBJ_2_PROP_CONSTRAINTS
            else []
        )
        valid_object_property_ids = (
            OBJ_2_PROP_CONSTRAINTS[entity_type]
            if entity_type in OBJ_2_PROP_CONSTRAINTS
            else []
        )

        entity_metadata_list.append(
            {
                "_id": i,
                "entity_type_id": entity_type,
                "label": label,
                "parent_type_ids": parents,
                "valid_subject_property_ids": valid_subject_property_ids,
                "valid_object_property_ids": valid_object_property_ids,
            }
        )

    valid_subject_property_ids = (
        SUBJ_2_PROP_CONSTRAINTS["<ANY SUBJECT>"]
        if "<ANY SUBJECT>" in SUBJ_2_PROP_CONSTRAINTS
        else []
    )

    valid_object_property_ids = (
        OBJ_2_PROP_CONSTRAINTS["<ANY OBJECT>"]
        if "<ANY SUBJECT>" in OBJ_2_PROP_CONSTRAINTS
        else []
    )

    entity_metadata_list.append(
        {
            "_id": i + 1,
            "entity_type_id": "ANY",
            "label": "ANY",
            "parent_type_ids": [],
            "valid_subject_property_ids": valid_subject_property_ids,
            "valid_object_property_ids": valid_object_property_ids,
        }
    )

    try:
        records = [EntityType(**record).model_dump() for record in entity_metadata_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_enum_entity_type(
    ENUM_ENTITY_TYPE_VALUES,
    ENTITY_TYPE_2_LABEL,
    db,
    collection_name: str = "enum_entity_type",
):
    logger.info(f"Starting to populate {collection_name} collection")

    records = []

    for idx, (entity_type_id, enum_values) in enumerate(
        ENUM_ENTITY_TYPE_VALUES.items()
    ):
        label = ENTITY_TYPE_2_LABEL.get(entity_type_id)

        if label is None:
            logger.warning(f"Missing label for enum entity type: {entity_type_id}")
            continue

        records.append(
            {
                "_id": idx,
                "entity_type_id": entity_type_id,
                "label": label,
                "enum_values": enum_values,
            }
        )

    if not records:
        logger.warning(f"No enum entity types to insert into {collection_name}")
        return

    collection = db.get_collection(collection_name)

    try:
        collection.insert_many(records, ordered=False)
    except Exception as e:
        logger.error(f"Failed to populate {collection_name}: {e}")
        raise

    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_entity_type_aliases(
    ENTITY_TYPE_2_LABEL,
    ENTITY_TYPE_2_ALIASES,
    qdrant_client,
    collection_name: str = "entity_type_aliases",
):
    logger.info(f"Starting to populate Qdrant collection: {collection_name}")

    qdrant = QdrantVectorStore(
        collection_name=collection_name, vector_size=embed_dim, client=qdrant_client
    )

    points = []
    point_id = 0

    for entity_type, aliases in tqdm(ENTITY_TYPE_2_ALIASES.items()):
        # 主 label
        label = ENTITY_TYPE_2_LABEL[entity_type]
        embedding = encoder.encode(label)

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "entity_type_id": entity_type,
                    "alias_label": label,
                    "is_canonical": True,
                },
            )
        )
        point_id += 1

        # aliases
        for alias in aliases:
            embedding = encoder.encode(alias)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "entity_type_id": entity_type,
                        "alias_label": alias,
                        "is_canonical": False,
                    },
                )
            )
            point_id += 1

    ids = [p.id for p in points]
    vectors = [p.vector for p in points]
    payloads = [p.payload for p in points]

    qdrant.upsert(
        ids=ids,
        vectors=vectors,
        payloads=payloads,
    )

    logger.info(
        f"Successfully populated Qdrant collection "
        f"{collection_name} with {len(points)} points"
    )


def populate_properties(
    PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name="properties"
):
    logger.info(f"Starting to populate {collection_name} collection")
    property_list = []

    for i, prop_id in enumerate(PROP_2_LABEL.keys()):
        property_list.append(
            {
                "_id": i,
                "property_id": prop_id,
                "label": PROP_2_LABEL[prop_id],
                "valid_subject_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Subject type constraint"
                ],
                "valid_object_type_ids": PROP_2_CONSTRAINT[prop_id][
                    "Value-type constraint"
                ],
            }
        )

    try:
        records = [Property(**record).model_dump() for record in property_list]
    except ValidationError as e:
        logger.error(f"Validation error while populating {collection_name}: {e}")

    collection = db.get_collection(collection_name)
    collection.insert_many(records)
    logger.info(f"Successfully populated {collection_name} with {len(records)} records")


def populate_property_aliases(
    PROP_2_LABEL,
    PROP_2_ALIASES,
    qdrant_client,
    collection_name: str = "property_aliases",
):
    logger.info(f"Starting to populate Qdrant collection: {collection_name}")

    qdrant = QdrantVectorStore(
        collection_name=collection_name,
        vector_size=embed_dim,
        client=qdrant_client,
    )

    points = []
    point_id = 0

    for prop, aliases in tqdm(PROP_2_ALIASES.items()):
        # 主 label（canonical）
        label = PROP_2_LABEL[prop]
        embedding = encoder.encode(label)

        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "property_id": prop,
                    "alias_label": label,
                    "is_canonical": True,
                },
            )
        )
        point_id += 1

        # aliases
        for alias in aliases:
            embedding = encoder.encode(alias)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "property_id": prop,
                        "alias_label": alias,
                        "is_canonical": False,
                    },
                )
            )
            point_id += 1

    # 拆 PointStruct → ids / vectors / payloads
    ids = [p.id for p in points]
    vectors = [p.vector for p in points]
    payloads = [p.payload for p in points]

    # 一次性 upsert（幂等）
    qdrant.upsert(
        ids=ids,
        vectors=vectors,
        payloads=payloads,
    )

    logger.info(
        f"Successfully populated Qdrant collection "
        f"{collection_name} with {len(points)} points"
    )


def create_search_index_for_entity_types(
    db,
    collection_name="entity_type_aliases",
    embedding_field_name="alias_text_embedding",
    index_name="entity_type_aliases",
):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    }
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_search_index_for_properties(
    db,
    collection_name="property_aliases",
    embedding_field_name="alias_text_embedding",
    prop_id_field_name="relation_id",
    index_name="property_aliases_ids",
):
    logger.info(f"Starting to create index {index_name} for {collection_name}")
    collection = db.get_collection(collection_name)
    vector_search_index_model = SearchIndexModel(
        definition={
            "mappings": {
                "dynamic": True,
                "fields": {
                    embedding_field_name: {
                        "dimensions": 768,
                        "similarity": "cosine",
                        "type": "knnVector",
                    },
                    prop_id_field_name: {"type": "token"},
                },
            }
        },
        name=index_name,
    )

    try:
        result = collection.create_search_index(model=vector_search_index_model)
        logger.info("Creating index...")
        time.sleep(20)
        logger.info(f"New index {index_name} created successfully: {result}")
    except Exception as e:
        logger.error(f"Error creating new vector search index {index_name}: {str(e)}")


def create_indexes(db):
    logger.info("Creating indexes for entity_types collection...")
    db.entity_types.create_index([("entity_type_id", 1)])
    db.entity_types.create_index([("label", 1)])

    logger.info("Creating indexes for entity_type_aliases collection...")
    db.entity_type_aliases.create_index([("entity_type_id", 1)])
    db.entity_type_aliases.create_index([("alias_label", 1)])

    logger.info("Creating indexes for properties collection...")
    db.properties.create_index([("property_id", 1)])

    # logger.info("Creating indexes for property_aliases collection...")
    # db.property_aliases.create_index("relation_id")

    logger.info("Creating indexes for entity_aliases collection...")
    db.entity_aliases.create_index([("entity_type", 1), ("sample_id", 1)])
    db.entity_aliases.create_index([("label", 1)])

    db.create_collection("triplets")
    logger.info("Creating indexes for triplets collection...")
    db.triplets.create_index([("sample_id", 1)])

    logger.info("All indexes created successfully")


def create_tmmkg_ontology_database(
    mongo_uri: str = "mongodb://localhost:27017/?directConnection=true",
    database: str = "tmmkg_ontology",
    qdrant_uri: str = "http://localhost:6333",
    mappings_dir: str | Path | None = None,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL_NAME,
    embedding_model_root: str | None = None,
    entity_types_collection: str = "entity_types",
    entity_type_aliases_collection: str = "entity_type_aliases",
    enum_entity_type_collection: str = "enum_entity_type",
    properties_collection: str = "properties",
    property_aliases_collection: str = "property_aliases",
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
    with open(mappings_path / "subj_constraint2prop.json", "r") as f:
        subj2prop_constraints = json.load(f)

    with open(mappings_path / "obj_constraint2prop.json", "r") as f:
        obj2prop_constraints = json.load(f)

    with open(mappings_path / "entity_type2label.json", "r") as f:
        ENTITY_TYPE_2_LABEL = json.load(f)

    with open(mappings_path / "entity_type2hierarchy.json", "r") as f:
        ENTITY_TYPE_2_HIERARCHY = json.load(f)

    with open(mappings_path / "entity_type2aliases.json", "r") as f:
        ENTITY_TYPE_2_ALIASES = json.load(f)

    with open(mappings_path / "enum_entity_type.json", "r") as f:
        ENUM_ENTITY_TYPE_VALUES = json.load(f)

    with open(mappings_path / "prop2constraints.json", "r") as f:
        PROP_2_CONSTRAINT = json.load(f)

    with open(mappings_path / "prop2label.json", "r") as f:
        PROP_2_LABEL = json.load(f)

    with open(mappings_path / "prop2aliases.json", "r") as f:
        PROP_2_ALIASES = json.load(f)

    logger.info("Successfully loaded all mapping files")

    # Connect to MongoDB
    mongo = MongoConnection(mongo_uri, database)
    db = mongo.connect()

    # Connect to qdrant
    qdrant = QdrantConnection(qdrant_uri)
    qdrant_client = qdrant.connect()

    # Drop specified collections only
    if drop_collections:
        logger.info("Dropping specified collections...")

        collections_to_drop = [
            entity_types_collection,
            entity_type_aliases_collection,
            enum_entity_type_collection,
            properties_collection,
            property_aliases_collection
        ]

        existing = set(db.list_collection_names())

        for collection_name in collections_to_drop:
            if collection_name in existing:
                logger.info(f"Dropping collection: {collection_name}")
                db.drop_collection(collection_name)
            else:
                logger.info(f"Collection not found, skip: {collection_name}")

        logger.info("Finished dropping specified collections.")
    # Populate collections
    populate_entity_types(
        ENTITY_TYPE_2_LABEL,
        ENTITY_TYPE_2_HIERARCHY,
        subj2prop_constraints,
        obj2prop_constraints,
        db,
        collection_name=entity_types_collection,
    )

    base_entity_type_aliases_collection = entity_type_aliases_collection

    physical_entity_type_aliases_collection = build_collection_name(
        base_entity_type_aliases_collection, encoder
    )

    populate_entity_type_aliases(
        ENTITY_TYPE_2_LABEL,
        ENTITY_TYPE_2_ALIASES,
        qdrant_client,
        collection_name=physical_entity_type_aliases_collection,
    )

    populate_enum_entity_type(
        ENUM_ENTITY_TYPE_VALUES,
        ENTITY_TYPE_2_LABEL,
        db,
        collection_name=enum_entity_type_collection,
    )

    populate_properties(
        PROP_2_LABEL, PROP_2_CONSTRAINT, db, collection_name=properties_collection
    )

    base_property_aliases_collection = property_aliases_collection

    physical_property_aliases_collection = build_collection_name(
        base_property_aliases_collection, encoder
    )

    populate_property_aliases(
        PROP_2_LABEL,
        PROP_2_ALIASES,
        qdrant_client,
        collection_name=physical_property_aliases_collection,
    )

    logger.info("Database population process completed")

    return db


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Populate MongoDB with Wikidata ontology data"
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to shared YAML config. Defaults to configs/tmmkg.yaml.",
    )

    args = parser.parse_args()
    config = load_config(args.config)
    infra = config.get("infra", {})
    mongo = infra.get("mongo", {})
    qdrant = infra.get("qdrant", {})
    embedding = config.get("embedding", {})
    ontology = config.get("ontology", {})
    collections = ontology.get("collections", {})
    model_root_env = embedding.get("model_root_env", "LLM_ROOT")

    create_tmmkg_ontology_database(
        mongo_uri=mongo.get("uri", "mongodb://localhost:27017/?directConnection=true"),
        database=ontology.get("database", "tmmkg_ontology"),
        qdrant_uri=qdrant.get("uri", "http://localhost:6333"),
        mappings_dir=project_path(ontology.get("mappings_dir")),
        embedding_model_name=embedding.get("model_name", DEFAULT_EMBEDDING_MODEL_NAME),
        embedding_model_root=os.getenv(model_root_env),
        entity_types_collection=collections.get("entity_types", "entity_types"),
        entity_type_aliases_collection=collections.get(
            "entity_type_aliases", "entity_type_aliases"
        ),
        enum_entity_type_collection=collections.get(
            "enum_entity_type", "enum_entity_type"
        ),
        properties_collection=collections.get("properties", "properties"),
        property_aliases_collection=collections.get(
            "property_aliases", "property_aliases"
        ),
        drop_collections=ontology.get("drop_collections", True),
    )
