import os
import logging

from qdrant_client import QdrantClient

from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.services.entity_resolver import EntityResolver
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from dotenv import load_dotenv, find_dotenv


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_ = load_dotenv(find_dotenv())


def main():
    logger.info("Starting EntityResolver demo")

    # -----------------------
    # 1. init encoder
    # -----------------------
    encoder, embed_dim = get_text_encoder(
        "Qwen3-Embedding-8B",
        model_root=os.getenv("LLM_ROOT"),
    )

    logger.info(f"TextEncoder initialized (dim={embed_dim})")

    # -----------------------
    # 2. init vector store
    # -----------------------
    base_collection = "entity_aliases"

    physical_collection = build_collection_name(
        base_collection,
        encoder,
    )

    qdrant_client = QdrantClient(url="http://localhost:6333")

    vector_store = QdrantVectorStore(
        collection_name=physical_collection,
        vector_size=embed_dim,
        client=qdrant_client,
    )

    logger.info(f"QdrantVectorStore initialized: {physical_collection}")

    # -----------------------
    # 3. init resolver
    # -----------------------
    resolver = EntityResolver(
        vector_store=vector_store,
        encoder=encoder,
        score_threshold=0.75,
    )

    logger.info("EntityResolver initialized")

    # -----------------------
    # 4. test input
    # -----------------------
    text = "发烧"
    entity_types = ["symptom"]  # 可改为 ["disease"]

    logger.info(f"Resolving entity for text='{text}', filter={entity_types}")

    # -----------------------
    # 5. resolve
    # -----------------------
    candidates = resolver.resolve(
        text=text,
        top_k=5,
        entity_types=entity_types,
    )

    # -----------------------
    # 6. output
    # -----------------------
    if not candidates:
        logger.info("No candidates found")
        return

    logger.info("Candidates:")

    for c in candidates:
        logger.info(
            f"- entity_id={c.entity_id}, "
            f"type={c.entity_type}, "
            f"alias='{c.alias_label}', "
            f"is_canonical={c.is_canonical}, "
            f"score={c.score:.4f}"
        )


if __name__ == "__main__":
    main()
