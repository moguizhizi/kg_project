import logging
import os
from dotenv import load_dotenv, find_dotenv
from qdrant_client import QdrantClient

from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.services.entity_type_resolver import (
    EntityTypeResolver,
    init_entity_type_resolver,
)
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore

# -----------------------
# logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


_ = load_dotenv(find_dotenv())


# -----------------------
# main
# -----------------------
def main():

    resolver = init_entity_type_resolver(
        model_name="Qwen3-Embedding-8B",
        base_collection="entity_type_aliases",
        qdrant_url="http://localhost:6333",
        score_threshold=0.75,
    )

    # -----------------------
    # 4. test input
    # -----------------------
    text = "药物"
    logger.info(f"Resolving entity type for text: '{text}'")

    # -----------------------
    # 5. resolve
    # -----------------------
    candidates = resolver.resolve(text, top_k=5)

    # -----------------------
    # 6. output
    # -----------------------
    if not candidates:
        logger.info("No candidates found")
        return

    logger.info("Candidates:")
    for c in candidates:
        logger.info(
            f"- entity_type_id={c.entity_type_id}, "
            f"alias='{c.alias_label}', "
            f"is_canonical={c.is_canonical}, "
            f"score={c.score:.4f}"
        )


if __name__ == "__main__":
    main()
