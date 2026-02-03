import logging
import os
from qdrant_client import QdrantClient
from typing import List
from TMMKG.meta_type import EntityTypeCandidate
from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from dotenv import load_dotenv, find_dotenv

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
# EntityTypeResolver
# -----------------------


class EntityTypeResolver:
    def __init__(
        self,
        vector_store,
        encoder,
        score_threshold: float = 0.75,
    ):
        """
        vector_store: VectorStore (e.g. QdrantVectorStore)
        encoder: TextEncoder
        """
        self.vector_store = vector_store
        self.encoder = encoder
        self.score_threshold = score_threshold

    def resolve(self, text: str, top_k: int = 5) -> List[EntityTypeCandidate]:
        if not text or not text.strip():
            return []

        vector = self.encoder.encode(text)

        hits = self.vector_store.search(
            query_vector=vector,
            top_k=top_k,
        )

        results: List[EntityTypeCandidate] = []

        for h in hits:
            score = h.get("score")
            if score is None or score < self.score_threshold:
                continue

            payload = h.get("payload") or {}

            entity_type_id = payload.get("entity_type_id")
            alias_label = payload.get("alias_label")

            if not entity_type_id or not alias_label:
                continue

            results.append(
                EntityTypeCandidate(
                    entity_type_id=entity_type_id,
                    alias_label=alias_label,
                    is_canonical=payload.get("is_canonical", False),
                    score=score,
                )
            )

        return results


# -----------------------
# main
# -----------------------
def main():
    logger.info("Starting EntityTypeResolver demo")

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

    base_entity_type_aliases_collection = "entity_type_aliases"

    physical_entity_type_aliases_collection = build_collection_name(
        base_entity_type_aliases_collection, encoder
    )

    qdrant_client = QdrantClient(url="http://localhost:6333")
    vector_store = QdrantVectorStore(
        collection_name=physical_entity_type_aliases_collection,
        vector_size=embed_dim,
        client=qdrant_client,
    )
    logger.info("QdrantVectorStore initialized")

    # -----------------------
    # 3. init resolver
    # -----------------------
    resolver = EntityTypeResolver(
        vector_store=vector_store,
        encoder=encoder,
        score_threshold=0.75,
    )
    logger.info("EntityTypeResolver initialized")

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
