# services/property_resolver.py

import logging
import os
from typing import List
from dataclasses import dataclass
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv

from TMMKG.services.encoder.registry import get_text_encoder
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
# PropertyCandidate
# -----------------------
@dataclass
class PropertyCandidate:
    property_id: str
    alias_label: str
    is_canonical: bool
    score: float


# -----------------------
# PropertyResolver
# -----------------------
class PropertyResolver:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        encoder,
        score_threshold: float = 0.75,
    ):
        """
        vector_store: VectorStore (QdrantVectorStore)
        encoder: TextEncoder
        """
        self.vector_store = vector_store
        self.encoder = encoder
        self.score_threshold = score_threshold

    def resolve(self, text: str, top_k: int = 5) -> List[PropertyCandidate]:
        if not text or not text.strip():
            return []

        vector = self.encoder.encode(text)

        hits = self.vector_store.search(
            query_vector=vector,
            top_k=top_k,
        )

        results: List[PropertyCandidate] = []

        for h in hits:
            score = h.get("score")
            if score is None or score < self.score_threshold:
                continue

            payload = h.get("payload") or {}

            property_id = payload.get("property_id")
            alias_label = payload.get("alias_label")

            if not property_id or not alias_label:
                continue

            results.append(
                PropertyCandidate(
                    property_id=property_id,
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
    logger.info("Starting PropertyResolver demo")

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
    base_property_aliases_collection = "property_aliases"

    physical_property_aliases_collection = build_collection_name(
        base_property_aliases_collection, encoder
    )

    qdrant_client = QdrantClient(url="http://localhost:6333")

    vector_store = QdrantVectorStore(
        collection_name=physical_property_aliases_collection,
        vector_size=embed_dim,
        client=qdrant_client,
    )
    logger.info("QdrantVectorStore initialized")

    # -----------------------
    # 3. init resolver
    # -----------------------
    resolver = PropertyResolver(
        vector_store=vector_store,
        encoder=encoder,
        score_threshold=0.75,
    )
    logger.info("PropertyResolver initialized")

    # -----------------------
    # 4. test input
    # -----------------------
    text = "出生日期"
    logger.info(f"Resolving property for text: '{text}'")

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
            f"- property_id={c.property_id}, "
            f"alias='{c.alias_label}', "
            f"is_canonical={c.is_canonical}, "
            f"score={c.score:.4f}"
        )


if __name__ == "__main__":
    main()
