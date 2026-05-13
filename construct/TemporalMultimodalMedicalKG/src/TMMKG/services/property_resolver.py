# services/property_resolver.py

import logging
from typing import List
from qdrant_client import QdrantClient

from TMMKG.meta_type import PropertyCandidate
from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore
from functools import lru_cache

# -----------------------
# logging
# -----------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


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


@lru_cache(maxsize=1)
def init_property_resolver(
    model_name: str = "Qwen3-Embedding-8B",
    model_root: str | None = None,
    base_collection: str = "property_aliases",
    qdrant_url: str = "http://localhost:6333",
    score_threshold: float = 0.75,
) -> PropertyResolver:
    """
    获取 PropertyResolver（单例）。

    自动缓存：
        - TextEncoder
        - QdrantVectorStore
        - Resolver

    Returns:
        PropertyResolver
    """

    logger.info("Initializing PropertyResolver...")

    # encoder 只加载一次
    encoder, embed_dim = get_text_encoder(
        model_name,
        model_root=model_root,
    )

    physical_collection = build_collection_name(base_collection, encoder)

    vector_store = QdrantVectorStore(
        collection_name=physical_collection,
        vector_size=embed_dim,
        client=QdrantClient(url=qdrant_url),
    )

    resolver = PropertyResolver(
        vector_store=vector_store,
        encoder=encoder,
        score_threshold=score_threshold,
    )

    logger.info("PropertyResolver ready")

    return resolver
