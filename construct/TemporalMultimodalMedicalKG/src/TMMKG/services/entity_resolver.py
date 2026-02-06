from typing import List, Optional

from TMMKG.meta_type import EntityCandidate
from TMMKG.services.encoder.registry import get_text_encoder
from TMMKG.vectorstores.base import build_collection_name
from TMMKG.vectorstores.qdrant import QdrantVectorStore

import os
import logging
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


class EntityResolver:
    def __init__(
        self,
        vector_store: QdrantVectorStore,
        encoder,
        score_threshold: float = 0.75,
    ):
        """
        vector_store: QdrantVectorStore (entity_aliases)
        encoder: TextEncoder
        """
        self.vector_store = vector_store
        self.encoder = encoder
        self.score_threshold = score_threshold

    def resolve(
        self,
        text: str,
        top_k: int = 5,
        entity_types: Optional[List[str]] = None,
        # e.g. ["disease"]
    ) -> List[EntityCandidate]:

        if not text or not text.strip():
            return []

        vector = self.encoder.encode(text)

        # 极重要：用 filter 限制类型
        query_filter = None
        if entity_types:
            query_filter = {
                "must": [
                    {
                        "key": "entity_type",
                        "match": {"any": entity_types},
                    }
                ]
            }

        hits = self.vector_store.search(
            query_vector=vector,
            top_k=top_k,
            filter=query_filter,
        )

        results: List[EntityCandidate] = []

        for h in hits:
            score = h.get("score")
            if score is None or score < self.score_threshold:
                continue

            payload = h.get("payload") or {}

            entity_id = payload.get("entity_id")
            alias_label = payload.get("alias_label")

            if not entity_id or not alias_label:
                continue

            results.append(
                EntityCandidate(
                    entity_id=entity_id,
                    entity_type=payload.get("entity_type"),
                    alias_label=alias_label,
                    is_canonical=payload.get("is_canonical", False),
                    score=score,
                )
            )

        return results


def init_entity_resolver(
    model_name: str = "Qwen3-Embedding-8B",
    base_collection: str = "entity_aliases",
    qdrant_url: str = "http://localhost:6333",
    score_threshold: float = 0.75,
) -> EntityResolver:
    """
    初始化 EntityResolver 对象（带向量存储和文本编码器）。

    Returns:
        EntityResolver: 可直接用于实体解析
    """

    logger.info("Starting EntityResolver initialization")

    # -----------------------
    # 1. init encoder
    # -----------------------
    encoder, embed_dim = get_text_encoder(
        model_name,
        model_root=os.getenv("LLM_ROOT"),
    )
    logger.info(f"TextEncoder initialized (dim={embed_dim})")

    # -----------------------
    # 2. init vector store
    # -----------------------
    physical_collection = build_collection_name(base_collection, encoder)
    qdrant_client = QdrantClient(url=qdrant_url)
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
        score_threshold=score_threshold,
    )
    logger.info("EntityResolver initialized successfully ✅")

    return resolver
