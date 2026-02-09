# services/property_resolver.py

import logging
import os
from typing import List
from qdrant_client import QdrantClient
from dotenv import load_dotenv, find_dotenv

from TMMKG.meta_type import PropertyCandidate
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
