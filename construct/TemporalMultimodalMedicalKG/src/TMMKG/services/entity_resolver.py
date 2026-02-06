from typing import List, Optional

from TMMKG.meta_type import EntityCandidate
from TMMKG.vectorstores.qdrant import QdrantVectorStore


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
