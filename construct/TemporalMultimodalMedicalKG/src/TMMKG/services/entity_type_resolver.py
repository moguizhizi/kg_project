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


import os
import logging
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def init_entity_type_resolver(
    model_name: str = "Qwen3-Embedding-8B",
    base_collection: str = "entity_type_aliases",
    qdrant_url: str = "http://localhost:6333",
    score_threshold: float = 0.75,
) -> EntityTypeResolver:
    """
    初始化 EntityTypeResolver 对象（带向量存储和文本编码器）。

    Args:
        model_name (str): 文本编码器模型名称
        base_collection (str): Qdrant collection 基础名
        qdrant_url (str): Qdrant 服务地址
        score_threshold (float): 相似度阈值

    Returns:
        EntityTypeResolver: 可直接用于实体类型解析
    """

    logger.info("Starting EntityTypeResolver initialization")

    # -----------------------
    # 1. 初始化文本编码器
    # -----------------------
    encoder, embed_dim = get_text_encoder(
        model_name,
        model_root=os.getenv("LLM_ROOT"),
    )
    logger.info(f"TextEncoder initialized (dim={embed_dim})")

    # -----------------------
    # 2. 初始化向量存储
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
    # 3. 初始化解析器
    # -----------------------
    resolver = EntityTypeResolver(
        vector_store=vector_store,
        encoder=encoder,
        score_threshold=score_threshold,
    )
    logger.info("EntityTypeResolver initialized successfully ✅")

    return resolver
