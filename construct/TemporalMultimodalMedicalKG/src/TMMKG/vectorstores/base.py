from abc import ABC, abstractmethod
from typing import List, Dict, Any
from typing import Protocol


class VectorStore(ABC):

    @abstractmethod
    def upsert(
        self,
        ids: List[str],
        vectors: List[List[float]],
        payloads: List[Dict[str, Any]],
    ):
        pass

    @abstractmethod
    def search(
        self,
        query_vector: List[float],
        top_k: int,
        filter: Dict[str, Any] | None = None,
    ):
        pass

    @abstractmethod
    def exists(self) -> bool:
        pass


class EncoderLike(Protocol):
    model_id: str
    pooling: str
    dim: int


def build_collection_name(base: str, encoder: EncoderLike) -> str:
    """
    Build a physical vector collection name bound to embedding space.

    Example:
        entity_type_aliases_contriever_mean_d768
    """
    return f"{base}" f"_{encoder.model_id}" f"_{encoder.pooling}" f"_d{encoder.dim}"
