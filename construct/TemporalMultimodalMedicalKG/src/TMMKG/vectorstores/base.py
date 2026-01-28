from abc import ABC, abstractmethod
from typing import List, Dict, Any

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
