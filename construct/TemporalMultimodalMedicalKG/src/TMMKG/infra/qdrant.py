from qdrant_client import QdrantClient
import logging

logger = logging.getLogger(__name__)


class QdrantConnection:
    def __init__(self, url: str):
        self.url = url
        self._client = None

    def connect(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url)
            logger.info(f"Connected to Qdrant at {self.url}")
        return self._client
