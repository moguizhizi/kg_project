from qdrant_client import QdrantClient
from .base import VectorStore

class QdrantVectorStore(VectorStore):

    def __init__(self, collection_name: str, host="localhost", port=6333):
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

    def exists(self) -> bool:
        cols = self.client.get_collections().collections
        return any(c.name == self.collection_name for c in cols)

    def upsert(self, ids, vectors, payloads):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    "id": i,
                    "vector": v,
                    "payload": p,
                }
                for i, v, p in zip(ids, vectors, payloads)
            ],
        )

    def search(self, query_vector, top_k, filter=None):
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=filter,
        )
