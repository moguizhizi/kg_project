from pymongo import MongoClient
import logging

logger = logging.getLogger(__name__)


class MongoConnection:
    def __init__(self, uri: str, database: str):
        self.uri = uri
        self.database_name = database
        self._client = None
        self._db = None

    def connect(self):
        if self._client is None:
            self._client = MongoClient(self.uri)
            self._db = self._client[self.database_name]
            logger.info(f"Connected to MongoDB: {self.database_name}")
        return self._db

    @property
    def db(self):
        if self._db is None:
            raise RuntimeError("MongoDB not connected. Call connect() first.")
        return self._db
