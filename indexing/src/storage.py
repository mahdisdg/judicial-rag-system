from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid

class VectorDB:
    def __init__(self, path: str, collection_name: str, vector_size: int):
        print(f"💾 Connecting to Qdrant at: {path}")
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.client.collection_exists(self.collection_name):
            print(f"🛠️ Creating Collection: {self.collection_name}")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )

    def upsert_batch(self, points_data: List[Dict[str, Any]]):
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=item['vector'],
                payload=item['payload']
            )
            for item in points_data
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)