from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import List, Dict, Any
import uuid
from .logger import logger
from .config import Config

class VectorDB:
    def __init__(self, path: str, collection_name: str, vector_size: int):
        logger.info(f"💾 Connecting to Qdrant Local at: {path}")
        
        self.client = QdrantClient(path=path)
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            logger.info("="*40)
            logger.info(f"🛠️  CREATING QDRANT INDEX: {self.collection_name}")
            logger.info(f"   - Vector Size: {self.vector_size}")
            logger.info(f"   - M (Edges): {Config.HNSW_M}")
            logger.info(f"   - ef_construct: {Config.HNSW_EF}")
            logger.info("="*40)

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                ),
                
                hnsw_config=models.HnswConfigDiff(
                    m=Config.HNSW_M,
                    ef_construct=Config.HNSW_EF
                )
            )
        else:
            logger.info(f"✅ Collection {self.collection_name} already exists.")

    def upsert_batch(self, points_data: List[Dict[str, Any]]):
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=item['vector'],
                payload=item['payload']
            )
            for item in points_data
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )