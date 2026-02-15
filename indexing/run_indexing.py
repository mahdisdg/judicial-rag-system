import json
import glob
import time
from tqdm import tqdm
from config.config import Config
from src.embedding import Embedder
from src.chunking import SemanticChunker
from src.storage import VectorDB
from src.logger import setup_logger

def main():
    # Setup Logging
    logger = setup_logger(Config.MODEL_NAME)
    
    logger.info(f"⏱️  Starting Indexing Pipeline")
    logger.info(f"⚙️  Model: {Config.MODEL_NAME} | Batch: {Config.BATCH_SIZE}")
    logger.info(f"📂 Database Path: {Config.DB_ROOT_DIR}")
    
    start_time = time.perf_counter()

    # Initialize Components
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    chunker = SemanticChunker(embedder, max_tokens=Config.MAX_TOKENS, similarity_threshold=Config.SEMANTIC_THRESHOLD)
    
    # Qdrant Setup
    db = VectorDB(
        path=str(Config.QDRANT_PATH),
        collection_name=Config.COLLECTION_NAME, 
        vector_size=embedder.get_dimension()
    )

    # Find Files
    file_pattern = str(Config.DATA_DIR / "judgments-*-clean.json")
    files = glob.glob(file_pattern)
    
    if not files:
        logger.error(f"No files found in {Config.DATA_DIR}")
        return

    logger.info(f"Found {len(files)} files to index.")

    # Process Loop
    total_chunks = 0

    for file_path in tqdm(files, desc="Processing Files"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read {file_path}: {e}")
            continue

        batch_points = []
        
        for item in data:
            text_full = item.get("text_full", "")
            title = item.get("clean_parts", {}).get("title", "")
            
            if not text_full: continue

            # Chunk
            chunks = chunker.chunk_text(text_full, title=title)
            if not chunks: continue
            
            # Embed
            vectors = embedder.embed(chunks)

            # Prepare Data
            for i, chunk_text in enumerate(chunks):
                payload = {
                    "doc_id": item["id"],
                    "chunk_index": i,
                    "text": chunk_text,
                    "metadata": item.get("metadata", {}),
                    "related_laws": item.get("related_laws", []),
                    "mentioned_laws": item.get("mentioned_laws", [])
                }
                
                batch_points.append({
                    "vector": vectors[i],
                    "payload": payload
                })

            # Batch Insert
            if len(batch_points) >= Config.BATCH_SIZE:
                db.upsert_batch(batch_points)
                total_chunks += len(batch_points)
                batch_points = []

        # Insert remaining
        if batch_points:
            db.upsert_batch(batch_points)
            total_chunks += len(batch_points)

    end_time = time.perf_counter()
    duration = end_time - start_time
    
    logger.info("="*40)
    logger.info(f"🎉 Indexing Complete!")
    logger.info(f"📊 Total Chunks: {total_chunks}")
    logger.info(f"⏱️  Time: {duration:.2f}s ({duration/60:.1f} min)")
    logger.info("="*40)

if __name__ == "__main__":
    main()