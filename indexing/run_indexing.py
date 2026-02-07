import json
import glob
import time
from tqdm import tqdm
from src.config import Config
from src.embedding import Embedder
from src.chunking import SemanticChunker
from src.storage import VectorDB
from src.logger import setup_logger

def main():
    # SETUP LOGGER
    # The filename will look like: indexing_intfloat-multilingual-e5-base_2023-10-25_14-30.log
    logger = setup_logger(Config.MODEL_NAME)
    
    logger.info(f"⏱️  Starting Semantic Indexing Pipeline...")
    logger.info(f"⚙️  Batch Size: {Config.BATCH_SIZE} | Max Tokens: {Config.MAX_TOKENS}")
    start_time = time.perf_counter()

    # Initialize Components
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    chunker = SemanticChunker(embedder, max_tokens=Config.MAX_TOKENS, similarity_threshold=Config.SEMANTIC_THRESHOLD)
    
    db_path_str = str(Config.QDRANT_PATH)
    db = VectorDB(
        path=db_path_str, 
        collection_name=Config.COLLECTION_NAME, 
        vector_size=embedder.get_dimension()
    )


    file_pattern = str(Config.DATA_DIR / "judgments-*-clean.json")
    files = glob.glob(file_pattern)
    logger.info(f"📂 Found {len(files)} files to index.")

    total_chunks = 0

    # Processing Loop
    for file_path in tqdm(files, desc="Processing Files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch_points = []
        
        for item in data:
            text_full = item.get("text_full", "")
            title = item.get("clean_parts", {}).get("title", "")
            
            if not text_full:
                continue

            # Semantic Chunking
            chunks = chunker.chunk_text(text_full, title=title)

            # Embed Chunks
            if not chunks: continue
            
            vectors = embedder.embed(chunks)

            # Prepare Payload
            for i, chunk_text in enumerate(chunks):
                payload = {
                    "doc_id": item["id"],
                    "chunk_index": i,
                    "text": chunk_text,  # This is the chunk content
                    "metadata": item.get("metadata", {}),
                    "related_laws": item.get("related_laws", []),
                    "mentioned_laws": item.get("mentioned_laws", []),
                    "original_full_text_id": item["id"] # Reference to full doc if needed
                }
                
                batch_points.append({
                    "vector": vectors[i],
                    "payload": payload
                })

            # Batch Upsert
            if len(batch_points) >= Config.BATCH_SIZE:
                db.upsert_batch(batch_points)
                total_chunks += len(batch_points)
                batch_points = []

        if batch_points:
            db.upsert_batch(batch_points)
            total_chunks += len(batch_points)

    # Final Logging
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    logger.info("="*40)
    logger.info(f"🎉 Indexing Complete for {Config.MODEL_NAME}")
    logger.info(f"📊 Total Chunks Indexed: {total_chunks}")
    logger.info(f"⏱️  Total Time: {duration:.2f} seconds")
    logger.info(f"🚀 Average Speed: {total_chunks/duration:.2f} chunks/sec")
    logger.info("="*40)

if __name__ == "__main__":
    main()