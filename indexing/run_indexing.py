import json
import glob
import time
from tqdm import tqdm
from src.config import Config
from src.embedding import Embedder
from src.chunking import SemanticChunker
from src.storage import VectorDB

def main():
    print(f"⏱️  Starting Semantic Indexing Pipeline...")
    start_time = time.perf_counter()

    # 1. Initialize
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    chunker = SemanticChunker(embedder, max_tokens=Config.MAX_TOKENS, similarity_threshold=Config.SEMANTIC_THRESHOLD)
    
    db = VectorDB(
        path=str(Config.QDRANT_PATH), 
        collection_name=Config.COLLECTION_NAME, 
        vector_size=embedder.get_dimension()
    )

    file_pattern = str(Config.DATA_DIR / "judgments-*-clean.json")
    files = glob.glob(file_pattern)
    print(f"📂 Found {len(files)} files.")

    total_chunks = 0

    for file_path in tqdm(files, desc="Processing Files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        batch_points = []
        
        for item in data:
            text_full = item.get("text_full", "")
            title = item.get("clean_parts", {}).get("title", "")
            
            if not text_full:
                continue

            # A. Semantic Chunking
            chunks = chunker.chunk_text(text_full, title=title)

            # B. Embed Chunks
            if not chunks: continue
            
            vectors = embedder.embed(chunks)

            # C. Prepare Payload
            for i, chunk_text in enumerate(chunks):
                payload = {
                    "doc_id": item["id"],
                    "chunk_index": i,
                    "text": chunk_text,  # This is the chunk content
                    "metadata": item.get("metadata", {}),
                    "related_laws": item.get("related_laws", []),
                    "original_full_text_id": item["id"] # Reference to full doc if needed
                }
                
                batch_points.append({
                    "vector": vectors[i],
                    "payload": payload
                })

            # D. Batch Upsert
            if len(batch_points) >= Config.BATCH_SIZE:
                db.upsert_batch(batch_points)
                total_chunks += len(batch_points)
                batch_points = []

        if batch_points:
            db.upsert_batch(batch_points)
            total_chunks += len(batch_points)

    end_time = time.perf_counter()
    print(f"\n🎉 Done! Indexed {total_chunks} chunks in {int(end_time - start_time)}s.")

if __name__ == "__main__":
    main()