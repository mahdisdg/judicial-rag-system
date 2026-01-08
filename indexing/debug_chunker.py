import json
import sys
from pathlib import Path
from src.config import Config
from src.embedding import Embedder
from src.chunking import SemanticChunker

def debug_one_file(file_name: str):
    file_path = Config.DATA_DIR / file_name
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return

    print("⏳ Initializing Models (This takes a moment)...")
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    chunker = SemanticChunker(embedder, max_tokens=Config.MAX_TOKENS, similarity_threshold=Config.SEMANTIC_THRESHOLD)

    print(f"📂 Reading: {file_name}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Just take the first document for testing
    doc = data[0]
    
    title = doc['clean_parts']['title']
    full_text = doc['text_full']
    
    print("\n" + "="*50)
    print(f"📄 DOCUMENT TITLE: {title}")
    print(f"📏 Original Length: {len(full_text)} characters")
    print("="*50 + "\n")

    # Run Chunking
    chunks = chunker.chunk_text(full_text, title=title)

    print(f"🧩 Result: Split into {len(chunks)} chunks based on semantic similarity.")
    print("-" * 30)

    for i, chunk in enumerate(chunks):
        print(f"\n🔸 CHUNK {i+1} ({len(chunk)} chars):")
        print(chunk)
        print("-" * 30)

if __name__ == "__main__":
    # Change this to one of your actual file names
    TEST_FILE = "judgments-2-clean.json" 
    debug_one_file(TEST_FILE)