import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# This ensures Python can find 'indexing' and 'retrieval' folders
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))

from config.config import Config
from indexing.src.embedding import Embedder
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker
from retrieval.src.pipeline import RetrievalPipeline

from rag_llm.src.llm_client import LLMClient
from rag_llm.src.rag_pipeline import RAGPipeline

def main():
    # Load API Keys
    load_dotenv()
    api_key = os.getenv("AVALAI_API_KEY")
    
    if not api_key:
        print("\n❌ ERROR: API Key missing!")
        return

    print("⏳ Initializing System (Loading Models & Database)...")

    # --- Setup Retrieval Components ---
    embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
    
    retriever = Retriever(
        qdrant_path=str(Config.QDRANT_PATH),
        collection_name=Config.COLLECTION_NAME,
        embedder=embedder
    )
    
    reranker = ReRanker("BAAI/bge-reranker-v2-m3")
    
    retrieval_pipe = RetrievalPipeline(retriever, reranker, embedder)

    # --- Setup Generation Components ---
    llm = LLMClient(
        model_name="gpt-4o-mini",
        api_key=api_key
    )

    # --- Connect RAG ---
    rag = RAGPipeline(
        retrieval_pipeline=retrieval_pipe,
        llm_client=llm
    )

    print("\n" + "="*50)
    print("⚖️  Smart Judicial Assistant (RAG System)")
    print("="*50)

    # --- Chat Loop ---
    while True:
        query = input("\n📝 سوال حقوقی خود را بپرسید (یا 'exit'):\n> ")
        if query.lower() in ['exit', 'quit']:
            print("خداحافظ!")
            break
        
        if not query.strip(): continue

        print("🤔 در حال جستجو و تفکر...")
        
        try:
            result = rag.run(query)

            # Print Answer
            print("\n" + "="*40)
            print("🤖 پاسخ هوشمند:")
            print("="*40)
            print(result["answer"])

            # Print Sources
            print("\n" + "-"*40)
            print("📚 منابع استناد شده:")
            
            # Smart Source Printing
            if result["used_docs"]:
                for doc_label in result["used_docs"]:
                    # Get details from the doc_map
                    info = result["documents"].get(doc_label)
                    if info:
                        doc_title = info['metadata'].get('title', 'بدون عنوان')
                        doc_id = info['real_doc_id']
                        print(f"✅ [{doc_label}] {doc_title} (شماره پرونده: {doc_id})")
            else:
                # If LLM didn't cite anything, list the top 3 docs anyway for debugging
                print("(منابع مستقیم در متن ذکر نشدند. اسناد مرتبط یافت شده:)")
                for i in range(1, min(4, len(result["documents"]) + 1)):
                    label = f"DOC_{i}"
                    if label in result["documents"]:
                        title = result["documents"][label]['metadata'].get('title')
                        print(f"- {title}")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()