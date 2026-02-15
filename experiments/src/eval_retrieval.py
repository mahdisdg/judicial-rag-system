import sys
import os
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config.config import Config
from indexing.src.embedding import Embedder
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker

class RetrievalEvaluator:
    def __init__(self):
        self.dataset_path = project_root / "experiments" / "data" / "golden_dataset.json"
        self.results_dir = project_root / "experiments" / "logs"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        print(f"⚙️  Loading Model: {Config.MODEL_NAME}")
        self.embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
        
        print(f"📂 Opening Database: {Config.QDRANT_PATH}")
        self.retriever = Retriever(str(Config.QDRANT_PATH), Config.COLLECTION_NAME, self.embedder)
        
        print("⚖️  Loading Reranker...")
        self.reranker = ReRanker("BAAI/bge-reranker-v2-m3")

    def load_dataset(self):
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def calculate_metrics(self, ground_truth_id, retrieved_ids):
        """Calculates Rank, Reciprocal Rank, and Hits for one query."""
        try:
            rank = retrieved_ids.index(str(ground_truth_id)) + 1
            reciprocal_rank = 1.0 / rank
            hit_at_5 = 1 if rank <= 5 else 0
            hit_at_10 = 1 if rank <= 10 else 0
        except ValueError:
            # Not found in the list
            rank = None
            reciprocal_rank = 0.0
            hit_at_5 = 0
            hit_at_10 = 0
            
        return rank, reciprocal_rank, hit_at_5, hit_at_10

    def run(self):
        dataset = self.load_dataset()
        results = []
        
        print(f"🚀 Starting Evaluation on {len(dataset)} queries...")
        print(f"ℹ️  Configuration: {Config.MODEL_NAME}")

        for item in tqdm(dataset):
            query = item['question']
            target_id = item['id']
            
            # --- VECTOR RETRIEVAL (Top 50) ---
            t0 = time.time()
            hits, _ = self.retriever.retrieve(query, top_k=50)
            t1 = time.time()
            
            # Extract IDs from Qdrant hits
            vector_ids = [h.payload.get('doc_id') for h in hits]
            
            v_rank, v_mrr, v_r5, v_r10 = self.calculate_metrics(target_id, vector_ids)

            # --- WITH RERANKING (Top 10) ---
            # We take the top 50 from vector search and rerank them
            texts = [h.payload.get('text', '') for h in hits]
            
            t2 = time.time()
            scores = self.reranker.rerank(query, texts)
            t3 = time.time()
            
            # Sort hits based on reranker scores
            ranked_pairs = sorted(zip(vector_ids, scores), key=lambda x: x[1], reverse=True)
            reranked_ids = [doc_id for doc_id, _ in ranked_pairs]
            
            rr_rank, rr_mrr, rr_r5, rr_r10 = self.calculate_metrics(target_id, reranked_ids)

            results.append({
                "query_id": target_id,
                # Vector Search Metrics
                "vector_rank": v_rank,
                "vector_mrr": v_mrr,
                "vector_recall@5": v_r5,
                "vector_recall@10": v_r10,
                "vector_time": t1 - t0,
                # Reranked Metrics
                "rerank_rank": rr_rank,
                "rerank_mrr": rr_mrr,
                "rerank_recall@5": rr_r5,
                "rerank_recall@10": rr_r10,
                "rerank_time": t3 - t2
            })

        # --- AGGREGATE RESULTS ---
        df = pd.DataFrame(results)
        
        # Calculate Averages
        summary = {
            "Model": Config.MODEL_NAME,
            "Vector_MRR": df["vector_mrr"].mean(),
            "Vector_Recall@5": df["vector_recall@5"].mean(),
            "Vector_Recall@10": df["vector_recall@10"].mean(),
            "Rerank_MRR": df["rerank_mrr"].mean(),
            "Rerank_Recall@5": df["rerank_recall@5"].mean(),
            "Rerank_Recall@10": df["rerank_recall@10"].mean(),
        }

        print("\n" + "="*40)
        print(f"📊 RESULTS FOR: {Config.MODEL_NAME}")
        print("="*40)
        print(f"🔹 Vector Search (No Rerank):")
        print(f"   MRR:        {summary['Vector_MRR']:.4f}")
        print(f"   Recall@5:   {summary['Vector_Recall@5']:.4f}")
        print(f"   Recall@10:  {summary['Vector_Recall@10']:.4f}")
        print("-" * 40)
        print(f"🔸 With Reranker (BGE-M3):")
        print(f"   MRR:        {summary['Rerank_MRR']:.4f}")
        print(f"   Recall@5:   {summary['Rerank_Recall@5']:.4f}")
        print(f"   Recall@10:  {summary['Rerank_Recall@10']:.4f}")
        print("="*40)

        # Save to CSV
        clean_name = Config.MODEL_NAME.split("/")[-1]
        output_file = self.results_dir / f"eval_retrieval_{clean_name}.csv"
        df.to_csv(output_file, index=False)
        print(f"📄 Detailed results saved to: {output_file}")

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.run()