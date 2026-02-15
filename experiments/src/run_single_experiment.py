import sys
import os
import json
import logging
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from config.config import Config
from indexing.src.embedding import Embedder
from retrieval.src.retriever import Retriever
from retrieval.src.reranker import ReRanker
from retrieval.src.pipeline import RetrievalPipeline
from rag_llm.src.llm_client import LLMClient
from rag_llm.src.rag_pipeline import RAGPipeline
from experiments.src.metrics_utils import MetricsCalculator

# --- DUMMY RERANKER (For "No Reranker" Mode) ---
class PassthroughReranker:
    def rerank(self, query, passages):
        # Return dummy scores preserving original order
        return [1.0 - (i * 0.01) for i in range(len(passages))]

class ExperimentRunner:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("AVALAI_API_KEY")
        self.dataset_path = project_root / "experiments" / "data" / "benchmark_20.json"
        self.metrics_calc = MetricsCalculator()
        
        # Setup Experiment Folder
        self.exp_dir = project_root / "experiments" / "logs" / Config.EXPERIMENT_NAME
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Logging
        self.logger = self._setup_logger()
        
        # Log Configuration
        self.logger.info(f"🧪 EXPERIMENT START: {Config.EXPERIMENT_NAME}")
        self.logger.info(f"   - Embedding: {Config.EMBEDDING_MODEL}")
        self.logger.info(f"   - DB Path: {Config.QDRANT_PATH}")
        self.logger.info(f"   - Reranker: {'ON' if Config.USE_RERANKER else 'OFF'}")
        self.logger.info(f"   - LLM: {Config.LLM_MODEL}")

    def _setup_logger(self):
        logger = logging.getLogger(Config.EXPERIMENT_NAME)
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        
        # File Handler (Execution Log - Changed to .log)
        f_handler = logging.FileHandler(self.exp_dir / "execution.log", encoding='utf-8')
        f_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(f_handler)
        
        # Console Handler
        c_handler = logging.StreamHandler()
        logger.addHandler(c_handler)
        return logger

    def setup_pipeline(self):
        self.logger.info("⚙️  Initializing Pipeline Components...")
        
        embedder = Embedder(Config.EMBEDDING_MODEL, is_e5=Config.IS_E5_MODEL)
        
        retriever = Retriever(
            qdrant_path=str(Config.QDRANT_PATH),
            collection_name=Config.COLLECTION_NAME,
            embedder=embedder
        )
        
        if Config.USE_RERANKER:
            reranker = ReRanker(Config.RERANKER_NAME)
            self.logger.info("   -> Reranker Enabled (BGE-M3)")
        else:
            reranker = PassthroughReranker()
            self.logger.info("   -> Reranker Disabled (Passthrough)")
            
        retrieval_pipe = RetrievalPipeline(retriever, reranker, embedder)
        
        llm = LLMClient(model_name=Config.LLM_MODEL, api_key=self.api_key)
        
        self.rag = RAGPipeline(retrieval_pipe, llm)

    def run(self):
        self.setup_pipeline()
        
        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
            
        results = []
        
        self.logger.info(f"🚀 Processing {len(dataset)} items...")
        
        for i, item in enumerate(tqdm(dataset)):
            time.sleep(60) # Rate limit safety between items

            query = item['question']
            gt_id = item['id']
            ref_answer = item['reference_answer']
            
            self.logger.info(f"\n--- Q{i+1}: {query} ---")
            
            # Run RAG
            start = time.time()
            try:
                res = self.rag.run(query)
                latency = time.time() - start
                
                # Extract Data
                generated_answer = res['answer']
                
                # Get Retrieved IDs
                retrieved_ids = [v['real_doc_id'] for v in res['documents'].values()]
                
                # Calculate Metrics
                ret_metrics = self.metrics_calc.calculate_retrieval(gt_id, retrieved_ids)
                gen_metrics = self.metrics_calc.calculate_generation(ref_answer, generated_answer)
                
                # Log Details to File
                self.logger.info(f"   📝 Generated Answer:\n{generated_answer}")
                self.logger.info(f"   🎯 Retrieval Metrics: {ret_metrics}")
                self.logger.info(f"   🎯 Generation Metrics: {gen_metrics}")
                self.logger.info(f"   ⏱️ Latency: {latency:.2f}s")
                
                # Store Row
                row = {
                    "id": gt_id,
                    "question": query,
                    "generated_answer": generated_answer,
                    "latency": latency,
                    **ret_metrics,
                    **gen_metrics
                }
                results.append(row)
                
            except Exception as e:
                self.logger.error(f"❌ Failed on Q{i+1}: {e}")

        # Save Matrix
        if results:
            df = pd.DataFrame(results)
            csv_path = self.exp_dir / "results_matrix.csv"
            df.to_csv(csv_path, index=False)
            
            # Calculate Averages for Console
            avg_metrics = df.mean(numeric_only=True)
            self.logger.info("\n" + "="*40)
            self.logger.info(f"📊 EXPERIMENT SUMMARY: {Config.EXPERIMENT_NAME}")
            self.logger.info("-" * 20)
            
            # Display ALL 9 Metrics
            metrics_to_show = [
                "Recall@5", "Recall@10", "MRR", "NDCG@10", # Retrieval
                "EM", "F1", "ROUGE-1", "ROUGE-L"           # Generation
            ]
            
            for metric in metrics_to_show:
                val = avg_metrics.get(metric, 0)
                self.logger.info(f"{metric:<10}: {val:.4f}")
                
            self.logger.info(f"Avg Latency: {avg_metrics.get('latency', 0):.2f}s")
            self.logger.info("="*40)
            self.logger.info(f"✅ Saved results to: {self.exp_dir}")

if __name__ == "__main__":
    runner = ExperimentRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Cleanup Qdrant connection
        if hasattr(runner, 'rag'):
            try:
                runner.rag.retrieval_pipeline.retriever.client.close()
                print("🔒 Qdrant Closed.")
            except:
                pass