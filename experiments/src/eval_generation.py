import sys
import os
import json
import time
import logging
import pandas as pd
from datetime import datetime
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

class GenerationEvaluator:
    def __init__(self):
        load_dotenv()
        self.dataset_path = project_root / "experiments" / "data" / "golden_dataset.json"
        
        # Define Log Directories
        self.log_dir = project_root / "experiments" / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.log_dir 
        
        self.api_key = os.getenv("AVALAI_API_KEY")
        
        # Setup Dual Loggers
        self.trace_logger, self.result_logger = self._setup_loggers()

        if not self.api_key:
            self.log(f"❌ API Key not found in .env", level="error")
            raise ValueError("API Key missing")

        # Setup Retrieval
        self.log("⚙️  Initializing Retrieval Engine...")
        self.embedder = Embedder(Config.MODEL_NAME, is_e5=Config.IS_E5_MODEL)
        self.retriever = Retriever(str(Config.QDRANT_PATH), Config.COLLECTION_NAME, self.embedder)
        self.reranker = ReRanker(Config.RERANKER_NAME)
        self.retrieval_pipe = RetrievalPipeline(self.retriever, self.reranker, self.embedder)
        
        # Judge Client
        self.judge_llm = LLMClient("gpt-4o-mini", api_key=self.api_key, temperature=0.0)

    def close(self):
        """Safely close resources."""
        if hasattr(self, 'retriever'):
            self.log("🔒 Closing Qdrant connection...")
            self.retriever.close()

    def _setup_loggers(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        # Trace Logger (Everything)
        trace_logger = logging.getLogger("TraceLogger")
        trace_logger.setLevel(logging.INFO)
        if trace_logger.hasHandlers(): trace_logger.handlers.clear()
        
        trace_file = self.log_dir / f"eval_trace_{timestamp}.log"
        t_handler = logging.FileHandler(trace_file, encoding='utf-8')
        t_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        trace_logger.addHandler(t_handler)
        
        # Console Output
        c_handler = logging.StreamHandler()
        c_handler.setFormatter(logging.Formatter('%(message)s'))
        trace_logger.addHandler(c_handler)

        # Result Logger (Metrics Only)
        result_logger = logging.getLogger("ResultLogger")
        result_logger.setLevel(logging.INFO)
        if result_logger.hasHandlers(): result_logger.handlers.clear()
        
        res_file = self.log_dir / f"eval_results_{timestamp}.log"
        r_handler = logging.FileHandler(res_file, encoding='utf-8')
        r_handler.setFormatter(logging.Formatter('%(message)s'))
        result_logger.addHandler(r_handler)

        return trace_logger, result_logger

    def log(self, message, level="info", to_results=False):
        if level == "info": self.trace_logger.info(message)
        elif level == "error": self.trace_logger.error(message)
        elif level == "warning": self.trace_logger.warning(message)
        
        if to_results:
            self.result_logger.info(message)

    def _retry_wrapper(self, func, description="API Call"):
        """
        Infinite Loop Retry. NEVER gives up.
        """
        attempt = 1
        bad_keywords = ["error", "متاسفانه", "خطا", "rate limit", "quota", "bad gateway", "timeout"]

        while True:
            try:
                result = func()
                
                # Check for textual error messages
                if isinstance(result, str):
                    for kw in bad_keywords:
                        if kw in result.lower() and len(result) < 100:
                            raise Exception(f"Soft API Error detected: {result}")
                
                if attempt > 1:
                    self.log(f"✅ {description} succeeded after {attempt} attempts.")
                
                return result

            except KeyboardInterrupt:
                self.log("🛑 Execution stopped by user.", level="error")
                raise # Exit the script

            except Exception as e:
                self.log(f"⏳ {description} Failed (Attempt {attempt}): {e}", level="warning")
                self.log("   --> Cooling down for 60 seconds...", level="info")
                time.sleep(60) 
                attempt += 1

    def get_judge_score(self, question, context, answer):
        system_prompt = "You are an impartial legal judge."
        user_prompt = f"""
Query: {question}
Context: {context[:2000]}...
Answer: {answer}

Task:
1. Faithfulness: Is the answer derived ONLY from context? (0=No, 1=Yes)
2. Reason: Short explanation.

Output JSON ONLY: {{"score": 0.0 to 1.0, "reason": "..."}}
"""
        def _call():
            resp = self.judge_llm.generate(system_prompt, user_prompt)
            clean = resp.replace("```json", "").replace("```", "").strip()
            return json.loads(clean)

        data = self._retry_wrapper(_call, description="Judge Evaluation")
        if data:
            return data.get("score", 0), data.get("reason", "N/A")
        return 0, "Judge Failed"

    def run(self):
        dataset = json.load(open(self.dataset_path, 'r', encoding='utf-8'))
        self.log(f"🚀 Starting Evaluation on {len(dataset)} questions", to_results=True)
        
        all_results_df = pd.DataFrame()
        output_file = self.results_dir / "final_generation_report.csv"

        for exp in Config.EXPERIMENTS:
            exp_name = exp["name"]
            model = exp["model"]
            temp = exp["temperature"]
            top_p = exp["top_p"]
            
            self.log(f"\n🧪 STARTING: {exp_name} | {model} | T={temp} | Top_p={top_p}", to_results=True)
            
            llm = LLMClient(model_name=model, api_key=self.api_key, temperature=temp, top_p=top_p)
            rag = RAGPipeline(self.retrieval_pipe, llm)
            
            exp_results = []

            for i, item in enumerate(dataset):
                query = item['question']
                target_doc_id = item['id']
                
                self.log(f"--- Q{i+1}: {query[:50]}... ---")
                
                # Run RAG (Infinite Retry)
                start = time.time()
                res = self._retry_wrapper(lambda: rag.run(query), description="RAG Gen")
                latency = time.time() - start

                # Judge (Infinite Retry)
                context_text = "".join([d.get('text', '') for d in res['documents'].values()])
                time.sleep(5) # Small gap between RAG and Judge
                score, reason = self.get_judge_score(query, context_text, res['answer'])
                
                # Calculate Metrics
                retrieved_ids = [info['real_doc_id'] for info in res['documents'].values()]
                citation_hit = 1 if target_doc_id in retrieved_ids else 0
                hallucination = 1 if score < 0.7 else 0

                row = {
                    "config": exp_name,
                    "model": model,
                    "query_id": target_doc_id,
                    "latency": latency,
                    "faithfulness": score,
                    "hallucination": hallucination,
                    "citation_recall": citation_hit,
                    "answer_len": len(res['answer'])
                }
                
                exp_results.append(row)
                
                # Save Incremental Progress to CSV (Safety)
                all_results_df = pd.concat([all_results_df, pd.DataFrame([row])])
                all_results_df.to_csv(output_file, index=False)
                
                self.log(f"   ⚖️ Score: {score} | Hit: {citation_hit} | Time: {latency:.1f}s")

            # --- End of Experiment Summary ---
            if exp_results:
                df = pd.DataFrame(exp_results)
                
                avg_faith = df['faithfulness'].mean()
                avg_lat = df['latency'].mean()
                hallucination_rate = df['hallucination'].mean() * 100
                citation_acc = df['citation_recall'].mean() * 100
                
                self.log("-" * 30, to_results=True)
                self.log(f"📊 RESULTS: {exp_name}", to_results=True)
                self.log(f"   🔹 Faithfulness:       {avg_faith:.4f}", to_results=True)
                self.log(f"   🔹 Hallucination Rate: {hallucination_rate:.1f}%", to_results=True)
                self.log(f"   🔹 Citation Recall:    {citation_acc:.1f}%", to_results=True)
                self.log(f"   🔹 Avg Latency:        {avg_lat:.2f}s", to_results=True)
                self.log("-" * 30, to_results=True)

if __name__ == "__main__":
    evaluator = GenerationEvaluator()
    try:
        evaluator.run()
    finally:
        evaluator.close()