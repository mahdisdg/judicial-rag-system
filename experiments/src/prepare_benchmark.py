import sys
import os
import json
import random
import time
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from rag_llm.src.llm_client import LLMClient

class BenchmarkPreparer:
    def __init__(self, source_dataset: str, output_path: str, target_size: int = 20):
        load_dotenv()
        self.source_path = Path(source_dataset)
        self.output_path = Path(output_path)
        self.target_size = target_size
        self.data_dir = project_root / "data" / "processed"
        
        self.api_key = os.getenv("AVALAI_API_KEY")
        self.llm = LLMClient(model_name="gpt-4o-mini", api_key=self.api_key)

    def load_doc_text(self, doc_id):
        """Finds the full text of the document by ID."""

        files = list(self.data_dir.glob("judgments-*-clean.json"))
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        # Match ID
                        if str(item['id']) == str(doc_id):
                            return item.get('text_full', '')
            except:
                continue
        return None

    def generate_reference_answer(self, question, context):
        """Generates a Gold Standard answer for metrics calculation."""
        prompt = f"""
You are a senior judge. Write a **perfect, concise, and factual answer** to the question based ONLY on the provided legal text.
This answer will be used as a "Ground Truth" to evaluate other AI models.

Text: {context[:4000]}

Question: {question}

Reference Answer (Persian):
"""
        return self.llm.generate("You are a helpful assistant.", prompt)

    def run(self):
        print(f"📂 Loading source: {self.source_path}")
        with open(self.source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Select 20 random items
        if len(data) > self.target_size:
            selection = random.sample(data, self.target_size)
        else:
            selection = data

        benchmark_data = []
        print(f"🚀 Generating Reference Answers for {len(selection)} items...")

        for item in tqdm(selection):
            doc_id = item['id']
            question = item['question']
            
            # Get Context
            context = self.load_doc_text(doc_id)
            if not context:
                print(f"⚠️ Doc {doc_id} not found, skipping.")
                continue

            # Generate Answer
            try:
                ref_answer = self.generate_reference_answer(question, context)
                time.sleep(60) # Rate limit safety
                
                benchmark_data.append({
                    "id": doc_id,
                    "question": question,
                    "reference_answer": ref_answer, # Needed for ROUGE/F1
                    "metadata": item.get('metadata', {})
                })
            except Exception as e:
                print(f"❌ Error on {doc_id}: {e}")

        # Save
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Benchmark Ready: {self.output_path}")

if __name__ == "__main__":
    SOURCE = project_root / "experiments" / "data" / "golden_dataset.json"
    OUTPUT = project_root / "experiments" / "data" / "benchmark_20.json"
    
    prep = BenchmarkPreparer(str(SOURCE), str(OUTPUT))
    prep.run()