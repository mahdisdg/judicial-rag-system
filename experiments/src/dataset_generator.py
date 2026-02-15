import sys
import os
import json
import random
import time
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# --- SETUP PATHS ---
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
sys.path.append(str(project_root))

from rag_llm.src.llm_client import LLMClient

# --- LOGGER SETUP ---
def setup_experiment_logger():
    log_dir = project_root / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_file = log_dir / f"generation_log_{timestamp}.log"
    
    logger = logging.getLogger("ExpGenerator")
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_experiment_logger()

class GoldenDatasetGenerator:
    def __init__(self, data_dir: str, output_dir: str, target_size: int = 50):
        load_dotenv()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_file = self.output_dir / "golden_dataset.json"
        self.target_size = target_size
        
        api_key = os.getenv("AVALAI_API_KEY")
        if not api_key:
            raise ValueError("API Key not found in .env")
            
        self.llm = LLMClient(model_name="gpt-4o-mini", api_key=api_key)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_current_dataset(self):
        if self.output_file.exists():
            try:
                with open(self.output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Simple validation to ensure file isn't just "[]" or empty
                    if isinstance(data, list):
                        logger.info(f"📂 Loaded existing dataset with {len(data)} items.")
                        return data
            except Exception:
                logger.warning("⚠️ Existing file corrupted or empty. Starting fresh.")
        return []

    def save_dataset(self, data):
        with open(self.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_source_documents(self):
        files = list(self.data_dir.glob("judgments-*-clean.json"))
        all_docs = []
        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    all_docs.extend(json.load(f))
            except Exception:
                continue
        return all_docs

    def generate_question(self, doc_text: str, title: str) -> str:
        """
        Generates a question with Retry logic + Error Detection.
        """
        system_prompt = "You are a legal expert. Generate a specific Persian search query based on the text."
        user_prompt = f"Title: {title}\nText: {doc_text[:1500]}\nTask: Write a Persian question for this text."
        
        retries = 0
        max_retries = 5 # Try 5 times before giving up on this document

        while retries < max_retries:
            try:
                start_time = time.time()
                question = self.llm.generate(system_prompt, user_prompt)
                duration = time.time() - start_time
                
                # --- VALIDATION CHECK ---
                # Check if the response is actually an error message from our LLMClient
                bad_keywords = ["error", "متاسفانه", "خطا", "rate limit", "quota"]
                
                is_invalid = False
                if not question or len(question) < 10:
                    is_invalid = True
                else:
                    for kw in bad_keywords:
                        if kw in question.lower() and len(question) < 100:
                            is_invalid = True
                            break
                
                if is_invalid:
                    raise Exception(f"API returned error message: {question}")

                # Success
                logger.info(f"✅ Generated in {duration:.2f}s")
                return question.replace("Question:", "").replace('"', '').strip()

            except Exception as e:
                retries += 1
                logger.warning(f"⚠️ Attempt {retries}/{max_retries} failed: {e}")
                
                if retries < max_retries:
                    logger.info("⏳ Hit Rate Limit or Connection Error. Sleeping for 60 seconds...")
                    time.sleep(60) # Wait 1 minute
                else:
                    logger.error("❌ Max retries reached for this doc. Skipping.")
                    return None

    def run(self):
        dataset = self.load_current_dataset()
        
        # Calculate remaining
        if len(dataset) >= self.target_size:
            logger.info("🎉 Dataset already complete.")
            return

        needed = self.target_size - len(dataset)
        logger.info(f"📉 Generating {needed} more items...")

        all_docs = self.load_source_documents()
        used_ids = {item['id'] for item in dataset}
        available_docs = [d for d in all_docs if d['id'] not in used_ids and len(d.get('text_short', '')) > 50]
        
        if not available_docs:
            logger.error("❌ No unused documents available!")
            return

        selection = random.sample(available_docs, min(needed, len(available_docs)))
        
        print(f"🚀 Starting generation for {len(selection)} items...")
        
        for doc in tqdm(selection, desc="Generating"):
            # Small pause between successful requests to prevent hitting rate limits too quickly
            time.sleep(2) 
            
            question = self.generate_question(
                doc.get('text_short', ''), 
                doc.get('metadata', {}).get('title', '')
            )
            
            if question:
                new_entry = {
                    "id": str(doc.get('id')),
                    "question": question,
                    "metadata": {
                        "title": doc.get('metadata', {}).get('title'),
                        "source_url": doc.get('metadata', {}).get('source_url')
                    }
                }
                dataset.append(new_entry)
                self.save_dataset(dataset) # Save immediately

        logger.info(f"🏁 Finished. Dataset size: {len(dataset)}")

if __name__ == "__main__":
    DATA_PATH = project_root / "data" / "processed"
    OUTPUT_PATH = project_root / "experiments" / "data"
    
    generator = GoldenDatasetGenerator(
        data_dir=str(DATA_PATH), 
        output_dir=str(OUTPUT_PATH),
        target_size=50 
    )
    generator.run()