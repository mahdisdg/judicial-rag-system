import json
import os
from pathlib import Path
# from tqdm import tqdm 

from src.text_cleaning import normalize_text, to_english_digits
from src.anonymization import Anonymizer, extract_related_laws

def process_file(page_number: int, secret: str) -> None:
    # Path configuration
    in_path = Path(f"../data/raw/judgments-{page_number}.json")
    
    if not in_path.exists():
        return

    out_path = Path(f"../data/processed/judgments-{page_number}-clean.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    anonymizer = Anonymizer(secret=secret)
    processed_docs = []

    for idx, item in enumerate(data):
        meta = item.get("metadata") or {}
        
        # 1. Get Raw Fields
        title_raw = meta.get("title") or ""
        message_raw = item.get("message") or ""
        decision_raw = item.get("decision_text") or ""

        # 2. Normalize & Standardize Digits
        title_norm = to_english_digits(normalize_text(title_raw))
        message_norm = to_english_digits(normalize_text(message_raw))
        decision_norm = to_english_digits(normalize_text(decision_raw))

        # 3. Anonymize
        title_anon = anonymizer.process(title_norm)
        message_anon = anonymizer.process(message_norm)
        decision_anon = anonymizer.process(decision_norm)

        # 4. Construct Texts
        # Short = Title + Summary (For Embedding)
        text_short = f"{title_anon}\n\n{message_anon}".strip()
        
        # Full = Title + Summary + Full Decision (For Context/Reranking)
        text_full = f"{title_anon}\n\n{message_anon}\n\n{decision_anon}".strip()

        # 5. Handle Laws (Corrected Logic)
        
        # A. Keep original scraped laws (with URLs) exactly as they are
        related_laws_original = item.get("related_laws", [])

        # B. Extract laws mentioned in the text (Strings only)
        mentioned_laws_extracted = extract_related_laws(text_full)

        # 6. Construct Output
        processed_docs.append({
            "id": f"{meta.get('case_number', 'doc')}_{idx}",
            "metadata": {
                **meta,
                "date_processed": meta.get("date") 
            },
            "text_short": text_short, 
            "text_full": text_full,   
            "clean_parts": {
                "title": title_anon,
                "message": message_anon,
                "decision": decision_anon
            },
            # Preserved: List of Dictionaries with URLs
            "related_laws": related_laws_original, 
            # New: List of Strings extracted from text
            "mentioned_laws": mentioned_laws_extracted 
        })

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=2)

    print(f"✅ Processed Page {page_number}: {len(processed_docs)} documents saved.")

if __name__ == "__main__":
    SECRET_KEY = os.getenv("ANON_SECRET", "MY_SUPER_SECRET_PROJECT_KEY_2024")
    
    # Adjust range to match your files
    for page in range(1, 1027): 
        process_file(page, SECRET_KEY)