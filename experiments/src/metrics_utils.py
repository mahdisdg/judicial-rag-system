import numpy as np
from rouge_score import rouge_scorer
from collections import Counter

class MetricsCalculator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def calculate_retrieval(self, ground_truth_id: str, retrieved_ids: list):
        """Calculates Recall@K, MRR, NDCG@K."""
        try:
            # Normalize IDs
            gt = str(ground_truth_id)
            retrieved = [str(x) for x in retrieved_ids]
            
            if gt in retrieved:
                rank = retrieved.index(gt) + 1
                mrr = 1.0 / rank
                recall_5 = 1 if rank <= 5 else 0
                recall_10 = 1 if rank <= 10 else 0
                ndcg_10 = 1.0 / (np.log2(rank + 1)) if rank <= 10 else 0.0
            else:
                mrr = 0.0
                recall_5 = 0
                recall_10 = 0
                ndcg_10 = 0.0
                
            return {
                "Recall@5": recall_5,
                "Recall@10": recall_10,
                "MRR": mrr,
                "NDCG@10": ndcg_10
            }
        except Exception:
            return {"Recall@5": 0, "Recall@10": 0, "MRR": 0, "NDCG@10": 0}

    def calculate_generation(self, reference: str, candidate: str):
        """Calculates Exact Match, F1, and ROUGE."""
        if not reference or not candidate:
            return {"EM": 0, "F1": 0, "ROUGE-1": 0, "ROUGE-L": 0}

        # Normalize
        ref_tokens = reference.lower().split()
        can_tokens = candidate.lower().split()

        # Exact Match (Strict)
        em = 1 if reference.strip() == candidate.strip() else 0

        # F1 Score (Token overlap)
        common = Counter(ref_tokens) & Counter(can_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            f1 = 0.0
        else:
            precision = 1.0 * num_same / len(can_tokens)
            recall = 1.0 * num_same / len(ref_tokens)
            f1 = (2 * precision * recall) / (precision + recall)

        # ROUGE
        scores = self.scorer.score(reference, candidate)
        
        return {
            "EM": em,
            "F1": round(f1, 4),
            "ROUGE-1": round(scores['rouge1'].fmeasure, 4),
            "ROUGE-L": round(scores['rougeL'].fmeasure, 4)
        }