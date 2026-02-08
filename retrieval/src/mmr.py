import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def mmr(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    lambda_: float = 0.7,
    k: int = 5
):
    selected = []
    candidates = list(range(len(doc_embeddings)))

    similarity_to_query = cosine_similarity(
        query_embedding.reshape(1, -1),
        doc_embeddings
    )[0]

    while len(selected) < k and candidates:
        scores = []

        for c in candidates:
            redundancy = max(
                cosine_similarity(
                    doc_embeddings[c].reshape(1, -1),
                    doc_embeddings[selected]
                )[0]
            ) if selected else 0.0

            score = (
                lambda_ * similarity_to_query[c]
                - (1 - lambda_) * redundancy
            )
            scores.append(score)

        best_idx = candidates[int(np.argmax(scores))]
        selected.append(best_idx)
        candidates.remove(best_idx)

    return selected
