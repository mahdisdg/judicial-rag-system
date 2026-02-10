from typing import List, Dict, Tuple, Any

class ContextBuilder:
    def __init__(self, max_docs: int = 8):
        self.max_docs = max_docs

    def build(self, hits: List[Dict[str, Any]]) -> Tuple[str, Dict[str, dict]]:
        """
        Builds the string context for the LLM from the pipeline results.
        
        Args:
            hits: List of dictionaries returned by RetrievalPipeline.run()
                  Format: [{'text':..., 'metadata':..., 'doc_id':..., 'score':...}]
        """
        context_blocks = []
        doc_map = {}

        # hits are already sorted by the pipeline
        for idx, hit in enumerate(hits[: self.max_docs], start=1):
            doc_label = f"DOC_{idx}"
            
            # Extract data from Dictionary
            text = hit.get("text", "").strip()
            meta = hit.get("metadata", {})
            title = meta.get("title", "بدون عنوان")
            real_doc_id = hit.get("doc_id", "unknown")

            if not text: continue

            # Create text block with Title for better LLM context
            block = f"[{doc_label}] (Title: {title})\n{text}"
            context_blocks.append(block)

            # Map for citations
            doc_map[doc_label] = {
                "point_id": hit.get("chunk_id"),
                "real_doc_id": real_doc_id,
                "score": hit.get("score", 0.0),
                "metadata": meta,
            }

        context_str = "\n\n".join(context_blocks)
        return context_str, doc_map