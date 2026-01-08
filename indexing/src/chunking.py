import re
import numpy as np
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, embedder, max_tokens: int = 512, similarity_threshold: float = 0.65, overlap_sentences: int = 1):
        self.embedder = embedder
        self.tokenizer = embedder.tokenizer
        self.max_tokens = max_tokens
        self.threshold = similarity_threshold
        self.overlap = overlap_sentences  # <--- NEW PARAMETER
        
        # Regex for Persian sentence splitting
        self.sentence_split_re = re.compile(r'([.?!؛\n]+)')

    def _split_sentences(self, text: str) -> List[str]:
        parts = self.sentence_split_re.split(text)
        sentences = []
        current = ""
        for part in parts:
            if self.sentence_split_re.match(part):
                current += part
                if len(current.strip()) > 0:
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())
        return sentences

    def chunk_text(self, text: str, title: str = "") -> List[str]:
        if not text: return []

        raw_sentences = self._split_sentences(text)
        if not raw_sentences: return []

        # 1. Safe Embedding (Split huge single sentences if necessary)
        valid_sentences = []
        for s in raw_sentences:
            if len(self.tokenizer.tokenize(s)) > self.max_tokens:
                valid_sentences.extend(self._force_split(s))
            else:
                valid_sentences.append(s)

        if not valid_sentences: return []

        embeddings = self.embedder.embed(valid_sentences)

        # 2. Calculate Similarities
        sims = []
        for i in range(len(embeddings) - 1):
            v1 = embeddings[i].reshape(1, -1)
            v2 = embeddings[i+1].reshape(1, -1)
            sim = cosine_similarity(v1, v2)[0][0]
            sims.append(sim)

        # 3. Grouping with Overlap
        chunks = []
        current_chunk_sentences = [valid_sentences[0]]
        
        for i in range(len(sims)):
            similarity = sims[i]
            next_sentence = valid_sentences[i+1]
            
            # Construct candidate text to check size
            # Note: We don't join with space here for counting to be precise with title
            candidate_text = " ".join(current_chunk_sentences + [next_sentence])
            token_count = len(self.tokenizer.tokenize(f"Title: {title}\n{candidate_text}"))

            should_split = False
            
            # Split condition A: Token limit reached
            if token_count >= self.max_tokens:
                should_split = True
            
            # Split condition B: Semantic shift (Low similarity)
            # Only split on semantics if we have at least 3 sentences (avoid tiny chunks)
            elif similarity < self.threshold and len(current_chunk_sentences) > 3:
                should_split = True

            if should_split:
                # Save current chunk
                chunks.append(" ".join(current_chunk_sentences))
                
                # START NEXT CHUNK WITH OVERLAP
                # Take the last 'self.overlap' sentences from current and start next chunk
                overlap_start = max(0, len(current_chunk_sentences) - self.overlap)
                current_chunk_sentences = current_chunk_sentences[overlap_start:] + [next_sentence]
            else:
                current_chunk_sentences.append(next_sentence)
        
        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

        # 4. Final Formatting
        final_chunks = []
        for chunk in chunks:
            enriched_chunk = f"Title: {title}\n{chunk}".strip()
            final_chunks.append(enriched_chunk)

        return final_chunks

    def _force_split(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_tokens:
            return [text]
        chunks = []
        start = 0
        step = self.max_tokens - 50 
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_str = self.tokenizer.decode(
                self.tokenizer.convert_tokens_to_ids(chunk_tokens),
                skip_special_tokens=True
            )
            chunks.append(chunk_str)
            start += step
        return chunks