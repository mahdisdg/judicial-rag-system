import re
from typing import List
from sklearn.metrics.pairwise import cosine_similarity

class SemanticChunker:
    def __init__(self, embedder, max_tokens: int = 512, similarity_threshold: float = 0.65, overlap_sentences: int = 1):
        self.embedder = embedder
        self.tokenizer = embedder.tokenizer
        self.max_tokens = max_tokens
        self.threshold = similarity_threshold
        self.overlap = overlap_sentences
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

        valid_sentences = []
        for s in raw_sentences:
            if len(self.tokenizer.tokenize(s)) > self.max_tokens:
                valid_sentences.extend(self._force_split(s))
            else:
                valid_sentences.append(s)

        if not valid_sentences: return []
        embeddings = self.embedder.embed(valid_sentences)

        sims = []
        for i in range(len(embeddings) - 1):
            v1 = embeddings[i].reshape(1, -1)
            v2 = embeddings[i+1].reshape(1, -1)
            sim = cosine_similarity(v1, v2)[0][0]
            sims.append(sim)

        chunks = []
        current_chunk = [valid_sentences[0]]
        
        for i in range(len(sims)):
            similarity = sims[i]
            next_s = valid_sentences[i+1]
            
            cand_text = " ".join(current_chunk + [next_s])
            token_count = len(self.tokenizer.tokenize(f"Title: {title}\n{cand_text}"))

            if token_count >= self.max_tokens or (similarity < self.threshold and len(current_chunk) > 3):
                chunks.append(" ".join(current_chunk))
                overlap_start = max(0, len(current_chunk) - self.overlap)
                current_chunk = current_chunk[overlap_start:] + [next_s]
            else:
                current_chunk.append(next_s)
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return [f"Title: {title}\n{c}".strip() for c in chunks]

    def _force_split(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) <= self.max_tokens: return [text]
        chunks = []
        start = 0
        step = self.max_tokens - 50 
        while start < len(tokens):
            end = min(start + self.max_tokens, len(tokens))
            chunk_str = self.tokenizer.decode(self.tokenizer.convert_tokens_to_ids(tokens[start:end]), skip_special_tokens=True)
            chunks.append(chunk_str)
            start += step
        return chunks