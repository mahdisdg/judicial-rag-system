from dataclasses import dataclass
from typing import List, Dict
import re


@dataclass
class Citation:
    doc_id: str
    point_id: str
    score: float
    metadata: Dict


@dataclass
class Answer:
    text: str
    citations: List[Citation]
    is_valid: bool
    error: str | None = None


class AnswerParser:
    """
    Parses LLM output, extracts citations, and maps them to retrieved documents.
    """

    CITATION_PATTERN = re.compile(r"\[(DOC_\d+)\]")

    def parse(
        self,
        llm_output: str,
        doc_map: Dict[str, Dict]
    ) -> Answer:
        """
        llm_output: raw text from LLM
        doc_map: output of ContextBuilder
        """

        found_doc_ids = self.CITATION_PATTERN.findall(llm_output)
        unique_doc_ids = list(set(found_doc_ids))

        if not unique_doc_ids:
            return Answer(
                text=llm_output,
                citations=[],
                is_valid=False,
                error="No citations found in LLM output",
            )

        citations: List[Citation] = []

        for doc_id in unique_doc_ids:
            if doc_id not in doc_map:
                return Answer(
                    text=llm_output,
                    citations=[],
                    is_valid=False,
                    error=f"Invalid citation reference: {doc_id}",
                )

            meta = doc_map[doc_id]

            citations.append(
                Citation(
                    doc_id=doc_id,
                    point_id=meta["point_id"],
                    score=meta["score"],
                    metadata=meta["metadata"],
                )
            )

        return Answer(
            text=llm_output,
            citations=citations,
            is_valid=True,
        )
