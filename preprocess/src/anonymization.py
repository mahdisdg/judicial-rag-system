import re
import hmac
import hashlib
from typing import List, Dict

class Anonymizer:
    def __init__(self, secret: str):
        self.secret = secret
        self.mapping: Dict[str, str] = {}
        
        # --- PATTERNS TO PRESERVE ---
        self.legal_refs_re = re.compile(r"(?:\bماده|\bاصل|\bبند|\bتبصره)\s*[0-9]+(?:\s*[ا-ی])?")
        self.date_re = re.compile(r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b")
        self.money_re = re.compile(r"(?:\d{1,3}(?:[,]\d{3})*|\d+)\s*(?:ریال|تومان|میلیون|میلیارد)")

        # --- PATTERNS TO REDACT ---
        self.name_context_re = re.compile(
            r"(?:آقای|خانم|وکیل|موکل|نامبرده|خواهان|خوانده|متهم)\s+" 
            r"([آ-ی\s\u200c]{3,30}?)"                                  
            r"(?=\s+(?:فرزند|به وکالت|به نشانی|مقیم|با وکالت|کد ملی|شماره شناسنامه))" 
        )
        self.dashed_name_re = re.compile(r"^[-]\s*([آ-ی\s\u200c]+)$", re.MULTILINE)
        self.sensitive_num_re = re.compile(r"\b\d{10,}\b") 
        self.case_id_context_re = re.compile(r"(?:شماره|کلاسه|دادنامه)\s*[:\s]\s*(\d+)")

    def _get_tag(self, category: str, value: str) -> str:
        value_clean = value.strip()
        if value_clean in self.mapping:
            return self.mapping[value_clean]
        digest = hmac.new(self.secret.encode(), value_clean.encode(), hashlib.sha256).hexdigest()[:8]
        tag = f"[{category}_{digest}]"
        self.mapping[value_clean] = tag
        return tag

    def process(self, text: str) -> str:
        if not text:
            return ""

        # PROTECTION
        protected_map = {}
        def protect(match):
            val = match.group(0)
            key = f"__PROTECTED_{len(protected_map)}__"
            protected_map[key] = val
            return key

        text = self.date_re.sub(protect, text)
        text = self.legal_refs_re.sub(protect, text)
        text = self.money_re.sub(protect, text)

        # REDACTION
        def redact_name(match):
            full_match = match.group(0)
            name_part = match.group(1)
            if len(name_part.split()) > 5: return full_match 
            tag = self._get_tag("PERSON", name_part)
            return full_match.replace(name_part, tag)

        text = self.name_context_re.sub(redact_name, text)
        text = self.dashed_name_re.sub(lambda m: f"- {self._get_tag('PERSON', m.group(1))}", text)
        text = self.sensitive_num_re.sub(lambda m: self._get_tag("NUM", m.group(0)), text)
        def redact_case_id(match):
            return match.group(0).replace(match.group(1), self._get_tag("CASE_ID", match.group(1)))
        text = self.case_id_context_re.sub(redact_case_id, text)

        # RESTORATION
        for key, val in protected_map.items():
            text = text.replace(key, val)

        return text

def extract_related_laws(text: str) -> List[str]:
    """
    Extracts legal citations. 
    Improved Regex: Stops before common connectors like " و ماده" (and Article)
    to avoid capturing long unrelated strings.
    """
    # Explanation:
    # 1. Starts with "ماده X" or "اصل X"
    # 2. Optional "مکرر"
    # 3. "قانون"
    # 4. Captures the Law Name until it hits a conjunction (و | که | به) or another "ماده"
    law_re = re.compile(
        r"((?:ماده|اصل)\s+\d+(?:\s+مکرر)?\s+قانون\s+[^0-9\n\r]+?)(?=\s+(?:و|که|به|ماده|تبصره|$))"
    )
    
    matches = law_re.findall(text)
    
    # Remove strict punctuation or extra spaces at the end
    cleaned_matches = []
    for m in matches:
        clean = m.strip()
        if len(clean) < 100: # Sanity check for length
            cleaned_matches.append(clean)
            
    return list(set(cleaned_matches))