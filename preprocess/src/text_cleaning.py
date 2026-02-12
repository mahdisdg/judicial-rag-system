import re
from typing import Optional
from shekar import Normalizer

# Initialize shekar normalizer
_NORMALIZER = Normalizer()

def normalize_text(text: Optional[str]) -> str:
    """
    Normalizes Persian text using 'shekar'.
    - Removes zero-width spaces where unnecessary.
    - Standardizes spacing.
    - Standardizes quotes and punctuation.
    """
    if not text:
        return ""
    
    # Basic whitespace cleanup before passing to shekar
    text = str(text).strip()
    
    # Use Shekar for standard normalization
    text = _NORMALIZER.normalize(text)
    
    # Ensure single newlines are preserved, but multiple are collapsed
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def to_english_digits(text: str) -> str:
    """
    Converts Persian/Arabic digits to English digits.
    """
    if not text:
        return ""
    
    persian_map = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    arabic_map = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
    
    return text.translate(persian_map).translate(arabic_map)