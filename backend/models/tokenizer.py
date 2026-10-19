"""
Tokenizer and sequence preparation helper for Contextual Model.
Formats dual-sequence input: [CLS] comment [SEP] preceding_context [SEP]
"""

import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

_cached_tokenizer = None

def get_context_tokenizer(model_name: str = "distilbert-base-uncased"):
    """Loads and caches the Hugging Face tokenizer."""
    global _cached_tokenizer
    if _cached_tokenizer is not None:
        return _cached_tokenizer

    try:
        from transformers import AutoTokenizer
        _cached_tokenizer = AutoTokenizer.from_pretrained(model_name)
        return _cached_tokenizer
    except Exception as e:
        logger.error(f"Error loading tokenizer for {model_name}: {e}")
        return None

def format_context_input(text: str, context_snippet: str = "") -> str:
    """
    Formats comment and context into a standardized representation.
    """
    cleaned_text = (text or "").strip()
    cleaned_ctx = (context_snippet or "").strip()
    if cleaned_ctx:
        return f"{cleaned_text} [SEP] Context: {cleaned_ctx}"
    return cleaned_text
