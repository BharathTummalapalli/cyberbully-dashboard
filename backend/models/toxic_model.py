"""
Base Neural Toxicity Model Wrapper.
Provides P_toxic probability from HuggingFace Toxic-BERT ('unitary/toxic-bert')
with keyword fallback lexicon.
"""

import os
import re
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

# Fallback toxic keyword weights
TOXIC_KEYWORDS = {
    "trash": 0.72, "loser": 0.85, "ugly": 0.81, "stupid": 0.68,
    "idiot": 0.79, "fat": 0.74, "gross": 0.65, "worthless": 0.88,
    "brainless": 0.76, "troll": 0.55, "annoying": 0.58, "hateful": 0.70,
    "jerk": 0.62, "pathetic": 0.82, "hate": 0.67, "banned": 0.52,
    "toxic": 0.69, "parasite": 0.80, "worst": 0.64, "delete": 0.48,
    "kill": 0.92, "disappointment": 0.71, "garbage": 0.59, "shame": 0.50
}

class ToxicModel:
    def __init__(self, model_name: str = "unitary/toxic-bert"):
        self.model_name = model_name
        self.version = "toxic-bert-v1"
        self._pipeline = None
        self.ml_available = False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            self.ml_available = True
        except ImportError:
            self.ml_available = False

    def load_model(self) -> bool:
        """Loads HuggingFace transformer pipeline if available."""
        if not self.ml_available:
            logger.warning("PyTorch/Transformers missing. ToxicModel will use Lexicon fallback.")
            return False

        if self._pipeline is not None:
            return True

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
            
            logger.info(f"Loading base toxicity model: {self.model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            device = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=device,
                top_k=None
            )
            logger.info("Base Toxic-BERT model loaded successfully.")
            return True
        except Exception as e:
            logger.error(f"Failed to load Toxic-BERT model: {e}. Falling back to Lexicon.", exc_info=True)
            self._pipeline = None
            return False

    def predict_p_toxic(self, text: str, use_real_model: bool = True) -> float:
        """
        Computes P_toxic in [0.0, 1.0].
        """
        if not text or not text.strip():
            return 0.0

        if use_real_model and self.ml_available:
            if self._pipeline is None:
                self.load_model()

            if self._pipeline is not None:
                try:
                    outputs = self._pipeline([text])[0]
                    # 'unitary/toxic-bert' returns labels: toxic, severe_toxic, obscene, threat, insult, identity_hate
                    toxic_score = 0.0
                    for label_dict in outputs:
                        if label_dict["label"] == "toxic":
                            toxic_score = float(label_dict["score"])
                            break
                    return max(0.0, min(1.0, toxic_score))
                except Exception as e:
                    logger.warning(f"Error in Toxic-BERT inference: {e}. Using Lexicon fallback.")

        # Keyword Lexicon Fallback
        words = re.findall(r"\b\w+\b", text.lower())
        matched = [w for w in set(words) if w in TOXIC_KEYWORDS]
        if matched:
            raw = 0.70 + sum(TOXIC_KEYWORDS[w] * 0.10 for w in matched)
            return min(0.98, raw)
        return 0.05
