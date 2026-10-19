"""
Unified Multi-Pillar Explainer.
Coordinates LIME local feature attributions with the 5-pillar Contextual Attribution Engine:
- Pillar 1: Lexical / Text Evidence
- Pillar 2: Conversational Evidence
- Pillar 3: Behavioral Evidence
- Pillar 4: Temporal Evidence
- Pillar 5: Network Threat Evidence
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional
import numpy as np

from .contextual_factors import generate_contextual_factors

logger = logging.getLogger(__name__)

# Fallback keywords for local mock attribution
TOXIC_KEYWORDS = {
    "trash": 0.72, "loser": 0.85, "ugly": 0.81, "stupid": 0.68,
    "idiot": 0.79, "fat": 0.74, "gross": 0.65, "worthless": 0.88,
    "brainless": 0.76, "troll": 0.55, "annoying": 0.58, "hateful": 0.70,
    "jerk": 0.62, "pathetic": 0.82, "hate": 0.67, "banned": 0.52,
    "toxic": 0.69, "parasite": 0.80, "worst": 0.64, "delete": 0.48,
    "kill": 0.92, "disappointment": 0.71, "garbage": 0.59, "shame": 0.50
}

class MultiPillarExplainer:
    def __init__(self):
        self._lime_explainer = None
        try:
            from lime.lime_text import LimeTextExplainer
            self._lime_explainer = LimeTextExplainer(class_names=["Safe", "Cyberbullying"])
        except ImportError:
            self._lime_explainer = None

    def explain(
        self,
        text: str,
        feature_dict: Dict[str, Any],
        p_toxic: float,
        p_context: float,
        p_final: float,
        context_data: Dict[str, Any],
        predictor_fn = None,
        skip_lime: bool = False
    ) -> Dict[str, Any]:
        """
        Generates full explainability payload.
        """
        lime_weights = {}
        highlighted = []

        # 1. Compute LIME text attributions if available
        if not skip_lime and self._lime_explainer is not None and predictor_fn is not None:
            try:
                exp = self._lime_explainer.explain_instance(
                    text,
                    predictor_fn,
                    num_features=5,
                    num_samples=250
                )
                for word, weight in exp.as_list():
                    lime_weights[word] = round(float(weight), 3)
                    if weight > 0.05:
                        highlighted.append(word)
            except Exception as e:
                logger.warning(f"LIME attribution skipped: {e}")

        # Fallback keyword attribution if LIME was skipped or failed
        if not lime_weights:
            words = re.findall(r"\b\w+\b", text.lower())
            for w in words:
                if w in TOXIC_KEYWORDS:
                    lime_weights[w] = round(TOXIC_KEYWORDS[w], 3)
                    highlighted.append(w)

        # 2. Generate 5-Pillar Contextual Factors
        factor_report = generate_contextual_factors(
            text=text,
            feature_dict=feature_dict,
            p_toxic=p_toxic,
            p_context=p_context,
            p_final=p_final,
            lime_weights=lime_weights,
            context_data=context_data
        )

        # Compile flat explanations list for backward compatibility with frontend
        flat_explanations = []
        for pillar_name in ["lexical", "conversational", "behavioral", "temporal", "network"]:
            items = factor_report["pillars"].get(pillar_name, [])
            flat_explanations.extend(items)

        return {
            "lime_weights": lime_weights,
            "highlighted_words": highlighted,
            "pillars": factor_report["pillars"],
            "summary": factor_report["summary"],
            "explanations": flat_explanations,
            "scores": factor_report["scores"]
        }
