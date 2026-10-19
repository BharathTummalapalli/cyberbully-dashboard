"""
Model Explainability and Inference Layer.

ARCHITECTURAL NOTE (VIVA DEFENSE):
This module provides model inference and LIME local feature attributions.
Base toxicity is predicted via HuggingFace's 'unitary/toxic-bert' (or a keyword lexicon fallback).
The base prediction is then modulated by the Rule-Based Context Adjustment Layer (in fusion.py)
via an additive clamped formula:
    S_fused = clamp(P_toxic + delta_neg + delta_slang + delta_sarcasm + delta_target, 0.0, 1.0)

NOTE ON MODEL TERMINOLOGY:
Deterministic linguistic heuristics are employed in this version.
Training a secondary parameterized contextual classifier is reserved for future work.
"""

import re
import random
import os
import logging
from typing import Dict, Any, List

# Try importing ML dependencies. Fallback to mock mode if not present or on low-resource machines.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    from lime.lime_text import LimeTextExplainer
    import numpy as np
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

import fusion

logger = logging.getLogger(__name__)

# List of toxic keywords for the fallback mock engine
TOXIC_KEYWORDS = {
    "trash": 0.72,
    "loser": 0.85,
    "ugly": 0.81,
    "stupid": 0.68,
    "idiot": 0.79,
    "fat": 0.74,
    "gross": 0.65,
    "worthless": 0.88,
    "brainless": 0.76,
    "troll": 0.55,
    "annoying": 0.58,
    "hateful": 0.70,
    "jerk": 0.62,
    "pathetic": 0.82,
    "hate": 0.67,
    "banned": 0.52,
    "toxic": 0.69,
    "parasite": 0.80,
    "worst": 0.64,
    "delete": 0.48,
    "kill": 0.92,
    "disappointment": 0.71,
    "garbage": 0.59,
    "shame": 0.50
}

# Global references for ML models
_nlp_pipeline = None
_explainer = None

def init_ml_model():
    """Initializes the HuggingFace model and LIME explainer if available."""
    global _nlp_pipeline, _explainer
    if not ML_AVAILABLE:
        logger.warning("ML dependencies missing. Backend will run in Keyword Fallback Mode.")
        return False
    
    try:
        model_name = "unitary/toxic-bert"
        logger.info(f"Loading HuggingFace model: {model_name}...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        device = 0 if torch.cuda.is_available() else -1
        _nlp_pipeline = pipeline(
            "text-classification", 
            model=model, 
            tokenizer=tokenizer, 
            device=device,
            top_k=None
        )
        
        _explainer = LimeTextExplainer(class_names=["Safe", "Cyberbullying"])
        logger.info("HuggingFace Model and LIME Explainer loaded successfully!")
        return True
    except Exception as e:
        logger.error(f"Failed to load HuggingFace model: {e}. Falling back to Keyword Fallback Mode.", exc_info=True)
        return False

def _validate_text(text: str):
    """Validates that input is a non-empty string and does not exceed 2000 characters."""
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    if not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only.")
    if len(text) > 2000:
        raise ValueError("Input text exceeds the maximum limit of 2000 characters.")

def get_base_mock_score(text: str) -> tuple[float, dict, list]:
    """Computes base toxicity probability using keyword matching without context rules."""
    words = re.findall(r'\b\w+\b', text.lower())
    matched_toxic = [w for w in set(words) if w in TOXIC_KEYWORDS]
    
    lime_weights = {}
    highlighted = []
    
    if matched_toxic:
        raw_conf = 70.0 + sum([TOXIC_KEYWORDS[w] * 10 for w in matched_toxic])
        p_toxic = min(raw_conf, 98.0) / 100.0
        
        matched_sorted = sorted(matched_toxic, key=lambda x: TOXIC_KEYWORDS[x], reverse=True)
        for w in matched_sorted[:5]:
            lime_weights[w] = round(TOXIC_KEYWORDS[w], 3)
            highlighted.append(w)
            
        neutral_words = [w for w in words if w not in TOXIC_KEYWORDS]
        random.shuffle(neutral_words)
        for w in neutral_words[:min(2, len(neutral_words))]:
            lime_weights[w] = round(random.uniform(0.01, 0.08), 3)
    else:
        p_toxic = round(random.uniform(0.03, 0.15), 4)
        neutral_words = list(set(words))
        random.shuffle(neutral_words)
        for w in neutral_words[:min(3, len(neutral_words))]:
            lime_weights[w] = round(random.uniform(-0.15, -0.01), 3)
            
    return p_toxic, lime_weights, highlighted

def get_lime_explanation(
    text: str, 
    use_real_model: bool = False, 
    bypass_context_rules: bool = False, 
    skip_lime: bool = False,
    target_user: str = "@anonymous",
    author_user: str = "@anonymous"
) -> Dict[str, Any]:
    """
    Main entry point for explainability.
    1. Computes base neural toxicity score P_toxic from Toxic-BERT (or keyword fallback).
    2. Modulates score using the Rule-Based Context Adjustment Layer (fusion.py) if not bypassed.
    3. Runs LIME feature attributions.
    """
    _validate_text(text)
    
    # -------------------------------------------------------------
    # 1. Base Model Inference (Toxic-BERT or Keyword Lexicon)
    # -------------------------------------------------------------
    p_toxic = 0.0
    mode = "mock"
    highlighted = []
    lime_weights = {}
    explanations = []

    if use_real_model and ML_AVAILABLE:
        if _nlp_pipeline is None:
            init_ml_model()

        if _nlp_pipeline is not None:
            try:
                mode = "real"
                def predictor_fn(texts):
                    pipeline_outputs = _nlp_pipeline(texts)
                    probs = []
                    for output in pipeline_outputs:
                        toxic_score = 0.0
                        for label_dict in output:
                            if label_dict['label'] == 'toxic':
                                toxic_score = label_dict['score']
                                break
                        probs.append([1.0 - toxic_score, toxic_score])
                    return np.array(probs)

                probs = predictor_fn([text])[0]
                p_toxic = float(probs[1])

                if not skip_lime:
                    try:
                        exp = _explainer.explain_instance(text, predictor_fn, num_features=5, num_samples=300)
                        for word, weight in exp.as_list():
                            lime_weights[word] = round(float(weight), 3)
                            if weight > 0.05:
                                highlighted.append(word)
                                explanations.append(f"Word '{word}' increased toxicity profile by {int(weight * 100)}%.")
                            elif weight < -0.05:
                                explanations.append(f"Word '{word}' supported Safe classification by {int(abs(weight) * 100)}%.")
                    except Exception as le:
                        logger.warning(f"LIME perturbation skipped due to error: {le}")
            except Exception as e:
                logger.error(f"Error in BERT inference: {e}. Falling back to lexicon.", exc_info=True)
                p_toxic, lime_weights, highlighted = get_base_mock_score(text)
                mode = "mock"
        else:
            p_toxic, lime_weights, highlighted = get_base_mock_score(text)
    else:
        p_toxic, lime_weights, highlighted = get_base_mock_score(text)

    # -------------------------------------------------------------
    # 2. Score Fusion via Rule-Based Context Adjustment Layer
    # -------------------------------------------------------------
    if bypass_context_rules:
        label = "Cyberbullying" if p_toxic >= 0.50 else "Safe"
        confidence = (p_toxic * 100.0) if label == "Cyberbullying" else ((1.0 - p_toxic) * 100.0)
        return {
            "prediction": label,
            "confidence": round(confidence, 1),
            "p_toxic": round(p_toxic, 4),
            "final_score": round(p_toxic, 4),
            "total_context_delta": 0.0,
            "explanations": explanations or [f"Base classifier assigned score {round(p_toxic, 3)}."],
            "highlighted": highlighted,
            "lime_weights": lime_weights,
            "mode": mode
        }

    fused = fusion.compute_fused_score(
        text=text,
        p_toxic=p_toxic,
        target_user=target_user,
        author_user=author_user
    )

    combined_exps = list(fused["explanations"])
    for exp in explanations:
        if exp not in combined_exps:
            combined_exps.append(exp)

    return {
        "prediction": fused["label"],
        "confidence": round(fused["confidence"], 1),
        "p_toxic": round(fused["p_toxic"], 4),
        "delta_neg": round(fused["delta_neg"], 2),
        "delta_slang": round(fused["delta_slang"], 2),
        "delta_sarcasm": round(fused["delta_sarcasm"], 2),
        "delta_target": round(fused["delta_target"], 2),
        "total_context_delta": round(fused["total_context_delta"], 4),
        "final_score": round(fused["s_fused"], 4),
        "status": fused["status"],
        "escalate_to_human": fused["escalate_to_human"],
        "escalation_reason": fused["escalation_reason"],
        "formula": fused["formula"],
        "explanations": combined_exps,
        "highlighted": highlighted,
        "lime_weights": lime_weights,
        "mode": mode
    }

def check_context_overrides(text: str) -> Dict[str, Any]:
    """
    Evaluates deterministic linguistic context rules (negation, slang, sarcasm)
    to check if an override is triggered. Retained for regression tests.
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+[']?\w*\b", text_lower)
    
    signals = fusion.analyze_context_signals(text)
    
    # Check for active (unnegated) toxic tokens
    toxic_indices = [i for i, w in enumerate(words) if w in fusion.TOXIC_TOKENS]
    active_toxic = []
    negated_toxic = []
    
    for idx in toxic_indices:
        w = words[idx]
        start = max(0, idx - 4)
        has_neg = any(words[j] in fusion.NEGATION_TOKENS for j in range(start, idx))
        if has_neg:
            negated_toxic.append(w)
        else:
            active_toxic.append(w)

    # 1. Sarcasm overrides everything to Cyberbullying
    if signals["delta_sarcasm"] > 0:
        return {
            "overridden": True,
            "prediction": "Cyberbullying",
            "explanation": "; ".join(signals["explanations"])
        }

    # 2. If there are active, unnegated insults, baseline must decide (no override)
    if len(active_toxic) > 0:
        return {
            "overridden": False,
            "prediction": "",
            "explanation": "Active unnegated insults present; standard classification applies."
        }

    # 3. Negation of all toxic words -> override to Safe
    if len(negated_toxic) > 0 and len(active_toxic) == 0:
        return {
            "overridden": True,
            "prediction": "Safe",
            "explanation": "; ".join(signals["explanations"])
        }

    # 4. Reclaimed slang without active insults -> override to Safe
    if signals["delta_slang"] < 0:
        return {
            "overridden": True,
            "prediction": "Safe",
            "explanation": "; ".join(signals["explanations"])
        }

    return {
        "overridden": False,
        "prediction": "",
        "explanation": "No contextual override triggered."
    }
