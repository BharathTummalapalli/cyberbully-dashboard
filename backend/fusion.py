"""
Learned Context-Aware Fusion Engine for Cyberbullying Detection.

UPGRADE SPECIFICATION:
Replaces legacy static rule-based clamping with the Learned Context Architecture:
1. Toxic-BERT -> P_toxic (Lexical & Direct Toxicity probability)
2. DistilBERT Context Sub-Model -> P_context + Multi-Task Auxiliary Outputs:
   - targeting_score (is_targeted)
   - repetition_score (is_repeated)
   - escalation_score (is_escalating)
   - sarcasm_score (is_sarcastic)
   - threat_score (is_threatening)
3. Multi-Source Context Collector -> 26 Numerical Features across:
   - Conversational Dynamics (turn count, response latency)
   - Behavioral History (prior attacks, target affinity, flag ratios)
   - Temporal Dynamics (attack velocity over 1h, 6h, 24h, 7d; escalation rate)
   - Social Network Graph (centrality metrics, edge weights, Abuser/Victim roles)
4. Context-Aware Fusion MLP -> P_final (Non-linear learned feature combination)
5. Decision Engine -> 3-Band Calibrated Escalation Policy

NOTE ON VIVA DEFENSE & BENCHMARKING:
The legacy heuristic rule engine is retained as `compute_rule_based_fused_score`
to power Tier 2 comparative evaluation against Tier 4 Learned Fusion.
"""

import re
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)

# =====================================================================
# 1. LEARNED CONTEXT-AWARE FUSION (PRIMARY PRODUCTION ENGINE)
# =====================================================================

def compute_fused_score(
    text: str, 
    p_toxic: Optional[float] = None, 
    target_user: str = "@anonymous", 
    author_user: str = "@anonymous",
    thread_id: Optional[str] = None,
    parent_id: Optional[str] = None,
    session = None
) -> Dict[str, Any]:
    """
    Executes learned multi-source score fusion:
    Toxic-BERT (P_toxic) + DistilBERT Context Model (P_context + 5 auxiliary heads)
    + Context Collector (26 features) -> Learned Fusion MLP -> P_final.
    """
    from inference.pipeline import pipeline
    from utils import resolve_target
    import db

    if not text or not text.strip():
        raise ValueError("Input text cannot be empty or whitespace only.")

    event_ts = datetime.utcnow()

    # Step 1: Target Resolution
    resolved_target = resolve_target(
        text=text,
        parent_username=target_user,
        author_username=author_user
    )

    # Step 2: Retrieve retrospective history for context collector
    history_records = []
    try:
        with (session or db.SessionLocal()) as s:
            history_records = db.get_all_posts(s)
    except Exception as e:
        logger.debug(f"Could not retrieve history records from DB: {e}")

    # Step 3: Multi-Source Context Collection
    context_data = pipeline.collector.collect(
        text=text,
        author=author_user,
        target_user=resolved_target,
        timestamp=event_ts,
        thread_id=thread_id,
        parent_id=parent_id,
        history_records=history_records
    )
    flat_features = context_data["flat_features"]

    # Step 4: Toxic-BERT Toxicity Probability
    if p_toxic is None:
        p_toxic = pipeline.toxic_model.predict_p_toxic(text, use_real_model=True)
    else:
        p_toxic = float(p_toxic)

    # Step 5: DistilBERT Context Sub-Model + Multi-Task Auxiliary Heads
    ctx_snippet = context_data.get("context_snippet", "")
    ctx_preds = pipeline.context_model.predict(text, ctx_snippet)
    p_context = float(ctx_preds["p_context"])

    # Step 6: Feature Assembly for Learned Fusion MLP
    feature_dict = {
        **flat_features,
        "p_toxic": p_toxic,
        "p_context": p_context,
        "is_targeted": float(ctx_preds["is_targeted"]),
        "is_repeated": float(ctx_preds["is_repeated"]),
        "is_escalating": float(ctx_preds["is_escalating"]),
        "is_sarcastic": float(ctx_preds["is_sarcastic"]),
        "is_threatening": float(ctx_preds["is_threatening"]),
    }

    # Step 7: Learned Fusion MLP Inference
    p_final = pipeline.fusion_model.predict_p_final(feature_dict)

    # Step 8: Calibrated 3-Band Decision Engine
    decision = pipeline.decision_engine.evaluate(
        p_final=p_final,
        p_toxic=p_toxic,
        p_context=p_context,
        context_signals=[
            f"Attack velocity: {int(feature_dict['v_1h'])} in 1h",
            f"Author graph role: {context_data['graph']['author_role']}"
        ]
    )

    # Compile contextual explanations
    explanations = []
    if ctx_preds["is_targeted"] > 0.5:
        explanations.append(f"Context sub-model flagged directed targeting against {resolved_target}.")
    if ctx_preds["is_repeated"] > 0.5:
        explanations.append("Context sub-model flagged repetitive harassment pattern.")
    if ctx_preds["is_escalating"] > 0.5:
        explanations.append("Context sub-model flagged conversational escalation.")
    if ctx_preds["is_sarcastic"] > 0.5:
        explanations.append("Context sub-model flagged sarcastic inversion.")
    if ctx_preds["is_threatening"] > 0.5:
        explanations.append("Context sub-model flagged implicit or explicit intimidation.")

    if feature_dict.get("v_1h", 0) > 2:
        explanations.append(f"High temporal attack velocity detected ({int(feature_dict['v_1h'])} attacks within 1h).")
    if context_data["graph"].get("author_role") == "Abuser":
        explanations.append("Network threat graph identifies author as a central Abuser.")

    formula_str = (
        f"FusionMLP(P_toxic={p_toxic:.4f}, P_context={p_context:.4f}, "
        f"v_1h={int(feature_dict['v_1h'])}, role={context_data['graph']['author_role']}) = {p_final:.4f}"
    )

    return {
        "p_toxic": round(p_toxic, 4),
        "p_context": round(p_context, 4),
        "s_fused": round(p_final, 4),
        "final_score": round(p_final, 4),
        "confidence": round(p_final * 100.0, 1),
        "label": decision["label"],
        "status": decision["status"],
        "escalate_to_human": decision["escalate_to_human"],
        "escalation_reason": decision["escalation_reason"],
        "auxiliary_scores": {
            "targeting": round(ctx_preds["is_targeted"], 4),
            "repetition": round(ctx_preds["is_repeated"], 4),
            "escalation": round(ctx_preds["is_escalating"], 4),
            "sarcasm": round(ctx_preds["is_sarcastic"], 4),
            "threat": round(ctx_preds["is_threatening"], 4)
        },
        "is_targeted": ctx_preds["is_targeted"],
        "is_repeated": ctx_preds["is_repeated"],
        "is_escalating": ctx_preds["is_escalating"],
        "is_sarcastic": ctx_preds["is_sarcastic"],
        "is_threatening": ctx_preds["is_threatening"],
        "context_features": flat_features,
        "author_role": context_data["graph"]["author_role"],
        "target_role": context_data["graph"]["target_role"],
        "explanations": explanations or ["Predictions derived from multi-pillar learned fusion."],
        "formula": formula_str,
        # Backward compatibility delta fields
        "delta_neg": 0.0,
        "delta_slang": 0.0,
        "delta_sarcasm": 0.0,
        "delta_target": 0.0,
        "total_context_delta": round(p_final - p_toxic, 4)
    }


# =====================================================================
# 2. LEGACY RULE-BASED CONTEXT LAYER (PRESERVED FOR TIER 2 DEFENSE)
# =====================================================================

NEGATION_TOKENS = [
    "not", "never", "no", "don't", "dont", "doesn't", "doesnt", 
    "isn't", "isnt", "aren't", "arent", "wasn't", "wasnt", 
    "weren't", "werent", "shouldn't", "shouldnt", "wouldn't", 
    "wouldnt", "couldn't", "couldnt", "can't", "cant", "cannot"
]

TOXIC_TOKENS = [
    "trash", "loser", "ugly", "stupid", "idiot", "fat", "gross", 
    "worthless", "brainless", "troll", "annoying", "hateful", "jerk", 
    "pathetic", "hate", "banned", "toxic", "parasite", "worst", 
    "delete", "kill", "disappointment", "garbage", "shame"
]

POSITIVE_SLANG_PATTERNS = [
    (r"\bkillin(?:g)?\s+it\b", -0.50, "Reclaimed slang 'killing it' detected"),
    (r"\bsick\s+(?:beat|skill|move|flow|track|play)\b", -0.50, "Colloquial praise 'sick [art/skill]' detected"),
    (r"\btrash\s+talk\b.*\b(?:friendly|fun|gg|game)\b", -0.45, "Friendly gaming banter 'trash talk' detected")
]

SARCASM_PATTERNS = [
    (r"\bsaid\s+no\s+one\s+ever\b", +0.50, "Trailing sarcastic qualifier 'said no one ever' detected"),
    (r"\bsaid\s+nobody\s+ever\b", +0.50, "Trailing sarcastic qualifier 'said nobody ever' detected"),
    (r"\byeah\s+right\b", +0.30, "Sarcastic acknowledgment 'yeah right' detected")
]

def analyze_context_signals(text: str, target_user: str = "@anonymous", author_user: str = "@anonymous") -> Dict[str, Any]:
    """
    Evaluates legacy rule-based contextual signals:
    - Negation lookback discount
    - Friendly slang / colloquial banter discount
    - Sarcasm inversion booster
    - Target directed hostility factor
    """
    text_lower = text.lower()
    words = re.findall(r"\b\w+[']?\w*\b", text_lower)
    
    delta_neg = 0.0
    delta_slang = 0.0
    delta_sarcasm = 0.0
    delta_target = 0.0
    explanations = []

    # 1. Slang check
    for pattern, delta, exp in POSITIVE_SLANG_PATTERNS:
        if re.search(pattern, text_lower):
            delta_slang = delta
            explanations.append(exp)
            break

    # 2. Negation lookback
    toxic_indices = [i for i, w in enumerate(words) if w in TOXIC_TOKENS]
    negated_toxic = []
    active_toxic = []

    for idx in toxic_indices:
        w = words[idx]
        start = max(0, idx - 4)
        has_neg = False
        neg_word = ""
        for j in range(start, idx):
            if words[j] in NEGATION_TOKENS:
                has_neg = True
                neg_word = words[j]
                break
        if has_neg:
            negated_toxic.append((w, neg_word))
        else:
            active_toxic.append(w)

    if negated_toxic and len(active_toxic) == 0:
        delta_neg = -0.55
        terms_str = ", ".join([f"'{n} {t}'" for t, n in negated_toxic])
        explanations.append(f"Syntactic negation resolved: {terms_str} discounts toxicity.")
    elif negated_toxic and len(active_toxic) > 0:
        delta_neg = -0.20
        explanations.append(f"Partial negation detected, but unnegated insults '{', '.join(active_toxic)}' remain active.")

    # 3. Sarcasm check
    for pattern, delta, exp in SARCASM_PATTERNS:
        if re.search(pattern, text_lower):
            delta_sarcasm = delta
            explanations.append(exp)
            break

    # 4. Target hostility factor
    if target_user and target_user != "@anonymous" and target_user != author_user:
        if len(toxic_indices) > 0 or delta_sarcasm > 0:
            delta_target = +0.10
            explanations.append(f"Direct personal mention of target {target_user} amplifies hostility risk.")

    return {
        "delta_neg": delta_neg,
        "delta_slang": delta_slang,
        "delta_sarcasm": delta_sarcasm,
        "delta_target": delta_target,
        "explanations": explanations
    }

def compute_rule_based_fused_score(
    text: str, 
    p_toxic: float, 
    target_user: str = "@anonymous", 
    author_user: str = "@anonymous"
) -> Dict[str, Any]:
    """
    Executes legacy additive clamped fusion for Tier 2 evaluation:
        S_fused = clamp(P_toxic + delta_neg + delta_slang + delta_sarcasm + delta_target, 0.0, 1.0)
    """
    signals = analyze_context_signals(text, target_user, author_user)
    d_neg = signals["delta_neg"]
    d_slang = signals["delta_slang"]
    d_sarcasm = signals["delta_sarcasm"]
    d_target = signals["delta_target"]
    
    raw_sum = p_toxic + d_neg + d_slang + d_sarcasm + d_target
    s_fused = max(0.0, min(1.0, raw_sum))
    
    if s_fused > 0.70:
        label = "Cyberbullying"
        escalate_to_human = False
        status_str = "AutoProcessed"
    elif s_fused >= 0.40:
        label = "Uncertain"
        escalate_to_human = True
        status_str = "EscalatedToHuman"
    else:
        label = "Safe"
        escalate_to_human = False
        status_str = "AutoProcessed"

    formula_str = (
        f"clamp({p_toxic:.4f} + ({d_neg:.2f}) + ({d_slang:.2f}) + "
        f"(+{d_sarcasm:.2f}) + (+{d_target:.2f}), 0.0, 1.0) = {s_fused:.4f}"
    )

    escalation_reason = ""
    if escalate_to_human:
        escalation_reason = (
            f"Confidence score {s_fused*100:.1f}% falls within uncertain decision band [40%, 70%]. "
            f"Context signals: {'; '.join(signals['explanations']) or 'Borderline base toxicity'}."
        )

    return {
        "p_toxic": p_toxic,
        "delta_neg": d_neg,
        "delta_slang": d_slang,
        "delta_sarcasm": d_sarcasm,
        "delta_target": d_target,
        "total_context_delta": d_neg + d_slang + d_sarcasm + d_target,
        "s_fused": s_fused,
        "final_score": s_fused,
        "confidence": s_fused * 100.0,
        "label": label,
        "status": status_str,
        "escalate_to_human": escalate_to_human,
        "escalation_reason": escalation_reason,
        "explanations": signals["explanations"],
        "formula": formula_str
    }

# Alias for backward compatibility
compute_legacy_rule_fusion = compute_rule_based_fused_score
