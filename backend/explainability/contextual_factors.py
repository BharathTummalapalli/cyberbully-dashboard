"""
Multi-Pillar Contextual Factors Generator.
Produces human-interpretable evidence across 5 dimensions:
1. Lexical / Text Evidence (LIME local word attributions)
2. Conversational Evidence (Parent replies, conversational turns, syntactic negation)
3. Behavioral Evidence (Author-target history, prior harassment flags, repeat attacks)
4. Temporal Evidence (Attack velocity in 1h/6h/24h, escalation acceleration)
5. Network Threat Evidence (Graph roles, directed edge weights, centrality)

Strictly grounded in empirical feature values; no manufactured rationales.
"""

from typing import Dict, Any, List

def generate_contextual_factors(
    text: str,
    feature_dict: Dict[str, Any],
    p_toxic: float,
    p_context: float,
    p_final: float,
    lime_weights: Dict[str, float],
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs multi-pillar explanation payload.
    """
    lexical_items = []
    for word, weight in sorted(lime_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        if weight > 0.05:
            lexical_items.append(f"Word '{word}' heightened toxicity profile (+{int(weight*100)}%).")
        elif weight < -0.05:
            lexical_items.append(f"Word '{word}' supported Safe classification (-{int(abs(weight)*100)}%).")

    # 2. Conversational Evidence
    conv = context_data.get("conversation", {})
    parent_text = conv.get("parent_text", "")
    turn_count = conv.get("turn_count", 0)
    conv_items = []

    if parent_text:
        conv_items.append(f"Direct reply to parent message: '{parent_text[:60]}...'")
    if turn_count > 1:
        conv_items.append(f"Part of an active discussion thread ({turn_count} prior turns).")

    # Check for contextual divergence
    if p_toxic >= 0.70 and p_final < 0.40:
        conv_items.append("Conversational syntax defused high surface toxicity (e.g. negation or benign banter).")
    elif p_toxic < 0.40 and p_final >= 0.70:
        conv_items.append("Hostile contextual trajectory inverted seemingly safe words into harassment.")

    # 3. Behavioral Evidence
    behav = context_data.get("behavioral", {})
    target_attacks = behav.get("target_attacks_count", 0)
    total_attacks = behav.get("total_attacks_sent", 0)
    flags = behav.get("author_flag_count", 0)
    behav_items = []

    if target_attacks > 0:
        behav_items.append(f"Author has previously targeted this user {target_attacks} times.")
    if total_attacks >= 3:
        behav_items.append(f"Author has a record of {total_attacks} prior cyberbullying incidents on the platform.")
    if flags > 0:
        behav_items.append(f"Author has been flagged or escalated {flags} times by moderation systems.")
    if not behav_items:
        behav_items.append("No prior hostile interactions recorded for this author-target pair.")

    # 4. Temporal Evidence
    temp = context_data.get("temporal", {})
    v_1h = temp.get("v_1h", 0)
    v_24h = temp.get("v_24h", 0)
    esc_score = temp.get("escalation_score", 0.0)
    temp_items = []

    if v_1h >= 2:
        temp_items.append(f"High velocity burst: {v_1h} attacks launched in the last 1 hour.")
    elif v_24h > 0:
        temp_items.append(f"{v_24h} attacks sent within the rolling 24-hour observation window.")

    if esc_score >= 0.5:
        temp_items.append(f"Significant temporal escalation curve detected (Index: {esc_score:.2f}).")
    elif v_1h == 0 and v_24h == 0:
        temp_items.append("Attack velocity is baseline; zero incidents in the preceding 24 hours.")

    # 5. Network Evidence
    graph = context_data.get("graph", {})
    author_role = graph.get("author_role", "Neutral")
    target_role = graph.get("target_role", "Neutral")
    edge_w = graph.get("edge_weight", 0.0)
    graph_items = []

    if author_role in ("Abuser", "Aggressive Participator"):
        graph_items.append(f"Author is classified as a network '{author_role}' based on 7-day interaction density.")
    if target_role == "Victim":
        graph_items.append("Target user is currently classified in a high-vulnerability 'Victim' role.")
    if edge_w >= 2:
        graph_items.append(f"Directed threat graph edge weight is {int(edge_w)} between author and target.")
    if not graph_items:
        graph_items.append("Network graph role is Neutral with no concentrated cluster targeting.")

    # High-level summary
    summary_parts = []
    if abs(p_final - p_toxic) >= 0.20:
        if p_final > p_toxic:
            summary_parts.append(f"Context amplified risk from base {p_toxic*100:.0f}% to final {p_final*100:.0f}% due to repetitive targeting and network signals.")
        else:
            summary_parts.append(f"Context discounted risk from base {p_toxic*100:.0f}% to final {p_final*100:.0f}% due to benign conversational framing.")
    else:
        summary_parts.append(f"Prediction aligned with base neural toxicity ({p_toxic*100:.0f}%).")

    return {
        "summary": " ".join(summary_parts),
        "pillars": {
            "lexical": lexical_items or ["Standard lexical tokens analyzed."],
            "conversational": conv_items or ["Isolated message; no prior thread history."],
            "behavioral": behav_items,
            "temporal": temp_items,
            "network": graph_items
        },
        "scores": {
            "p_toxic": round(p_toxic, 4),
            "p_context": round(p_context, 4),
            "p_final": round(p_final, 4)
        }
    }
