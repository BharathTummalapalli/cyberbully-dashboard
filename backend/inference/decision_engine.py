"""
Decision Engine for Cyberbullying Moderation.
Configurable three-band decision policy:
- P_final < 0.40           -> Safe (Auto-Approved)
- 0.40 <= P_final <= 0.70  -> Uncertain (EscalatedToHuman -> review_queue)
- P_final > 0.70           -> Cyberbullying (Moderation / Flagged)

Thresholds are explicitly parameterized and configurable via environment variables.
"""

import os
from typing import Dict, Any, List, Optional

class DecisionEngine:
    def __init__(
        self,
        safe_threshold: Optional[float] = None,
        bullying_threshold: Optional[float] = None
    ):
        # Allow environment variable override for calibration
        self.safe_threshold = (
            safe_threshold if safe_threshold is not None 
            else float(os.getenv("DECISION_THRESHOLD_SAFE", "0.40"))
        )
        self.bullying_threshold = (
            bullying_threshold if bullying_threshold is not None 
            else float(os.getenv("DECISION_THRESHOLD_BULLYING", "0.70"))
        )

    def evaluate(
        self,
        p_final: float,
        p_toxic: float = 0.0,
        p_context: float = 0.0,
        context_signals: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Classifies final probability into operational bands and generates escalation metadata.

        Args:
            p_final: Final cyberbullying probability in [0.0, 1.0].
            p_toxic: Base toxicity probability from Toxic-BERT.
            p_context: Contextual probability from Contextual Sub-Model.
            context_signals: List of human-readable contextual rationales.

        Returns:
            Dict containing:
                - label: 'Safe', 'Uncertain', or 'Cyberbullying'
                - status: 'AutoProcessed' or 'EscalatedToHuman'
                - escalate_to_human: bool
                - escalation_reason: str
                - decision_bands: Dict with active thresholds
        """
        p_final_clamped = max(0.0, min(1.0, float(p_final)))
        signals = context_signals or []

        if p_final_clamped > self.bullying_threshold:
            label = "Cyberbullying"
            status = "AutoProcessed"
            escalate_to_human = False
            escalation_reason = ""
        elif p_final_clamped >= self.safe_threshold:
            label = "Uncertain"
            status = "EscalatedToHuman"
            escalate_to_human = True
            
            reasons = []
            if signals:
                reasons.extend(signals[:2])
            else:
                reasons.append(f"Borderline score between base toxicity ({p_toxic*100:.1f}%) and context ({p_context*100:.1f}%)")
            
            escalation_reason = (
                f"Confidence score {p_final_clamped*100:.1f}% falls within uncertain decision band "
                f"[{self.safe_threshold*100:.0f}%, {self.bullying_threshold*100:.0f}%]. "
                f"Contextual signals: {'; '.join(reasons)}."
            )
        else:
            label = "Safe"
            status = "AutoProcessed"
            escalate_to_human = False
            escalation_reason = ""

        return {
            "label": label,
            "status": status,
            "escalate_to_human": escalate_to_human,
            "escalation_reason": escalation_reason,
            "decision_bands": {
                "safe_max": self.safe_threshold,
                "bullying_min": self.bullying_threshold
            }
        }
