"""
Behavioral Context Extractor.
Analyzes author and target interaction history:
- Attacks sent, repeated targeting, distinct victims, prior moderation flags.
Strictly ensures temporal leakage prevention: only records prior to current_timestamp are queried.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional

class BehavioralContextExtractor:
    def __init__(self):
        pass

    def extract(
        self,
        author: str,
        target_user: str,
        timestamp: datetime,
        history_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Extracts behavioral history metrics.

        Args:
            author: Author identifier.
            target_user: Target user identifier.
            timestamp: Reference timestamp (for strictly temporal retrospective filtering).
            history_records: List of past log records.

        Returns:
            Dict containing behavioral features:
                - total_attacks_sent: int
                - target_attacks_count: int
                - distinct_targets_count: int
                - author_target_interactions: int
                - author_flag_count: int
                - author_hidden_count: int
                - author_approved_count: int
                - target_prior_victimization: int
                - repetition_ratio: float (fraction of interactions to this target that were attacks)
        """
        defaults = {
            "total_attacks_sent": 0,
            "target_attacks_count": 0,
            "distinct_targets_count": 0,
            "author_target_interactions": 0,
            "author_flag_count": 0,
            "author_hidden_count": 0,
            "author_approved_count": 0,
            "target_prior_victimization": 0,
            "repetition_ratio": 0.0
        }

        if not history_records or author == "@anonymous":
            return defaults

        # Filter strictly before current timestamp (TEMPORAL LEAKAGE PREVENTION)
        prior_records = []
        for r in history_records:
            r_ts = r.get("timestamp")
            if isinstance(r_ts, str):
                try:
                    r_ts = datetime.fromisoformat(r_ts)
                except Exception:
                    continue
            if r_ts and r_ts < timestamp:
                prior_records.append({**r, "parsed_ts": r_ts})

        total_attacks_sent = 0
        target_attacks_count = 0
        targets_set = set()
        author_target_interactions = 0
        author_flag_count = 0
        author_hidden_count = 0
        author_approved_count = 0
        target_prior_victimization = 0

        target_valid = (target_user and target_user != "@anonymous" and target_user != author)

        for r in prior_records:
            r_author = r.get("author") or r.get("user_id") or r.get("username")
            r_target = r.get("target_user")
            is_cyberbullying = (
                r.get("prediction") == "Cyberbullying" or 
                r.get("label") == "Cyberbullying" or 
                float(r.get("final_score") or r.get("toxicity_score") or 0.0) >= 0.70
            )
            is_flagged = (
                r.get("status") == "EscalatedToHuman" or 
                r.get("escalate_to_human") is True or 
                r.get("label") == "Uncertain"
            )
            mod_action = r.get("moderator_action")

            # Author metrics
            if r_author == author:
                # Count prior moderation audit
                if mod_action == "Hidden":
                    author_hidden_count += 1
                elif mod_action == "Approved":
                    author_approved_count += 1

                if is_flagged:
                    author_flag_count += 1

                if is_cyberbullying:
                    total_attacks_sent += 1
                    if r_target and r_target != "@anonymous":
                        targets_set.add(r_target)

                # Interactions specifically with this target
                if target_valid and r_target == target_user:
                    author_target_interactions += 1
                    if is_cyberbullying:
                        target_attacks_count += 1

            # Target victimization by any author
            if target_valid and r_target == target_user and is_cyberbullying:
                target_prior_victimization += 1

        repetition_ratio = 0.0
        if author_target_interactions > 0:
            repetition_ratio = target_attacks_count / author_target_interactions

        return {
            "total_attacks_sent": total_attacks_sent,
            "target_attacks_count": target_attacks_count,
            "distinct_targets_count": len(targets_set),
            "author_target_interactions": author_target_interactions,
            "author_flag_count": author_flag_count,
            "author_hidden_count": author_hidden_count,
            "author_approved_count": author_approved_count,
            "target_prior_victimization": target_prior_victimization,
            "repetition_ratio": round(repetition_ratio, 4)
        }
