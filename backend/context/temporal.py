"""
Temporal Context Extractor.
Calculates rolling attack velocities (1h, 6h, 24h, 7d), time since last attack, and escalation rate.
Strictly ensures temporal leakage prevention: only records prior to current_timestamp are queried.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

class TemporalContextExtractor:
    def __init__(self):
        pass

    def extract(
        self,
        author: str,
        timestamp: datetime,
        history_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Calculates temporal attack activity features.

        Args:
            author: Author identifier.
            timestamp: Reference timestamp.
            history_records: Past logs.

        Returns:
            Dict containing:
                - v_1h: int (attacks sent in last 1 hour)
                - v_6h: int (attacks sent in last 6 hours)
                - v_24h: int (attacks sent in last 24 hours)
                - v_7d: int (attacks sent in last 7 days)
                - time_since_prev_attack_hours: float (hours since author's last attack, or 168.0 if none)
                - escalation_rate: float ((v_1h + 1) / (v_24h / 24 + 1))
                - escalation_score: float (0.0 to 1.0 normalized index)
        """
        defaults = {
            "v_1h": 0,
            "v_6h": 0,
            "v_24h": 0,
            "v_7d": 0,
            "time_since_prev_attack_hours": 168.0,
            "escalation_rate": 1.0,
            "escalation_score": 0.0
        }

        if not history_records or author == "@anonymous":
            return defaults

        # Filter strictly strictly before current timestamp (TEMPORAL LEAKAGE PREVENTION)
        author_attacks = []
        for r in history_records:
            r_author = r.get("author") or r.get("user_id") or r.get("username")
            if r_author != author:
                continue

            is_cyberbullying = (
                r.get("prediction") == "Cyberbullying" or 
                r.get("label") == "Cyberbullying" or 
                float(r.get("final_score") or r.get("toxicity_score") or 0.0) >= 0.70
            )
            if not is_cyberbullying:
                continue

            r_ts = r.get("timestamp")
            if isinstance(r_ts, str):
                try:
                    r_ts = datetime.fromisoformat(r_ts)
                except Exception:
                    continue

            # Strict temporal inequality
            if r_ts and r_ts < timestamp:
                author_attacks.append(r_ts)

        if not author_attacks:
            return defaults

        # Sort descending (most recent first)
        author_attacks.sort(reverse=True)

        t_1h = timestamp - timedelta(hours=1)
        t_6h = timestamp - timedelta(hours=6)
        t_24h = timestamp - timedelta(hours=24)
        t_7d = timestamp - timedelta(days=7)

        v_1h = sum(1 for ts in author_attacks if ts >= t_1h)
        v_6h = sum(1 for ts in author_attacks if ts >= t_6h)
        v_24h = sum(1 for ts in author_attacks if ts >= t_24h)
        v_7d = sum(1 for ts in author_attacks if ts >= t_7d)

        # Time since most recent attack in hours
        most_recent_attack = author_attacks[0]
        time_since_prev = max(0.0, (timestamp - most_recent_attack).total_seconds() / 3600.0)

        # Numerically stable escalation rate
        # Baseline = 1.0 when velocities are 0.
        escalation_rate = (v_1h + 1.0) / ((v_24h / 24.0) + 1.0)

        # Normalized escalation score: 0 when no recent 1h acceleration, up to 1.0 for surging attacks
        if v_1h > 0:
            escalation_score = min(1.0, max(0.0, (escalation_rate - 1.0) / 4.0))
        else:
            escalation_score = 0.0

        return {
            "v_1h": v_1h,
            "v_6h": v_6h,
            "v_24h": v_24h,
            "v_7d": v_7d,
            "time_since_prev_attack_hours": round(time_since_prev, 2),
            "escalation_rate": round(escalation_rate, 4),
            "escalation_score": round(escalation_score, 4)
        }
