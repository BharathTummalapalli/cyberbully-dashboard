"""
Multi-Source Context Collector.
Coordinates:
1. Conversation Context (Parent message, recent thread messages, turns, time deltas)
2. Behavioral Context (Targeting frequency, attacks sent, prior moderation actions)
3. Temporal Context (1h, 6h, 24h, 7d attack velocities, escalation rate)
4. Network Graph Context (Centrality, roles, edge weights, clustering)

Strictly enforces Temporal and Retrospective Isolation to prevent data leakage.
"""

from datetime import datetime
from typing import Dict, Any, List, Optional
import logging

from .conversation import ConversationContextExtractor
from .behavioral import BehavioralContextExtractor
from .temporal import TemporalContextExtractor
from .graph_features import GraphFeatureExtractor

logger = logging.getLogger(__name__)

class ContextCollector:
    def __init__(self):
        self.conv_extractor = ConversationContextExtractor()
        self.behavioral_extractor = BehavioralContextExtractor()
        self.temporal_extractor = TemporalContextExtractor()
        self.graph_extractor = GraphFeatureExtractor()

    def collect(
        self,
        text: str,
        author: str,
        target_user: str,
        timestamp: Optional[datetime] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        history_records: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Gathers all 4 context dimensions for an incoming comment.

        Args:
            text: Current comment text.
            author: Author handle.
            target_user: Target handle.
            timestamp: Event timestamp. If None, defaults to datetime.utcnow().
            thread_id: Optional conversation thread ID.
            parent_id: Optional direct parent comment ID.
            history_records: List of past log records (must only include records available at or before timestamp).

        Returns:
            Dict containing conversation, behavioral, temporal, graph, and flat_features.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        elif isinstance(timestamp, str):
            try:
                timestamp = datetime.fromisoformat(timestamp)
            except Exception:
                timestamp = datetime.utcnow()

        history = history_records or []

        # 1. Extract Conversation Context
        conv_ctx = self.conv_extractor.extract(
            text=text,
            author=author,
            target_user=target_user,
            timestamp=timestamp,
            thread_id=thread_id,
            parent_id=parent_id,
            history_records=history
        )

        # 2. Extract Behavioral Context
        behav_ctx = self.behavioral_extractor.extract(
            author=author,
            target_user=target_user,
            timestamp=timestamp,
            history_records=history
        )

        # 3. Extract Temporal Context
        temp_ctx = self.temporal_extractor.extract(
            author=author,
            timestamp=timestamp,
            history_records=history
        )

        # 4. Extract Graph Context
        graph_ctx = self.graph_extractor.extract(
            author=author,
            target_user=target_user,
            timestamp=timestamp,
            history_records=history
        )

        # Assemble flat numerical features dictionary
        flat = {
            # Conversation features
            "conv_turn_count": float(conv_ctx["turn_count"]),
            "conv_time_delta_seconds": float(conv_ctx["time_delta_seconds"]),
            
            # Behavioral features
            "total_attacks_sent": float(behav_ctx["total_attacks_sent"]),
            "target_attacks_count": float(behav_ctx["target_attacks_count"]),
            "distinct_targets_count": float(behav_ctx["distinct_targets_count"]),
            "author_target_interactions": float(behav_ctx["author_target_interactions"]),
            "author_flag_count": float(behav_ctx["author_flag_count"]),
            "author_hidden_count": float(behav_ctx["author_hidden_count"]),
            "author_approved_count": float(behav_ctx["author_approved_count"]),
            "target_prior_victimization": float(behav_ctx["target_prior_victimization"]),
            "repetition_ratio": float(behav_ctx["repetition_ratio"]),

            # Temporal features
            "v_1h": float(temp_ctx["v_1h"]),
            "v_6h": float(temp_ctx["v_6h"]),
            "v_24h": float(temp_ctx["v_24h"]),
            "v_7d": float(temp_ctx["v_7d"]),
            "time_since_prev_attack_hours": float(temp_ctx["time_since_prev_attack_hours"]),
            "escalation_rate": float(temp_ctx["escalation_rate"]),
            "escalation_score": float(temp_ctx["escalation_score"]),

            # Graph features
            "author_out_centrality": float(graph_ctx["author_out_centrality"]),
            "target_in_centrality": float(graph_ctx["target_in_centrality"]),
            "edge_weight": float(graph_ctx["edge_weight"]),
            "author_clustering": float(graph_ctx["author_clustering"]),
            "author_role_abuser": float(graph_ctx["author_role_abuser"]),
            "author_role_participator": float(graph_ctx["author_role_participator"]),
            "target_role_victim": float(graph_ctx["target_role_victim"]),
            "graph_threat_score": float(graph_ctx["graph_threat_score"]),
        }

        return {
            "timestamp": timestamp.isoformat(),
            "author": author,
            "target_user": target_user,
            "conversation": conv_ctx,
            "behavioral": behav_ctx,
            "temporal": temp_ctx,
            "graph": graph_ctx,
            "flat_features": flat,
            "context_snippet": conv_ctx["context_snippet"]
        }
