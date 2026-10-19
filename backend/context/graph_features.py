"""
Threat Graph Feature Extractor.
Extracts numerical topological metrics from the rolling 7-day NetworkX interaction graph:
- In/Out degree centralities, edge weights, clustering coefficients, and role indicators.
Strictly ensures temporal leakage prevention: the graph snapshot contains ONLY incidents prior to current_timestamp.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import networkx as nx

class GraphFeatureExtractor:
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
        Builds a NetworkX directed graph from incidents in the 7 days prior to timestamp,
        and extracts topological network features.

        Args:
            author: Author identifier.
            target_user: Target identifier.
            timestamp: Reference event timestamp.
            history_records: Past logs.

        Returns:
            Dict of graph features:
                - author_out_centrality: float
                - target_in_centrality: float
                - edge_weight: float (attacks author -> target in last 7 days)
                - author_clustering: float
                - author_role: str ('Abuser', 'Victim', 'Aggressive Participator', 'Neutral')
                - target_role: str
                - author_role_abuser: float (1.0 or 0.0)
                - author_role_participator: float (1.0 or 0.0)
                - target_role_victim: float (1.0 or 0.0)
                - graph_threat_score: float (composite 0.0 to 1.0)
        """
        defaults = {
            "author_out_centrality": 0.0,
            "target_in_centrality": 0.0,
            "edge_weight": 0.0,
            "author_clustering": 0.0,
            "author_role": "Neutral",
            "target_role": "Neutral",
            "author_role_abuser": 0.0,
            "author_role_participator": 0.0,
            "target_role_victim": 0.0,
            "graph_threat_score": 0.0
        }

        if not history_records:
            return defaults

        # Filter for rolling 7-day window STRICTLY PRIOR to timestamp (NO LEAKAGE)
        seven_days_ago = timestamp - timedelta(days=7)
        G = nx.DiGraph()

        attacks_sent_7d = {}
        attacks_received_7d = {}

        for r in history_records:
            is_cyberbullying = (
                r.get("prediction") == "Cyberbullying" or 
                r.get("label") == "Cyberbullying" or 
                float(r.get("final_score") or r.get("toxicity_score") or 0.0) >= 0.70
            )
            if not is_cyberbullying:
                continue

            u = r.get("author") or r.get("user_id") or r.get("username")
            v = r.get("target_user")
            if not u or not v or u == "@anonymous" or v == "@anonymous" or u == v:
                continue

            r_ts = r.get("timestamp")
            if isinstance(r_ts, str):
                try:
                    r_ts = datetime.fromisoformat(r_ts)
                except Exception:
                    continue

            # Strict temporal check: seven_days_ago <= r_ts < timestamp
            if r_ts and seven_days_ago <= r_ts < timestamp:
                attacks_sent_7d[u] = attacks_sent_7d.get(u, 0) + 1
                attacks_received_7d[v] = attacks_received_7d.get(v, 0) + 1

                if G.has_edge(u, v):
                    G[u][v]["weight"] += 1
                else:
                    G.add_edge(u, v, weight=1)

        if len(G.nodes) == 0:
            return defaults

        # Calculate centralities
        out_centrality = nx.out_degree_centrality(G)
        in_centrality = nx.in_degree_centrality(G)

        # Clustering coefficient on undirected projection
        try:
            clustering_coef = nx.clustering(G.to_undirected())
        except Exception:
            clustering_coef = {}

        # Node metrics for author & target
        author_out = float(out_centrality.get(author, 0.0))
        target_in = float(in_centrality.get(target_user, 0.0))
        author_clust = float(clustering_coef.get(author, 0.0))

        edge_w = 0.0
        if G.has_edge(author, target_user):
            edge_w = float(G[author][target_user].get("weight", 1))

        # Role assignment based on rolling 7-day attacks sent/received
        sent_a = attacks_sent_7d.get(author, 0)
        rec_a = attacks_received_7d.get(author, 0)
        if sent_a >= 3 and rec_a >= 3:
            author_role = "Aggressive Participator"
        elif sent_a >= 3:
            author_role = "Abuser"
        elif rec_a >= 3:
            author_role = "Victim"
        else:
            author_role = "Neutral"

        sent_t = attacks_sent_7d.get(target_user, 0)
        rec_t = attacks_received_7d.get(target_user, 0)
        if sent_t >= 3 and rec_t >= 3:
            target_role = "Aggressive Participator"
        elif sent_t >= 3:
            target_role = "Abuser"
        elif rec_t >= 3:
            target_role = "Victim"
        else:
            target_role = "Neutral"

        # One-hot encodings
        author_role_abuser = 1.0 if author_role == "Abuser" else 0.0
        author_role_participator = 1.0 if author_role == "Aggressive Participator" else 0.0
        target_role_victim = 1.0 if target_role in ("Victim", "Aggressive Participator") else 0.0

        # Composite graph threat score
        # Elevated when author is high out-centrality or has established directed edge to target
        threat_score = min(
            1.0,
            (author_out * 0.4) + (min(edge_w, 5.0) / 5.0 * 0.4) + (0.2 if author_role == "Abuser" else 0.0)
        )

        return {
            "author_out_centrality": round(author_out, 4),
            "target_in_centrality": round(target_in, 4),
            "edge_weight": float(edge_w),
            "author_clustering": round(author_clust, 4),
            "author_role": author_role,
            "target_role": target_role,
            "author_role_abuser": author_role_abuser,
            "author_role_participator": author_role_participator,
            "target_role_victim": target_role_victim,
            "graph_threat_score": round(threat_score, 4)
        }
