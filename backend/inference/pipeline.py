"""
Unified End-to-End Context-Aware Inference Pipeline.
Coordinates:
1. Target Resolution
2. Context Collection (Conversation, Behavioral, Temporal, Network Graph)
3. Base Toxicity Inference (Toxic-BERT -> P_toxic)
4. Contextual Sub-Model Inference (DistilBERT + Auxiliary Heads -> P_context)
5. Learned Fusion Feature Assembly, Normalization & MLP Scoring (-> P_final)
6. Decision Engine Evaluation (Safe / Uncertain / Cyberbullying)
7. Multi-Pillar Explainability (LIME + 5 Contextual Pillars)
8. Audit Trail Persistence to Database
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from utils import resolve_target
from context.collector import ContextCollector
from models.toxic_model import ToxicModel
from models.context_model import ContextModelWrapper
from models.fusion_model import FusionModelWrapper, FEATURE_SCHEMA
from inference.decision_engine import DecisionEngine
from explainability.explainer import MultiPillarExplainer
import db

logger = logging.getLogger(__name__)

class InferencePipeline:
    def __init__(self):
        self.collector = ContextCollector()
        self.toxic_model = ToxicModel()
        
        weights_dir = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
        context_weights = os.path.join(weights_dir, "context_submodel.pt")
        fusion_weights = os.path.join(weights_dir, "fusion_mlp.pt")
        scaler_path = os.path.join(weights_dir, "feature_scaler.json")

        self.context_model = ContextModelWrapper(
            weights_path=context_weights if os.path.exists(context_weights) else None
        )
        self.fusion_model = FusionModelWrapper(
            weights_path=fusion_weights if os.path.exists(fusion_weights) else None,
            scaler_path=scaler_path if os.path.exists(scaler_path) else None
        )
        self.decision_engine = DecisionEngine()
        self.explainer = MultiPillarExplainer()

    def reload_models(self):
        """Reloads trained model checkpoints and scaler if updated."""
        weights_dir = os.path.join(os.path.dirname(__file__), "..", "models", "weights")
        context_weights = os.path.join(weights_dir, "context_submodel.pt")
        fusion_weights = os.path.join(weights_dir, "fusion_mlp.pt")
        scaler_path = os.path.join(weights_dir, "feature_scaler.json")

        if os.path.exists(context_weights):
            self.context_model.load_weights(context_weights)
        if os.path.exists(scaler_path):
            self.fusion_model.load_scaler(scaler_path)
        if os.path.exists(fusion_weights):
            self.fusion_model.load_weights(fusion_weights)

    def process(
        self,
        text: str,
        author: str = "@anonymous",
        target_user: Optional[str] = None,
        parent_username: Optional[str] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        use_real_model: bool = True,
        skip_lime: bool = True,
        persist: bool = True,
        session = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end classification pipeline for an incoming post or comment.
        """
        if not text or not text.strip():
            raise ValueError("Input text cannot be empty or whitespace only.")

        event_ts = datetime.utcnow()
        post_id = f"evt-{str(uuid.uuid4())[:8]}"

        # Step 1: Target Resolution
        resolved_target = resolve_target(
            text=text,
            parent_username=target_user or parent_username,
            author_username=author
        )

        # Step 2: Fetch recent history for context collection (strictly retrospective)
        history_records = []
        try:
            # Query recent posts from database
            with (session or db.SessionLocal()) as s:
                history_records = db.get_all_posts(s)
        except Exception as e:
            logger.warning(f"Could not retrieve history records from DB: {e}")

        # Step 3: Context Collection across 4 dimensions
        context_data = self.collector.collect(
            text=text,
            author=author,
            target_user=resolved_target,
            timestamp=event_ts,
            thread_id=thread_id,
            parent_id=parent_id,
            history_records=history_records
        )
        flat_features = context_data["flat_features"]

        # Step 4: Base Neural Toxicity Inference (Toxic-BERT)
        p_toxic = self.toxic_model.predict_p_toxic(text, use_real_model=use_real_model)

        # Step 5: Contextual Sub-Model Inference (DistilBERT + Auxiliary Heads)
        ctx_snippet = context_data.get("context_snippet", "")
        ctx_preds = self.context_model.predict(text, ctx_snippet)
        p_context = ctx_preds["p_context"]

        # Step 6: Assemble Complete Feature Dictionary
        feature_dict = {
            **flat_features,
            "p_toxic": p_toxic,
            "p_context": p_context,
            "is_targeted": ctx_preds["is_targeted"],
            "is_repeated": ctx_preds["is_repeated"],
            "is_escalating": ctx_preds["is_escalating"],
            "is_sarcastic": ctx_preds["is_sarcastic"],
            "is_threatening": ctx_preds["is_threatening"],
        }

        # Step 7: Learned Fusion MLP Scoring (P_final)
        p_final = self.fusion_model.predict_p_final(feature_dict)

        # Step 8: Decision Engine Evaluation
        decision = self.decision_engine.evaluate(
            p_final=p_final,
            p_toxic=p_toxic,
            p_context=p_context,
            context_signals=[
                f"Attack velocity: {int(feature_dict['v_1h'])} attacks in last 1 hour",
                f"Author graph role: {context_data['graph']['author_role']}"
            ]
        )

        # Step 9: Multi-Pillar Explainability
        explain_payload = self.explainer.explain(
            text=text,
            feature_dict=feature_dict,
            p_toxic=p_toxic,
            p_context=p_context,
            p_final=p_final,
            context_data=context_data,
            skip_lime=skip_lime
        )

        # Step 10: Database Persistence
        db_payload = {
            "id": post_id,
            "text": text,
            "author": author,
            "target_user": resolved_target,
            "toxicity_score": p_toxic,
            "context_delta": round(p_final - p_toxic, 4),
            "p_toxic": p_toxic,
            "p_context": p_context,
            "final_score": p_final,
            "targeting_score": ctx_preds["is_targeted"],
            "repetition_score": feature_dict["repetition_ratio"],
            "escalation_score": feature_dict["escalation_score"],
            "graph_score": feature_dict["graph_threat_score"],
            "prediction": decision["label"],
            "status": decision["status"],
            "moderator_action": "Pending",
            "mode": "real" if (use_real_model and self.toxic_model.ml_available) else "mock",
            "model_version": self.toxic_model.version,
            "context_model_version": self.context_model.version,
            "fusion_model_version": self.fusion_model.version,
            "thread_id": thread_id,
            "parent_id": parent_id,
            "escalate_to_human": decision["escalate_to_human"],
            "escalation_reason": decision["escalation_reason"],
            "timestamp": event_ts.isoformat()
        }

        if persist:
            try:
                db.log_prediction(db_payload, session=session)
            except Exception as e:
                logger.error(f"Error persisting prediction log: {e}", exc_info=True)

        return {
            "id": post_id,
            "text": text,
            "author": author,
            "username": author,
            "target_user": resolved_target,
            "p_toxic": round(p_toxic, 4),
            "p_context": round(p_context, 4),
            "final_score": round(p_final, 4),
            "confidence": round(p_final * 100, 1),
            "prediction": decision["label"],
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
            "context_features": {
                "v_1h": int(feature_dict["v_1h"]),
                "v_24h": int(feature_dict["v_24h"]),
                "v_7d": int(feature_dict["v_7d"]),
                "escalation_rate": round(feature_dict["escalation_rate"], 2),
                "target_attacks_count": int(feature_dict["target_attacks_count"]),
                "author_role": context_data["graph"]["author_role"],
                "target_role": context_data["graph"]["target_role"],
                "edge_weight": feature_dict["edge_weight"]
            },
            "pillars": explain_payload["pillars"],
            "summary": explain_payload["summary"],
            "explanations": explain_payload["explanations"],
            "highlighted_words": explain_payload["highlighted_words"],
            "lime_weights": explain_payload["lime_weights"],
            "moderator_action": "Pending",
            "model_version": self.toxic_model.version,
            "context_model_version": self.context_model.version,
            "fusion_model_version": self.fusion_model.version,
            "timestamp": event_ts.isoformat()
        }

# Global Pipeline Singleton
pipeline = InferencePipeline()
