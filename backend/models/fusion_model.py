"""
Learned Context-Aware Fusion Model Architecture.
Replaces the static clamped addition with an optimized, calibrated Neural Network.
Combines:
- Base toxicity probability P_toxic
- Contextual sub-model probability P_context + Multi-task auxiliary logits
- Conversational turn count and time deltas
- Behavioral targeting, flag history, and repetition ratios
- Temporal velocity and escalation metrics
- Network graph centralities, roles, and edge weights

Central Dynamic Feature Schema:
Dimensionality is dynamically computed: input_dim = len(FEATURE_SCHEMA).
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Central Feature Schema - Dynamic Input Dimension Specification
FEATURE_SCHEMA = [
    # 1. Core Model Signals
    "p_toxic",
    "p_context",
    "is_targeted",
    "is_repeated",
    "is_escalating",
    "is_sarcastic",
    "is_threatening",
    
    # 2. Conversational Context Features
    "conv_turn_count",
    "conv_time_delta_seconds",
    
    # 3. Behavioral History Features
    "total_attacks_sent",
    "target_attacks_count",
    "distinct_targets_count",
    "author_target_interactions",
    "author_flag_count",
    "author_hidden_count",
    "author_approved_count",
    "target_prior_victimization",
    "repetition_ratio",
    
    # 4. Temporal Dynamics Features
    "v_1h",
    "v_6h",
    "v_24h",
    "v_7d",
    "time_since_prev_attack_hours",
    "escalation_rate",
    "escalation_score",
    
    # 5. Network Graph Features
    "author_out_centrality",
    "target_in_centrality",
    "edge_weight",
    "author_clustering",
    "author_role_abuser",
    "author_role_participator",
    "target_role_victim",
    "graph_threat_score"
]

class ContextAwareFusionMLP(nn.Module):
    """
    Feed-Forward Neural Network for Multi-Source Feature Fusion.
    """
    def __init__(self, input_dim: int = len(FEATURE_SCHEMA), dropout_rate: float = 0.2):
        super().__init__()
        self.input_dim = input_dim
        self.version = "fusion-mlp-v1"

        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw scalar logits."""
        return self.network(x).squeeze(-1)

class FusionModelWrapper:
    """
    Inference wrapper for feature normalization and fused probability scoring.
    """
    def __init__(
        self,
        weights_path: Optional[str] = None,
        scaler_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        self.version = "fusion-mlp-v1"
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.input_dim = len(FEATURE_SCHEMA)
        self.model: Optional[ContextAwareFusionMLP] = None
        self.scaler: Optional[Dict[str, Any]] = None
        self.is_loaded = False

        if scaler_path and os.path.exists(scaler_path):
            self.load_scaler(scaler_path)

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_scaler(self, scaler_path: str) -> bool:
        """Loads normalization parameters computed strictly on training data."""
        try:
            with open(scaler_path, "r", encoding="utf-8") as f:
                self.scaler = json.load(f)
            logger.info(f"Loaded feature scaler parameters from {scaler_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load scaler from {scaler_path}: {e}")
            return False

    def load_weights(self, weights_path: str) -> bool:
        """Loads trained PyTorch MLP checkpoint."""
        try:
            self.model = ContextAwareFusionMLP(input_dim=self.input_dim)
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Loaded Fusion MLP weights from {weights_path} to {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load fusion weights: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def extract_vector(self, feature_dict: Dict[str, Any]) -> np.ndarray:
        """
        Orders features according to FEATURE_SCHEMA into a 1D numpy array.
        """
        vector = []
        for name in FEATURE_SCHEMA:
            val = feature_dict.get(name, 0.0)
            if val is None or not (isinstance(val, (int, float))):
                val = 0.0
            vector.append(float(val))
        return np.array(vector, dtype=np.float32)

    def scale_features(self, vector: np.ndarray) -> np.ndarray:
        """
        Applies z-score standardization using training scaler parameters:
        z = (x - mean) / (std + eps)
        """
        if self.scaler is None:
            return vector

        mean = np.array(self.scaler.get("mean", [0.0] * len(vector)), dtype=np.float32)
        std = np.array(self.scaler.get("std", [1.0] * len(vector)), dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std) # Prevent division by zero

        scaled = (vector - mean) / std
        return scaled

    def predict_p_final(self, feature_dict: Dict[str, Any]) -> float:
        """
        Executes end-to-end feature assembly, scaling, and MLP inference.
        Returns P_final in [0.0, 1.0].
        """
        raw_vec = self.extract_vector(feature_dict)
        scaled_vec = self.scale_features(raw_vec)

        if self.is_loaded and self.model is not None:
            try:
                x_tensor = torch.tensor(scaled_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    logit = self.model(x_tensor)
                    p_final = torch.sigmoid(logit).item()
                return max(0.0, min(1.0, float(p_final)))
            except Exception as e:
                logger.error(f"Error in Fusion MLP inference: {e}. Falling back to weighted combination.")

        # Fallback weighted combination if weights not loaded yet
        p_tox = float(feature_dict.get("p_toxic", 0.0))
        p_ctx = float(feature_dict.get("p_context", 0.0))
        esc = float(feature_dict.get("escalation_score", 0.0))
        rep = float(feature_dict.get("repetition_ratio", 0.0))
        threat = float(feature_dict.get("graph_threat_score", 0.0))

        # Balanced baseline formula
        blended = (0.45 * p_tox) + (0.35 * p_ctx) + (0.10 * esc) + (0.05 * rep) + (0.05 * threat)
        return max(0.0, min(1.0, blended))
