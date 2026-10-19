"""
Contextual Sub-Model Architecture.
Encodes dual-sequence text (Current Comment + Conversational Context) using DistilBERT.
Features a shared transformer representation with Multi-Task Auxiliary Prediction Heads:
1. P_context: Context-aware cyberbullying probability (Primary)
2. is_targeted: Direct targeted attack probability
3. is_repeated: Repetitive harassment indicator
4. is_escalating: Escalating hostility indicator
5. is_sarcastic: Sarcastic inversion indicator
6. is_threatening: Overt threat indicator
"""

import os
import logging
from typing import Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

class MultiTaskContextualTransformer(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased", dropout_rate: float = 0.2):
        super().__init__()
        from transformers import AutoModel, AutoConfig

        self.model_name = model_name
        self.version = "context-distilbert-v1"
        
        # Load transformer backbone
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        hidden_size = self.config.hidden_size

        self.dropout = nn.Dropout(dropout_rate)

        # Multi-task prediction heads (logits)
        self.head_cyberbullying = nn.Linear(hidden_size, 1) # P_context
        self.head_targeted = nn.Linear(hidden_size, 1)      # is_targeted
        self.head_repeated = nn.Linear(hidden_size, 1)      # is_repeated
        self.head_escalating = nn.Linear(hidden_size, 1)    # is_escalating
        self.head_sarcastic = nn.Linear(hidden_size, 1)     # is_sarcastic
        self.head_threatening = nn.Linear(hidden_size, 1)   # is_threatening

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass producing raw logits for each task head.
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Extract [CLS] representation (first token)
        cls_rep = outputs.last_hidden_state[:, 0, :]
        cls_rep = self.dropout(cls_rep)

        return {
            "logit_cyberbullying": self.head_cyberbullying(cls_rep).squeeze(-1),
            "logit_targeted": self.head_targeted(cls_rep).squeeze(-1),
            "logit_repeated": self.head_repeated(cls_rep).squeeze(-1),
            "logit_escalating": self.head_escalating(cls_rep).squeeze(-1),
            "logit_sarcastic": self.head_sarcastic(cls_rep).squeeze(-1),
            "logit_threatening": self.head_threatening(cls_rep).squeeze(-1),
            "cls_embedding": cls_rep
        }

class ContextModelWrapper:
    """
    Inference and lifecycle wrapper around the MultiTaskContextualTransformer.
    """
    def __init__(self, weights_path: Optional[str] = None, device: Optional[str] = None):
        self.version = "context-distilbert-v1"
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[MultiTaskContextualTransformer] = None
        self.tokenizer = None
        self.is_loaded = False

        if weights_path and os.path.exists(weights_path):
            self.load_weights(weights_path)

    def load_weights(self, weights_path: str) -> bool:
        """Loads trained checkpoint weights."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            self.model = MultiTaskContextualTransformer()
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            self.is_loaded = True
            logger.info(f"Loaded Contextual Sub-Model weights from {weights_path} to {self.device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load context model weights: {e}", exc_info=True)
            self.is_loaded = False
            return False

    def predict(self, text: str, context_snippet: str = "") -> Dict[str, float]:
        """
        Runs multi-task contextual inference on a single comment + context pair.

        Returns probabilities in [0.0, 1.0].
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            # Fallback estimation if weights not loaded yet
            return self._heuristic_fallback(text, context_snippet)

        try:
            formatted_text = text
            if context_snippet:
                formatted_text = f"{text} [SEP] Context: {context_snippet}"

            inputs = self.tokenizer(
                formatted_text,
                max_length=128,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)

            with torch.no_grad():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

            # Convert logits to probabilities via sigmoid
            p_context = torch.sigmoid(logits["logit_cyberbullying"]).item()
            p_targeted = torch.sigmoid(logits["logit_targeted"]).item()
            p_repeated = torch.sigmoid(logits["logit_repeated"]).item()
            p_escalating = torch.sigmoid(logits["logit_escalating"]).item()
            p_sarcastic = torch.sigmoid(logits["logit_sarcastic"]).item()
            p_threatening = torch.sigmoid(logits["logit_threatening"]).item()

            return {
                "p_context": round(float(p_context), 4),
                "is_targeted": round(float(p_targeted), 4),
                "is_repeated": round(float(p_repeated), 4),
                "is_escalating": round(float(p_escalating), 4),
                "is_sarcastic": round(float(p_sarcastic), 4),
                "is_threatening": round(float(p_threatening), 4)
            }
        except Exception as e:
            logger.error(f"Error in context model inference: {e}", exc_info=True)
            return self._heuristic_fallback(text, context_snippet)

    def _heuristic_fallback(self, text: str, context_snippet: str) -> Dict[str, float]:
        """Graceful fallback prior to initial training."""
        t_low = text.lower()
        has_insult = any(w in t_low for w in ["trash", "loser", "ugly", "stupid", "idiot", "fat", "gross", "worthless"])
        has_neg = any(w in t_low for w in ["not", "never", "don't", "dont", "no"])
        has_slang = "killing it" in t_low or "sick beat" in t_low
        has_sarcasm = "said no one ever" in t_low or "said nobody ever" in t_low
        has_threat = any(w in t_low for w in ["kill", "delete", "destroy", "die"])

        p_ctx = 0.85 if has_insult and not has_neg else (0.25 if has_neg or has_slang else 0.05)
        if has_sarcasm:
            p_ctx = 0.78

        return {
            "p_context": p_ctx,
            "is_targeted": 0.80 if ("@" in text or "@" in context_snippet) else 0.20,
            "is_repeated": 0.10,
            "is_escalating": 0.10,
            "is_sarcastic": 0.90 if has_sarcasm else 0.05,
            "is_threatening": 0.85 if has_threat else 0.05
        }
