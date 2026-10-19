"""
Models package for CyberShield.
Exports:
- SQLAlchemy ORM models: Base, User, Prediction, ReviewQueue, ModeratorAction, FairnessMetric
- Neural & Fusion ML models: ToxicModel, MultiTaskContextualTransformer, ContextModelWrapper,
                             ContextAwareFusionMLP, FusionModelWrapper, FEATURE_SCHEMA
- Tokenizer utilities
"""

from .db_models import (
    Base,
    User,
    Prediction,
    ReviewQueue,
    ModeratorAction,
    FairnessMetric,
    generate_uuid
)

from .toxic_model import ToxicModel
from .context_model import MultiTaskContextualTransformer, ContextModelWrapper
from .fusion_model import ContextAwareFusionMLP, FusionModelWrapper, FEATURE_SCHEMA
from .tokenizer import get_context_tokenizer, format_context_input

__all__ = [
    "Base",
    "User",
    "Prediction",
    "ReviewQueue",
    "ModeratorAction",
    "FairnessMetric",
    "generate_uuid",
    "ToxicModel",
    "MultiTaskContextualTransformer",
    "ContextModelWrapper",
    "ContextAwareFusionMLP",
    "FusionModelWrapper",
    "FEATURE_SCHEMA",
    "get_context_tokenizer",
    "format_context_input",
]
