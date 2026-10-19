"""
Explainability package providing 5-pillar contextual feature attribution and LIME.
"""

from .contextual_factors import generate_contextual_factors
from .explainer import MultiPillarExplainer

__all__ = [
    "generate_contextual_factors",
    "MultiPillarExplainer",
]
