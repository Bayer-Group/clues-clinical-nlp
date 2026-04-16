"""
CLUES: Conditional Language Uncertainty via Entropy and Schur complement.

A framework for decomposing semantic uncertainty in LLM outputs into:
- Ambiguity (H_I): uncertainty from input interpretation diversity
- Instability (H_{R|I}): uncertainty from model inconsistency

Paper: "Disentangling Ambiguity from Instability in Large Language Models:
        A Clinical Text-to-SQL Case Study" (ClinicalNLP 2026 Workshop at LREC 2026)

GitHub: https://github.com/Bayer-Group/clues-clinical-nlp
"""

from clues.config import (
    UncertaintyRegime,
    UncertaintyThresholds,
    UncertaintyResult,
    DEFAULT_THRESHOLDS,
)
from clues.decomposition import UncertaintyDecomposition
from clues.disambiguation import generate_question_interpretations, generate_explore_meanings
from clues.visualization import plot_uncertainty_decomposition
from clues.llm import GeminiClient

__all__ = [
    # Core
    "UncertaintyDecomposition",
    # Config
    "UncertaintyRegime",
    "UncertaintyThresholds",
    "UncertaintyResult",
    "DEFAULT_THRESHOLDS",
    # Disambiguation
    "generate_question_interpretations",
    "generate_explore_meanings",
    # Visualization
    "plot_uncertainty_decomposition",
    # LLM Client
    "GeminiClient",
]
