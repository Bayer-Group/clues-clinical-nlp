"""
Pydantic schemas for CLUES disambiguation and similarity scoring.
"""

from pydantic import BaseModel, Field, field_validator
from typing import List
import logging

logger = logging.getLogger(__name__)


class DisambiguationResponse(BaseModel):
    """Schema for the complete disambiguation analysis."""

    original_question: str = Field(..., description="The original question that was analyzed")
    is_ambiguous: bool = Field(
        ..., description="Whether the question contains meaningful ambiguities"
    )
    ambiguity_score: float = Field(
        ...,
        description="Overall ambiguity score from 0 (unambiguous) to 1 (highly ambiguous)",
    )
    final_interpretations: List[str] = Field(
        ...,
        description="Final list of distinct question interpretations that represent all meaningful combinations of ambiguity resolutions",
    )


class ExploreMeaningsResponse(BaseModel):
    """
    Schema for 'Explore Meanings' prompt — Stage 1 uncertainty estimation.

    Generates diverse semantic interpretations for measuring aleatoric uncertainty H(I).
    """

    original_question: str = Field(..., description="The original question analyzed")
    interpretations: List[str] = Field(
        ...,
        description="Diverse semantic interpretations exploring different possible meanings.",
    )


class EntailmentScore(BaseModel):
    """Schema for LLM-based entailment scoring."""

    score: float = Field(
        description="A number between 0 and 1 representing the entailment score",
        ge=0.0,
        le=1.0,
    )

    @field_validator("score")
    def validate_score_range(cls, v):
        if v < 0.0 or v > 1.0:
            logger.warning(
                f"Entailment score {v} is outside valid range [0,1]. Clamping."
            )
            return max(0.0, min(1.0, v))
        return v
