"""
CLUES Disambiguation: Demo of iterative question disambiguation.

Demonstrates how CLUES resolves implicit ambiguities in epidemiological questions
using two complementary approaches:
1. Guided disambiguation: identifies ambiguity types, scores them, generates interpretations
2. Exploratory meanings: generates diverse interpretations for uncertainty estimation

Usage (requires GEMINI_API_KEY in .env):
    uv run python examples/run_disambiguation.py

Paper: "Disentangling Ambiguity from Instability in Large Language Models:
        A Clinical Text-to-SQL Case Study" (ClinicalNLP 2026 Workshop at LREC 2026)
"""

import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clues import GeminiClient, generate_explore_meanings, generate_question_interpretations

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Example epidemiological questions with varying levels of ambiguity.
# The first two are from Table 2 of the paper.
# ---------------------------------------------------------------------------

QUESTIONS = [
    "How many patients > 17 yo have atopic dermatitis (all codes). Breakdown by code",
    "How many patients with chronic kidney disease never took heparin before their chronic kidney disease diagnosis?",
    "How many patients over 18 have atopic dermatitis?",
    "Among people with CKD, which percentage will develop ischemic stroke within 5 years of diagnosis? Group by year after diagnosis",
]


async def main():
    client = GeminiClient()

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*70}")
        print(f"Example {i}: {question}")
        print(f"{'='*70}")

        # --- 1. Guided disambiguation ---
        print(f"\n--- Guided Disambiguation ---")
        result = await generate_question_interpretations(question, client=client)
        print(f"  Ambiguous: {result.is_ambiguous}")
        print(f"  Ambiguity score: {result.ambiguity_score:.2f}")
        print(f"  Interpretations ({len(result.final_interpretations)}):")
        for j, interp in enumerate(result.final_interpretations, 1):
            print(f"    {j}. {interp}")

        # --- 2. Exploratory meanings ---
        print(f"\n--- Exploratory Meanings ---")
        explore = await generate_explore_meanings(
            question, num_interpretations=5, client=client
        )
        print(f"  Interpretations ({len(explore.interpretations)}):")
        for j, interp in enumerate(explore.interpretations, 1):
            print(f"    {j}. {interp}")


if __name__ == "__main__":
    asyncio.run(main())
