"""
CLUES: Minimal example demonstrating uncertainty decomposition.

Usage:
    # Live mode (requires GEMINI_API_KEY in .env):
    uv run python examples/run_clues.py

    # Offline mode (no API key needed, uses pre-computed matrices):
    uv run python examples/run_clues.py --offline

Paper: "Disentangling Ambiguity from Instability in Large Language Models:
        A Clinical Text-to-SQL Case Study" (ClinicalNLP 2026 Workshop at LREC 2026)
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, TypedDict

import numpy as np

# Add parent directory so clues package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clues.config import UncertaintyThresholds
from clues.decomposition import UncertaintyDecomposition
from clues.visualization import plot_uncertainty_decomposition

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds calibrated for the pre-computed example matrices below.
# In production, calibrate on your own data (the paper uses median-based thresholds).
# ---------------------------------------------------------------------------

THRESHOLDS = UncertaintyThresholds(h_i_threshold=0.005, h_r_given_i_threshold=0.003)


class Example(TypedDict):
    question: str
    interpretations: List[str]
    answers: List[str]
    w_ii: np.ndarray
    w_rr: np.ndarray


# ---------------------------------------------------------------------------
# Fictitious but realistic clinical examples.
# Pre-computed similarity matrices are illustrative examples designed to
# demonstrate each regime. Live Gemini API will produce different scores.
# ---------------------------------------------------------------------------

EXAMPLES: dict[str, Example] = {
    "ambiguous": {
        "question": "How many patients over 18 have atopic dermatitis?",
        "interpretations": [
            "Count of unique patients whose age at the time of their first diagnosis of atopic dermatitis was greater than 18 years.",
            "Count of unique patients whose current age is greater than 18 years and who have any diagnosis of atopic dermatitis at any time in their history.",
            "Count of unique patients aged over 18 at any diagnosis of atopic dermatitis, considering all recorded diagnoses (not just the first).",
        ],
        "answers": [
            # Interpretation 1 answers (age at first diagnosis > 18)
            "There are 12,847 unique patients whose age at first diagnosis of atopic dermatitis was over 18.",
            "A total of 12,851 patients were over 18 years old at the time of their first atopic dermatitis diagnosis.",
            "12,845 unique patients had their first atopic dermatitis diagnosis after the age of 18.",
            # Interpretation 2 answers (current age > 18, any diagnosis)
            "There are 28,493 unique patients currently over 18 with any atopic dermatitis diagnosis.",
            "28,501 patients aged over 18 currently have at least one atopic dermatitis diagnosis on record.",
            "A total of 28,490 unique patients over the age of 18 have atopic dermatitis in their history.",
            # Interpretation 3 answers (age > 18 at any diagnosis)
            "21,376 unique patients were over 18 at the time of at least one atopic dermatitis diagnosis.",
            "There are 21,380 patients who had an atopic dermatitis diagnosis while over 18 years old.",
            "21,372 unique patients had atopic dermatitis diagnosed at an age greater than 18.",
        ],
        # Pre-computed similarity matrices for offline mode
        "w_ii": np.array([
            [1.00, 0.35, 0.55],
            [0.35, 1.00, 0.50],
            [0.55, 0.50, 1.00],
        ]),
        "w_rr": np.array([
            # I1-R1, I1-R2, I1-R3, I2-R1, I2-R2, I2-R3, I3-R1, I3-R2, I3-R3
            [1.00, 0.95, 0.95, 0.15, 0.15, 0.15, 0.30, 0.30, 0.30],
            [0.95, 1.00, 0.95, 0.15, 0.15, 0.15, 0.30, 0.30, 0.30],
            [0.95, 0.95, 1.00, 0.15, 0.15, 0.15, 0.30, 0.30, 0.30],
            [0.15, 0.15, 0.15, 1.00, 0.95, 0.95, 0.35, 0.35, 0.35],
            [0.15, 0.15, 0.15, 0.95, 1.00, 0.95, 0.35, 0.35, 0.35],
            [0.15, 0.15, 0.15, 0.95, 0.95, 1.00, 0.35, 0.35, 0.35],
            [0.30, 0.30, 0.30, 0.35, 0.35, 0.35, 1.00, 0.95, 0.95],
            [0.30, 0.30, 0.30, 0.35, 0.35, 0.35, 0.95, 1.00, 0.95],
            [0.30, 0.30, 0.30, 0.35, 0.35, 0.35, 0.95, 0.95, 1.00],
        ]),
    },
    "confident": {
        "question": "What is the total count of unique female patients aged 18-55 with a first diagnosis of endometriosis between 2015 and 2020?",
        "interpretations": [
            "Count of unique female patients aged 18-55 at the time of their first endometriosis diagnosis, where the first diagnosis occurred between January 1, 2015 and December 31, 2020.",
            "Total number of distinct female patients between 18 and 55 years old who received their initial endometriosis diagnosis in the 2015-2020 period.",
            "Number of unique female patients whose first-ever endometriosis diagnosis was recorded between 2015 and 2020, and who were aged 18 to 55 at that time.",
        ],
        "answers": [
            # All interpretations produce very similar answers
            "There are 4,217 unique female patients aged 18-55 with a first endometriosis diagnosis between 2015 and 2020.",
            "A total of 4,217 female patients aged 18-55 had their first endometriosis diagnosis during 2015-2020.",
            "4,218 unique female patients aged 18-55 received their first endometriosis diagnosis between 2015 and 2020.",
            "The count is 4,217 unique female patients aged 18-55 with a first endometriosis diagnosis in the 2015-2020 period.",
            "4,217 distinct female patients between 18 and 55 years were first diagnosed with endometriosis between 2015 and 2020.",
            "There are 4,216 unique female patients aged 18-55 with an initial endometriosis diagnosis in 2015-2020.",
            "A total of 4,217 unique female patients aged 18-55 had their first endometriosis diagnosis recorded between 2015 and 2020.",
            "4,218 female patients aged 18 to 55 had their first diagnosis of endometriosis between 2015 and 2020.",
            "The total is 4,217 unique female patients aged 18-55 first diagnosed with endometriosis during 2015-2020.",
        ],
        "w_ii": np.array([
            [1.00, 0.95, 0.93],
            [0.95, 1.00, 0.94],
            [0.93, 0.94, 1.00],
        ]),
        "w_rr": np.array([
            [1.00, 0.98, 0.95, 0.98, 0.97, 0.94, 0.98, 0.95, 0.97],
            [0.98, 1.00, 0.95, 0.97, 0.98, 0.94, 0.97, 0.95, 0.98],
            [0.95, 0.95, 1.00, 0.95, 0.95, 0.93, 0.95, 0.97, 0.95],
            [0.98, 0.97, 0.95, 1.00, 0.97, 0.94, 0.98, 0.95, 0.97],
            [0.97, 0.98, 0.95, 0.97, 1.00, 0.94, 0.97, 0.95, 0.98],
            [0.94, 0.94, 0.93, 0.94, 0.94, 1.00, 0.94, 0.93, 0.94],
            [0.98, 0.97, 0.95, 0.98, 0.97, 0.94, 1.00, 0.95, 0.97],
            [0.95, 0.95, 0.97, 0.95, 0.95, 0.93, 0.95, 1.00, 0.95],
            [0.97, 0.98, 0.95, 0.97, 0.98, 0.94, 0.97, 0.95, 1.00],
        ]),
    },
    "unstable": {
        "question": "List the top 5 most prescribed medications for hypertension patients.",
        "interpretations": [
            "Identify the top 5 medications by total prescription count among patients with a diagnosis of hypertension at any time in their history.",
            "Determine the 5 most frequently prescribed medications for patients diagnosed with hypertension, ranked by number of unique patients.",
            "Find the top 5 drugs by prescription volume for patients whose primary diagnosis includes hypertension.",
        ],
        "answers": [
            # Interpretation 1 answers (vary in ordering / numbers)
            "Top 5 medications by prescription count for hypertension patients: 1. Amlodipine (245,891), 2. Lisinopril (231,456), 3. Metoprolol (198,234), 4. Losartan (187,654), 5. Hydrochlorothiazide (176,432).",
            "The top 5 prescribed medications for hypertension are: 1. Lisinopril (233,102), 2. Amlodipine (244,567), 3. Losartan (189,234), 4. Metoprolol (196,789), 5. Atenolol (165,432).",
            "Top 5 hypertension medications: 1. Amlodipine (246,123), 2. Lisinopril (230,890), 3. Metoprolol (197,456), 4. Losartan (188,901), 5. Valsartan (171,234).",
            # Interpretation 2 answers (vary in results)
            "By unique patients: 1. Amlodipine (189,432), 2. Lisinopril (178,901), 3. Metoprolol (156,234), 4. Losartan (145,678), 5. Hydrochlorothiazide (134,567).",
            "Top 5 by patient count: 1. Lisinopril (180,234), 2. Amlodipine (188,567), 3. Losartan (147,890), 4. Metoprolol (155,123), 5. Atenolol (128,456).",
            "Most prescribed to unique patients: 1. Amlodipine (190,123), 2. Lisinopril (179,456), 3. Metoprolol (157,890), 4. Hydrochlorothiazide (136,234), 5. Losartan (144,567).",
            # Interpretation 3 answers (vary in results)
            "Top 5 for primary hypertension: 1. Amlodipine (201,345), 2. Lisinopril (192,678), 3. Losartan (167,890), 4. Metoprolol (158,234), 5. Olmesartan (142,567).",
            "Primary hypertension top medications: 1. Lisinopril (194,123), 2. Amlodipine (200,456), 3. Metoprolol (160,789), 4. Losartan (166,012), 5. Valsartan (139,345).",
            "For patients with primary hypertension diagnosis: 1. Amlodipine (202,678), 2. Lisinopril (191,234), 3. Losartan (168,567), 4. Metoprolol (159,890), 5. Hydrochlorothiazide (140,123).",
        ],
        "w_ii": np.array([
            [1.00, 0.80, 0.75],
            [0.80, 1.00, 0.78],
            [0.75, 0.78, 1.00],
        ]),
        "w_rr": np.array([
            [1.00, 0.60, 0.65, 0.45, 0.40, 0.48, 0.50, 0.42, 0.52],
            [0.60, 1.00, 0.58, 0.42, 0.50, 0.45, 0.45, 0.48, 0.47],
            [0.65, 0.58, 1.00, 0.48, 0.42, 0.50, 0.52, 0.45, 0.55],
            [0.45, 0.42, 0.48, 1.00, 0.55, 0.60, 0.50, 0.48, 0.52],
            [0.40, 0.50, 0.42, 0.55, 1.00, 0.58, 0.45, 0.52, 0.48],
            [0.48, 0.45, 0.50, 0.60, 0.58, 1.00, 0.52, 0.50, 0.55],
            [0.50, 0.45, 0.52, 0.50, 0.45, 0.52, 1.00, 0.55, 0.62],
            [0.42, 0.48, 0.45, 0.48, 0.52, 0.50, 0.55, 1.00, 0.58],
            [0.52, 0.47, 0.55, 0.52, 0.48, 0.55, 0.62, 0.58, 1.00],
        ]),
    },
}


def get_mapping(n_interp: int, m_per_interp: int):
    """Create result-to-interpretation mapping: [0,0,0,1,1,1,2,2,2,...]"""
    return [i for i in range(n_interp) for _ in range(m_per_interp)]


async def run_offline(output_dir: Path):
    """Run CLUES with pre-computed similarity matrices (no API calls)."""
    ud = UncertaintyDecomposition(t=10.0)

    for name, example in EXAMPLES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Example: {name.upper()}")
        logger.info(f"Question: {example['question']}")
        logger.info(f"{'='*60}")

        n_interp = len(example["interpretations"])
        m_per_interp = len(example["answers"]) // n_interp
        mapping = get_mapping(n_interp, m_per_interp)

        result = await ud.compute_decomposed_uncertainty_from_matrices(
            w_ii=example["w_ii"],
            w_rr=example["w_rr"],
            mapping=mapping,
            thresholds=THRESHOLDS,
        )

        example_dir = output_dir / name
        plot_uncertainty_decomposition(
            result, example["interpretations"], example["answers"], str(example_dir)
        )
        logger.info(f"Plots saved to {example_dir.as_posix()}")

        _print_results(name, example, result)


async def run_live(output_dir: Path):
    """Run CLUES with live Gemini API calls for similarity computation."""
    from clues.llm import GeminiClient

    client = GeminiClient()
    ud = UncertaintyDecomposition(t=10.0)

    for name, example in EXAMPLES.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Example: {name.upper()}")
        logger.info(f"Question: {example['question']}")
        logger.info(f"{'='*60}")

        n_interp = len(example["interpretations"])
        m_per_interp = len(example["answers"]) // n_interp
        mapping = get_mapping(n_interp, m_per_interp)

        result = await ud.compute_decomposed_uncertainty(
            interpretations=example["interpretations"],
            results=example["answers"],
            mapping=mapping,
            client=client,
            thresholds=THRESHOLDS,
        )

        example_dir = output_dir / name
        plot_uncertainty_decomposition(
            result, example["interpretations"], example["answers"], str(example_dir)
        )
        logger.info(f"Plots saved to {example_dir.as_posix()}")

        _print_results(name, example, result)


def _print_results(name, example, result):
    """Print formatted CLUES results."""
    ur = result["uncertainty_result"]
    e = result["entropies"]

    print(f"\n--- {name.upper()} ---")
    print(f"Question: {example['question']}")
    print(f"  H(I)   = {e['H_I']:.4f}  (ambiguity)")
    print(f"  H(R)   = {e['H_R']:.4f}  (result diversity)")
    print(f"  H(R|I) = {e['H(R|I)']:.4f}  (instability)")
    print(f"  Regime: {ur.regime.value}")
    print(f"  Action: {ur.recommended_action}")
    print()


def main():
    parser = argparse.ArgumentParser(description="CLUES uncertainty decomposition demo")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use pre-computed similarity matrices (no API key needed)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory for plots (default: output)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()

    if args.offline:
        logger.info("Running in OFFLINE mode (pre-computed matrices)")
        asyncio.run(run_offline(output_dir))
    else:
        logger.info("Running in LIVE mode (calling Gemini API)")
        asyncio.run(run_live(output_dir))


if __name__ == "__main__":
    main()
