# CLUES: Conditional Language Uncertainty via Entropy and Schur Complement

A framework for decomposing semantic uncertainty in LLM outputs into **ambiguity** (from input interpretation diversity) and **instability** (from model inconsistency).

**Authors**: Angelo Ziletti, Leonardo D'Ambrosi

> **Paper**: *Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study*
> **arXiv**: [https://arxiv.org/abs/2602.12015](https://arxiv.org/abs/2602.12015)
> Accepted at the [8th Clinical NLP Workshop (ClinicalNLP 2026)](https://clinical-nlp.github.io/2026/) at LREC 2026, Palma de Mallorca, Spain

## Overview

When an LLM answers a question multiple times, variation in its outputs can stem from two distinct sources:

1. **Ambiguity (H_I)**: The input question has multiple valid interpretations (e.g., "patients over 18" — at diagnosis or currently?)
2. **Instability (H_{R|I})**: The model produces inconsistent answers even for a single, clear interpretation

CLUES disentangles these two sources using a bipartite semantic graph and the **Schur complement** of its similarity matrix, guaranteeing non-negative conditional entropy by construction.

### The Four Uncertainty Regimes

| Regime | H(I) | H(R\|I) | Recommended Action |
|--------|-------|---------|-------------------|
| **I. Confident** | Low | Low | Auto-answer |
| **II. Ambiguity** | High | Low | Ask user to clarify |
| **III. Instability** | Low | High | Flag for human review |
| **IV. Compound** | High | High | Clarify + review |

## Installation

Requires Python >= 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

For Jupyter notebook support:

```bash
uv sync --extra dev
```

## Quick Start

### 1. Question Disambiguation (requires Gemini API key)

CLUES can automatically resolve implicit ambiguities in epidemiological questions — such as age at diagnosis vs. current age, first vs. any occurrence, or temporal boundaries. This works out of the box and is useful for any researcher working with clinical data.

1. Copy `.env.example` to `.env` and add your Gemini API key:

```bash
cp .env.example .env
# Edit .env with your key: GEMINI_API_KEY="your-key-here"
```

2. Run the disambiguation demo:

```bash
uv run python examples/run_disambiguation.py
```

**Python API:**

```python
import asyncio
from clues import GeminiClient, generate_question_interpretations

client = GeminiClient()
result = asyncio.get_event_loop().run_until_complete(
    generate_question_interpretations(
        "How many patients over 18 have atopic dermatitis?",
        client=client,
    )
)

print(f"Ambiguous: {result.is_ambiguous}")
print(f"Score: {result.ambiguity_score:.2f}")
for i, interp in enumerate(result.final_interpretations, 1):
    print(f"  {i}. {interp}")
```

A step-by-step notebook is available:

```bash
uv run jupyter notebook examples/disambiguation_demo.ipynb
```

### 2. Full Uncertainty Decomposition

The full CLUES pipeline decomposes semantic uncertainty into ambiguity (H_I) and instability (H_{R|I}) using a bipartite semantic graph and the Schur complement. This may require threshold calibration for your specific domain (the paper uses median-based thresholds).

#### Offline Mode (no API key needed)

Run with pre-computed similarity matrices:

```bash
uv run python examples/run_clues.py --offline
```

#### Live Mode (requires Gemini API key)

Run with live LLM-based similarity computation:

```bash
uv run python examples/run_clues.py
```

#### Python API

```python
import asyncio
import numpy as np
from clues import UncertaintyDecomposition

# Pre-computed similarity matrices
w_ii = np.array([          # Interpretation similarity (N x N)
    [1.0, 0.35, 0.55],
    [0.35, 1.0, 0.50],
    [0.55, 0.50, 1.0],
])
w_rr = np.eye(9)           # Answer similarity (N*M x N*M)
mapping = [0, 0, 0, 1, 1, 1, 2, 2, 2]  # Which interpretation produced which answer

ud = UncertaintyDecomposition(t=10.0)
result = asyncio.get_event_loop().run_until_complete(
    ud.compute_decomposed_uncertainty_from_matrices(
        w_ii=w_ii, w_rr=w_rr, mapping=mapping,
    )
)

entropies = result["entropies"]
print(f"Ambiguity H(I):    {entropies['H_I']:.4f}")
print(f"Instability H(R|I): {entropies['H(R|I)']:.4f}")
print(f"Regime: {result['uncertainty_result'].regime.value}")
```

#### Jupyter Notebook

A step-by-step walkthrough is available:

```bash
uv run jupyter notebook examples/clues_demo.ipynb
```

## Method

CLUES constructs a block similarity matrix over a bipartite graph of interpretations (I) and results (R):

```
W = [ W_II   W_IR ]
    [ W_RI   W_RR ]
```

Entropy is computed via the **heat kernel** K = exp(-τL) and **von Neumann entropy** H = -Σ λ_i log(λ_i), where L is the graph Laplacian.

The **Schur complement** S = W_RR - W_RI · inv(W_II + εI) · W_IR isolates the residual result similarity after factoring out interpretation differences, yielding H(R|I) ≥ 0 by construction.

The chain rule H(I,R) = H(I) + H(R|I) decomposes total uncertainty into its two sources.

## Project Structure

```
clues/
├── clues/                  # Python package
│   ├── __init__.py         # Public API
│   ├── config.py           # Regimes, thresholds, result dataclass
│   ├── entropy.py          # Heat kernel + von Neumann entropy
│   ├── schur.py            # Schur complement H(R|I)
│   ├── decomposition.py    # Full decomposition pipeline
│   ├── disambiguation.py   # LLM-based interpretation generation
│   ├── visualization.py    # Plots and heatmaps
│   ├── schemas.py          # Pydantic schemas
│   └── llm.py              # Minimal Gemini client
└── examples/
    ├── run_disambiguation.py      # Disambiguation demo (live)
    ├── disambiguation_demo.ipynb  # Disambiguation notebook
    ├── run_clues.py               # Full CLUES demo (offline + live)
    └── clues_demo.ipynb           # Full CLUES notebook
```

## Citation

```bibtex
@misc{ziletti2026disentanglingambiguityinstabilitylarge,
      title={Disentangling Ambiguity from Instability in Large Language Models: A Clinical Text-to-SQL Case Study},
      author={Angelo Ziletti and Leonardo D'Ambrosi},
      year={2026},
      eprint={2602.12015},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2602.12015},
}
```

## License

MIT
