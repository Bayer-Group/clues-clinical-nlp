"""
Kernel Laplacian Entropy (KLE) for CLUES framework.

Implements heat kernel computation and von Neumann entropy over semantic graphs.
Provides prompt-based similarity graph construction via LLM entailment scoring.

References:
    - Nikitin et al. (2024). Kernel Language Entropy.
    - Paper Section 3.2: Heat Kernel Regularization.
"""

from typing import List, Any
import asyncio
import logging
import networkx as nx
import numpy as np
import scipy.linalg

from clues.schemas import EntailmentScore

logger = logging.getLogger(__name__)


class KernelLaplacianEntropy:
    """
    Kernel Laplacian Entropy (KLE) implementation.

    Computes von Neumann entropy over heat-kernel-regularized semantic graphs.

    Args:
        t: Heat kernel diffusion parameter (default 10.0).
           Controls granularity: small t preserves fine-grained distinctions,
           large t diffuses similarity, merging nearby nodes.
        norm_lapl: Whether to use normalized Laplacian (default True, recommended).
        gamma: Input kernel sharpening parameter (default 1.0 = no sharpening).
               Values > 1 suppress off-diagonal noise while preserving diagonal.
    """

    def __init__(self, t: float = 10.0, norm_lapl: bool = True, gamma: float = 1.0):
        self.t = t
        self.norm_lapl = norm_lapl
        self.gamma = gamma

        if gamma < 1.0:
            logger.warning(f"gamma={gamma} < 1.0 may cause unexpected behavior. Using gamma=1.0.")
            self.gamma = 1.0

    @staticmethod
    def sharpen_similarity_matrix(matrix: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """
        Apply element-wise power transformation to sharpen a similarity matrix.

        W_new[i,j] = W_old[i,j] ** gamma

        Raising similarity scores s in [0,1] to a power gamma > 1 suppresses
        off-diagonal noise while preserving the diagonal (1.0).

        Args:
            matrix: Similarity matrix with values in [0, 1].
            gamma: Sharpening exponent (default 1.0 = no sharpening).

        Returns:
            Sharpened similarity matrix.
        """
        if gamma == 1.0:
            return matrix.copy()

        sharpened = np.power(matrix, gamma)
        np.fill_diagonal(sharpened, 1.0)

        if matrix.shape[0] > 1:
            mask = ~np.eye(matrix.shape[0], dtype=bool)
            mean_before = np.mean(matrix[mask])
            mean_after = np.mean(sharpened[mask])
            logger.debug(
                f"Kernel sharpening (gamma={gamma}): mean off-diagonal {mean_before:.4f} -> {mean_after:.4f}"
            )

        return sharpened

    def compute_entropy_from_matrix(self, matrix: np.ndarray) -> float:
        """Compute von Neumann entropy from a similarity matrix."""
        G = nx.Graph()
        n = matrix.shape[0]
        G.add_nodes_from(range(n))

        for i in range(n):
            for j in range(n):
                if matrix[i, j] > 0:
                    G.add_edge(i, j, weight=matrix[i, j])

        kernel = self.heat_kernel(G)
        entropy = self.vn_entropy(kernel)
        return entropy

    def heat_kernel(self, G: nx.Graph) -> np.ndarray:
        """
        Compute heat kernel K = exp(-t * L) from graph.

        Args:
            G: NetworkX graph with weighted edges.

        Returns:
            Heat kernel matrix.
        """
        if self.norm_lapl:
            L = nx.normalized_laplacian_matrix(G).toarray()
        else:
            L = nx.laplacian_matrix(G).toarray()

        try:
            K = scipy.linalg.expm(-self.t * L)
        except Exception as e:
            eigenvalues, eigenvectors = np.linalg.eigh(L)
            K = eigenvectors @ np.diag(np.exp(-self.t * eigenvalues)) @ eigenvectors.T
            logger.warning(f"Used eigendecomposition fallback due to: {e}")

        return K

    def vn_entropy(
        self,
        kernel: np.ndarray,
        scale: bool = True,
        normalize: bool = True,
        jitter: float = 1e-10,
    ) -> float:
        """
        Compute von Neumann entropy of kernel matrix.

        H = -sum(lambda_i * log(lambda_i))

        Args:
            kernel: Heat kernel matrix.
            scale: If True, scale to [0, 1] by dividing by log(n).
            normalize: If True, trace-normalize the kernel first.
            jitter: Small value added for numerical stability.

        Returns:
            Von Neumann entropy value.
        """
        n = kernel.shape[0]

        if np.all(kernel == 0):
            raise ValueError("Invalid kernel matrix: all elements are zero")
        if np.isnan(kernel).any():
            raise ValueError("Invalid kernel matrix: contains NaN values")

        if normalize:
            trace = np.trace(kernel)
            if trace <= 0:
                raise ValueError(f"Invalid kernel matrix: trace is non-positive ({trace})")
            kernel = kernel / trace

        kernel = kernel + np.eye(n) * jitter

        eigenvalues = np.linalg.eigvalsh(kernel)
        positive_eigenvalues = eigenvalues[eigenvalues > 0]

        if len(positive_eigenvalues) == 0:
            raise ValueError("Invalid kernel matrix: no positive eigenvalues found")

        entropy = -np.sum(eigenvalues * np.log(eigenvalues))

        if scale:
            entropy = entropy / np.log(n)

        if scale:
            entropy = max(0.0, min(1.0, entropy))
        else:
            entropy = max(0.0, min(np.log(n), entropy))

        return entropy

    async def get_prompt_similarity_graph(
        self,
        contents: List[str],
        client: Any,
        max_concurrency: int = 25,
        custom_prompt: str = None,
    ) -> nx.Graph:
        """
        Construct a similarity graph using LLM-based entailment scoring.

        For each pair of texts, compute bidirectional entailment scores
        and average them to get a symmetric similarity weight.

        Args:
            contents: List of text strings to compare.
            client: GeminiClient instance for LLM calls.
            max_concurrency: Maximum number of concurrent API calls.
            custom_prompt: Optional custom prompt template with {text1} and {text2}.

        Returns:
            NetworkX graph with similarity-weighted edges.
        """
        semaphore = asyncio.Semaphore(max_concurrency)

        async def get_entailment_with_semaphore(text1: str, text2: str) -> float:
            async with semaphore:
                return await self.check_entailment_prompt(text1, text2, client, custom_prompt)

        tasks = []
        edge_indices = []
        for i in range(len(contents)):
            for j in range(i + 1, len(contents)):
                tasks.append(get_entailment_with_semaphore(contents[i], contents[j]))
                tasks.append(get_entailment_with_semaphore(contents[j], contents[i]))
                edge_indices.append((i, j))

        all_weights = await asyncio.gather(*tasks, return_exceptions=True)

        edges = []
        for i in range(len(contents)):
            edges.append((i, i, 1.0))

        for idx, (i, j) in enumerate(edge_indices):
            weight_1to2 = all_weights[idx * 2]
            weight_2to1 = all_weights[idx * 2 + 1]

            if isinstance(weight_1to2, Exception) or isinstance(weight_2to1, Exception):
                logger.error(f"Failed to calculate similarity for pair ({i}, {j}). Skipping.")
                continue

            combined_weight = (weight_1to2 + weight_2to1) / 2
            if combined_weight > 0:
                edges.append((i, j, combined_weight))

        G = nx.Graph()
        G.add_nodes_from(range(len(contents)))
        G.add_weighted_edges_from(edges)
        return G

    @staticmethod
    async def check_entailment_prompt(
        text1: str, text2: str, client: Any, custom_prompt: str = None
    ) -> float:
        """
        Check entailment between two texts using LLM.

        Args:
            text1: First text.
            text2: Second text.
            client: GeminiClient instance.
            custom_prompt: Optional custom prompt template.

        Returns:
            Entailment score between 0 and 1.
        """
        if custom_prompt:
            prompt = custom_prompt.format(text1=text1, text2=text2)
        else:
            prompt = f"""Task: Determine if the first statement entails the second statement.

Pay special attention to numerical values in both statements:
- Same numbers -> higher score
- Different numbers -> lower score
- Small differences (within 5%) -> minimal impact

Statement 1: "{text1}"
Statement 2: "{text2}"

Return a single number between 0 and 1:
- 1.0: Perfect entailment
- 0.0: No entailment / contradiction
- Between: Partial entailment

Your response should be ONLY a number between 0 and 1."""

        try:
            response = await client.get_response(prompt=prompt, response_schema=EntailmentScore)
            return response.score
        except Exception as e:
            logger.error(f"Error in entailment check: {e}")
            raise ValueError(f"Entailment check failed: {e}") from e
