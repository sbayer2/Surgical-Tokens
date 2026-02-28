"""Clustering validation: verify that clip embeddings cluster by surgical phase
without any labels (unsupervised validation of embedding quality).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logger = logging.getLogger(__name__)


def evaluate_clustering(
    embeddings: np.ndarray | torch.Tensor,
    n_clusters_list: list[int] = [5, 8, 12],
    true_labels: np.ndarray | None = None,
) -> dict:
    """Evaluate clustering quality of embeddings at various k values.

    Args:
        embeddings: (N, D) array of clip embeddings.
        n_clusters_list: List of k values to test.
        true_labels: Optional ground-truth phase labels for supervised metrics.

    Returns:
        Dict of results per k value.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    results = {}

    for k in n_clusters_list:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)

        metrics = {
            "n_clusters": k,
            "silhouette": float(silhouette_score(embeddings, cluster_labels)),
            "calinski_harabasz": float(calinski_harabasz_score(embeddings, cluster_labels)),
            "davies_bouldin": float(davies_bouldin_score(embeddings, cluster_labels)),
            "inertia": float(kmeans.inertia_),
        }

        # If we have ground truth, compute supervised metrics
        if true_labels is not None:
            metrics["ari"] = float(adjusted_rand_score(true_labels, cluster_labels))
            metrics["nmi"] = float(normalized_mutual_info_score(true_labels, cluster_labels))

        results[k] = metrics
        logger.info(
            f"k={k}: silhouette={metrics['silhouette']:.3f}, "
            f"CH={metrics['calinski_harabasz']:.1f}, "
            f"DB={metrics['davies_bouldin']:.3f}"
        )

    return results


def plot_embedding_space(
    embeddings: np.ndarray | torch.Tensor,
    labels: np.ndarray | None = None,
    method: str = "tsne",
    output_path: str | None = None,
    title: str = "Clip Embedding Space",
) -> None:
    """Visualize embedding space using dimensionality reduction.

    Args:
        embeddings: (N, D) embeddings.
        labels: Optional labels for coloring (phase indices or cluster assignments).
        method: "tsne" or "pca".
        output_path: If provided, save figure to this path.
        title: Plot title.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    else:
        reducer = PCA(n_components=2)

    coords = reducer.fit_transform(embeddings)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if labels is not None:
        scatter = ax.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap="tab10", alpha=0.6, s=20,
        )
        plt.colorbar(scatter, ax=ax, label="Phase / Cluster")
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)

    ax.set_title(title)
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved embedding plot to {output_path}")

    plt.close(fig)


def plot_codebook_utilization(
    token_indices: np.ndarray | torch.Tensor,
    codebook_size: int,
    output_path: str | None = None,
) -> dict:
    """Analyze and plot codebook token utilization.

    Args:
        token_indices: Array of assigned codebook indices.
        codebook_size: Total codebook size.
        output_path: If provided, save figure.

    Returns:
        Utilization statistics.
    """
    if isinstance(token_indices, torch.Tensor):
        token_indices = token_indices.numpy()

    counts = np.bincount(token_indices, minlength=codebook_size)
    used = np.sum(counts > 0)
    entropy = -np.sum((counts / counts.sum()) * np.log2(counts / counts.sum() + 1e-10))
    max_entropy = np.log2(codebook_size)

    stats = {
        "used_tokens": int(used),
        "total_tokens": codebook_size,
        "utilization": float(used / codebook_size),
        "entropy": float(entropy),
        "max_entropy": float(max_entropy),
        "normalized_entropy": float(entropy / max_entropy),
    }

    logger.info(
        f"Codebook utilization: {used}/{codebook_size} ({stats['utilization']:.1%}), "
        f"entropy: {entropy:.2f}/{max_entropy:.2f} ({stats['normalized_entropy']:.1%})"
    )

    if output_path:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram of token frequencies
        axes[0].bar(range(codebook_size), counts, width=1.0, alpha=0.7)
        axes[0].set_xlabel("Token Index")
        axes[0].set_ylabel("Frequency")
        axes[0].set_title("Codebook Token Frequency")

        # Sorted frequency distribution
        sorted_counts = np.sort(counts)[::-1]
        axes[1].plot(sorted_counts)
        axes[1].set_xlabel("Token Rank")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Token Frequency (Ranked)")
        axes[1].set_yscale("log")

        fig.suptitle(f"Codebook Utilization: {stats['utilization']:.1%}")
        fig.tight_layout()

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return stats
