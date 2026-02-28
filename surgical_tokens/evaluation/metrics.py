"""Custom metrics for surgical procedure evaluation."""

from __future__ import annotations

import numpy as np


def procedural_coherence_score(
    token_sequence: list[int],
    transition_matrix: np.ndarray,
) -> float:
    """Measure how well a token sequence follows learned transition patterns.

    A high coherence score means the model generates sequences that follow
    the natural progression patterns seen in training data.

    Args:
        token_sequence: List of codebook token indices.
        transition_matrix: (V, V) matrix of transition probabilities learned from data.

    Returns:
        Mean log-probability of transitions (higher = more coherent).
    """
    if len(token_sequence) < 2:
        return 0.0

    log_probs = []
    for i in range(len(token_sequence) - 1):
        src = token_sequence[i]
        dst = token_sequence[i + 1]
        prob = transition_matrix[src, dst]
        log_probs.append(np.log(prob + 1e-10))

    return float(np.mean(log_probs))


def build_transition_matrix(
    sequences: list[list[int]],
    vocab_size: int,
    smoothing: float = 1e-6,
) -> np.ndarray:
    """Build a token transition probability matrix from training sequences.

    Args:
        sequences: List of token sequences.
        vocab_size: Size of the token vocabulary.
        smoothing: Laplace smoothing constant.

    Returns:
        (vocab_size, vocab_size) transition probability matrix.
    """
    counts = np.full((vocab_size, vocab_size), smoothing)

    for seq in sequences:
        for i in range(len(seq) - 1):
            counts[seq[i], seq[i + 1]] += 1

    # Normalize rows
    row_sums = counts.sum(axis=1, keepdims=True)
    return counts / row_sums


def outcome_prediction_accuracy(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> dict:
    """Evaluate binary classification (good vs better) accuracy.

    Args:
        predicted_labels: Model's predicted labels (0=good, 1=better).
        true_labels: Ground truth labels.

    Returns:
        Dict with accuracy, precision, recall, F1.
    """
    tp = np.sum((predicted_labels == 1) & (true_labels == 1))
    fp = np.sum((predicted_labels == 1) & (true_labels == 0))
    fn = np.sum((predicted_labels == 0) & (true_labels == 1))
    tn = np.sum((predicted_labels == 0) & (true_labels == 0))

    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
