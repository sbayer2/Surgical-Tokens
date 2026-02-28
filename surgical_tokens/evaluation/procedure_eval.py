"""Procedure-level evaluation: compare model assessments against expert skill ratings.

Evaluates:
  1. Correlation between model-predicted quality and expert OSATS-style ratings
  2. Procedure generation quality (does the model produce plausible sequences?)
  3. DPO alignment (does the model actually prefer "better" procedures?)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from scipy.stats import kendalltau, spearmanr

logger = logging.getLogger(__name__)


def evaluate_skill_correlation(
    model_scores: np.ndarray,
    expert_ratings: np.ndarray,
) -> dict:
    """Compute correlation between model quality scores and expert ratings.

    Args:
        model_scores: (N,) array of model-assigned quality scores.
        expert_ratings: (N,) array of expert OSATS-style ratings (1-5).

    Returns:
        Dict with correlation metrics.
    """
    spearman_r, spearman_p = spearmanr(model_scores, expert_ratings)
    kendall_tau, kendall_p = kendalltau(model_scores, expert_ratings)

    results = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
        "n_samples": len(model_scores),
    }

    logger.info(
        f"Skill correlation: Spearman r={spearman_r:.3f} (p={spearman_p:.4f}), "
        f"Kendall τ={kendall_tau:.3f} (p={kendall_p:.4f})"
    )
    return results


def evaluate_preference_accuracy(
    model,
    chosen_sequences: list[list[int]],
    rejected_sequences: list[list[int]],
    pad_token: int = 2,
    device: str = "cuda",
) -> dict:
    """Evaluate whether the model assigns higher log-prob to chosen vs rejected.

    This is the core alignment metric: after DPO training, the model should
    prefer "better" procedures over "good" ones.
    """
    model.eval()
    model = model.to(device)

    correct = 0
    total = 0
    margins = []

    for chosen, rejected in zip(chosen_sequences, rejected_sequences):
        chosen_lp = _sequence_log_prob(model, chosen, pad_token, device)
        rejected_lp = _sequence_log_prob(model, rejected, pad_token, device)

        margin = chosen_lp - rejected_lp
        margins.append(margin)

        if chosen_lp > rejected_lp:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    margins = np.array(margins)

    results = {
        "preference_accuracy": float(accuracy),
        "mean_margin": float(np.mean(margins)),
        "median_margin": float(np.median(margins)),
        "n_pairs": total,
    }

    logger.info(
        f"Preference accuracy: {accuracy:.1%} ({correct}/{total}), "
        f"mean margin: {np.mean(margins):.4f}"
    )
    return results


@torch.no_grad()
def _sequence_log_prob(
    model,
    token_sequence: list[int],
    pad_token: int,
    device: str,
) -> float:
    """Compute the log probability of a token sequence under the model."""
    tokens = [0] + token_sequence + [1]  # BOS + seq + EOS
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    outputs = model(input_ids=input_ids)
    logits = outputs.logits  # (1, seq_len, vocab_size)

    # Shift: predict token[t+1] from logits[t]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()

    # Log softmax
    log_probs = torch.log_softmax(shift_logits, dim=-1)

    # Gather the log probs for actual tokens
    token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

    # Mask padding
    mask = (shift_labels != pad_token).float()
    seq_log_prob = (token_log_probs * mask).sum() / mask.sum()

    return float(seq_log_prob.cpu())


def evaluate_generation_quality(
    generated_sequences: list[list[int]],
    reference_sequences: list[list[int]],
    codebook_size: int,
) -> dict:
    """Evaluate quality of model-generated procedure sequences.

    Computes:
      - Token distribution similarity (KL divergence from reference)
      - Sequence length statistics
      - Repetition rate (lower is better)
      - Unique n-gram ratios
    """
    # Token distributions
    gen_tokens = [t for seq in generated_sequences for t in seq]
    ref_tokens = [t for seq in reference_sequences for t in seq]

    gen_hist = np.bincount(gen_tokens, minlength=codebook_size).astype(float)
    ref_hist = np.bincount(ref_tokens, minlength=codebook_size).astype(float)
    gen_hist /= gen_hist.sum() + 1e-10
    ref_hist /= ref_hist.sum() + 1e-10

    kl_div = float(np.sum(ref_hist * np.log((ref_hist + 1e-10) / (gen_hist + 1e-10))))

    # Length statistics
    gen_lengths = [len(s) for s in generated_sequences]
    ref_lengths = [len(s) for s in reference_sequences]

    # Repetition rate (consecutive duplicate tokens)
    def repetition_rate(sequences):
        total, repeated = 0, 0
        for seq in sequences:
            for i in range(1, len(seq)):
                total += 1
                if seq[i] == seq[i - 1]:
                    repeated += 1
        return repeated / total if total > 0 else 0

    # Unique n-gram ratio
    def unique_ngram_ratio(sequences, n=3):
        all_ngrams = []
        for seq in sequences:
            for i in range(len(seq) - n + 1):
                all_ngrams.append(tuple(seq[i : i + n]))
        return len(set(all_ngrams)) / len(all_ngrams) if all_ngrams else 0

    results = {
        "kl_divergence": kl_div,
        "gen_mean_length": float(np.mean(gen_lengths)),
        "ref_mean_length": float(np.mean(ref_lengths)),
        "repetition_rate": repetition_rate(generated_sequences),
        "ref_repetition_rate": repetition_rate(reference_sequences),
        "unique_trigram_ratio": unique_ngram_ratio(generated_sequences, 3),
        "ref_unique_trigram_ratio": unique_ngram_ratio(reference_sequences, 3),
    }

    logger.info(
        f"Generation quality: KL={kl_div:.4f}, "
        f"rep_rate={results['repetition_rate']:.3f}, "
        f"trigram_diversity={results['unique_trigram_ratio']:.3f}"
    )
    return results
