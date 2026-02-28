"""Full training pipeline: pretrain sequence model → DPO alignment.

Orchestrates the complete training workflow from tokenized procedures
through pretraining and DPO fine-tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from surgical_tokens.config import get_paths, load_config
from surgical_tokens.data.case_log import generate_dpo_pairs, label_outcomes, load_case_log
from surgical_tokens.training.dpo import train_dpo
from surgical_tokens.training.pretrain import pretrain_sequence_model

logger = logging.getLogger(__name__)


def build_procedure_sequences(
    tokens_path: str,
    clip_mapping_path: str,
) -> dict[str, list[int]]:
    """Assemble per-clip tokens into per-procedure token sequences.

    Args:
        tokens_path: Path to tokenized_clips.pt (from vocabulary pipeline).
        clip_mapping_path: Path to clip_mapping.json (clip_id → video_id mapping).

    Returns:
        Dict mapping video_id → list of token indices (ordered by clip_index).
    """
    token_data = torch.load(tokens_path, weights_only=True)
    tokens = token_data["tokens"]
    clip_paths = token_data.get("clip_paths", [])

    with open(clip_mapping_path) as f:
        clip_mapping = json.load(f)

    # Build clip_id → token index mapping
    clip_id_to_idx = {}
    for i, path in enumerate(clip_paths):
        # Extract clip_id from path
        clip_id = Path(path).stem
        clip_id_to_idx[clip_id] = i

    # Group by video_id and order by clip_index
    procedures = {}
    for entry in clip_mapping:
        vid = entry["video_id"]
        clip_id = entry["clip_id"]
        clip_idx = entry["clip_index"]

        if clip_id in clip_id_to_idx:
            token_idx = clip_id_to_idx[clip_id]
            token_val = int(tokens[token_idx].item())
            if vid not in procedures:
                procedures[vid] = []
            procedures[vid].append((clip_idx, token_val))

    # Sort by clip_index and extract just tokens
    for vid in procedures:
        procedures[vid] = [t for _, t in sorted(procedures[vid])]

    logger.info(f"Built {len(procedures)} procedure sequences")
    return procedures


def run_training_pipeline(
    case_log_path: str,
    tokens_path: str,
    clip_mapping_path: str,
    config_path: str = "configs/default.yaml",
    skip_pretrain: bool = False,
    pretrained_model_path: str | None = None,
) -> dict:
    """Run the full training pipeline.

    Steps:
        1. Build procedure sequences from tokenized clips
        2. Load case log and compute outcome labels
        3. Pretrain sequence model (next token prediction)
        4. Generate DPO pairs from outcome labels
        5. Run DPO training

    Returns:
        Dict with paths to all trained models.
    """
    cfg = load_config(config_path)
    paths = get_paths(cfg)

    # 1. Build procedure sequences
    logger.info("=== Building Procedure Sequences ===")
    procedures = build_procedure_sequences(tokens_path, clip_mapping_path)
    all_sequences = list(procedures.values())

    # 2. Load and label case log
    logger.info("=== Loading Case Log & Computing Outcomes ===")
    case_log = load_case_log(case_log_path)
    case_log = label_outcomes(case_log, cfg)

    # 3. Pretrain
    pretrain_dir = str(paths.models_dir / "pretrain")
    if not skip_pretrain:
        logger.info("=== Pretraining Sequence Model ===")

        # Load codebook weights for embedding initialization
        codebook_path = paths.codebook_dir / "codebook.pt"
        codebook_weights = None
        if codebook_path.exists():
            cb_data = torch.load(codebook_path, weights_only=True)
            codebook_weights = cb_data["codebook_weights"]

        # Train/val split
        rng = np.random.default_rng(cfg["infra"]["seed"])
        indices = rng.permutation(len(all_sequences))
        split = int(0.9 * len(indices))
        train_seqs = [all_sequences[i] for i in indices[:split]]
        val_seqs = [all_sequences[i] for i in indices[split:]]

        pretrain_sequence_model(
            train_sequences=train_seqs,
            val_sequences=val_seqs,
            cfg=cfg,
            output_dir=pretrain_dir,
            codebook_weights=codebook_weights,
        )
        pretrained_model_path = str(Path(pretrain_dir) / "final")

    # 4. Generate DPO pairs
    logger.info("=== Generating DPO Pairs ===")
    dpo_pairs = generate_dpo_pairs(case_log)

    # Map DPO pairs to token sequences
    chosen_seqs = []
    rejected_seqs = []
    for pair in dpo_pairs:
        if pair.chosen_video_id in procedures and pair.rejected_video_id in procedures:
            chosen_seqs.append(procedures[pair.chosen_video_id])
            rejected_seqs.append(procedures[pair.rejected_video_id])

    logger.info(f"Mapped {len(chosen_seqs)} DPO pairs to token sequences")

    if len(chosen_seqs) < 10:
        logger.warning("Very few DPO pairs — consider adjusting percentile thresholds")

    # 5. DPO training
    dpo_dir = str(paths.models_dir / "dpo")
    logger.info("=== DPO Training ===")
    train_dpo(
        model_path=pretrained_model_path,
        chosen_sequences=chosen_seqs,
        rejected_sequences=rejected_seqs,
        cfg=cfg,
        output_dir=dpo_dir,
    )

    return {
        "pretrain_model": pretrained_model_path,
        "dpo_model": str(Path(dpo_dir) / "final"),
        "n_procedures": len(procedures),
        "n_dpo_pairs": len(chosen_seqs),
    }
