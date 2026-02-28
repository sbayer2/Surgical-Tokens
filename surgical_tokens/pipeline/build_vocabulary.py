"""Pipeline: embeddings → VQ-VAE training → codebook vocabulary → tokenized procedures."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from surgical_tokens.config import get_paths, load_config
from surgical_tokens.encoding.sparse_encoder import SparseEncoder, train_sparse_encoder

logger = logging.getLogger(__name__)


def run_vocabulary_pipeline(
    embeddings_path: str,
    config_path: str = "configs/default.yaml",
) -> dict:
    """Train VQ-VAE and build clip token vocabulary.

    Steps:
        1. Load clip embeddings
        2. Train sparse encoder (VQ-VAE)
        3. Tokenize all embeddings
        4. Save codebook and tokenized sequences

    Returns:
        Dict with paths and stats.
    """
    cfg = load_config(config_path)
    paths = get_paths(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load embeddings
    logger.info("Loading embeddings...")
    data = torch.load(embeddings_path, weights_only=True)
    embeddings = data["embeddings"]
    clip_paths = data.get("clip_paths", [])
    logger.info(f"Loaded {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")

    # 2. Train sparse encoder
    logger.info("=== Training VQ-VAE Sparse Encoder ===")
    encoder = train_sparse_encoder(
        embeddings=embeddings,
        cfg=cfg,
        device=device,
        output_dir=str(paths.codebook_dir),
    )

    # 3. Tokenize all embeddings
    logger.info("Tokenizing embeddings...")
    encoder.eval()
    encoder = encoder.to(device)
    with torch.no_grad():
        tokens = encoder.tokenize(embeddings.to(device))
    tokens = tokens.cpu()

    # Offset by 3 for special tokens (BOS=0, EOS=1, PAD=2)
    tokens = tokens + 3

    # 4. Save
    codebook_weights = encoder.vq.embedding.weight.detach().cpu()
    torch.save({
        "codebook_weights": codebook_weights,
        "codebook_size": cfg["sparse_encoder"]["codebook_size"],
        "embedding_dim": cfg["sparse_encoder"]["embedding_dim"],
    }, paths.codebook_dir / "codebook.pt")

    torch.save({
        "tokens": tokens,
        "clip_paths": clip_paths,
    }, paths.codebook_dir / "tokenized_clips.pt")

    # Stats
    unique_tokens = len(tokens.unique())
    utilization = unique_tokens / cfg["sparse_encoder"]["codebook_size"]

    results = {
        "codebook_size": cfg["sparse_encoder"]["codebook_size"],
        "unique_tokens_used": unique_tokens,
        "utilization": float(utilization),
        "codebook_path": str(paths.codebook_dir / "codebook.pt"),
        "tokens_path": str(paths.codebook_dir / "tokenized_clips.pt"),
    }

    logger.info(f"Codebook utilization: {unique_tokens}/{cfg['sparse_encoder']['codebook_size']} ({utilization:.1%})")
    return results
