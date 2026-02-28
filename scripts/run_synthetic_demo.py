#!/usr/bin/env python3
"""End-to-end demo using synthetic data.

Runs the entire pipeline on synthetic data to validate
that all components work together before using real ACS data.

Usage:
    python scripts/run_synthetic_demo.py [--output-dir ./demo_output]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from surgical_tokens.config import load_config
from surgical_tokens.data.case_log import generate_dpo_pairs, label_outcomes
from surgical_tokens.data.synthetic import (
    generate_full_synthetic_dataset,
    generate_synthetic_procedures,
)
from surgical_tokens.encoding.sparse_encoder import SparseEncoder, train_sparse_encoder
from surgical_tokens.evaluation.clustering import (
    evaluate_clustering,
    plot_codebook_utilization,
    plot_embedding_space,
)
from surgical_tokens.evaluation.metrics import build_transition_matrix, procedural_coherence_score
from surgical_tokens.evaluation.procedure_eval import (
    evaluate_generation_quality,
    evaluate_preference_accuracy,
)
from surgical_tokens.models.sequence_model import (
    ProcedureDataset,
    create_sequence_model,
    generate_procedure,
)
from surgical_tokens.training.distributed import estimate_memory_requirements, setup_distributed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(output_dir: str = "./demo_output"):
    output_dir = Path(output_dir)
    cfg = load_config("configs/default.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ─── System Info ──────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("SURGICAL TOKENS — SYNTHETIC DEMO")
    logger.info("=" * 60)
    env = setup_distributed()
    mem = estimate_memory_requirements(cfg)
    logger.info(f"Device: {device}, Model: {mem['n_params_millions']}M params")

    # ─── 1. Generate Synthetic Data ──────────────────────────
    logger.info("\n=== STAGE 1: Generating Synthetic Data ===")
    data_dir = output_dir / "synthetic_data"
    artifacts = generate_full_synthetic_dataset(
        output_dir=data_dir,
        n_cases=50,
        n_clips_per_case=10,
        embedding_dim=cfg["sparse_encoder"]["embedding_dim"],
    )
    logger.info(f"Generated: {json.dumps(artifacts, indent=2)}")

    # ─── 2. Clustering Validation ────────────────────────────
    logger.info("\n=== STAGE 2: Clustering Validation (No Labels) ===")
    emb_data = torch.load(artifacts["embeddings"], weights_only=False)
    embeddings = emb_data["embeddings"]
    phase_labels = emb_data["phase_labels"]

    cluster_results = evaluate_clustering(
        embeddings.numpy(),
        n_clusters_list=cfg["evaluation"]["clustering"]["n_clusters"],
        true_labels=phase_labels,
    )

    eval_dir = output_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    plot_embedding_space(
        embeddings.numpy(),
        labels=phase_labels,
        method="tsne",
        output_path=str(eval_dir / "embedding_tsne.png"),
        title="Synthetic Clip Embeddings (colored by surgical phase)",
    )

    # ─── 3. Train VQ-VAE ────────────────────────────────────
    logger.info("\n=== STAGE 3: Training VQ-VAE Sparse Encoder ===")
    # Use smaller config for demo speed
    demo_sparse_cfg = {**cfg, "sparse_encoder": {
        **cfg["sparse_encoder"],
        "codebook_size": 256,  # Smaller for demo
        "training": {**cfg["sparse_encoder"]["training"], "epochs": 10, "batch_size": 64},
    }}

    encoder = train_sparse_encoder(
        embeddings=embeddings,
        cfg=demo_sparse_cfg,
        device=device,
        output_dir=str(output_dir / "codebook"),
    )

    # Tokenize
    encoder.eval()
    with torch.no_grad():
        tokens = encoder.tokenize(embeddings.to(device)).cpu()

    plot_codebook_utilization(
        tokens.numpy(),
        codebook_size=256,
        output_path=str(eval_dir / "codebook_utilization.png"),
    )

    # ─── 4. Sequence Model (Quick Test) ─────────────────────
    logger.info("\n=== STAGE 4: Sequence Model Quick Test ===")
    # Use synthetic procedure sequences
    with open(artifacts["procedures"]) as f:
        procedures = json.load(f)

    # Remap tokens to smaller codebook range
    all_seqs = []
    for proc in procedures:
        remapped = [t % 256 + 3 for t in proc["tokens"]]  # +3 for special tokens
        all_seqs.append(remapped)

    # Create a small model for demo
    demo_model_cfg = {**cfg, "sequence_model": {
        **cfg["sequence_model"],
        "vocab_size": 256,
        "n_embd": 128,
        "n_layer": 4,
        "n_head": 4,
        "n_positions": 256,
        "pretraining": {
            **cfg["sequence_model"]["pretraining"],
            "epochs": 2,
            "batch_size": 8,
        },
    }}

    model = create_sequence_model(demo_model_cfg)
    model = model.to(device)

    # Quick training loop (without full Trainer for demo speed)
    dataset = ProcedureDataset(
        sequences=all_seqs,
        max_length=256,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

    model.train()
    for epoch in range(2):
        total_loss = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        logger.info(f"Epoch {epoch+1}: loss={total_loss/len(loader):.4f}")

    # Generate some procedures
    logger.info("Generating sample procedures...")
    model.eval()
    generated = []
    for _ in range(10):
        seq = generate_procedure(
            model, max_length=30, temperature=0.9, device=device,
        )
        generated.append(seq)
        logger.info(f"  Generated: {seq[:15]}{'...' if len(seq) > 15 else ''}")

    # ─── 5. Generation Quality Metrics ──────────────────────
    logger.info("\n=== STAGE 5: Evaluation Metrics ===")
    gen_quality = evaluate_generation_quality(
        generated_sequences=generated,
        reference_sequences=all_seqs[:20],
        codebook_size=256 + 3,
    )
    logger.info(f"Generation quality: {json.dumps(gen_quality, indent=2)}")

    # Transition coherence
    transition_matrix = build_transition_matrix(all_seqs, vocab_size=256 + 3)
    coherence_scores = [procedural_coherence_score(seq, transition_matrix) for seq in generated]
    ref_coherence = [procedural_coherence_score(seq, transition_matrix) for seq in all_seqs[:20]]
    logger.info(
        f"Coherence — generated: {np.mean(coherence_scores):.4f}, "
        f"reference: {np.mean(ref_coherence):.4f}"
    )

    # ─── 6. Case Log & Outcome Labeling Demo ────────────────
    logger.info("\n=== STAGE 6: Outcome Labeling & DPO Pairs ===")
    import pandas as pd

    case_log = pd.read_csv(artifacts["case_log"])
    labeled = label_outcomes(case_log, cfg)
    dpo_pairs = generate_dpo_pairs(labeled, max_pairs=100)
    logger.info(f"Generated {len(dpo_pairs)} DPO pairs from {len(labeled)} cases")

    # ─── Summary ─────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("DEMO COMPLETE — Summary")
    logger.info("=" * 60)
    logger.info(f"Synthetic data:     {data_dir}")
    logger.info(f"Evaluation plots:   {eval_dir}")
    logger.info(f"Codebook:           {output_dir / 'codebook'}")
    logger.info(f"Embeddings:         {embeddings.shape}")
    logger.info(f"Codebook tokens:    {tokens.shape}")
    logger.info(f"Procedures:         {len(all_seqs)}")
    logger.info(f"DPO pairs:          {len(dpo_pairs)}")
    logger.info(f"Best clustering k:  {max(cluster_results, key=lambda k: cluster_results[k]['silhouette'])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./demo_output")
    args = parser.parse_args()
    main(args.output_dir)
