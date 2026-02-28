"""Pretraining: next clip-token prediction on procedure sequences.

Standard autoregressive language modeling objective, but over surgical clip tokens
instead of text tokens. Uses HuggingFace Trainer with Accelerate for multi-GPU.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from transformers import Trainer, TrainingArguments

from surgical_tokens.models.sequence_model import ProcedureDataset, create_sequence_model

logger = logging.getLogger(__name__)


def pretrain_sequence_model(
    train_sequences: list[list[int]],
    val_sequences: list[list[int]] | None = None,
    cfg: dict = None,
    output_dir: str = "./outputs/pretrain",
    codebook_weights: torch.Tensor | None = None,
) -> None:
    """Pretrain the clip-token sequence model.

    Args:
        train_sequences: List of token sequences for training.
        val_sequences: Optional validation sequences.
        cfg: Full config dict.
        output_dir: Where to save checkpoints.
        codebook_weights: Optional codebook embeddings to initialize from.
    """
    sm_cfg = cfg["sequence_model"]
    train_cfg = sm_cfg["pretraining"]
    special = sm_cfg["special_tokens"]

    # Create model
    model = create_sequence_model(cfg)

    # Optionally initialize from codebook
    if codebook_weights is not None:
        from surgical_tokens.models.sequence_model import initialize_embeddings_from_codebook

        model = initialize_embeddings_from_codebook(
            model, codebook_weights, sm_cfg["n_embd"]
        )

    # Create datasets
    train_dataset = ProcedureDataset(
        sequences=train_sequences,
        max_length=sm_cfg["n_positions"],
        bos_token=special["bos"],
        eos_token=special["eos"],
        pad_token=special["pad"],
    )
    val_dataset = None
    if val_sequences:
        val_dataset = ProcedureDataset(
            sequences=val_sequences,
            max_length=sm_cfg["n_positions"],
            bos_token=special["bos"],
            eos_token=special["eos"],
            pad_token=special["pad"],
        )

    # Determine dtype
    dtype_map = {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}
    mixed_precision = dtype_map.get(cfg["infra"]["dtype"], "no")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["lr"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        bf16=mixed_precision == "bf16",
        fp16=mixed_precision == "fp16",
        logging_steps=cfg["infra"]["logging"]["log_every_n_steps"],
        save_strategy="epoch",
        eval_strategy="epoch" if val_dataset else "no",
        save_total_limit=3,
        dataloader_num_workers=cfg["infra"]["num_workers"],
        dataloader_pin_memory=cfg["infra"]["pin_memory"],
        seed=cfg["infra"]["seed"],
        report_to="wandb",
        run_name="surgical-tokens-pretrain",
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    logger.info("Starting sequence model pretraining")
    trainer.train()
    trainer.save_model(Path(output_dir) / "final")
    logger.info(f"Pretraining complete. Model saved to {output_dir}/final")
