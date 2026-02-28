"""DPO training for surgical procedure preference optimization.

Uses trl's DPOTrainer to optimize the sequence model to prefer
"better" procedure compositions (top 30% outcomes) over "good" ones.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from datasets import Dataset as HFDataset
from transformers import (
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    TrainingArguments,
)
from trl import DPOConfig, DPOTrainer

logger = logging.getLogger(__name__)


class ClipTokenTokenizer:
    """Minimal tokenizer interface for clip tokens.

    trl's DPOTrainer expects a tokenizer-like object. This wraps our
    clip token vocabulary to satisfy that interface.
    """

    def __init__(self, vocab_size: int, max_length: int = 512, pad_token_id: int = 2):
        self.vocab_size = vocab_size
        self.model_max_length = max_length
        self.pad_token_id = pad_token_id
        self.eos_token_id = 1
        self.bos_token_id = 0
        self.padding_side = "right"
        self.is_fast = False

    @property
    def pad_token(self):
        return str(self.pad_token_id)

    @property
    def eos_token(self):
        return str(self.eos_token_id)

    def __call__(self, text, **kwargs):
        """Convert token string sequences to tensors."""
        if isinstance(text, str):
            text = [text]

        batch_ids = []
        batch_masks = []
        for t in text:
            ids = [int(x) for x in t.strip().split()]
            mask = [1] * len(ids)
            # Pad
            pad_len = self.model_max_length - len(ids)
            if pad_len > 0:
                ids = ids + [self.pad_token_id] * pad_len
                mask = mask + [0] * pad_len
            else:
                ids = ids[: self.model_max_length]
                mask = mask[: self.model_max_length]
            batch_ids.append(ids)
            batch_masks.append(mask)

        return {
            "input_ids": torch.tensor(batch_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_masks, dtype=torch.long),
        }

    def decode(self, ids, **kwargs):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(i) for i in ids)


def prepare_dpo_dataset(
    chosen_sequences: list[list[int]],
    rejected_sequences: list[list[int]],
    bos_token: int = 0,
    eos_token: int = 1,
) -> HFDataset:
    """Prepare a HuggingFace Dataset for DPO training.

    DPOTrainer expects a dataset with 'chosen' and 'rejected' columns
    containing text (which our tokenizer will parse).
    """
    assert len(chosen_sequences) == len(rejected_sequences)

    def tokens_to_str(tokens: list[int]) -> str:
        return " ".join(str(t) for t in [bos_token] + tokens + [eos_token])

    data = {
        "chosen": [tokens_to_str(seq) for seq in chosen_sequences],
        "rejected": [tokens_to_str(seq) for seq in rejected_sequences],
        "prompt": [""] * len(chosen_sequences),  # Unconditional generation
    }

    return HFDataset.from_dict(data)


def train_dpo(
    model_path: str,
    chosen_sequences: list[list[int]],
    rejected_sequences: list[list[int]],
    cfg: dict,
    output_dir: str = "./outputs/dpo",
) -> None:
    """Run DPO training on the pretrained sequence model.

    Args:
        model_path: Path to pretrained sequence model checkpoint.
        chosen_sequences: Token sequences from "better" procedures.
        rejected_sequences: Token sequences from "good" procedures.
        cfg: Full config dict.
        output_dir: Where to save the DPO-trained model.
    """
    dpo_cfg = cfg["dpo"]
    train_cfg = dpo_cfg["training"]
    sm_cfg = cfg["sequence_model"]
    special = sm_cfg["special_tokens"]

    # Load pretrained model
    model = GPT2LMHeadModel.from_pretrained(model_path)
    ref_model = GPT2LMHeadModel.from_pretrained(model_path)

    # Create tokenizer wrapper
    total_vocab = sm_cfg["vocab_size"] + 3
    tokenizer = ClipTokenTokenizer(
        vocab_size=total_vocab,
        max_length=sm_cfg["n_positions"],
        pad_token_id=special["pad"],
    )

    # Prepare dataset
    dataset = prepare_dpo_dataset(
        chosen_sequences=chosen_sequences,
        rejected_sequences=rejected_sequences,
        bos_token=special["bos"],
        eos_token=special["eos"],
    )

    # Split 90/10
    split = dataset.train_test_split(test_size=0.1, seed=cfg["infra"]["seed"])

    # Determine dtype
    dtype_map = {"bf16": "bf16", "fp16": "fp16", "fp32": "no"}
    mixed_precision = dtype_map.get(cfg["infra"]["dtype"], "no")

    dpo_config = DPOConfig(
        output_dir=output_dir,
        beta=dpo_cfg["beta"],
        loss_type=dpo_cfg["loss_type"],
        label_smoothing=dpo_cfg["label_smoothing"],
        num_train_epochs=train_cfg["epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        per_device_eval_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["lr"],
        warmup_ratio=train_cfg["warmup_ratio"],
        max_grad_norm=train_cfg["max_grad_norm"],
        bf16=mixed_precision == "bf16",
        fp16=mixed_precision == "fp16",
        logging_steps=cfg["infra"]["logging"]["log_every_n_steps"],
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        seed=cfg["infra"]["seed"],
        report_to="wandb",
        run_name="surgical-tokens-dpo",
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=split["train"],
        eval_dataset=split["test"],
        processing_class=tokenizer,
    )

    logger.info(
        f"Starting DPO training: beta={dpo_cfg['beta']}, "
        f"{len(split['train'])} train pairs, {len(split['test'])} eval pairs"
    )
    trainer.train()
    trainer.save_model(Path(output_dir) / "final")
    logger.info(f"DPO training complete. Model saved to {output_dir}/final")
