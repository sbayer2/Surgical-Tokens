"""GPT-style sequence model over surgical clip tokens.

Replaces the standard text token embedding with clip codebook embeddings.
Learns to predict the next clip token given previous clip tokens in a procedure.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import GPT2Config, GPT2LMHeadModel

logger = logging.getLogger(__name__)


class ProcedureDataset(Dataset):
    """Dataset of surgical procedure token sequences for next-token prediction."""

    def __init__(
        self,
        sequences: list[list[int]],
        max_length: int = 512,
        bos_token: int = 0,
        eos_token: int = 1,
        pad_token: int = 2,
    ):
        self.sequences = sequences
        self.max_length = max_length
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Add special tokens
        tokens = [self.bos] + seq + [self.eos]

        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.eos]

        # Pad
        attention_mask = [1] * len(tokens)
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.pad] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        input_ids = torch.tensor(tokens, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        # Labels: shift right (next token prediction), mask padding with -100
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class DPOPairDataset(Dataset):
    """Dataset of (chosen, rejected) procedure pairs for DPO training."""

    def __init__(
        self,
        chosen_sequences: list[list[int]],
        rejected_sequences: list[list[int]],
        max_length: int = 512,
        bos_token: int = 0,
        eos_token: int = 1,
        pad_token: int = 2,
    ):
        assert len(chosen_sequences) == len(rejected_sequences)
        self.chosen = chosen_sequences
        self.rejected = rejected_sequences
        self.max_length = max_length
        self.bos = bos_token
        self.eos = eos_token
        self.pad = pad_token

    def __len__(self):
        return len(self.chosen)

    def _encode(self, seq: list[int]) -> dict:
        tokens = [self.bos] + seq + [self.eos]
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length - 1] + [self.eos]

        attention_mask = [1] * len(tokens)
        padding_length = self.max_length - len(tokens)
        tokens = tokens + [self.pad] * padding_length
        attention_mask = attention_mask + [0] * padding_length

        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def __getitem__(self, idx):
        chosen = self._encode(self.chosen[idx])
        rejected = self._encode(self.rejected[idx])

        return {
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }


def create_sequence_model(cfg: dict) -> GPT2LMHeadModel:
    """Create a GPT2-based sequence model configured for clip tokens.

    The vocabulary is the codebook (discrete clip tokens) plus special tokens.
    """
    sm_cfg = cfg["sequence_model"]
    special = sm_cfg["special_tokens"]

    # Total vocab = codebook_size + number of special tokens (already included in codebook range)
    # Special tokens occupy indices 0, 1, 2 — codebook starts at 3
    total_vocab = sm_cfg["vocab_size"] + 3  # +3 for BOS, EOS, PAD

    config = GPT2Config(
        vocab_size=total_vocab,
        n_positions=sm_cfg["n_positions"],
        n_embd=sm_cfg["n_embd"],
        n_layer=sm_cfg["n_layer"],
        n_head=sm_cfg["n_head"],
        resid_pdrop=sm_cfg["dropout"],
        embd_pdrop=sm_cfg["dropout"],
        attn_pdrop=sm_cfg["dropout"],
        bos_token_id=special["bos"],
        eos_token_id=special["eos"],
        pad_token_id=special["pad"],
    )

    model = GPT2LMHeadModel(config)
    logger.info(
        f"Created sequence model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params, "
        f"vocab={total_vocab}, layers={sm_cfg['n_layer']}, dim={sm_cfg['n_embd']}"
    )
    return model


def initialize_embeddings_from_codebook(
    model: GPT2LMHeadModel,
    codebook_weights: torch.Tensor,
    projection_dim: int,
) -> GPT2LMHeadModel:
    """Initialize the token embedding layer from the VQ-VAE codebook.

    Projects codebook embeddings (768d) down to model dimension (e.g., 512d)
    and uses them to initialize the embedding table.
    """
    codebook_dim = codebook_weights.shape[1]
    codebook_size = codebook_weights.shape[0]

    # Learn a projection from codebook dim to model dim
    projector = nn.Linear(codebook_dim, projection_dim, bias=False)
    nn.init.orthogonal_(projector.weight)

    with torch.no_grad():
        projected = projector(codebook_weights)

        # model.transformer.wte has shape (total_vocab, n_embd)
        # First 3 entries are special tokens — leave randomly initialized
        # Entries 3 onwards get codebook values
        n_special = 3
        embed_weight = model.transformer.wte.weight
        n_to_copy = min(codebook_size, embed_weight.shape[0] - n_special)
        embed_weight[n_special : n_special + n_to_copy] = projected[:n_to_copy]

    logger.info(
        f"Initialized {n_to_copy} embeddings from codebook "
        f"({codebook_dim}d → {projection_dim}d)"
    )
    return model


@torch.no_grad()
def generate_procedure(
    model: GPT2LMHeadModel,
    prompt_tokens: Optional[list[int]] = None,
    max_length: int = 100,
    temperature: float = 0.8,
    top_k: int = 50,
    bos_token: int = 0,
    eos_token: int = 1,
    device: str = "cuda",
) -> list[int]:
    """Generate a procedure token sequence autoregressively.

    Args:
        model: Trained sequence model.
        prompt_tokens: Optional initial tokens to condition on.
        max_length: Maximum sequence length.
        temperature: Sampling temperature.
        top_k: Top-k sampling.
        bos_token: BOS token id.
        eos_token: EOS token id.
        device: Device.

    Returns:
        List of generated token ids (excluding BOS/EOS).
    """
    model.eval()

    if prompt_tokens is None:
        input_ids = torch.tensor([[bos_token]], dtype=torch.long, device=device)
    else:
        input_ids = torch.tensor(
            [[bos_token] + prompt_tokens], dtype=torch.long, device=device
        )

    generated = []

    for _ in range(max_length):
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            logits[logits < values[:, -1:]] = float("-inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eos_token:
            break

        generated.append(next_token.item())
        input_ids = torch.cat([input_ids, next_token], dim=1)

    return generated
