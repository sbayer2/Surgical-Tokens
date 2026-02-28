"""VQ-VAE sparse encoder for discretizing clip embeddings into a procedural vocabulary.

Takes dense VideoMAE embeddings and maps them to a finite codebook,
converting each clip into one or more discrete tokens.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VectorQuantizer(nn.Module):
    """Vector Quantizer with EMA codebook updates.

    Based on the VQ-VAE paper (van den Oord et al., 2017) with
    exponential moving average codebook learning.
    """

    def __init__(
        self,
        codebook_size: int = 2048,
        embedding_dim: int = 768,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay

        # Codebook
        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", self.embedding.weight.clone())

    def forward(self, z: torch.Tensor) -> dict:
        """Quantize input embeddings.

        Args:
            z: (B, D) continuous embeddings

        Returns:
            dict with keys:
                quantized: (B, D) quantized embeddings
                indices: (B,) codebook indices
                commitment_loss: scalar
                codebook_loss: scalar
                perplexity: scalar (codebook utilization)
        """
        # Compute distances to codebook
        distances = (
            torch.sum(z ** 2, dim=1, keepdim=True)
            + torch.sum(self.embedding.weight ** 2, dim=1)
            - 2 * torch.matmul(z, self.embedding.weight.t())
        )

        # Nearest codebook entry
        indices = torch.argmin(distances, dim=1)
        quantized = self.embedding(indices)

        # EMA update (training only)
        if self.training:
            with torch.no_grad():
                encodings = F.one_hot(indices, self.codebook_size).float()
                self.ema_cluster_size.mul_(self.decay).add_(
                    encodings.sum(0), alpha=1 - self.decay
                )
                dw = encodings.t() @ z
                self.ema_weight.mul_(self.decay).add_(dw, alpha=1 - self.decay)

                # Laplace smoothing
                n = self.ema_cluster_size.sum()
                cluster_size = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.codebook_size * 1e-5)
                    * n
                )
                self.embedding.weight.data.copy_(self.ema_weight / cluster_size.unsqueeze(1))

        # Losses
        commitment_loss = F.mse_loss(z, quantized.detach())
        codebook_loss = F.mse_loss(quantized, z.detach())

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity (codebook utilization metric)
        avg_probs = torch.histc(
            indices.float(), bins=self.codebook_size, min=0, max=self.codebook_size - 1
        ) / indices.numel()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return {
            "quantized": quantized,
            "indices": indices,
            "commitment_loss": commitment_loss,
            "codebook_loss": codebook_loss,
            "perplexity": perplexity,
        }

    def encode(self, z: torch.Tensor) -> torch.Tensor:
        """Encode to codebook indices without gradient tracking."""
        with torch.no_grad():
            distances = (
                torch.sum(z ** 2, dim=1, keepdim=True)
                + torch.sum(self.embedding.weight ** 2, dim=1)
                - 2 * torch.matmul(z, self.embedding.weight.t())
            )
            return torch.argmin(distances, dim=1)

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode codebook indices back to embeddings."""
        return self.embedding(indices)


class SparseEncoder(nn.Module):
    """Full sparse encoding module: encoder → VQ → decoder.

    Learns to compress dense embeddings through a bottleneck with
    discrete codebook, then reconstruct.
    """

    def __init__(
        self,
        input_dim: int = 768,
        codebook_size: int = 2048,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
    ):
        super().__init__()

        # Pre-quantization projection (optional dimensionality adjustment)
        self.pre_quant = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
        )

        self.vq = VectorQuantizer(
            codebook_size=codebook_size,
            embedding_dim=input_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        # Post-quantization decoder
        self.post_quant = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass through encoder → VQ → decoder.

        Args:
            x: (B, D) input embeddings

        Returns:
            dict with reconstructed, indices, and all losses
        """
        z = self.pre_quant(x)
        vq_out = self.vq(z)

        reconstructed = self.post_quant(vq_out["quantized"])
        recon_loss = F.mse_loss(reconstructed, x)

        return {
            "reconstructed": reconstructed,
            "indices": vq_out["indices"],
            "recon_loss": recon_loss,
            "commitment_loss": vq_out["commitment_loss"],
            "codebook_loss": vq_out["codebook_loss"],
            "perplexity": vq_out["perplexity"],
        }

    def tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Convert embeddings to codebook tokens."""
        z = self.pre_quant(x)
        return self.vq.encode(z)

    def detokenize(self, indices: torch.Tensor) -> torch.Tensor:
        """Convert codebook tokens back to embeddings."""
        z_q = self.vq.decode(indices)
        return self.post_quant(z_q)


def train_sparse_encoder(
    embeddings: torch.Tensor,
    cfg: dict,
    device: str = "cuda",
    output_dir: str | None = None,
) -> SparseEncoder:
    """Train the VQ-VAE sparse encoder on clip embeddings.

    Args:
        embeddings: (N, D) tensor of clip embeddings.
        cfg: Full config dict.
        device: Training device.
        output_dir: If provided, save checkpoints here.

    Returns:
        Trained SparseEncoder.
    """
    se_cfg = cfg["sparse_encoder"]
    train_cfg = se_cfg["training"]

    model = SparseEncoder(
        input_dim=se_cfg["embedding_dim"],
        codebook_size=se_cfg["codebook_size"],
        commitment_cost=se_cfg["commitment_cost"],
        decay=se_cfg["decay"],
    ).to(device)

    dataset = TensorDataset(embeddings)
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"])

    # Warmup + cosine schedule
    total_steps = train_cfg["epochs"] * len(loader)
    warmup_steps = train_cfg.get("warmup_steps", 500)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    import numpy as np

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    logger.info(f"Training sparse encoder: {train_cfg['epochs']} epochs, {len(loader)} batches")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_perplexity = 0

        for (batch,) in loader:
            batch = batch.to(device)
            out = model(batch)

            loss = (
                out["recon_loss"]
                + se_cfg["commitment_cost"] * out["commitment_loss"]
                + out["codebook_loss"]
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            epoch_perplexity += out["perplexity"].item()

        avg_loss = epoch_loss / len(loader)
        avg_perplexity = epoch_perplexity / len(loader)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch+1}/{train_cfg['epochs']} — "
                f"loss: {avg_loss:.4f}, perplexity: {avg_perplexity:.1f}/{se_cfg['codebook_size']}"
            )

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(output_dir) / "sparse_encoder.pt"
        torch.save(model.state_dict(), save_path)
        logger.info(f"Saved sparse encoder to {save_path}")

    return model
