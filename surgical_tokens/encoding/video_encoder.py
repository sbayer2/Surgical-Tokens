"""VideoMAE-based clip embedding extraction.

Loads a pretrained VideoMAE model, feeds it 2-minute video clips,
and extracts latent representations from the penultimate layer.
Supports optional LoRA fine-tuning for surgical domain adaptation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


class VideoClipDataset(Dataset):
    """Dataset that loads video clips and samples frames for VideoMAE."""

    def __init__(
        self,
        clip_paths: list[str],
        num_frames: int = 16,
        frame_size: int = 224,
    ):
        self.clip_paths = clip_paths
        self.num_frames = num_frames
        self.frame_size = frame_size

    def __len__(self):
        return len(self.clip_paths)

    def __getitem__(self, idx):
        path = self.clip_paths[idx]
        frames = self._load_frames(path)
        return {"pixel_values": frames, "clip_path": path}

    def _load_frames(self, path: str) -> torch.Tensor:
        """Load and uniformly sample frames from a video clip."""
        try:
            from decord import VideoReader, cpu

            vr = VideoReader(path, ctx=cpu(0))
            total = len(vr)

            # Uniform sampling
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)
            frames = vr.get_batch(indices).asnumpy()  # (T, H, W, C)
        except ImportError:
            # Fallback to OpenCV
            import cv2

            cap = cv2.VideoCapture(path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = np.linspace(0, total - 1, self.num_frames, dtype=int)

            frames_list = []
            for i in sorted(indices):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if ret:
                    frames_list.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            frames = np.stack(frames_list)

        # Resize
        import cv2

        resized = np.stack([
            cv2.resize(f, (self.frame_size, self.frame_size))
            for f in frames
        ])

        # Normalize to [0, 1] and convert to (T, C, H, W)
        tensor = torch.from_numpy(resized).float() / 255.0
        tensor = tensor.permute(0, 3, 1, 2)  # (T, C, H, W)
        return tensor


class SurgicalVideoEncoder(nn.Module):
    """Wrapper around VideoMAE for extracting clip embeddings.

    Extracts the [CLS] token representation from the penultimate layer.
    """

    def __init__(
        self,
        model_name: str = "MCG-NJU/videomae-base",
        use_lora: bool = False,
        lora_rank: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()
        from transformers import VideoMAEModel, VideoMAEImageProcessor

        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name)
        self.embedding_dim = self.model.config.hidden_size

        if use_lora:
            self._apply_lora(lora_rank, lora_alpha)

        # Freeze base model if using LoRA
        if use_lora:
            for name, param in self.model.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False

    def _apply_lora(self, rank: int, alpha: int):
        """Apply LoRA adapters to attention layers."""
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=rank,
            lora_alpha=alpha,
            target_modules=["query", "value"],
            lora_dropout=0.05,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info(f"Applied LoRA (rank={rank}, alpha={alpha}) to VideoMAE")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract embeddings from video frames.

        Args:
            pixel_values: (B, T, C, H, W) tensor of video frames

        Returns:
            (B, embedding_dim) tensor of clip embeddings
        """
        # VideoMAE expects (B, T, C, H, W)
        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)

        # Use penultimate layer's [CLS] token (first token)
        penultimate = outputs.hidden_states[-2]  # (B, num_patches+1, hidden_dim)
        cls_embedding = penultimate[:, 0, :]  # (B, hidden_dim)

        return cls_embedding

    def preprocess(self, frames: torch.Tensor) -> torch.Tensor:
        """Apply VideoMAE preprocessing to raw frames.

        Args:
            frames: (B, T, C, H, W) in [0, 1]

        Returns:
            Preprocessed tensor ready for forward pass
        """
        # VideoMAE processor expects list of numpy frames
        # But in batch mode we handle normalization ourselves
        mean = torch.tensor(self.processor.image_mean).view(1, 1, 3, 1, 1).to(frames.device)
        std = torch.tensor(self.processor.image_std).view(1, 1, 3, 1, 1).to(frames.device)
        return (frames - mean) / std


def extract_embeddings(
    clip_paths: list[str],
    cfg: dict,
    device: str = "cuda",
    output_path: str | None = None,
) -> torch.Tensor:
    """Extract embeddings for a list of video clips.

    Args:
        clip_paths: Paths to video clip files.
        cfg: Full config dict.
        device: Device to use.
        output_path: If provided, save embeddings to this path.

    Returns:
        (N, embedding_dim) tensor of embeddings.
    """
    enc_cfg = cfg["video_encoder"]

    encoder = SurgicalVideoEncoder(
        model_name=enc_cfg["model_name"],
        use_lora=enc_cfg.get("use_lora", False),
        lora_rank=enc_cfg.get("lora_rank", 16),
        lora_alpha=enc_cfg.get("lora_alpha", 32),
    ).to(device)
    encoder.eval()

    dataset = VideoClipDataset(
        clip_paths=clip_paths,
        num_frames=enc_cfg["num_frames"],
        frame_size=enc_cfg["frame_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=enc_cfg["batch_size"],
        num_workers=cfg["infra"]["num_workers"],
        pin_memory=cfg["infra"]["pin_memory"],
    )

    all_embeddings = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Extracting embeddings"):
            frames = batch["pixel_values"].to(device)
            # Reshape from (B, T, C, H, W) if needed
            if frames.dim() == 4:
                # Single frame case — shouldn't happen but handle gracefully
                frames = frames.unsqueeze(1)

            frames = encoder.preprocess(frames)
            emb = encoder(frames)
            all_embeddings.append(emb.cpu())

    embeddings = torch.cat(all_embeddings, dim=0)
    logger.info(f"Extracted embeddings: {embeddings.shape}")

    if output_path:
        torch.save({
            "embeddings": embeddings,
            "clip_paths": clip_paths,
        }, output_path)
        logger.info(f"Saved embeddings to {output_path}")

    return embeddings
