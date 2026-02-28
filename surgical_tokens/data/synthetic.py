"""Synthetic data generators for testing the pipeline without real ACS data.

Generates:
  - Fake video clips (solid color frames with random movement patterns)
  - Synthetic case logs with realistic distributions
  - Pre-computed fake embeddings for downstream testing
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

PROCEDURE_TYPES = [
    "laparoscopic_cholecystectomy",
    "laparoscopic_appendectomy",
    "inguinal_hernia_repair",
    "laparoscopic_colectomy",
    "open_appendectomy",
]

SURGICAL_PHASES = [
    "port_placement",
    "exposure",
    "dissection",
    "clipping",
    "division",
    "extraction",
    "hemostasis",
    "closure",
]


def generate_synthetic_video(
    output_path: str | Path,
    duration_sec: float = 120.0,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> None:
    """Generate a synthetic surgical-ish video clip.

    Creates a video with moving colored shapes on a dark red background
    (simulating abdominal cavity with instrument movement).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    rng = np.random.default_rng(hash(str(output_path)) % (2**31))
    n_frames = int(duration_sec * fps)

    # Background: dark red-brown (abdominal cavity color)
    bg_color = np.array([40, 20, 60], dtype=np.uint8)  # BGR

    # Instrument positions (two instruments)
    instr1_pos = np.array([width * 0.3, height * 0.5])
    instr2_pos = np.array([width * 0.7, height * 0.5])

    for i in range(n_frames):
        frame = np.full((height, width, 3), bg_color, dtype=np.uint8)

        # Add some tissue-like texture noise
        noise = rng.integers(0, 20, (height, width, 3), dtype=np.uint8)
        frame = cv2.add(frame, noise)

        # Move instruments with sinusoidal + random motion
        t = i / fps
        dx1 = 50 * np.sin(t * 0.5) + rng.normal(0, 3)
        dy1 = 30 * np.cos(t * 0.3) + rng.normal(0, 3)
        dx2 = 40 * np.cos(t * 0.4) + rng.normal(0, 3)
        dy2 = 35 * np.sin(t * 0.6) + rng.normal(0, 3)

        p1 = (int(instr1_pos[0] + dx1), int(instr1_pos[1] + dy1))
        p2 = (int(instr2_pos[0] + dx2), int(instr2_pos[1] + dy2))

        # Draw instruments as lines
        cv2.line(frame, (0, height), p1, (180, 180, 180), 3)
        cv2.line(frame, (width, height), p2, (180, 180, 180), 3)

        # Instrument tips
        cv2.circle(frame, p1, 8, (200, 200, 200), -1)
        cv2.circle(frame, p2, 8, (200, 200, 200), -1)

        writer.write(frame)

    writer.release()


def generate_synthetic_case_log(
    n_cases: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic case log with realistic distributions."""
    rng = np.random.default_rng(seed)

    procedures = rng.choice(PROCEDURE_TYPES, size=n_cases)

    # Base operative times by procedure (minutes)
    base_times = {
        "laparoscopic_cholecystectomy": (45, 15),
        "laparoscopic_appendectomy": (35, 10),
        "inguinal_hernia_repair": (55, 20),
        "laparoscopic_colectomy": (120, 35),
        "open_appendectomy": (40, 12),
    }

    rows = []
    for i in range(n_cases):
        proc = procedures[i]
        mu, sigma = base_times[proc]
        pgy = rng.integers(1, 8)

        # More experienced surgeons tend to be faster with less blood loss
        experience_factor = 1.0 + 0.15 * max(0, 4 - pgy)

        op_time = max(15, rng.normal(mu * experience_factor, sigma))
        ebl = max(0, rng.normal(50 * experience_factor, 30))
        converted = 1 if rng.random() < 0.05 * experience_factor else 0
        complications = 1 if rng.random() < 0.08 * experience_factor else 0

        rows.append({
            "video_id": f"case_{i:04d}",
            "procedure_type": proc,
            "operative_time_min": round(op_time, 1),
            "estimated_blood_loss_ml": round(ebl, 1),
            "converted_to_open": converted,
            "complications_30d": complications,
            "surgeon_pgy": pgy,
            "case_volume": rng.integers(5, 500),
        })

    return pd.DataFrame(rows)


def generate_synthetic_embeddings(
    n_clips: int = 500,
    embedding_dim: int = 768,
    n_phases: int = 8,
    seed: int = 42,
) -> tuple[torch.Tensor, np.ndarray, list[str]]:
    """Generate synthetic clip embeddings that cluster by surgical phase.

    Returns:
        embeddings: (n_clips, embedding_dim) tensor
        phase_labels: (n_clips,) array of phase indices (for validation)
        clip_ids: list of clip identifier strings
    """
    rng = np.random.default_rng(seed)

    # Create phase centroids in embedding space
    centroids = rng.standard_normal((n_phases, embedding_dim)).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True) * 5.0

    # Assign clips to phases
    phase_labels = rng.integers(0, n_phases, size=n_clips)

    # Generate embeddings around centroids
    embeddings = np.zeros((n_clips, embedding_dim), dtype=np.float32)
    for i in range(n_clips):
        phase = phase_labels[i]
        embeddings[i] = centroids[phase] + rng.standard_normal(embedding_dim).astype(np.float32) * 0.5

    clip_ids = [f"clip_{i:04d}" for i in range(n_clips)]

    return torch.from_numpy(embeddings), phase_labels, clip_ids


def generate_synthetic_procedures(
    n_procedures: int = 50,
    min_clips: int = 8,
    max_clips: int = 40,
    codebook_size: int = 2048,
    seed: int = 42,
) -> list[dict]:
    """Generate synthetic procedure token sequences for sequence model testing.

    Each procedure is a sequence of codebook indices with associated metadata.
    """
    rng = np.random.default_rng(seed)
    procedures = []

    for i in range(n_procedures):
        n_clips = rng.integers(min_clips, max_clips + 1)

        # Token sequence — surgical procedures have structure, so we bias
        # toward sequential token ranges (phases progress in order)
        phase_progression = np.sort(rng.integers(0, codebook_size, size=n_clips))
        # Add noise to break perfect ordering
        noise = rng.integers(-50, 50, size=n_clips)
        tokens = [int(t) for t in np.clip(phase_progression + noise, 3, codebook_size - 1)]

        procedures.append({
            "video_id": f"case_{i:04d}",
            "tokens": tokens,
            "n_clips": int(n_clips),
        })

    return procedures


def generate_full_synthetic_dataset(
    output_dir: str | Path,
    n_cases: int = 50,
    n_clips_per_case: int = 10,
    embedding_dim: int = 768,
    generate_videos: bool = False,
) -> dict:
    """Generate a complete synthetic dataset for end-to-end pipeline testing.

    Args:
        output_dir: Where to write all synthetic data.
        n_cases: Number of surgical cases.
        n_clips_per_case: Average clips per case.
        embedding_dim: Dimension of clip embeddings.
        generate_videos: If True, actually create video files (slow).

    Returns:
        Dict with paths to all generated artifacts.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Case log
    case_log = generate_synthetic_case_log(n_cases=n_cases)
    case_log_path = output_dir / "case_log.csv"
    case_log.to_csv(case_log_path, index=False)
    logger.info(f"Generated case log: {case_log_path}")

    # 2. Embeddings
    n_clips = n_cases * n_clips_per_case
    embeddings, phase_labels, clip_ids = generate_synthetic_embeddings(
        n_clips=n_clips,
        embedding_dim=embedding_dim,
    )
    emb_path = output_dir / "embeddings.pt"
    torch.save({
        "embeddings": embeddings,
        "phase_labels": torch.from_numpy(phase_labels.astype(np.int64)),
        "clip_ids": clip_ids,
    }, emb_path)
    logger.info(f"Generated embeddings: {emb_path}")

    # 3. Clip-to-case mapping
    rng = np.random.default_rng(42)
    clip_mapping = []
    clip_idx = 0
    for _, row in case_log.iterrows():
        n = rng.integers(max(1, n_clips_per_case - 3), n_clips_per_case + 4)
        for j in range(min(n, n_clips - clip_idx)):
            clip_mapping.append({
                "clip_id": clip_ids[clip_idx],
                "video_id": row["video_id"],
                "clip_index": j,
            })
            clip_idx += 1
            if clip_idx >= n_clips:
                break
        if clip_idx >= n_clips:
            break

    mapping_path = output_dir / "clip_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(clip_mapping, f, indent=2)

    # 4. Synthetic videos (optional, slow)
    if generate_videos:
        video_dir = output_dir / "videos"
        for _, row in case_log.iterrows():
            generate_synthetic_video(
                video_dir / f"{row['video_id']}.mp4",
                duration_sec=120.0,
            )
        logger.info(f"Generated {n_cases} synthetic videos")

    # 5. Procedure token sequences
    procedures = generate_synthetic_procedures(n_procedures=n_cases)
    proc_path = output_dir / "procedures.json"
    with open(proc_path, "w") as f:
        json.dump(procedures, f, indent=2)

    return {
        "case_log": str(case_log_path),
        "embeddings": str(emb_path),
        "clip_mapping": str(mapping_path),
        "procedures": str(proc_path),
    }
