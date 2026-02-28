"""End-to-end pipeline: raw videos → segmented clips → quality filter → embeddings."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from surgical_tokens.config import get_paths, load_config
from surgical_tokens.data.quality_filter import filter_clips
from surgical_tokens.data.segmentation import segment_video_batch
from surgical_tokens.encoding.video_encoder import extract_embeddings

logger = logging.getLogger(__name__)


def run_extraction_pipeline(
    video_dir: str,
    config_path: str = "configs/default.yaml",
) -> dict:
    """Run the full extraction pipeline.

    Steps:
        1. Segment videos into clips
        2. Filter clips for quality
        3. Extract VideoMAE embeddings

    Returns:
        Dict with paths to all outputs.
    """
    cfg = load_config(config_path)
    paths = get_paths(cfg)

    # 1. Segment
    logger.info("=== Stage 1: Video Segmentation ===")
    clips = segment_video_batch(
        video_dir=video_dir,
        output_dir=paths.clips_dir,
        cfg=cfg,
    )
    clip_paths = [c.output_path for c in clips]
    logger.info(f"Segmented {len(clip_paths)} clips from videos")

    # 2. Quality filter
    logger.info("=== Stage 2: Quality Filtering ===")
    passed_paths, scores = filter_clips(clip_paths, cfg)
    logger.info(f"Quality filter: {len(passed_paths)}/{len(clip_paths)} passed")

    # Save filter results
    filter_results = paths.clips_dir / "quality_scores.json"
    with open(filter_results, "w") as f:
        json.dump(
            [{"path": s.clip_path, "passed": s.passed, "reason": s.rejection_reason}
             for s in scores],
            f, indent=2,
        )

    # 3. Extract embeddings
    logger.info("=== Stage 3: Embedding Extraction ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    emb_path = str(paths.embeddings_dir / "clip_embeddings.pt")
    embeddings = extract_embeddings(
        clip_paths=passed_paths,
        cfg=cfg,
        device=device,
        output_path=emb_path,
    )

    return {
        "n_clips_total": len(clip_paths),
        "n_clips_passed": len(passed_paths),
        "embeddings_shape": list(embeddings.shape),
        "embeddings_path": emb_path,
    }
