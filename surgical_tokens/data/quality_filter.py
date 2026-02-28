"""Quality filtering for surgical video clips.

Drops clips that are:
  - Out of body (instrument insertion/removal) — low entropy, low action
  - Fogged/obscured lens — low entropy, abnormal brightness
  - Pure irrigation/suction with no meaningful action — low optical flow
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityScore:
    """Quality assessment for a single clip."""

    clip_path: str
    mean_brightness: float
    mean_entropy: float
    action_ratio: float  # Fraction of frames with meaningful optical flow
    passed: bool
    rejection_reason: str | None = None


def frame_entropy(frame_gray: np.ndarray) -> float:
    """Compute Shannon entropy of a grayscale frame."""
    hist = cv2.calcHist([frame_gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))


def compute_optical_flow_magnitude(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute mean optical flow magnitude between two frames."""
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))


def assess_clip_quality(
    clip_path: str | Path,
    min_brightness: float = 30.0,
    max_brightness: float = 240.0,
    min_entropy: float = 4.0,
    min_action_ratio: float = 0.3,
    flow_threshold: float = 2.0,
    sample_fps: float = 1.0,
) -> QualityScore:
    """Assess quality of a video clip by sampling frames.

    Args:
        clip_path: Path to video clip.
        min_brightness: Reject if mean brightness below this.
        max_brightness: Reject if mean brightness above this (whiteout).
        min_entropy: Reject if mean frame entropy below this.
        min_action_ratio: Reject if fraction of frames with flow > threshold is below this.
        flow_threshold: Optical flow magnitude threshold for "action".
        sample_fps: FPS at which to sample frames for assessment.
    """
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        return QualityScore(
            clip_path=str(clip_path),
            mean_brightness=0, mean_entropy=0, action_ratio=0,
            passed=False, rejection_reason="cannot_open",
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, int(fps / sample_fps))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    brightnesses = []
    entropies = []
    flow_above = 0
    flow_total = 0
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightnesses.append(float(np.mean(gray)))
            entropies.append(frame_entropy(gray))

            if prev_gray is not None:
                mag = compute_optical_flow_magnitude(prev_gray, gray)
                flow_total += 1
                if mag > flow_threshold:
                    flow_above += 1

            prev_gray = gray

        frame_idx += 1

    cap.release()

    if not brightnesses:
        return QualityScore(
            clip_path=str(clip_path),
            mean_brightness=0, mean_entropy=0, action_ratio=0,
            passed=False, rejection_reason="no_frames",
        )

    mean_bright = float(np.mean(brightnesses))
    mean_ent = float(np.mean(entropies))
    action_ratio = flow_above / max(flow_total, 1)

    # Apply filters
    reason = None
    if mean_bright < min_brightness:
        reason = "too_dark"
    elif mean_bright > max_brightness:
        reason = "too_bright"
    elif mean_ent < min_entropy:
        reason = "low_entropy"
    elif action_ratio < min_action_ratio:
        reason = "low_action"

    return QualityScore(
        clip_path=str(clip_path),
        mean_brightness=mean_bright,
        mean_entropy=mean_ent,
        action_ratio=action_ratio,
        passed=reason is None,
        rejection_reason=reason,
    )


def filter_clips(
    clip_paths: list[str | Path],
    cfg: dict,
) -> tuple[list[str], list[QualityScore]]:
    """Filter a batch of clips, returning paths that pass and all scores.

    Returns:
        (passed_paths, all_scores)
    """
    qf_cfg = cfg["quality_filter"]
    passed = []
    scores = []

    for path in clip_paths:
        score = assess_clip_quality(
            clip_path=path,
            min_brightness=qf_cfg["min_brightness"],
            max_brightness=qf_cfg["max_brightness"],
            min_entropy=qf_cfg["min_entropy"],
            min_action_ratio=qf_cfg["min_action_ratio"],
            flow_threshold=qf_cfg["flow_threshold"],
        )
        scores.append(score)
        if score.passed:
            passed.append(str(path))
        else:
            logger.debug(f"Rejected {path}: {score.rejection_reason}")

    n_rejected = len(clip_paths) - len(passed)
    logger.info(f"Quality filter: {len(passed)}/{len(clip_paths)} passed ({n_rejected} rejected)")
    return passed, scores
