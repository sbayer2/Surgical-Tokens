"""Video segmentation: scene detection + fixed-length clip extraction.

Pipeline:
  1. Run PySceneDetect to find meaningful scene boundaries
  2. Segment into fixed-length clips (default 2 min) with overlap (default 30s)
  3. Write clips to disk with metadata
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClipMetadata:
    """Metadata for a single extracted clip."""

    video_id: str
    clip_index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    scene_boundaries: list[float]  # Scene cuts within this clip (relative to clip start)
    output_path: str


def detect_scenes(
    video_path: str | Path,
    threshold: float = 27.0,
    min_scene_len_sec: float = 5.0,
    fps_sample: int = 2,
) -> list[float]:
    """Detect scene boundaries using PySceneDetect's ContentDetector.

    Returns list of timestamps (seconds) where scene changes occur.
    """
    from scenedetect import ContentDetector, open_video, SceneManager

    video = open_video(str(video_path))
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=threshold,
            min_scene_len=int(min_scene_len_sec * fps_sample),
        )
    )

    # Downsampled detection pass
    scene_manager.detect_scenes(video, frame_skip=max(1, int(video.frame_rate / fps_sample) - 1))
    scene_list = scene_manager.get_scene_list()

    boundaries = [scene[0].get_seconds() for scene in scene_list]
    logger.info(f"Detected {len(boundaries)} scene boundaries in {video_path}")
    return boundaries


def get_video_duration(video_path: str | Path) -> float:
    """Get video duration in seconds using ffprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def compute_clip_windows(
    total_duration: float,
    clip_duration: float = 120.0,
    overlap: float = 30.0,
) -> list[tuple[float, float]]:
    """Compute (start, end) windows for fixed-length clips with overlap."""
    stride = clip_duration - overlap
    windows = []
    start = 0.0
    while start + clip_duration <= total_duration:
        windows.append((start, start + clip_duration))
        start += stride
    # Handle final partial clip — include if > 50% of target duration
    if start < total_duration and (total_duration - start) > clip_duration * 0.5:
        windows.append((start, total_duration))
    return windows


def extract_clip_ffmpeg(
    video_path: str | Path,
    start_sec: float,
    end_sec: float,
    output_path: str | Path,
) -> None:
    """Extract a clip using ffmpeg with stream copy (fast, no re-encode)."""
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-to", str(end_sec),
            "-i", str(video_path),
            "-c", "copy",
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ],
        capture_output=True,
        check=True,
    )


def segment_video(
    video_path: str | Path,
    output_dir: str | Path,
    clip_duration_sec: float = 120.0,
    overlap_sec: float = 30.0,
    scene_threshold: float = 27.0,
    min_scene_duration_sec: float = 5.0,
    fps_sample: int = 2,
) -> list[ClipMetadata]:
    """Full segmentation pipeline for a single video.

    Returns:
        List of ClipMetadata for each extracted clip.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_id = video_path.stem
    duration = get_video_duration(video_path)
    logger.info(f"Video {video_id}: {duration:.1f}s")

    # Detect scenes
    scene_boundaries = detect_scenes(
        video_path,
        threshold=scene_threshold,
        min_scene_len_sec=min_scene_duration_sec,
        fps_sample=fps_sample,
    )

    # Compute clip windows
    windows = compute_clip_windows(duration, clip_duration_sec, overlap_sec)
    logger.info(f"Extracting {len(windows)} clips from {video_id}")

    clips = []
    for i, (start, end) in enumerate(windows):
        clip_filename = f"{video_id}_clip{i:04d}.mp4"
        clip_path = output_dir / clip_filename

        # Find scene boundaries within this clip window
        local_boundaries = [
            b - start for b in scene_boundaries if start <= b < end
        ]

        extract_clip_ffmpeg(video_path, start, end, clip_path)

        meta = ClipMetadata(
            video_id=video_id,
            clip_index=i,
            start_sec=start,
            end_sec=end,
            duration_sec=end - start,
            scene_boundaries=local_boundaries,
            output_path=str(clip_path),
        )
        clips.append(meta)

    # Write metadata
    meta_path = output_dir / f"{video_id}_clips.json"
    with open(meta_path, "w") as f:
        json.dump([asdict(c) for c in clips], f, indent=2)

    return clips


def segment_video_batch(
    video_dir: str | Path,
    output_dir: str | Path,
    cfg: dict,
    extensions: tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov"),
) -> list[ClipMetadata]:
    """Segment all videos in a directory."""
    video_dir = Path(video_dir)
    all_clips = []
    videos = sorted(p for p in video_dir.iterdir() if p.suffix.lower() in extensions)
    logger.info(f"Found {len(videos)} videos in {video_dir}")

    seg_cfg = cfg["segmentation"]
    for video_path in videos:
        clips = segment_video(
            video_path=video_path,
            output_dir=Path(output_dir) / video_path.stem,
            clip_duration_sec=seg_cfg["clip_duration_sec"],
            overlap_sec=seg_cfg["overlap_sec"],
            scene_threshold=seg_cfg["scene_threshold"],
            min_scene_duration_sec=seg_cfg["min_scene_duration_sec"],
            fps_sample=seg_cfg["fps_sample"],
        )
        all_clips.extend(clips)

    return all_clips
