"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML config and return as nested dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class Paths:
    """Standard project paths derived from config output_dir."""

    output_dir: Path
    raw_video_dir: Path = field(init=False)
    clips_dir: Path = field(init=False)
    embeddings_dir: Path = field(init=False)
    codebook_dir: Path = field(init=False)
    sequences_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    eval_dir: Path = field(init=False)

    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.raw_video_dir = self.output_dir / "raw_videos"
        self.clips_dir = self.output_dir / "clips"
        self.embeddings_dir = self.output_dir / "embeddings"
        self.codebook_dir = self.output_dir / "codebook"
        self.sequences_dir = self.output_dir / "sequences"
        self.models_dir = self.output_dir / "models"
        self.eval_dir = self.output_dir / "evaluation"

    def ensure_all(self):
        """Create all directories."""
        for f in [
            self.raw_video_dir,
            self.clips_dir,
            self.embeddings_dir,
            self.codebook_dir,
            self.sequences_dir,
            self.models_dir,
            self.eval_dir,
        ]:
            f.mkdir(parents=True, exist_ok=True)


def get_paths(cfg: dict) -> Paths:
    p = Paths(output_dir=cfg["infra"]["output_dir"])
    p.ensure_all()
    return p
