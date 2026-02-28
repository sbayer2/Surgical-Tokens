"""Reasoning model as process reward judge.

Uses Claude or GPT-4 with image understanding to evaluate
assembled procedure sequences. Scores intermediate states
and final outcomes as a process reward model.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are an expert surgical procedure evaluator. You assess surgical video sequences
for quality across multiple dimensions. You evaluate based on:

1. **Anatomical Safety**: Proper identification and preservation of critical structures
2. **Instrument Handling**: Appropriate instrument selection, grip, and movement
3. **Tissue Respect**: Gentle tissue manipulation, appropriate traction/counter-traction
4. **Procedural Flow**: Logical progression of surgical steps, minimal backtracking
5. **Efficiency**: Purposeful movements, minimal wasted motion, appropriate pace

For each criterion, provide a score from 1-5:
  1 = Poor / Unsafe
  2 = Below average
  3 = Competent
  4 = Above average
  5 = Expert level

Respond in JSON format with scores and brief justifications."""

JUDGE_EVALUATION_PROMPT = """Evaluate this surgical procedure sequence. You are seeing {n_clips} clips
from a {procedure_type} procedure, represented by key frames extracted at regular intervals.

For each clip, assess the surgical quality based on the visible anatomy, instrument positioning,
tissue handling, and procedural context.

Provide:
1. Per-clip scores (1-5) for each criterion
2. An overall procedure score (1-5) for each criterion
3. A composite quality score (1-5)
4. Brief narrative assessment

Respond as JSON:
{{
  "clip_scores": [
    {{"clip_index": 0, "anatomical_safety": X, "instrument_handling": X, "tissue_respect": X, "procedural_flow": X, "efficiency": X}},
    ...
  ],
  "overall_scores": {{
    "anatomical_safety": X,
    "instrument_handling": X,
    "tissue_respect": X,
    "procedural_flow": X,
    "efficiency": X
  }},
  "composite_score": X,
  "assessment": "brief narrative"
}}"""


@dataclass
class JudgeScore:
    """Structured output from the judge model."""

    clip_scores: list[dict]
    overall_scores: dict[str, float]
    composite_score: float
    assessment: str
    raw_response: str


def extract_keyframes(
    clip_path: str | Path,
    n_frames: int = 3,
) -> list[np.ndarray]:
    """Extract evenly-spaced key frames from a video clip.

    Returns list of BGR numpy arrays.
    """
    cap = cv2.VideoCapture(str(clip_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def frame_to_base64(frame: np.ndarray, max_size: int = 512) -> str:
    """Convert a frame to base64-encoded JPEG for API submission."""
    h, w = frame.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def judge_procedure_anthropic(
    clip_paths: list[str | Path],
    procedure_type: str,
    cfg: dict,
) -> JudgeScore:
    """Evaluate a procedure using Claude as the judge.

    Args:
        clip_paths: Ordered list of clip video paths.
        procedure_type: Name of the surgical procedure.
        cfg: Full config dict.

    Returns:
        JudgeScore with structured evaluation.
    """
    import anthropic

    judge_cfg = cfg["judge"]
    client = anthropic.Anthropic()

    # Extract key frames from each clip
    content_blocks = []
    for i, clip_path in enumerate(clip_paths):
        frames = extract_keyframes(clip_path, n_frames=judge_cfg["keyframes_per_clip"])
        for j, frame in enumerate(frames):
            b64 = frame_to_base64(frame)
            content_blocks.append({
                "type": "text",
                "text": f"[Clip {i+1}/{len(clip_paths)}, Frame {j+1}]",
            })
            content_blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": b64,
                },
            })

    # Add evaluation prompt
    content_blocks.append({
        "type": "text",
        "text": JUDGE_EVALUATION_PROMPT.format(
            n_clips=len(clip_paths),
            procedure_type=procedure_type,
        ),
    })

    response = client.messages.create(
        model=judge_cfg["model"],
        max_tokens=judge_cfg["max_tokens"],
        temperature=judge_cfg["temperature"],
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content_blocks}],
    )

    raw = response.content[0].text

    # Parse JSON response
    try:
        # Find JSON in response
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])

        return JudgeScore(
            clip_scores=parsed.get("clip_scores", []),
            overall_scores=parsed.get("overall_scores", {}),
            composite_score=float(parsed.get("composite_score", 0)),
            assessment=parsed.get("assessment", ""),
            raw_response=raw,
        )
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Failed to parse judge response: {e}")
        return JudgeScore(
            clip_scores=[],
            overall_scores={},
            composite_score=0.0,
            assessment=raw,
            raw_response=raw,
        )


def compute_process_rewards(
    clip_paths: list[str | Path],
    procedure_type: str,
    cfg: dict,
) -> list[float]:
    """Compute per-step process rewards using the judge model.

    Evaluates progressively longer prefixes of the procedure to get
    intermediate rewards (process reward model pattern).

    Args:
        clip_paths: Ordered list of clip video paths.
        procedure_type: Name of the surgical procedure.
        cfg: Full config dict.

    Returns:
        List of reward scores, one per clip position.
    """
    rewards = []
    for i in range(1, len(clip_paths) + 1):
        prefix = clip_paths[:i]
        score = judge_procedure_anthropic(prefix, procedure_type, cfg)
        rewards.append(score.composite_score)
        logger.debug(f"Process reward at step {i}: {score.composite_score}")

    return rewards
