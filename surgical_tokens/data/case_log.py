"""Case log and outcome metadata management.

Loads surgical case logs, computes composite outcome scores,
and generates soft binary labels / DPO pairs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class DPOPair:
    """A preference pair for DPO training."""

    chosen_video_id: str      # "better" procedure
    rejected_video_id: str    # "good" procedure
    chosen_score: float
    rejected_score: float


def load_case_log(path: str) -> pd.DataFrame:
    """Load case log CSV with expected columns.

    Expected columns:
        video_id: str — matches video filename stem
        operative_time_min: float
        estimated_blood_loss_ml: float
        converted_to_open: bool (0/1)
        complications_30d: int (count)
        surgeon_pgy: int (post-graduate year)
        case_volume: int (surgeon's total case count for this procedure)
        procedure_type: str (e.g., "cholecystectomy", "appendectomy")
    """
    df = pd.read_csv(path)
    required = [
        "video_id", "operative_time_min", "estimated_blood_loss_ml",
        "converted_to_open", "complications_30d",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Case log missing columns: {missing}")
    return df


def compute_composite_score(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """Compute composite outcome score (lower = better).

    All metrics are oriented so that lower is better, then combined
    with configurable weights.
    """
    outcome_cfg = cfg["outcomes"]
    weights = outcome_cfg["composite_weights"]

    # Copy to avoid mutating
    scores = pd.DataFrame(index=df.index)
    scores["operative_time"] = df["operative_time_min"].astype(float)
    scores["blood_loss"] = df["estimated_blood_loss_ml"].astype(float)
    scores["conversion"] = df["converted_to_open"].astype(float)
    scores["complications_30d"] = df["complications_30d"].astype(float)

    if outcome_cfg.get("normalize", True):
        for col in ["operative_time", "blood_loss", "complications_30d"]:
            mu = scores[col].mean()
            std = scores[col].std()
            if std > 0:
                scores[col] = (scores[col] - mu) / std

    composite = (
        weights["operative_time"] * scores["operative_time"]
        + weights["blood_loss"] * scores["blood_loss"]
        + weights["conversion"] * scores["conversion"]
        + weights["complications_30d"] * scores["complications_30d"]
    )
    return composite


def label_outcomes(
    df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """Add composite score and soft binary label to case log.

    Adds columns:
        composite_score: float (lower = better)
        label: str ("better" or "good")
    """
    df = df.copy()
    df["composite_score"] = compute_composite_score(df, cfg)
    percentile = cfg["outcomes"]["better_percentile"]
    threshold = np.percentile(df["composite_score"], percentile)

    # Lower score = better outcome → top percentile by low score
    df["label"] = np.where(df["composite_score"] <= threshold, "better", "good")

    n_better = (df["label"] == "better").sum()
    logger.info(f"Labeled {n_better} 'better' / {len(df) - n_better} 'good' out of {len(df)}")
    return df


def generate_dpo_pairs(
    df: pd.DataFrame,
    max_pairs: int | None = None,
    same_procedure_only: bool = True,
) -> list[DPOPair]:
    """Generate DPO preference pairs from labeled case log.

    Each pair consists of a "better" procedure and a "good" procedure.
    If same_procedure_only, only pairs within the same procedure_type are generated.
    """
    better = df[df["label"] == "better"]
    good = df[df["label"] == "good"]

    pairs = []

    if same_procedure_only and "procedure_type" in df.columns:
        for proc_type in df["procedure_type"].unique():
            b = better[better["procedure_type"] == proc_type]
            g = good[good["procedure_type"] == proc_type]
            for _, b_row in b.iterrows():
                for _, g_row in g.iterrows():
                    pairs.append(DPOPair(
                        chosen_video_id=b_row["video_id"],
                        rejected_video_id=g_row["video_id"],
                        chosen_score=float(b_row["composite_score"]),
                        rejected_score=float(g_row["composite_score"]),
                    ))
    else:
        for _, b_row in better.iterrows():
            for _, g_row in good.iterrows():
                pairs.append(DPOPair(
                    chosen_video_id=b_row["video_id"],
                    rejected_video_id=g_row["video_id"],
                    chosen_score=float(b_row["composite_score"]),
                    rejected_score=float(g_row["composite_score"]),
                ))

    # Subsample if too many pairs
    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    logger.info(f"Generated {len(pairs)} DPO pairs")
    return pairs
