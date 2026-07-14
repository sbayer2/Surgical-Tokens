"""Stage 2: the response-prediction assay (P1-P5), leave-one-video-out CV.

Question: does the ONSET embedding carry consequence essence (what happens
next) or only appearance? For each response target we compare, under
leave-one-video-out CV over the 10 training videos (video-level splits = P5):

  predictors:  {VideoMAE, V-JEPA2} x {temporal onset, single-frame (B3)}
  baseline:    predict-mean / majority (B1); permutation null (B2)
  P2 delta:    temporal_score - single_frame_score  (the dynamics content)
  P1:          is V-JEPA2's delta > VideoMAE's delta?

Targets (registered structure):
  - bleeding_resp   (binary, ~50/50): a CONSEQUENCE; dynamics should help
  - tool_pos_resp   (cx,cy regression): the TRAJECTORY; dynamics should help
  - phase_resp      (7-class): an APPEARANCE-dominated CONTROL; expect
                    temporal ~= single-frame (low delta) -- if trajectory/
                    bleeding show a delta and phase does NOT, that contrast
                    is the essence signal, not a probe artifact.

Reserved: fixed Validation + fine-regime Test set (final confirmation only).
"""

from __future__ import annotations

import argparse
import csv
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

ENCODERS = {"VideoMAE": ("vm_onset", "vm_frame"), "V-JEPA2": ("vj_onset", "vj_frame")}


def load(emb_path, csv_path):
    E = np.load(emb_path)
    rows = list(csv.DictReader(open(csv_path)))
    return E, rows


def logo_predict(X, y, groups, kind):
    """Pooled out-of-fold predictions under leave-one-video-out CV."""
    logo = LeaveOneGroupOut()
    pred = np.zeros_like(y, dtype=float) if kind == "reg" else np.zeros(len(y))
    if kind == "reg" and y.ndim == 2:
        pred = np.zeros_like(y, dtype=float)
    for tr, te in logo.split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        if kind == "reg":
            m = Ridge(alpha=10.0).fit(Xtr, y[tr])
            pred[te] = m.predict(Xte)
        else:
            # a fold's train slice can be single-class; fall back to that class.
            if len(np.unique(y[tr])) < 2:
                pred[te] = y[tr][0]
            else:
                m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y[tr])
                pred[te] = m.predict(Xte)
    return pred


def score(y, pred, kind):
    if kind == "reg":
        return r2_score(y, pred)  # >0 beats predict-mean
    return balanced_accuracy_score(y, pred)  # 0.5 = chance for balanced binary


def perm_pvalue(X, y, groups, kind, observed, n=100, seed=0):
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n):
        yp = y.copy()
        # permute labels WITHIN the group structure preserved (shuffle all rows)
        idx = rng.permutation(len(y))
        s = score(y, logo_predict(X, yp[idx], groups, kind), kind)
        ge += (s >= observed)
    return (ge + 1) / (n + 1)


def run_target(name, E, rows, y, kind, mask=None, n_perm=100):
    groups = np.array([r["video_id"] for r in rows])
    if mask is not None:
        groups = groups[mask]
    print(f"\n=== {name} ({kind}, n={len(y)}, "
          f"{len(np.unique(groups))} videos) ===")
    print(f"{'encoder':9s} {'temporal':>9s} {'1-frame':>9s} {'delta':>7s} "
          f"{'p(temp)':>8s}")
    deltas = {}
    for enc, (onk, frk) in ENCODERS.items():
        Xo = E[onk][mask] if mask is not None else E[onk]
        Xf = E[frk][mask] if mask is not None else E[frk]
        so = score(y, logo_predict(Xo, y, groups, kind), kind)
        sf = score(y, logo_predict(Xf, y, groups, kind), kind)
        p = perm_pvalue(Xo, y, groups, kind, so, n=n_perm)
        deltas[enc] = so - sf
        print(f"{enc:9s} {so:9.3f} {sf:9.3f} {so - sf:+7.3f} {p:8.3f}")
    print(f"P1 (V-JEPA2 delta {deltas['V-JEPA2']:+.3f} vs VideoMAE "
          f"{deltas['VideoMAE']:+.3f}): "
          f"{'V-JEPA2 larger' if deltas['V-JEPA2'] > deltas['VideoMAE'] else 'VideoMAE larger/equal'}")
    return deltas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--split", default="Training")
    ap.add_argument("--n-perm", type=int, default=100)
    args = ap.parse_args()

    d = os.path.expanduser(args.emb_dir)
    E, rows = load(os.path.join(d, f"emb_{args.split}.npz"),
                   os.path.join(d, f"clips_{args.split}.csv"))
    print(f"[stage2] {len(rows)} clips, LOGO-CV over "
          f"{len(set(r['video_id'] for r in rows))} videos, "
          f"{args.n_perm} permutations")

    # CONSEQUENCE: bleeding in the response half (binary, balanced)
    yb = np.array([int(r["bleeding_resp"]) for r in rows])
    run_target("bleeding_resp [CONSEQUENCE]", E, rows, yb, "cls", n_perm=args.n_perm)

    # TRAJECTORY: response tool centroid (regression), tool present both halves
    mask = np.array([float(r["tool_cx_onset"]) >= 0 and float(r["tool_cx_resp"]) >= 0
                     for r in rows])
    yt = np.array([[float(r["tool_cx_resp"]), float(r["tool_cy_resp"])]
                   for r, m in zip(rows, mask) if m])
    run_target("tool_pos_resp [TRAJECTORY]", E, rows, yt, "reg", mask=mask,
               n_perm=args.n_perm)

    # CONTROL: phase in the response half (appearance-dominated; expect ~0 delta)
    yp = np.array([int(r["phase_resp"]) for r in rows])
    run_target("phase_resp [APPEARANCE CONTROL]", E, rows, yp, "cls",
               n_perm=args.n_perm)


if __name__ == "__main__":
    main()
