"""B4: the random-weights-encoder baseline (F6 discipline).

Is V-JEPA2's temporal advantage (S1) a property of the TRAINED model, or just
of the architecture ("16-frame transformer > 1 frame")? Re-embed the clips
with RANDOMLY-INITIALIZED VideoMAE and V-JEPA2 (same configs, random weights),
recompute the temporal-vs-single-frame delta for the targets that showed
signal (bleeding, phase), and compare random-delta vs trained-delta.

If random shows ~0 delta and trained shows the S1 delta, the dynamics-encoding
was LEARNED. If random already shows it, S1's delta is architectural.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from extract_embeddings_stage1 import (CLIP, HALF, build_clips, frame_key,  # noqa: E402
                                       load_imgs)
from stage2_response_prediction import logo_predict, score  # noqa: E402

VIDEOMAE_ID = "MCG-NJU/videomae-base"
VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"


class RandomEncoders:
    """Same architecture as the trained encoders, random weights."""

    def __init__(self, device):
        from transformers import (AutoImageProcessor, AutoVideoProcessor,
                                   VideoMAEConfig, VideoMAEModel, VJEPA2Config,
                                   VJEPA2Model)
        self.device = device
        torch.manual_seed(0)
        self.vm_proc = AutoImageProcessor.from_pretrained(VIDEOMAE_ID)
        self.vm = VideoMAEModel(VideoMAEConfig.from_pretrained(VIDEOMAE_ID)).to(device).eval()
        self.vj_proc = AutoVideoProcessor.from_pretrained(VJEPA2_ID)
        self.vj = VJEPA2Model(VJEPA2Config.from_pretrained(VJEPA2_ID)).to(device).eval()

    @torch.no_grad()
    def embed(self, imgs16):
        vm_in = self.vm_proc(imgs16, return_tensors="pt").to(self.device)
        vm = self.vm(**vm_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
        video = np.stack([np.asarray(f) for f in imgs16])
        vj_in = self.vj_proc(video, return_tensors="pt").to(self.device)
        vj = self.vj(**vj_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
        return vm, vj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--emb-dir", required=True)
    ap.add_argument("--split", default="Training")
    args = ap.parse_args()

    root = os.path.expanduser(args.data)
    emb_dir = os.path.expanduser(args.emb_dir)
    rand_path = os.path.join(emb_dir, f"emb_{args.split}_RANDOM.npz")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    if not os.path.exists(rand_path):
        print(f"[b4] device={device}; building RANDOM encoders + embedding ...", flush=True)
        enc = RandomEncoders(device)
        E = {k: [] for k in ("vm_onset", "vm_frame", "vj_onset", "vj_frame")}
        n = 0
        for vdir in sorted(glob.glob(f"{root}/{args.split}/VID*")):
            for vid, paths, objs in build_clips(vdir):
                imgs = load_imgs(paths)
                vm_o, vj_o = enc.embed(imgs[:HALF])
                vm_f, vj_f = enc.embed([imgs[0]] * HALF)
                for key, val in (("vm_onset", vm_o), ("vm_frame", vm_f),
                                 ("vj_onset", vj_o), ("vj_frame", vj_f)):
                    E[key].append(val)
                n += 1
                if n % 100 == 0:
                    print(f"[b4] {n} clips", flush=True)
        np.savez(rand_path, **{k: np.stack(v) for k, v in E.items()})
        print(f"[b4] wrote {rand_path} ({n} clips)", flush=True)

    R = np.load(rand_path)
    T = np.load(os.path.join(emb_dir, f"emb_{args.split}.npz"))
    rows = list(csv.DictReader(open(os.path.join(emb_dir, f"clips_{args.split}.csv"))))
    groups = np.array([r["video_id"] for r in rows])

    def delta(E, onk, frk, y, kind):
        so = score(y, logo_predict(E[onk], y, groups, kind), kind)
        sf = score(y, logo_predict(E[frk], y, groups, kind), kind)
        return so, sf, so - sf

    print(f"\n[b4] TRAINED vs RANDOM temporal deltas ({len(rows)} clips, LOGO-CV)")
    for name, y, kind in (("bleeding_resp", np.array([int(r["bleeding_resp"]) for r in rows]), "cls"),
                          ("phase_resp", np.array([int(r["phase_resp"]) for r in rows]), "cls")):
        print(f"\n=== {name} ===")
        print(f"{'encoder':9s} {'trained Δ':>10s} {'random Δ':>10s}  verdict")
        for enc, (onk, frk) in (("VideoMAE", ("vm_onset", "vm_frame")),
                                ("V-JEPA2", ("vj_onset", "vj_frame"))):
            _, _, td = delta(T, onk, frk, y, kind)
            _, _, rd = delta(R, onk, frk, y, kind)
            verdict = "LEARNED" if (td - rd) > 0.03 else "architectural/none"
            print(f"{enc:9s} {td:+10.3f} {rd:+10.3f}  {verdict}")


if __name__ == "__main__":
    main()
