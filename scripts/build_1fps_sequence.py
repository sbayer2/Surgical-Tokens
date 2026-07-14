"""Build a per-timestep frozen-embedding sequence for the horizon sweep.

For each training video, embed 1 fps frames (already on disk as PNGs; no
decode) with V-JEPA2 single-frame, subsampled to a STEP-second grid, plus the
per-timestep labels (bleeding, phase). Batched forwards for speed. Output:
one <out>/seq_<VID>.npz per video with emb (T, 1024), bleeding (T,), phase (T,),
frame_key (T,). The horizon sweep forms (context, future-target) pairs from
these sequences.
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image

VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
SIZE = 224


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="Training")
    ap.add_argument("--step", type=int, default=2, help="seconds between grid points")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoVideoProcessor, VJEPA2Model
    root, out = os.path.expanduser(args.data), os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    proc = AutoVideoProcessor.from_pretrained(VJEPA2_ID)
    model = VJEPA2Model.from_pretrained(VJEPA2_ID).to(device).eval()
    print(f"[seq] device={device}, step={args.step}s", flush=True)

    @torch.no_grad()
    def embed_frames(imgs):
        # each frame -> a static 16-frame clip; batch across frames.
        out_emb = []
        for i in range(0, len(imgs), args.batch):
            batch = imgs[i:i + args.batch]
            vids = np.stack([np.stack([np.asarray(f)] * 16) for f in batch])  # (B,16,H,W,C)
            inp = proc(list(vids), return_tensors="pt").to(device)
            e = model(**inp).last_hidden_state.mean(dim=1).float().cpu().numpy()
            out_emb.append(e)
        return np.concatenate(out_emb, 0)

    for vd in sorted(glob.glob(f"{root}/{args.split}/VID*")):
        vid = os.path.basename(vd)
        ann = json.load(open(os.path.join(vd, f"{vid}.json")))["annotations"]
        pngs = {int(os.path.splitext(os.path.basename(p))[0]): p
                for p in glob.glob(os.path.join(vd, "Frames", "*.png"))}
        keys = sorted(k for k in (int(k) for k in ann) if k in pngs)
        # STEP-second grid (annotations are 1 fps at stride 25).
        grid = keys[::args.step]
        imgs = [Image.open(pngs[k]).convert("RGB").resize((SIZE, SIZE)) for k in grid]
        emb = embed_frames(imgs)
        bleeding = np.array([int(any(o.get("bleeding", 0) for o in ann[str(k)]))
                             for k in grid])
        phase = np.array([ann[str(k)][0].get("phase", -1) for k in grid])
        np.savez(os.path.join(out, f"seq_{vid}.npz"), emb=emb, bleeding=bleeding,
                 phase=phase, frame_key=np.array(grid))
        print(f"[seq] {vid}: {len(grid)} timesteps -> seq_{vid}.npz", flush=True)
    print("[seq] done", flush=True)


if __name__ == "__main__":
    main()
