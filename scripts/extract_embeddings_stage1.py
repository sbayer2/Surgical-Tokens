"""Stage 1: extract frozen-encoder embeddings for the surgical assay (coarse regime).

For each 32-frame clip (1 fps => 32 s of procedure), embed three segments with
BOTH frozen encoders:
  - onset    = frames[0:16]   (the action-onset input for response prediction)
  - response = frames[16:32]  (the response target)
  - frame    = frames[0] x16  (static clip => B3 single-frame appearance baseline)
Encoders: E-recon VideoMAE-base (768-d), E-pred V-JEPA2 ViT-L (1024-d).

Alongside, record the aligned label targets Stage 2 needs (all from the exact
frame keys, so labels and embeddings never drift):
  phase_onset/response, phase_changed, operator (P5 covariate),
  bleeding_onset/response, and primary-tool bbox centroid at onset/response
  (the trajectory response target).

Writes per split: <out>/emb_<split>.npz (embedding arrays, row-aligned) and
<out>/clips_<split>.csv (metadata + labels). Video-level split discipline is
preserved because each clip carries its video_id.

    python scripts/extract_embeddings_stage1.py \
        --data ~/datasets/cholectrack20 --out ~/datasets/cholectrack20_emb \
        --splits Training Validation
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import time

import numpy as np
import torch
from PIL import Image

VIDEOMAE_ID = "MCG-NJU/videomae-base"
VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
CLIP = 32          # frames per clip (1 fps -> 32 s)
HALF = 16          # onset / response segment length (VideoMAE fixed num_frames)
SIZE = 224


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def frame_key(png: str) -> int:
    return int(os.path.splitext(os.path.basename(png))[0])


def bbox_centroid(objs) -> tuple:
    """Centroid of the first object's clamped tool bbox [x,y,w,h] -> (cx,cy), or
    (-1,-1) if no tool box in this frame."""
    for o in objs:
        bb = o.get("tool_bbox")
        if bb:
            x, y, w, h = (clamp01(bb[0]), clamp01(bb[1]),
                          clamp01(bb[2]), clamp01(bb[3]))
            return (x + w / 2.0, y + h / 2.0)
    return (-1.0, -1.0)


def frame_phase(objs) -> int:
    return objs[0].get("phase", -1) if objs else -1


def frame_bleeding(objs) -> int:
    return int(any(o.get("bleeding", 0) for o in objs))


def frame_operator(objs) -> int:
    return objs[0].get("operator", -1) if objs else -1


def build_clips(vdir: str):
    """Yield (video_id, [32 frame png paths], [32 annotation-object lists]) for
    non-overlapping windows of matched, contiguously-strided annotated frames."""
    vid = os.path.basename(vdir)
    ann = json.load(open(os.path.join(vdir, f"{vid}.json")))["annotations"]
    pngs = {frame_key(p): p for p in glob.glob(os.path.join(vdir, "Frames", "*.png"))}
    keys = sorted(k for k in (int(k) for k in ann) if k in pngs)  # matched only
    # split into runs of constant stride (typically 25), then chunk each run.
    if not keys:
        return
    stride = min((keys[i + 1] - keys[i] for i in range(len(keys) - 1)), default=25)
    run = [keys[0]]
    runs = []
    for k in keys[1:]:
        if k - run[-1] == stride:
            run.append(k)
        else:
            runs.append(run); run = [k]
    runs.append(run)
    for r in runs:
        for i in range(0, len(r) - CLIP + 1, CLIP):
            win = r[i:i + CLIP]
            yield (vid, [pngs[k] for k in win], [ann[str(k)] for k in win])


def load_imgs(paths) -> list:
    return [Image.open(p).convert("RGB").resize((SIZE, SIZE)) for p in paths]


class Encoders:
    def __init__(self, device):
        from transformers import (AutoImageProcessor, AutoVideoProcessor,
                                   VideoMAEModel, VJEPA2Model)
        self.device = device
        self.vm_proc = AutoImageProcessor.from_pretrained(VIDEOMAE_ID)
        self.vm = VideoMAEModel.from_pretrained(VIDEOMAE_ID).to(device).eval()
        self.vj_proc = AutoVideoProcessor.from_pretrained(VJEPA2_ID)
        self.vj = VJEPA2Model.from_pretrained(VJEPA2_ID).to(device).eval()

    @torch.no_grad()
    def embed(self, imgs16) -> tuple:
        vm_in = self.vm_proc(imgs16, return_tensors="pt").to(self.device)
        vm = self.vm(**vm_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
        video = np.stack([np.asarray(f) for f in imgs16])
        vj_in = self.vj_proc(video, return_tensors="pt").to(self.device)
        vj = self.vj(**vj_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
        return vm, vj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["Training", "Validation"])
    args = ap.parse_args()

    root = os.path.expanduser(args.data)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[stage1] device={device}; loading encoders ...", flush=True)
    enc = Encoders(device)

    for split in args.splits:
        vdirs = sorted(glob.glob(f"{root}/{split}/VID*"))
        rows, E = [], {k: [] for k in
                       ("vm_onset", "vm_resp", "vm_frame", "vj_onset", "vj_resp", "vj_frame")}
        t0, n = time.time(), 0
        for vdir in vdirs:
            for vid, paths, objs in build_clips(vdir):
                imgs = load_imgs(paths)
                onset, resp = imgs[:HALF], imgs[HALF:CLIP]
                frame16 = [imgs[0]] * HALF
                vm_o, vj_o = enc.embed(onset)
                vm_r, vj_r = enc.embed(resp)
                vm_f, vj_f = enc.embed(frame16)
                for key, val in (("vm_onset", vm_o), ("vm_resp", vm_r), ("vm_frame", vm_f),
                                 ("vj_onset", vj_o), ("vj_resp", vj_r), ("vj_frame", vj_f)):
                    E[key].append(val)
                cx_o, cy_o = bbox_centroid(objs[0])
                cx_r, cy_r = bbox_centroid(objs[HALF])
                rows.append({
                    "video_id": vid, "frame_start": frame_key(paths[0]),
                    "phase_onset": frame_phase(objs[0]),
                    "phase_resp": frame_phase(objs[HALF]),
                    "phase_changed": int(frame_phase(objs[0]) != frame_phase(objs[HALF])),
                    "operator": frame_operator(objs[0]),
                    "bleeding_onset": frame_bleeding(objs[0]),
                    "bleeding_resp": frame_bleeding(objs[HALF]),
                    "tool_cx_onset": cx_o, "tool_cy_onset": cy_o,
                    "tool_cx_resp": cx_r, "tool_cy_resp": cy_r,
                })
                n += 1
                if n % 50 == 0:
                    print(f"[stage1] {split}: {n} clips ({(time.time()-t0)/n:.2f}s/clip)",
                          flush=True)
        np.savez(os.path.join(out, f"emb_{split}.npz"),
                 **{k: np.stack(v) for k, v in E.items()})
        with open(os.path.join(out, f"clips_{split}.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print(f"[stage1] {split}: {n} clips from {len(vdirs)} videos -> "
              f"emb_{split}.npz + clips_{split}.csv ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
