"""Fine-regime check: does 25 fps test video reveal response dynamics that 1 fps missed?

The 25 fps raw video exists ONLY in the 8 test videos. With fixed-16-frame
encoders, three onset variants ending at the same time T, predicting the label
15 s in the FUTURE (T+375 frames), LOGO-CV over the 8 videos:

  - dense  = 16 frames @ 25 fps  (0.6 s burst; captures fast motion)
  - sparse = 16 frames @ 1 fps   (16 s window; S1-style; the "worn-stairs" long span)
  - frame  = 1 frame @ T x16     (single-frame appearance baseline, B3)

x {trained encoders, random-weights encoders (B4)}. Reports, per target:
score for each variant, the trained-minus-random LEARNED component, and the
dense-vs-sparse contrast (the density / worn-stairs question).

NOTE: this necessarily uses the reserved test set (25 fps lives nowhere else);
it answers the density question, distinct from the final P1-P5 confirmation.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import av
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from stage2_response_prediction import logo_predict, score  # noqa: E402

VIDEOMAE_ID = "MCG-NJU/videomae-base"
VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"
GAP = 375         # predict 15 s (375 @ 25 fps) into the future
SPAN = 375        # sparse window span (15 s) preceding T
SIZE = 224
STRIDE_1FPS = 25


def build_encoders(device, random_weights: bool):
    from transformers import (AutoImageProcessor, AutoVideoProcessor,
                              VideoMAEConfig, VideoMAEModel, VJEPA2Config, VJEPA2Model)
    vm_proc = AutoImageProcessor.from_pretrained(VIDEOMAE_ID)
    vj_proc = AutoVideoProcessor.from_pretrained(VJEPA2_ID)
    if random_weights:
        torch.manual_seed(0)
        vm = VideoMAEModel(VideoMAEConfig.from_pretrained(VIDEOMAE_ID))
        vj = VJEPA2Model(VJEPA2Config.from_pretrained(VJEPA2_ID))
    else:
        vm = VideoMAEModel.from_pretrained(VIDEOMAE_ID)
        vj = VJEPA2Model.from_pretrained(VJEPA2_ID)
    return vm_proc, vm.to(device).eval(), vj_proc, vj.to(device).eval()


@torch.no_grad()
def embed(imgs16, vm_proc, vm, vj_proc, vj, device):
    vm_in = vm_proc(imgs16, return_tensors="pt").to(device)
    e_vm = vm(**vm_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
    video = np.stack([np.asarray(f) for f in imgs16])
    vj_in = vj_proc(video, return_tensors="pt").to(device)
    e_vj = vj(**vj_in).last_hidden_state.mean(dim=1)[0].float().cpu().numpy()
    return e_vm, e_vj


def decode_segment(path, start, end):
    """Decode mp4 frames [start, end] (25 fps indices) -> {idx: PIL}. One seek."""
    frames = {}
    container = av.open(path)
    stream = container.streams.video[0]
    tb = stream.time_base
    container.seek(int(start / float(stream.average_rate) / tb), stream=stream)
    for frame in container.decode(video=0):
        i = frame.pts * tb * float(stream.average_rate)
        i = int(round(i))
        if i < start:
            continue
        if i > end:
            break
        frames[i] = frame.to_image().convert("RGB").resize((SIZE, SIZE))
    container.close()
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-video", type=int, default=30)
    args = ap.parse_args()

    root = os.path.expanduser(args.data)
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cache = os.path.join(out, "fine_regime_emb.npz")

    if not os.path.exists(cache):
        print(f"[fine] device={device}; loading trained + random encoders ...", flush=True)
        T = build_encoders(device, False)
        R = build_encoders(device, True)
        cols = {f"{w}_{v}": [] for w in ("t", "r") for v in ("dense", "sparse", "frame")}
        meta = {"video": [], "bleeding": [], "phase": []}
        for vd in sorted(glob.glob(f"{root}/Testing/VID*")):
            vid = os.path.basename(vd)
            ann = json.load(open(os.path.join(vd, f"{vid}.json")))["annotations"]
            keys = sorted(int(k) for k in ann)
            usable = [T0 for T0 in keys
                      if (T0 - SPAN) >= keys[0] and (T0 + GAP) in set(keys)]
            picks = usable[::max(1, len(usable) // args.per_video)][:args.per_video]
            mp4 = os.path.join(vd, f"{vid}.mp4")
            print(f"[fine] {vid}: {len(picks)} anchors", flush=True)
            for T0 in picks:
                seg = decode_segment(mp4, T0 - SPAN, T0)
                if T0 not in seg:
                    continue
                dense = [seg[i] for i in range(T0 - 15, T0 + 1) if i in seg]
                sparse = [seg[i] for i in range(T0 - SPAN, T0 + 1, STRIDE_1FPS) if i in seg]
                if len(dense) < 16 or len(sparse) < 16:
                    continue
                dense, sparse = dense[-16:], sparse[-16:]
                frame16 = [seg[T0]] * 16
                for tag, enc in (("t", T), ("r", R)):
                    for v, imgs in (("dense", dense), ("sparse", sparse), ("frame", frame16)):
                        e_vm, e_vj = embed(imgs, *enc, device)
                        cols[f"{tag}_{v}"].append(np.concatenate([e_vm, e_vj]))
                tgt = ann[str(T0 + GAP)]
                meta["video"].append(vid)
                meta["bleeding"].append(int(any(o.get("bleeding", 0) for o in tgt)))
                meta["phase"].append(int(tgt[0].get("phase", -1)))
        np.savez(cache, **{k: np.stack(v) for k, v in cols.items()},
                 video=np.array(meta["video"]), bleeding=np.array(meta["bleeding"]),
                 phase=np.array(meta["phase"]))
        print(f"[fine] wrote {cache} ({len(meta['video'])} anchors)", flush=True)

    D = np.load(cache, allow_pickle=True)
    groups = D["video"]
    print(f"\n[fine] {len(groups)} anchors, {len(set(groups))} test videos, LOGO-CV, "
          "predicting label 15 s ahead")
    for tname, y, kind in (("bleeding", D["bleeding"], "cls"),
                           ("phase", D["phase"], "cls")):
        print(f"\n=== {tname} (15 s future) ===")
        print(f"{'weights':8s} {'dense':>7s} {'sparse':>7s} {'frame':>7s}  "
              f"{'dense-sparse':>12s}")
        sc = {}
        for w, tag in (("trained", "t"), ("random", "r")):
            s = {v: score(y, logo_predict(D[f"{tag}_{v}"], y, groups, kind), kind)
                 for v in ("dense", "sparse", "frame")}
            sc[w] = s
            print(f"{w:8s} {s['dense']:7.3f} {s['sparse']:7.3f} {s['frame']:7.3f}  "
                  f"{s['dense'] - s['sparse']:+12.3f}")
        print(f"LEARNED (trained-random): dense {sc['trained']['dense']-sc['random']['dense']:+.3f}  "
              f"sparse {sc['trained']['sparse']-sc['random']['sparse']:+.3f}")


if __name__ == "__main__":
    main()
