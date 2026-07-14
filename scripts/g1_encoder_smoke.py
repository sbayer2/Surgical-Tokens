"""G1 gate: both frozen encoders embed real CholecTrack20 clips, BOTH regimes, on MPS.

Not the assay — the gate that unblocks Stage 1 (see docs/ASSAY_PLAN.md):
  - E-recon: VideoMAE-base (masked pixel reconstruction; appearance-biased)
  - E-pred : V-JEPA 2 (latent prediction; the essence candidate)

Two clip sources, matching the plan's regimes:
  - coarse: 1 fps extracted PNG frames (Training/Validation)
  - fine:   25 fps raw .mp4 decoded with PyAV (Testing). PyAV ships native
    Apple-silicon (arm64) wheels over ffmpeg; this is the video path the
    fine-regime assay will use, so it must pass here.

Prints embedding shapes + wall-clock per (regime, encoder). Exits non-zero
if any of the four combinations fails.

    python scripts/g1_encoder_smoke.py --data ~/datasets/cholectrack20 \
        --coarse-video Training/VID02 --fine-video Testing/VID01 --n-frames 16
"""

from __future__ import annotations

import argparse
import glob
import os
import time

import numpy as np
import torch
from PIL import Image

VIDEOMAE_ID = "MCG-NJU/videomae-base"
VJEPA2_ID = "facebook/vjepa2-vitl-fpc64-256"


def load_clip_frames(video_dir: str, n_frames: int, size: int = 224) -> list:
    """Coarse regime: a contiguous window of extracted PNG frames from the
    middle of the procedure (mid-op = active manipulation)."""
    frames = sorted(glob.glob(os.path.join(video_dir, "Frames", "*.png")))
    if len(frames) < n_frames:
        raise SystemExit(f"only {len(frames)} frames in {video_dir}")
    start = len(frames) // 2 - n_frames // 2
    picks = frames[start:start + n_frames]
    return [Image.open(p).convert("RGB").resize((size, size)) for p in picks]


def load_clip_mp4(path: str, n_frames: int, size: int = 224) -> list:
    """Fine regime: decode a contiguous window from the middle of a 25 fps
    .mp4 with PyAV (native arm64 ffmpeg). Seeks near the midpoint so we don't
    decode an hour-long surgery to reach it."""
    import av

    if not os.path.exists(path):
        raise SystemExit(f"no video at {path}")
    container = av.open(path)
    stream = container.streams.video[0]
    if stream.duration:
        container.seek(int(stream.duration * 0.5), stream=stream)
    out = []
    for frame in container.decode(video=0):
        out.append(frame.to_image().convert("RGB").resize((size, size)))
        if len(out) >= n_frames:
            break
    container.close()
    if len(out) < n_frames:
        raise SystemExit(f"only decoded {len(out)} frames from {path}")
    return out


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()


def run_videomae(clip, device) -> tuple:
    from transformers import AutoImageProcessor, VideoMAEModel

    proc = AutoImageProcessor.from_pretrained(VIDEOMAE_ID)
    model = VideoMAEModel.from_pretrained(VIDEOMAE_ID).to(device).eval()
    inputs = proc(clip, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    _sync(device)
    emb = out.last_hidden_state.mean(dim=1)  # pooled clip embedding
    return emb.shape, float(emb.float().mean()), time.time() - t0


def run_vjepa2(clip, device) -> tuple:
    from transformers import AutoVideoProcessor, VJEPA2Model

    proc = AutoVideoProcessor.from_pretrained(VJEPA2_ID)
    model = VJEPA2Model.from_pretrained(VJEPA2_ID).to(device).eval()
    video = np.stack([np.asarray(f) for f in clip])  # (T, H, W, C)
    inputs = proc(video, return_tensors="pt").to(device)
    t0 = time.time()
    with torch.no_grad():
        out = model(**inputs)
    _sync(device)
    emb = out.last_hidden_state.mean(dim=1)
    return emb.shape, float(emb.float().mean()), time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--coarse-video", default="Training/VID02")
    ap.add_argument("--fine-video", default="Testing/VID01")
    ap.add_argument("--n-frames", type=int, default=16)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[g1] device={device}", flush=True)
    root = os.path.expanduser(args.data)

    regimes = []
    cdir = os.path.join(root, args.coarse_video)
    regimes.append(("coarse/1fps-frames", args.coarse_video,
                    load_clip_frames(cdir, args.n_frames)))
    fbase = os.path.basename(args.fine_video)
    fpath = os.path.join(root, args.fine_video, f"{fbase}.mp4")
    regimes.append(("fine/25fps-mp4", args.fine_video,
                    load_clip_mp4(fpath, args.n_frames)))

    ok = True
    encoders = (("E-recon VideoMAE", run_videomae), ("E-pred V-JEPA2", run_vjepa2))
    for regime, vid, clip in regimes:
        print(f"[g1] {regime}: {len(clip)}-frame clip from {vid} "
              f"({clip[0].size[0]}px)", flush=True)
        for name, fn in encoders:
            try:
                shape, mean, secs = fn(clip, device)
                print(f"[g1]   {name:18s}: emb {tuple(shape)}  "
                      f"mean={mean:+.4f}  {secs:.1f}s", flush=True)
            except Exception as e:  # noqa: BLE001 - gate reports, never crashes silently
                ok = False
                print(f"[g1]   {name:18s}: FAILED — {type(e).__name__}: {e}",
                      flush=True)
    print(f"[g1] {'PASS' if ok else 'FAIL'} — 4/4 combos" if ok
          else "[g1] FAIL — see above", flush=True)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
