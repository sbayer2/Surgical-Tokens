"""G2 gate: label <-> clip alignment sanity on CholecTrack20.

Programmatic checks across sampled videos + a visual overlay (draw the tool
bboxes on real frames) so alignment is *seen*, not just asserted:

  - every annotated frame key has a matching Frames/<key>.png (coarse regime)
  - tool_bbox coords are normalized [0,1]; phase in [0,6]; operator present
  - render N frames with bboxes drawn -> <out>/overlay_*.png for eyeballing

    python scripts/g2_alignment_check.py --data ~/datasets/cholectrack20 \
        --out /tmp/g2_overlays --n-overlays 6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random

from PIL import Image, ImageDraw

TOOLS = ["grasper", "bipolar", "hook", "scissors", "clipper", "irrigator", "specimen-bag"]


def check_video(vdir: str) -> dict:
    vid = os.path.basename(vdir)
    d = json.load(open(os.path.join(vdir, f"{vid}.json")))
    ann = d["annotations"]
    frame_files = {os.path.splitext(os.path.basename(p))[0].lstrip("0") or "0": p
                   for p in glob.glob(os.path.join(vdir, "Frames", "*.png"))}
    keys = list(ann)
    matched = sum(1 for k in keys if k in frame_files)
    bad_bbox = bad_phase = no_op = 0
    for k in keys:
        for obj in ann[k]:
            bb = obj.get("tool_bbox")
            if bb and not all(0.0 <= c <= 1.0 for c in bb):
                bad_bbox += 1
            if not (0 <= obj.get("phase", -1) <= 6):
                bad_phase += 1
            if obj.get("operator", -1) < 0:
                no_op += 1
    return {"vid": vid, "n_ann_frames": len(keys), "n_png": len(frame_files),
            "matched": matched, "bad_bbox": bad_bbox, "bad_phase": bad_phase,
            "no_operator": no_op}


def draw_overlay(vdir: str, out_dir: str, n: int, rng: random.Random) -> int:
    vid = os.path.basename(vdir)
    d = json.load(open(os.path.join(vdir, f"{vid}.json")))
    ann = d["annotations"]
    made = 0
    keys = [k for k in ann if any(o.get("tool_bbox") for o in ann[k])]
    rng.shuffle(keys)
    for k in keys[:n]:
        png = os.path.join(vdir, "Frames", f"{int(k):06d}.png")
        if not os.path.exists(png):
            continue
        im = Image.open(png).convert("RGB")
        W, H = im.size
        dr = ImageDraw.Draw(im)
        for obj in ann[k]:
            bb = obj.get("tool_bbox")
            if not bb:
                continue
            x, y, w, h = bb[0] * W, bb[1] * H, bb[2] * W, bb[3] * H
            dr.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
            label = TOOLS[obj["instrument"]] if 0 <= obj.get("instrument", -1) < 7 else "?"
            dr.text((x + 2, y + 2), f"{label} p{obj.get('phase')}", fill=(255, 255, 0))
        im.save(os.path.join(out_dir, f"overlay_{vid}_{k}.png"))
        made += 1
    return made


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="/tmp/g2_overlays")
    ap.add_argument("--n-overlays", type=int, default=6)
    args = ap.parse_args()

    root = os.path.expanduser(args.data)
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)

    coarse = sorted(glob.glob(f"{root}/Training/VID*") + glob.glob(f"{root}/Validation/VID*"))
    print(f"[g2] checking {len(coarse)} coarse-regime videos ...")
    tot_bad = 0
    for vdir in coarse:
        r = check_video(vdir)
        flag = "" if (r["matched"] == r["n_ann_frames"] and r["bad_bbox"] == 0
                      and r["bad_phase"] == 0) else "  <-- CHECK"
        tot_bad += (r["n_ann_frames"] - r["matched"]) + r["bad_bbox"] + r["bad_phase"]
        print(f"[g2] {r['vid']}: {r['n_ann_frames']} ann / {r['n_png']} png / "
              f"{r['matched']} matched | bad_bbox={r['bad_bbox']} "
              f"bad_phase={r['bad_phase']} no_op={r['no_operator']}{flag}")

    # Visual overlays from a couple of videos for eyeballing.
    made = 0
    for vdir in coarse[:2]:
        made += draw_overlay(vdir, args.out, args.n_overlays // 2 + 1, rng)
    print(f"[g2] wrote {made} overlay images -> {args.out}")
    print(f"[g2] {'PASS' if tot_bad == 0 else 'CHECK (see flags)'} — "
          f"{tot_bad} alignment/validity issues")


if __name__ == "__main__":
    main()
