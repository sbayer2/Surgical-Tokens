"""The horizon sweep: does long-horizon latent prediction build better response
representations? (H1, the worn-stairs hypothesis, vs monotone H0.)

Per horizon H (in grid-steps), train a small context transformer g to predict
the future frozen embedding e_{t+H} from a context window ending at t
(JEPA-style latent prediction). Freeze g, extract the context vector c_t, and
probe c_t -> response label (bleeding) via LOGO-CV over videos. The LEARNED
gain = trained-g probe minus random-g probe. H1 predicts this gain is
non-monotone with an interior peak (H ~ 2-5 s); H0 predicts monotone decrease.

Sub-second horizons need 25 fps (deferred); at the 0.5 fps grid (step=2 s) the
sweepable horizons are H*2 s each.
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler

W = 8              # context window (timesteps)
CDIM = 256         # context vector dim


class ContextTransformer(nn.Module):
    def __init__(self, d_in, d_ctx=CDIM):
        super().__init__()
        self.proj = nn.Linear(d_in, d_ctx)
        layer = nn.TransformerEncoderLayer(d_ctx, nhead=4, dim_feedforward=512,
                                           batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.predict = nn.Linear(d_ctx, d_in)  # predict future frozen embedding

    def context(self, x):                      # x: (B, W, d_in) -> c: (B, d_ctx)
        return self.enc(self.proj(x))[:, -1, :]

    def forward(self, x):
        return self.predict(self.context(x))


def make_pairs(seqs, H):
    """(context window, future target, video, label_at_context_end) over all videos."""
    X, Y, G, L = [], [], [], []
    for vid, s in seqs.items():
        emb, ble = s["emb"], s["bleeding"]
        for t in range(W - 1, len(emb) - H):
            X.append(emb[t - W + 1:t + 1])
            Y.append(emb[t + H])
            G.append(vid)
            L.append(ble[t])
    return (np.stack(X).astype("float32"), np.stack(Y).astype("float32"),
            np.array(G), np.array(L))


def train_g(X, Y, groups, device, epochs=40, seed=0, random_only=False):
    torch.manual_seed(seed)
    g = ContextTransformer(X.shape[-1]).to(device)
    if random_only:
        g.eval()
        with torch.no_grad():
            return g.context(torch.tensor(X, device=device)).cpu().numpy()
    opt = torch.optim.Adam(g.parameters(), lr=1e-3)
    Xt = torch.tensor(X, device=device)
    Yt = torch.tensor(Y, device=device)
    n = len(X)
    g.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 128):
            idx = perm[i:i + 128]
            opt.zero_grad()
            loss = nn.functional.mse_loss(g(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
    g.eval()
    with torch.no_grad():
        return g.context(Xt).cpu().numpy()


def probe(C, y, groups):
    """LOGO-CV balanced accuracy of C -> y."""
    logo = LeaveOneGroupOut()
    pred = np.zeros(len(y))
    for tr, te in logo.split(C, y, groups):
        if len(np.unique(y[tr])) < 2:
            pred[te] = y[tr][0]
            continue
        sc = StandardScaler().fit(C[tr])
        m = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(C[tr]), y[tr])
        pred[te] = m.predict(sc.transform(C[te]))
    return balanced_accuracy_score(y, pred)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-dir", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--step-sec", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    seqs = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(args.seq_dir), "seq_*.npz"))):
        vid = os.path.basename(f)[4:-4]
        d = np.load(f)
        seqs[vid] = {"emb": d["emb"], "bleeding": d["bleeding"]}
    print(f"[sweep] {len(seqs)} videos, device={device}, context W={W} "
          f"({W*args.step_sec}s), grid step={args.step_sec}s, {args.seeds} seeds")
    print(f"\n{'H (sec)':>8s} {'n_pairs':>8s} {'learned mean':>13s} {'std':>7s} "
          f"{'per-seed':>28s}")

    means, stds, secs = [], [], []
    for H in args.horizons:
        X, Y, G, L = make_pairs(seqs, H)
        gains = []
        for seed in range(args.seeds):
            Ct = train_g(X, Y, G, device, seed=seed)
            Cr = train_g(X, Y, G, device, seed=seed, random_only=True)
            gains.append(probe(Ct, L, G) - probe(Cr, L, G))
        m, sd = float(np.mean(gains)), float(np.std(gains))
        means.append(m); stds.append(sd); secs.append(H * args.step_sec)
        print(f"{H*args.step_sec:8d} {len(X):8d} {m:+13.3f} {sd:7.3f}  "
              f"{'['+' '.join(f'{g:+.3f}' for g in gains)+']':>28s}")

    peak_i = int(np.argmax(means))
    peak = secs[peak_i]
    print(f"\n[sweep] learned-gain curve: peak {means[peak_i]:+.3f}±{stds[peak_i]:.3f} at H={peak}s")
    # separation: is the peak above BOTH ends by more than pooled std?
    interior = peak not in (secs[0], secs[-1])
    pooled = (stds[peak_i] + max(stds[0], stds[-1])) / 2
    sep = min(means[peak_i] - means[0], means[peak_i] - means[-1])
    if interior and sep > pooled:
        print(f"[sweep] INTERIOR PEAK, separation {sep:+.3f} > pooled std {pooled:.3f} "
              f"-> H1 (worn-stairs) SUPPORTED")
    elif interior:
        print(f"[sweep] interior peak but separation {sep:+.3f} <= pooled std {pooled:.3f} "
              f"-> WITHIN NOISE (underpowered; not a finding)")
    else:
        print(f"[sweep] peak at boundary ({peak}s) -> H0 / inconclusive")


if __name__ == "__main__":
    main()
