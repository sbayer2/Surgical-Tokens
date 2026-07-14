# Surgical Tokens — findings

Empirical record for the surgical assay (`docs/ASSAY_PLAN.md`), same discipline
as the parent project `sbayer2/Project-Free-Robot-Agent-World` (`docs/FINDINGS.md`,
F1–F17): preregistered predictions, negatives at equal prominence, own
predictions graded honestly including when wrong.

---

## S1 — B4-corrected NULL: frozen encoders show no *learned* response-consequence dynamics on 1 fps surgical video (the temporal advantage is mostly architectural)

*Run 2026-07-14 on CholecTrack20 (coarse regime, 1 fps). 410 clips from 10
training videos, leave-one-video-out CV (video-level = P5), 100-perm null.
Deltas are dimensionality-fair (temporal − single-frame within an encoder);
absolute scores are not.*

**Setup.** Each 32-frame clip (32 s) → onset (0–15) + response (16–31). From
the **onset** embedding predict response-half targets, comparing {VideoMAE
(E-recon), V-JEPA2 (E-pred)} × {temporal onset, single-frame (B3)}. The P2
"dynamics content" = temporal − single-frame delta.

**The naive result (before the control) looked like a clean P1 win:** V-JEPA2
showed positive temporal deltas (bleeding +0.061, phase +0.167, both p≤.01)
while VideoMAE showed none. It would have been easy to report "latent-
prediction carries surgical dynamics, masked-reconstruction doesn't."

**B4 (random-weights encoder, F6 discipline) overturned it.** Re-embedding
with randomly-initialized encoders of the same architecture, the *learned*
component is trained Δ − random Δ:

| encoder × target | trained Δ | random Δ | learned |
|---|---|---|---|
| VideoMAE · bleeding | −0.015 | +0.081 | **−0.096** |
| V-JEPA2 · bleeding [consequence] | +0.061 | +0.051 | **+0.010 ≈ 0** |
| VideoMAE · phase | −0.010 | +0.064 | **−0.074** |
| V-JEPA2 · phase [context] | +0.167 | +0.074 | **+0.093** |

**Findings (corrected):**
1. **A random 16-frame video transformer already shows the temporal-over-
   single-frame advantage** (random deltas +0.05…+0.08). Much of "temporal
   beats one frame" is architecture — 16 frames pool more than 1 — not
   learning. F6, reproduced on a new instrument.
2. **The consequence target (bleeding) carries NO learned dynamics.**
   V-JEPA2's +0.061 is within its own random baseline (+0.051). The apparent
   consequence-prediction was an architectural artifact. The naive headline
   is false.
3. **The only learned signal is V-JEPA2 on phase (+0.093 over random)** — and
   phase is procedural *context*, not tissue-response. The sole surviving
   learned effect is in the "wrong" place for the essence claim (and phase was
   the target predicted to be an appearance control — graded wrong twice over).
4. **Masked-reconstruction training reduces temporal informativeness below
   random init** (VideoMAE learned deltas −0.096, −0.074): training VideoMAE
   to reconstruct pixels makes its pooled embedding *more* appearance-locked
   than a random transformer. A clean, if unflattering, observation about MAE
   objectives.

**Verdict — P2 FAILS on the frozen-encoder / coarse-regime configuration.**
On 1 fps surgical video, frozen general-video encoders do not demonstrably
carry learned response-*consequence* dynamics beyond the architectural
baseline. Per the plan's registered P2-fail branch, the essence question is
NOT answered here and the next steps are forced, not optional:
- **Fine regime (25 fps test video, reserved):** 1 fps may sit below the
  Nyquist rate of tissue/instrument motion; the dynamics may simply not be
  sampled. This is the first thing to try.
- **Fine-tuning** a predictive encoder on surgical video (the frozen shortcut
  is insufficient) — a NEW preregistration, per the plan.

**What the process bought us.** The B4 control converted a would-be false
positive into an honest null, exactly as F6 did before any Mac time was spent
in the parent project. The naive P1 "win" was mostly architecture; the one
learned signal is real but off-target. Reporting the null at equal prominence
is the point of the preregistration.

**Limitations.** n=410, 10 videos, single random-init seed (random deltas are
themselves noisy — the ±0.01 bleeding gap is within noise; the +0.09 phase gap
is larger but one seed); coarse regime only; permutation floor p≤.01. The
fine-regime run is the decisive follow-up.

Reproduce: `scripts/extract_embeddings_stage1.py` → `scripts/stage2_response_prediction.py` → `scripts/b4_random_baseline.py --emb-dir ~/datasets/ct20_emb`.
Artifacts: `~/datasets/ct20_emb/` (gitignored; embeddings + labels, no raw video).

---

## S2 — Fine regime (25 fps): future surgical state is strongly predictable from APPEARANCE, not motion; density is not the bottleneck

*Run 2026-07-14 on the 8 test videos (25 fps lives only there; this spends
reserved-set independence for the density question, distinct from final
P-confirmation). 240 anchors (30/video), LOGO-CV, predicting the label 15 s
ahead from three onset variants x {trained, random-B4} encoders. First-order
trained-vs-random gains are robust at this n; second-order temporal deltas are
noisy over 8 videos.*

Predicting **bleeding 15 s in the future** (balanced accuracy):

| variant | trained | random | learned |
|---|---|---|---|
| single-frame (B3) | **0.688** | 0.512 | **+0.176** |
| sparse (16 s @ 1 fps) | 0.655 | 0.492 | +0.163 |
| dense (0.6 s @ 25 fps) | 0.652 | 0.511 | +0.141 |

**Findings:**
1. **The representations ARE predictively useful (corrects S1's pessimism).**
   Trained encoders predict 15-s-future bleeding/phase far above random
   (learned +0.14…+0.18). S1's "null" was specifically about the temporal
   *delta* (motion-over-frame); it did not isolate that appearance itself is
   strongly predictive of the near future. The procedural grammar is visible
   in a single frame.
2. **The predictive power is APPEARANCE, not dynamics.** For trained encoders
   single-frame ≥ both temporal variants (bleeding 0.688 > 0.655 ≈ 0.652).
   Motion does not beat a clean frame.
3. **Density is NOT the bottleneck — the Nyquist hypothesis is NOT supported.**
   Even at full 25 fps, dense motion adds nothing over appearance (dense ≤
   single-frame). "1 fps was too sparse to see dynamics" (the S1 next-step
   guess) is wrong.
4. **Sparse ≥ dense, weakly.** The 16 s window is never worse than the 0.6 s
   burst and edges it on phase (+0.032). Faint support for the worn-stairs
   direction (long span ≥ short burst), but both lose to single-frame, so it
   is a hint, not a vindication — its real test is the horizon sweep.

**Load-bearing caveat.** Embeddings are **mean-pooled** over space-time tokens,
which weakens explicit motion representation (16 identical frames give a clean
appearance vector; real frames add motion the pooling cannot cleanly expose).
So "dynamics don't help" is entangled with "mean-pooling cannot show dynamics."
This is precisely why the preregistered **horizon sweep** (`docs/HORIZON_SWEEP_PLAN.md`)
— which trains a temporally-aware head to predict the future rather than
pooling it away — is the correct test of whether learned long-horizon dynamics
beat the (now strong) appearance baseline of 0.69.

Reproduce: `scripts/fine_regime_check.py --data ~/datasets/cholectrack20 --out ~/datasets/ct20_fine`.

---

## S3 — Horizon sweep: underpowered NULL — the single-seed "interior peak" was seed-0 luck (F9→F10 again)

*Run 2026-07-14. Latent-prediction head (2-layer context transformer, ~1.6 M
params) trained per horizon H to predict the frozen V-JEPA2 embedding H ahead;
context vector probed → future bleeding, LOGO-CV; learned gain = trained −
random-init head. 0.5 fps grid (step 2 s), horizons {2,4,8,16,32 s}, 10 videos.*

**Single-seed (seed 0) suggested H1:** learned gain +0.027/+0.038/+0.065/
+0.022/+0.025 — an interior peak at 8 s, which the script auto-classified as
"H1 SUPPORTED." **Multi-seed (5 seeds) overturned it:**

| H | learned gain (mean ± std) |
|---|---|
| 2 s | −0.002 ± 0.025 |
| 4 s | −0.016 ± 0.043 |
| 8 s | −0.006 ± 0.048 |
| 16 s | −0.037 ± 0.041 |
| 32 s | −0.028 ± 0.037 |

Every mean is ≈ 0 or slightly negative; per-seed swings (0.15 range at H=8 s)
dwarf every between-horizon difference. **Seed 0 was the most positive seed in
every row** — the apparent 8 s peak was one lucky draw. Third instance this
session of a naive single-seed positive corrected to null by the proper
control (cf. F10 basins, S1 B4).

**Verdict — H1 NOT supported by this experiment; NOT refuted either.** The
experiment is underpowered to detect a horizon effect if one exists — the
prereg's data-scale gate fired ("if it underpowers, report that, not a
horizon curve"). Two causes: (1) the random-init head already scores ~0.60
because the frozen embeddings carry strong appearance signal (S2), so the
trainable head must add atop a strong baseline; (2) a ~0.04 candidate effect
against ~0.04 seed noise is invisible at 10 videos.

**What would actually test the worn-stairs hypothesis** (each a new
preregistration): (a) scale to hundreds of videos (Cholec80/CholecT50); (b)
fine-tune the ENCODER at each horizon, not a head on frozen features (the
frozen appearance baseline is the ceiling here); (c) a temporally-structured
readout instead of last-token pooling. The hypothesis remains live; only this
underpowered test of it is closed.

Reproduce: `scripts/build_1fps_sequence.py` → `scripts/horizon_sweep.py --seq-dir ~/datasets/ct20_seq --seeds 5`.
