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
