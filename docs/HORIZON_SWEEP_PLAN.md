# Preregistered: the prediction-horizon sweep (the "worn-stairs" experiment)

> Status: **PREREGISTERED 2026-07-14**, frozen before the training runs.
> Amendments are dated additions, not edits (parent-project discipline).
> Motivated by S1 (frozen-encoder null) and by the hypothesis that sparse,
> long-horizon prediction filters for invariant surgical structure the way
> worn stairs record the integral of use rather than any single footstep.

## The hypothesis, stated so it can be graded

**H1 (primary — the worn-stairs / invariance claim):** a self-supervised
model trained to predict surgical representations at a LONG horizon (seconds)
learns representations that predict held-out surgical response BETTER than the
same model trained at a SHORT horizon (sub-second) — even though the long
horizon sees "less" motion. Mechanism (registered): long gaps defeat
interpolation, so only slowly-varying, physically-meaningful state predicts
across them (slow-feature analysis; predictive information). Short horizons
let the model interpolate pixel micro-motion and learn nothing physical.

**The counter-hypothesis this is tested against (H0):** prediction quality is
monotone in temporal density / inverse in horizon — nearer targets are easier
and yield better representations (the conventional expectation). Under H0 the
sweep curve falls with horizon; under H1 it rises then falls (an optimal
horizon in the seconds range).

This is the crux the parent project's method exists for: counterintuitive-true
and counterintuitive-false look identical until measured. The sweep measures it.

## Design

Two orthogonal knobs, deliberately separated (they were conflated in the
initial framing):
- **input density** — fps the encoder sees within its 16-frame clip (fixed
  here; the fine-regime check addresses density separately).
- **prediction horizon H** — how far ahead the self-supervised target sits.
  THIS is the swept variable: H ∈ {0.2 s, 0.5 s, 1 s, 2 s, 5 s, 10 s}.

**Task.** Predict the future representation: from an onset clip ending at time
t, predict the (frozen-encoder) embedding of the clip at t+H. A small trainable
transformer head (the "attention heads" of the proposal) is trained with this
objective; the frozen encoder supplies both inputs and targets (JEPA-style
latent prediction, no pixel decode). Trained per horizon H.

**Data.** Coarse-regime training frames (1 fps, 10 videos) for H up to the 1 fps
limit; the 25 fps test video extends the *dense* end if the fine-regime check
shows density matters. Video-level splits throughout (P5).

**Readout (what "better representations" means).** After training each
horizon's head, FREEZE it and evaluate its onset representation on the S1
response targets (bleeding consequence, tool trajectory) via the same LOGO-CV
probe + B4 random baseline. The metric is the LEARNED response-prediction gain
(trained head minus random head) as a function of H. H1 predicts this gain is
non-monotone with a peak at H in the seconds range; H0 predicts monotone
decrease.

## Registered predictions

- **P1 (the headline):** the learned response-prediction gain is HIGHER at
  H ≈ 2–5 s than at H ≈ 0.2 s. A non-monotone curve peaking in the seconds
  range confirms the worn-stairs hypothesis. Falsifier: monotone-decreasing
  curve (nearer is strictly better) refutes it.
- **P2:** the short-horizon head (H=0.2 s) barely beats its random baseline
  (it interpolates, learns little) — echoing S1's frozen-encoder null.
- **P3:** the long-horizon advantage is larger for the CONSEQUENCE target
  (bleeding) than for phase (context) — dynamics prediction, not appearance.
- **P4 (honesty guard):** at very long H (10 s) the target becomes nearly
  unpredictable (decorrelated), so the curve falls again — the peak is
  interior, not "longer is always better." If gain keeps rising to 10 s, the
  mechanism is not what H1 claims and needs rethinking.

## Gates / caveats registered up front

- **Data scale:** ~200–500 clips over 10 videos is small for training. The
  head is deliberately tiny (a 2-layer transformer, <1 M params); if it
  overfits (train gain >> val gain across all H), the experiment is
  underpowered and reports that, not a horizon curve.
- **The frozen-encoder-target confound:** targets are frozen-encoder
  embeddings, so the ceiling is what that encoder captures. If S1's encoders
  miss tissue dynamics entirely, no horizon recovers them — which is why the
  fine-regime check (does 25 fps reveal dynamics at all) runs FIRST and gates
  whether a dense-input arm is added.
- **Compute:** 6 horizons × (train head + LOGO eval) on an M5 Pro — hours, not
  days; embeddings are cached from Stage 1.

## Order of operations

1. Fine-regime check (running) — does input density reveal dynamics 1 fps
   missed? Gates whether the sweep needs a 25 fps dense-input arm.
2. This horizon sweep — the primary test of H1.
3. Only if H1 holds: fine-tune the encoder itself at the winning horizon (a
   further preregistration), the real "train the model weights" step.
