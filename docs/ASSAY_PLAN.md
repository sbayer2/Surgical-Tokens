# The Surgical Assay — preregistered run design

> Status: **PREREGISTERED 2026-07-13** — this document and its registered
> predictions (P1–P5) are frozen in commit history *before* any dataset was
> downloaded or any embedding computed. Amendments must be dated additions,
> not edits, mirroring the discipline in
> `sbayer2/Project-Free-Robot-Agent-World` (`docs/GSO_EXPERIMENT.md`,
> `docs/FINDINGS.md` F1–F17), whose instruments this plan transfers.

## Question

Does an embedding of surgical video clips carry **consequence essence** —
what the tissue is about to do under the applied action — or only
**appearance** (what the scene looks like)?

A clip of manipulation contains both halves of the answer in one sample: the
instrument applies force (the probe is in the frame) and the tissue's
response is recorded in the following frames. This plan measures whether
that physics actually ends up *in the embedding array*, using four
instruments validated on pseudo-marble:

1. the **response-prediction test** (behavior head, found in nature),
2. the **baseline-subtraction discipline** (credit nothing the architecture
   or static appearance gets for free),
3. the **bit-counting bottleneck** (how many bits is a surgical moment?),
4. the **coherence metric** (do appearance and consequence share one code?).

Philosophy: *pour in enormous context, demand out a tiny essence, and own
the instrument that checks the essence is real.*

## Data (free; no institutional compute)

- **Primary: Cholec80** (CAMMA, University of Strasbourg) — 80 laparoscopic
  cholecystectomy videos with expert **surgical-phase labels** (7 phases)
  and **tool-presence labels** (7 instruments). ~60–80 GB. Access by
  research request (institutional email). The labels are the exact answer
  key — the MuJoCo of this experiment.
- **Upgrade if granted: CholecT50** — ⟨instrument, verb, target⟩ action
  triplets: literal action ground truth for response prediction.
- **Reserved, out of scope here: AutoLaparo** (hysterectomy) — the future
  cross-procedure transfer test. One question per experiment.
- **Clip unit:** 8–16 s at ~5 fps (16–32 frames), NOT 2-minute segments —
  response prediction needs within-clip temporal structure and sample count
  (~25–40k clips from 80 videos).

## Stage 1 — Two frozen encoders (the controlled comparison)

Embed every clip with both, inference only (no fine-tuning until the frozen
comparison earns it):

- **E-recon** — VideoMAE-base (masked *pixel reconstruction*; the
  appearance-biased incumbent already in this repo's pipeline).
- **E-pred** — V-JEPA public checkpoint (*latent prediction*; the essence
  candidate — the objective that can only reduce loss by encoding the state
  that determines the future).

## Stage 2 — Response-prediction test (core assay)

Split each clip: first half = action onset, second half = response. From
the first-half embedding predict:

- (a) the second-half **embedding** (ridge / small-MLP probe),
- (b) **phase transition within the next N seconds** (exact label),
- (c) **tool appearance/disappearance** (exact label),
- (d) next **verb** (CholecT50 arm, if granted).

**Baselines (F6 discipline — every reported number is a gain over ALL of
these):**

| id | baseline | what it controls for |
|---|---|---|
| B1 | majority-class / predict-mean | task triviality |
| B2 | shuffled first↔second half pairs | chance level of the probe |
| B3 | **single-frame embedding** | static appearance — the temporal-minus-single-frame delta IS the measured dynamics content |
| B4 | random-weights encoder | architectural prior (untrained nets already score; F6) |

**Split discipline:** all probes trained/evaluated with **video-level
splits** (never clip-level) — clips from one patient share appearance; this
is the category-holdout lesson (P5 below).

## Stage 3 — Bit-counting bottleneck (F17 transfer)

FSQ-quantize the winning encoder's embeddings: k ∈ {1, 2, 3, 4, 6, 8, 16,
32} ternary dims × 3 seeds; each bottleneck model trains two heads —
**appearance** (embedding reconstruction) and **response** (Stage-2
targets). Output: two rate–distortion curves → *how many bits is a surgical
moment?* **Codebook utilization (perplexity) co-reported with every number**
— naive sequence/coherence metrics are maximized by a collapsed codebook
(F10's law; VQ literature calls it codebook collapse).

## Stage 4 — Coherence metric (the nudge test)

On each two-head bottleneck model: perturb the latent; measure whether
directions that move the appearance head also move the response head —
**minus the same-width untrained baseline**, co-reported with prediction
gain and utilization (the full F10 reporting law: coherence alone rewards
degeneracy).

## Registered predictions (frozen with this commit)

- **P1 (objective choice):** E-pred beats E-recon on response prediction
  after baseline subtraction. Failure ⇒ the generative-vs-predictive story
  needs revision at this scale.
- **P2 (dynamics content, the falsifier with teeth):** temporal clips beat
  single-frame (B3) for both encoders, larger delta for E-pred. If temporal
  ≈ single-frame everywhere, frozen general-video embeddings carry NO
  surgical dynamics — the essence claim dies for frozen encoders, and any
  fine-tuning continuation must be a new preregistration.
- **P3 (bit count):** response knee ≪ appearance knee; point band: response
  knee ≤ 6 trits (7 phases ≈ 2 trits, plus transition state).
- **P4 (scarcity manufactures unity):** narrower codes → higher coherence
  *with intact prediction* — first replication of pseudo-marble F17 outside
  its authored world.
- **P5 (confound watch):** the video-identity shortcut (patient-specific
  appearance) is the expected contaminant; video-level splits are the
  defense; a post-hoc check correlates probe error with visual
  nearest-neighbor distance to training videos.

## Gates and budget (Apple M5 Pro, 64 GB — laptop-class by design)

| gate | check | cost |
|---|---|---|
| G0 | Cholec80 access granted; labels parse | request form; days of waiting (out of our control — file FIRST) |
| G1 | VideoMAE + V-JEPA inference runs on MPS/MLX | an evening |
| G2 | label↔clip alignment sanity (eyeball 20 clips) | an hour |
| Stage 1 | ~25–40k clips × 2 encoders | ~3–5 h unattended |
| Stage 2 | probes + baselines | minutes per probe |
| Stage 3 | 24 bottleneck trainings | ~1 h |
| Stage 4 | coherence sweeps | ~1 h |

Peak storage ~90 GB (videos); everything downstream is megabytes. Total:
**one weekend, mostly unattended.**

## What each outcome means

- **P1+P2 pass** → the essence objective is validated on real interaction
  video; the LSVCM pipeline should be rebuilt around E-pred before any
  sequence-model or DPO work.
- **P2 fails** → frozen general-video encoders don't carry surgical
  dynamics; fine-tuning becomes the *measured* next step, not an assumption.
- **P3/P4 replicate** → F17 generalizes beyond a synthetic world: narrow
  discrete codes both size the essence and manufacture appearance↔response
  unity — a design law candidate for essence-bearing world models.
- **Null everywhere** → published at equal prominence: that is what the
  preregistration is for.

## Explicitly out of scope (each is its own future preregistration)

- Cross-procedure transfer (appendectomy→cholecystectomy; needs a second
  procedure dataset — AutoLaparo).
- The repo's GPT sequence-model and DPO stages — they wait until the tokens
  beneath them are certified to mean something.
- The Claude/GPT judge — must first be validated per-dimension against the
  expert phase labels (prediction from pseudo-marble F11: reliable on
  discrete judgments, unreliable on continuous quality scores).
- Surgeon-identity confound gate for DPO (outcome ~ PGY/volume): required
  before any preference optimization, specified in the project page at
  the ai-research vault (`wiki/surgical-tokens.md`).

---

## AMENDMENT 2026-07-13 (same day, pre-data): access reality + two temporal regimes

Investigation of the CAMMA distribution channels (TF-Cholec80 repo, CAMMA
datasets page, CholecT50 docs) before any download:

**Access:** TF-Cholec80's scripted S3 download currently returns 403
(gated). Current routes, both Google Forms (institutional email):
- Cholec80 (full 25 fps videos + phase/tool labels):
  https://docs.google.com/forms/d/1GwZFM3-GhEduBs1d5QzbfFksKmS1OqXZAz8keYi-wKI
- CholecT50 (50 videos as 1 fps frames + ⟨instrument, verb, target⟩
  triplets): https://forms.gle/GbMj8TwNoNpMUJuv9

**License correction:** CC-BY-NC-SA 4.0 — non-commercial, share-alike
redistribution IS permitted with citation (the original issue text said "no
redistribution"; too strict). Videos still stay out of the repo as a size
and courtesy matter.

**Temporal-regime split (plan clarification, registered before data):** the
frame-based releases are 1 fps, so the assay runs in two regimes with
different claims:
- **Coarse regime (1 fps — CholecT50 frames or Cholec80-at-1fps):** clips
  are 16-frame ≈ 16 s windows. P1/P2 here measure *procedural* dynamics
  (phase flow, tool events, triplet verbs) — NOT tissue mechanics. All
  Stage-2/3/4 machinery applies unchanged.
- **Fine regime (25 fps — Cholec80 video grant):** 16-frame ≈ 0.6–3 s clips
  at native or lightly-strided rate; this is the *tissue-response* arm the
  essence claim ultimately needs (deformation, tension, bleeding onset).
- Registered expectation: P2's temporal-vs-single-frame delta is larger in
  the fine regime than the coarse one. If the coarse regime shows NO delta,
  that does not falsify the essence claim (1 fps may simply be below the
  dynamics' Nyquist rate); only the fine regime can falsify it.

G0 therefore splits: G0a = CholecT50 form (triplets, coarse regime); G0b =
Cholec80 form (videos, fine regime). File both immediately; stages proceed
with whichever grants first.

---

## AMENDMENT 2026-07-13 (evening): G0c GRANTED — CholecTrack20

CAMMA granted access to **CholecTrack20** (20 laparoscopic cholecystectomy
videos; CVPR 2025 dataset): multi-class multi-tool **tracking trajectories**
(visibility / intracorporeal movement / lifelong paths), surgical phases,
scene visual-challenge labels, **surgeon-operator labels**, bleeding/smoke/
adverse-event annotations. Annotations at 1 fps; **raw 25 fps video for the
test set**. License CC-BY-NC-SA 4.0 + DUA (no re-identification; security
care; research-only; publication code must be open — this repo already is).
The access key is personal to the grantee's email and is NOT recorded in
this repository or its issues.

**Adaptations (registered before download):**

- **New primary response target — tool-trajectory forecasting:** from a
  clip-onset embedding, predict subsequent tool positions/paths (dense,
  continuous dynamics label — the closest surgical analog to the parent
  project's 21-dim behavior vector). Phase-transition and tool-event targets
  remain as secondary.
- **New consequence target:** bleeding/adverse-event occurrence within the
  following window (Brier-scored — the F11 calibration lesson).
- **P5 upgrade:** surgeon-operator labels make the identity confound
  directly measurable: probes must beat operator-only covariates, and a
  registered check regresses embedding-probe error on operator identity.
- **Scale honesty:** n=20 videos ⇒ video-level splits are tighter; the
  official CholecTrack20 splits are used as published; all results reported
  with per-video spread. Cholec80/CholecT50 requests remain filed (G0a/G0b)
  for scale-up and the triplet arm.
- **Regime mapping:** train/val = coarse regime (1 fps annotations);
  official test set = fine regime (25 fps raw video) — evaluation lands
  exactly where the falsifying arm needs it.

Citation obligation: Nwoye et al., CholecTrack20 (CVPR 2025), in any
publication arising.
