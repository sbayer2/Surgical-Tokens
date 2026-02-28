# Surgical Tokens

Surgical procedure modeling through clip-token sequence learning and Direct Preference Optimization (DPO). This project converts surgical video into discrete token sequences — then trains a language model to learn procedural grammar and align it toward better surgical outcomes.

## Project Scope

**What this does:** Takes raw surgical procedure video, segments it into clips, encodes each clip into a discrete "surgical token" via VQ-VAE, and trains a GPT-2-style sequence model on the resulting token sequences. The model learns the temporal grammar of surgery — which steps follow which, what sequences correlate with better outcomes — and is then fine-tuned with DPO to prefer procedure flows associated with fewer complications.

**Intended use:**
- Surgical education: generate and evaluate procedural plans
- Research: quantify surgical skill through token-level analysis
- Quality improvement: identify procedural patterns correlated with outcomes
- Simulation: procedural sequence generation for training curricula

**This is NOT:**
- An autonomous surgical system
- A real-time intraoperative guidance tool
- A diagnostic or clinical decision-making tool

## Architecture

```
Surgical Video
    │
    ▼
┌─────────────────┐
│  Scene Detection │  PySceneDetect + quality filtering
│  & Clip Segment  │  2-min clips with 30s overlap
└────────┬────────┘
         ▼
┌─────────────────┐
│  Video Encoder   │  VideoMAE (MCG-NJU/videomae-base)
│  768-d embeddings│  16 frames per clip → 768-d vector
└────────┬────────┘
         ▼
┌─────────────────┐
│  VQ-VAE Sparse   │  Vector quantization → 2048 discrete tokens
│  Encoder         │  EMA codebook updates, commitment loss
└────────┬────────┘
         ▼
┌─────────────────┐
│  Sequence Model  │  GPT-2 (8 layers, 512-d, 8 heads)
│  Pretraining     │  Next-token prediction on procedure sequences
└────────┬────────┘
         ▼
┌─────────────────┐
│  DPO Alignment   │  Preferred vs. rejected procedure pairs
│                  │  Outcome-labeled via composite surgical metrics
└────────┬────────┘
         ▼
┌─────────────────┐
│  Judge (Claude)  │  Reasoning model scores procedure quality
│                  │  on safety, flow, efficiency, tissue respect
└─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### Installation

```bash
git clone git@github.com:sbayer2/Surgical-Tokens.git
cd Surgical-Tokens
pip install -e .
```

### Run the Synthetic Demo (No GPU Required)

Validates the entire pipeline end-to-end using synthetic data:

```bash
python scripts/run_synthetic_demo.py --output-dir ./demo_output
```

This runs all 6 stages (~10 seconds on CPU):
1. **Synthetic data generation** — 50 cases, 500 clip embeddings, case log
2. **Clustering validation** — unsupervised phase discovery via K-Means
3. **VQ-VAE training** — learns a 256-token codebook from embeddings
4. **Sequence model** — trains a small GPT-2 on tokenized procedures
5. **Evaluation** — KL divergence, coherence scores, trigram diversity
6. **DPO pair generation** — outcome labeling and preference pair creation

### Full Pipeline (Requires GPU + Data)

```bash
# 1. Extract clip embeddings from surgical video
surtok extract-embeddings --video-dir ./data/videos --output ./outputs/embeddings.pt

# 2. Build the discrete vocabulary (VQ-VAE codebook)
surtok build-vocabulary --embeddings ./outputs/embeddings.pt --output ./outputs/codebook/

# 3. Train the sequence model
surtok train --config configs/default.yaml --stage pretrain

# 4. Run DPO alignment
surtok train --config configs/default.yaml --stage dpo
```

## GPU Setup

You do not need your own GPU hardware. Several cloud options provide easy access:

### Google Colab (Free)

The fastest way to get started. Colab provides free T4 GPU access (usage limits apply).

```python
# In a Colab notebook:
!git clone https://github.com/sbayer2/Surgical-Tokens.git
%cd Surgical-Tokens
!pip install -e .
!python scripts/run_synthetic_demo.py
```

Select **Runtime > Change runtime type > T4 GPU** for GPU acceleration.

### HuggingFace Spaces with ZeroGPU ($9/month)

ZeroGPU dynamically allocates NVIDIA H200 GPU slices (~70 GB VRAM) only when your function runs, then releases them immediately. Requires a [HuggingFace PRO account](https://huggingface.co/pricing) ($9/month) to create Spaces; free users can use existing ZeroGPU Spaces.

1. Create a new Space at [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Gradio** SDK and **ZeroGPU** hardware
3. Upload model weights to a HuggingFace model repo
4. Use the `@spaces.GPU` decorator to request GPU on-demand:

```python
import spaces
import torch

model = YourModel()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))

@spaces.GPU  # GPU allocated only during this call
def predict(input_data):
    model.to("cuda")
    with torch.no_grad():
        return model(input_data.to("cuda")).cpu()
```

PRO quota: ~25 min of H200 compute per day across all your Spaces.

### HuggingFace Inference Endpoints (Pay-per-use)

For production or heavy training workloads:

1. Push your trained model to a HuggingFace repo
2. Go to [ui.endpoints.huggingface.co](https://ui.endpoints.huggingface.co)
3. Create an endpoint, select your model, choose GPU tier:
   - **NVIDIA T4** (~$0.60/hr) — sufficient for inference and light fine-tuning
   - **NVIDIA A10G** (~$1.30/hr) — recommended for DPO training
   - **NVIDIA A100** (~$6.50/hr) — full pretraining runs
4. The endpoint auto-scales and can pause when idle (pay only for active time)

### Lightning AI (Free Tier)

[Lightning AI Studios](https://lightning.ai) provides free GPU credits with a full Linux environment:

```bash
# Inside a Lightning Studio with GPU:
git clone https://github.com/sbayer2/Surgical-Tokens.git
cd Surgical-Tokens && pip install -e .
python scripts/run_synthetic_demo.py
```

### Estimated Compute Requirements

| Stage | GPU Memory | Time (A100) | Time (T4) |
|-------|-----------|-------------|-----------|
| Video encoding (1000 clips) | ~4 GB | ~15 min | ~1 hr |
| VQ-VAE training (50 epochs) | ~2 GB | ~5 min | ~20 min |
| Sequence pretraining (100 epochs) | ~6 GB | ~2 hr | ~8 hr |
| DPO alignment (3 epochs) | ~8 GB | ~30 min | ~2 hr |

## Obtaining Surgical Video Data

This project is designed to work with surgical procedure videos. Below are pathways to access video data for research and education.

### ACS (American College of Surgeons) Resources

The ACS maintains the [Online Video Library](https://www.facs.org/for-medical-professionals/education/tools-and-platforms/acs-online-video-library/) — 3,000+ peer-reviewed surgical procedure videos distributed via [CineMed](https://cine-med.com/acsonline/). Access paths for non-members:

| Method | Details |
|--------|---------|
| **Institutional subscription** | Many university hospitals hold site licenses. Check your medical library. |
| **Individual subscription** | Non-members can purchase directly from CineMed (1-800-633-0004). |
| **VBE Webinar Series** | [Live sessions are free](https://www.facs.org/for-medical-professionals/education/programs/vbe-webinar-series/) for all, including non-members. Recordings: $80/webinar. |
| **ACS Learning Portal** | Some content at [learning.facs.org](https://learning.facs.org/) is accessible to guest accounts. |
| **Student/Resident membership** | Reduced rates for trainees at [facs.org/member-services](https://www.facs.org/member-services/). |

### Public Surgical Video Datasets (Open Access)

These peer-reviewed datasets are freely available for research — **no ACS membership required**:

| Dataset | Procedure | Size | Annotations | Access |
|---------|-----------|------|-------------|--------|
| **Cholec80** | Lap. cholecystectomy | 80 videos | Phase labels, tool presence | [CAMMA](http://camma.unistra.fr/datasets) |
| **CholecT50** | Lap. cholecystectomy | 50 videos | Action triplets (instrument, verb, target) | [GitHub](https://github.com/CAMMA-public/cholect50) |
| **JIGSAWS** | Robotic suturing/knot-tying | 103 clips | Gesture labels, skill ratings, kinematics | [JHU](https://cirl.lcsr.jhu.edu/research/hmm/datasets/jigsaws_release/) |
| **AutoLaparo** | Lap. hysterectomy | 21 videos | Phase, motion, segmentation | [autolaparo.github.io](https://autolaparo.github.io) |
| **HeiChole** | Lap. cholecystectomy | 33 videos | Surgical actions, workflow | Heidelberg University |
| **SurgToolLoc** | Da Vinci robotic | 24,695 clips | Tool presence labels | [Grand Challenge](https://surgtoolloc.grand-challenge.org) |
| **Endoscapes2023** | Lap. cholecystectomy | 201 videos | Segmentation masks, CVS assessments | [Scientific Data](https://www.nature.com/articles/s41597-025-04642-4) |

**Cholec80** is the recommended starting point — the most widely cited benchmark in surgical video understanding.

### CAMMA Datasets (University of Strasbourg)

The [CAMMA research group](http://camma.unistra.fr) at IHU Strasbourg maintains the most comprehensive open surgical video repository. To access:

1. Visit [camma.unistra.fr/datasets](http://camma.unistra.fr/datasets)
2. Fill out the data use agreement form
3. Receive download credentials (typically within a few business days)
4. Contact: camma.dataset@gmail.com

See also: [list-of-surgical-tool-datasets](https://github.com/luiscarlosgph/list-of-surgical-tool-datasets) for a curated index organized by task.

### Using Your Own Data

If you have institutional access to surgical video:

1. Place `.mp4` files in `./data/videos/`
2. Prepare a case log CSV with columns: `video_id`, `operative_time_min`, `blood_loss_ml`, `complications_30d`, `conversion_to_open`
3. Run the extraction pipeline: `surtok extract-embeddings --video-dir ./data/videos`

## Project Structure

```
surgical_tokens/
├── configs/
│   └── default.yaml              # All hyperparameters
├── scripts/
│   └── run_synthetic_demo.py     # End-to-end demo script
├── surgical_tokens/
│   ├── data/
│   │   ├── case_log.py           # Outcome labeling & DPO pair generation
│   │   ├── quality_filter.py     # Clip quality filtering (brightness, entropy, motion)
│   │   ├── segmentation.py       # Video → clip segmentation
│   │   └── synthetic.py          # Synthetic data generation for testing
│   ├── encoding/
│   │   ├── sparse_encoder.py     # VQ-VAE codebook learning
│   │   └── video_encoder.py      # VideoMAE clip encoder
│   ├── evaluation/
│   │   ├── clustering.py         # Unsupervised clustering metrics & plots
│   │   ├── metrics.py            # Transition matrices, coherence scores
│   │   └── procedure_eval.py     # Generation quality & preference accuracy
│   ├── models/
│   │   ├── judge.py              # Claude/GPT reasoning judge
│   │   └── sequence_model.py     # GPT-2 procedure sequence model
│   ├── pipeline/
│   │   ├── build_vocabulary.py   # VQ-VAE training pipeline
│   │   ├── extract_embeddings.py # Video → embedding extraction
│   │   └── train_pipeline.py     # Full training orchestration
│   └── training/
│       ├── distributed.py        # Multi-GPU / distributed setup
│       ├── dpo.py                # DPO training loop
│       └── pretrain.py           # Sequence model pretraining
├── tests/
│   └── test_synthetic.py
├── configs/
│   └── default.yaml
├── pyproject.toml
└── LICENSE
```

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

```yaml
sparse_encoder:
  codebook_size: 2048       # Discrete vocabulary size
  embedding_dim: 768        # Must match video encoder output

sequence_model:
  architecture: "gpt2"
  n_embd: 512
  n_layer: 8
  n_head: 8

dpo:
  beta: 0.1                 # Lower = stronger preference signal
```

## How It Works

1. **Video Segmentation**: Raw surgical video is split into 2-minute clips with 30-second overlaps using PySceneDetect for scene-aware boundaries.

2. **Clip Encoding**: Each clip is encoded to a 768-d vector by VideoMAE, a self-supervised video transformer pretrained on large-scale video.

3. **Discrete Tokenization**: A VQ-VAE maps each 768-d embedding to one of 2048 discrete codebook entries. Each entry represents a reusable "surgical action primitive."

4. **Sequence Learning**: A GPT-2 model learns next-token prediction over procedure sequences — effectively learning the grammar of surgery (which steps follow which).

5. **Outcome Labeling**: Surgical case logs (operative time, blood loss, complications, conversion rate) are composited into a quality score. Top 30% are labeled "better."

6. **DPO Alignment**: The model is fine-tuned with Direct Preference Optimization using (better, good) procedure pairs, learning to prefer sequences associated with better outcomes.

7. **Judge Evaluation**: An optional reasoning model (Claude) evaluates generated procedures on anatomical safety, instrument handling, tissue respect, procedural flow, and efficiency.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{surgical_tokens_2026,
  title={Surgical Tokens: Procedure Modeling via Clip-Token Sequence Learning and DPO},
  author={sbayer2},
  year={2026},
  url={https://github.com/sbayer2/Surgical-Tokens}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.
