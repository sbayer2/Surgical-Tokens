"""Tests for synthetic data generation and core pipeline components."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from surgical_tokens.data.case_log import (
    compute_composite_score,
    generate_dpo_pairs,
    label_outcomes,
)
from surgical_tokens.data.synthetic import (
    generate_full_synthetic_dataset,
    generate_synthetic_case_log,
    generate_synthetic_embeddings,
    generate_synthetic_procedures,
)
from surgical_tokens.encoding.sparse_encoder import SparseEncoder, VectorQuantizer
from surgical_tokens.evaluation.clustering import evaluate_clustering
from surgical_tokens.evaluation.metrics import (
    build_transition_matrix,
    outcome_prediction_accuracy,
    procedural_coherence_score,
)
from surgical_tokens.models.sequence_model import (
    DPOPairDataset,
    ProcedureDataset,
    create_sequence_model,
)
from surgical_tokens.config import load_config


@pytest.fixture
def cfg():
    return load_config("configs/default.yaml")


@pytest.fixture
def synthetic_case_log():
    return generate_synthetic_case_log(n_cases=30)


@pytest.fixture
def synthetic_embeddings():
    return generate_synthetic_embeddings(n_clips=100, embedding_dim=768)


class TestSyntheticData:
    def test_case_log_schema(self, synthetic_case_log):
        df = synthetic_case_log
        assert len(df) == 30
        assert "video_id" in df.columns
        assert "operative_time_min" in df.columns
        assert "estimated_blood_loss_ml" in df.columns
        assert "converted_to_open" in df.columns
        assert "complications_30d" in df.columns
        assert df["operative_time_min"].min() > 0
        assert df["estimated_blood_loss_ml"].min() >= 0

    def test_embeddings_shape(self, synthetic_embeddings):
        emb, labels, ids = synthetic_embeddings
        assert emb.shape == (100, 768)
        assert len(labels) == 100
        assert len(ids) == 100
        assert labels.max() < 8

    def test_procedures(self):
        procs = generate_synthetic_procedures(n_procedures=20, codebook_size=256)
        assert len(procs) == 20
        for p in procs:
            assert "tokens" in p
            assert len(p["tokens"]) >= 8

    def test_full_synthetic_dataset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_full_synthetic_dataset(
                output_dir=tmpdir, n_cases=10, n_clips_per_case=5,
            )
            assert Path(result["case_log"]).exists()
            assert Path(result["embeddings"]).exists()
            assert Path(result["clip_mapping"]).exists()
            assert Path(result["procedures"]).exists()


class TestCaseLog:
    def test_composite_score(self, synthetic_case_log, cfg):
        scores = compute_composite_score(synthetic_case_log, cfg)
        assert len(scores) == len(synthetic_case_log)
        assert scores.dtype == np.float64

    def test_labeling(self, synthetic_case_log, cfg):
        labeled = label_outcomes(synthetic_case_log, cfg)
        assert "label" in labeled.columns
        assert set(labeled["label"].unique()) <= {"good", "better"}
        n_better = (labeled["label"] == "better").sum()
        assert n_better > 0
        assert n_better < len(labeled)

    def test_dpo_pairs(self, synthetic_case_log, cfg):
        labeled = label_outcomes(synthetic_case_log, cfg)
        pairs = generate_dpo_pairs(labeled, max_pairs=50)
        assert len(pairs) <= 50
        for p in pairs:
            assert p.chosen_score <= p.rejected_score  # Lower = better


class TestVectorQuantizer:
    def test_forward(self):
        vq = VectorQuantizer(codebook_size=64, embedding_dim=32)
        z = torch.randn(16, 32)
        out = vq(z)
        assert out["quantized"].shape == (16, 32)
        assert out["indices"].shape == (16,)
        assert out["indices"].max() < 64

    def test_encode_decode(self):
        vq = VectorQuantizer(codebook_size=64, embedding_dim=32)
        z = torch.randn(16, 32)
        indices = vq.encode(z)
        decoded = vq.decode(indices)
        assert decoded.shape == (16, 32)


class TestSparseEncoder:
    def test_forward(self):
        enc = SparseEncoder(input_dim=32, codebook_size=64)
        x = torch.randn(16, 32)
        out = enc(x)
        assert out["reconstructed"].shape == (16, 32)
        assert out["indices"].shape == (16,)

    def test_tokenize_detokenize(self):
        enc = SparseEncoder(input_dim=32, codebook_size=64)
        x = torch.randn(16, 32)
        tokens = enc.tokenize(x)
        reconstructed = enc.detokenize(tokens)
        assert tokens.shape == (16,)
        assert reconstructed.shape == (16, 32)


class TestSequenceModel:
    def test_create_model(self, cfg):
        model = create_sequence_model(cfg)
        assert model is not None
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    def test_forward_pass(self, cfg):
        model = create_sequence_model(cfg)
        batch_size = 4
        seq_len = 32
        input_ids = torch.randint(0, cfg["sequence_model"]["vocab_size"], (batch_size, seq_len))
        outputs = model(input_ids=input_ids)
        assert outputs.logits.shape == (batch_size, seq_len, cfg["sequence_model"]["vocab_size"] + 3)

    def test_procedure_dataset(self):
        seqs = [[10, 20, 30, 40, 50], [15, 25, 35]]
        ds = ProcedureDataset(seqs, max_length=16)
        assert len(ds) == 2
        item = ds[0]
        assert item["input_ids"].shape == (16,)
        assert item["attention_mask"].shape == (16,)
        assert item["labels"].shape == (16,)
        assert item["input_ids"][0] == 0  # BOS

    def test_dpo_pair_dataset(self):
        chosen = [[10, 20, 30]]
        rejected = [[40, 50, 60]]
        ds = DPOPairDataset(chosen, rejected, max_length=16)
        item = ds[0]
        assert "chosen_input_ids" in item
        assert "rejected_input_ids" in item


class TestClustering:
    def test_evaluate_clustering(self, synthetic_embeddings):
        emb, labels, _ = synthetic_embeddings
        results = evaluate_clustering(emb.numpy(), n_clusters_list=[4, 8], true_labels=labels)
        assert 4 in results
        assert 8 in results
        assert "silhouette" in results[4]
        assert "ari" in results[4]


class TestMetrics:
    def test_transition_matrix(self):
        seqs = [[1, 2, 3, 4], [1, 2, 4], [2, 3, 4]]
        tm = build_transition_matrix(seqs, vocab_size=5)
        assert tm.shape == (5, 5)
        assert np.allclose(tm.sum(axis=1), 1.0)

    def test_coherence_score(self):
        seqs = [[1, 2, 3, 4]] * 10
        tm = build_transition_matrix(seqs, vocab_size=5)
        score = procedural_coherence_score([1, 2, 3, 4], tm)
        assert score > -10  # Should be reasonably high for matching sequences

    def test_outcome_accuracy(self):
        pred = np.array([1, 1, 0, 0, 1])
        true = np.array([1, 0, 0, 1, 1])
        result = outcome_prediction_accuracy(pred, true)
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["f1"] <= 1
