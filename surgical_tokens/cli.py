"""CLI entry point for the surgical tokens pipeline.

Usage:
    surtok generate-synthetic --output-dir ./data/synthetic
    surtok extract --video-dir ./videos --config configs/default.yaml
    surtok build-vocab --embeddings ./outputs/embeddings/clip_embeddings.pt
    surtok train --case-log ./data/case_log.csv --tokens ./outputs/codebook/tokenized_clips.pt
    surtok evaluate --model ./outputs/models/dpo/final --embeddings ./outputs/embeddings/clip_embeddings.pt
    surtok info --config configs/default.yaml
"""

from __future__ import annotations

import json
import logging
import sys

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(name="surtok", help="Surgical Tokens: clip-token sequence learning pipeline")
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


@app.command()
def generate_synthetic(
    output_dir: str = typer.Option("./data/synthetic", help="Output directory"),
    n_cases: int = typer.Option(50, help="Number of synthetic cases"),
    generate_videos: bool = typer.Option(False, help="Generate actual video files (slow)"),
):
    """Generate synthetic dataset for pipeline testing."""
    from surgical_tokens.data.synthetic import generate_full_synthetic_dataset

    console.print("[bold blue]Generating synthetic dataset...[/bold blue]")
    result = generate_full_synthetic_dataset(
        output_dir=output_dir,
        n_cases=n_cases,
        generate_videos=generate_videos,
    )
    console.print("[bold green]Done![/bold green]")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@app.command()
def extract(
    video_dir: str = typer.Option(..., help="Directory containing raw surgical videos"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
):
    """Run video → clips → embeddings extraction pipeline."""
    from surgical_tokens.pipeline.extract_embeddings import run_extraction_pipeline

    result = run_extraction_pipeline(video_dir=video_dir, config_path=config)
    console.print("[bold green]Extraction complete![/bold green]")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@app.command()
def build_vocab(
    embeddings: str = typer.Option(..., help="Path to clip_embeddings.pt"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
):
    """Train VQ-VAE and build procedural vocabulary."""
    from surgical_tokens.pipeline.build_vocabulary import run_vocabulary_pipeline

    result = run_vocabulary_pipeline(embeddings_path=embeddings, config_path=config)
    console.print("[bold green]Vocabulary built![/bold green]")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@app.command()
def train(
    case_log: str = typer.Option(..., help="Path to case_log.csv"),
    tokens: str = typer.Option(..., help="Path to tokenized_clips.pt"),
    clip_mapping: str = typer.Option(..., help="Path to clip_mapping.json"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
    skip_pretrain: bool = typer.Option(False, help="Skip pretraining (use existing model)"),
    pretrained_model: str = typer.Option(None, help="Path to pretrained model (if skipping)"),
):
    """Run pretraining + DPO training pipeline."""
    from surgical_tokens.pipeline.train_pipeline import run_training_pipeline

    result = run_training_pipeline(
        case_log_path=case_log,
        tokens_path=tokens,
        clip_mapping_path=clip_mapping,
        config_path=config,
        skip_pretrain=skip_pretrain,
        pretrained_model_path=pretrained_model,
    )
    console.print("[bold green]Training complete![/bold green]")
    for k, v in result.items():
        console.print(f"  {k}: {v}")


@app.command()
def evaluate(
    model_path: str = typer.Option(..., help="Path to trained model"),
    embeddings: str = typer.Option(None, help="Path to clip_embeddings.pt (for clustering eval)"),
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
):
    """Run evaluation suite on trained model."""
    import numpy as np
    import torch

    from surgical_tokens.config import get_paths, load_config
    from surgical_tokens.evaluation.clustering import (
        evaluate_clustering,
        plot_codebook_utilization,
        plot_embedding_space,
    )

    cfg = load_config(config)
    paths = get_paths(cfg)

    if embeddings:
        console.print("[bold blue]Running clustering evaluation...[/bold blue]")
        data = torch.load(embeddings, weights_only=True)
        emb = data["embeddings"]
        labels = data.get("phase_labels", None)

        results = evaluate_clustering(
            emb.numpy(),
            n_clusters_list=cfg["evaluation"]["clustering"]["n_clusters"],
            true_labels=labels,
        )

        table = Table(title="Clustering Results")
        table.add_column("k")
        table.add_column("Silhouette")
        table.add_column("Calinski-Harabasz")
        table.add_column("Davies-Bouldin")

        for k, m in results.items():
            table.add_row(
                str(k),
                f"{m['silhouette']:.3f}",
                f"{m['calinski_harabasz']:.1f}",
                f"{m['davies_bouldin']:.3f}",
            )
        console.print(table)

        plot_embedding_space(
            emb.numpy(),
            labels=labels,
            output_path=str(paths.eval_dir / "embedding_space.png"),
        )
        console.print(f"Saved embedding visualization to {paths.eval_dir}/embedding_space.png")


@app.command()
def info(
    config: str = typer.Option("configs/default.yaml", help="Config file path"),
):
    """Show project info and memory estimates."""
    from surgical_tokens.config import load_config
    from surgical_tokens.training.distributed import estimate_memory_requirements, setup_distributed

    cfg = load_config(config)
    env = setup_distributed()
    mem = estimate_memory_requirements(cfg)

    table = Table(title="Surgical Tokens — System Info")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("GPUs", f"{env['n_gpus']} ({', '.join(env['gpu_names']) or 'none'})")
    table.add_row("Total VRAM", f"{env['total_memory_gb']} GB")
    table.add_row("BF16 Support", str(env["bf16_supported"]))
    table.add_row("---", "---")
    table.add_row("Sequence Model Params", f"{mem['n_params_millions']}M")
    table.add_row("Estimated Training VRAM", f"{mem['total_estimated_gb']} GB")
    table.add_row("Recommended GPUs", str(mem["recommended_gpus"]))
    table.add_row("---", "---")
    table.add_row("Codebook Size", str(cfg["sparse_encoder"]["codebook_size"]))
    table.add_row("Clip Duration", f"{cfg['segmentation']['clip_duration_sec']}s")
    table.add_row("DPO Beta", str(cfg["dpo"]["beta"]))

    console.print(table)


if __name__ == "__main__":
    app()
