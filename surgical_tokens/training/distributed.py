"""Multi-GPU training utilities using Accelerate.

Provides helpers for setting up distributed training and
handling device placement across multiple GPUs.
"""

from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)


def setup_distributed() -> dict:
    """Detect and configure distributed training environment.

    Returns:
        Dict with training environment info.
    """
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_names = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        total_memory_gb = sum(
            torch.cuda.get_device_properties(i).total_mem / 1e9
            for i in range(n_gpus)
        )
    else:
        n_gpus = 0
        gpu_names = []
        total_memory_gb = 0

    env = {
        "n_gpus": n_gpus,
        "gpu_names": gpu_names,
        "total_memory_gb": round(total_memory_gb, 1),
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }

    logger.info(
        f"Distributed env: {n_gpus} GPUs, "
        f"{total_memory_gb:.1f}GB total VRAM, "
        f"bf16={'yes' if env['bf16_supported'] else 'no'}"
    )

    return env


def get_accelerate_config(cfg: dict) -> dict:
    """Generate accelerate config from project config.

    Returns a dict suitable for writing to accelerate_config.yaml.
    """
    env = setup_distributed()

    config = {
        "compute_environment": "LOCAL_MACHINE",
        "distributed_type": "MULTI_GPU" if env["n_gpus"] > 1 else "NO",
        "num_processes": max(env["n_gpus"], 1),
        "mixed_precision": cfg["infra"]["dtype"] if cfg["infra"]["dtype"] != "fp32" else "no",
        "gradient_accumulation_steps": cfg["sequence_model"]["pretraining"]["gradient_accumulation_steps"],
    }

    if env["n_gpus"] > 1:
        config["deepspeed_config"] = {
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {"device": "none"},
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "overlap_comm": True,
            },
            "bf16": {"enabled": cfg["infra"]["dtype"] == "bf16"},
            "fp16": {"enabled": cfg["infra"]["dtype"] == "fp16"},
            "train_micro_batch_size_per_gpu": cfg["sequence_model"]["pretraining"]["batch_size"],
            "gradient_accumulation_steps": cfg["sequence_model"]["pretraining"]["gradient_accumulation_steps"],
        }

    return config


def estimate_memory_requirements(cfg: dict) -> dict:
    """Estimate VRAM requirements for each training stage."""
    sm_cfg = cfg["sequence_model"]

    # Rough parameter count estimate for GPT2-style model
    n_params = (
        sm_cfg["n_embd"] * (sm_cfg["vocab_size"] + 3)  # Embedding
        + sm_cfg["n_layer"] * (
            4 * sm_cfg["n_embd"] ** 2  # Attention
            + 8 * sm_cfg["n_embd"] ** 2  # FFN
        )
    )

    # Memory estimates (bytes)
    param_bytes = n_params * (2 if cfg["infra"]["dtype"] != "fp32" else 4)
    grad_bytes = param_bytes
    optimizer_bytes = param_bytes * 2  # Adam states
    activation_bytes = (
        sm_cfg["pretraining"]["batch_size"]
        * sm_cfg["n_positions"]
        * sm_cfg["n_embd"]
        * sm_cfg["n_layer"]
        * 2  # bf16
    )

    total_gb = (param_bytes + grad_bytes + optimizer_bytes + activation_bytes) / 1e9

    return {
        "n_params_millions": round(n_params / 1e6, 1),
        "param_memory_gb": round(param_bytes / 1e9, 2),
        "total_estimated_gb": round(total_gb, 2),
        "recommended_gpus": max(1, int(total_gb / 20) + 1),  # ~20GB usable per GPU
    }
