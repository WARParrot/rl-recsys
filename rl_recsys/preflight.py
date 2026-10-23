"""Small forward/backward runtime preflight for configured model jobs."""
from __future__ import annotations

import json
import resource
import time
from typing import Any

import torch

from .config import validate_config
from .models import Policy, RewardModel


def run_preflight(config: dict[str, Any], *, device: str | None = None) -> dict[str, Any]:
    validate_config(config)
    requested = torch.device(device) if device else None
    if requested is not None and requested.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU required by --device cuda, but CUDA is unavailable")
    if requested is None:
        if torch.cuda.is_available():
            requested = torch.device("cuda")
        elif not config["train"].get("allow_cpu", False):
            raise RuntimeError("GPU required: CUDA is unavailable and train.allow_cpu is false")
        else:
            requested = torch.device("cpu")
    if requested.type == "cpu" and not config["train"].get("allow_cpu", False):
        raise RuntimeError("GPU required: CPU preflight is disabled because train.allow_cpu is false")

    # Keep the check representative while bounding it independently of production batch sizes.
    batch_size = min(2, int(config["train"]["batch_size"]))
    length = min(8, int(config["data"]["context_length"]))
    vocab_size = int(config["data"]["catalog_size"]) + 2
    context = torch.randint(2, vocab_size, (batch_size, length), device=requested)
    items = torch.randint(2, vocab_size, (batch_size,), device=requested)
    if requested.type == "cuda":
        torch.cuda.reset_peak_memory_stats(requested)
        torch.cuda.synchronize(requested)

    started = time.perf_counter()
    policy = Policy(vocab_size, length, config["model"]).to(requested)
    policy_output = policy(context)
    policy_loss = policy_output.square().mean()
    policy_loss.backward()
    if requested.type == "cuda":
        torch.cuda.synchronize(requested)
    policy_seconds = time.perf_counter() - started

    started = time.perf_counter()
    reward = RewardModel(vocab_size, length, config["model"]).to(requested)
    reward_output = reward(context, items)
    reward_loss = reward_output.square().mean()
    reward_loss.backward()
    if requested.type == "cuda":
        torch.cuda.synchronize(requested)
    reward_seconds = time.perf_counter() - started

    memory: dict[str, Any] = {
        "cpu_max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2),
    }
    if requested.type == "cuda":
        memory.update({
            "gpu_allocated_mb": round(torch.cuda.memory_allocated(requested) / 2**20, 2),
            "gpu_reserved_mb": round(torch.cuda.memory_reserved(requested) / 2**20, 2),
            "gpu_peak_allocated_mb": round(torch.cuda.max_memory_allocated(requested) / 2**20, 2),
        })
    return {
        "status": "ok",
        "device": str(requested),
        "required_gpu": not config["train"].get("allow_cpu", False),
        "smoke_shape": {"batch_size": batch_size, "context_length": length, "catalog_size": vocab_size - 2},
        "timing_seconds": {"policy_forward_backward": policy_seconds, "reward_forward_backward": reward_seconds,
                           "total": policy_seconds + reward_seconds},
        "memory": memory,
        "outputs": {"policy": list(policy_output.shape), "reward": list(reward_output.shape)},
    }


def preflight_json(config: dict[str, Any], *, device: str | None = None) -> str:
    return json.dumps(run_preflight(config, device=device), indent=2)
