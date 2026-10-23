"""Configuration validation shared by command-line entry points."""
from __future__ import annotations

import math
from collections.abc import Mapping


class ConfigValidationError(ValueError):
    """Raised when a run configuration cannot be executed safely."""


def _positive(cfg: Mapping, path: str, errors: list[str]) -> None:
    parts = path.split(".")
    value = cfg
    try:
        for part in parts:
            value = value[part]
    except (KeyError, TypeError):
        errors.append(f"{path} is required and must be positive")
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        errors.append(f"{path} must be positive")


def validate_config(config: Mapping) -> Mapping:
    """Validate invariants that otherwise fail only after expensive work starts."""
    errors: list[str] = []
    if not isinstance(config, Mapping):
        raise ConfigValidationError("configuration must be an object")

    for path in ("train.sft_steps", "train.reward_steps", "train.rl_steps",
                 "train.batch_size", "evaluation.batch_size", "data.catalog_size", "evaluation.k"):
        _positive(config, path, errors)
    _positive(config, "rl.seconds_cap", errors)

    try:
        quantiles = config["data"]["quantiles"]
        if not quantiles or any(isinstance(q, bool) or not isinstance(q, (int, float))
                                or not math.isfinite(q) or not 0 < q < 1 for q in quantiles):
            errors.append("data.quantiles must be nonempty numbers strictly between 0 and 1")
        elif list(quantiles) != sorted(set(quantiles)):
            errors.append("data.quantiles must be strictly increasing and unique")
    except (KeyError, TypeError):
        errors.append("data.quantiles must be nonempty")

    try:
        temperature = config["rl"]["sampling_temperature"]
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            errors.append("rl.sampling_temperature must be positive")
    except (KeyError, TypeError):
        errors.append("rl.sampling_temperature is required and must be positive")

    for key in ("rewards", "horizons", "betas"):
        try:
            values = config["rl"][key]
            if not values:
                errors.append(f"rl.{key} must be nonempty")
        except (KeyError, TypeError):
            errors.append(f"rl.{key} must be nonempty")
    for key in ("horizons", "betas"):
        try:
            values = config["rl"][key]
            if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
                errors.append(f"rl.{key} must contain only positive finite numbers")
        except (KeyError, TypeError):
            pass

    try:
        rewards = config["rl"]["rewards"]
        if any(not isinstance(v, str) or not v.strip() for v in rewards):
            errors.append("rl.rewards must contain nonempty names")
    except (KeyError, TypeError):
        pass

    # Candidate names can be supplied explicitly; the generated RL grid must also be collision-free.
    try:
        explicit = config.get("candidates", [])
        names = [c["name"] if isinstance(c, Mapping) else c for c in explicit]
        names += [f"rl_{r}_h{h}_b{b:g}" for r in config["rl"]["rewards"]
                  for h in config["rl"]["horizons"] for b in config["rl"]["betas"]]
        if len(names) != len(set(names)):
            errors.append("candidate names must be unique")
    except (KeyError, TypeError, ValueError):
        errors.append("candidate names must be unique")

    if errors:
        raise ConfigValidationError("Invalid configuration: " + "; ".join(errors))
    return config
