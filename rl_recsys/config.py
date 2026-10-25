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


def validate_temporal_config(data: Mapping) -> None:
    """Shared by full config validation and direct array preparation."""
    if not isinstance(data, Mapping):
        raise ConfigValidationError("data must be an object")
    legacy = data.get("allow_derived_cutoffs", False)
    if not isinstance(legacy, bool):
        raise ConfigValidationError("data.allow_derived_cutoffs must be boolean")
    cutoffs = data.get("cutoffs")
    if cutoffs is not None:
        if (not isinstance(cutoffs, (list, tuple)) or len(cutoffs) != 3
                or any(isinstance(x, bool) or not isinstance(x, int)
                       or not 0 <= x <= 2**63 - 1 for x in cutoffs)
                or any(a >= b for a, b in zip(cutoffs, cutoffs[1:]))):
            raise ConfigValidationError("data.cutoffs must contain three strictly increasing nonnegative int64 timestamps")
    elif not legacy:
        raise ConfigValidationError("data.cutoffs is required; allow_derived_cutoffs is only for legacy exploratory runs")
    quantiles = data.get("quantiles")
    if quantiles is not None or cutoffs is None:
        if (not isinstance(quantiles, (list, tuple)) or len(quantiles) != 3
                or any(isinstance(q, bool) or not isinstance(q, (int, float))
                       or not math.isfinite(q) or not 0 < q < 1 for q in quantiles)
                or any(a >= b for a, b in zip(quantiles, quantiles[1:]))):
            raise ConfigValidationError("data.quantiles must contain three strictly increasing numbers between 0 and 1")


def validate_config(config: Mapping) -> Mapping:
    """Validate invariants that otherwise fail only after expensive work starts."""
    errors: list[str] = []
    if not isinstance(config, Mapping):
        raise ConfigValidationError("configuration must be an object")

    if "sample_seed" in config:
        value = config["sample_seed"]
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**32:
            errors.append("sample_seed must be a nonnegative 32-bit integer")

    for path in ("train.sft_steps", "train.reward_steps", "train.rl_steps",
                 "train.batch_size", "evaluation.batch_size", "data.catalog_size", "evaluation.k"):
        _positive(config, path, errors)
    _positive(config, "rl.seconds_cap", errors)
    try:
        threshold = config.get("reward_audit", {}).get("min_relative_improvement", 0.0)
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold < 1:
            errors.append("reward_audit.min_relative_improvement must be in [0, 1)")
    except AttributeError:
        errors.append("reward_audit must be an object")

    try:
        validate_temporal_config(config["data"])
    except (KeyError, TypeError, ConfigValidationError) as exc:
        errors.append(str(exc))

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
