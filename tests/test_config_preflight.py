import copy
import json
from pathlib import Path

import pytest

from rl_recsys.config import ConfigValidationError, validate_config
from rl_recsys.preflight import run_preflight


@pytest.fixture
def config():
    return json.loads(Path("configs/pilot.json").read_text())


def test_validate_config_accepts_pilot(config):
    assert validate_config(config) is config


@pytest.mark.parametrize(("path", "value"), [
    (("train", "sft_steps"), 0),
    (("evaluation", "k"), 0),
    (("data", "catalog_size"), -1),
    (("rl", "seconds_cap"), 0),
    (("rl", "sampling_temperature"), 0),
])
def test_validate_config_rejects_nonpositive(config, path, value):
    bad = copy.deepcopy(config)
    bad[path[0]][path[1]] = value
    with pytest.raises(ConfigValidationError, match="Invalid configuration"):
        validate_config(bad)


def test_validate_config_rejects_bad_quantiles_and_empty_grid(config):
    bad = copy.deepcopy(config)
    bad["data"]["quantiles"] = [0.8, 0.8]
    bad["rl"]["rewards"] = []
    bad["rl"]["betas"] = []
    with pytest.raises(ConfigValidationError):
        validate_config(bad)


def test_validate_config_rejects_duplicate_candidate_names(config):
    bad = copy.deepcopy(config)
    bad["candidates"] = [{"name": "same"}, {"name": "same"}]
    with pytest.raises(ConfigValidationError, match="candidate names"):
        validate_config(bad)


def test_cpu_preflight_reports_timing_and_memory(config):
    cfg = copy.deepcopy(config)
    cfg["data"]["catalog_size"] = 8
    cfg["data"]["context_length"] = 4
    cfg["train"]["batch_size"] = 2
    cfg["train"]["allow_cpu"] = True
    cfg["model"].update(hidden=8, layers=1, heads=2)
    result = run_preflight(cfg, device="cpu")
    assert result["status"] == "ok"
    assert result["device"] == "cpu"
    assert result["timing_seconds"]["total"] > 0
    assert result["memory"]["cpu_max_rss_mb"] > 0
    assert result["outputs"] == {"policy": [2, 8], "reward": [2, 2]}


def test_preflight_fails_clearly_when_gpu_required(config):
    cfg = copy.deepcopy(config)
    cfg["train"]["allow_cpu"] = False
    with pytest.raises(RuntimeError, match="GPU required"):
        run_preflight(cfg, device="cpu")
