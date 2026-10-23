import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from rl_recsys.config import ConfigValidationError, validate_config
from rl_recsys import preflight
from rl_recsys.preflight import run_preflight


@pytest.fixture
def config():
    cfg = json.loads(Path("configs/pilot.json").read_text())
    cfg["data"]["cutoffs"] = [100, 200, 300]
    return cfg


def test_validate_config_accepts_pilot_with_explicit_cutoffs(config):
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


@pytest.mark.parametrize('cutoffs', [[], [1, 2], [1, 2, 3, 4], [1, 1, 3],
                                      [3, 2, 1], [-1, 2, 3], [True, 2, 3],
                                      [1., 2, 3], [1, 2, 2**63], '123'])
def test_validate_config_rejects_invalid_calendar_cutoffs(config, cutoffs):
    config['data']['cutoffs'] = cutoffs
    with pytest.raises(ConfigValidationError, match='cutoffs'):
        validate_config(config)


def test_legacy_quantiles_require_explicit_opt_in(config):
    config['data'].pop('cutoffs')
    config['data']['allow_derived_cutoffs'] = True
    assert validate_config(config) is config


def test_validate_config_rejects_missing_frozen_cutoffs(config):
    bad = copy.deepcopy(config)
    bad["data"].pop("cutoffs", None)
    bad["data"]["allow_derived_cutoffs"] = False
    with pytest.raises(ConfigValidationError, match="cutoffs"):
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


@pytest.mark.parametrize("resource_available", [True, False])
def test_cpu_preflight_reports_timing_and_memory(config, monkeypatch, resource_available):
    if not resource_available:
        monkeypatch.setattr(preflight, "resource", None)
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
    rss = result["memory"]["cpu_max_rss_mb"]
    assert rss is None or rss >= 0
    if not resource_available:
        assert rss is None
    assert json.loads(json.dumps(result, allow_nan=False)) == result
    assert result["outputs"] == {"policy": [2, 8], "reward": [2, 2]}


@pytest.mark.parametrize(("platform", "usage", "expected"), [
    ("linux", 2048, 2.0),
    ("darwin", 2 * 2**20, 2.0),
    ("linux", 0, 0.0),
    ("linux", -1, None),
    ("linux", float("nan"), None),
    ("linux", float("inf"), None),
    ("linux", None, None),
])
def test_cpu_rss_normalization(monkeypatch, platform, usage, expected):
    monkeypatch.setattr(preflight.sys, "platform", platform)
    monkeypatch.setattr(preflight, "resource", SimpleNamespace(
        RUSAGE_SELF=0, getrusage=lambda _: SimpleNamespace(ru_maxrss=usage)))
    assert preflight._cpu_max_rss_mb() == expected


def test_cpu_rss_unavailable(monkeypatch):
    monkeypatch.setattr(preflight, "resource", None)
    assert preflight._cpu_max_rss_mb() is None


@pytest.mark.parametrize("error", [OSError, AttributeError])
def test_cpu_rss_query_failure(monkeypatch, error):
    def unavailable(_):
        raise error("RSS unavailable")

    monkeypatch.setattr(preflight, "resource", SimpleNamespace(RUSAGE_SELF=0, getrusage=unavailable))
    assert preflight._cpu_max_rss_mb() is None


def test_preflight_fails_clearly_when_gpu_required(config):
    cfg = copy.deepcopy(config)
    cfg["train"]["allow_cpu"] = False
    with pytest.raises(RuntimeError, match="GPU required"):
        run_preflight(cfg, device="cpu")
