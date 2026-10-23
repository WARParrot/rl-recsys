import json
from pathlib import Path

import pytest

from rl_recsys import cli
from rl_recsys.config import ConfigValidationError


@pytest.mark.parametrize("command", ["prepare", "train", "evaluate"])
def test_invalid_config_stops_before_expensive_work(tmp_path, monkeypatch, command):
    cfg = json.loads(Path("configs/pilot.json").read_text())
    cfg["train"]["sft_steps"] = 0
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    (tmp_path / "provenance.json").write_text(json.dumps({"config": cfg}))

    def unexpected(*args, **kwargs):
        pytest.fail("Expensive work started before config validation")

    monkeypatch.setattr(cli, "prepare", unexpected)
    monkeypatch.setattr(cli, "train", unexpected)
    monkeypatch.setattr(cli, "final_evaluation", unexpected)
    argv = [command]
    if command != "evaluate":
        argv += ["--config", str(config_path)]
    if command != "prepare":
        argv += ["--run", str(tmp_path)]
    with pytest.raises(ConfigValidationError):
        cli.main(argv)
