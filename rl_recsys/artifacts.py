"""Atomic evaluation artifacts; only a validated completion marker denotes success.

Hashes detect corruption, not malicious replacement or scientific validity.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np


EVALUATION_FILES = ("results.json", "per_example.npz", "evaluation-provenance.json")
COMPLETION_FILE = "EVALUATION_COMPLETE.json"


@contextmanager
def _atomic_file(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path, value):
    # Serialize before opening the temporary file, rejecting NaN/Inf.
    payload = json.dumps(value, indent=2, allow_nan=False).encode("utf-8")
    with _atomic_file(path) as stream:
        stream.write(payload)


def atomic_npz(path, arrays):
    with _atomic_file(path) as stream:
        # A file handle avoids numpy silently appending '.npz' to the temp name.
        np.savez_compressed(stream, **arrays)


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complete_evaluation(run_dir, results, arrays, provenance):
    """Publish outputs then marker last. Partial/previous evaluations are not overwritten.

    A failure leaves no completion marker. Retained partial outputs intentionally
    block retry, since test data may already have been inspected.
    """
    run = Path(run_dir)
    if any((run / name).exists() for name in (*EVALUATION_FILES, COMPLETION_FILE)):
        raise FileExistsError("Evaluation artifacts already exist; no silent retry/overwrite")
    atomic_npz(run / "per_example.npz", arrays)
    atomic_json(run / "evaluation-provenance.json", provenance)
    atomic_json(run / "results.json", results)
    manifest = {"schema_version": 1, "status": "complete",
                "sha256": {name: _sha256(run / name) for name in EVALUATION_FILES}}
    atomic_json(run / COMPLETION_FILE, manifest)
    return manifest


def validate_evaluation_artifacts(run_dir):
    """Return the completion manifest or raise ValueError for incomplete/corrupt output."""
    run = Path(run_dir)
    try:
        manifest = json.loads((run / COMPLETION_FILE).read_text(encoding="utf-8"))
        if (not isinstance(manifest, dict) or manifest.get("schema_version") != 1
                or manifest.get("status") != "complete"
                or not isinstance(manifest.get("sha256"), dict)
                or set(manifest["sha256"]) != set(EVALUATION_FILES)):
            raise ValueError("Invalid evaluation completion manifest")
        for name in EVALUATION_FILES:
            if _sha256(run / name) != manifest["sha256"][name]:
                raise ValueError(f"Evaluation artifact hash mismatch: {name}")
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Incomplete or unreadable evaluation artifacts: {exc}") from exc
    return manifest
