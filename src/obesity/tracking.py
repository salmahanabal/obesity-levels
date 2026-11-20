"""Aides MLflow : configuration, tags communs."""
from __future__ import annotations

import subprocess

import mlflow

from obesity import config as cfg


def setup(experiment: str) -> None:
    """Configure le backend SQLite et l'experience."""
    mlflow.set_tracking_uri(cfg.load_config()["mlflow"]["tracking_uri"])
    mlflow.set_experiment(experiment)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cfg.ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def common_tags(dataset_hash: str) -> dict[str, str]:
    return {
        "git_commit": git_commit(),
        "dataset_hash": dataset_hash,
        "split_seed": str(cfg.SEED),
    }
