"""Chargement de config.yaml et resolution des chemins (aucun chemin absolu en dur)."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """Lit config.yaml a la racine du projet."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(key: str) -> Path:
    """Retourne le chemin absolu associe a une cle de `paths`."""
    return ROOT / load_config()["paths"][key]


SEED: int = load_config()["seed"]
