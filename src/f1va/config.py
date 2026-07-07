"""Caricamento e validazione della configurazione."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    detection: dict[str, Any] = field(default_factory=dict)
    tracking: dict[str, Any] = field(default_factory=dict)
    homography: dict[str, Any] = field(default_factory=dict)
    speed: dict[str, Any] = field(default_factory=dict)
    strategy: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: data.get(k, {}) for k in cls.__dataclass_fields__})


def load_config(path: str | Path = "configs/default.yaml") -> Config:
    return Config.load(path)
