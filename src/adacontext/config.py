from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load configuration and resolve project-relative output paths."""

    config_path = Path(path).resolve() if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config = deepcopy(config)
    if overrides:
        config = _deep_update(config, deepcopy(overrides))

    root = config_path.parent.parent if config_path.name == "default.yaml" else PROJECT_ROOT
    config["_project_root"] = str(root)
    for key in ("artifacts_dir", "logs_dir", "reports_dir"):
        value = Path(config["project"][key])
        if not value.is_absolute():
            value = root / value
        config["project"][key] = str(value.resolve())

    model_path = Path(config["adaptive"]["budget_model_path"])
    if not model_path.is_absolute():
        model_path = root / model_path
    config["adaptive"]["budget_model_path"] = str(model_path.resolve())
    return config


def ensure_output_directories(config: dict[str, Any]) -> None:
    for key in ("artifacts_dir", "logs_dir", "reports_dir"):
        Path(config["project"][key]).mkdir(parents=True, exist_ok=True)

