from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "tmmkg.yaml"


def project_path(path: str | Path | None) -> Path | None:
    if path is None:
        return None

    resolved = Path(path)
    if resolved.is_absolute():
        return resolved

    return PROJECT_ROOT / resolved


def load_config(config_path: str | Path | None = None) -> dict:
    path = project_path(config_path) if config_path else DEFAULT_CONFIG_PATH

    with open(path, "r") as f:
        return yaml.safe_load(f) or {}
