import yaml
from pathlib import Path
from typing import Any, Dict


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    cfg_path = Path(path)
    with cfg_path.open("r") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


