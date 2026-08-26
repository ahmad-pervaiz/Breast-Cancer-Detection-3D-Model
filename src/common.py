"""Small shared utilities: config loading/merging, seeding, device detection."""
import copy
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: Path, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
    return cfg


def save_config(cfg: Dict[str, Any], out_path: Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def resolve_path(cfg: Dict[str, Any], key: str) -> Path:
    """Config output paths (checkpoint_dir, log_dir, ...) are stored relative to
    the project root; dataset_root is already absolute (points at FINAL DATASET)."""
    value = Path(cfg[key])
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic where practical; some ops have no deterministic CPU/GPU
    # kernel, so we don't set torch.use_deterministic_algorithms(True) (would
    # raise on those ops) - this is a documented, deliberate trade-off.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)}), "
              f"CUDA version: {torch.version.cuda}")
    else:
        device = torch.device("cpu")
        print("Device: CPU (no CUDA GPU detected)")
    return device


def deep_copy_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return copy.deepcopy(cfg)
