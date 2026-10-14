"""Shared utilities for the invisible-logic v2 pipeline."""
import os
import json
import glob
from pathlib import Path

import yaml

BASE_DIR = Path("/home/whiteowl/last_shot/v2")
V1_DIR = Path("/home/whiteowl/last_shot")


def load_config(config_name: str) -> dict:
    """Load a YAML config from configs/ directory."""
    path = BASE_DIR / "configs" / config_name
    with open(path) as f:
        return yaml.safe_load(f)


def stage_complete(marker_path: str) -> bool:
    """Check if a pipeline stage is already complete."""
    return Path(marker_path).exists()


def mark_complete(marker_path: str):
    """Create a completion marker file."""
    Path(marker_path).parent.mkdir(parents=True, exist_ok=True)
    Path(marker_path).touch()


def find_latest_checkpoint(output_dir: str) -> str | None:
    """Find latest checkpoint-N directory for resume."""
    checkpoints = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda x: int(x.split("-")[-1]),
    )
    return checkpoints[-1] if checkpoints else None


def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file."""
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(data: list[dict], path: str):
    """Save data as JSONL."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
