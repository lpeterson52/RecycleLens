#!/usr/bin/env python3
"""Populate the mounted model volume.

Usage (environment variables):
- MODEL_PATH: full destination path for the model (preferred)
- MODEL_DIR: destination directory (joined with default model name)
- LOCAL_MODEL_PATH: path to a local model file inside the container to copy
- MODEL_URL: URL to download the model from (used if LOCAL_MODEL_PATH not provided)

Examples:
  MODEL_DIR=/data/models LOCAL_MODEL_PATH=/tmp/y.pt python scripts/populate_model.py
  MODEL_DIR=/data/models MODEL_URL=https://.../y.pt python scripts/populate_model.py
"""
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import requests

MODEL_NAME = "yolov8n-waste-12cls-best.pt"


def get_dest_path() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    env_dir = os.environ.get("MODEL_DIR")
    if env_path:
        return Path(env_path)
    if env_dir:
        return Path(env_dir) / MODEL_NAME
    # default to app/models (same as repo layout)
    return Path(__file__).resolve().parents[1] / "app" / "models" / MODEL_NAME


def copy_local(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def download(url: str, dest: Path, timeout: int = 60) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def main() -> int:
    dest = get_dest_path()
    print(f"Destination model path: {dest}")

    local = os.environ.get("LOCAL_MODEL_PATH")
    url = os.environ.get("MODEL_URL")

    if local:
        src = Path(local)
        if not src.exists():
            print(f"LOCAL_MODEL_PATH set but file does not exist: {src}", file=sys.stderr)
            return 2
        print(f"Copying local model {src} -> {dest}")
        copy_local(src, dest)
        print("Copy complete")
        return 0

    if url:
        print(f"Downloading model from {url} -> {dest}")
        try:
            download(url, dest)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return 3
        print("Download complete")
        return 0

    print("Neither LOCAL_MODEL_PATH nor MODEL_URL set. Nothing to do.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
