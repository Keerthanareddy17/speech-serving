"""Downloads LibriTTS-R train-clean-100 (24 kHz, CC BY 4.0) to LOCAL disk (never Drive, never git).

Source: the Hugging Face parquet mirror blabble-io/libritts_r, split train.clean.100 (18 parquet files,
9.24 GB, 33,232 clips; each row has the WAV bytes, normalised + original text, speaker, chapter, id).
Fallback: OpenSLR SLR141 (one 8.1 GB tar). Local disk because Drive's FUSE mount is slow for heavy reads
and would hide the pipeline's real bottleneck.
"""

from __future__ import annotations

import time
from pathlib import Path

REPO = "blabble-io/libritts_r"
SPLIT_DIR = "data/train.clean.100"
LOCAL_DIR = Path("/content/data/libritts_r")


def download(max_files: int | None = None, local_dir: Path = LOCAL_DIR) -> list[Path]:
    """Fetch the split's parquet files (optionally only the first `max_files`). Returns the local paths."""
    from huggingface_hub import HfApi, snapshot_download

    files = sorted(f.path for f in HfApi().list_repo_tree(REPO, repo_type="dataset", path_in_repo=SPLIT_DIR))
    files = files[:max_files] if max_files else files
    t0 = time.time()
    snapshot_download(REPO, repo_type="dataset", allow_patterns=files, local_dir=str(local_dir), max_workers=8)
    print(f"downloaded {len(files)} files in {time.time() - t0:.0f} s")
    return [local_dir / f for f in files]


def local_files(local_dir: Path = LOCAL_DIR) -> list[Path]:
    return sorted((local_dir / SPLIT_DIR).glob("*.parquet"))
