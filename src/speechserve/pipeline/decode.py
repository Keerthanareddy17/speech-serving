"""Decode stage (+ VAD, timed separately): one parquet row group -> trimmed mono float32 24 kHz clips.

Runs inside CPU worker processes. The unit of work is one parquet row group (~100 clips), so workers read
their own slice of the file (no audio passes through the parent before decoding).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass

import numpy as np

from . import vad as V

TARGET_SR = 24_000


@dataclass
class Clip:
    id: str
    speaker_id: str
    chapter_id: str
    text: str
    audio: np.ndarray        # float32, mono, 24 kHz, trimmed
    orig_s: float            # duration before trimming


def decode_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV bytes -> (mono float32 at 24 kHz, original sample rate); resampled only if not 24 kHz."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]
    if sr != TARGET_SR:
        from math import gcd
        g = gcd(sr, TARGET_SR)
        audio = resample_poly(audio, TARGET_SR // g, sr // g).astype(np.float32)
    return audio, sr


def decode_row_group(task) -> dict:
    """Worker entry point. task = (parquet path, row group index, VadConfig)."""
    import pyarrow.parquet as pq

    path, rg, vad_cfg = task
    t0 = time.perf_counter()
    table = pq.ParquetFile(path).read_row_group(rg, columns=["audio", "text_normalized", "speaker_id",
                                                             "chapter_id", "id"])
    read_s = time.perf_counter() - t0
    clips, dropped, decode_s, vad_s, orig_total, resampled = [], [], 0.0, 0.0, 0.0, 0
    for row in table.to_pylist():
        t = time.perf_counter()
        audio, sr_in = decode_wav(row["audio"]["bytes"])
        resampled += sr_in != TARGET_SR
        decode_s += time.perf_counter() - t
        orig = len(audio) / TARGET_SR
        orig_total += orig
        t = time.perf_counter()
        trimmed = V.trim(audio, TARGET_SR, vad_cfg)
        vad_s += time.perf_counter() - t
        if trimmed is None:
            dropped.append({"id": row["id"], "orig_s": orig})
            continue
        clips.append(Clip(row["id"], row["speaker_id"], row["chapter_id"], row["text_normalized"],
                          np.ascontiguousarray(trimmed, dtype=np.float32), orig))
    return {"clips": clips, "dropped": dropped, "read_s": read_s, "decode_s": decode_s, "vad_s": vad_s,
            "orig_s": orig_total, "resampled": resampled}
