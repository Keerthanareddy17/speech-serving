"""Voice activity detection: trims leading/trailing silence and drops clips that are too short.

LibriTTS-R is already cut into sentence-length clips, so no splitting: webrtcvad (aggressiveness 2, 30 ms
frames) runs on a 16 kHz int16 copy (detection only; the 24 kHz audio is what gets trimmed). Speech starts
at the first run of `min_run` voiced frames and ends after the last such run; `pad_ms` is kept on each side.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VAD_SR = 16_000
FRAME_MS = 30


@dataclass
class VadConfig:
    aggressiveness: int = 2
    min_run: int = 3          # consecutive voiced 30 ms frames that count as speech
    pad_ms: int = 100         # silence kept before the first / after the last speech frame
    min_speech_s: float = 0.5 # clips shorter than this after trimming are dropped


_VADS: dict[int, object] = {}


def _vad(aggr: int):
    import webrtcvad
    if aggr not in _VADS:
        _VADS[aggr] = webrtcvad.Vad(aggr)
    return _VADS[aggr]


def speech_bounds(audio: np.ndarray, sr: int, cfg: VadConfig) -> tuple[int, int] | None:
    """(start, end) sample indices of the speech region in `audio` (at `sr`), or None if no speech."""
    from scipy.signal import resample_poly

    x16 = audio if sr == VAD_SR else resample_poly(audio, VAD_SR, sr)
    pcm = (np.clip(x16, -1, 1) * 32767).astype("<i2")
    flen = VAD_SR * FRAME_MS // 1000
    n = len(pcm) // flen
    if n == 0:
        return None
    vad = _vad(cfg.aggressiveness)
    raw = pcm[: n * flen].tobytes()
    voiced = np.fromiter((vad.is_speech(raw[2 * i * flen: 2 * (i + 1) * flen], VAD_SR) for i in range(n)),
                         dtype=bool, count=n)
    # frames that are part of a run of >= min_run voiced frames
    run = np.convolve(voiced.astype(int), np.ones(cfg.min_run, int), "valid") == cfg.min_run
    starts = np.flatnonzero(run)
    if len(starts) == 0:
        return None
    first, last = starts[0], starts[-1] + cfg.min_run          # frame indices [first, last)
    pad = int(cfg.pad_ms / FRAME_MS)
    first, last = max(0, first - pad), min(n, last + pad)
    scale = sr / VAD_SR
    return int(first * flen * scale), min(len(audio), int(last * flen * scale))


def trim(audio: np.ndarray, sr: int, cfg: VadConfig) -> np.ndarray | None:
    """Trimmed audio, or None if it has no speech or is shorter than `min_speech_s` after trimming."""
    b = speech_bounds(audio, sr, cfg)
    if b is None or (b[1] - b[0]) / sr < cfg.min_speech_s:
        return None
    return audio[b[0]: b[1]]
