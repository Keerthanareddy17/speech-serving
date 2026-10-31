"""Tokenize stage: batched SNAC encode on the GPU, with length bucketing, pinned memory and prefetching.

Clips arrive from the CPU workers into a buffer. When the buffer is full (or input ends), it is sorted by
length and cut into batches of similar length; each batch is zero-padded to its longest clip (rounded up
to SNAC's 2048-sample frame), encoded in one call, and each clip's codes are trimmed back to its own
ceil(len / 2048) frames. One frame = 2048 samples = 85.3 ms = 7 Orpheus tokens.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

import numpy as np

FRAME = 2048


@dataclass
class TokenizeConfig:
    batch_clips: int | None = 16          # fixed clips per batch (baseline); None = use max_batch_samples
    max_batch_samples: int = 2_400_000    # padded samples per batch when batch_clips is None (~100 s audio)
    bucket_buffer: int = 512              # clips collected before sorting into length buckets
    pin_memory: bool = False
    prefetch: bool = False                # pad + pin + copy the next batch on a side thread
    # torch.compile the encoder's conv stack (step 7 fix): ~60% of encode time is memory-bound
    # elementwise work (Snake activations, residual adds) that compilation fuses; ~1.25x faster.
    # (Compiling SNAC.encode itself hits a dynamo guard bug in its padding code, so only .encoder.)
    compile: bool = False


def make_batches(clips: list, cfg: TokenizeConfig) -> list[list]:
    """Sort by length and cut into batches (fixed count, or by padded-sample budget)."""
    clips = sorted(clips, key=lambda c: len(c.audio))
    batches, cur = [], []
    for c in clips:
        if cfg.batch_clips is not None:
            if len(cur) == cfg.batch_clips:
                batches.append(cur)
                cur = []
        else:
            padded = -(-len(c.audio) // FRAME) * FRAME          # longest so far (sorted ascending)
            if cur and padded * (len(cur) + 1) > cfg.max_batch_samples:
                batches.append(cur)
                cur = []
        cur.append(c)
    if cur:
        batches.append(cur)
    return batches


class Encoder:
    def __init__(self, cfg: TokenizeConfig, device: str = "cuda"):
        import torch
        from snac import SNAC

        from .. import config as C
        self.torch, self.cfg, self.device = torch, cfg, device
        self.model = SNAC.from_pretrained(C.SNAC_MODEL).eval().to(device)
        self.encoder = torch.compile(self.model.encoder, dynamic=True) if cfg.compile else self.model.encoder
        self.gpu_s = 0.0          # time inside encode calls (incl. waiting for the GPU)
        self.prep_s = 0.0         # padding + host->device copy
        self.padded_samples = self.real_samples = 0
        self.batches = 0

    def _prep(self, batch):
        torch = self.torch
        t = time.perf_counter()
        n = max(len(c.audio) for c in batch)
        n = -(-n // FRAME) * FRAME
        x = np.zeros((len(batch), 1, n), dtype=np.float32)
        for i, c in enumerate(batch):
            x[i, 0, : len(c.audio)] = c.audio
        xt = torch.from_numpy(x)
        if self.cfg.pin_memory:
            xt = xt.pin_memory()
        xt = xt.to(self.device, non_blocking=self.cfg.pin_memory)
        self.prep_s += time.perf_counter() - t
        self.padded_samples += x.size
        self.real_samples += sum(len(c.audio) for c in batch)
        return batch, xt

    def _encode(self, batch, xt):
        torch = self.torch
        t = time.perf_counter()
        with torch.inference_mode():
            z = self.encoder(self.model.preprocess(xt))       # same as SNAC.encode, encoder possibly compiled
            _, codes = self.model.quantizer(z)
        codes = [c.cpu().numpy() for c in codes]               # waits for the GPU
        self.gpu_s += time.perf_counter() - t
        self.batches += 1
        out = []
        for i, c in enumerate(batch):
            nf = -(-len(c.audio) // FRAME)
            out.append((c, codes[0][i, :nf], codes[1][i, : 2 * nf], codes[2][i, : 4 * nf]))
        return out

    def encode_batches(self, batches):
        """Yield (clip, l0, l1, l2) for every clip. With prefetch, batch k+1 is padded and copied to the
        GPU on a side thread while batch k encodes."""
        if not self.cfg.prefetch:
            for b in batches:
                yield from self._encode(*self._prep(b))
            return
        q: queue.Queue = queue.Queue(maxsize=2)

        def producer():
            for b in batches:
                q.put(self._prep(b))
            q.put(None)

        th = threading.Thread(target=producer, daemon=True)
        th.start()
        while (item := q.get()) is not None:
            yield from self._encode(*item)
        th.join()
