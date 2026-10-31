"""BatchedChunkDecoder tests.

1. Logic (CPU, fake model): windows from many concurrent streams are grouped by length, capped at
   max_batch, decoded in fewer calls, and every stream gets back exactly its own slice.
2. Numerics (real SNAC, noise off): batched output equals per-window output for the same windows,
   up to fp32 rounding (<= 1e-5). TF32 is switched off for this check: PyTorch's default lets cuDNN run fp32 convolutions in
   TF32 (~1e-3 relative precision), and cuDNN may pick a different algorithm per batch shape, so with TF32
   on, single and batched decodes differ by ~1e-3 (both equally far from a float64 reference).
   Runs on the GPU if available, else on the CPU; skipped if the SNAC weights cannot be loaded.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from speechserve import config as C
from speechserve.codec import BatchedChunkDecoder, Window

SPF = C.SAMPLES_PER_FRAME


class FakeSNAC:
    """Frame k of a window decodes to SPF samples equal to that frame's first code."""

    def __init__(self):
        self._pool = ThreadPoolExecutor(max_workers=1)
        self.batch_sizes = []

    def set_noise(self, enabled):
        pass

    def decode_many(self, frames_list):
        self.batch_sizes.append(len(frames_list))
        assert len({len(f) for f in frames_list}) == 1, "one call must only stack same-length windows"
        return [np.repeat(np.asarray(f)[:, 0].astype(np.float32), SPF) for f in frames_list], 1.0


def window_frames(stream, n_frames):
    """Frames whose first code encodes (stream, frame index) so slices can be checked."""
    return [[stream * 100 + k] + [0] * 6 for k in range(n_frames)]


def test_batched_logic_groups_caps_and_slices():
    fake = FakeSNAC()
    dec = BatchedChunkDecoder(snac=fake, max_batch=4)
    jobs = []
    for s in range(10):                        # 10 streams; 8 have 4-frame windows, 2 have tail windows
        n = 4 if s < 8 else 3
        frames = window_frames(s, n)
        win = Window(0, n, 1, 2) if n == 4 else Window(0, 3, 1, 3)
        jobs.append((s, frames, win))

    async def run():
        return await asyncio.gather(*(dec.decode_window(f, w) for _, f, w in jobs))

    chunks = asyncio.run(run())
    for (s, frames, win), c in zip(jobs, chunks):
        expected = np.repeat(np.array([s * 100 + k for k in range(win.emit_start, win.emit_end)], np.float32), SPF)
        assert np.array_equal(c.pcm, expected), f"stream {s} got someone else's audio"
    assert max(fake.batch_sizes) <= 4
    assert sum(fake.batch_sizes) == 10
    assert len(fake.batch_sizes) < 10           # fewer calls than windows
    st = dec.stats()
    assert st["snac_windows_total"] == 10 and st["snac_calls_total"] == len(fake.batch_sizes)


def test_batched_logic_frame_cap_limits_long_windows():
    fake = FakeSNAC()
    dec = BatchedChunkDecoder(snac=fake, max_batch=64, max_frames=21)   # 7-frame windows -> at most 3 per call
    frames = [window_frames(s, 7) for s in range(7)]

    async def run():
        return await asyncio.gather(*(dec.decode_window(f, Window(0, 7, 1, 5)) for f in frames))

    asyncio.run(run())
    assert max(fake.batch_sizes) <= 3 and sum(fake.batch_sizes) == 7
    assert dec.batch_limit(4) == 5 and dec.batch_limit(30) == 1


def test_batched_logic_cancelled_stream_is_skipped():
    fake = FakeSNAC()
    dec = BatchedChunkDecoder(snac=fake, max_batch=8)

    async def run():
        t1 = asyncio.create_task(dec.decode_window(window_frames(1, 4), Window(0, 4, 1, 2)))
        t2 = asyncio.create_task(dec.decode_window(window_frames(2, 4), Window(0, 4, 1, 2)))
        await asyncio.sleep(0)                  # both queued
        t1.cancel()
        c2 = await t2
        assert c2.pcm[0] == 201
        with pytest.raises(asyncio.CancelledError):
            await t1

    asyncio.run(run())


@pytest.fixture(scope="module")
def snac():
    torch = pytest.importorskip("torch")
    pytest.importorskip("snac")
    from speechserve.codec import SNACDecoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        dec = SNACDecoder(device=device)
    except Exception as e:  # weights not cached / no network
        pytest.skip(f"SNAC not available: {e}")
    dec.set_noise(False)
    tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    yield dec
    torch.backends.cudnn.allow_tf32 = tf32
    dec.set_noise(True)


def test_batched_matches_per_window_noise_off(snac):
    rng = np.random.default_rng(0)
    wins = [rng.integers(0, C.CODEBOOK_SIZE, size=(4, 7)) for _ in range(16)]
    single = [snac.decode(w)[0] for w in wins]
    batched, _ = snac.decode_many(wins)
    max_diff = max(float(np.abs(a - b).max()) for a, b in zip(single, batched))
    assert max_diff <= 1e-5, max_diff  # fp32 rounding only (a slicing bug would be ~signal size, 1e-2)


def test_batched_decoder_end_to_end_matches_per_stream_noise_off(snac):
    """Through decode_window: concurrent streams via the batched worker == one call per window."""
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, C.CODEBOOK_SIZE, size=(6, 7)).tolist() for _ in range(8)]
    win = Window(1, 5, 2, 3)
    dec = BatchedChunkDecoder(snac=snac, max_batch=64)

    async def run():
        return await asyncio.gather(*(dec.decode_window(f, win) for f in frames))

    chunks = asyncio.run(run())
    assert dec.stats()["snac_calls_total"] < len(frames)
    for f, c in zip(frames, chunks):
        ref = snac.decode(f[win.decode_start:win.decode_end])[0][win.keep_samples]
        assert c.pcm.shape == ref.shape
        assert float(np.abs(c.pcm - ref).max()) <= 1e-5
