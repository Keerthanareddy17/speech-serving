"""SNAC codec wrapper: turns streamed Orpheus audio token ids into PCM chunks as soon as enough frames exist.

Three layers, from pure logic to GPU:

1. `FrameAssembler` (CPU, no torch): raw vLLM ids -> 7-code SNAC frames. Ids below `<custom_token_10>`
   (e.g. the end-of-speech stop token) are skipped; position i among audio tokens maps to
   code = id - 128266 - (i % 7) * 4096 (upstream decoder.py `turn_token_into_id`).

2. `ChunkPlanner` (CPU, no torch): decides *when* to call SNAC and *which samples to keep*.
   SNAC's decoder is a stack of convolutions, so audio near the edge of a short decode is wrong (the
   convolutions see zero padding instead of the neighbouring frames). Upstream streaming
   (canopyai/Orpheus-TTS @ e64661fe6d, orpheus_tts_pypi/orpheus_tts/decoder.py, `tokens_decoder` +
   `convert_to_audio`) handles this with a sliding window: every time a new frame completes and at least
   4 frames exist, decode the last 4 frames (28 tokens) and keep samples [2048:4096], i.e. only the
   second frame. Each emitted frame therefore had 1 frame of real left context and 2 of right context.
   Generalised here: to emit frames [a, b) we decode frames [a - left, b + right) and keep the middle.
   `first_chunk_frames` sets b - a for the first chunk (bigger = later first audio, but more audio
   buffered up front); `chunk_frames` for every later chunk. Defaults reproduce upstream exactly,
   including never emitting frame 0. One deliberate difference: at end of stream we flush the last
   frames with whatever right context exists (upstream drops the final 2 frames, ~171 ms).

3. `SNACDecoder` (GPU): runs SNAC decode calls on one dedicated worker thread, so the asyncio event
   loop (and the vLLM output handler on it) is not blocked while the GPU decodes, and times each call.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from . import config as C


# ---- 1. ids -> frames --------------------------------------------------------------------------------
def token_to_code(token_id: int, position: int) -> int:
    """Audio token id at `position` (index among audio tokens) -> SNAC code in [0, 4096)."""
    code = token_id - C.AUDIO_TOKEN_BASE - (position % C.TOKENS_PER_FRAME) * C.CODEBOOK_SIZE
    if not 0 <= code < C.CODEBOOK_SIZE:
        raise ValueError(f"audio token {token_id} out of range for frame position {position % 7}")
    return code


class FrameAssembler:
    def __init__(self):
        self.frames: list[list[int]] = []
        self._partial: list[int] = []
        self.n_audio_tokens = 0
        self.n_tokens = 0          # every id pushed, incl. non-audio ones

    def push(self, token_ids: Iterable[int]) -> int:
        """Add new ids; returns the number of complete frames so far."""
        for t in token_ids:
            self.n_tokens += 1
            if t < C.AUDIO_TOKEN_BASE:
                continue
            self._partial.append(token_to_code(t, self.n_audio_tokens))
            self.n_audio_tokens += 1
            if len(self._partial) == C.TOKENS_PER_FRAME:
                self.frames.append(self._partial)
                self._partial = []
        return len(self.frames)


def frames_to_layers(frames) -> list[np.ndarray]:
    """[n, 7] codes -> the 3 SNAC layers (upstream `convert_to_audio`): [0]; [1, 4]; [2, 3, 5, 6]."""
    f = np.asarray(frames, dtype=np.int64).reshape(-1, C.TOKENS_PER_FRAME)
    return [f[:, 0], f[:, [1, 4]].reshape(-1), f[:, [2, 3, 5, 6]].reshape(-1)]


# ---- 2. when to decode, what to keep -----------------------------------------------------------------
@dataclass(frozen=True)
class Window:
    decode_start: int   # frames [decode_start, decode_end) go into SNAC
    decode_end: int
    emit_start: int     # frames [emit_start, emit_end) of the result are kept
    emit_end: int

    @property
    def keep_samples(self) -> slice:
        s = (self.emit_start - self.decode_start) * C.SAMPLES_PER_FRAME
        return slice(s, s + (self.emit_end - self.emit_start) * C.SAMPLES_PER_FRAME)


class ChunkPlanner:
    def __init__(self, cfg: C.ChunkConfig | None = None):
        self.cfg = cfg or C.ChunkConfig()
        assert self.cfg.first_chunk_frames >= 1 and self.cfg.chunk_frames >= 1
        self.next_emit = 1 if self.cfg.skip_first_frame else 0
        self.n_chunks = 0

    def plan(self, n_frames: int, final: bool = False) -> list[Window]:
        """Windows that can be decoded now that `n_frames` frames exist (`final`: no more will come)."""
        cfg, out = self.cfg, []
        while self.next_emit < n_frames:
            a = self.next_emit
            b = a + (cfg.first_chunk_frames if self.n_chunks == 0 else cfg.chunk_frames)
            if b + cfg.right_context <= n_frames:
                win = Window(max(0, a - cfg.left_context), b + cfg.right_context, a, b)
            elif final and cfg.flush_tail:
                # Last chunk: everything left, with whatever right context exists.
                b = n_frames
                win = Window(max(0, a - cfg.left_context), n_frames, a, b)
            else:
                break
            out.append(win)
            self.next_emit = b
            self.n_chunks += 1
        return out


# ---- 3. GPU decode ------------------------------------------------------------------------------------
@dataclass
class AudioChunk:
    pcm: np.ndarray        # float32, mono, 24 kHz
    window: Window
    decode_ms: float       # SNAC call incl. copy back to CPU
    ready_s: float         # time.perf_counter() when the chunk was ready


class SNACDecoder:
    def __init__(self, model: str = C.SNAC_MODEL, device: str = "cuda"):
        import torch
        from snac import SNAC

        self._torch = torch
        self.device = device
        self.model = SNAC.from_pretrained(model).eval().to(device)
        # One worker thread: decode calls from all streams are serialised on the GPU, off the event loop.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snac")

    def set_noise(self, enabled: bool) -> None:
        """snac_24khz has `noise: true`: decoder NoiseBlocks add random noise, so two decodes of the
        same codes differ slightly. Disabling it makes decode deterministic (used for comparisons only)."""
        from snac.layers import NoiseBlock

        for m in self.model.modules():
            if isinstance(m, NoiseBlock):
                if enabled:
                    m.__dict__.pop("forward", None)
                else:
                    m.forward = lambda x: x

    def decode(self, frames) -> tuple[np.ndarray, float]:
        """Synchronous decode of [n, 7] frames -> (float32 samples, ms)."""
        torch = self._torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            codes = [torch.from_numpy(x).unsqueeze(0).to(self.device) for x in frames_to_layers(frames)]
            audio = self.model.decode(codes).squeeze().float().cpu().numpy()  # .cpu() waits for the GPU
        return audio, (time.perf_counter() - t0) * 1000

    async def decode_async(self, frames) -> tuple[np.ndarray, float]:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self.decode, frames)

    async def decode_window(self, frames: list[list[int]], win: Window) -> AudioChunk:
        audio, ms = await self.decode_async(frames[win.decode_start:win.decode_end])
        return AudioChunk(audio[win.keep_samples], win, ms, time.perf_counter())


async def stream_audio(token_stream: AsyncIterator[list[int]], decoder: SNACDecoder,
                       chunk_cfg: C.ChunkConfig | None = None,
                       assembler: FrameAssembler | None = None) -> AsyncIterator[AudioChunk]:
    """Token-id stream -> PCM chunks, each yielded as soon as its window is decodable.

    Pass your own `assembler` to read the token/frame counts (and frames) after the stream ends.
    """
    asm = assembler if assembler is not None else FrameAssembler()
    planner = ChunkPlanner(chunk_cfg)
    async for ids in token_stream:
        for win in planner.plan(asm.push(ids)):
            yield await decoder.decode_window(asm.frames, win)
    for win in planner.plan(len(asm.frames), final=True):
        yield await decoder.decode_window(asm.frames, win)
