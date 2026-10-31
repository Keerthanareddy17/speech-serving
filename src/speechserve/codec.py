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
from contextlib import aclosing
from dataclasses import dataclass
from typing import Protocol

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
    """Decides when to decode which window.

    First chunk: `first_chunk_frames` frames as soon as they (plus right context) exist, so TTFA is
    unchanged. Later chunks: upstream uses 1 frame per window (`chunk_frames=1`). With a larger cap, a
    chunk of c frames needs a window of c + left + right frames, so SNAC work per emitted frame falls from
    4x (c=1) to 1.75x (c=4) or 1.4x (c=8), but the chunk can only be sent once all its frames exist.

    Waiting for a big chunk is only safe while the client still has enough audio buffered. Given timing
    hints (see `plan`), the planner waits for a full `chunk_frames` chunk only if the frames it still needs
    are expected before the client's buffer runs out (minus a safety margin); otherwise it emits the
    largest power-of-two chunk available now (at least 1 frame, i.e. upstream behaviour). Without hints
    it always waits for full chunks (used by tests and offline decoding).
    """

    def __init__(self, cfg: C.ChunkConfig | None = None):
        self.cfg = cfg or C.ChunkConfig()
        assert self.cfg.first_chunk_frames >= 1 and self.cfg.chunk_frames >= 1
        std = self.cfg.left_context + 1 + self.cfg.right_context
        # Early first chunk (step 6c): wait for W < 4 frames and emit frame 0 with W-1 right frames.
        self.early = 1 <= self.cfg.first_wait_frames < std
        if self.early:
            assert self.cfg.first_chunk_frames == 1 and self.cfg.chunk_frames == 1, "early first chunk: 1-frame chunks only"
        self.next_emit = 0 if self.early else (1 if self.cfg.skip_first_frame else 0)
        self.n_chunks = 0

    def _window(self, a: int, b: int, end: int) -> Window:
        return Window(max(0, a - self.cfg.left_context), end, a, b)

    def plan(self, n_frames: int, final: bool = False, buffer_s: float | None = None,
             frame_s: float | None = None, decode_s: float = 0.0) -> list[Window]:
        """Windows that can be decoded now that `n_frames` frames exist (`final`: no more will come).

        Timing hints (all optional): `buffer_s` = audio the client has buffered but not yet played,
        `frame_s` = recent time per generated frame for this stream, `decode_s` = recent SNAC time
        (incl. queueing) for one chunk.
        """
        cfg, out, right = self.cfg, [], self.cfg.right_context
        while self.next_emit < n_frames:
            a = self.next_emit
            if self.early:
                w = self._early_window(a, n_frames, final, buffer_s, frame_s, decode_s)
                if w is None:
                    break
                out.append(w)
                self.next_emit = w.emit_end
                self.n_chunks += 1
                continue
            if self.n_chunks == 0:
                b = a + cfg.first_chunk_frames
            else:
                avail = n_frames - right - a              # frames that could be emitted right now
                if avail >= cfg.chunk_frames:
                    b = a + cfg.chunk_frames
                elif avail >= 1 and not final:
                    if buffer_s is not None and frame_s is not None:
                        wait_s = (a + cfg.chunk_frames + right - n_frames) * frame_s
                        if wait_s + decode_s + cfg.chunk_safety_ms / 1000 < buffer_s:
                            break                         # safe to wait for a full-size chunk
                        b = a + (1 << (avail.bit_length() - 1))   # largest power of two <= avail
                    else:
                        break                             # no timing info: wait for full chunks
                else:
                    b = a + max(avail, 0)                 # final: handled below
            if b > a and b + right <= n_frames:
                win = self._window(a, b, b + right)
            elif final and cfg.flush_tail:
                # Last chunk: everything left, with whatever right context exists. (Tail windows are
                # shorter than the standard 4 frames, so with --snac-graphs they run eagerly.)
                b = n_frames
                win = self._window(a, b, n_frames)
            else:
                break
            out.append(win)
            self.next_emit = b
            self.n_chunks += 1
        return out


    def _early_window(self, a, n, final, buffer_s, frame_s, decode_s):
        """Early-first-chunk mode: one frame per chunk, right context between W-1 and the standard 2."""
        cfg, right = self.cfg, self.cfg.right_context
        min_right = cfg.first_wait_frames - 1
        if self.n_chunks == 0:                         # first chunk: frame 0 as soon as W frames exist
            return Window(0, cfg.first_wait_frames, 0, 1) if n >= cfg.first_wait_frames else (
                Window(0, n, 0, 1) if final else None)
        if a + 1 + right <= n:                         # standard context available
            return Window(max(0, a - cfg.left_context), a + 1 + right, a, a + 1)
        if final:
            return Window(max(0, a - cfg.left_context), n, a, a + 1)
        if a + 1 + min_right > n:                      # not even the minimum context yet
            return None
        if buffer_s is not None and frame_s is not None:
            wait_s = (a + 1 + right - n) * frame_s
            if wait_s + decode_s + cfg.chunk_safety_ms / 1000 < buffer_s:
                return None                            # safe to wait for full right context
            return Window(max(0, a - cfg.left_context), n, a, a + 1)   # send now with what exists
        return None                                    # no timing info: wait for full context


# ---- 3. GPU decode ------------------------------------------------------------------------------------
@dataclass
class AudioChunk:
    pcm: np.ndarray        # float32, mono, 24 kHz
    window: Window
    decode_ms: float       # SNAC call incl. copy back to CPU
    ready_s: float         # time.perf_counter() when the chunk was ready
    wait_ms: float = 0.0   # time queued behind other decode calls before this one started


class ChunkDecoder(Protocol):
    """What `stream_audio` (and the server) need from a decoder. The baseline `SNACDecoder` decodes
    each stream's windows one call at a time; a batched decoder (step 6) can implement the same method
    by grouping windows from many streams into one SNAC call."""

    async def decode_window(self, frames: list[list[int]], win: Window) -> AudioChunk: ...

    def stats(self) -> dict: ...


class SNACDecoder:
    """Baseline decoder: one SNAC call per window, all streams' calls queued on one worker thread."""

    def __init__(self, model: str = C.SNAC_MODEL, device: str = "cuda", dtype: str = "float32"):
        import torch
        from snac import SNAC

        self._torch = torch
        self.device = device
        self.dtype = dtype
        # Codes are integers; the decoder's weights/activations run in `dtype`; output is returned as fp32.
        self.model = SNAC.from_pretrained(model).eval().to(device=device, dtype=getattr(torch, dtype))
        self.graphs: dict[tuple[int, int], tuple] = {}   # (batch, window frames) -> (graph, inputs, output)
        # One worker thread: decode calls from all streams are serialised on the GPU, off the event loop.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="snac")
        self.calls = 0
        self.busy_s = 0.0          # total time the worker spent decoding
        self.pending = 0           # calls submitted but not finished

    def stats(self) -> dict:
        return {"decoder": "per_stream", "snac_calls_total": self.calls, "snac_windows_total": self.calls,
                "snac_busy_s_total": round(self.busy_s, 3), "snac_pending": self.pending}

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
        if self.graphs:
            audio, ms = self.decode_many([np.asarray(frames)])
            return audio[0], ms
        torch = self._torch
        t0 = time.perf_counter()
        with torch.inference_mode():
            codes = [torch.from_numpy(x).unsqueeze(0).to(self.device) for x in frames_to_layers(frames)]
            audio = self.model.decode(codes).squeeze().float().cpu().numpy()  # .cpu() waits for the GPU
        return audio, (time.perf_counter() - t0) * 1000

    def capture_graphs(self, window_frames: int, max_batch: int) -> float:
        """Record one CUDA graph per batch size 1..max_batch for `window_frames`-frame windows (step 6b).
        A graph replays SNAC's whole kernel sequence with one launch; at small batches SNAC was launch-bound
        (batch 1: 6.8 ms eager vs 1.6 ms graphed, fp32). All graphs share one memory pool: replays happen one
        at a time on the single decode thread, and each output is copied to the CPU right after its replay.
        Captured largest batch first, so smaller graphs reuse the big blocks (ascending order made the pool
        grow to 2.1 GiB and run out of memory next to vLLM)."""
        torch = self._torch
        t0 = time.perf_counter()
        pool = self._pool_handle = getattr(self, "_pool_handle", None) or torch.cuda.graph_pool_handle()
        for b in range(max_batch, 0, -1):
            shapes = [(b, window_frames), (b, 2 * window_frames), (b, 4 * window_frames)]
            inputs = [torch.zeros(sh, dtype=torch.long, device=self.device) for sh in shapes]
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.inference_mode():
                for _ in range(2):
                    self.model.decode(inputs)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode(), torch.cuda.graph(g, pool=pool):
                out = self.model.decode(inputs)
            self.graphs[(b, window_frames)] = (g, inputs, out)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    def decode_many(self, frames_list) -> tuple[list[np.ndarray], float]:
        """Synchronous decode of several same-length windows in ONE SNAC call -> (samples per window, ms).
        Each layer is stacked along the batch dimension: (B, n), (B, 2n), (B, 4n); no padding needed.
        Uses a recorded CUDA graph for this exact shape if one exists, else runs eagerly."""
        torch = self._torch
        t0 = time.perf_counter()
        layers = [np.stack(x) for x in zip(*(frames_to_layers(f) for f in frames_list))]
        graph = self.graphs.get((len(frames_list), len(frames_list[0])))
        with torch.inference_mode():
            codes = [torch.from_numpy(x).to(self.device) for x in layers]
            if graph is not None:
                g, inputs, out = graph
                for dst, src in zip(inputs, codes):
                    dst.copy_(src)
                g.replay()
                audio = out.float().cpu().numpy()
            else:
                audio = self.model.decode(codes).float().cpu().numpy()  # (B, 1, T)
        return [audio[i, 0] for i in range(len(frames_list))], (time.perf_counter() - t0) * 1000

    async def decode_async(self, frames) -> tuple[np.ndarray, float]:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self.decode, frames)

    async def decode_window(self, frames: list[list[int]], win: Window) -> AudioChunk:
        submitted = time.perf_counter()
        started = []

        def job():
            started.append(time.perf_counter())
            return self.decode(frames[win.decode_start:win.decode_end])

        self.pending += 1
        try:
            audio, ms = await asyncio.get_running_loop().run_in_executor(self._pool, job)
        finally:
            self.pending -= 1
        self.calls += 1
        self.busy_s += ms / 1000
        return AudioChunk(audio[win.keep_samples], win, ms, time.perf_counter(),
                          wait_ms=(started[0] - submitted) * 1000)


@dataclass
class _Pending:
    frames: np.ndarray        # [n_frames, 7] for this window
    win: Window
    fut: asyncio.Future
    submitted: float


class BatchedChunkDecoder:
    """One decode worker for all streams: every pending window goes into a shared queue; the worker takes
    all of them (same window length, oldest first, at most `max_batch`), stacks them, runs ONE SNAC call
    and returns each stream its slice. While a batch decodes, new windows pile up, so batches grow with
    load by themselves; `max_wait_ms` > 0 additionally lets an idle worker wait that long for more."""

    def __init__(self, snac: SNACDecoder | None = None, max_batch: int = 64, max_wait_ms: float = 0.0,
                 max_frames: int = 320):
        assert max_batch >= 1 and max_wait_ms >= 0 and max_frames >= 1
        self.snac = snac or SNACDecoder()
        self.max_batch, self.max_wait_ms = max_batch, max_wait_ms
        # Memory cap per call, in frames (batch x window length): SNAC's activations grow with both, and
        # vLLM leaves only ~2.7 GiB free (64 x 11-frame windows ran out of memory in step 6b).
        self.max_frames = max_frames
        self._queue: list[_Pending] = []
        self._wakeup: asyncio.Event | None = None
        self._task: asyncio.Task | None = None
        self._inflight = 0
        self.calls = self.windows = self.batch_max = 0
        self.busy_s = 0.0

    def set_noise(self, enabled: bool) -> None:
        self.snac.set_noise(enabled)

    def stats(self) -> dict:
        return {"decoder": "batched", "snac_calls_total": self.calls, "snac_windows_total": self.windows,
                "snac_busy_s_total": round(self.busy_s, 3), "snac_pending": len(self._queue) + self._inflight,
                "snac_batch_max_seen": self.batch_max, "snac_max_batch": self.max_batch,
                "snac_max_frames": self.max_frames, "snac_max_wait_ms": self.max_wait_ms}

    async def decode_window(self, frames: list[list[int]], win: Window) -> AudioChunk:
        if self._task is None or self._task.done():
            self._wakeup = asyncio.Event()
            self._task = asyncio.create_task(self._worker())
        item = _Pending(np.asarray(frames[win.decode_start:win.decode_end], dtype=np.int64), win,
                        asyncio.get_running_loop().create_future(), time.perf_counter())
        self._queue.append(item)
        self._wakeup.set()
        audio, ms, wait_ms = await item.fut
        return AudioChunk(audio[win.keep_samples], win, ms, time.perf_counter(), wait_ms=wait_ms)

    def batch_limit(self, window_frames: int) -> int:
        return max(1, min(self.max_batch, self.max_frames // window_frames))

    def _take_batch(self) -> list[_Pending]:
        n = len(self._queue[0].frames)
        batch = [it for it in self._queue if len(it.frames) == n][: self.batch_limit(n)]
        taken = set(map(id, batch))
        self._queue = [it for it in self._queue if id(it) not in taken]
        return batch

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._wakeup.wait()
            self._wakeup.clear()
            if self.max_wait_ms > 0 and 0 < len(self._queue) < self.max_batch:
                await asyncio.sleep(self.max_wait_ms / 1000)
            while self._queue:
                batch = [it for it in self._take_batch() if not it.fut.done()]  # skip cancelled streams
                if not batch:
                    continue
                started = time.perf_counter()
                self._inflight = len(batch)
                try:
                    audios, ms = await loop.run_in_executor(self.snac._pool, self.snac.decode_many,
                                                            [it.frames for it in batch])
                except Exception as e:  # deliver the error to every waiting stream
                    for it in batch:
                        if not it.fut.done():
                            it.fut.set_exception(e)
                    continue
                finally:
                    self._inflight = 0
                self.calls += 1
                self.windows += len(batch)
                self.batch_max = max(self.batch_max, len(batch))
                self.busy_s += ms / 1000
                for it, audio in zip(batch, audios):
                    if not it.fut.done():
                        it.fut.set_result((audio, ms, (started - it.submitted) * 1000))


async def stream_audio(token_stream: AsyncIterator[list[int]], decoder: ChunkDecoder,
                       chunk_cfg: C.ChunkConfig | None = None,
                       assembler: FrameAssembler | None = None,
                       timing: dict | None = None) -> AsyncIterator[AudioChunk]:
    """Token-id stream -> PCM chunks, each yielded as soon as its window is decodable.

    Pass your own `assembler` to read the token/frame counts (and frames) after the stream ends.
    Closing this generator early also closes `token_stream`, so the engine aborts the request.

    For growing chunks (chunk_frames > 1) it tracks what the planner needs: the client's buffer (audio
    sent minus time since the first chunk, i.e. assuming playback starts when the first chunk arrives),
    this stream's time per generated frame since then, and its last SNAC time.
    """
    asm = assembler if assembler is not None else FrameAssembler()
    cfg = chunk_cfg or C.ChunkConfig()
    planner = ChunkPlanner(cfg)
    t_first = None            # when the first chunk was ready (= playback start)
    n_at_first = 0
    sent_s = 0.0
    last_decode_s = 0.0

    adaptive = cfg.chunk_frames > 1 or planner.early

    def hints(n: int) -> dict:
        if not adaptive or t_first is None or n <= n_at_first:
            return {}
        now = time.perf_counter()
        return {"buffer_s": sent_s - (now - t_first), "frame_s": (now - t_first) / (n - n_at_first),
                "decode_s": last_decode_s}

    async def emit(win):
        nonlocal t_first, n_at_first, sent_s, last_decode_s
        if t_first is None and timing is not None:     # TTFA breakdown (step 6c)
            timing["first_window_planned_s"] = time.perf_counter()
            timing["first_window_audio_tokens"] = asm.n_audio_tokens
        chunk = await decoder.decode_window(asm.frames, win)
        if t_first is None:
            t_first, n_at_first = chunk.ready_s, len(asm.frames)
            if timing is not None:
                timing["first_chunk_ready_s"] = chunk.ready_s
                timing["first_chunk_decode_ms"] = chunk.decode_ms
                timing["first_chunk_wait_ms"] = chunk.wait_ms
        sent_s += len(chunk.pcm) / C.SAMPLE_RATE
        last_decode_s = (chunk.decode_ms + chunk.wait_ms) / 1000
        return chunk

    async with aclosing(token_stream):
        async for ids in token_stream:
            n = asm.push(ids)
            for win in planner.plan(n, **hints(n)):
                yield await emit(win)
    for win in planner.plan(len(asm.frames), final=True):
        yield await emit(win)
