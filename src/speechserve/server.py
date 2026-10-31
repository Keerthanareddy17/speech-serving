"""FastAPI app exposing a websocket endpoint that streams synthesized audio chunks to clients in realtime.

One process holds ONE vLLM engine and ONE SNAC decoder, shared by every connection; vLLM batches the
concurrent requests, and decode calls go through the `ChunkDecoder` interface (baseline: per-stream
SNAC calls on one worker thread; step 6 can swap in a batched decoder).

Protocol on /ws/tts (a connection may send several requests, one after another):
  client -> {"text": "...", "voice": "tara"}
  server -> binary messages: PCM16 little-endian, mono, 24 kHz, one per decoded chunk
  server -> {"type": "done", ...timings}   or   {"type": "error", "message": "..."}

Settings come from env vars (set by scripts/run_server.py): SPEECHSERVE_PRESET, SPEECHSERVE_LOG.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import aclosing, asynccontextmanager

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketDisconnected

# What a send/receive raises when the client has gone away (uvicorn's ClientDisconnected is an OSError).
DISCONNECTS = (WebSocketDisconnect, WebSocketDisconnected, OSError)

from . import config as C
from .codec import ChunkDecoder, FrameAssembler, SNACDecoder, stream_audio
from .engine import TTSEngine
from .metrics import DEFAULT_LOG, GPUStats, ServerMetrics

WARMUP_TEXT = "Hi, I'm testing a realtime speech server on a single L4 GPU."
MAX_TEXT_CHARS = 2000


class State:
    engine: TTSEngine
    decoder: ChunkDecoder
    metrics: ServerMetrics
    gpu: GPUStats
    chunk_cfg: C.ChunkConfig
    preset: str


S = State()


async def synthesize(text: str, voice: str, send_bytes) -> dict:
    """Run one request, sending PCM16 chunks through `send_bytes`. Returns its timings."""
    info: dict = {}
    asm = FrameAssembler()
    t0 = time.perf_counter()
    ttfa = None
    audio_samples = 0
    n_chunks = 0
    decode_ms = wait_ms = 0.0
    async with aclosing(stream_audio(S.engine.stream(text, voice, info=info), S.decoder, S.chunk_cfg, asm)) as chunks:
        async for c in chunks:
            await send_bytes((np.clip(c.pcm, -1.0, 1.0) * 32767).astype("<i2").tobytes())
            if ttfa is None:
                ttfa = (time.perf_counter() - t0) * 1000
            audio_samples += len(c.pcm)
            n_chunks += 1
            decode_ms += c.decode_ms
            wait_ms += c.wait_ms
    wall = time.perf_counter() - t0
    audio_s = audio_samples / C.SAMPLE_RATE
    return {
        "ttfa_ms": None if ttfa is None else round(ttfa, 1),
        "tokens": asm.n_tokens,
        "audio_s": round(audio_s, 3),
        "wall_s": round(wall, 3),
        "rtf": round(wall / audio_s, 3) if audio_s else None,
        "queue_ms": round(info.get("queue_ms", 0.0), 1),     # waiting in vLLM's scheduler
        "prefill_ms": round(info.get("prefill_ms", 0.0), 1),
        "preemptions": info.get("preemptions", 0),
        "finish_reason": info.get("finish_reason"),
        "chunks": n_chunks,
        "snac_ms_mean": round(decode_ms / n_chunks, 2) if n_chunks else None,
        "snac_wait_ms_mean": round(wait_ms / n_chunks, 2) if n_chunks else None,  # queued behind other streams
    }


# vLLM 0.32's Triton top-p sampler splits each row over S blocks, S = a power of two chosen from the batch
# size and SM count (topk_topp_triton.py `_topp_split_count`; on the L4: bs 1 -> 16, 2-3 -> 8, 4-7 -> 4,
# 8-14 -> 2, 15-29 -> 1), and above batch 64 it switches kernel. S is a compile-time constant, so the
# first batch with a new S compiles 3 kernels (~3 s) while the whole engine stalls. vLLM's own JIT warmup
# does not cover these. A first batch of 32 also compiled something, although its S equals batch 16's, so
# rather than predict every bucket we run one short simultaneous batch for every size 2..64 (+ 80, 128).
# Compiled kernels are cached on disk (~/.triton), so after the first ever start this takes ~10 s.
WARMUP_BATCH_SIZES = (*range(2, 65), 80, 128)


async def warmup_batch_sizes() -> None:
    async def drain(i):
        async for _ in S.engine.stream(WARMUP_TEXT, C.DEFAULT_VOICE, max_tokens=8):
            pass

    for n in WARMUP_BATCH_SIZES:
        t0 = time.perf_counter()
        await asyncio.gather(*(drain(i) for i in range(n)))
        dt = time.perf_counter() - t0
        if dt > 0.5:
            print(f"warmup batch {n}: {dt:.2f} s (kernel compile)", flush=True)
    print(f"warmup batch sizes {WARMUP_BATCH_SIZES[0]}..{WARMUP_BATCH_SIZES[-1]} done", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.preset = os.environ.get("SPEECHSERVE_PRESET", C.DEFAULT_PRESET)
    S.chunk_cfg = C.ChunkConfig()
    S.metrics = ServerMetrics(os.environ.get("SPEECHSERVE_LOG", DEFAULT_LOG))
    S.engine = TTSEngine(S.preset)
    S.decoder = SNACDecoder()
    S.gpu = GPUStats()
    # Warmup: the first request after load is slower and samples differently (step 2), so no user gets it.
    await synthesize(WARMUP_TEXT, C.DEFAULT_VOICE, lambda b: asyncio.sleep(0))
    await warmup_batch_sizes()
    print(f"speechserve ready: preset={S.preset}", flush=True)
    yield
    S.engine.shutdown()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "preset": S.preset}


@app.get("/metrics")
async def metrics():
    return {**S.metrics.snapshot(), **S.gpu.snapshot(), **S.decoder.stats(), "preset": S.preset}


@app.websocket("/ws/tts")
async def ws_tts(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            req = await ws.receive_json()
            text, voice = str(req.get("text", "")).strip(), str(req.get("voice", C.DEFAULT_VOICE))
            rid = uuid.uuid4().hex[:12]
            if not text or len(text) > MAX_TEXT_CHARS:
                await ws.send_json({"type": "error", "request_id": rid,
                                    "message": f"text must be 1..{MAX_TEXT_CHARS} characters"})
                continue
            S.metrics.request_started()
            record = {"request_id": rid, "preset": S.preset, "voice": voice, "text_chars": len(text)}
            try:
                timings = await synthesize(text, voice, ws.send_bytes)
            except DISCONNECTS:
                S.metrics.request_finished({**record, "status": "disconnected"})
                return
            except Exception as e:  # report to the client, keep serving others
                S.metrics.request_finished({**record, "status": "error", "error": repr(e)})
                await ws.send_json({"type": "error", "request_id": rid, "message": repr(e)})
                continue
            S.metrics.request_finished({**record, "status": "ok", **timings})
            await ws.send_json({"type": "done", "request_id": rid, **timings})
    except DISCONNECTS:
        pass
