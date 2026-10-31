"""Concurrent websocket load generator that measures latency and throughput of the server under load.

`ws_request` runs one request on an open connection and measures it from the client's side: TTFA is the
time from sending the JSON to receiving the first audio bytes, and gaps/buffer-needed use the step 2
definitions on the chunk arrival times.
"""

from __future__ import annotations

import json
import time

from . import config as C
from .metrics import playback_metrics

BYTES_PER_SAMPLE = 2  # PCM16

TEXTS = {
    "sentence": "Hi, I'm testing a realtime speech server on a single L4 GPU.",  # step 1 sentence
    "paragraph": ("Streaming speech synthesis is mostly a race against the listener. The model has to "
                  "produce each slice of audio before the previous one finishes playing, or the voice "
                  "stutters. On a single small GPU, that means every millisecond of decoding matters, "
                  "from the matrix math all the way to the final audio codec."),  # step 2 paragraph
}


async def ws_request(ws, text: str, voice: str = C.DEFAULT_VOICE, keep_audio: bool = False,
                     on_chunk=None) -> dict:
    """One request on an open websocket. Returns client-side measurements plus the server's `done` dict.

    `on_chunk(t, n_samples)` is called for every audio message (used for throughput accounting)."""
    t0 = time.perf_counter()
    await ws.send(json.dumps({"text": text, "voice": voice}))
    arrivals, durations, audio = [], [], []
    while True:
        msg = await ws.recv()
        t = time.perf_counter()
        if isinstance(msg, bytes):
            n = len(msg) // BYTES_PER_SAMPLE
            arrivals.append(t - t0)
            durations.append(n / C.SAMPLE_RATE)
            if keep_audio:
                audio.append(msg)
            if on_chunk:
                on_chunk(t, n)
            continue
        server = json.loads(msg)
        break
    t_end = time.perf_counter()
    row = {"t_start": t0, "t_end": t_end, "status": server.get("type"), "server": server}
    if server.get("type") != "done" or not arrivals:
        row["error"] = server.get("message", "no audio received")
        return row
    gaps, gap_s, need = playback_metrics(arrivals, durations)
    audio_s = sum(durations)
    row.update({
        "client_ttfa_ms": arrivals[0] * 1000,
        "audio_s": audio_s,
        "wall_s": t_end - t0,
        "client_rtf": (t_end - t0) / audio_s,
        "gap_count": gaps,
        "gap_total_ms": gap_s * 1000,
        "buffer_needed_ms": need * 1000,
        "chunks": len(arrivals),
    })
    if keep_audio:
        row["pcm16"] = b"".join(audio)
    return row


# ---- load sweep ----------------------------------------------------------------------------------------
import asyncio  # noqa: E402
import urllib.request  # noqa: E402

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

REQUEST_TIMEOUT_S = 900


def http_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=10) as r:
        return json.loads(r.read())


async def poll_metrics(metrics_url: str, samples: list, stop: asyncio.Event, every_s: float = 1.0):
    """Sample the server's /metrics once a second (GPU memory/util, SNAC worker busy time, streams)."""
    while not stop.is_set():
        try:
            m = await asyncio.to_thread(http_json, metrics_url)
            samples.append((time.perf_counter(), m))
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop.wait(), every_s)
        except asyncio.TimeoutError:
            pass


class Level:
    """N concurrent streams. Each: 1 warmup request, then measured requests back to back (alternating
    sentence/paragraph) until `duration_s` after the level start AND at least `min_measured` requests,
    then unmeasured filler requests until every stream is done measuring, so concurrency stays at N for
    the whole life of every measured request."""

    def __init__(self, url, n, duration_s, min_measured, voice=C.DEFAULT_VOICE):
        self.url, self.n, self.duration_s, self.min_measured, self.voice = url, n, duration_s, min_measured, voice
        self.rows: list[dict] = []
        self.chunks: list[tuple[float, int]] = []          # (arrival time, samples), for throughput
        self.warm_done = 0
        self.measuring_streams = n
        self.all_measured = asyncio.Event()
        self.t_all_warm = self.t_last_measured_end = None

    def _done_measuring(self) -> None:
        self.measuring_streams -= 1
        if self.measuring_streams == 0:
            self.all_measured.set()

    async def stream(self, sid: int):
        kinds = ["sentence", "paragraph"] if sid % 2 == 0 else ["paragraph", "sentence"]
        i, measured, measuring = 0, 0, True
        try:
            async with connect(self.url, max_size=None, ping_interval=None, open_timeout=60) as ws:
                while True:
                    kind = kinds[i % 2]
                    if i == 0:
                        phase = "warmup"
                    elif measuring and (measured < self.min_measured or time.perf_counter() < self.t_end):
                        phase = "measured"
                    else:
                        phase = "filler"
                        if measuring:                       # this stream just finished measuring
                            measuring = False
                            self._done_measuring()
                    try:
                        r = await asyncio.wait_for(
                            ws_request(ws, TEXTS[kind], self.voice,
                                       on_chunk=lambda t, n: self.chunks.append((t, n))),
                            REQUEST_TIMEOUT_S)
                    except asyncio.TimeoutError:
                        r = {"t_start": None, "t_end": time.perf_counter(), "status": "timeout", "server": {},
                             "error": f"no reply within {REQUEST_TIMEOUT_S} s"}
                    except Exception as e:
                        r = {"t_start": None, "t_end": time.perf_counter(), "status": "exception", "server": {},
                             "error": repr(e)}
                    self.rows.append({"stream": sid, "seq": i, "kind": kind, "phase": phase, **r})
                    if r["status"] in ("timeout", "exception"):
                        return                              # connection unusable; the row is recorded
                    if phase == "measured":
                        measured += 1
                        self.t_last_measured_end = max(self.t_last_measured_end or 0, r["t_end"])
                    if i == 0:
                        self.warm_done += 1
                        if self.warm_done == self.n:
                            self.t_all_warm = time.perf_counter()
                    i += 1
        except Exception as e:                              # e.g. could not connect
            self.rows.append({"stream": sid, "seq": i, "kind": None, "phase": "connect", "t_start": None,
                              "t_end": time.perf_counter(), "status": "exception", "server": {}, "error": repr(e)})
        finally:
            if measuring:                                   # stopped early: don't block the level
                self._done_measuring()

    async def run(self) -> None:
        self.t0 = time.perf_counter()
        self.t_end = self.t0 + self.duration_s
        tasks = [asyncio.create_task(self.stream(sid)) for sid in range(self.n)]
        await self.all_measured.wait()
        for t in tasks:
            t.cancel()                                      # closes the sockets; server aborts fillers
        await asyncio.gather(*tasks, return_exceptions=True)
        self.t_stop = time.perf_counter()


def pct(values, q):
    v = [x for x in values if x is not None]
    return round(float(np.percentile(v, q)), 3) if v else None


def summarize_level(level: Level, samples: list, cpu_s: float) -> dict:
    m = [r for r in level.rows if r["phase"] == "measured"]
    ok = [r for r in m if r["status"] == "done"]
    errors = [r for r in level.rows if r["status"] != "done"]  # any phase: an error is an error
    # throughput while all N streams are busy: from the last warmup end to the last measured end
    t_a, t_b = level.t_all_warm, level.t_last_measured_end
    thr = None
    if t_a and t_b and t_b > t_a:
        samp = sum(n for t, n in level.chunks if t_a <= t <= t_b)
        thr = samp / C.SAMPLE_RATE / (t_b - t_a)
    win = [(t, s) for t, s in samples if t_a and t_b and t_a <= t <= t_b] or samples
    snac_busy = None
    if len(win) >= 2:
        (ta, a), (tb, b) = win[0], win[-1]
        snac_busy = (b["snac_busy_s_total"] - a["snac_busy_s_total"]) / (tb - ta)
    wall = level.t_stop - level.t0
    return {
        "streams": level.n,
        "measured_requests": len(m),
        "measured_sentence": sum(r["kind"] == "sentence" for r in m),
        "measured_paragraph": sum(r["kind"] == "paragraph" for r in m),
        "errors": len(errors),
        "ttfa_ms_p50": pct([r["client_ttfa_ms"] for r in ok], 50),
        "ttfa_ms_p95": pct([r["client_ttfa_ms"] for r in ok], 95),
        "rtf_p50": pct([r["client_rtf"] for r in ok], 50),
        "rtf_p95": pct([r["client_rtf"] for r in ok], 95),
        "pct_requests_with_gaps": round(100 * sum(r["gap_count"] > 0 for r in ok) / len(ok), 1) if ok else None,
        "gap_count_p95": pct([r["gap_count"] for r in ok], 95),
        "buffer_needed_ms_p95": pct([r["buffer_needed_ms"] for r in ok], 95),
        "buffer_needed_ms_max": pct([r["buffer_needed_ms"] for r in ok], 100),
        "throughput_audio_s_per_s": round(thr, 3) if thr else None,
        "throughput_window_s": round(t_b - t_a, 1) if t_a and t_b else None,
        "server_queue_ms_p95": pct([r["server"].get("queue_ms") for r in ok], 95),
        "snac_ms_per_chunk_p50": pct([r["server"].get("snac_ms_mean") for r in ok], 50),
        "snac_wait_ms_per_chunk_p50": pct([r["server"].get("snac_wait_ms_mean") for r in ok], 50),
        "snac_busy_fraction": round(snac_busy, 3) if snac_busy is not None else None,
        "gpu_util_pct_mean": round(float(np.mean([s["gpu_util_pct"] for _, s in win])), 1) if win else None,
        "gpu_mem_used_gib_peak": max((s["gpu_mem_used_gib"] for _, s in samples), default=None),
        "client_cpu_pct_of_one_core": round(100 * cpu_s / wall, 1),
        "level_wall_s": round(wall, 1),
    }


ROW_FIELDS = ["streams", "stream", "seq", "kind", "phase", "status", "error", "client_ttfa_ms", "client_rtf",
              "audio_s", "wall_s", "gap_count", "gap_total_ms", "buffer_needed_ms", "chunks",
              "server_ttfa_ms", "server_rtf", "tokens", "queue_ms", "prefill_ms", "snac_ms_mean",
              "snac_wait_ms_mean", "t_start_rel_s"]


def flatten_row(level: Level, r: dict) -> dict:
    s = r.get("server", {})
    out = {"streams": level.n, **{k: r.get(k) for k in ROW_FIELDS if k in r},
           "server_ttfa_ms": s.get("ttfa_ms"), "server_rtf": s.get("rtf"), "tokens": s.get("tokens"),
           "queue_ms": s.get("queue_ms"), "prefill_ms": s.get("prefill_ms"),
           "snac_ms_mean": s.get("snac_ms_mean"), "snac_wait_ms_mean": s.get("snac_wait_ms_mean"),
           "t_start_rel_s": round(r["t_start"] - level.t0, 3) if r.get("t_start") else None}
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


async def run_level(url: str, metrics_url: str, n: int, duration_s: float, min_measured: int) -> tuple[dict, list]:
    proc = psutil.Process()
    samples: list = []
    stop = asyncio.Event()
    poller = asyncio.create_task(poll_metrics(metrics_url, samples, stop))
    cpu0 = sum(proc.cpu_times()[:2])
    level = Level(url, n, duration_s, min_measured)
    await level.run()
    cpu_s = sum(proc.cpu_times()[:2]) - cpu0
    stop.set()
    await poller
    return summarize_level(level, samples, cpu_s), [flatten_row(level, r) for r in level.rows]
