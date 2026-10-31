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
