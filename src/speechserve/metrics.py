"""Server-side metrics: live counters for /metrics and one structured JSON log line per request.

The log lives outside the repo (default /content/logs/server.jsonl), since it grows without bound and may
contain user text.
"""

from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DEFAULT_LOG = "/content/logs/server.jsonl"


def playback_metrics(arrivals, durations):
    """Playback starts when chunk 0 arrives. A gap = a chunk arriving after the audio before it ended.
    buffer_needed = the start delay that would make every chunk arrive in time.
    Returns (gap_count, total_gap_s, buffer_needed_s). Same definitions in step 2, the client and the load test."""
    a0, played, gaps, gap_s, need = arrivals[0], 0.0, 0, 0.0, 0.0
    play_end = a0
    for a, d in zip(arrivals, durations):
        need = max(need, a - (a0 + played))   # lateness vs. an uninterrupted schedule
        if a > play_end + 1e-9 and played > 0:
            gaps += 1
            gap_s += a - play_end
        play_end = max(play_end, a) + d
        played += d
    return gaps, gap_s, need


class ServerMetrics:
    def __init__(self, log_path: str | Path = DEFAULT_LOG, window: int = 200):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("a", buffering=1)  # line-buffered: each request is on disk at once
        self.active_streams = 0
        self.total_requests = 0
        self.errors = 0
        self.recent_ttfa_ms: deque[float] = deque(maxlen=window)  # last `window` successful requests
        self.started = time.time()

    def request_started(self) -> None:
        self.active_streams += 1
        self.total_requests += 1

    def request_finished(self, record: dict) -> None:
        """`record` must contain "status" ("ok" / "error" / "disconnected"); it is logged as one line."""
        self.active_streams -= 1
        if record["status"] == "error":
            self.errors += 1
        elif record["status"] == "ok" and record.get("ttfa_ms") is not None:
            self.recent_ttfa_ms.append(record["ttfa_ms"])
        line = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
        self._log.write(json.dumps(line) + "\n")

    def snapshot(self) -> dict:
        t = np.array(self.recent_ttfa_ms) if self.recent_ttfa_ms else None
        return {
            "uptime_s": round(time.time() - self.started, 1),
            "active_streams": self.active_streams,
            "total_requests": self.total_requests,
            "errors": self.errors,
            "recent_ttfa_n": len(self.recent_ttfa_ms),
            "recent_ttfa_p50_ms": None if t is None else round(float(np.percentile(t, 50)), 1),
            "recent_ttfa_p95_ms": None if t is None else round(float(np.percentile(t, 95)), 1),
        }


class GPUStats:
    """Device-wide GPU memory and utilisation via NVML (covers the vLLM engine-core process too)."""

    def __init__(self, index: int = 0):
        import pynvml

        pynvml.nvmlInit()
        self._nv = pynvml
        self._h = pynvml.nvmlDeviceGetHandleByIndex(index)

    def snapshot(self) -> dict:
        mem = self._nv.nvmlDeviceGetMemoryInfo(self._h)
        util = self._nv.nvmlDeviceGetUtilizationRates(self._h)
        return {"gpu_mem_used_gib": round(mem.used / 2**30, 2), "gpu_mem_total_gib": round(mem.total / 2**30, 2),
                "gpu_util_pct": util.gpu}
