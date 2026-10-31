"""Step 4: sweep concurrent websocket streams against a running server and record the baseline.

  python scripts/run_server.py            # in another tmux window
  python scripts/run_loadtest.py [--levels 1 2 4 8 16 32 64] [--duration 60] [--min-measured 2]

Writes results/baseline_sweep.csv (one row per request), results/baseline_sweep_levels.csv (one row per
level) and results/baseline_sweep.json (level summaries + metadata + max streams).
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve.loadtest import REQUEST_TIMEOUT_S, ROW_FIELDS, TEXTS, http_json, run_level  # noqa: E402


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


async def wait_idle(metrics_url, timeout=120):
    """Between levels: wait until the server has no active streams (aborted fillers have drained)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = await asyncio.to_thread(http_json, metrics_url)
        if m["active_streams"] == 0 and m["snac_pending"] == 0:
            return
        await asyncio.sleep(0.5)


def max_streams(levels):
    """Highest N with p95 RTF < 1.0, no request with gaps, and no errors (checked from N=1 upward)."""
    best = None
    for s in levels:
        if s["errors"] == 0 and s["rtf_p95"] is not None and s["rtf_p95"] < 1.0 and s["pct_requests_with_gaps"] == 0:
            best = s["streams"]
        else:
            break
    return best


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1:8000")
    ap.add_argument("--levels", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--duration", type=float, default=60.0, help="seconds per level to keep starting measured requests")
    ap.add_argument("--min-measured", type=int, default=2, help="measured requests each stream completes at least")
    ap.add_argument("--out", default="results/baseline_sweep")
    args = ap.parse_args()
    url, metrics_url = f"ws://{args.host}/ws/tts", f"http://{args.host}/metrics"
    health = http_json(f"http://{args.host}/health")

    levels, rows = [], []
    out = ROOT / args.out
    for n in args.levels:
        await wait_idle(metrics_url)
        print(f"--- N={n}", flush=True)
        summary, level_rows = await run_level(url, metrics_url, n, args.duration, args.min_measured)
        levels.append(summary)
        rows += level_rows
        s = summary
        print(f"N={n:3d} reqs {s['measured_requests']:4d} err {s['errors']} | TTFA p50 {s['ttfa_ms_p50']} p95 "
              f"{s['ttfa_ms_p95']} ms | RTF p50 {s['rtf_p50']} p95 {s['rtf_p95']} | gaps {s['pct_requests_with_gaps']}% "
              f"| buffer p95 {s['buffer_needed_ms_p95']} ms | thr {s['throughput_audio_s_per_s']} audio-s/s | "
              f"SNAC busy {s['snac_busy_fraction']} batch {s['snac_batch_mean']} wait p50 "
              f"{s['snac_wait_ms_per_chunk_p50']} ms | vLLM {s['vllm_tokens_per_s']} tok/s {s['vllm_ms_per_token']} "
              f"ms/tok | GPU "
              f"{s['gpu_util_pct_mean']}% {s['gpu_mem_used_gib_peak']} GiB | client CPU {s['client_cpu_pct_of_one_core']}% "
              f"| {s['level_wall_s']} s", flush=True)
        # save after every level so a crash keeps what was measured
        write_csv(f"{out}.csv", rows, ROW_FIELDS)
        write_csv(f"{out}_levels.csv", levels, list(levels[0]))
        result = {
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "gpu": health["gpu"], "model": health["model"], "vllm_version": health["vllm_version"],
            "torch_version": health["torch_version"], "python_version": health["python_version"],
            "config": {"server": health, "levels": args.levels, "duration_s": args.duration,
                       "min_measured_per_stream": args.min_measured, "request_timeout_s": REQUEST_TIMEOUT_S,
                       "texts": TEXTS,
                       "method": "each stream: 1 warmup request, then measured requests back to back alternating "
                                 "sentence/paragraph until duration elapsed and min_measured done, then unmeasured "
                                 "fillers until all streams finish measuring (concurrency stays at N)"},
            "max_streams": max_streams(levels),
            "max_streams_rule": "highest N (from 1 up) with p95 RTF < 1.0, no request with gaps, no errors",
            "levels": levels,
        }
        Path(f"{out}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"baseline max streams: {max_streams(levels)}  -> {out}.csv/_levels.csv/.json")


if __name__ == "__main__":
    asyncio.run(main())
