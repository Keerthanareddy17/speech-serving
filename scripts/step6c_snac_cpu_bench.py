"""Step 6c: can the CPU decode SNAC fast enough to take it off the GPU?

  python scripts/step6c_snac_cpu_bench.py  -> results/step6c_snac_cpu_bench.json

fp32 SNAC on the CPU, our 4-frame windows. Part 1: one process, torch.set_num_threads in THREADS, batches in
BATCHES (ms per call, windows/s). Part 2: several worker processes with fewer threads each, each decoding
batch-8 calls back to back for a fixed time: aggregate windows/s. Need: ~11.72 windows/s per stream
(N=8: ~94/s, N=12: ~141/s).
"""

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

THREADS = [1, 2, 4, 6, 8, 12]
BATCHES = [1, 4, 8, 16]
POOLS = [(2, 6), (3, 4), (4, 3), (6, 2), (12, 1)]   # (processes, threads each)
POOL_BATCH, POOL_SECONDS = 8, 8.0
WINDOWS_PER_STREAM = 7 * 24000 / 2048 / 7          # 11.72 frames (= windows) per second per stream
OUT = ROOT / "results/step6c_snac_cpu_bench.json"


def make_inputs(b, seed=0):
    import numpy as np

    from speechserve import config as C
    rng = np.random.default_rng(seed)
    return [rng.integers(0, C.CODEBOOK_SIZE, size=(4, 7)) for _ in range(b)]


def pool_worker(args):
    threads, seconds, barrier_t = args
    import torch
    torch.set_num_threads(threads)
    from speechserve.codec import SNACDecoder
    dec = SNACDecoder(device="cpu")
    wins = make_inputs(POOL_BATCH, seed=os.getpid())
    for _ in range(2):
        dec.decode_many(wins)
    while time.time() < barrier_t:                 # start all workers together
        time.sleep(0.01)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        dec.decode_many(wins)
        n += POOL_BATCH
    return n / (time.perf_counter() - t0)


def main():
    import numpy as np
    import torch

    from speechserve.codec import SNACDecoder
    dec = SNACDecoder(device="cpu")
    single = []
    for t in THREADS:
        torch.set_num_threads(t)
        for b in BATCHES:
            wins = make_inputs(b)
            for _ in range(2):
                dec.decode_many(wins)
            times = []
            t_end = time.perf_counter() + 3.0
            while len(times) < 3 or (time.perf_counter() < t_end and len(times) < 20):
                t0 = time.perf_counter()
                dec.decode_many(wins)
                times.append((time.perf_counter() - t0) * 1000)
            ms = float(np.median(times))
            row = {"threads": t, "batch": b, "call_ms_p50": round(ms, 1), "windows_per_s": round(b / ms * 1000, 1),
                   "streams_supported": round(b / ms * 1000 / WINDOWS_PER_STREAM, 1)}
            single.append(row)
            print(f"1 proc x {t:2d} threads, batch {b:2d}: {ms:7.1f} ms/call, {row['windows_per_s']:6.1f} windows/s "
                  f"(~{row['streams_supported']} streams)", flush=True)

    pools = []
    ctx = mp.get_context("spawn")
    for procs, threads in POOLS:
        with ctx.Pool(procs) as pool:
            start = time.time() + 20 + 2 * procs    # time for workers to load SNAC
            rates = pool.map(pool_worker, [(threads, POOL_SECONDS, start)] * procs)
        total = sum(rates)
        pools.append({"processes": procs, "threads_each": threads, "batch": POOL_BATCH,
                      "windows_per_s_total": round(total, 1), "streams_supported": round(total / WINDOWS_PER_STREAM, 1),
                      "per_process_call_ms": round(POOL_BATCH / (total / procs) * 1000, 1)})
        print(f"{procs:2d} procs x {threads:2d} threads, batch {POOL_BATCH}: {total:6.1f} windows/s total "
              f"(~{total / WINDOWS_PER_STREAM:.1f} streams), {pools[-1]['per_process_call_ms']} ms per call", flush=True)

    out = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": "n/a (CPU decode)", "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count(),
        "model": "hubertsiuzdak/snac_24khz", "vllm_version": "n/a (SNAC alone)", "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"dtype": "float32", "window_frames": 4, "noise": "on (default)",
                   "need_windows_per_s_per_stream": round(WINDOWS_PER_STREAM, 2),
                   "need_examples": {"N=6": round(6 * WINDOWS_PER_STREAM), "N=8": round(8 * WINDOWS_PER_STREAM),
                                     "N=12": round(12 * WINDOWS_PER_STREAM)},
                   "pool_seconds": POOL_SECONDS},
        "single_process": single, "process_pools": pools,
        "best_windows_per_s": max([r["windows_per_s"] for r in single] + [p["windows_per_s_total"] for p in pools]),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"best {out['best_windows_per_s']} windows/s -> saved {OUT}")


if __name__ == "__main__":
    main()
