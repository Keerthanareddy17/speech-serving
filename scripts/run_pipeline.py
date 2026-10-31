"""Step 7: audio tokenization pipeline, end to end: parquet -> decode -> VAD trim -> SNAC encode -> shards.

  python scripts/run_pipeline.py --label baseline --files 4                    # 4 of 18 parquet files
  python scripts/run_pipeline.py --label tuned --files 4 --workers 10 --budget-batches --pin --prefetch
  python scripts/run_pipeline.py --label full --workers 10 --budget-batches --pin --prefetch --copy-to-drive

Data must already be on local disk (src/speechserve/pipeline/download.py). Shards are written to
/content/data/shards/<label>/ (local) and, with --copy-to-drive, copied to
/content/drive/MyDrive/speech-infra/shards/<label>/. Never into git. Each run is saved under its label in
results/pipeline.json: config, per-stage busy time, hours of audio per minute per stage, CPU/GPU
utilisation, and how long stages waited on each other.
"""

import argparse
import json
import multiprocessing as mp
import platform
import shutil
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve.pipeline.decode import decode_row_group  # noqa: E402
from speechserve.pipeline.download import local_files  # noqa: E402
from speechserve.pipeline.vad import VadConfig  # noqa: E402

OUT_JSON = ROOT / "results/pipeline.json"
LOCAL_SHARDS = Path("/content/data/shards")
DRIVE_SHARDS = Path("/content/drive/MyDrive/speech-infra/shards")


class Sampler(threading.Thread):
    """System-wide CPU % and GPU utilisation once per second."""

    def __init__(self):
        super().__init__(daemon=True)
        import psutil
        import pynvml
        pynvml.nvmlInit()
        self.psutil, self.nv, self.h = psutil, pynvml, pynvml.nvmlDeviceGetHandleByIndex(0)
        self.cpu, self.gpu, self.gpu_mem = [], [], []
        self.stop = threading.Event()

    def run(self):
        self.psutil.cpu_percent(None)
        while not self.stop.wait(1.0):
            self.cpu.append(self.psutil.cpu_percent(None))
            self.gpu.append(self.nv.nvmlDeviceGetUtilizationRates(self.h).gpu)
            self.gpu_mem.append(self.nv.nvmlDeviceGetMemoryInfo(self.h).used / 2**30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--files", type=int, default=None, help="use only the first N parquet files")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--inflight-per-worker", type=int, default=3)
    ap.add_argument("--batch-clips", type=int, default=16, help="fixed clips per GPU batch (ignored with --budget-batches)")
    ap.add_argument("--budget-batches", action="store_true", help="batch by padded-sample budget instead of clip count")
    ap.add_argument("--max-batch-samples", type=int, default=2_400_000)
    ap.add_argument("--bucket-buffer", type=int, default=512)
    ap.add_argument("--pin", action="store_true")
    ap.add_argument("--prefetch", action="store_true")
    ap.add_argument("--compile", action="store_true", help="torch.compile the SNAC encoder conv stack")
    ap.add_argument("--clips-per-shard", type=int, default=2500)
    ap.add_argument("--copy-to-drive", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    files = local_files()[: args.files] if args.files else local_files()
    vad_cfg = VadConfig()
    tasks = [(str(f), rg, vad_cfg) for f in files for rg in range(pq.ParquetFile(f).metadata.num_row_groups)]
    out_dir = LOCAL_SHARDS / args.label
    if out_dir.exists():
        shutil.rmtree(out_dir)

    # CPU workers first (spawned before CUDA is initialised in this process)
    pool = mp.get_context("spawn").Pool(args.workers)
    pool.map(abs, range(args.workers))                         # make sure workers are up before timing

    from speechserve.pipeline.shard import ShardWriter
    from speechserve.pipeline.tokenize import Encoder, TokenizeConfig, make_batches
    import torch
    import vllm  # noqa: F401  (version for the record)

    tcfg = TokenizeConfig(batch_clips=None if args.budget_batches else args.batch_clips,
                          max_batch_samples=args.max_batch_samples, bucket_buffer=args.bucket_buffer,
                          pin_memory=args.pin, prefetch=args.prefetch, compile=args.compile)
    enc = Encoder(tcfg)
    writer = ShardWriter(out_dir, args.clips_per_shard)
    sampler = Sampler()

    stats = {"read_s": 0.0, "decode_s": 0.0, "vad_s": 0.0, "orig_s": 0.0, "resampled": 0, "dropped": []}
    wait_decode_s = 0.0          # main thread blocked waiting for decoded clips (GPU idle for lack of input)
    buffer, pending, todo = [], deque(), deque(tasks)

    def submit():
        while todo and len(pending) < args.workers * args.inflight_per_worker:
            pending.append(pool.apply_async(decode_row_group, (todo.popleft(),)))

    def flush_buffer():
        nonlocal buffer
        if buffer:
            for clip, l0, l1, l2 in enc.encode_batches(make_batches(buffer, tcfg)):
                writer.add(clip, l0, l1, l2)
            buffer = []

    sampler.start()
    t_start = time.perf_counter()
    submit()
    while pending:
        t = time.perf_counter()
        r = pending.popleft().get()
        wait_decode_s += time.perf_counter() - t
        submit()
        for k in ("read_s", "decode_s", "vad_s", "orig_s", "resampled"):
            stats[k] += r[k]
        stats["dropped"] += r["dropped"]
        buffer += r["clips"]
        if len(buffer) >= tcfg.bucket_buffer:
            flush_buffer()
    flush_buffer()
    torch.cuda.synchronize()
    wall = time.perf_counter() - t_start
    sampler.stop.set()
    sampler.join()
    pool.close()
    pool.join()

    orig_h, trimmed_h = stats["orig_s"] / 3600, writer.audio_s / 3600
    worker_busy = stats["read_s"] + stats["decode_s"] + stats["vad_s"]
    per_min = lambda hours, seconds: round(hours / (seconds / 60), 2) if seconds else None  # noqa: E731
    manifest = writer.manifest({
        "source": "LibriTTS-R train-clean-100 (blabble-io/libritts_r, split train.clean.100), CC BY 4.0",
        "parquet_files": [Path(f).name for f in files], "orig_audio_hours": round(orig_h, 4),
        "dropped_clips": len(stats["dropped"]), "vad": vars(vad_cfg), "snac_model": "hubertsiuzdak/snac_24khz",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds")})

    import numpy as np
    run = {
        "label": args.label,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0), "model": "hubertsiuzdak/snac_24khz (encoder)",
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "python_version": platform.python_version(), "cpu_count": mp.cpu_count(),
        "config": {**{k: v for k, v in vars(args).items()}, "vad": vars(vad_cfg), "tokenize": vars(tcfg),
                   "parquet_files": len(files), "row_groups": len(tasks)},
        "data": {"clips_in": manifest["clips"] + len(stats["dropped"]), "clips_out": manifest["clips"],
                 "dropped": len(stats["dropped"]), "resampled": stats["resampled"],
                 "orig_audio_hours": round(orig_h, 3), "trimmed_audio_hours": round(trimmed_h, 3),
                 "trimmed_fraction": round(1 - trimmed_h / orig_h, 4), "tokens": manifest["tokens"],
                 "tokens_per_audio_hour": manifest["tokens_per_audio_hour"],
                 "tokens_per_audio_second": manifest["tokens_per_audio_second"],
                 "token_count_mismatches": manifest["token_count_mismatches"], "shards": len(manifest["shards"]),
                 "shard_bytes": sum(s["bytes"] for s in manifest["shards"])},
        "wall_s": round(wall, 1),
        "end_to_end_audio_hours_per_min": per_min(orig_h, wall),
        "stages": {
            # CPU stages: busy time summed over workers; capacity = what `workers` processes could do
            "read_parquet": {"busy_s": round(stats["read_s"], 1), "hours_per_min_capacity": per_min(orig_h, stats["read_s"] / args.workers)},
            "decode": {"busy_s": round(stats["decode_s"], 1), "hours_per_min_capacity": per_min(orig_h, stats["decode_s"] / args.workers)},
            "vad": {"busy_s": round(stats["vad_s"], 1), "hours_per_min_capacity": per_min(orig_h, stats["vad_s"] / args.workers)},
            "tokenize_prep": {"busy_s": round(enc.prep_s, 1), "note": "pad + pin + host->device copy (main or prefetch thread)"},
            "tokenize_gpu": {"busy_s": round(enc.gpu_s, 1), "hours_per_min_capacity": per_min(trimmed_h, enc.gpu_s),
                             "batches": enc.batches, "mean_clips_per_batch": round(manifest["clips"] / max(enc.batches, 1), 1),
                             "padding_overhead": round(enc.padded_samples / max(enc.real_samples, 1) - 1, 4)},
            "shard": {"busy_s": round(writer.write_s, 1), "hours_per_min_capacity": per_min(trimmed_h, writer.write_s)},
        },
        "waits": {"main_waiting_for_decode_s": round(wait_decode_s, 1),
                  "main_waiting_fraction_of_wall": round(wait_decode_s / wall, 3),
                  "worker_utilisation": round(worker_busy / (args.workers * wall), 3)},
        "utilisation": {"cpu_pct_mean": round(float(np.mean(sampler.cpu)), 1) if sampler.cpu else None,
                        "gpu_pct_mean": round(float(np.mean(sampler.gpu)), 1) if sampler.gpu else None,
                        "gpu_mem_gib_peak": round(max(sampler.gpu_mem), 2) if sampler.gpu_mem else None},
        "shards_dir": str(out_dir),
    }
    if args.copy_to_drive:
        t = time.perf_counter()
        dst = DRIVE_SHARDS / args.label
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(out_dir, dst)
        run["drive_copy"] = {"dir": str(dst), "seconds": round(time.perf_counter() - t, 1)}

    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {"runs": {}}
    results["runs"][args.label] = run
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
    s = run["stages"]
    print(f"RESULT {args.label}: {orig_h:.2f} h audio ({run['data']['clips_out']} clips) in {wall:.0f} s = "
          f"{run['end_to_end_audio_hours_per_min']} h/min | decode cap {s['decode']['hours_per_min_capacity']} "
          f"vad cap {s['vad']['hours_per_min_capacity']} gpu cap {s['tokenize_gpu']['hours_per_min_capacity']} h/min "
          f"(batch {s['tokenize_gpu']['mean_clips_per_batch']}, pad {s['tokenize_gpu']['padding_overhead']}) | "
          f"wait-for-decode {run['waits']['main_waiting_fraction_of_wall']:.0%}, worker util {run['waits']['worker_utilisation']} | "
          f"CPU {run['utilisation']['cpu_pct_mean']}% GPU {run['utilisation']['gpu_pct_mean']}% | "
          f"tokens/h {run['data']['tokens_per_audio_hour']} mismatches {run['data']['token_count_mismatches']}")


if __name__ == "__main__":
    main()
