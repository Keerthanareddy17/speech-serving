"""Step 6a micro-benchmark: SNAC decode time vs batch size for our fixed 4-frame windows (SNAC alone on the GPU).

  python scripts/step6a_snac_bench.py   -> results/step6a_snac_batch_bench.json

Each window is what the streaming codec decodes per emitted frame (1 left + 1 emitted + 2 right context
frames = 28 tokens), and emits 2048 samples (85.3 ms) of audio. "call" = what the server's decoder does:
codes to GPU, SNAC decode, audio back to CPU. "gpu" = CUDA-event time of the decode alone.
"""

import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402
from speechserve.codec import SNACDecoder, frames_to_layers  # noqa: E402

BATCHES = [1, 2, 4, 8, 16, 32, 64]
WINDOW_FRAMES = 4
WARMUP, REPS = 20, 100
OUT = ROOT / "results/step6a_snac_batch_bench.json"


def main():
    dec = SNACDecoder()
    model = dec.model
    rng = np.random.default_rng(0)
    results = []
    for b in BATCHES:
        # b different random windows, stacked into one batch: layer shapes (b,4), (b,8), (b,16)
        wins = [rng.integers(0, C.CODEBOOK_SIZE, size=(WINDOW_FRAMES, 7)) for _ in range(b)]
        layers = [np.stack(x) for x in zip(*(frames_to_layers(w) for w in wins))]

        def call():
            with torch.inference_mode():
                codes = [torch.from_numpy(x).to("cuda") for x in layers]
                return model.decode(codes).float().cpu().numpy()

        for _ in range(WARMUP):
            out = call()
        assert out.shape == (b, 1, WINDOW_FRAMES * C.SAMPLES_PER_FRAME), out.shape

        wall = []
        for _ in range(REPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            call()
            wall.append((time.perf_counter() - t0) * 1000)

        codes = [torch.from_numpy(x).to("cuda") for x in layers]
        gpu = []
        with torch.inference_mode():
            for _ in range(REPS):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                model.decode(codes)
                e.record()
                torch.cuda.synchronize()
                gpu.append(s.elapsed_time(e))

        call_ms = float(np.median(wall))
        frame_s = C.SAMPLES_PER_FRAME / C.SAMPLE_RATE
        row = {
            "batch": b,
            "call_ms_p50": round(call_ms, 2), "call_ms_p95": round(float(np.percentile(wall, 95)), 2),
            "gpu_ms_p50": round(float(np.median(gpu)), 2),
            "call_ms_per_window": round(call_ms / b, 3),
            "windows_per_s": round(b / call_ms * 1000, 1),
            # one window -> one emitted frame of audio
            "audio_s_per_s_ceiling": round(b / call_ms * 1000 * frame_s, 2),
            "realtime_streams_ceiling": round(b / call_ms * 1000 * frame_s, 1),
        }
        results.append(row)
        print(f"batch {b:3d}: call {row['call_ms_p50']:6.2f} ms (p95 {row['call_ms_p95']:6.2f}), gpu "
              f"{row['gpu_ms_p50']:6.2f} ms, {row['call_ms_per_window']:6.3f} ms/window, "
              f"ceiling {row['audio_s_per_s_ceiling']:6.1f} audio-s/s", flush=True)

    base = results[0]["call_ms_p50"]
    out = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": C.SNAC_MODEL,
        "vllm_version": "n/a (SNAC alone; no vLLM engine loaded)",
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"window_frames": WINDOW_FRAMES, "tokens_per_window": WINDOW_FRAMES * 7, "dtype": "float32",
                   "noise": "on (model default)", "warmup_calls": WARMUP, "timed_calls": REPS,
                   "call": "numpy codes -> cuda, SNAC.decode, .float().cpu().numpy() (what the server does)",
                   "gpu_alone": True},
        "results": results,
        "speedup_vs_batch1_per_window": {r["batch"]: round(base * r["batch"] / r["call_ms_p50"], 1) for r in results},
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
