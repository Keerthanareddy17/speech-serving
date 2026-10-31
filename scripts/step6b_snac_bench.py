"""Step 6b micro-benchmark: SNAC decode cost per call for precision / compile / CUDA-graph variants.

  python scripts/step6b_snac_bench.py [--variants fp32 fp16 bf16 fp32_compile ...]
  -> results/step6b_snac_bench.json

SNAC alone on the GPU. Shapes: our window lengths (4 frames = first chunk / cap 1, 7 frames = cap-4 chunk)
at several batch sizes. Each variant is measured on its own. Error = max |variant - fp32| on the same
windows with noise OFF (fp32 with TF32 as the server runs it), relative to the fp32 signal RMS.
"""

import argparse
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

LENGTHS = [4, 7]
BATCHES = [1, 4, 8, 16, 32]
WARMUP, REPS = 10, 50
OUT = ROOT / "results/step6b_snac_bench.json"


def layers_for(windows):
    return [np.stack(x) for x in zip(*(frames_to_layers(w) for w in windows))]


class Variant:
    """dtype + optional torch.compile (dynamic shapes) or CUDA graphs (one graph per exact shape)."""

    def __init__(self, name, dtype, mode):
        self.name, self.mode = name, mode
        self.dec = SNACDecoder(dtype=dtype)
        self.dec.set_noise(False)
        self.model = self.dec.model
        self.decode_fn = self.model.decode
        if mode == "compile":
            self.decode_fn = torch.compile(self.model.decode, dynamic=True)
        self.graphs = {}

    def _graph(self, shapes):
        static = [torch.zeros(s, dtype=torch.long, device="cuda") for s in shapes]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self.model.decode(static)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(g):
            out = self.model.decode(static)
        return g, static, out

    def __call__(self, layers):
        with torch.inference_mode():
            codes = [torch.from_numpy(x).to("cuda") for x in layers]
            if self.mode == "graph":
                key = tuple(tuple(c.shape) for c in codes)
                if key not in self.graphs:
                    self.graphs[key] = self._graph(key)
                g, static, out = self.graphs[key]
                for dst, src in zip(static, codes):
                    dst.copy_(src)
                g.replay()
                return out.float().cpu().numpy()
            return self.decode_fn(codes).float().cpu().numpy()


VARIANTS = {
    "fp32": ("float32", None), "fp16": ("float16", None), "bf16": ("bfloat16", None),
    "fp32_compile": ("float32", "compile"), "fp16_compile": ("float16", "compile"),
    "fp32_graph": ("float32", "graph"), "fp16_graph": ("float16", "graph"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    inputs = {(n, b): layers_for([rng.integers(0, C.CODEBOOK_SIZE, size=(n, 7)) for _ in range(b)])
              for n in LENGTHS for b in BATCHES}

    ref = Variant("fp32_ref", "float32", None)
    ref_out = {k: ref(v) for k, v in inputs.items()}
    del ref
    torch.cuda.empty_cache()

    results = {}
    for name in args.variants:
        dtype, mode = VARIANTS[name]
        t0 = time.perf_counter()
        v = Variant(name, dtype, mode)
        rows, setup_s = [], 0.0
        for (n, b), layers in inputs.items():
            ts = time.perf_counter()
            for _ in range(WARMUP):          # includes compile / graph capture for this shape
                out = v(layers)
            setup_s += time.perf_counter() - ts
            times = []
            for _ in range(REPS):
                torch.cuda.synchronize()
                t = time.perf_counter()
                v(layers)
                times.append((time.perf_counter() - t) * 1000)
            r = ref_out[(n, b)]
            rms = float(np.sqrt(np.mean(r ** 2)))
            rows.append({"window_frames": n, "batch": b, "call_ms_p50": round(float(np.median(times)), 3),
                         "ms_per_window": round(float(np.median(times)) / b, 3),
                         "max_abs_err_vs_fp32": round(float(np.abs(out - r).max()), 6),
                         "rel_rms_err_vs_fp32": round(float(np.sqrt(np.mean((out - r) ** 2))) / rms, 6)})
        results[name] = {"dtype": dtype, "mode": mode, "setup_s": round(setup_s, 1),
                         "total_s": round(time.perf_counter() - t0, 1), "rows": rows}
        print(name, " ".join(f"{r['window_frames']}x{r['batch']}:{r['call_ms_p50']:.2f}ms" for r in rows),
              f"| max rel rms err {max(r['rel_rms_err_vs_fp32'] for r in rows):.1e} | setup {setup_s:.0f}s",
              flush=True)
        del v
        torch.cuda.empty_cache()

    base = {(r["window_frames"], r["batch"]): r["call_ms_p50"] for r in results.get("fp32", {"rows": []})["rows"]}
    for name, res in results.items():
        if base:
            res["speedup_vs_fp32"] = {f"{r['window_frames']}x{r['batch']}": round(base[(r['window_frames'], r['batch'])] / r["call_ms_p50"], 2)
                                      for r in res["rows"]}
    out = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gpu": torch.cuda.get_device_name(0),
        "model": C.SNAC_MODEL, "vllm_version": "n/a (SNAC alone)", "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"lengths": LENGTHS, "batches": BATCHES, "warmup": WARMUP, "reps": REPS, "noise": "off",
                   "tf32": torch.backends.cudnn.allow_tf32, "reference": "fp32 (TF32 on, as served)",
                   "call": "numpy codes -> cuda, decode, .float().cpu() (as the server)"},
        "results": results,
    }
    prev = json.loads(OUT.read_text()) if OUT.exists() else None
    if prev:
        prev["results"].update(results)
        out["results"] = prev["results"]
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
