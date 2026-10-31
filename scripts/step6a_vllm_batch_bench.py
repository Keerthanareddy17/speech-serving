"""Step 6a: vLLM decode step time vs batch size, alone on the GPU (no SNAC, no server).

  python scripts/step6a_vllm_batch_bench.py [--preset w4a16_lmhead]  -> results/step6a_vllm_batch_bench.json

N requests start together and each generates exactly STEPS tokens (end-of-speech ignored), so every decode
step runs at batch N. ms/step = wall time / STEPS (prefill of ~25-token prompts is negligible). Compare
with the server's per-token time under load to separate vLLM's own scaling from GPU sharing with SNAC.
"""

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

BATCHES = [1, 2, 4, 8, 16, 32, 64]
STEPS = 300
TEXT = "Hi, I'm testing a realtime speech server on a single L4 GPU."
REALTIME_TOK_S = 7 * 24000 / 2048   # 82.03 tokens/s per stream


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="w4a16_lmhead")
    args = ap.parse_args()

    import torch
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from speechserve import config as C
    from speechserve.engine import build_prompt_ids

    engine_args = C.engine_args(args.preset)
    tok = AutoTokenizer.from_pretrained(C.TOKENIZER)
    prompt = TokensPrompt(prompt_token_ids=build_prompt_ids(tok, TEXT, C.DEFAULT_VOICE))
    llm = LLM(**engine_args)
    base = {k: v for k, v in C.SamplingConfig().to_dict().items() if k not in ("max_tokens", "stop_token_ids", "seed")}

    def run(n):
        params = [SamplingParams(**base, max_tokens=STEPS, ignore_eos=True, stop_token_ids=[], seed=i,
                                 detokenize=False) for i in range(n)]
        t0 = time.perf_counter()
        outs = llm.generate([prompt] * n, params, use_tqdm=False)
        dt = time.perf_counter() - t0
        assert all(len(o.outputs[0].token_ids) == STEPS for o in outs)
        return dt

    rows = []
    for n in BATCHES:
        run(n)                                  # warmup: first batch of a size may JIT-compile (step 3)
        dt = min(run(n) for _ in range(2))
        ms_step = 1000 * dt / STEPS
        per_stream = 1000 / ms_step
        rows.append({"batch": n, "ms_per_step": round(ms_step, 2), "tokens_per_s_total": round(n * per_stream, 1),
                     "tokens_per_s_per_stream": round(per_stream, 1),
                     "realtime_per_stream": per_stream >= REALTIME_TOK_S,
                     "audio_s_per_s_total": round(n * per_stream / REALTIME_TOK_S, 2)})
        print(f"batch {n:3d}: {ms_step:6.2f} ms/step, {n * per_stream:7.1f} tok/s total, "
              f"{per_stream:6.1f} tok/s/stream ({'realtime' if per_stream >= REALTIME_TOK_S else 'too slow'})",
              flush=True)

    out = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0), "model": engine_args["model"], "vllm_version": vllm.__version__,
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "config": {"preset": args.preset, "engine": engine_args, "sampling": {**base, "ignore_eos": True},
                   "steps_per_request": STEPS, "text": TEXT, "engine_class": "offline LLM (no SNAC, no server)",
                   "realtime_tokens_per_s_per_stream": round(REALTIME_TOK_S, 2), "timing": "best of 2 after warmup"},
        "results": rows,
    }
    path = ROOT / "results/step6a_vllm_batch_bench.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
