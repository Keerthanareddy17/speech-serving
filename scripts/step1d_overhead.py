"""Step 1d: measure one decode-overhead variant on W4A16 (same sentence/seed as step 1), one per process.

  python scripts/step1d_overhead.py <variant> [--save-wav results/samples/step1d_best.wav]
Appends a row to results/step1d_overhead.json. Variants stack in the order listed in VARIANTS.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Variant env vars must be set before vllm is imported.
VARIANTS = {
    # name: (what changed, precision, engine overrides, sampling overrides, env, warmup)
    "baseline": ("W4A16 as in step 1c: short default-params warmup", "w4a16", {}, {}, {}, "short"),
    "matched_warmup": ("warmup uses the identical SamplingParams (methodology fix)", "w4a16", {}, {}, {}, "matched"),
    "no_detok": ("+ detokenize=False", "w4a16", {}, {"detokenize": False}, {}, "matched"),
    "async_off": ("measure-only: async_scheduling=False (vLLM default is on)", "w4a16",
                  {"async_scheduling": False}, {"detokenize": False}, {}, "matched"),
    "graph_full_decode_only": ("cudagraph_mode FULL_DECODE_ONLY (default FULL_AND_PIECEWISE)", "w4a16",
                               {"compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY"}},
                               {"detokenize": False}, {}, "matched"),
    "flashinfer_sampler_off": ("measure-only: VLLM_USE_FLASHINFER_SAMPLER=0 (vLLM default is on)", "w4a16", {},
                               {"detokenize": False}, {"VLLM_USE_FLASHINFER_SAMPLER": "0"}, "matched"),
    "rep_penalty_1.0": ("measure-only: repetition_penalty 1.0 instead of 1.3 (not adopted)", "w4a16", {},
                        {"detokenize": False, "repetition_penalty": 1.0}, {}, "matched"),
    "lmhead_w4a16": ("+ lm_head quantized W4A16 (GPTQ, same calibration)", "w4a16_lmhead", {},
                     {"detokenize": False}, {}, "matched"),
}

parser = argparse.ArgumentParser()
parser.add_argument("variant", choices=VARIANTS)
parser.add_argument("--save-wav", help="also SNAC-decode and save the audio here")
args = parser.parse_args()
note, precision, engine_over, sampling_over, env, warmup = VARIANTS[args.variant]
os.environ.update(env)

import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402
from snac import SNAC  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402

from step1_generate import (ENGINE_ARGS, PRECISIONS, ROOT, SAMPLE_RATE, SAMPLING, SNAC_MODEL,  # noqa: E402
                            TEXT, TOKENIZER, VOICE, build_prompt_ids, tokens_to_snac_codes)

OUT_JSON = ROOT / "results/step1d_overhead.json"
SAMPLES_PER_FRAME = 2048  # SNAC 24 kHz: one 7-token frame decodes to 2048 samples


def main():
    engine_args = {**ENGINE_ARGS, **PRECISIONS[precision], **engine_over}
    sampling = {**SAMPLING, **sampling_over}
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, TEXT, VOICE))

    llm = LLM(**engine_args)
    warm = SamplingParams(max_tokens=16) if warmup == "short" else SamplingParams(**sampling)
    llm.generate(prompt, warm, use_tqdm=False)

    t0 = time.perf_counter()
    out = llm.generate(prompt, SamplingParams(**sampling), use_tqdm=False)[0].outputs[0]
    gen_s = time.perf_counter() - t0

    codes, n_audio = tokens_to_snac_codes(out.token_ids)
    duration_s = (n_audio // 7) * SAMPLES_PER_FRAME / SAMPLE_RATE
    n = len(out.token_ids)
    row = {
        "variant": args.variant,
        "what_changed": note,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": engine_args["model"],
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"engine": engine_args, "sampling": sampling, "env": env, "warmup": warmup,
                   "text": TEXT, "voice": VOICE},
        "generated_tokens": n,
        "finish_reason": out.finish_reason,
        "generation_s": round(gen_s, 3),
        "ms_per_token": round(1000 * gen_s / n, 2),
        "tokens_per_s": round(n / gen_s, 1),
        "audio_duration_s": round(duration_s, 3),
        "rtf_generation": round(gen_s / duration_s, 3),
    }
    if args.save_wav:
        snac = SNAC.from_pretrained(SNAC_MODEL).eval().to("cuda")
        with torch.inference_mode():
            audio = snac.decode([c.to("cuda") for c in codes]).squeeze().float().cpu().numpy()
        wav = ROOT / args.save_wav
        wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav, audio, SAMPLE_RATE, subtype="PCM_16")
        row["output_wav"] = str(wav.relative_to(ROOT))

    rows = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else []
    rows = [r for r in rows if r["variant"] != args.variant] + [row]
    OUT_JSON.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"RESULT {args.variant:24s} {row['ms_per_token']:6.2f} ms/tok {row['tokens_per_s']:6.1f} tok/s "
          f"RTF {row['rtf_generation']:.3f} ({n} tokens, {out.finish_reason}) | {note}")


if __name__ == "__main__":
    sys.exit(main())
