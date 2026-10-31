"""Step 1d: torch-profile one generation (same sentence/seed as step 1) and write the raw trace to a temp dir.

  python scripts/profile_decode.py --precision w4a16 --out /tmp/prof_w4a16
Analyse the trace with scripts/analyze_trace.py. Traces are large; they don't go into git.
"""

import argparse
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from step1_generate import ENGINE_ARGS, PRECISIONS, SAMPLING, TEXT, TOKENIZER, VOICE, build_prompt_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", choices=PRECISIONS, default="w4a16")
    parser.add_argument("--out", required=True, help="absolute directory for the trace")
    args = parser.parse_args()
    out = Path(args.out).resolve()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    prompt = TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, TEXT, VOICE))
    llm = LLM(**{**ENGINE_ARGS, **PRECISIONS[args.precision]},
              profiler_config={"profiler": "torch", "torch_profiler_dir": str(out),
                               "torch_profiler_with_stack": False})

    llm.generate(prompt, SamplingParams(max_tokens=16), use_tqdm=False)  # warmup, not profiled
    llm.start_profile()
    o = llm.generate(prompt, SamplingParams(**SAMPLING), use_tqdm=False)[0].outputs[0]
    llm.stop_profile()
    print(f"profiled {len(o.token_ids)} tokens -> {out}")


if __name__ == "__main__":
    main()
