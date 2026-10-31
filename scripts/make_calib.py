"""Step 1c: build a GPTQ calibration set from real Orpheus bf16 outputs (prompt + generated audio tokens).

Each sentence in calib_sentences.txt is spoken by 2 of the 8 voices (rotating), giving 128 sequences.
Output (Drive, not git): /content/drive/MyDrive/speech-infra/calib/orpheus_calib_128.jsonl + manifest.json
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import vllm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

# Same prompt format, engine settings and sampling as step 1 (scripts/ is on sys.path when run directly).
from step1_generate import ENGINE_ARGS, SAMPLING, TOKENIZER, build_prompt_ids

# Voices of the finetune-prod model, from the Orpheus-TTS README.
VOICES = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
VOICES_PER_SENTENCE = 2
# Long sentences need more than step 1's 1200 tokens (~14 s of audio) to finish.
MAX_TOKENS = 2048

SENTENCES = Path(__file__).with_name("calib_sentences.txt")
OUT_DIR = Path("/content/drive/MyDrive/speech-infra/calib")


def main():
    sentences = [s.strip() for s in SENTENCES.read_text().splitlines() if s.strip()]
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    requests = []
    for i, text in enumerate(sentences):
        for j in range(VOICES_PER_SENTENCE):
            voice = VOICES[(i * VOICES_PER_SENTENCE + j) % len(VOICES)]
            requests.append({"id": len(requests), "voice": voice, "text": text, "seed": len(requests)})

    prompts = [TokensPrompt(prompt_token_ids=build_prompt_ids(tokenizer, r["text"], r["voice"])) for r in requests]
    params = [SamplingParams(**{**SAMPLING, "max_tokens": MAX_TOKENS, "seed": r["seed"]}) for r in requests]

    llm = LLM(**ENGINE_ARGS)
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, params)
    gen_s = time.perf_counter() - t0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"orpheus_calib_{len(requests)}.jsonl"
    with out_path.open("w") as f:
        for r, o in zip(requests, outputs):
            gen = o.outputs[0]
            f.write(json.dumps({**r, "prompt_ids": o.prompt_token_ids, "output_ids": list(gen.token_ids),
                                "finish_reason": gen.finish_reason}) + "\n")

    lengths = [len(o.prompt_token_ids) + len(o.outputs[0].token_ids) for o in outputs]
    finish = [o.outputs[0].finish_reason for o in outputs]
    manifest = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": ENGINE_ARGS["model"],
        "vllm_version": vllm.__version__,
        "config": {"engine": ENGINE_ARGS, "sampling": {**SAMPLING, "max_tokens": MAX_TOKENS, "seed": "per-request id"},
                   "voices": VOICES, "voices_per_sentence": VOICES_PER_SENTENCE,
                   "sentences_file": "scripts/calib_sentences.txt", "n_sentences": len(sentences)},
        "n_sequences": len(requests),
        "generation_s": round(gen_s, 1),
        "total_tokens": sum(lengths),
        "seq_len_min": min(lengths),
        "seq_len_median": sorted(lengths)[len(lengths) // 2],
        "seq_len_max": max(lengths),
        "finish_reasons": {k: finish.count(k) for k in set(finish)},
        "file": str(out_path),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
