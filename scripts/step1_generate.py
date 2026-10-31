"""Step 1: generate one WAV with Orpheus 3B (vLLM offline, bf16) + SNAC on a single GPU. No server, no streaming.

Prompt format, special token ids and the token -> SNAC code mapping are taken from canopyai/Orpheus-TTS
(commit e64661fe6d):
  - orpheus_tts_pypi/orpheus_tts/engine_class.py  `_format_prompt`        (prompt layout)
  - orpheus_tts_pypi/orpheus_tts/decoder.py       `turn_token_into_id`    (code = N - 10 - (i % 7) * 4096
                                                                           for "<custom_token_N>")
  - orpheus_tts_pypi/orpheus_tts/decoder.py       `convert_to_audio`      (7 tokens/frame -> 3 SNAC layers)
  - realtime_streaming_example/main.py            stop_token_ids=[128258] (end of speech)
"""

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import vllm
from snac import SNAC
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

MODEL = "canopylabs/orpheus-3b-0.1-ft"
# Upstream engine_class.py loads the tokenizer from the pretrained repo; it is the same Llama-3 tokenizer.
TOKENIZER = "canopylabs/orpheus-3b-0.1-pretrained"
SNAC_MODEL = "hubertsiuzdak/snac_24khz"
TEXT = "Hi, I'm testing a realtime speech server on a single L4 GPU."
VOICE = "tara"
SAMPLE_RATE = 24_000

ROOT = Path(__file__).resolve().parents[1]
OUT_WAV = ROOT / "results/samples/step1_tara.wav"
OUT_JSON = ROOT / "results/step1_generate.json"

# Special token ids (engine_class.py `_format_prompt`)
START_OF_HUMAN = 128259
END_OF_TEXT = 128009
END_OF_HUMAN = 128260
START_OF_AI = 128261
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258  # stop token, realtime_streaming_example/main.py
# "<custom_token_N>" has id 128256 + N, and decoder.py subtracts 10 more, so code 0 of layer
# position 0 is id 128266. Checked against the tokenizer at runtime below.
AUDIO_TOKEN_BASE = 128266
CODEBOOK_SIZE = 4096

ENGINE_ARGS = dict(
    model=MODEL,
    tokenizer=TOKENIZER,
    dtype="bfloat16",
    # The checkpoint advertises Llama-3.2's 131k context; its KV cache would not fit next to the
    # weights on a 24 GB L4. Prompt + 1200 audio tokens needs far less.
    max_model_len=4096,
    gpu_memory_utilization=0.85,
)
# Sampling defaults from engine_class.py `generate_tokens_sync` (with the corrected stop token).
SAMPLING = dict(
    temperature=0.6,
    top_p=0.8,
    repetition_penalty=1.3,
    max_tokens=1200,
    stop_token_ids=[END_OF_SPEECH],
    seed=0,
)


def build_prompt_ids(tokenizer, text, voice):
    text_ids = tokenizer(f"{voice}: {text}").input_ids  # includes Llama BOS 128000, as upstream
    return [START_OF_HUMAN, *text_ids, END_OF_TEXT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]


def tokens_to_snac_codes(token_ids):
    """Orpheus output ids -> 3 SNAC code layers, following decoder.py `convert_to_audio`."""
    audio = [t for t in token_ids if t >= AUDIO_TOKEN_BASE]
    n_frames = len(audio) // 7
    audio = audio[: n_frames * 7]
    codes = [t - AUDIO_TOKEN_BASE - (i % 7) * CODEBOOK_SIZE for i, t in enumerate(audio)]
    if any(c < 0 or c >= CODEBOOK_SIZE for c in codes):
        raise ValueError("audio token out of range for its frame position")
    f = np.array(codes, dtype=np.int64).reshape(n_frames, 7)
    layer0 = f[:, 0]                          # 1 code/frame  (~12 Hz)
    layer1 = f[:, [1, 4]].reshape(-1)         # 2 codes/frame (~24 Hz)
    layer2 = f[:, [2, 3, 5, 6]].reshape(-1)   # 4 codes/frame (~47 Hz)
    return [torch.from_numpy(x).unsqueeze(0) for x in (layer0, layer1, layer2)], len(audio)


def main():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
    assert tokenizer.convert_tokens_to_ids("<custom_token_10>") == AUDIO_TOKEN_BASE, "token offset mismatch"
    prompt_ids = build_prompt_ids(tokenizer, TEXT, VOICE)

    t0 = time.perf_counter()
    llm = LLM(**ENGINE_ARGS)
    load_s = time.perf_counter() - t0

    # One short warmup request so the timed run excludes first-request overheads.
    llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), SamplingParams(max_tokens=16), use_tqdm=False)

    params = SamplingParams(**SAMPLING)
    t0 = time.perf_counter()
    out = llm.generate(TokensPrompt(prompt_token_ids=prompt_ids), params, use_tqdm=False)[0].outputs[0]
    gen_s = time.perf_counter() - t0

    codes, n_audio_tokens = tokens_to_snac_codes(out.token_ids)

    snac = SNAC.from_pretrained(SNAC_MODEL).eval().to("cuda")
    t0 = time.perf_counter()
    with torch.inference_mode():
        audio = snac.decode([c.to("cuda") for c in codes]).squeeze().float().cpu().numpy()
    torch.cuda.synchronize()
    decode_s = time.perf_counter() - t0

    OUT_WAV.parent.mkdir(parents=True, exist_ok=True)
    sf.write(OUT_WAV, audio, SAMPLE_RATE, subtype="PCM_16")
    duration_s = len(audio) / SAMPLE_RATE

    result = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": MODEL,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"engine": ENGINE_ARGS, "sampling": SAMPLING, "text": TEXT, "voice": VOICE,
                   "snac_model": SNAC_MODEL},
        "model_load_s": round(load_s, 2),
        "generation_s": round(gen_s, 3),
        "snac_decode_s": round(decode_s, 3),
        "generated_tokens": len(out.token_ids),
        "audio_tokens": n_audio_tokens,
        "finish_reason": out.finish_reason,
        "audio_duration_s": round(duration_s, 3),
        "rtf_generation": round(gen_s / duration_s, 3),
        "rtf_total": round((gen_s + decode_s) / duration_s, 3),
        "output_wav": str(OUT_WAV.relative_to(ROOT)),
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")

    print(f"model load time   : {load_s:.1f} s")
    print(f"generation time   : {gen_s:.2f} s  ({len(out.token_ids)} tokens, finish={out.finish_reason})")
    print(f"audio tokens      : {n_audio_tokens}  ({n_audio_tokens // 7} SNAC frames)")
    print(f"SNAC decode time  : {decode_s:.3f} s")
    print(f"audio duration    : {duration_s:.2f} s")
    print(f"real-time factor  : {gen_s / duration_s:.3f} (generation) / {(gen_s + decode_s) / duration_s:.3f} (gen+decode)"
          "  [<1 = faster than realtime]")
    print(f"saved             : {OUT_WAV}  +  {OUT_JSON}")


if __name__ == "__main__":
    main()
