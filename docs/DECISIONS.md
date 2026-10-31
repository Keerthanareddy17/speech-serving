# Decisions

## Environment (step 1, 2026-10-31)

Colab, NVIDIA L4 (23 GB), driver 580.82.07 (CUDA 13.0), Python 3.13.16, 52 GB RAM.

| Package      | Version        | Notes                                                        |
|--------------|----------------|--------------------------------------------------------------|
| vllm         | 0.32.0         | latest stable at the time; installed from PyPI wheel         |
| torch        | 2.13.0+cu130   | pinned by vllm 0.32.0 (replaced Colab's preinstalled 2.11.0) |
| snac         | 1.2.1          |                                                              |
| transformers | 5.19.0         | pulled down from 5.21 by vllm's `<5.20` pin                  |
| cuDNN        | 9.20 (nvidia-cudnn-cu13 9.20.0.48) | pinned by torch                          |

### Fix needed: stray cuDNN 9.27 (cu12) files
Symptom: vLLM generated fine, but SNAC's first `conv1d` failed with
`CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.

Cause: Colab preinstalls `nvidia-cudnn-cu12` 9.27, and `nvidia-cudnn-cu13` unpacks into the same
`site-packages/nvidia/cudnn/lib/` directory. Installing cu13 9.20 overwrote the shared filenames but
left two 9.27-only sub-libraries (`libcudnn_ext.so.9`, `libcudnn_engines_tensor_ir.so.9`), which
cuDNN 9.20 then loaded and rejected.

Fix (nothing else depends on the cu12 package):
```
pip uninstall -y nvidia-cudnn-cu12
pip install --force-reinstall --no-deps nvidia-cudnn-cu13==9.20.0.48
```

## Model and serving stack

- Model `canopylabs/orpheus-3b-0.1-ft` (gated; HF auth from `~/.cache/huggingface`), loaded with the
  vLLM offline `LLM` class, `dtype=bfloat16`. The checkpoint is stored in fp32 (~15 GB on disk) and cast
  on load (~6.4 GB of weights on GPU).
- Tokenizer from `canopylabs/orpheus-3b-0.1-pretrained`, as upstream `engine_class.py` does.
- The `orpheus-speech` pip wrapper is **not** used (it pins an old vllm); its prompt format and
  decoder logic are reimplemented in this repo's code.
- L4 flags:
  - `max_model_len=4096`: the checkpoint advertises a 131k context, and vLLM must fit KV cache for at
    least one max-length sequence, which does not fit next to the weights on 24 GB.
  - `gpu_memory_utilization=0.85`: leaves room for SNAC on the same GPU. That gives 11.4 GiB of KV
    cache (~107k tokens, about 26 concurrent 4k-token requests).
  - vLLM picked the FLASH_ATTN backend and CUDA graphs by default; no extra flags were needed.

## Codec and streaming

Source of truth: canopyai/Orpheus-TTS @ `e64661fe6d` (2025-12-05).

- **Prompt** (`orpheus_tts_pypi/orpheus_tts/engine_class.py`, `_format_prompt`):
  `[128259] + tokenize("{voice}: {text}") + [128009, 128260, 128261, 128257]`
  (start-of-human, text with Llama BOS, end-of-text, end-of-human, start-of-AI, start-of-speech).
  These ids are passed straight to vLLM (`TokensPrompt`). Upstream decodes them to a string and lets vLLM
  re-tokenize it, which risks a different split.
- **Stop token**: `128258` (end-of-speech), from `realtime_streaming_example/main.py` and the Baseten
  example. The pip wrapper's default `stop_token_ids=[49158]` is not an Orpheus special token, so it
  is not used.
- **Sampling defaults** (`generate_tokens_sync`): temperature 0.6, top_p 0.8, repetition_penalty 1.3,
  max_tokens 1200. The README says repetition_penalty >= 1.1 is required for stable output.
- **Token -> SNAC codes** (`orpheus_tts_pypi/orpheus_tts/decoder.py`, `turn_token_into_id` and
  `convert_to_audio`):
  - code = N - 10 - (i % 7) * 4096 for `<custom_token_N>`, i.e. code = id - 128266 - (i % 7) * 4096.
    The script asserts `<custom_token_10>` == 128266.
  - Each frame of 7 tokens splits into SNAC layers: layer 0 = [0]; layer 1 = [1, 4];
    layer 2 = [2, 3, 5, 6]. Codes must be in [0, 4096).
  - Decoder: `hubertsiuzdak/snac_24khz`; 2048 samples per frame at 24 kHz, so ~11.7 frames/s.
- Step 1 decodes the whole sequence in one SNAC call. Upstream streaming decodes a sliding window of
  the last 4 frames (28 tokens) and keeps samples [2048:4096]; that comes in the streaming step.

## Single-stream speed on the L4 (step 1 finding)

`results/step1_generate.json`: 463 tokens in 12.6 s, about **37 tok/s**. That gives 5.63 s of audio, so
**RTF 2.24**: slower than realtime.

Realtime needs 7 tokens x 11.7 frames/s, about **82 tok/s** per stream. Each decoding step of a single
stream reads all ~6.4 GB of bf16 weights, and the L4 has ~300 GB/s of memory bandwidth, so one stream
tops out around 45 tok/s. **bf16 Orpheus 3B cannot stream in realtime on one L4.** Candidates to
measure next, one at a time: FP8 weights (Ada supports FP8; halves bytes per token), and batching
(throughput rises with concurrency because the weights are read once per step for all streams).

## Precision check (step 1b, 2026-10-31)

`results/step1b_precision.json`, from `scripts/step1_generate.py --precision {bf16,fp8,fp8_kv}`. Same
sentence and seed as step 1, one fresh process per run, batch size 1.

> **Corrected in step 1d.** The step 1b/1c speeds were measured while `step1_generate.py` captured the
> engine's stdout/stderr to read its startup log. That capture added ~7 ms/token of CPU work to the
> engine loop, which shows up once the GPU step is shorter than that (bf16 was unaffected). The tok/s
> below are re-measured without it (2026-10-31, step 1d). The old numbers are in brackets and under
> `superseded` in the JSON. Memory and kernel columns come from the original runs' startup log and are
> unaffected. See `results/step1d_profile.md`.

| precision | tok/s [old] | ms/token | RTF (gen) | weights | KV cache            | attention  | load time |
|-----------|-------------|----------|-----------|---------|---------------------|------------|-----------|
| bf16      | 36.5 [36.7] | 27.4     | 2.25      | 6.2 GiB | 11.4 GiB, 107k tok  | FLASH_ATTN | 44 s      |
| fp8       | 55.5 [40.1] | 18.0     | 1.48      | 3.73 GiB| 13.2 GiB, 124k tok  | FLASH_ATTN | 98 s      |
| fp8 + fp8 KV | 55.5 [41.3] | 18.0  | 1.48      | 3.73 GiB| 12.8 GiB, 239k tok  | FLASHINFER | 140 s     |

- The bf16 rerun matched step 1 exactly (463 tokens, 12.60 s), so the seed is deterministic. The
  capture didn't affect bf16 (see the correction above).
- **fp8 does not reach realtime (~82 tok/s), but it scales with bytes read as expected:** 1.5x faster
  for 40% fewer weight bytes. The original "fp8 is inefficient at batch 1" hypothesis came from the
  capture artefact and is withdrawn.
- fp8 weights are 3.73 GiB, not ~3.1, because the tied 156,940 x 3072 embedding / lm_head
  (~0.9 GiB) stays bf16 and is still read every step.
- The fp8 KV cache roughly doubles KV capacity (239k vs 124k tokens). That matters for concurrency,
  not single-stream speed. vLLM switched attention to FLASHINFER for it.
- Online fp8 quantization adds ~1–1.5 min to model load.
- Same seed, different weights: fp8 produced a different token sequence (442 tokens, 5.38 s). All
  three WAVs have sane levels (no clipping, ~60% voiced); quality still needs a listen or a WER check.
## fp8 weight-only via Marlin (step 1c, 2026-10-31)

vLLM 0.32 has no `VLLM_TEST_FORCE_FP8_MARLIN` any more. Its replacement is the engine setting
`linear_backend="marlin"` (CLI `--linear-backend marlin`), which `--precision fp8_marlin` uses. The
default fp8 path on the L4 picks `CutlassFP8ScaledMMLinearKernel` (W8A8); with marlin it picks
`MarlinFP8ScaledMMLinearKernel` (fp8 weights, bf16 activations).
Result (corrected in step 1d): **61.7 tok/s** [43.2] vs 55.5 for CUTLASS fp8. Weight-only is ~10%
faster at batch 1; not enough for realtime.

## W4A16 GPTQ (step 1c, 2026-10-31)

### Calibration set: the model's own outputs
GPTQ rounds weights so that each layer's output stays accurate *for the inputs it sees during
calibration*. Orpheus spends almost all its steps predicting audio tokens, so calibrating on ordinary
text would protect the wrong behaviour. `scripts/make_calib.py` uses bf16 vLLM, the step-1 prompt
format and the step-1 sampling to generate 128 sequences:
- 64 sentences (`scripts/calib_sentences.txt`, 2–45 words, with questions, numbers, instructions,
  narrative, technical and support dialogue, and some emotive tags);
- each spoken by 2 of the 8 voices (each voice 16 times), one seed per sequence;
- `max_tokens=2048`.

All 128 ended on end-of-speech. That's 56k tokens, 125–890 per sequence (median 411), in 50 s. Saved
as prompt + generated ids to `/content/drive/MyDrive/speech-infra/calib/orpheus_calib_128.jsonl`
(+ `manifest.json`).

### Quantization environment: separate venv
- llm-compressor must not share vLLM's environment. 0.13–0.15 need compressed-tensors >= 0.18, but
  vllm 0.32 pins `compressed-tensors==0.17.0`. 0.12.0.3 (compressed-tensors 0.17.1, the same format
  family vLLM reads) pins torch 2.12 / transformers 5.10.
- So it lives in `/content/venvs/quant` (`scripts/setup_quant_venv.sh`, `requirements-quant.txt`).
- Two Colab traps:
  - `python -m venv` fails (no `ensurepip`), so use `uv venv`.
  - Colab sets `UV_SYSTEM_PYTHON=true`, which makes `uv pip install` write into the **system**
    python even with `VIRTUAL_ENV` set. That happened once: it downgraded torch, transformers,
    triton, tokenizers, compressed-tensors and cuda-toolkit under vLLM. It was repaired by
    reinstalling the exact previous versions with `--no-deps`, and verified with a bf16 rerun that
    reproduced step 1 exactly. The setup script now clears `UV_SYSTEM_PYTHON`, passes
    `--python <venv>`, and aborts after a dry run unless uv targets the venv.

### Quantization
`scripts/quantize_w4a16.py`: `GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])`,
model loaded in bf16, 128 calibration sequences.
- Saved config: int4, symmetric, group size 128, `pack-quantized`, ignore `[lm_head]`.
- Embeddings stay bf16 and stay tied to lm_head.
- Took **559 s** on the L4. Output **2.25 GiB** at
  `/content/drive/MyDrive/speech-infra/models/orpheus-3b-w4a16/` (Drive, not git).
- Record: `results/step1c_quantize.json`.

### Result
vLLM loads it as `CompressedTensorsWNA16` and uses **`MarlinLinearKernel`**. No flags were needed.

Speeds corrected in step 1d (old, capture-inflated numbers in brackets; see the correction note in
the step 1b section):

| precision  | tok/s [old]   | ms/token | RTF (gen) | weights  | KV capacity | linear kernel                  |
|------------|---------------|----------|-----------|----------|-------------|--------------------------------|
| bf16       | 36.5 [36.7]   | 27.4     | 2.25      | 6.2 GiB  | 107k tok    | (unquantized)                  |
| fp8        | 55.5 [40.1]   | 18.0     | 1.48      | 3.73 GiB | 124k tok    | CutlassFP8ScaledMMLinearKernel |
| fp8_marlin | 61.7 [43.2]   | 16.2     | 1.33      | 3.58 GiB | 125k tok    | MarlinFP8ScaledMMLinearKernel  |
| **w4a16**  | **91.0** [56.1] | **11.0** | **0.91** | 2.31 GiB | 138k tok | MarlinLinearKernel             |

- **w4a16 is faster than realtime** (≤ 12.2 ms/token needed). The step 1c report of 17.8 ms/token
  and "~9.5 ms of unexplained overhead" came from the capture artefact.
- Different weights, same seed: w4a16 produced a different token sequence (435 tokens, 5.29 s).
  Listening check: the audio sounds clean.

## Decode overhead and lm_head (step 1d, 2026-10-31)

### Profile
Full summary: `results/step1d_profile.md` (`scripts/profile_decode.py` + `scripts/analyze_trace.py`,
vLLM's torch profiler).

Per decode step on W4A16 the engine needs **10.8 ms with the GPU 99% busy**:
- 6.0 ms: 4-bit Marlin matmuls (~250 GB/s);
- 3.8 ms: bf16 lm_head gemv;
- ~0.8 ms: attention and norms;
- 0.13 ms: sampling.

There is no idle-gap overhead to cut. The step 1c gap came from the measurement script (see
correction above).

### Overhead variants
`results/step1d_overhead.json`, from `scripts/step1d_overhead.py <variant>`. One fresh process per
row, same sentence and seed. Rows from `no_detok` on are stacked on `detokenize=False`.

| variant | what changed | ms/token | tok/s | RTF | kept? |
|---|---|---|---|---|---|
| baseline | W4A16 as in step 1c, short warmup | 10.86 | 92.1 | 0.89 | — |
| matched_warmup | warmup with identical SamplingParams | 10.85 | 92.2 | 0.89 | yes (methodology) |
| no_detok | `detokenize=False` | 10.88 | 91.9 | 0.90 | yes: neutral at batch 1, and the text output is never used |
| async_off | measure-only: `async_scheduling=False` | 12.90 | 77.5 | 1.06 | default (on) kept: worth ~2 ms/token |
| graph_full_decode_only | `cudagraph_mode=FULL_DECODE_ONLY` | 10.97 | 91.1 | 0.90 | no: default FULL_AND_PIECEWISE already graphs decode fully |
| flashinfer_sampler_off | measure-only: `VLLM_USE_FLASHINFER_SAMPLER=0` | 10.95 | 91.3 | 0.90 | default (on) kept; sampling is ~0.13 ms |
| rep_penalty_1.0 | measure-only: repetition_penalty 1.0 | 11.10 | 90.1 | 0.91 | **no**: cost ~0 per token, and without it generation never stopped (ran to the 1200-token limit), confirming upstream's stability note |
| **lmhead_w4a16** | **+ lm_head quantized W4A16** | **8.11** | **123.3** | **0.67** | **yes** |

Best stack: vLLM 0.32 defaults (async scheduling, FULL_AND_PIECEWISE graphs, FlashInfer sampling) +
`detokenize=False` + W4A16 with lm_head quantized. Audio: `results/samples/step1d_best.wav`.

### lm_head quantization
`scripts/quantize_w4a16.py --lm-head`: same GPTQ W4A16 group 128 and same 128 calibration sequences.
Two details:
- **Untie first.** lm_head shares its matrix with the embedding table. The script gives lm_head its
  own copy and sets `tie_word_embeddings=False`, so the embedding lookup stays bf16. Took 576 s.
  Output 2.48 GiB (bigger on disk than the 2.25 GiB w4a16 because the bf16 embedding is now stored
  separately, but each decode step reads ~0.7 GiB less).
- **Name lm_head in the targets.** llm-compressor's `targets=["Linear"]` matches HF's `nn.Linear`,
  including lm_head. vLLM matches by layer name or by substring of its own class names, and its lm_head
  is `ParallelLMHead`, so it loaded lm_head unquantized and failed on `lm_head.weight_packed`. Adding
  `"lm_head"` to the config's targets fixes it (metadata only). The script now does it, and
  `results/step1d_quantize_lmhead.json` records the patch.
- vLLM then runs lm_head on `MarlinLinearKernel` too: weights 2.54 GiB in GPU memory, 136k-token KV.
- Quality risk: lm_head directly scores every next token, so its rounding affects token choice most.
  The levels in `step1d_best.wav` are sane; it needs a listen and, later, a WER check.

### Measurement notes
- KV capacity is now read from vLLM's cache config in the parent process. It differs by a few percent
  from the "GPU KV cache size" startup log line used in step 1b/1c (e.g. fp8 128k vs 124k tokens).
- Audio duration in `step1d_overhead.json` is computed from the frame count
  (frames × 2048 / 24000), which equals the SNAC output length.

## Tokenization pipeline

## Measurement methodology

- One warmup request before each timed run, so one-time startup costs don't count.
- Fixed sampling seed, so reruns produce the same tokens.
- Every result JSON records date, GPU, model, vllm and torch versions, and the full config.
