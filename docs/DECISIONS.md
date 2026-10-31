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

## Streaming engine + SNAC decode (step 2, 2026-10-31)

### Pieces
- `src/speechserve/config.py`: model presets (`bf16`, `w4a16`, `w4a16_lmhead`; default
  `w4a16_lmhead`), engine flags, sampling, the Orpheus token layout, and chunk settings. It contains
  the step 1 values, not re-derived ones. One change: **`max_tokens` 1200 → 3000**. 1200 tokens is
  only ~14.6 s of audio, so longer paragraphs were cut off. Prompt + 3000 still fits in
  `max_model_len=4096`.
- `src/speechserve/engine.py`: `TTSEngine` wraps vLLM 0.32's `AsyncLLM` (the engine core runs in its
  own process). `stream(text, voice)` is an async generator of new token ids
  (`output_kind=DELTA`, `detokenize=False`). Each call is its own vLLM request with a unique id, so one
  instance serves many concurrent streams. A request whose consumer stops early is aborted.
- `src/speechserve/codec.py`: three layers.
  - `FrameAssembler`: ids → 7-code frames.
  - `ChunkPlanner`: when to decode and which samples to keep.
  - `SNACDecoder`: GPU decode on one dedicated worker thread, off the event loop, timed per call.
- Streaming decode follows upstream `decoder.py` `tokens_decoder` / `convert_to_audio`
  (canopyai/Orpheus-TTS @ e64661fe6d). Once 4 frames exist, every new frame triggers a decode of the
  last 4 frames, keeping samples [2048:4096]. So each emitted frame gets 1 frame of left context and
  2 of right context. Generalised: to emit frames [a, b), decode [a−1, b+2).
  `ChunkConfig.first_chunk_frames` / `chunk_frames` set the chunk sizes (tuning comes in step 6). The
  defaults (1/1) reproduce upstream exactly, including never emitting frame 0.
  Deliberate difference from upstream: at end of stream the last 2 frames (~171 ms) are flushed with
  whatever right context exists; upstream drops them.
- `tests/test_chunking.py`: CPU-only checks.
  - The id → code mapping and the 3-layer layout.
  - The default plan equals upstream's windows.
  - With a fake decoder, stitching emits every frame exactly once, in order, for several chunk sizes.

### Single-stream results
`results/step2_single_stream.json`, from `scripts/step2_stream.py`. Each config runs in its own process,
with one warmup stream per text and then 10 timed streams per text. Default chunking (upstream). SNAC
fp32 on the same GPU. p50 shown; p95 is within 1% except where noted.

| config | text | tokens | audio | TTFA | gaps | buffer needed | RTF | SNAC/chunk | SNAC total |
|---|---|---|---|---|---|---|---|---|---|
| w4a16_lmhead | sentence | 498 | 5.97 s | 247 ms | 0 | 0 ms | 0.716 | 16.8 ms | 1.16 s |
| w4a16_lmhead | paragraph | 1345 | 16.30 s | 245 ms | 0 | 0 ms | 0.725 | 15.0 ms | 2.86 s |
| w4a16 | sentence | 421 | 5.04 s | 327 ms | 0 | 0 ms | 0.958 | 17.1 ms | 0.99 s |
| w4a16 | paragraph | 1492 | 18.09 s | 324 ms | 0 (1 run: one 0.3 ms gap) | 0 ms (max 0.3) | 0.963 | 15.2 ms (p95 17.0) | 3.20 s |

- **TTFA is mostly waiting for 4 frames.** Prefill plus 28 tokens is ~230 ms at 8.1 ms/token, or
  ~305 ms at 10.9; one SNAC call adds ~17 ms. Step 6 can only lower it by changing the first window.
- **Both configs stream without stalls, but w4a16 has almost no headroom.** Streaming RTF is
  0.96 vs 0.89 generation-only. The 2 frames of right context give a small built-in lead, which is
  why buffer needed stays ~0.
- **SNAC costs ~0.7–0.8 ms per token of engine time.** Streaming runs at 8.8 ms/token (lmhead) and
  11.7 (w4a16), vs 8.1 / 10.9 generation-only. SNAC decodes on the same GPU while vLLM generates, and
  takes ~15–17 ms for every 4-frame window, about a quarter of the wall time. Most of that is likely
  kernel-launch overhead on tiny inputs, not compute (not profiled yet).
- **Concern for step 3:** one SNAC worker at ~15 ms per frame spends ~180 ms per second of audio per
  stream, so it would saturate at ~5–6 concurrent streams. SNAC decode needs batching across streams
  (or bigger chunks, or graphs/compile) before the server can scale.

### Streaming vs whole-sequence decode (same tokens)
On the last paragraph run, the run's own frames are also decoded in one SNAC call. The comparison
covers frames 1..N−1, since frame 0 is never emitted.

**SNAC 24 kHz is not deterministic:** its config has `noise: true`, so decoder `NoiseBlock`s add
random noise on every decode. So there are three comparisons:

| config | whole vs whole (noise floor) | stream vs whole, noise on | stream vs whole, noise off |
|---|---|---|---|
| w4a16_lmhead | 23.8 dB SNR | 21.4 dB | 25.1 dB, corr 0.9985 |
| w4a16 | 25.7 dB | 22.6 dB | 25.7 dB, corr 0.9987 |

- With noise off, the windowing error (25 dB) is no larger than the difference between two whole
  decodes caused by SNAC's own noise.
- **No clicks.** Across 189 / 210 chunk boundaries:
  - the largest sample-to-sample jump at a boundary is 0.07 / 0.09, below the 99.9th-percentile jump
    anywhere in the clip (0.17 / 0.14);
  - it exceeds the whole decode's jump at the same sample by at most 0.03 / 0.04.
- The error is slightly higher within 64 samples of a boundary (RMS 0.0034 vs 0.0023, noise off).
  That's where the truncated context shows, but it stays small.
- Stitched streams: `results/samples/step2_w4a16_lmhead_stream.wav`, `step2_w4a16_stream.wav`.
- The production path keeps noise on (model default); noise off was used only for this comparison.

### Same seed, different tokens: engine class and first request
Step 2 produced 421 / 498 tokens for the step 1 sentence, while step 1d produced 435 / 463. A check
(w4a16, one process per engine, two requests each):

| engine | 1st request after load | 2nd request |
|---|---|---|
| offline `LLM` | 386 (max_tokens 1200) | 435 (max_tokens 3000) |
| `AsyncLLM` | 491 (max_tokens 1200) | 421 (max_tokens 3000) |

- `max_tokens` has no effect (step 1d got 435 at 1200 after a warmup).
- Each engine class is deterministic after warmup, but `LLM` and `AsyncLLM` give different
  sequences. Most likely their different defaults lead to different kernel shapes and numerics
  (not investigated further).
- **The first request after load differs from later ones in its tokens, not only its speed.** Warmup
  is needed for reproducible tokens as well as timings.
- So step 2 token counts are comparable within step 2, not with step 1.

## Websocket server (step 3, 2026-10-31)

### Design
- `src/speechserve/server.py` (FastAPI, one uvicorn worker). Startup loads **one** `TTSEngine`
  (default `w4a16_lmhead`) and **one** SNAC decoder, shared by every connection. vLLM batches the
  concurrent requests.
- `/ws/tts`: the client sends `{text, voice}`. The server sends binary PCM16 24 kHz chunks as they
  are decoded, then `{"type": "done", ...}` with that request's timings: TTFA, tokens, audio s, RTF,
  `queue_ms` (vLLM scheduler wait, from vLLM's per-request `queued_ts`/`scheduled_ts`), SNAC ms and
  SNAC wait ms per chunk. A connection can send several requests in a row. A client that disconnects
  mid-stream has its vLLM request aborted, and the log shows it as `disconnected`, not as an error.
- Decoding goes through the `ChunkDecoder` interface (`codec.py`). The baseline `SNACDecoder` runs one
  SNAC call per window, with every stream's calls queued on **one** worker thread, off the event loop.
  Step 6 can swap in a batched decoder without touching the server.
- `/health`; `/metrics` returns active streams, total requests, errors, recent TTFA p50/p95, GPU
  memory and utilisation (NVML, device-wide), and the SNAC worker's call count, busy time and pending
  calls.
- `src/speechserve/metrics.py`: one JSON line per request in `/content/logs/server.jsonl`, outside
  the repo. It records text length, not the text.
- `scripts/run_server.py` (run in tmux window `server`); `scripts/ws_client.py` (`-n` for concurrent
  clients).

### Warmup: first batch of a new size stalled the engine ~3 s
The first time 3 requests ran together, TTFA was 3.3 s. The same 3 requests again took 0.26–0.29 s.
The first batch of 4 again took 3.0 s. New batch sizes with staggered starts, and new combined prompt
lengths, did not reproduce it.

Cause: vLLM 0.32's Triton top-p sampler (`v1/sample/ops/topk_topp_triton.py`) splits each row over
`S` blocks. `S` is a compile-time constant picked from the batch size and the SM count
(`_topp_split_count`; on the L4's 58 SMs: batch 1 → 16, 2–3 → 8, 4–7 → 4, 8–14 → 2, 15–29 → 1). Above
batch 64 a different kernel is used.
- The first batch with a new `S` compiles 3 kernels (~3 s), and the whole engine stalls while it
  does, so every active stream stalls too.
- vLLM's JIT monitor only warns once per kernel name, so the later compiles were silent.
- vLLM's own JIT warmup runs but compiles nothing here ("finished in 0.00s").
- A first batch of 32 also compiled something, although its `S` equals batch 16's.

Fix: after the normal warmup stream, server startup runs one short (8-token) simultaneous batch for
every size 2–64, plus 80 and 128.
- Kernels are cached on disk (`~/.triton`), so after the first-ever start this takes ~10 s.
- With it, 3 concurrent paragraphs get TTFA 0.26–0.29 s on a fresh server.
- Anyone measuring latency on a fresh vLLM process should warm every batch size they will hit.

### Sanity checks (fresh server, warmup fix in)
| test | client TTFA | server TTFA | RTF | gaps | errors |
|---|---|---|---|---|---|
| sentence ×1 | 248 ms | 247 ms | 0.719 | 0 | 0 |
| paragraph ×1 | 247 ms | 247 ms | 0.729 | 0 | 0 |
| paragraph ×3 at once | 260 / 275 / 291 ms | 259 / 274 / 288 ms | 0.885 | 0 | 0 |
| sentence ×3 at once | 256 / 275 / 292 ms | 256 / 273 / 290 ms | 0.83 | 0 | 0 |

- Client and server TTFA agree within 3 ms, so websocket and localhost overhead is negligible.
- With 3 streams the TTFAs are staggered by ~15 ms: their first SNAC calls queue behind each other
  (per-chunk SNAC wait of 0 / 5 / 20 ms).
- `results/samples/step3_ws.wav` (the paragraph, received over the socket) has the same 1345 tokens
  as step 2's run and the same length. It correlates 0.998 with `step2_w4a16_lmhead_stream.wav` (the
  difference is SNAC's noise), with no clipping.
- A client disconnect mid-stream: logged as `disconnected`, errors 0, GPU idle within a second
  (request aborted).

## Tokenization pipeline

## Measurement methodology

- One warmup request before each timed run, so one-time startup costs don't count.
- Fixed sampling seed, so reruns produce the same tokens.
- Every result JSON records date, GPU, model, vllm and torch versions, and the full config.
