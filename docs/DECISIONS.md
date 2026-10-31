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

## Baseline load test (step 4, 2026-10-31)

### Method
`scripts/run_loadtest.py` → `src/speechserve/loadtest.py`, against the step 3 server (w4a16_lmhead,
per-stream SNAC on one worker thread). The client runs in a separate process on the same machine.
- Each of the N streams opens its own websocket. Even-numbered streams start with the sentence, odd
  ones with the paragraph, then each alternates.
- The first request of each stream is **warmup** (not counted). After it, a stream keeps starting
  **measured** requests until 60 s after the level start, and completes at least 2 measured requests
  (one of each text).
- That minimum is a deliberate deviation from a pure fixed duration. At N=64 a single paragraph takes
  ~3 min, so a 60 s window would have measured only one kind of text.
- A stream that is done measuring keeps sending unmeasured **fillers** until every stream is done, so
  every measured request runs with all N streams active. Then all sockets close and the server aborts
  the fillers.
- Per request: client TTFA, client RTF, gaps and buffer needed (step 2 definitions), errors, plus the
  server's `done` timings.
- Throughput: audio seconds received per wall second, counted from the end of the last warmup to the
  end of the last measured request (all N streams busy).
- Server-side state comes from `/metrics`, polled once a second. "SNAC busy" = the SNAC worker's
  busy-time increase divided by wall time over the same window.
- Files: `results/baseline_sweep.csv` (one row per request, 415 rows), `results/baseline_sweep_levels.csv`
  (one row per level), `results/baseline_sweep.json` (levels + metadata + verdict).

### Results
| N | measured reqs | TTFA p50 / p95 | RTF p50 / p95 | reqs with gaps | buffer needed p95 | throughput (audio s/s) | SNAC busy | SNAC wait/chunk p50 | SNAC ms/call | client CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 247 / 249 ms | 0.73 / 0.73 | 0% | 0 ms | 1.38 | 0.25 | 0.1 ms | 15.2 | 0.5% |
| 2 | 13 | 273 / 293 ms | 0.80 / 0.81 | 0% | 0 ms | 2.50 | 0.45 | 1.1 ms | 15.5 | 0.7% |
| 4 | 20 | 333 / 381 ms | 0.94 / 0.96 | 15% (3 reqs, one 0.2–2.5 ms gap each) | 1.2 ms | 4.29 | 0.77 | 9.6 ms | 15.5 | 1.0% |
| 8 | 24 | 458 / 531 ms | 1.42 / 1.46 | 100% | 7.7 s | 5.72 | 1.00 | 103 ms | 15.2 | 1.3% |
| 16 | 32 | 565 / 712 ms | 2.61 / 2.89 | 100% | 30 s | 6.00 | 1.00 | 207 ms | 14.0 | 1.3% |
| 32 | 64 | 846 / 1047 ms | 4.66 / 5.89 | 100% | 72 s | 6.61 | 1.00 | 385 ms | 12.6 | 1.4% |
| 64 | 128 | 1400 / 2155 ms | 9.35 / 11.6 | 100% | 163 s | 6.68 | 1.00 | 789 ms | 12.5 | 1.4% |

- Errors: 0 at every level. vLLM scheduler queue time: 0 ms at every level (p95).
- GPU memory: 20.16 GiB, flat at every level. vLLM preallocates its KV cache, so this number doesn't
  change with load.
- NVML GPU utilisation reads 100% even at N=1 (it only measures whether *any* kernel is running), so
  it is not a saturation signal here. It falls to 82% at N=64, while vLLM is starved.

**Baseline max streams: 2** (highest N with p95 RTF < 1.0 and no request with a gap).
- N=4 misses only on gaps: 3 of 20 requests each had one gap of 0.2–2.5 ms, and p95 RTF was 0.96.
- With a client playback buffer of a few ms, N=4 would pass.

### Where it breaks, and why
**Per-stream SNAC decoding is the bottleneck, as predicted.** The data confirms the 5–6 stream
estimate as the hard ceiling. In practice it breaks earlier because the SNAC calls queue.
- **The ceiling is SNAC calls per second.** All streams share one SNAC worker thread, and each 85 ms
  frame of audio costs one ~15 ms call. The worker can therefore produce at most about
  1 / 15 ms × 85 ms ≈ **5.6 s of audio per second**, for all streams together. Realtime for N streams
  needs N audio s/s, so N ≤ ~5.6.
- **The data matches that ceiling:**
  - SNAC busy grows with load (0.25 / 0.45 / 0.77 at N = 1 / 2 / 4) and reaches 1.00 from N=8 on.
  - Throughput then flattens at 5.7–6.7 audio s/s.
  - RTF grows roughly as N / 6.6: at N=64 every stream gets ~0.1 s of audio per second.
- **Why it breaks at 4, not ~5.6:** SNAC is a single queue. At 77% busy, a chunk sometimes waits
  behind 3 other streams' decodes (SNAC wait p50 rises from 0.1 ms to 9.6 ms), and at a 1.07× realtime
  margin per stream an occasional wait shows up as a 1–2 ms gap.
- **Throughput rises from 5.7 to 6.7 past saturation, because SNAC calls get faster**
  (15.2 → 12.5 ms). Once SNAC is the limit, vLLM finishes its tokens early and leaves the GPU to SNAC,
  so there is less contention between the two.
- **TTFA grows with N because the first chunk waits behind every other stream's decode**
  (~N × 12.5 ms at saturation: 789 ms SNAC wait at N=64). vLLM never queued a request.
- **It's not the language model.** vLLM queue time is 0 at every level, and GPU utilisation drops as N
  grows, so vLLM waits on SNAC, not the other way round. Its own limit is still unmeasured, because
  SNAC saturates first.
- **It's not the client.** It used at most 1.4% of one CPU core.
- **What to do next (step 6):** batch SNAC across streams, one call for every stream's pending window,
  through the `ChunkDecoder` interface. Bigger chunks (fewer calls per audio second) and graphs or
  compile for SNAC are the other levers.

### Warmup buckets (update to step 3)
On a later restart the Triton disk cache did not help: compiles happened again at batch 2, 4, 8, 15
and 30 (plus 80 for the kernel used above 64). Warming every size 2–64 is what makes startup safe;
15 and 30 also show that more than `S` depends on batch size.

## Quality baseline (step 5, 2026-10-31)

### Harness
`scripts/run_quality.py` → `src/speechserve/quality.py`.
- **Sentences:** `src/speechserve/quality_sentences.txt`, 30 fixed sentences: short, numbers/dates/
  money/codes, multilingual names, punctuation and questions, long, technical, 2 tongue twisters.
  None overlap the GPTQ calibration set. **Keep this file fixed.**
- **Generation:** one process per config, through the server's path (`TTSEngine.stream` →
  `stream_audio`, per-stream SNAC), voice `tara`, seeds 0 and 1, one clip at a time (batch 1, so
  reproducible), after a warmup request.
  - Clips and manifest go to `/content/drive/MyDrive/speech-infra/quality/baseline/<config>/` (~14 MB
    each).
  - 3 examples per config (sentences 4, 10, 28; seed 0) go to `results/samples/quality/`.
- **Transcription:** faster-whisper 1.2.1 `large-v3`, fp16 on GPU, beam 5, English,
  `condition_on_previous_text=False`, in its own process after generation. Two local workarounds:
  - PyAV 19 breaks faster-whisper's file reader (`metadata_errors`), so clips are read with soundfile
    and resampled 24 → 16 kHz with scipy.
  - CTranslate2 needs CUDA 12 `libcublas.so.12`, which is installed (`nvidia-cublas-cu12`) but not on
    the loader path, so it is preloaded by full path.
- **Scoring:** Whisper's English normaliser (`WhisperTokenizer.normalize`) on both sides, then jiwer
  WER/CER per clip.
  - That normaliser handles written numbers unevenly: "4:30" → "4 30" but Whisper's "4.30" stays;
    "6:05" → "6 5"; "3rd" kept on one side, "3" on the other; "555-0142" → "555 142" vs "5550142".
  - So a supplementary **format-insensitive** WER (`make_format_insensitive`) also removes separators
    between digits, ordinal suffixes and the "p m"/"pm" split, on both sides.
- **"Within noise" test:** paired bootstrap over the 30 sentences (mean of the 2 seeds per sentence),
  10,000 resamples, 95% CI of the mean-WER difference vs bf16. Plus the seed-to-seed WER change within
  each config.
- **Outputs:** `results/quality_baseline.json`, `results/quality_baseline_clips.csv` (180 rows).

### Results (60 clips per config, all ended on end-of-speech)
| config | WER mean | median | p90 | clips > 20% | CER | corpus WER | format-insensitive WER mean / corpus | clips > 20% (fmt) |
|---|---|---|---|---|---|---|---|---|
| bf16 | 2.5% | 0 | 10.0% | 2 | 0.9% | 2.5% | 1.6% / 1.4% | 1 |
| w4a16 | 4.2% | 0 | 15.4% | 5 | 1.0% | 3.9% | 1.3% / 1.2% | 1 |
| w4a16_lmhead | 2.9% | 0 | 10.2% | 3 | 0.7% | 2.7% | 0.8% / 0.7% | 0 |

Difference vs bf16 (mean WER, 95% CI, paired over sentences):

| | official normaliser | format-insensitive |
|---|---|---|
| w4a16 − bf16 | +1.75 pts [−0.14, +4.30] | −0.24 pts [−2.49, +1.60] |
| w4a16_lmhead − bf16 | +0.40 pts [−0.62, +1.73] | −0.75 pts [−2.28, +0.30] |

- Seed noise (mean absolute WER change between seeds, same sentence): 1.0–2.5 points.
- Audio lengths match within 3%: 304 / 299 / 308 s total.

### What the errors are
- **Formatting (all configs alike):** the time, date, ordinal and phone-number cases above. They
  account for most clips over 20%, and they explain w4a16's higher official WER: its two phone-number
  clips were transcribed "5550142" (WER 0.40 each), versus "555 80142"-style for the others.
- **Whisper errors:** "…came back to life **thank you**" (a classic hallucination on trailing
  silence; bf16) and "Rahman" for "Raman" (a spelling variant; lmhead).
- **Likely model (pronunciation) slips, 1 word each:**
  - bf16: gate "47 B" → "D", "the" → "a", "Tomasz" → "Thomas", "peck" → "pack", two odd phone-number
    readings.
  - w4a16: "engineers" → "engineer" (both seeds), "nvidia-smi" → "smile" (both seeds), "OAuth" →
    "olof", "Nakamura" → "Makamura", "picked" → "cacked", "peck" → "pack".
  - w4a16_lmhead: "CUDA" → "kruda", "surely" → "shorely", "peck" → "pack", one phone-number reading.
- **Every config** mangles "Peter Piper … peck" at least once. The tongue twister is hard for Orpheus
  itself, not a quantization effect.

### Verdict
**w4a16_lmhead is not meaningfully worse than bf16; it stays the default.**
- Its WER difference is +0.4 points on the official normaliser and −0.75 on the format-insensitive
  one. Both CIs include 0 and are smaller than the seed-to-seed noise.
- w4a16 is also within noise. Its official +1.75 points comes from the phone-number formatting case.
- **Resolution:** this harness only detects differences of about ±2 WER points (30 sentences × 2
  seeds).
- **Not measured:** WER measures intelligibility, not naturalness, prosody or codec artefacts. That
  needs a listen (examples in `results/samples/quality/`) or an automatic MOS model later.

## Batched SNAC decoding (step 6a, 2026-10-31)

### Micro-benchmark: what batching can buy
`results/step6a_snac_batch_bench.json` (`scripts/step6a_snac_batch_bench.py`): SNAC alone on the GPU,
fp32, our 4-frame windows. Each window emits one 85 ms frame.

| batch | ms per call | ms per window | ceiling (audio s/s) |
|---|---|---|---|
| 1 | 7.4 | 7.4 | 11.5 |
| 4 | 7.4 | 1.85 | 46 |
| 8 | 8.3 | 1.04 | 82 |
| 16 | 18.8 | 1.18 | 73 |
| 64 | 92.8 | 1.45 | 59 |

- **Alone, a batch-1 call is 7.4 ms, half the ~15 ms seen in the server.** Sharing the GPU with vLLM
  already doubled it in step 4.
- **Up to batch 8 a call costs about the same as batch 1.** GPU time equals wall time, so this isn't CPU
  launch overhead: the kernels are latency-bound on tiny inputs.
- **Past batch 8 it becomes compute-bound** at ~1.2–1.45 ms per window. Batching raises SNAC's ceiling
  by ~5–7×.

### Implementation
`BatchedChunkDecoder` (`codec.py`), behind the same `ChunkDecoder` interface. Selected with
`run_server.py --decoder batched`; `per_stream` stays the default.
- Streams put their pending window in one queue. One worker takes all windows of the same length
  (oldest first, up to `--snac-max-batch`, default 64), stacks them (no padding), runs **one** SNAC call
  (`SNACDecoder.decode_many`) and resolves each stream's future with its slice.
- Windows queue up while a batch decodes, so batches grow with load. `--snac-max-wait-ms` (default 0)
  optionally holds an idle worker to gather more. Windows from streams that disconnected are skipped.
- Server startup also runs one SNAC call per batch size 1..max_batch (4 s).
- `/metrics` adds `snac_windows_total` (so batch mean = Δwindows / Δcalls) and vLLM's own Prometheus
  counters: generated tokens, inter-token latency, running and waiting requests, KV usage.

### Correctness
- `tests/test_batched_decoder.py`:
  - logic on CPU with a fake model: grouping by length, the cap, each stream gets its own slice, and
    cancelled streams are skipped;
  - real SNAC with noise off: batched output = per-window output.
- **TF32 finding:** PyTorch lets cuDNN run fp32 convolutions in TF32 (`cudnn.allow_tf32=True` by
  default), and cuDNN may pick a different algorithm per batch shape. With TF32 on, batch-64 and
  single decodes differ by up to 1.65e-3. With TF32 off they are identical (0.0 at batch 64,
  ≤ 1.7e-6 at batch 16).
  - Against a float64 CPU decode, TF32 single and TF32 batched are equally far off (8.6e-4).
    Batching costs no accuracy; TF32 itself is the approximation.
  - The test disables TF32 and requires ≤ 1e-5. The server keeps PyTorch's default; TF32 on/off is a
    separate measurement for later.
- `results/samples/step6a_batched.wav` (paragraph via the batched server): same 1345 tokens and length
  as `step3_ws.wav`, correlation 0.998 (SNAC noise), no clipping.

### Sweep (same settings as step 4)
`results/batched_sweep.csv` / `_levels.csv` / `.json`.

| N | TTFA p50 / p95 | RTF p50 / p95 | reqs with gaps | buffer p95 | throughput (audio s/s) | SNAC busy | SNAC batch mean | SNAC ms/call | vLLM ms/token | client CPU |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 245 / 245 ms | 0.72 / 0.72 | 0% | 0 | 1.39 | 0.25 | 1.0 | 15.2 | 8.7 | 0.5% |
| 2 | 277 / 286 ms | 0.80 / 0.81 | 0% | 0 | 2.53 | 0.40 | 1.1 | 15.3 | 9.5 | 0.7% |
| 4 | 315 / 341 ms | 0.89 / 0.93 | 0% | 0 | 4.53 | 0.52 | 1.6 | 15.8 | 10.6 | 1.0% |
| 8 | 336 / 383 ms | 1.02 / 1.08 | 78% | 1.3 s | 7.86 | 0.42 | 3.7 | 16.8 | 12.2 | 1.6% |
| 16 | 460 / 603 ms | 1.48 / 1.52 | 100% | 9.2 s | 11.0 | 0.61 | 3.6 | 20.0 | 17.6 | 2.0% |
| 32 | 746 / 970 ms | 2.27 / 2.44 | 100% | 25 s | 14.4 | 0.61 | 6.2 | 36.2 | 26.8 | 2.2% |
| 64 | 1352 / 1466 ms | 3.93 / 4.30 | 100% | 55 s | 16.6 | 0.61 | 13.9 | 61.7 | 46.5 | 2.4% |

Errors 0 at every level. GPU memory 21.26 GiB (+1.1 GiB of SNAC batch activations kept by PyTorch's
caching allocator).

Side by side with the baseline:

| N | baseline RTF p95 / gaps | batched RTF p95 / gaps | baseline throughput | batched throughput |
|---|---|---|---|---|
| 1 | 0.73 / 0% | 0.72 / 0% | 1.38 | 1.39 |
| 2 | 0.81 / 0% | 0.81 / 0% | 2.50 | 2.53 |
| 4 | 0.96 / 15% | **0.93 / 0%** | 4.29 | 4.53 |
| 8 | 1.46 / 100% | 1.08 / 78% | 5.72 | 7.86 |
| 16 | 2.89 / 100% | 1.52 / 100% | 6.00 | 11.0 |
| 32 | 5.89 / 100% | 2.44 / 100% | 6.61 | 14.4 |
| 64 | 11.6 / 100% | 4.30 / 100% | 6.68 | 16.6 |

**Max streams: 2 → 4.** N=8 fails at RTF p50 1.02 (5–7 were not tested). The throughput ceiling went
from 6.7 to 16.6 audio s/s (2.5×), and TTFA at N=64 dropped from 1.40 s to 1.35 s (p95 2.16 → 1.47 s).

### The new bottleneck: SNAC and vLLM sharing one GPU
`results/step6a_vllm_batch_bench.json` (`scripts/step6a_vllm_batch_bench.py`): vLLM alone (offline
`LLM`, no SNAC), N requests decoding together.

| N | vLLM alone ms/step | vLLM in server ms/token | ratio |
|---|---|---|---|
| 1 | 7.9 | 8.7 | 1.1× |
| 8 | 9.0 | 12.2 | 1.4× |
| 16 | 10.2 | 17.6 | 1.7× |
| 32 | 13.0 | 26.8 | 2.1× |
| 64 | 19.8 | 46.5 | 2.3× |

- **It's not vLLM's own scaling.** Alone, vLLM stays at or under the 12.2 ms realtime limit up to
  ~28 streams (10.2 ms at 16, 13.0 at 32).
- **It's not the decode worker running out of capacity.** SNAC busy plateaus at ~0.61, so the worker
  is idle ~40% of the time and its queue wait stays ~1 ms.
- **It's the two sharing the GPU.**
  - Each slows the other: vLLM's step time grows to 1.4–2.3× its standalone value, and SNAC calls take
    2–4× longer than alone (61.7 ms in the server for ~14 windows, vs ~17 ms alone).
  - They run in **different processes** (vLLM's engine core, and the server process with SNAC), so
    the GPU time-slices between two CUDA contexts instead of overlapping their kernels.
  - At N=8, vLLM's per-token time lands exactly on the realtime limit, which is why it breaks there.
- **SNAC does about 4× the necessary work:** each emitted frame decodes a 4-frame window. And with
  `max_wait_ms=0`, batches stay small at moderate load (mean 3.6 at N=16), where a call costs nearly
  as much as batch 8.
- **Next levers, in rough order of expected payoff:**
  1. larger `chunk_frames` (e.g. 4 frames per window of 7: 1.75× instead of 4× decode work per frame;
     first chunk unchanged, so TTFA unchanged);
  2. SNAC in fp16/bf16, or compiled / CUDA-graphed;
  3. a small `max_wait_ms` to form bigger batches;
  4. later: SNAC in the same process or CUDA stream as vLLM, or MPS.

  Change one at a time.

### Optimisation log
`results/optimizations.csv` started: step 4 baseline, step 6a batched SNAC (adopted), and fp8
(step 1b, tried, not adopted: 18 ms/token, not realtime even at one stream; W4A16 variants are faster
with no measurable WER loss).

## SNAC load reduction (step 6b, 2026-10-31)

`--decoder batched` is now the server default (`per_stream` stays as a flag). Each change was tried
**one at a time** against a reference screen of the 6a config: N = 4, 6, 8, 30 s per level, same
method as before. Rows are in `results/optimizations.csv`, raw screens in `results/screen/`.
SNAC-only changes were micro-benchmarked first (`results/step6b_snac_bench.json`), and audio-changing
ones were checked against an fp32, noise-off, whole-sequence decode of the same tokens
(`results/step6b_audio_check.json`, `scripts/step6b_audio_check.py`).

### What was tried
| change | max passing (of 4/6/8) | N=8 RTF p95 | vLLM ms/token at 8 | SNAC busy at 8 | decision |
|---|---|---|---|---|---|
| reference (6a: batched, chunk 1, fp32, wait 0) | 6 | 1.047 | 12.16 | 0.39 | — |
| growing chunks, cap 4 | 4 | 1.082 | 12.09 | 0.44 | rejected |
| growing chunks, cap 8 | 6 | 1.058 | 12.33 | 0.46 | rejected |
| SNAC fp16 (eager) | 4 | 1.108 | 12.33 | 0.50 | rejected |
| SNAC CUDA graphs (fp32) | 6 | 1.081 | 12.60 | 0.48 | rejected |
| fp16 + graphs, torch.compile | — | — | — | — | not screened |
| engine core in-process / MPS | — | — | — | — | not possible here |
| max_wait 1 ms | 4 | 1.086 | 12.51 | 0.52 | rejected |
| max_wait 2 ms | 4 | 1.082 | 12.60 | 0.52 | rejected |

**1. Growing later chunks** (`--chunk-frames`, `ChunkPlanner`).
- **Idea:** a chunk of c frames needs a (c+3)-frame window, so SNAC work per emitted frame falls
  from 4× to 1.75× (c=4) or 1.4× (c=8).
- **Trade-off:** a chunk can only be sent once all its frames exist. With a fixed c=4 after the
  1-frame first chunk, the client's 85 ms buffer would run dry and **every stream would gap after the
  first chunk** (~150 ms).
- **Implementation:** chunks are buffer-aware. The server tracks the client's buffer (audio sent
  minus time since the first chunk) and this stream's frame rate. It waits for a full-size chunk only
  if the missing frames are expected before the buffer runs out, minus the last SNAC time and 50 ms.
  Otherwise it sends what it has, in power-of-two pieces. The first chunk is unchanged, so TTFA is
  unchanged.
- **Audio:** bigger chunks agree *better* with the whole decode (SNR 25.1 → 27.7 / 30.5 / 33.2 dB
  for caps 1 / 2 / 4 / 8; no boundary clicks).
- **Rejected:** at the breaking point (N=8) streams run at realtime, so there's no buffer slack and
  chunks stay at 1 frame. Mixed window lengths also split SNAC batches (2.6–2.8 vs 4.1), and SNAC got
  *busier*. Savings only appeared at loads that already pass (N=4 SNAC busy 0.41 → 0.24). Cap 4 also
  produced small gaps at N=6 (≤ 9 ms).

**2. SNAC precision, graphs, compile** (micro-benchmark, 4-frame windows, ms per call alone):

| | b=1 | b=4 | b=8 | b=16 | error vs fp32 |
|---|---|---|---|---|---|
| fp32 | 6.8 | 7.3 | 7.9 | 17.0 | — |
| fp16 | 7.4 | 7.3 | 7.4 | 9.9 | 3.3e-3 rel. RMS; 53.9 dB SNR on the paragraph |
| bf16 | 7.1 | 7.1 | 7.2 | 10.1 | 2.7e-2 (8× fp16's error, so dropped) |
| fp32 compile / fp16 compile | 2.8 / 3.0 | 3.4 / 3.0 | 5.9 / 3.5 | 13.6 / 6.6 | ≈ dtype |
| fp32 graph / fp16 graph | 1.6 / 1.1 | 3.4 / 2.3 | 6.2 / 4.2 | 16.1 / 8.8 | ≈ dtype |

- **The 6a "~7 ms floor" was CPU launch overhead.** Graphs cut batch 1 to 1.1–1.6 ms.
- **fp16 eager: rejected.** It's slower in the server: each new (batch, length) shape costs ~1 s of
  cuDNN setup (warmup 68 s vs 4 s), and unwarmed tail shapes hit it live (TTFA p95 0.9–1.1 s).
- **fp32 graphs: rejected.** They're 2–4× faster alone, but in the server SNAC calls only dropped
  from 16–18 to 13.5 ms. A SNAC call's in-server time is dominated by waiting for its GPU time slice
  next to vLLM's process, not by its own work. Faster calls also shrank batches (2.3 vs 4.1).
- **fp16 + graphs and compile: not screened**, for the same reason: they cut SNAC kernel time, which
  isn't what costs time in the server.
- Two fixes found on the way, both kept in the code:
  - graphs must be captured largest batch first, or the shared pool grows to 2.1 GiB and runs out of
    memory next to vLLM (largest first: 1.31 GiB fp32 / 0.69 GiB fp16);
  - **`--snac-max-frames` (default 320)** caps frames per SNAC call (batch × window length). The cap-8
    warmup (64 × 11-frame windows) ran out of memory without it. It's inactive for 4-frame windows at
    max_batch 64.

**3. Same process: not possible with vLLM 0.32's AsyncLLM.**
- `EngineCoreClient.make_client` raises `NotImplementedError` for asyncio without multiprocessing.
  `VLLM_ENABLE_V1_MULTIPROCESSING=0` only affects the sync `LLMEngine`, which would need a custom
  threaded serving loop and likely lose async scheduling (~2 ms/token).
- **MPS** (the other way to stop time-slicing): `nvidia-cuda-mps-control` exits with code 1 before
  logging, in this container.
- Took ~3 minutes of the 20-minute box; nothing was changed.

**4. `--snac-max-wait-ms` 1 and 2: rejected.** Batches didn't grow (2.8–2.9 vs 4.1 at N=8).
Streams' windows already arrive together, since they come out of the same vLLM step, so the wait
only adds latency (an asyncio sleep of "1 ms" costs ~4 ms here) and caused gaps at N=6.

### Final config and fine sweep
Final config = 6a batched SNAC (now the default), chunk cap 1, fp32 eager, max_wait 0, plus the
inactive frame cap. `results/step6b_final_sweep.csv` / `_levels.csv` / `.json`, 60 s per level:

| N | reqs | TTFA p50 / p95 | RTF p50 / p95 | reqs with gaps | buffer p95 | throughput (audio s/s) | SNAC busy | batch | SNAC ms/call | vLLM ms/token | client CPU |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7 | 247 / 249 ms | 0.72 / 0.73 | 0% | 0 | 1.38 | 0.24 | 1.0 | 15.0 | 8.7 | 0.5% |
| 4 | 24 | 307 / 318 ms | 0.88 / 0.90 | 0% | 0 | 4.57 | 0.45 | 1.8 | 15.6 | 10.5 | 1.0% |
| 5 | 25 | 311 / 340 ms | 0.92 / 0.95 | 0% | 0 | 5.45 | 0.47 | 2.1 | 16.0 | 11.0 | 1.1% |
| **6** | 30 | **325 / 340 ms** | 0.95 / **0.986** | **0%** | 0 | 6.31 | 0.45 | 2.6 | 16.1 | 11.4 | 1.2% |
| 7 | 32 | 324 / 366 ms | 0.99 / 1.04 | 44% | 50 ms | 7.16 | 0.39 | 3.6 | 16.8 | 11.8 | 1.4% |
| 8 | 36 | 342 / 392 ms | 1.06 / 1.11 | 78% | 1.8 s | 7.53 | 0.60 | 2.3 | 16.4 | 12.8 | 1.6% |
| 10 | 45 | 383 / 409 ms | 1.14 / 1.18 | 96% | 3.1 s | 8.85 | 0.45 | 3.8 | 17.4 | 13.6 | 1.7% |
| 12 | 48 | 419 / 466 ms | 1.22 / 1.28 | 96% | 4.5 s | 9.94 | 0.47 | 4.3 | 17.9 | 14.6 | 1.7% |
| 16 | 48 | 438 / 599 ms | 1.47 / 1.49 | 100% | 8.6 s | 11.33 | 0.44 | 6.6 | 22.2 | 17.0 | 1.8% |

**Max streams: 6.** Errors 0 at every level.
- This is not a 6b gain: 6a's sweep jumped from 4 to 8, and the reference screen of the 6a config
  already passed N=6.
- N=7 fails narrowly (RTF p95 1.04, 44% of requests with gaps up to ~50 ms).

**Quality:** `results/step6b_quality.json`. The step 5 harness on w4a16_lmhead through the batched
decoder:
- all 60 clips have the same tokens as step 5 (vLLM is deterministic for sequential seeded requests);
- WER 2.87% (format-insensitive 0.84%), identical to step 5; difference 0.0 on all 30 sentences.

That confirms the decoder path doesn't change intelligibility. It is not an independent quality sample.

`results/samples/step6b_final.wav`: same tokens and length as `step3_ws.wav`, correlation 0.998, no
clipping.

### The limit now
**vLLM's per-token time, inflated by sharing the GPU with SNAC in another process.**
- In the server, vLLM takes 11.4 ms/token at N=6 and 11.8 at N=7. The realtime limit is 12.2, and
  p95 tails cross it at 7. Alone, vLLM would take ~8.6 ms at these batch sizes.
- SNAC is only ~45% busy, so neither part is saturated alone.
- Every SNAC-side change that shrinks SNAC's own work left vLLM's per-token time unchanged.
  The two processes take turns on the GPU, so the time vLLM loses depends on how often SNAC runs, not
  on how much work each call does.
- **Options this step couldn't use:**
  - run SNAC inside vLLM's process, or vLLM inside ours: needs a custom serving loop around the sync
    engine;
  - MPS: blocked in this container;
  - **decode SNAC on the CPU** (12 cores, ~94 windows/s needed at N=8): removes the GPU contention
    entirely; worth a micro-benchmark next.

## First chunk and CPU SNAC (step 6c, 2026-10-31)

### TTFA breakdown at N=1 (6b config, 10 requests, p50 / p95)
The server now returns `ttfa_first_output_ms`, `ttfa_first_window_ms`, `ttfa_first_chunk_ready_ms` and
`first_chunk_decode/wait_ms` per request; the load test copies them into its CSV.

| part | p50 / p95 |
|---|---|
| request → first vLLM output in the server (queue 0, prefill 14.2 ms, IPC) | 16.9 / 18.0 ms |
| **first output → 28th audio token (27 tokens × 7.8 ms)** | **211.8 / 214.0 ms** |
| first SNAC call | 16.0 / 17.7 ms |
| server send + websocket | 0.9 / 1.0 ms |
| client TTFA | 245.5 / 249.4 ms |

86% of TTFA is waiting for the tokens the first window needs. That is the only part a smaller first
chunk can shrink.

### Smaller first chunk (`--first-wait-frames`, server default now 2)
- **W = 4 (upstream):** decode frames 0–3, emit frame 1 (frame 0 is never played).
- **W < 4:** decode frames 0..W−1 and emit **frame 0** with W−1 right frames.
  - W=3 gives frame 0 the standard 2 right frames.
  - W ≤ 2 would leave the client only 85 ms of audio before the next standard window, so the next
    frames may also go out with fewer right frames (≥ W−1). This uses the 6b buffer-aware rule: wait for
    2 right frames only if the client's buffer covers it.
- `ChunkConfig.first_wait_frames` defaults to 4 (the upstream reference used by tests and offline
  tools); the server defaults to 2.

Audio check (`results/step6b_audio_check.json`, variants `first_wait*` and `fw*_<clip>`): noise off,
against the whole-sequence decode. The paragraph starts in near-silence, so 3 clips that start speaking
immediately were added ("Yes, absolutely.", "See you tomorrow at noon.", "Peter Piper…").

| W | first 300 ms SNR (yes / see you / Peter Piper) | right frames, first 6 frames | first-boundary jump | whole-clip SNR |
|---|---|---|---|---|
| 4 | 27.4 / 30.1 / 29.1 dB | 2 2 2 2 2 2 | ≈ whole | 23.5 / 28.5 / 25.6 |
| 3 | 29.0 / 34.7 / 30.0 | 2 2 2 2 2 2 | ≈ whole | same |
| 2 | 26.8 / 31.5 / 27.8 | 1 1 1 1 2 1 | ≈ whole | 23.5 / 28.3 / 25.6 |
| 1 | 8.7 / 11.3 / 8.3 | 0 0 0 0 0 0 | 3–15× whole | 13.8 / 14.0 / 25.5 |

W=1 was rejected on audio and not screened.

Server screens (`results/screen/fw*`):

| W | TTFA N=1 (10 reqs) p50 / p95 | N=4 RTF p95 / TTFA p50 | N=6 RTF p95 / gaps / TTFA p50 |
|---|---|---|---|
| 4 | 246 / 249 ms | 0.876 / 306 ms (6b ref) | 0.960 / 0% / 318 ms (6b ref) |
| 3 | 190 / 194 ms | 0.911 / 247 ms | 0.966 / 0% / 263 ms |
| **2** | **135 / 137 ms** | 0.908 / 189 ms | **0.998** / 0% / 196 ms |

**Adopted W=2**: the smallest size with clean audio and capacity 6. N=6 sits at the edge; the 60 s
final sweep confirmed it (0.996).
- Listening examples: `results/samples/step6c/fw{4,3,2}_{yes,see_you,peter_piper}.wav`. The tokens
  are identical across W, so they differ only in how the start was chunked.
- `results/samples/step6c_first_chunk.wav`: the paragraph at W=2, 85 ms longer than 6b's (frame 0),
  otherwise correlation 0.998.

**Quality** (`results/step6c_quality.json`, the step 5 harness unchanged, W=2):
- WER 3.1% vs 2.87%; difference **+0.24 points, 95% CI [−0.53, +1.01]**; format-insensitive +0.10.
  Within noise (seed noise 2.3 points).
- Same tokens on all 60 clips; only the audio differs (frame 0 played, SNAC noise, early context).

### Speech starts ~0.6 s into the audio (finding, nothing built)
In the 60 step 5 w4a16_lmhead clips, audible speech (first 20 ms window above 10% of the clip's loud
level) starts p10 389 / **p50 634** / p90 923 ms into the played audio. Orpheus generates a lead-in of
near-silence.
- Time to first *speech* at N=1 is therefore ~TTFA + 0.6 s: ~880 ms before 6c, ~770 ms now (frame 0
  adds 85 ms of lead-in back).
- Sending generated frames only once speech starts could bring first speech forward by ~0.3 s at N=1.
  Candidate for later.

### Prefix caching (record only)
- On by default in vLLM 0.32 (`enable_prefix_caching=True`, block size 16).
- Our prompts share only a 5-token voice prefix (`[128259, 128000, "t", "ara", ":"]`), less than one
  block, so different texts get no reuse. Identical repeated prompts do (e.g. the load test's two
  texts); prefill is ~14 ms, so the effect is small.

### SNAC on the CPU (time box 25 min, used ~5): rejected
`results/step6c_snac_cpu_bench.json`, fp32, 12 cores.
- Best total: **47.6 windows/s** (12 processes × 1 thread, batch 8, 2.0 s per call) = ~4 streams.
- Best at low delay: 38.9 windows/s (12 threads, batch 4, 103 ms per call).
- N=8 needs ~94/s and N=12 ~141/s. Cost per window rises with batch size on the CPU.
- The server flag was not built (stop rule).

### Final 6c sweep (`results/step6c_final_sweep.*`, 60 s per level)
| N | TTFA p50 / p95, 6b → 6c | RTF p95, 6b → 6c | gaps, 6b → 6c | throughput, 6b → 6c | vLLM ms/token, 6b → 6c | SNAC busy, 6b → 6c |
|---|---|---|---|---|---|---|
| 1 | 247 / 249 → **137 / 138 ms** | 0.729 → 0.729 | 0 → 0% | 1.38 → 1.38 | 8.74 → 8.78 | 0.24 → 0.28 |
| 4 | 307 / 318 → 184 / 198 | 0.903 → 0.927 | 0 → 0% | 4.57 → 4.51 | 10.5 → 10.7 | 0.45 → 0.57 |
| 5 | 311 / 340 → 194 / 207 | 0.953 → 0.979 | 0 → 0% | 5.45 → 5.37 | 11.0 → 11.3 | 0.47 → 0.59 |
| **6** | 325 / 340 → **187 / 214** | 0.986 → **0.996** | 0 → **0%** | 6.31 → 6.18 | 11.4 → 11.7 | 0.45 → 0.61 |
| 7 | 324 / 366 → 208 / 220 | 1.036 → 1.038 | 44 → 56% | 7.16 → 7.07 | 11.8 → 12.0 | 0.39 → 0.53 |
| 8 | 342 / 392 → 206 / 236 | 1.108 → 1.066 | 78 → 67% | 7.53 → 7.92 | 12.8 → 12.2 | 0.60 → 0.51 |
| 10 | 383 / 409 → 222 / 244 | 1.18 → 1.17 | 96 → 98% | 8.85 → 8.89 | 13.6 → 13.6 | 0.45 → 0.52 |
| 12 | 419 / 466 → 218 / 245 | 1.28 → 1.21 | 96 → 89% | 9.94 → 10.3 | 14.6 → 14.1 | 0.47 → 0.46 |
| 16 | 438 / 599 → 256 / 270 | 1.49 → 1.48 | 100 → 100% | 11.3 → 11.4 | 17.0 → 17.1 | 0.44 → 0.59 |

**Max streams: 6** (unchanged), errors 0 at every level. TTFA fell by ~110–200 ms at every level.
- **The price is a thinner margin at N=4–6** (RTF p95 +0.01–0.03, SNAC busy +0.1). During each
  stream's start-up ramp, its 3-frame windows can't share a SNAC call with the others' 4-frame windows,
  so there are more calls and vLLM's per-token time rises slightly.
- **Limit unchanged** (see 6b): vLLM's per-token time, inflated by sharing the GPU with SNAC in another
  process.

## Tokenization pipeline (step 7, 2026-10-31)

### Data
- **Source:** LibriTTS-R train-clean-100 (CC BY 4.0) from the HF parquet mirror `blabble-io/libritts_r`,
  split `train.clean.100`: 18 parquet files, 9.24 GB, 33,232 clips of 24 kHz WAV bytes, with
  normalised text, speaker, chapter and id.
- **Download:** to local disk (`/content/data/libritts_r/`) at ~300 MB/s, the whole split in **28 s**.
  The time box didn't apply, so **all 53.55 h** were used.

### Pipeline (`src/speechserve/pipeline/`, `scripts/run_pipeline.py`)
1. **decode** (CPU worker processes; one parquet row group of ~100 clips per task): WAV bytes →
   mono float32. All clips were 24 kHz, so 0 were resampled.
2. **vad** (same workers, timed separately): webrtcvad, aggressiveness 2, 30 ms frames, on a 16 kHz
   copy. Speech = runs of ≥ 3 voiced frames; 100 ms kept on each side; clips < 0.5 s after trimming are
   dropped. No splitting: LibriTTS-R is already sentence-length. The clips are already tight, so
   trimming removed only 0.2% of the audio; 119 clips (0.36%) were dropped.
3. **tokenize** (GPU): SNAC 24 kHz encode.
   - Clips are bucketed by length (buffer of 512, sorted) and batched up to a padded-sample budget;
     padding overhead is 7.7%.
   - Each clip's codes are trimmed back to its own ceil(len / 2048) frames.
   - Optional pinned memory, prefetch thread and `torch.compile`.
4. **shard:** codes → Orpheus ids via the new `codec.layers_to_frames` + `frames_to_token_ids`, the
   exact inverse of what the server decodes (round-trip test in `tests/test_chunking.py`).
   - Format: zstd parquet shards of 2,500 clips (id, speaker_id, chapter_id, text, duration_s,
     orig_duration_s, n_frames, n_tokens, tokens list<int32>), plus `manifest.json`.
   - Written to `/content/data/shards/<label>/`, then copied to
     `/content/drive/MyDrive/speech-infra/shards/<label>/`. Never in git.

### Measurement and the fix (`results/pipeline.json`)
First 4 of 18 files (12.28 h of audio) for before/after, then the full split:

| run | batching | pin / prefetch / compile | wall | end-to-end h/min | GPU-stage capacity h/min | clips/batch | GPU util | GPU mem peak |
|---|---|---|---|---|---|---|---|---|
| baseline (4 files) | 16 clips | – / – / – | 134 s | 5.48 | 5.76 | 15.7 | 93.5% | 22.1 GiB |
| tuned (4 files) | 4M samples | ✓ / ✓ / ✓ | 140 s | 5.27 | 5.39 | 24.5 | 77.2% | 6.6 GiB |
| **tuned, no prefetch** (4 files) | 4M samples | ✓ / – / ✓ | 120 s | **6.14** | 6.42 | 24.5 | 86.4% | 6.5 GiB |
| tuned, 8M (4 files) | 8M samples | ✓ / – / ✓ | 121 s | 6.10 | 6.47 | 49.1 | 85.5% | 22.2 GiB |
| **full (18 files, 53.55 h)** | 4M samples | ✓ / – / ✓ | 481 s | **6.68** | 6.90 | 25.3 | 94.0% | 8.0 GiB |

In every run, 4 CPU workers could have decoded 357–474 h/min and VAD'd 67–68 h/min. Workers were
busy 11–13% of the time, system CPU was 14–16%, and the main thread waited for decoded clips 0.5–3% of
the time. Shard writing could handle ~600–1000 h/min.

**Slowest stage: SNAC encode on the GPU**, ~5.8 h/min at baseline with the GPU 93% busy.
- **Why:** the encoder runs at the full 24 kHz sample rate. A profile showed **~60% of its GPU time
  is element-by-element arithmetic** (27% the fused Snake activation, 34% other elementwise kernels),
  17% depthwise convolutions and only 17% matrix-multiply convolutions. So it's limited by memory
  bandwidth, not arithmetic.
- **Options measured** on fixed batches (`encode_microbench` in the JSON):
  - `cudnn.benchmark`: +0%.
  - fp16 autocast: +12%, but changes 3–8% of the codes.
  - bf16: changes 27–42% of the codes.
  - **`torch.compile` of the encoder** (fuses the elementwise chains): **1.25×** with 99.4–99.9% of
    codes identical. Compiling `SNAC.encode` itself hits a dynamo guard bug in its padding code, so only
    `.encoder` is compiled.
- **Fix:** compile the encoder **and** batch by a 4M-sample budget. Compiled kernels are cheaper per
  sample but cost more per call, so they need big batches: on 40 real batches, compile + 4M gave 7.15
  h/min vs 5.6 eager. The budget also caps memory (22 → 6.6–8 GiB).
- **Prefetch hurt (5.27 h/min)**, so it's off. The compiled encoder launches its fused kernels from
  Python, and a prefetch thread competing for Python's lock (GIL) left the GPU idle 23% of the time.
- **8M budget:** no gain (6.10) and memory back to 22 GiB, so 4M stays.
- **Result:** +12% on the same 4 files (5.48 → 6.14 h/min). The full run reached 6.68 h/min, partly
  because the one-time ~7 s compile is spread over 4× more audio.
- **The pipeline is still GPU-bound.** Further gains would need a cheaper encoder (a custom fused
  Snake/depthwise kernel), fp16 codes (they differ from fp32), or a second GPU. CPU stages have 10×
  headroom.

### Output and validation (`results/pipeline.json` "validation", `scripts/validate_shards.py`)
- **Totals:** 33,113 clips, 53.44 h after trimming (53.55 h before), 15,898,071 tokens, 14 shards,
  32.9 MB (zstd).
- **Tokens per audio hour: 297,483** (82.63/s). That's 7 × 24000 / 2048 = 82.03/s, plus each clip's
  last partial frame rounded up (~3.5 tokens per clip).
- **Every clip passed:** n_tokens = 7 × n_frames, n_frames = ceil(duration × 24000 / 2048), and every
  id in its position's range (0 failures).
- **5 random clips decoded back** with SNAC (fp32, noise off) against the original trimmed audio:
  - waveform SNR 1.9–5.8 dB (mean 3.6). A ~1 kbps neural codec keeps how speech sounds, not the exact
    waveform (phase), so this is expected;
  - **log-mel SNR 16.8–19.4 dB (mean 18.1)**, log-mel L1 0.69.
- **Same clips re-encoded alone with eager SNAC:** 97.5–98.4% token agreement (batch padding near clip
  ends + compile), and identical reconstruction quality (log-mel SNR 18.10 vs 18.08 dB).
- **Listening pair:** `results/samples/pipeline/5678_43301_000018_000001_{original,reconstructed}.wav`.

## Measurement methodology

- One warmup request before each timed run, so one-time startup costs don't count.
- Fixed sampling seed, so reruns produce the same tokens.
- Every result JSON records date, GPU, model, vllm and torch versions, and the full config.
