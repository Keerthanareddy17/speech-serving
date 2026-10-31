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
- We do **not** use the `orpheus-speech` pip wrapper (it pins an old vllm); we reimplement its prompt
  format and decoder logic in our own code.
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
  We pass these ids straight to vLLM (`TokensPrompt`). Upstream decodes them to a string and lets vLLM
  re-tokenize it, which risks a different split.
- **Stop token**: `128258` (end-of-speech), from `realtime_streaming_example/main.py` and the Baseten
  example. The pip wrapper's default `stop_token_ids=[49158]` is not an Orpheus special token, so we
  don't use it.
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

## Tokenization pipeline

## Measurement methodology

- One warmup request before each timed run, so one-time startup costs don't count.
- Fixed sampling seed, so reruns produce the same tokens.
- Every result JSON records date, GPU, model, vllm and torch versions, and the full config.
