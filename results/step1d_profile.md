# Step 1d profile: W4A16 decode on the L4

- **Date:** 2026-10-31
- **Setup:** NVIDIA L4; model `orpheus-3b-w4a16` (GPTQ W4A16, group 128, lm_head bf16); vLLM 0.32.0; torch 2.13.0+cu130.
- **Run:** same sentence/voice/seed/sampling as step 1 (temperature 0.6, top_p 0.8, repetition_penalty 1.3); batch size 1.
- **Method:** vLLM's built-in torch profiler (`profiler_config={"profiler": "torch"}`, stack tracing off), wrapped around one request after a warmup. Reproduce with `scripts/profile_decode.py` + `scripts/analyze_trace.py`. The raw 7 MB trace is not kept.
- **Caveat:** the profiler adds some CPU-side overhead. Kernel durations are reliable.

## In plain words

Each decode step produces one audio token and takes **~10.8 ms**. During that time the GPU is busy
**99%** of the time. There is essentially no idle gap between kernels: CUDA graphs plus vLLM's async
scheduling keep the next step's CPU preparation hidden behind the current step's GPU work.

Almost all of the 10.8 ms goes to reading weights from GPU memory:

- **6.0 ms: the 4-bit Marlin matmuls.** 112 per step: 28 layers × (QKV, output, gate/up, down).
  They read ~1.5 GB of packed weights at ~250 GB/s, which is close to the L4's practical bandwidth.
- **3.8 ms: lm_head.** One bf16 matrix-vector product of 3072 × 156,940 that turns the final hidden
  state into a score for every vocabulary entry. It is ~0.96 GB, so a third of each step goes to
  that single matrix. It is the obvious next target.
- **~0.8 ms:** attention, norms and small fused ops.
- **0.13 ms: sampling and logits processing** (repetition penalty, temperature, top-p, gumbel
  sampling). This is negligible.

**So the "~9.5 ms of non-weight overhead" from step 1c does not exist inside the engine.** The engine
produces tokens at ~92 tok/s, already above the ~82 tok/s realtime line.

## Where the step 1c number came from

The 17.8 ms/token in step 1c was created by the measurement script. From step 1b on,
`step1_generate.py` redirected stdout/stderr at the OS level while the engine started, to read its
startup log (weights and KV cache sizes). The engine process inherited that redirection.

An A/B run of the same script with *only* the redirection removed went from 7.70 s to **4.76 s** for
the same 435 tokens. The mechanism is not identified: the engine wrote nothing per step to the log.

It added ~7 ms of CPU work per step in the engine loop. Async scheduling hides CPU work only while it
is shorter than the GPU step:

- bf16 (~27 ms of GPU per step) was unaffected: 12.60 s with and without the redirection.
- The faster quantized runs were slowed. All fp8/fp8_kv/fp8_marlin/w4a16 speeds in step 1b/1c were
  too slow and are re-measured in step 1d.

The redirection was removed in step 1d. Memory numbers from step 1b/1c (weights, KV capacity, kernel
names) came from that startup log and are unaffected.

Trace: `rank0.1793427765676970756.pt.trace.json.gz`, 435 decode steps (1 token each), batch size 1.

## Per decode step (averages)

| | ms/step | share |
|---|---|---|
| **wall time (GPU decode window / steps)** | **10.81** | 100% |
| GPU busy (union of kernels) | 10.75 | 99% |
| GPU idle (gaps between kernels) | 0.06 | 1% |
| &nbsp;&nbsp;kernels: model forward | 6.81 | 63% |
| &nbsp;&nbsp;kernels: lm_head | 3.78 | 35% |
| &nbsp;&nbsp;kernels: sampling | 0.13 | 1% |
| &nbsp;&nbsp;kernels: bookkeeping | 0.02 | 0% |

## Where the idle time sits (gap between kernel categories)

| gap | ms/step |
|---|---|
| model forward -> model forward | 0.02 |
| bookkeeping -> bookkeeping | 0.02 |
| sampling -> sampling | 0.01 |
| bookkeeping -> model forward | 0.00 |
| sampling -> bookkeeping | 0.00 |
| bookkeeping -> sampling | 0.00 |

## Top 10 kernels by GPU time

| kernel | category | calls/step | ms/step |
|---|---|---|---|
| `void marlin::Marlin<...>` | model forward | 112 | 6.034 |
| `std::enable_if<...>` | lm_head | 1 | 3.781 |
| `void flash::flash_fwd_splitkv_kernel<...>` | model forward | 28 | 0.274 |
| `void flash::flash_fwd_splitkv_combine_kernel<...>` | model forward | 28 | 0.167 |
| `triton_red_fused_fused_add_rms_norm_2` | model forward | 28 | 0.076 |
| `void vllm::reshape_and_cache_flash_kernel<...>` | model forward | 28 | 0.067 |
| `triton_red_fused_fused_add_rms_norm_0` | model forward | 28 | 0.065 |
| `_topp_sb_step_kernel` | sampling | 5 | 0.050 |
| `triton_poi_fused_3` | model forward | 27 | 0.047 |
| `triton_poi_fused_mul_silu_slice_1` | model forward | 28 | 0.037 |
