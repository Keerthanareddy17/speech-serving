"""Scaling math: how the measured single-L4 numbers translate to a much bigger scale.

Standard library only (no GPU, no numpy). Run:  python scripts/scaling_math.py

Reads MEASURED inputs from results/*.json (never retyped here), combines them with the ASSUMED
constants below, writes results/scaling.json and regenerates docs/SCALING.md between marker comments.
Every number in the doc comes from this script.
"""

import argparse
import datetime
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOC_BEGIN = "<!-- BEGIN GENERATED: scripts/scaling_math.py -->"
DOC_END = "<!-- END GENERATED -->"

# ----------------------------------------------------------------------------------------------
# ASSUMPTIONS. None of these were measured in this project. Each has a one-line source or reason.
# ----------------------------------------------------------------------------------------------

# Training hardware (an "80 GB GPU": H100 SXM chosen as one concrete example).
GPU_MEM_GB = 80                # name of the GPU class in the task: 80 GB cards (A100-80GB / H100-80GB)
USABLE_MEM_FRACTION = 0.90     # rule of thumb: CUDA context, NCCL buffers, fragmentation eat ~10%
PEAK_BF16_TFLOPS = 989.0       # NVIDIA H100 SXM spec sheet, dense bf16 (A100 would be 312)
MFU = 0.40                     # model FLOPs utilisation; 35-50% is typical for well-tuned dense LLM training
NODE_GPUS = 8                  # standard 8-GPU server
NVLINK_GBPS = 450.0            # H100 NVLink: 900 GB/s total = ~450 GB/s per direction per GPU (spec sheet)
INTERNODE_GBPS = 50.0          # one 400 Gb/s NIC per GPU = 50 GB/s (typical DGX-style cluster)
CLUSTER_SIZES = (256, 1024)    # illustrative cluster sizes for wall-clock; assumes perfect linear scaling

# Mixed-precision Adam, bytes per parameter (the ZeRO paper's accounting).
BYTES_WEIGHT_BF16 = 2          # bf16 working copy used in forward/backward
BYTES_GRAD_BF16 = 2            # bf16 gradients
BYTES_MASTER_FP32 = 4          # fp32 master copy of the weights
BYTES_ADAM_M = 4               # fp32 Adam first moment
BYTES_ADAM_V = 4               # fp32 Adam second moment

# Training setup for the activation estimate and communication volumes.
SEQ_LEN = 4096                 # same as the serving max_model_len; ~50 s of audio at 82 tokens/s
MICRO_BATCH = 2                # sequences per GPU per forward/backward
GRAD_ACCUM = 1                 # micro-batches per optimizer step (communication is amortised if > 1)
ACT_COEF = 34                  # bytes per (token * hidden) per layer, bf16, no attention-score matrix
                               # (Korthikanti et al. 2022 "Reducing Activation Recomputation": 34*s*b*h
                               # once flash attention removes the 5*a*s^2*b term). Approximation for SwiGLU.
CKPT_RECOMPUTE_FACTOR = 8 / 6  # checkpointing re-runs the forward pass: 8*N*D instead of 6*N*D FLOPs
CKPT_SAVED_BYTES = 2           # with checkpointing only each layer's bf16 input (2 bytes/elem) is stored
LOGITS_BYTES = 8               # fp32 logits + fp32 logits-gradient per (token * vocab)
LOGIT_CHUNK_TOKENS = 1024      # chunked/fused cross-entropy: logits are computed and discarded this many
                               # tokens at a time (recomputed in backward), so only one chunk is alive.
                               # The chunk size is an assumption; common implementations use 512-4096.

# Model shapes (layers, hidden, ffn, heads, kv heads). head_dim = 128 everywhere.
VOCAB = 156_940                # Orpheus vocabulary (tied embedding / lm_head): docs/DECISIONS.md, step1d_profile.md
MODEL_SHAPES = {
    # Orpheus 3B: layers (28), hidden size (3072) and vocab (156,940) are RECORDED in this repo
    # (docs/DECISIONS.md "tied 156,940 x 3072 embedding", results/step1d_profile.md "28 layers",
    # "3072 x 156,940"). The FFN width (8192) and head counts (24 query / 8 KV) are NOT recorded in
    # the repo; they are the Llama-3.2-3B values, ASSUMED. Cross-check: the layout implies the bf16 weight
    # size that vLLM measured (results/step1b_precision.json, weights_gib). See `orpheus_check` in compute().
    "Orpheus 3B (reference)": dict(L=28, d=3072, ffn=8192, heads=24, kv=8, nominal=None),
    # The two target sizes. Nominal parameter count is what the task names; the shape is OUR assumed
    # Llama-style layout chosen to land near it (only used for activations; params use the nominal count).
    "800M": dict(L=12, d=2048, ffn=4608, heads=16, kv=8, nominal=800e6),
    "4.8B": dict(L=32, d=3584, ffn=9728, heads=28, kv=4, nominal=4.8e9),
}
HEAD_DIM = 128
SNAC_TOKENS_PER_FRAME = 7      # Orpheus: 7 tokens per SNAC frame (docs/DECISIONS.md)
SNAC_SAMPLES_PER_FRAME = 2048  # SNAC 24 kHz
SNAC_SAMPLE_RATE = 24000

# Data / serving assumptions.
DATASET_HOURS = 2_000_000      # the scale named in the task
TARGET_TOKENIZE_DAYS = 7       # illustrative deadline for the tokenization pass
SERVE_CONCURRENCY = (100, 1000)  # the two concurrency targets named in the task
H100_MEM_BW_GBPS = 3350.0      # H100 SXM HBM3, spec sheet
L4_MEM_BW_GBPS = 300.0         # L4 spec sheet (also the number used in docs/DECISIONS.md)

# ----------------------------------------------------------------------------------------------
# Measured inputs (read from results files)
# ----------------------------------------------------------------------------------------------


def _load(rel):
    with open(ROOT / rel) as f:
        return json.load(f)


def load_measured(root=ROOT):
    """Pull every measured input out of the results files. Returns a dict, each value with its source."""
    pipe = _load("results/pipeline.json")["runs"]["full"]
    sweep = _load("results/step6c_final_sweep.json")
    overhead = _load("results/step1d_overhead.json")
    vbench = _load("results/step6a_vllm_batch_bench.json")
    prec = _load("results/step1b_precision.json")

    best = next(r for r in overhead if r["variant"] == "lmhead_w4a16")
    lv = {r["streams"]: r for r in sweep["levels"]}
    passing = sorted(n for n, r in lv.items()
                     if r["rtf_p95"] < 1.0 and r["pct_requests_with_gaps"] == 0 and r["errors"] == 0)
    max_streams = sweep["max_streams"]
    stages = pipe["stages"]
    workers = pipe["config"]["workers"]
    cpu_caps = {k: v["hours_per_min_capacity"] for k, v in stages.items()
                if k in ("read_parquet", "decode", "vad", "shard")}
    vad_cap = cpu_caps["vad"]
    e2e = pipe["end_to_end_audio_hours_per_min"]
    alone = [(r["batch"], r["ms_per_step"]) for r in vbench["results"]]
    rt_tok_s = vbench["config"]["realtime_tokens_per_s_per_stream"]

    return {
        "tokens_per_audio_hour": pipe["data"]["tokens_per_audio_hour"],
        "tokens_per_audio_second": pipe["data"]["tokens_per_audio_second"],
        "pipeline_hours_per_min": e2e,
        "pipeline_gpu_stage_hours_per_min": stages["tokenize_gpu"]["hours_per_min_capacity"],
        "pipeline_gpu_util_pct": pipe["utilisation"]["gpu_pct_mean"],
        "pipeline_workers": workers,
        "pipeline_cpu_stage_caps": cpu_caps,
        "pipeline_cpu_headroom_x": vad_cap / e2e,
        "pipeline_vad_hours_per_min_per_worker": vad_cap / workers,
        "pipeline_audio_hours": pipe["data"]["trimmed_audio_hours"],
        "pipeline_shard_bytes": pipe["data"]["shard_bytes"],
        "pipeline_clips": pipe["data"]["clips_out"],
        "max_streams": max_streams,
        "passing_levels": passing,
        "next_lower_passing": max([n for n in passing if n < max_streams], default=max_streams),
        "ttfa_ms_p50_n1": lv[1]["ttfa_ms_p50"],
        "ttfa_ms_p50_nmax": lv[max_streams]["ttfa_ms_p50"],
        "ttfa_ms_p95_nmax": lv[max_streams]["ttfa_ms_p95"],
        "rtf_p95_nmax": lv[max_streams]["rtf_p95"],
        "vllm_ms_per_token_nmax": lv[max_streams]["vllm_ms_per_token"],
        "first_failing_level": min(n for n in lv if n > max_streams),
        "decode_ms_per_token_b1": best["ms_per_token"],
        "decode_tokens_per_s_b1": best["tokens_per_s"],
        "realtime_tokens_per_s_per_stream": rt_tok_s,
        "vllm_alone_ms_per_step": alone,
        "orpheus_bf16_weights_gib": prec["bf16"]["weights_gib"],
        "gpu": sweep["gpu"],
        "vllm_version": sweep["vllm_version"],
        "torch_version": sweep["torch_version"],
        "serve_model": sweep["model"],
    }


# ----------------------------------------------------------------------------------------------
# Core formulas (pure functions, tested in tests/test_scaling_math.py)
# ----------------------------------------------------------------------------------------------


def params_from_shape(L, d, ffn, heads, kv, vocab=VOCAB, head_dim=HEAD_DIM):
    """Parameter count of a Llama-style decoder with a tied embedding.
    Per layer: attention q,o (d x heads*hd each) + k,v (d x kv*hd each) + SwiGLU MLP 3 * d * ffn."""
    attn = 2 * d * heads * head_dim + 2 * d * kv * head_dim
    mlp = 3 * d * ffn
    return L * (attn + mlp) + vocab * d


def train_state_bytes(n_params):
    """Weights + grads + optimizer state for mixed-precision Adam. 2 + 2 + (4 + 4 + 4) = 16 bytes/param."""
    parts = {
        "weights_bf16": BYTES_WEIGHT_BF16 * n_params,
        "grads_bf16": BYTES_GRAD_BF16 * n_params,
        "master_fp32": BYTES_MASTER_FP32 * n_params,
        "adam_m": BYTES_ADAM_M * n_params,
        "adam_v": BYTES_ADAM_V * n_params,
    }
    parts["optimizer"] = parts["master_fp32"] + parts["adam_m"] + parts["adam_v"]
    parts["total"] = parts["weights_bf16"] + parts["grads_bf16"] + parts["optimizer"]
    return parts


def logits_bytes(tokens, vocab=VOCAB, chunked=False):
    """Naive loss materialises logits for every token. A chunked/fused cross-entropy only keeps
    LOGIT_CHUNK_TOKENS tokens of logits alive at a time."""
    live = min(tokens, LOGIT_CHUNK_TOKENS) if chunked else tokens
    return LOGITS_BYTES * live * vocab


def activation_bytes(shape, seq, batch, checkpointing, chunked_loss=False):
    """Activation memory for one micro-batch on one GPU (no tensor parallelism).

    No checkpointing: every layer keeps its intermediates: L * ACT_COEF * s * b * d.
    Checkpointing:    keep only each layer's input (2 bytes * s*b*d per layer) and recompute one layer
                      at a time in backward, so the peak adds one layer's full intermediates.
    Both add the logits: tokens * vocab * LOGITS_BYTES for a naive loss, or only one chunk of tokens for a
    chunked/fused cross-entropy. Logits do not depend on the number of layers."""
    L, d = shape["L"], shape["d"]
    sbd = seq * batch * d
    if checkpointing:
        layers = L * CKPT_SAVED_BYTES * sbd + ACT_COEF * sbd
    else:
        layers = L * ACT_COEF * sbd
    return layers + logits_bytes(seq * batch, chunked=chunked_loss)


def unit_param_counts(shape, vocab=VOCAB):
    """(params per transformer layer, params in the embedding). FSDP gathers one unit at a time."""
    d = shape["d"]
    per_layer = 2 * d * shape["heads"] * HEAD_DIM + 2 * d * shape["kv"] * HEAD_DIM + 3 * d * shape["ffn"]
    return per_layer, vocab * d


def per_gpu_memory(n_params, shape, strategy, seq=SEQ_LEN, batch=MICRO_BATCH):
    """Per-GPU memory for a strategy dict: kind in ddp|fsdp|hsdp, gpus, tp.

    State (16 bytes/param) is divided by tp (tensor parallel splits every weight) and by the shard-group
    size: 1 for DDP, gpus/tp for FSDP, NODE_GPUS/tp for HSDP (shard inside one node, replicate across
    nodes). Sharded strategies also hold a gathered working set: two layers + the embedding unit in bf16
    (param + grad).
    Tensor parallel with sequence parallelism also divides activations (and the vocab-parallel logits) by tp."""
    kind, gpus, tp = strategy["kind"], strategy["gpus"], strategy.get("tp", 1)
    if kind == "ddp":
        shard = 1
    elif kind == "fsdp":
        shard = gpus // tp
    elif kind == "hsdp":
        shard = NODE_GPUS // tp
    else:
        raise ValueError(kind)
    state = train_state_bytes(n_params)["total"] / (tp * shard)
    if shard > 1:
        per_layer, embed = unit_param_counts(shape)
        work = (2 * per_layer + embed) * (BYTES_WEIGHT_BF16 + BYTES_GRAD_BF16) / tp
    else:
        work = 0.0
    act_ckpt = activation_bytes(shape, seq, batch, True) / tp
    act_ckpt_chunked = activation_bytes(shape, seq, batch, True, chunked_loss=True) / tp
    act_full = activation_bytes(shape, seq, batch, False) / tp
    return {"state": state, "gathered": work, "act_ckpt": act_ckpt, "act_ckpt_chunked": act_ckpt_chunked,
            "act_full": act_full,
            "total_ckpt": state + work + act_ckpt, "total_ckpt_chunked": state + work + act_ckpt_chunked,
            "total_full": state + work + act_full, "shard": shard}


def comm_volumes(n_params, shape, strategy, seq=SEQ_LEN, batch=MICRO_BATCH):
    """Bytes each GPU sends per micro-batch step, split into (intra_node, inter_node).

    Ring collective on a group of size D moves (D-1)/D of the tensor per GPU per pass; an all-reduce is two
    passes (reduce-scatter + all-gather).
      DDP:  all-reduce of bf16 grads:                        2 * (D-1)/D * 2N
      FSDP: all-gather params (fwd) + all-gather (bwd)
            + reduce-scatter grads, each (D-1)/D * 2N/tp:    3 * (D-1)/D * 2N/tp
      HSDP: FSDP inside the node (D=8), then an all-reduce of each GPU's grad *shard* (2N/8) across the
            R node-replicas:                                  2 * (R-1)/R * 2N/8
      TP:   per layer, 4 all-reduces of a (s*b*d) bf16 tensor over the tp group (fwd+bwd), intra-node.
    A flat group larger than one node is treated as inter-node bound (a ring runs at its slowest link)."""
    kind, gpus, tp = strategy["kind"], strategy["gpus"], strategy.get("tp", 1)
    two_n = 2 * n_params
    intra = inter = 0.0
    if kind == "ddp":
        vol = 2 * (gpus - 1) / gpus * two_n
        inter, intra = (vol, 0.0) if gpus > NODE_GPUS else (0.0, vol)
    elif kind == "fsdp":
        d = gpus // tp
        vol = 3 * (d - 1) / d * two_n / tp
        if gpus > NODE_GPUS:
            inter += vol
        else:
            intra += vol
    elif kind == "hsdp":
        r = gpus // NODE_GPUS
        intra += 3 * (NODE_GPUS - 1) / NODE_GPUS * two_n
        inter += 2 * (r - 1) / r * (two_n / NODE_GPUS)
    if tp > 1:
        tensor = seq * batch * shape["d"] * 2
        intra += shape["L"] * 4 * 2 * (tp - 1) / tp * tensor
    return intra, inter


def compute_seconds_per_microstep(n_params, tp=1, seq=SEQ_LEN, batch=MICRO_BATCH):
    """6 * N * tokens FLOPs per micro-batch per data-parallel rank, shared across tp GPUs, at peak * MFU."""
    return 6 * n_params * seq * batch / tp / (PEAK_BF16_TFLOPS * 1e12 * MFU)


def comm_seconds(intra, inter):
    return intra / (NVLINK_GBPS * 1e9) + inter / (INTERNODE_GBPS * 1e9)


def train_flops(n_params, tokens):
    """The 6*N*D rule: 2 FLOPs per parameter per token forward, 4 backward."""
    return 6 * n_params * tokens


def gpu_hours(flops):
    return flops / (PEAK_BF16_TFLOPS * 1e12 * MFU) / 3600


def gpus_needed(streams, per_gpu):
    return math.ceil(streams / per_gpu)


def interp_limit(points, limit):
    """Largest x (linearly interpolated between measured points) where y stays <= limit. points = [(x, y)]."""
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= limit < y1:
            return x0 + (limit - y0) / (y1 - y0) * (x1 - x0)
    return pts[-1][0] if pts[-1][1] <= limit else pts[0][0]


# ----------------------------------------------------------------------------------------------
# Compute everything
# ----------------------------------------------------------------------------------------------

STRATEGIES = [
    ("DDP, 16 GPUs", dict(kind="ddp", gpus=16, tp=1)),
    ("FSDP, 8 GPUs (1 node)", dict(kind="fsdp", gpus=8, tp=1)),
    ("FSDP, 16 GPUs (2 nodes)", dict(kind="fsdp", gpus=16, tp=1)),
    ("HSDP, 8-shard x 2 replicas (16 GPUs)", dict(kind="hsdp", gpus=16, tp=1)),
    ("TP=2 + FSDP over 8 (16 GPUs)", dict(kind="fsdp", gpus=16, tp=2)),
    ("TP=4 + FSDP over 4 (16 GPUs)", dict(kind="fsdp", gpus=16, tp=4)),
]


def compute(m):
    out = {"assumptions": {}, "measured": m}
    usable = GPU_MEM_GB * 1e9 * USABLE_MEM_FRACTION
    tok_micro = SEQ_LEN * MICRO_BATCH

    # a. memory per model
    models = {}
    for name, sh in MODEL_SHAPES.items():
        shape_params = params_from_shape(sh["L"], sh["d"], sh["ffn"], sh["heads"], sh["kv"])
        n = sh["nominal"] or shape_params
        st = train_state_bytes(n)
        a_full = activation_bytes(sh, SEQ_LEN, MICRO_BATCH, False)
        a_ckpt = activation_bytes(sh, SEQ_LEN, MICRO_BATCH, True)
        a_ckpt_ch = activation_bytes(sh, SEQ_LEN, MICRO_BATCH, True, chunked_loss=True)
        models[name] = {
            "n_params": n, "shape": sh, "shape_params": shape_params, "state": st,
            "act_full": a_full, "act_ckpt": a_ckpt, "act_ckpt_chunked": a_ckpt_ch,
            "logits": logits_bytes(tok_micro), "logits_chunked": logits_bytes(tok_micro, chunked=True),
            "one_gpu_full": st["total"] + a_full, "one_gpu_ckpt": st["total"] + a_ckpt,
            "one_gpu_ckpt_chunked": st["total"] + a_ckpt_ch,
        }
    out["models"] = models
    orph = MODEL_SHAPES["Orpheus 3B (reference)"]
    implied = params_from_shape(orph["L"], orph["d"], orph["ffn"], orph["heads"], orph["kv"])
    implied_gib = BYTES_WEIGHT_BF16 * implied / 2 ** 30
    out["orpheus_check"] = {
        "implied_params": implied, "implied_bf16_gib": implied_gib,
        "measured_bf16_gib": m["orpheus_bf16_weights_gib"],
        "relative_diff": implied_gib / m["orpheus_bf16_weights_gib"] - 1,
    }
    out["usable_bytes"] = usable

    # b. strategies
    rows = []
    for name, sh in MODEL_SHAPES.items():
        n = models[name]["n_params"]
        for sname, s in STRATEGIES:
            mem = per_gpu_memory(n, sh, s)
            intra, inter = comm_volumes(n, sh, s)
            comp = compute_seconds_per_microstep(n, s["tp"])
            rows.append({
                "model": name, "strategy": sname, **s, **mem,
                "fits_ckpt": mem["total_ckpt"] <= usable, "fits_full": mem["total_full"] <= usable,
                "fits_ckpt_chunked": mem["total_ckpt_chunked"] <= usable,
                "comm_intra": intra, "comm_inter": inter,
                "comm_s": comm_seconds(intra, inter), "compute_s": comp,
                "dp_size": s["gpus"] // s["tp"],
                "global_batch_tokens": (s["gpus"] // s["tp"]) * tok_micro * GRAD_ACCUM,
            })
    out["strategies"] = rows

    # c. data and compute
    tokens = DATASET_HOURS * m["tokens_per_audio_hour"]
    comp = {}
    for name in ("800M", "4.8B", "Orpheus 3B (reference)"):
        n = models[name]["n_params"]
        fl = train_flops(n, tokens)
        h = gpu_hours(fl)
        comp[name] = {"flops": fl, "gpu_hours": h,
                      "gpu_hours_with_ckpt": h * CKPT_RECOMPUTE_FACTOR,
                      "days_on": {str(c): h / c / 24 for c in CLUSTER_SIZES}}
    out["data"] = {"hours": DATASET_HOURS, "tokens": tokens, "compute": comp,
                   "effective_flops_per_gpu": PEAK_BF16_TFLOPS * 1e12 * MFU}

    # d. pipeline
    h_per_hour = m["pipeline_hours_per_min"] * 60
    l4_hours = DATASET_HOURS / h_per_hour
    shard_bytes_per_audio_hour = m["pipeline_shard_bytes"] / m["pipeline_audio_hours"]
    out["pipeline"] = {
        "l4_hours": l4_hours, "l4_days": l4_hours / 24,
        "l4_gpus_for_deadline": math.ceil(l4_hours / (TARGET_TOKENIZE_DAYS * 24)),
        "wall_days": {str(k): l4_hours / k / 24 for k in (16, 64, 256)},
        "shard_bytes_total": shard_bytes_per_audio_hour * DATASET_HOURS,
        "shard_bytes_per_audio_hour": shard_bytes_per_audio_hour,
        "vad_workers_per_gpu": m["pipeline_hours_per_min"] / m["pipeline_vad_hours_per_min_per_worker"],
    }

    # e. serving
    per_gpu = m["max_streams"]
    cons = max(1, m["next_lower_passing"])
    alone_ok = [b for b, ms in m["vllm_alone_ms_per_step"]
                if ms <= 1000 / m["realtime_tokens_per_s_per_stream"]]
    alone_interp = interp_limit(m["vllm_alone_ms_per_step"], 1000 / m["realtime_tokens_per_s_per_stream"])
    out["serving"] = {
        "per_gpu": per_gpu, "conservative_per_gpu": cons,
        "gpus": {str(c): gpus_needed(c, per_gpu) for c in SERVE_CONCURRENCY},
        "gpus_conservative": {str(c): gpus_needed(c, cons) for c in SERVE_CONCURRENCY},
        "realtime_budget_ms": 1000 / m["realtime_tokens_per_s_per_stream"],
        "vllm_alone_max_measured_batch": max(alone_ok),
        "vllm_alone_interp_limit": alone_interp,
        "gpus_if_alone_limit": {str(c): gpus_needed(c, int(alone_interp)) for c in SERVE_CONCURRENCY},
        "bandwidth_ratio": H100_MEM_BW_GBPS / L4_MEM_BW_GBPS,
    }
    return out


# ----------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------


def gb(x):
    return f"{x / 1e9:,.1f}"


def sci(x):
    return f"{x:.2e}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render(r):
    m = r["measured"]
    mod = r["models"]
    usable_gb = r["usable_bytes"] / 1e9
    S = r["strategies"]
    tok_micro = SEQ_LEN * MICRO_BATCH
    out = []
    w = out.append

    w(f"_Generated by `scripts/scaling_math.py` on {r['date']}. Do not edit between the markers; "
      "rerun the script. GB = 10^9 bytes._\n")
    w("**Reading guide.** Anything marked **[measured]** comes from this project's own results files on one "
      f"{m['gpu']}. Anything marked **[assumed]** is a general, labelled assumption (constants at the top of "
      "the script). No claims are made about any particular organisation's internal systems.\n")

    # ---- a
    w("## a. Training memory\n")
    a48, a8, a3 = mod["4.8B"], mod["800M"], mod["Orpheus 3B (reference)"]
    w(f"**Training state costs {r['bytes_per_param']} bytes per parameter, so an unsharded 4.8B model needs "
      f"{gb(a48['state']['total'])} GB before a single activation, which cannot fit on one "
      f"{GPU_MEM_GB} GB GPU; 800M ({gb(a8['one_gpu_ckpt'])} GB with checkpointing and a naive loss) can.**\n")
    w("In plain words: each parameter is stored several times. Training keeps a bf16 copy to compute with, "
      "a bf16 gradient, and, because Adam needs high precision, an fp32 master copy plus two fp32 running "
      f"averages (m and v). {BYTES_WEIGHT_BF16} + {BYTES_GRAD_BF16} + {BYTES_MASTER_FP32} + {BYTES_ADAM_M} + "
      f"{BYTES_ADAM_V} = **{r['bytes_per_param']} bytes/param** **[assumed: standard "
      "mixed-precision Adam accounting]**.\n")
    w(f"Activation assumptions **[assumed]**: sequence length {SEQ_LEN}, micro-batch {MICRO_BATCH} "
      f"({tok_micro:,} tokens per GPU per step), bf16, flash attention (no s×s score matrix).\n")
    w("```\n"
      f"state            = {r['bytes_per_param']} * N_params\n"
      f"activations      = L * {ACT_COEF} * s * b * d  +  logits            (no checkpointing)\n"
      f"activations_ckpt = L * {CKPT_SAVED_BYTES} * s * b * d  +  {ACT_COEF} * s * b * d  +  logits\n"
      f"logits (naive)   = s * b * V * {LOGITS_BYTES} bytes   (fp32 logits + fp32 gradient, all tokens at once)\n"
      f"logits (chunked) = min(s*b, {LOGIT_CHUNK_TOKENS}) * V * {LOGITS_BYTES} bytes   (one chunk alive at a time)\n"
      "```\n"
      "Without checkpointing every layer keeps all its intermediate tensors for the backward pass. With "
      "checkpointing each layer keeps only its input and recomputes the rest during backward, one layer at "
      f"a time, so memory drops a lot but you pay one extra forward pass ({CKPT_RECOMPUTE_FACTOR - 1:.0%} more compute). "
      f"The logits term is large because the vocabulary is {VOCAB:,} (audio tokens). A **naive loss** "
      "computes the scores for every token in the micro-batch at once. A **chunked / fused cross-entropy** "
      f"computes them {LOGIT_CHUNK_TOKENS} tokens at a time, turns each chunk straight into a loss and "
      "gradient, and recomputes it in backward, so the full logits tensor never exists. Both are shown "
      f"below; the chunk size ({LOGIT_CHUNK_TOKENS}) is **[assumed]**.\n")
    rows = []
    for name in ("Orpheus 3B (reference)", "800M", "4.8B"):
        x = mod[name]
        s = x["state"]
        rows.append([name, f"{x['n_params'] / 1e9:.2f}B", gb(s["weights_bf16"]), gb(s["grads_bf16"]),
                     gb(s["optimizer"]), gb(s["total"]), gb(x["logits"]), gb(x["logits_chunked"]),
                     gb(x["act_full"]), gb(x["act_ckpt"]), gb(x["act_ckpt_chunked"]),
                     gb(x["one_gpu_full"]), gb(x["one_gpu_ckpt"]), gb(x["one_gpu_ckpt_chunked"])])
    w(table(["model", "params", "weights", "grads", "optimizer", "state total", "logits (naive)",
             "logits (chunked)", "activations (no ckpt, naive loss)", "activations (ckpt, naive loss)",
             "activations (ckpt, chunked loss)", "total one GPU (no ckpt, naive)",
             "total one GPU (ckpt, naive)", "total one GPU (ckpt, chunked)"], rows))
    w("")
    oc = r["orpheus_check"]
    w("All columns in GB **[assumed]**. Model shapes used for activations. **Orpheus 3B:** 28 layers, hidden "
      f"size 3072 and vocabulary {VOCAB:,} are the real values recorded in this repo "
      "(`docs/DECISIONS.md`, `results/step1d_profile.md`). The FFN width and head counts are not recorded "
      "anywhere in the repo, so they are the Llama-3.2-3B values **[assumed]**. Cross-check **[measured]**: "
      f"that layout gives {oc['implied_params'] / 1e9:.2f}B parameters, i.e. {oc['implied_bf16_gib']:.2f} GiB in "
      f"bf16, against {oc['measured_bf16_gib']} GiB of weights that vLLM reported "
      f"(`results/step1b_precision.json`), a {oc['relative_diff']:+.1%} difference. The 800M and 4.8B "
      "layouts are project-defined Llama-style layouts sized to land near the nominal counts:\n")
    w(table(["model", "layers", "hidden", "ffn", "heads / kv heads", "params implied by shape"],
            [[n, x["shape"]["L"], x["shape"]["d"], x["shape"]["ffn"],
              f"{x['shape']['heads']} / {x['shape']['kv']}", f"{x['shape_params'] / 1e9:.2f}B"]
             for n, x in mod.items()]))
    w("")

    # ---- b
    w("## b. How it splits across GPUs\n")
    nofit48 = [s["strategy"] for s in S if s["model"] == "4.8B" and not s["fits_ckpt"]]
    nofit48_ch = [s["strategy"] for s in S if s["model"] == "4.8B" and not s["fits_ckpt_chunked"]]
    hs = next(s for s in S if s["model"] == "4.8B" and s["kind"] == "hsdp")
    f16 = next(s for s in S if s["model"] == "4.8B" and s["kind"] == "fsdp" and s["gpus"] == 16 and s["tp"] == 1)
    w(f"**For 4.8B on {GPU_MEM_GB} GB GPUs (usable {usable_gb:.0f} GB), everything fits with "
      f"activation checkpointing except {', '.join(nofit48) if nofit48 else 'nothing'} (same with a chunked "
      f"loss: {', '.join(nofit48_ch) if nofit48_ch else 'nothing'} does not fit); HSDP sends "
      f"{gb(hs['comm_inter'])} GB per GPU across nodes per step versus {gb(f16['comm_inter'])} GB for "
      f"16-way FSDP ({f16['comm_inter'] / hs['comm_inter']:.0f}× less).**\n")
    w("In plain words:\n"
      "- **DDP** copies the whole model on every GPU and only averages gradients. Simple, but memory "
      "per GPU never shrinks.\n"
      "- **FSDP** splits (shards) the 16 bytes/param across GPUs. Each layer is all-gathered just before "
      "use and freed after, and gradients are reduce-scattered. Memory per GPU = 16N / GPUs + a small "
      "gathered working set.\n"
      f"- **HSDP** shards only inside one {NODE_GPUS}-GPU node (fast NVLink) and keeps a full replica "
      "per node. Across nodes only the small gradient shard is averaged.\n"
      "- **TP** (tensor parallel) splits each matrix multiply across 2 or 4 GPUs in a node; it cuts "
      "weights and activations per GPU but needs an all-reduce inside every layer, so it stays within a "
      "node.\n")
    w("Per-GPU communication volume, per micro-batch step (accumulation "
      f"{GRAD_ACCUM}), ring collectives, N = parameters, D = group size:\n")
    w("```\n"
      "DDP   : 2 * (D-1)/D * 2N                       (all-reduce of bf16 grads)\n"
      "FSDP  : 3 * (D-1)/D * 2N / tp                  (all-gather fwd, all-gather bwd, reduce-scatter)\n"
      f"HSDP  : 3 * ({NODE_GPUS - 1}/{NODE_GPUS}) * 2N  [in node]  +  2 * (R-1)/R * 2N/{NODE_GPUS}  "
      "[across nodes, R = node replicas]\n"
      "TP    : L * 4 * 2 * (tp-1)/tp * (s*b*d*2)      (in node; checkpointing adds another forward's worth, not modelled)\n"
      "```\n"
      "Why HSDP cuts cross-node traffic: with FSDP over 16 GPUs the three full-model collectives run in "
      "one ring that crosses the node boundary, so all of that volume rides the slow inter-node link. With "
      "HSDP the full-model collectives stay inside the node on NVLink, and the only cross-node message "
      f"is one all-reduce of the gradient *shard* (1/{NODE_GPUS} of the gradients). A flat group bigger than a node "
      "is treated as inter-node bound **[assumed simplification: real NCCL uses hierarchical algorithms "
      "and does better]**.\n")
    w(f"Bandwidths **[assumed]**: intra-node {NVLINK_GBPS:.0f} GB/s per GPU, inter-node "
      f"{INTERNODE_GBPS:.0f} GB/s per GPU. Compute time = 6·N·tokens / (peak {PEAK_BF16_TFLOPS:.0f} TFLOPs × "
      f"MFU {MFU}). Comm time = intra/NVLink + inter/network, summed (an upper bound: real runs overlap "
      "comm with compute).\n")
    for name in ("4.8B", "800M", "Orpheus 3B (reference)"):
        w(f"**{name}**\n")
        rows = []
        for s in [x for x in S if x["model"] == name]:
            def cell(total, ok):
                return f"{gb(total)} " + ("fits" if ok else "**NO**")
            rows.append([s["strategy"], gb(s["state"]), gb(s["gathered"]),
                         cell(s["total_ckpt"], s["fits_ckpt"]),
                         cell(s["total_ckpt_chunked"], s["fits_ckpt_chunked"]),
                         cell(s["total_full"], s["fits_full"]),
                         gb(s["comm_intra"]), gb(s["comm_inter"]),
                         f"{s['comm_s'] * 1e3:,.0f}", f"{s['compute_s'] * 1e3:,.0f}"])
        w(table(["strategy", "state GB", "gathered GB", "total GB (ckpt, naive loss)",
                 "total GB (ckpt, chunked loss)", "total GB (no ckpt, naive loss)",
                 "in-node GB sent", "cross-node GB sent", "comm ms", "compute ms"], rows))
        w("")
    w(f"Global batch implied: data-parallel size × {tok_micro:,} tokens, e.g. "
      f"{S[0]['global_batch_tokens']:,} tokens for 16-way data parallel **[assumed setup]**. A comm time "
      "above the compute time means the step would be network-bound unless overlap hides it or gradient "
      "accumulation amortises it.\n")

    # ---- c
    d = r["data"]
    w("## c. Data and training compute\n")
    c48, c8 = d["compute"]["4.8B"], d["compute"]["800M"]
    w(f"**{d['hours']:,} hours of audio is {d['tokens']:.3g} SNAC tokens; one epoch costs about "
      f"{c8['gpu_hours']:,.0f} GPU-hours for 800M and {c48['gpu_hours']:,.0f} GPU-hours for 4.8B at an assumed "
      f"{MFU:.0%} MFU.**\n")
    w(f"Tokens = hours × tokens per audio hour = {d['hours']:,} × {m['tokens_per_audio_hour']:,} "
      f"**[measured: results/pipeline.json, {m['pipeline_clips']:,} clips; the pure codec rate would be "
      f"{SNAC_TOKENS_PER_FRAME} × {SNAC_SAMPLE_RATE}/{SNAC_SAMPLES_PER_FRAME} × 3600 = "
      f"{SNAC_TOKENS_PER_FRAME * SNAC_SAMPLE_RATE / SNAC_SAMPLES_PER_FRAME * 3600:,.0f}; the measured rate adds "
      "the rounding-up of each clip's last frame]**. It counts audio tokens only, not "
      "the text prompt tokens.\n")
    w("```\n"
      "training FLOPs = 6 * N * D          (N = parameters, D = tokens)\n"
      "GPU-hours      = FLOPs / (peak FLOPs/s * MFU) / 3600\n"
      "```\n"
      "In plain words: every token passes through every parameter once forward (2 FLOPs per parameter: one "
      "multiply, one add) and the backward pass costs twice that, hence 6. MFU says what fraction of the "
      f"GPU's peak actually achieved: {PEAK_BF16_TFLOPS:.0f} TFLOPs × {MFU} = "
      f"{d['effective_flops_per_gpu'] / 1e12:.0f} TFLOPs effective **[assumed]**.\n")
    rows = []
    for name in ("800M", "4.8B", "Orpheus 3B (reference)"):
        c = d["compute"][name]
        rows.append([name, sci(c["flops"]), f"{c['gpu_hours']:,.0f}", f"{c['gpu_hours_with_ckpt']:,.0f}"]
                    + [f"{c['days_on'][str(k)]:.1f}" for k in CLUSTER_SIZES])
    w(table(["model", "6·N·D FLOPs", "GPU-hours", "GPU-hours with checkpointing"]
            + [f"days on {k} GPUs" for k in CLUSTER_SIZES], rows))
    w("\nDays assume perfect linear scaling and one epoch, so they are lower bounds. 6ND ignores attention "
      "FLOPs, which are small at this sequence length.\n")

    # ---- d
    p = r["pipeline"]
    w("## d. Data pipeline at that scale\n")
    w(f"**Tokenizing {DATASET_HOURS:,} hours takes about {p['l4_hours']:,.0f} L4-hours at the measured "
      f"{m['pipeline_hours_per_min']} audio hours per minute, and it is bound by the GPU (SNAC encode), "
      f"not the CPU, which has about {m['pipeline_cpu_headroom_x']:.0f}× headroom.**\n")
    w("```\n"
      "L4-hours = hours of audio / (audio-hours per minute * 60)\n"
      "```\n"
      f"= {DATASET_HOURS:,} / ({m['pipeline_hours_per_min']} × 60) **[measured rate: results/pipeline.json, "
      f"full run of {m['pipeline_audio_hours']:.1f} h]**.\n")
    caps = m["pipeline_cpu_stage_caps"]
    w(table(["stage (full run)", "capacity, audio-hours / min", "vs end-to-end"],
            [["SNAC encode on GPU", m["pipeline_gpu_stage_hours_per_min"],
              f"GPU {m['pipeline_gpu_util_pct']:.0f}% busy"]]
            + [[k, v, f"{v / m['pipeline_hours_per_min']:.1f}×"] for k, v in caps.items()]
            + [["**end-to-end**", m["pipeline_hours_per_min"], "—"]]))
    w(f"\nThe slowest CPU stage (VAD, {m['pipeline_workers']} workers) still has "
      f"{m['pipeline_cpu_headroom_x']:.1f}× the GPU's rate, i.e. one L4 needs only about "
      f"{p['vad_workers_per_gpu']:.2f} of a worker's time for VAD **[derived from measured]**. "
      "Adding CPUs would not speed it up; adding GPUs would.\n")
    w("**How it parallelises.** Each clip is encoded independently, so shard the parquet files across "
      "machines: one GPU (plus a few CPU cores) per shard group, each writing its own output shards. No "
      "GPU ever talks to another, so throughput scales linearly with GPU count (until storage or network "
      "reads become the limit, which was not measured).\n")
    w(table(["L4 GPUs", "wall-clock days (ideal linear scaling)"],
            [[k, f"{v:.1f}"] for k, v in p["wall_days"].items()]
            + [[f"{p['l4_gpus_for_deadline']} (to finish in {TARGET_TOKENIZE_DAYS} days **[assumed target]**)",
                f"{TARGET_TOKENIZE_DAYS}.0"]]))
    w(f"\nOutput size at the measured zstd-parquet density: {p['shard_bytes_per_audio_hour'] / 1e3:,.0f} kB "
      f"per audio hour, so about {p['shard_bytes_total'] / 1e12:.2f} TB for {DATASET_HOURS:,} hours "
      "**[extrapolated from measured]**. Only LibriTTS-R on one machine was run; not measured: reading "
      "from remote storage and other GPU types.\n")

    # ---- e
    s = r["serving"]
    w("## e. Serving\n")
    g = s["gpus"]
    w(f"**At the measured {s['per_gpu']} realtime streams per L4 you need {g['100']} GPUs for 100 "
      f"concurrent streams and {g['1000']} for 1,000; planning with a safety margin "
      f"({s['conservative_per_gpu']} per GPU, the next lower level that passed) gives "
      f"{s['gpus_conservative']['100']} and {s['gpus_conservative']['1000']}.**\n")
    w("```\n"
      "GPUs = ceil(concurrent streams / streams per GPU)\n"
      "```\n"
      f"Measured **[results/step6c_final_sweep.json]**: max {s['per_gpu']} streams where p95 RTF < 1.0 with no "
      f"gaps (RTF p95 {m['rtf_p95_nmax']}, TTFA p50 {m['ttfa_ms_p50_nmax']:.0f} ms / p95 "
      f"{m['ttfa_ms_p95_nmax']:.0f} ms at {s['per_gpu']} streams; TTFA p50 {m['ttfa_ms_p50_n1']:.0f} ms at "
      f"1 stream). Level {m['first_failing_level']} already fails. At {s['per_gpu']} streams the margin is "
      "thin (RTF p95 so close to 1.0), hence the conservative figures. **[assumed]** for the GPU counts: "
      "every stream is continuously generating, load is spread perfectly across GPUs, and one websocket "
      "stays on one GPU (no cross-GPU communication). Real traffic has idle time and bursts; this is "
      "peak-concurrency sizing, not a traffic forecast.\n")
    w(f"Where the limit comes from **[measured, DECISIONS step 6b]**: a realtime stream needs "
      f"{m['realtime_tokens_per_s_per_stream']:.2f} tokens/s, a budget of {s['realtime_budget_ms']:.1f} "
      f"ms/token. Alone, single-stream decode takes {m['decode_ms_per_token_b1']} ms/token "
      f"({m['decode_tokens_per_s_b1']} tok/s, step 1d). In the server at {s['per_gpu']} streams vLLM takes "
      f"{m['vllm_ms_per_token_nmax']} ms/token, because vLLM and SNAC run in two processes and the GPU "
      "time-slices between them.\n")
    w("What would raise per-GPU capacity:\n"
      f"- **One-process serving or MPS** (stop the time-slicing). In the step 6a benchmark vLLM *alone* "
      f"stayed within the {s['realtime_budget_ms']:.1f} ms budget up to batch {s['vllm_alone_max_measured_batch']} "
      f"measured, and about {s['vllm_alone_interp_limit']:.0f} by linear interpolation. That would be "
      f"{s['gpus_if_alone_limit']['100']} / {s['gpus_if_alone_limit']['1000']} GPUs for 100 / 1,000 "
      "streams. **This is an upper bound, not a result:** SNAC also needs GPU time and neither fix could be "
      "run here (vLLM 0.32's AsyncLLM cannot run in-process; MPS failed in the test container).\n"
      f"- **Bigger GPUs.** Decode is mostly memory-bandwidth bound. An H100's bandwidth "
      f"({H100_MEM_BW_GBPS:.0f} GB/s **[assumed spec]**) is {s['bandwidth_ratio']:.0f}× an L4's "
      f"({L4_MEM_BW_GBPS:.0f} GB/s **[assumed spec]**), so per-GPU capacity would rise, but it has not been "
      "measured, and cost per stream is a separate question.\n"
      "- Reducing SNAC's share of GPU time did not help in the step 6b tests, so more SNAC "
      "micro-optimisation is not the lever.\n")

    # ---- f
    w("## f. What was measured vs what was assumed\n")
    rows = [
        ["Tokens per audio hour", f"{m['tokens_per_audio_hour']:,}", "measured", "results/pipeline.json (full run)"],
        ["Pipeline throughput, end-to-end", f"{m['pipeline_hours_per_min']} audio h/min on 1 L4", "measured",
         "results/pipeline.json (full run)"],
        ["Pipeline CPU headroom", f"{m['pipeline_cpu_headroom_x']:.1f}×", "measured (derived)", "results/pipeline.json"],
        ["Max realtime streams per L4", m["max_streams"], "measured", "results/step6c_final_sweep.json"],
        ["TTFA p50 at max streams", f"{m['ttfa_ms_p50_nmax']:.0f} ms", "measured", "results/step6c_final_sweep.json"],
        ["Decode time per token, 1 stream", f"{m['decode_ms_per_token_b1']} ms", "measured",
         "results/step1d_overhead.json (lmhead_w4a16)"],
        ["vLLM alone, ms/step vs batch", "table in step 6a", "measured", "results/step6a_vllm_batch_bench.json"],
        ["GPU memory / bf16 peak", f"{GPU_MEM_GB} GB / {PEAK_BF16_TFLOPS:.0f} TFLOPs", "ASSUMED",
         "H100 SXM spec sheet"],
        ["Usable memory fraction", USABLE_MEM_FRACTION, "ASSUMED", "CUDA context, NCCL, fragmentation"],
        ["MFU", MFU, "ASSUMED", "typical for tuned dense LLM training; the biggest lever on GPU-hours"],
        ["Bytes per parameter (training)", 16, "ASSUMED", "mixed-precision Adam (ZeRO accounting)"],
        ["Sequence length / micro-batch", f"{SEQ_LEN} / {MICRO_BATCH}", "ASSUMED", "the serving max_model_len; small micro-batch"],
        ["Activation coefficient", ACT_COEF, "ASSUMED", "Korthikanti et al. 2022, flash attention"],
        ["Orpheus layers / hidden / vocab", f"{MODEL_SHAPES['Orpheus 3B (reference)']['L']} / {MODEL_SHAPES['Orpheus 3B (reference)']['d']} / {VOCAB:,}", "recorded in repo", "docs/DECISIONS.md, results/step1d_profile.md"],
        ["Orpheus FFN width and head counts", "see section a", "ASSUMED", "Llama-3.2-3B values; not recorded in repo; weight-size cross-check in section a"],
        ["800M / 4.8B layouts", "see section a", "ASSUMED", "project-defined Llama-style layouts"],
        ["Chunked-loss chunk size", f"{LOGIT_CHUNK_TOKENS} tokens", "ASSUMED", "typical fused cross-entropy chunk"],
        ["NVLink / inter-node bandwidth", f"{NVLINK_GBPS:.0f} / {INTERNODE_GBPS:.0f} GB/s per GPU", "ASSUMED",
         "H100 / 400 Gb NIC per GPU"],
        ["Dataset size", f"{DATASET_HOURS:,} hours", "ASSUMED", "the scale named in the task"],
        ["Linear scaling of tokenization and serving", "—", "ASSUMED", "no cross-GPU communication in either"],
    ]
    w(table(["quantity", "value", "status", "source / reason"], rows))
    w("\n**What has NOT been run.** No multi-GPU training of any kind: no DDP, FSDP, HSDP or tensor "
      "parallelism, and no multi-node job. No model has been trained in this project; sections "
      "a–c are arithmetic, not measurements. No more than one 53-hour dataset has been tokenized on one "
      "L4, nothing has been served on more than one GPU, and never measured an H100/A100. All 80 GB numbers, "
      "bandwidths and the MFU are assumptions.\n")
    return "\n".join(out)


def update_doc(text, doc_path):
    block = f"{DOC_BEGIN}\n\n{text}\n{DOC_END}"
    if doc_path.exists() and DOC_BEGIN in doc_path.read_text() and DOC_END in doc_path.read_text():
        old = doc_path.read_text()
        head = old.split(DOC_BEGIN)[0]
        tail = old.split(DOC_END, 1)[1]
        doc_path.write_text(head + block + tail)
    else:
        doc_path.write_text("# Scaling notes\n\nHow the measured single-L4 numbers translate to a much larger "
                            "scale. Sections a-f below.\n\n" + block + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-write", action="store_true", help="print the doc, write nothing")
    args = ap.parse_args()
    m = load_measured()
    r = compute(m)
    r["bytes_per_param"] = train_state_bytes(1)["total"]
    r["date"] = datetime.date.today().isoformat()
    text = render(r)
    if args.no_write:
        print(text)
        return
    meta = {
        "date": r["date"],
        "gpu": f"none used; measured inputs are from {m['gpu']}",
        "model": f"Orpheus 3B ({m['serve_model']}) + SNAC 24 kHz; training models are hypothetical 800M / 4.8B",
        "vllm_version": m["vllm_version"],
        "torch_version": m["torch_version"],
    }
    out = {**meta, "assumed_constants": {k: v for k, v in globals().items()
                                         if k.isupper() and isinstance(v, (int, float, tuple, dict))
                                         and k not in ("MODEL_SHAPES", "STRATEGIES")},
           "model_shapes": MODEL_SHAPES, "results": r}
    (ROOT / "results" / "scaling.json").write_text(json.dumps(out, indent=2, default=str) + "\n")
    update_doc(text, ROOT / "docs" / "SCALING.md")
    print("wrote results/scaling.json and docs/SCALING.md")


if __name__ == "__main__":
    main()
