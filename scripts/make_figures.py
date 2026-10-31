"""Draw the README charts from the files in results/ (nothing is typed in by hand).

  python scripts/make_figures.py          # needs matplotlib; writes results/figures/*.png

Figures:
  ttfa_vs_streams.png      time to first audio (p50) vs concurrent streams: step 4 baseline, 6a batched, 6c final
  rtf_vs_streams.png       real-time factor (p95) vs streams for the same three, realtime line at 1.0
  precision_ladder.png     single-stream decode tok/s per precision, with the realtime line
  pipeline_stages.png      capacity of each tokenization pipeline stage (hours of audio per minute)
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = RES / "figures"

# Okabe-Ito colour-blind-safe palette.
GREY, BLUE, ORANGE, GREEN, RED = "#8c8c8c", "#0072B2", "#D55E00", "#009E73", "#B2182B"
INK, MUTED = "#222222", "#666666"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 150,
    "legend.frameon": False,
})


def read_levels(name):
    with open(RES / name) as f:
        return [{k: float(v) if v not in ("", None) else None for k, v in row.items()} for row in csv.DictReader(f)]


def load(name):
    with open(RES / name) as f:
        return json.load(f)


def footer(fig, meta, sources):
    fig.text(0.01, 0.005, f"{meta['gpu']} · vLLM {meta['vllm_version']} · {meta['date'][:10]}\nsource: {sources}",
             fontsize=7.5, color=MUTED, ha="left", va="bottom", linespacing=1.4)


SWEEPS = [
    ("Step 4 baseline (per-stream SNAC)", "baseline_sweep", GREY),
    ("Step 6a batched SNAC", "batched_sweep", BLUE),
    ("Step 6c final (batched + early first chunk)", "step6c_final_sweep", ORANGE),
]


def sweep_figure(metric, ylabel, title, fname, hline=None, logy=False):
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    meta = load("step6c_final_sweep.json")
    for label, stem, color in SWEEPS:
        rows = read_levels(f"{stem}_levels.csv")
        xs = [r["streams"] for r in rows]
        ys = [r[metric] for r in rows]
        ax.plot(xs, ys, marker="o", ms=4.5, lw=2, color=color, label=label)
    ax.set_xscale("log", base=2)
    ticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    if logy:
        ax.set_yscale("log")
        ax.set_yticks([0.5, 1, 2, 4, 8, 12])
        ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    if hline is not None:
        ax.axhline(hline, color=RED, lw=1.5, ls="--")
        ax.annotate("realtime limit (1.0)", xy=(1.0, hline), xytext=(1.05, hline * 1.07), color=RED, fontsize=10)
    ax.set_xlabel("concurrent streams on one GPU")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.legend(loc="upper left")
    ax.margins(x=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    footer(fig, meta, "results/{baseline,batched,step6c_final}_sweep_levels.csv")
    fig.savefig(OUT / fname)
    plt.close(fig)


def precision_figure():
    p = load("step1b_precision.json")
    d = load("step1d_overhead.json")
    best = next(r for r in d if r["variant"] == "lmhead_w4a16")
    bench = load("step6a_vllm_batch_bench.json")
    realtime = bench["config"]["realtime_tokens_per_s_per_stream"]
    bars = [
        ("bf16", p["bf16"]["tokens_per_s"], GREY),
        ("fp8\n(W8A8)", p["fp8"]["tokens_per_s"], GREY),
        ("W4A16", p["w4a16"]["tokens_per_s"], BLUE),
        ("W4A16 +\n4-bit lm_head", best["tokens_per_s"], ORANGE),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    xs = range(len(bars))
    ax.bar(xs, [b[1] for b in bars], color=[b[2] for b in bars], width=0.62)
    for x, b in zip(xs, bars):
        ax.text(x, b[1] + 2, f"{b[1]:.1f}", ha="center", va="bottom", fontsize=11, color=INK)
    ax.axhline(realtime, color=RED, lw=1.5, ls="--")
    ax.annotate(f"realtime need: {realtime:.1f} tok/s per stream", xy=(-0.4, realtime),
                xytext=(-0.4, realtime + 3), color=RED, fontsize=10)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("single-stream decode speed (tokens/s)")
    ax.set_title("Precision ladder: tokens per second at batch size 1", loc="left", fontsize=12, color=INK)
    ax.set_ylim(0, max(b[1] for b in bars) * 1.15)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    footer(fig, {"gpu": p["bf16"]["gpu"], "vllm_version": p["bf16"]["vllm_version"], "date": best["date"]},
           "results/step1b_precision.json (bf16, fp8, W4A16), step1d_overhead.json (+ lm_head),\n"
           "step6a_vllm_batch_bench.json (realtime need)")
    fig.savefig(OUT / "precision_ladder.png")
    plt.close(fig)


def pipeline_figure():
    run = load("pipeline.json")["runs"]["full"]
    st = run["stages"]
    rows = [
        ("read parquet (CPU)", st["read_parquet"]["hours_per_min_capacity"], GREY),
        ("decode WAV (CPU)", st["decode"]["hours_per_min_capacity"], GREY),
        ("VAD trim (CPU)", st["vad"]["hours_per_min_capacity"], GREY),
        ("SNAC encode (GPU)", st["tokenize_gpu"]["hours_per_min_capacity"], ORANGE),
        ("write shards (CPU)", st["shard"]["hours_per_min_capacity"], GREY),
    ]
    e2e = run["end_to_end_audio_hours_per_min"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ys = range(len(rows))
    ax.barh(list(ys), [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for y, r in zip(ys, rows):
        ax.text(r[1] * 1.08, y, f"{r[1]:,.1f}", va="center", fontsize=10, color=INK)
    ax.axvline(e2e, color=RED, lw=1.5, ls="--")
    ax.text(e2e * 5, 3.0, f"dashed line: end-to-end {e2e} h/min", color=RED, fontsize=10, va="center")
    ax.set_yticks(list(ys))
    ax.set_yticklabels([r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlim(1, max(r[1] for r in rows) * 4)
    ax.set_xlabel("stage capacity (hours of audio per minute, log scale)")
    ax.set_title("Tokenization pipeline: the GPU encoder is the slowest stage", loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    footer(fig, run, f"results/pipeline.json (run '{run['label']}', {run['data']['trimmed_audio_hours']:.1f} h of audio)")
    fig.savefig(OUT / "pipeline_stages.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sweep_figure("ttfa_ms_p50", "time to first audio, p50 (ms)", "Time to first audio vs concurrent streams",
                 "ttfa_vs_streams.png")
    sweep_figure("rtf_p95", "real-time factor, p95 (lower is better; log scale)",
                 "Real-time factor vs concurrent streams", "rtf_vs_streams.png", hline=1.0, logy=True)
    precision_figure()
    pipeline_figure()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))


if __name__ == "__main__":
    main()
