"""Render README.md from docs/README.template.md and the files in results/.

  python scripts/make_readme.py            # rewrite README.md
  python scripts/make_readme.py --check    # exit 1 if README.md is out of date (also run by the tests)

The template holds the prose. Every number, table and figure caption value is a {{placeholder}} filled here
from results/*.json, results/*.csv and results/step1d_profile.md, so the README cannot drift from the
measurements. Standard library only.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
TEMPLATE = ROOT / "docs" / "README.template.md"
README = ROOT / "README.md"


def jload(name):
    with open(RES / name) as f:
        return json.load(f)


def cload(name):
    with open(RES / name) as f:
        return list(csv.DictReader(f))


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def pct(x, nd=1):
    return f"{x * 100:.{nd}f}%"


def pts(x):
    return f"{x * 100:+.2f}"


def profile_value(label):
    """Read one 'ms/step' value from the table in results/step1d_profile.md."""
    for line in (RES / "step1d_profile.md").read_text().splitlines():
        if label in line:
            cells = [c.strip().strip("*") for c in line.strip().strip("|").split("|")]
            return float(cells[1]), cells[2]
    raise KeyError(label)


# ----------------------------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------------------------


def build(res=None):
    prec = jload("step1b_precision.json")
    d1d = jload("step1d_overhead.json")
    lmhead = next(r for r in d1d if r["variant"] == "lmhead_w4a16")
    bench = jload("step6a_vllm_batch_bench.json")
    sweep = jload("step6c_final_sweep.json")
    lv6c = {int(r["streams"]): r for r in sweep["levels"]}
    base = {int(r["streams"]): r for r in map(lambda x: {k: float(v) if v else None for k, v in x.items()},
                                              cload("baseline_sweep_levels.csv"))}
    bat = {int(r["streams"]): r for r in map(lambda x: {k: float(v) if v else None for k, v in x.items()},
                                             cload("batched_sweep_levels.csv"))}
    qb = jload("quality_baseline.json")
    q6 = jload("step6c_quality.json")
    pipe = jload("pipeline.json")
    full = pipe["runs"]["full"]
    scal = jload("scaling.json")
    opt = {r["id"]: r for r in cload("optimizations.csv")}

    rt_tok = bench["config"]["realtime_tokens_per_s_per_stream"]
    rt_ms = 1000 / rt_tok
    max_n = sweep["max_streams"]
    passing = sorted(n for n, r in lv6c.items()
                     if r["rtf_p95"] < 1.0 and r["pct_requests_with_gaps"] == 0 and r["errors"] == 0)
    margin_n = max(n for n in passing if n < max_n)
    top = lv6c[max_n]
    first_fail = min(n for n in lv6c if n > max_n)

    F = {}

    # ---- precision table
    def prow(label, tps, rtf, note):
        return [label, f"{1000 / tps:.1f}", f"{tps:.1f}", f"{rtf:.2f}", "yes" if tps >= rt_tok else "no", note]

    p = prec
    rows = [
        prow("bf16", p["bf16"]["tokens_per_s"], p["bf16"]["rtf_generation"], "baseline"),
        prow("fp8, W8A8 (CUTLASS)", p["fp8"]["tokens_per_s"], p["fp8"]["rtf_generation"], "rejected: not realtime"),
        prow("fp8, weight-only (Marlin)", p["fp8_marlin"]["tokens_per_s"], p["fp8_marlin"]["rtf_generation"],
             "rejected: not realtime"),
        prow("W4A16 (GPTQ, group 128)", p["w4a16"]["tokens_per_s"], p["w4a16"]["rtf_generation"],
             "first realtime config"),
        prow("**W4A16 + 4-bit lm_head**", lmhead["tokens_per_s"], lmhead["rtf_generation"], "**served default**"),
    ]
    F["t_precision"] = md_table(["precision", "ms / token", "tokens / s", "RTF (generation only)",
                                 f"≥ {rt_tok:.1f} tok/s needed", "status"], rows)

    # ---- quality
    qpc = qb["per_config"]
    cmp_ = qb["comparisons"]
    cmp6 = q6["comparisons"]["w4a16_lmhead_vs_baseline_csv"]
    s6 = q6["per_config"]["w4a16_lmhead"]
    n_sent = qb["config"]["n_sentences"]
    seeds = len(qb["config"]["seeds"])

    def qrow(name, key, diff_key):
        s, sf = qpc[key]["summary"], qpc[key]["summary_format_insensitive"]
        if diff_key is None:
            diff, diff_f = "reference", "reference"
        else:
            c, cf = cmp_[diff_key], cmp_[diff_key + "_format_insensitive"]
            diff = f"{pts(c['mean_diff'])} pts [{pts(c['ci95'][0])}, {pts(c['ci95'][1])}]"
            diff_f = f"{pts(cf['mean_diff'])} pts [{pts(cf['ci95'][0])}, {pts(cf['ci95'][1])}]"
        return [name, s["clips"], pct(s["wer_mean"]), pct(sf["wer_mean"]), pct(s["cer_mean"]), diff, diff_f]

    qrows = [qrow("bf16", "bf16", None), qrow("W4A16", "w4a16", "w4a16_minus_bf16"),
             qrow("W4A16 + 4-bit lm_head (step 5)", "w4a16_lmhead", "w4a16_lmhead_minus_bf16")]
    sfin, sfin_f = s6["summary"], s6["summary_format_insensitive"]
    qrows.append(["W4A16 + 4-bit lm_head, final server config (step 6c)", sfin["clips"], pct(sfin["wer_mean"]),
                  pct(sfin_f["wer_mean"]), pct(sfin["cer_mean"]), "not run against bf16 (see below)", "—"])
    F["t_quality"] = md_table(["config", "clips", "WER mean", "WER mean, format-insensitive", "CER mean",
                               "Δ WER vs bf16 [95% CI]", "Δ WER vs bf16, format-insensitive [95% CI]"], qrows)
    d_off = cmp_["w4a16_lmhead_minus_bf16"]
    F["lm_vs_bf16"] = f"{pts(d_off['mean_diff'])} pts [{pts(d_off['ci95'][0])}, {pts(d_off['ci95'][1])}]"
    F["final_vs_step5"] = (f"{pts(cmp6['diff_official']['mean_diff'])} pts "
                           f"[{pts(cmp6['diff_official']['ci95'][0])}, {pts(cmp6['diff_official']['ci95'][1])}]")
    F["n_sent"], F["n_seeds"] = n_sent, seeds
    F["n_clips"] = qpc["bf16"]["summary"]["clips"]
    F["voice"] = qb["config"]["voice"]
    F["whisper"] = qb["config"]["generation"]["whisper"]["whisper_model"]
    F["boot_resamples"] = re.search(r"(\d[\d,]*) resamples", qb["config"]["bootstrap"]).group(1)
    F["seed_noise_lo"] = f'{min(qpc[k]["seed_noise"]["mean_abs_seed_diff_per_sentence"] for k in qpc) * 100:.1f}'
    F["seed_noise_hi"] = f'{max(qpc[k]["seed_noise"]["mean_abs_seed_diff_per_sentence"] for k in qpc) * 100:.1f}'
    F["bf16_wer"] = pct(qpc["bf16"]["summary"]["wer_mean"])
    F["lm_wer"] = pct(qpc["w4a16_lmhead"]["summary"]["wer_mean"])
    F["w4_wer"] = pct(qpc["w4a16"]["summary"]["wer_mean"])
    F["fin_wer"] = pct(sfin["wer_mean"])
    F["lm_wer_fmt"] = pct(qpc["w4a16_lmhead"]["summary_format_insensitive"]["wer_mean"])
    F["bf16_wer_fmt"] = pct(qpc["bf16"]["summary_format_insensitive"]["wer_mean"])

    # ---- headline table
    pb_h = full["data"]["tokens_per_audio_hour"]
    rows = [
        ["Single-stream decode speed", f"{lmhead['tokens_per_s']:.1f} tok/s ({lmhead['ms_per_token']} ms/token, "
         f"RTF {lmhead['rtf_generation']:.2f}) vs {prec['bf16']['tokens_per_s']:.1f} tok/s for bf16",
         "step1d_overhead.json, step1b_precision.json"],
        ["Time to first audio, 1 stream", f"{lv6c[1]['ttfa_ms_p50']:.0f} ms p50 ({lv6c[1]['ttfa_ms_p95']:.0f} ms p95)",
         "step6c_final_sweep.json"],
        [f"Time to first audio, {max_n} streams",
         f"{top['ttfa_ms_p50']:.0f} ms p50 ({top['ttfa_ms_p95']:.0f} ms p95)", "step6c_final_sweep.json"],
        ["Max realtime streams per L4",
         f"**{max_n}** (p95 RTF {top['rtf_p95']}, no gaps); **{margin_n}** with one level of margin "
         f"(p95 RTF {lv6c[margin_n]['rtf_p95']}); {first_fail} fails (p95 RTF {lv6c[first_fail]['rtf_p95']})",
         "step6c_final_sweep.json"],
        ["WER vs bf16 (4-bit lm_head, step 5)",
         f"{F['lm_wer']} vs {F['bf16_wer']}: {F['lm_vs_bf16']}, within noise", "quality_baseline.json"],
        ["Tokenization pipeline throughput", f"{full['end_to_end_audio_hours_per_min']} hours of audio / min on one L4 "
         f"(GPU stage alone: {full['stages']['tokenize_gpu']['hours_per_min_capacity']})", "pipeline.json"],
        ["Tokens per audio hour (SNAC 24 kHz, 7 tokens per frame)", f"{pb_h:,}", "pipeline.json"],
    ]
    F["t_headline"] = md_table(["metric", "result", "source (results/)"], rows)

    # ---- optimisation ladder
    reasons = {
        "baseline_per_stream_snac": "One SNAC call per window on one worker thread; the worker saturates and throughput flattens.",
        "batched_snac": "One SNAC call for every stream's pending window. SNAC no longer saturates; vLLM and SNAC now contend for the GPU.",
        "fp8_weights": "Not realtime even for one stream; W4A16 is faster with no measurable WER loss.",
        "6b_chunk4": "At the limit there is no playback-buffer slack, so chunks stay small; mixed window lengths split SNAC batches; small gaps at 6 streams.",
        "6b_chunk8": "No capacity gain; savings only appear at loads that already pass.",
        "6b_fp16": "Each new (batch, window length) shape costs seconds of cuDNN setup; unwarmed shapes hit it live and TTFA p95 spikes. Audio was fine.",
        "6b_fp32_graph": "2-4× faster alone, but in the server SNAC time is mostly waiting for a GPU time slice shared with vLLM's process.",
        "6b_fp16_graph_compile": "Not screened: it shortens SNAC kernel time, which is not what limits the server.",
        "6b_same_process": "vLLM's async engine always runs its core in a child process; MPS did not start in the test container.",
        "6b_wait1": "Batches did not grow (windows already arrive together); the wait only adds latency and gaps.",
        "6b_wait2": "Same as 1 ms: no larger batches, more gaps.",
        "6b_final": "Fine sweep of the 6a configuration: 6a's coarse sweep had skipped levels 5-7.",
        "6c_first_wait3": "Works, but superseded by the 2-frame first chunk.",
        "6c_first_wait2": "Smallest first chunk with clean audio; time to first audio drops at every load level.",
        "6c_first_wait1": "Audio not clean: the first frames have no right context.",
        "6c_snac_cpu": "The CPU cannot keep up with the windows per second that 8 streams need.",
        "6c_final": "Server default. Lower time to first audio at every level; capacity unchanged, margin at 6 streams thinner.",
    }
    order = ["baseline_per_stream_snac", "batched_snac", "fp8_weights", "6b_chunk4", "6b_chunk8", "6b_fp16",
             "6b_fp32_graph", "6b_fp16_graph_compile", "6b_same_process", "6b_wait1", "6b_wait2", "6b_final",
             "6c_first_wait3", "6c_first_wait2", "6c_first_wait1", "6c_snac_cpu", "6c_final"]
    rows = []
    for i in order:
        r = opt[i]
        ms = r["max_streams"] if r["levels"] else "—"
        t1 = f"{float(r['ttfa_p50_ms_n1']):.0f}" if r["ttfa_p50_ms_n1"] else "—"
        status = r["status"].replace("candidate (superseded by first_wait2)", "superseded")
        rows.append([r["step"], r["change"].split(" (")[0] if len(r["change"]) > 70 else r["change"],
                     f"**{status}**" if status == "adopted" else status, ms, t1, reasons[i]])
    F["t_ladder"] = md_table(["step", "change", "outcome", "max realtime streams", "TTFA p50 at 1 stream (ms)", "why"],
                             rows)
    F["ladder_note_n"] = len(opt)

    # ---- pipeline runs
    runs = pipe["runs"]
    names = {"baseline": "baseline (16 clips / batch)", "tuned": "tuned: 4M-sample batches, pinned, prefetch, compile",
             "tuned_noprefetch": "tuned without prefetch", "tuned_8m": "tuned, 8M-sample batches",
             "full": "**full split, tuned without prefetch**"}
    rows = []
    for k, r in runs.items():
        hrs = r["data"]["trimmed_audio_hours"]
        rows.append([names[k], f"{hrs:.1f}", r["end_to_end_audio_hours_per_min"],
                     r["stages"]["tokenize_gpu"]["hours_per_min_capacity"], f"{r['utilisation']['gpu_pct_mean']:.0f}%",
                     f"{r['utilisation']['gpu_mem_gib_peak']:.1f}"])
    F["t_pipeline_runs"] = md_table(["run", "audio (h)", "end-to-end (h / min)", "GPU stage capacity (h / min)",
                                     "GPU busy", "peak GPU memory (GiB)"], rows)

    # ---- facts for prose
    F["rt_tok"], F["rt_ms"] = f"{rt_tok:.1f}", f"{rt_ms:.1f}"
    F["bf16_tps"], F["bf16_ms"] = f"{prec['bf16']['tokens_per_s']:.1f}", f"{1000 / prec['bf16']['tokens_per_s']:.1f}"
    F["fp8_tps"] = f"{prec['fp8']['tokens_per_s']:.1f}"
    F["w4_tps"] = f"{prec['w4a16']['tokens_per_s']:.1f}"
    F["lm_tps"], F["lm_ms"] = f"{lmhead['tokens_per_s']:.1f}", f"{lmhead['ms_per_token']}"
    F["w_bf16"], F["w_fp8"], F["w_w4"] = (prec["bf16"]["weights_gib"], prec["fp8"]["weights_gib"],
                                          prec["w4a16"]["weights_gib"])
    F["tps_ratio"] = f"{lmhead['tokens_per_s'] / prec['bf16']['tokens_per_s']:.1f}"
    step_ms, _ = profile_value("wall time")
    lm_ms, lm_share = profile_value("kernels: lm_head")
    marlin_ms, _ = profile_value("kernels: model forward")
    F["prof_step"], F["prof_lmhead"], F["prof_lmhead_share"] = f"{step_ms:.1f}", f"{lm_ms:.1f}", lm_share
    F["prof_forward"] = f"{marlin_ms:.1f}"
    F["prof_busy"] = profile_value("GPU busy")[1]
    sup = prec["w4a16"]["superseded"]
    F["art_old_tps"], F["art_old_ms"] = f"{sup['tokens_per_s']:.1f}", f"{1000 / sup['tokens_per_s']:.1f}"
    F["art_new_tps"] = F["w4_tps"]
    F["base_max"] = opt["baseline_per_stream_snac"]["max_streams"]
    F["bat_max"] = opt["batched_snac"]["max_streams"]
    sat = min(n for n, r in base.items() if r["snac_busy_fraction"] >= 0.99)
    F["base_snac_sat_n"] = sat
    F["base_ceiling"] = f"{base[64]['throughput_audio_s_per_s']:.1f}"
    F["bat_ceiling"] = f"{bat[64]['throughput_audio_s_per_s']:.1f}"
    F["ceiling_x"] = f"{bat[64]['throughput_audio_s_per_s'] / base[64]['throughput_audio_s_per_s']:.1f}"
    alone = {r["batch"]: r["ms_per_step"] for r in bench["results"]}
    F["alone_8"] = alone[8]
    F["server_8"] = f"{lv6c[8]['vllm_ms_per_token']}"
    F["snac_busy_8"] = f"{lv6c[8]['snac_busy_fraction']}"
    F["alone_16"], F["alone_32"] = alone[16], alone[32]
    F["ttfa1_old"] = f"{base[1]['ttfa_ms_p50']:.0f}"
    F["ttfa1_new"] = f"{lv6c[1]['ttfa_ms_p50']:.0f}"
    F["max_n"], F["margin_n"] = max_n, margin_n
    F["ttfa_max"] = f"{top['ttfa_ms_p50']:.0f}"
    F["vllm_ms_max"] = f"{top['vllm_ms_per_token']}"
    F["rtf_top"] = f"{top['rtf_p95']}"
    F["gpu"], F["vllm"], F["torch"] = sweep["gpu"], sweep["vllm_version"], sweep["torch_version"]
    mb = pipe["encode_microbench"]
    opts = {o["variant"]: o for o in mb["options"]}
    fp16 = opts["fp16 autocast"]
    comp = opts["torch.compile(encoder, dynamic=True), fp32"]
    F["fp16_gain"] = f"{(fp16['audio_hours_per_min'] / opts['fp32 eager']['audio_hours_per_min'] - 1) * 100:.0f}%"
    F["fp16_diff_lo"] = f"{(1 - max(fp16['code_agreement'])) * 100:.0f}"
    F["fp16_diff_hi"] = f"{(1 - min(fp16['code_agreement'])) * 100:.0f}"
    F["compile_x"] = comp["speedup"].replace("x on 3 batch shapes", "×").replace("-", "–")
    F["compile_agree_lo"] = f"{min(comp['code_agreement']) * 100:.1f}"
    F["compile_agree_hi"] = f"{max(comp['code_agreement']) * 100:.1f}"
    F["gpu_cap"] = full["stages"]["tokenize_gpu"]["hours_per_min_capacity"]
    F["e2e"] = full["end_to_end_audio_hours_per_min"]
    F["gpu_util_full"] = f"{full['utilisation']['gpu_pct_mean']:.0f}%"
    F["cpu_headroom"] = f"{scal['results']['measured']['pipeline_cpu_headroom_x']:.0f}"
    F["pipe_hours"] = f"{full['data']['trimmed_audio_hours']:.1f}"
    F["pipe_clips"] = f"{full['data']['clips_out']:,}"
    F["tok_per_hour"] = f"{pb_h:,}"
    F["prof_gemm_share"] = f"{mb['profile_share_of_gpu_time']['gemm_convs'] * 100:.0f}%"
    F["prof_elem_share"] = (f"{(mb['profile_share_of_gpu_time']['snake_activation_fused'] + mb['profile_share_of_gpu_time']['other_elementwise']) * 100:.0f}%")
    r_ = scal["results"]
    F["sc_gpus_100"] = r_["serving"]["gpus"]["100"]
    F["sc_gpus_1000"] = r_["serving"]["gpus"]["1000"]
    F["sc_gpus_100_c"] = r_["serving"]["gpus_conservative"]["100"]
    F["sc_gpus_1000_c"] = r_["serving"]["gpus_conservative"]["1000"]
    F["sc_hours"] = f"{r_['data']['hours']:,}"
    F["sc_tokens"] = f"{r_['data']['tokens']:.2e}"
    F["sc_l4h"] = f"{r_['pipeline']['l4_hours']:,.0f}"
    F["sc_800m"] = f"{r_['data']['compute']['800M']['gpu_hours']:,.0f}"
    F["sc_48b"] = f"{r_['data']['compute']['4.8B']['gpu_hours']:,.0f}"
    F["sc_mfu"] = f"{scal['assumed_constants']['MFU']:.0%}"
    F["vocab"] = f"{scal['assumed_constants']['VOCAB']:,}"
    F["sc_state_bytes"] = r_["bytes_per_param"]
    return F


def render(template_text, F):
    def sub(m):
        key = m.group(1)
        if key not in F:
            raise KeyError(f"README template placeholder {{{{{key}}}}} has no value")
        return str(F[key])

    body = re.sub(r"\{\{(\w+)\}\}", sub, template_text)
    header = ("<!-- Generated by scripts/make_readme.py from docs/README.template.md and results/. "
              "Edit the template, not this file. -->\n")
    return header + body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = render(TEMPLATE.read_text(), build())
    if args.check:
        if README.read_text() != text:
            print("README.md is out of date: run python scripts/make_readme.py")
            sys.exit(1)
        print("README.md is up to date")
        return
    README.write_text(text)
    print("wrote README.md")


if __name__ == "__main__":
    main()
