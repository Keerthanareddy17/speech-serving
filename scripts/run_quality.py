"""Step 5: quality baseline. Synthesize the fixed sentence set with each config, transcribe with Whisper,
score WER/CER.

  python scripts/run_quality.py                       # generate (one process per config), transcribe, score
  python scripts/run_quality.py --skip-generate       # re-transcribe + re-score existing clips
  python scripts/run_quality.py --score-only          # re-score existing transcripts

Clips go to Drive (/content/drive/MyDrive/speech-infra/quality/<run>/<preset>/); 3 examples per config go
to results/samples/quality/. Results: results/quality_baseline.json + results/quality_baseline_clips.csv.
"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import quality as Q  # noqa: E402

PRESETS = ["bf16", "w4a16", "w4a16_lmhead"]
DRIVE = Path("/content/drive/MyDrive/speech-infra/quality")
EXAMPLES = [4, 10, 28]   # a date, the names, a tongue twister (seed 0) -> results/samples/quality/
OUT_JSON = ROOT / "results/quality_baseline.json"
OUT_CSV = ROOT / "results/quality_baseline_clips.csv"
CSV_FIELDS = ["preset", "sid", "seed", "file", "tokens", "audio_s", "finish_reason", "wer", "cer",
              "ref_words", "word_errors", "wer_fmt", "cer_fmt", "word_errors_fmt", "expected", "transcribed",
              "ref_norm", "hyp_norm", "ref_fmt", "hyp_fmt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", nargs="+", default=PRESETS)
    ap.add_argument("--run", default="baseline", help="subfolder under the Drive quality dir")
    ap.add_argument("--skip-generate", action="store_true")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--worker-generate", help=argparse.SUPPRESS)
    ap.add_argument("--worker-transcribe", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    run_dir = DRIVE / args.run
    dirs = [run_dir / p for p in args.presets]

    if args.worker_generate:                                  # child process: one engine
        Q.run_generate(args.worker_generate, run_dir / args.worker_generate)
        return
    if args.worker_transcribe:                                # child process: Whisper only
        Q.transcribe(dirs)
        return

    if not (args.skip_generate or args.score_only):
        for p in args.presets:
            subprocess.run([sys.executable, __file__, "--run", args.run, "--worker-generate", p], check=True)
    if not args.score_only:
        subprocess.run([sys.executable, __file__, "--run", args.run, "--presets", *args.presets,
                        "--worker-transcribe"], check=True)

    # ---- score ----
    normalize = Q.make_normalizer()
    normalize_fmt = Q.make_format_insensitive(normalize)
    all_rows, per, meta = [], {}, {}
    for p, d in zip(args.presets, dirs):
        manifest = json.loads((d / "manifest.json").read_text())
        tr = json.loads((d / "transcripts.json").read_text())
        rows = Q.score_clips(manifest, tr["transcripts"], normalize, normalize_fmt)
        all_rows += rows
        worst = sorted(rows, key=lambda r: (-r["wer"], r["sid"], r["seed"]))[:5]
        worst_fmt = sorted(rows, key=lambda r: (-r["wer_fmt"], r["sid"], r["seed"]))[:5]
        cols = ("file", "wer", "wer_fmt", "cer", "expected", "transcribed", "ref_norm", "hyp_norm")
        per[p] = {
            "model": manifest["model"],
            "summary": Q.summarize(rows),
            "summary_format_insensitive": Q.summarize(rows, "wer_fmt"),
            "seed_noise": Q.seed_noise(rows),
            "seed_noise_format_insensitive": Q.seed_noise(rows, "wer_fmt"),
            "worst5": [{k: r[k] for k in cols} for r in worst],
            "worst5_format_insensitive": [{k: r[k] for k in cols} for r in worst_fmt],
        }
        meta[p] = {k: manifest[k] for k in ("date", "gpu", "model", "vllm_version", "torch_version", "engine",
                                            "sampling", "chunk", "voice", "seeds", "path")}
        meta["whisper"] = tr["meta"]
        qdir = ROOT / "results/samples/quality"
        qdir.mkdir(parents=True, exist_ok=True)
        for sid in EXAMPLES:
            shutil.copy(d / Q.clip_name(sid, 0), qdir / f"{p}_{Q.clip_name(sid, 0)}")

    comparisons = {}
    if "bf16" in per:
        base = [r for r in all_rows if r["preset"] == "bf16"]
        for p in args.presets:
            if p != "bf16":
                other = [r for r in all_rows if r["preset"] == p]
                comparisons[f"{p}_minus_bf16"] = Q.paired_bootstrap(base, other)
                comparisons[f"{p}_minus_bf16_format_insensitive"] = Q.paired_bootstrap(base, other, key="wer_fmt")

    first = next(iter(meta.values()))
    result = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": first["gpu"], "vllm_version": first["vllm_version"], "torch_version": first["torch_version"],
        "models": {p: per[p]["model"] for p in per},
        "config": {"sentences_file": str(Q.SENTENCES_FILE.relative_to(ROOT)), "n_sentences": len(Q.load_sentences()),
                   "seeds": list(Q.DEFAULT_SEEDS), "voice": Q.VOICE, "clips_dir": str(run_dir),
                   "normalizer": "Whisper EnglishTextNormalizer (transformers WhisperTokenizer.normalize)",
                   "format_insensitive": "supplementary: also ignores number formatting (quality.make_format_insensitive)",
                   "metrics": "jiwer wer/cer per clip on normalised text; corpus_wer = total errors / total words",
                   "bootstrap": "paired over sentences (mean of seeds), 10000 resamples, 95% percentile CI",
                   "generation": meta},
        "per_config": per,
        "comparisons": comparisons,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2) + "\n")
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    for p in per:
        s = per[p]["summary"]
        print(f"RESULT {p:13s} WER mean {s['wer_mean']:.3f} median {s['wer_median']:.3f} p90 {s['wer_p90']:.3f} "
              f"| >20%: {s['clips_wer_gt_20pct']} | CER {s['cer_mean']:.3f} | corpus WER {s['corpus_wer']:.3f} "
              f"| not stopped {s['not_stopped_clips']}")
        s = per[p]["summary_format_insensitive"]
        print(f"  fmt-insens.   WER mean {s['wer_mean']:.3f} median {s['wer_median']:.3f} p90 {s['wer_p90']:.3f} "
              f"| >20%: {s['clips_wer_gt_20pct']} | CER {s['cer_mean']:.3f} | corpus WER {s['corpus_wer']:.3f}")
    for k, v in comparisons.items():
        print(f"COMPARE {k}: {v}")
    print(f"saved {OUT_JSON} + {OUT_CSV}")


if __name__ == "__main__":
    main()
