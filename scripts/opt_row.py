"""Append (or replace) a row in results/optimizations.csv from a load-test JSON (step 6b screening/final).

  python scripts/opt_row.py --id chunk4 --step 6b --phase screen --json results/screen/chunk4.json \
      --change "growing chunks, cap 4" --status adopted --reason "..."

max_streams = highest tested level N that passes (p95 RTF < 1.0, no request with a gap, no errors), with every
lower tested level also passing; 0 if the lowest tested level already fails. "top" = highest tested level.
"""

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/optimizations.csv"
FIELDS = ["id", "step", "phase", "change", "status", "decision_reason", "levels", "max_streams",
          "ttfa_p50_ms_n1", "ttfa_p95_ms_n1", "ttfa_p50_ms_at_max", "ttfa_p95_ms_at_max",
          "throughput_ceiling_audio_s_per_s", "top_level", "rtf_p95_at_top", "pct_gaps_at_top",
          "vllm_ms_per_token_at_top", "snac_ms_per_call_at_top", "snac_busy_at_top", "snac_batch_mean_at_top",
          "gpu", "model", "vllm_version", "date", "source", "notes"]


def passes(level):
    return (level["errors"] == 0 and level["rtf_p95"] is not None and level["rtf_p95"] < 1.0
            and level["pct_requests_with_gaps"] == 0)


def row_from_json(path, **kw):
    R = json.loads(Path(path).read_text())
    levels = R["levels"]
    best = None
    for lv in levels:
        if passes(lv):
            best = lv
        else:
            break
    by = {lv["streams"]: lv for lv in levels}
    top = levels[-1]
    return {
        **kw,
        "levels": " ".join(str(lv["streams"]) for lv in levels),
        "max_streams": best["streams"] if best else 0,
        "ttfa_p50_ms_n1": by[1]["ttfa_ms_p50"] if 1 in by else "",
        "ttfa_p95_ms_n1": by[1]["ttfa_ms_p95"] if 1 in by else "",
        "ttfa_p50_ms_at_max": best["ttfa_ms_p50"] if best else "",
        "ttfa_p95_ms_at_max": best["ttfa_ms_p95"] if best else "",
        "throughput_ceiling_audio_s_per_s": max(lv["throughput_audio_s_per_s"] or 0 for lv in levels),
        "top_level": top["streams"], "rtf_p95_at_top": top["rtf_p95"], "pct_gaps_at_top": top["pct_requests_with_gaps"],
        "vllm_ms_per_token_at_top": top.get("vllm_ms_per_token"), "snac_ms_per_call_at_top": top.get("snac_ms_per_chunk_p50"),
        "snac_busy_at_top": top.get("snac_busy_fraction"), "snac_batch_mean_at_top": top.get("snac_batch_mean"),
        "gpu": R["gpu"], "model": R["model"], "vllm_version": R["vllm_version"], "date": R["date"][:10],
        "source": str(Path(path).resolve().relative_to(ROOT)),
    }


def upsert(row):
    rows = list(csv.DictReader(CSV.open())) if CSV.exists() else []
    rows = [r for r in rows if r["id"] != row["id"]] + [row]
    with CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--step", required=True)
    ap.add_argument("--phase", required=True, help="screen | sweep | final | micro")
    ap.add_argument("--json", help="load-test JSON; omit for rows without a load test")
    ap.add_argument("--change", required=True)
    ap.add_argument("--status", required=True, help="adopted | rejected | reference | baseline | tried, not adopted")
    ap.add_argument("--reason", default="")
    ap.add_argument("--notes", default="")
    a = ap.parse_args()
    kw = dict(id=a.id, step=a.step, phase=a.phase, change=a.change, status=a.status, decision_reason=a.reason,
              notes=a.notes)
    row = row_from_json(a.json, **kw) if a.json else kw
    upsert(row)
    print({k: row.get(k) for k in ("id", "status", "levels", "max_streams", "ttfa_p50_ms_at_max",
                                    "throughput_ceiling_audio_s_per_s", "vllm_ms_per_token_at_top",
                                    "snac_ms_per_call_at_top", "snac_busy_at_top")})


if __name__ == "__main__":
    main()
