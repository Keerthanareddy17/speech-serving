"""Step 1d: summarise a vLLM torch-profiler trace into a per-decode-step breakdown (markdown to stdout).

  python scripts/analyze_trace.py /tmp/prof_w4a16 > results/step1d_profile.md

GPU kernels are classified as:
  model forward  - replayed from a CUDA graph (non-zero "graph id")
  lm_head        - the bf16 gemv computing the 157k-wide logits (runs eagerly, outside the graph)
  sampling       - logits processing + sampling kernels (penalties, temperature, top-p, gumbel, argmax, ...)
  bookkeeping    - other eager kernels and copies (block tables, slot mappings, token bookkeeping)
Idle = time inside the decode window when no kernel is running on the GPU.
"""

import collections
import gzip
import json
import re
import sys
from pathlib import Path

DECODE_STEP = "execute_context_0(0)_generation_1(1)"  # vLLM's annotation for a 1-token decode step
SAMPLING = re.compile(r"penalt|temperature|topp|topk|top_k|top_p|gumbel|sample|argmax|reduce_kernel|softmax|"
                      r"logit|index_elementwise|gather|exponential|cumsum|sort", re.I)


def classify(e):
    if e["args"].get("graph id"):
        return "model forward"
    if "gemv" in e["name"]:
        return "lm_head"
    if SAMPLING.search(e["name"]):
        return "sampling"
    return "bookkeeping"


def union_ms(intervals):
    total, end = 0.0, -1.0
    for s, e in sorted(intervals):
        if e > end:
            total += e - max(s, end)
            end = e
    return total / 1000


def main():
    trace = next(Path(sys.argv[1]).glob("*.trace.json.gz"))
    events = [e for e in json.load(gzip.open(trace))["traceEvents"] if e.get("ph") == "X"]
    steps = sorted((e for e in events if e.get("cat") == "gpu_user_annotation" and e["name"] == DECODE_STEP),
                   key=lambda e: e["ts"])
    n = len(steps)
    # Decode window on the GPU: first decode step start to last decode step end.
    t0, t1 = steps[0]["ts"], steps[-1]["ts"] + steps[-1]["dur"]
    gpu = sorted(((e["ts"], e["ts"] + e["dur"], e) for e in events
                  if e.get("cat") in ("kernel", "gpu_memcpy") and t0 <= e["ts"] < t1), key=lambda x: x[0])

    window = (t1 - t0) / 1000
    busy = union_ms([(s, e) for s, e, _ in gpu])
    by_cat = collections.Counter()
    by_kernel = collections.Counter()
    calls = collections.Counter()
    for s, e, ev in gpu:
        cat = classify(ev) if ev.get("cat") == "kernel" else "bookkeeping"
        by_cat[cat] += (e - s) / 1000
        by_kernel[(ev["name"], cat)] += (e - s) / 1000
        calls[(ev["name"], cat)] += 1

    # Idle gaps, labelled by the category of the kernel before -> after.
    gaps = collections.Counter()
    end, prev = gpu[0][1], gpu[0][2]
    for s, e, ev in gpu[1:]:
        if s > end:
            before = classify(prev) if prev.get("cat") == "kernel" else "bookkeeping"
            after = classify(ev) if ev.get("cat") == "kernel" else "bookkeeping"
            gaps[f"{before} -> {after}"] += (s - end) / 1000
        if e > end:
            end, prev = e, ev

    per = lambda ms: ms / n  # noqa: E731
    print(f"Trace: `{trace.name}`, {n} decode steps (1 token each), batch size 1.\n")
    print("## Per decode step (averages)\n")
    print("| | ms/step | share |\n|---|---|---|")
    print(f"| **wall time (GPU decode window / steps)** | **{per(window):.2f}** | 100% |")
    print(f"| GPU busy (union of kernels) | {per(busy):.2f} | {busy / window:.0%} |")
    print(f"| GPU idle (gaps between kernels) | {per(window - busy):.2f} | {(window - busy) / window:.0%} |")
    for cat in ("model forward", "lm_head", "sampling", "bookkeeping"):
        print(f"| &nbsp;&nbsp;kernels: {cat} | {per(by_cat[cat]):.2f} | {by_cat[cat] / window:.0%} |")
    print("\n## Where the idle time sits (gap between kernel categories)\n")
    print("| gap | ms/step |\n|---|---|")
    for k, v in gaps.most_common(6):
        print(f"| {k} | {per(v):.2f} |")
    print("\n## Top 10 kernels by GPU time\n")
    print("| kernel | category | calls/step | ms/step |\n|---|---|---|---|")
    for (name, cat), ms in by_kernel.most_common(10):
        short = re.sub(r"<.*", "<...>", name)[:70]
        print(f"| `{short}` | {cat} | {calls[(name, cat)] / n:.0f} | {per(ms):.3f} |")


if __name__ == "__main__":
    main()
