"""Step 6b "nothing broke" check: does a streaming change alter the audio vs a whole-sequence decode?

  python scripts/step6b_audio_check.py --variant chunk4 --chunk-frames 4
  python scripts/step6b_audio_check.py --variant fp16 --snac-dtype float16
Appends/updates results/step6b_audio_check.json.

First run: generates the step 2 paragraph once with the engine (w4a16_lmhead, seed 0) and saves its frames
plus each frame's arrival time to Drive. Every variant then replays the chunk planner on exactly those
frames, with timing hints derived from the recorded arrival times (so growing chunks grow as they would
live), decodes each window with the variant's SNAC settings and noise OFF, stitches, and compares to an
fp32, noise-off, whole-sequence decode of the same frames (the reference).
"""

import argparse
import asyncio
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402
from speechserve.codec import ChunkPlanner, FrameAssembler, SNACDecoder  # noqa: E402
from speechserve.loadtest import TEXTS  # noqa: E402

FRAMES_FILE = Path("/content/drive/MyDrive/speech-infra/step6b/paragraph_frames.json")
FRAMES_DIR = FRAMES_FILE.parent
# step 6c: clips that start with speech immediately (the paragraph starts in near-silence)
EXTRA_TEXTS = {"yes": "Yes, absolutely.", "see_you": "See you tomorrow at noon.",
               "peter_piper": "Peter Piper picked a peck of pickled peppers."}


def frames_file(name):
    return FRAMES_FILE if name == "paragraph" else FRAMES_DIR / f"frames_{name}.json"
OUT = ROOT / "results/step6b_audio_check.json"
DECODE_S = 0.02  # assumed SNAC time per chunk in the replay (only affects when chunks are allowed to grow)


async def record_frames(names):
    from speechserve.engine import TTSEngine

    engine = TTSEngine("w4a16_lmhead")
    async for _ in engine.stream(TEXTS["sentence"]):        # warmup (step 2: first request differs)
        pass
    for name in names:
        text = TEXTS["paragraph"] if name == "paragraph" else EXTRA_TEXTS[name]
        asm, times, t0 = FrameAssembler(), [], time.perf_counter()
        async for ids in engine.stream(text):
            before = len(asm.frames)
            asm.push(ids)
            times += [time.perf_counter() - t0] * (len(asm.frames) - before)
        FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        frames_file(name).write_text(json.dumps({"preset": "w4a16_lmhead", "text": text, "seed": 0,
                                                 "frames": asm.frames, "arrival_s": times}))
    engine.shutdown()


def replay_windows(frames, arrival, cfg):
    """Run the planner frame by frame with the hints stream_audio would compute live."""
    p, wins, t_first, n_first, sent = ChunkPlanner(cfg), [], None, 0, 0.0
    for n in range(1, len(frames) + 1):
        now = arrival[n - 1]
        hints = {}
        if (cfg.chunk_frames > 1 or p.early) and t_first is not None and n > n_first:
            hints = {"buffer_s": sent - (now - t_first), "frame_s": (now - t_first) / (n - n_first),
                     "decode_s": DECODE_S}
        for w in p.plan(n, **hints):
            if t_first is None:
                t_first, n_first = now + DECODE_S, n
            sent += (w.emit_end - w.emit_start) * C.SAMPLES_PER_FRAME / C.SAMPLE_RATE
            wins.append(w)
    return wins + p.plan(len(frames), final=True)


def compare(x, ref):
    err = x - ref
    return {"snr_db": round(float(10 * np.log10(np.sum(ref ** 2) / max(np.sum(err ** 2), 1e-20))), 2),
            "max_abs_diff": round(float(np.abs(err).max()), 5), "correlation": round(float(np.corrcoef(x, ref)[0, 1]), 5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--chunk-frames", type=int, default=1)
    ap.add_argument("--snac-dtype", default="float32")
    ap.add_argument("--snac-graphs", action="store_true")
    ap.add_argument("--first-wait-frames", type=int, default=4, help="step 6c: frames the first chunk waits for")
    ap.add_argument("--text", default="paragraph", choices=["paragraph", *EXTRA_TEXTS])
    args = ap.parse_args()
    missing = [n for n in ["paragraph", *EXTRA_TEXTS] if not frames_file(n).exists()]
    if missing:
        asyncio.run(record_frames(missing))
    data = json.loads(frames_file(args.text).read_text())
    frames, arrival = data["frames"], data["arrival_s"]

    import torch

    ref_dec = SNACDecoder()                                   # fp32 reference
    ref_dec.set_noise(False)
    whole, _ = ref_dec.decode(frames)
    dec = ref_dec if args.snac_dtype == "float32" and not args.snac_graphs else SNACDecoder(dtype=args.snac_dtype)
    dec.set_noise(False)
    if args.snac_graphs:
        dec.capture_graphs(4, 1)                               # replay windows one at a time, like decode()

    cfg = C.ChunkConfig(chunk_frames=args.chunk_frames, first_wait_frames=args.first_wait_frames)
    wins = replay_windows(frames, arrival, cfg)
    pieces = [dec.decode(frames[w.decode_start:w.decode_end])[0][w.keep_samples] for w in wins]
    stitched = np.concatenate(pieces)
    start = wins[0].emit_start
    ref = whole[start * C.SAMPLES_PER_FRAME: start * C.SAMPLES_PER_FRAME + len(stitched)]
    assert len(ref) == len(stitched)
    bounds = np.cumsum([len(p) for p in pieces])[:-1]
    dx, dref = np.abs(np.diff(stitched)), np.abs(np.diff(ref))
    sizes = [w.emit_end - w.emit_start for w in wins[1:-1]]   # later chunks, excluding first and tail
    decoded = sum(w.decode_end - w.decode_start for w in wins)
    emitted = sum(w.emit_end - w.emit_start for w in wins)
    row = {
        "variant": args.variant,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0), "model": data["preset"] + " tokens; SNAC " + C.SNAC_MODEL,
        "vllm_version": __import__("importlib.metadata").metadata.version("vllm"),
        "torch_version": torch.__version__, "python_version": platform.python_version(),
        "text": data["text"],
        "config": {"chunk": cfg.to_dict(), "snac_dtype": args.snac_dtype, "snac_graphs": args.snac_graphs, "noise": "off",
                   "tf32": torch.backends.cudnn.allow_tf32, "reference": "fp32 whole-sequence decode, noise off",
                   "frames_file": str(frames_file(args.text)), "replay_decode_s": DECODE_S},
        "frames": len(frames), "windows": len(wins),
        "later_chunk_sizes": {str(k): sizes.count(k) for k in sorted(set(sizes))},
        "snac_frames_decoded_per_emitted_frame": round(decoded / emitted, 3),
        "stream_vs_whole": compare(stitched, ref),
        "boundary": {"n": len(bounds), "max_jump": round(float(max(dx[b - 1] for b in bounds)), 4),
                     "max_jump_excess_vs_whole": round(float(max(dx[b - 1] - dref[b - 1] for b in bounds)), 4),
                     "p99_9_jump_anywhere": round(float(np.percentile(dx, 99.9)), 4)},
    }
    # step 6c: the start of the clip (what the listener hears first) and the first chunk boundary
    head = int(0.3 * C.SAMPLE_RATE)
    b0 = int(bounds[0])
    row["first_300ms"] = compare(stitched[:head], ref[:head])
    row["first_300ms"]["ref_rms"] = round(float(np.sqrt(np.mean(ref[:head] ** 2))), 4)   # is there speech?
    row["first_boundary"] = {"sample": b0, "jump": round(float(dx[b0 - 1]), 4),
                             "whole_jump_same_sample": round(float(dref[b0 - 1]), 4),
                             "excess": round(float(dx[b0 - 1] - dref[b0 - 1]), 4)}
    row["starts_at_frame"] = start
    row["right_context_first10"] = [w.decode_end - w.emit_end for w in wins[:10]]
    if args.snac_dtype != "float32":
        import torch as _t
        from speechserve.codec import frames_to_layers
        with _t.inference_mode():                              # whole sequence, eager, in the variant dtype
            codes = [_t.from_numpy(x).unsqueeze(0).cuda() for x in frames_to_layers(frames)]
            w2 = dec.model.decode(codes).float().squeeze().cpu().numpy()
        row["whole_dtype_vs_whole_fp32"] = compare(w2, whole)
    rows = json.loads(OUT.read_text()) if OUT.exists() else []
    rows = [r for r in rows if r["variant"] != args.variant] + [row]
    OUT.write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps({k: row[k] for k in ("variant", "windows", "starts_at_frame", "right_context_first10",
                                          "stream_vs_whole", "first_300ms", "first_boundary", "boundary")}
                     | ({"whole_dtype_vs_whole_fp32": row["whole_dtype_vs_whole_fp32"]} if "whole_dtype_vs_whole_fp32" in row else {})))


if __name__ == "__main__":
    main()
