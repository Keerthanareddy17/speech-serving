"""Step 2: single-stream streaming measurements (engine + SNAC, in-process, no server).

  python scripts/step2_stream.py                         # runs each config in its own process
  python scripts/step2_stream.py --configs w4a16_lmhead  # just one
Writes results/step2_single_stream.json and results/samples/step2_<config>_stream.wav.

Per run: time to first audio (TTFA), playback gaps and the buffer needed to avoid them, tokens, audio
seconds, RTF, and SNAC decode time per chunk. The last paragraph run is also decoded as one whole
sequence and compared with the stitched stream (same tokens, no new generation).
"""

import argparse
import asyncio
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402

OUT_JSON = ROOT / "results/step2_single_stream.json"
CONFIGS = ["w4a16_lmhead", "w4a16"]
TEXTS = {
    "sentence": "Hi, I'm testing a realtime speech server on a single L4 GPU.",  # step 1 sentence
    "paragraph": ("Streaming speech synthesis is mostly a race against the listener. The model has to "
                  "produce each slice of audio before the previous one finishes playing, or the voice "
                  "stutters. On a single small GPU, that means every millisecond of decoding matters, "
                  "from the matrix math all the way to the final audio codec."),
}
RUNS = 10


# ---- per-run metrics -----------------------------------------------------------------------------------
def playback_metrics(arrivals, durations):
    """Playback starts when chunk 0 arrives. A gap = a chunk arriving after the audio before it ended.
    buffer_needed = the start delay that would make every chunk arrive in time."""
    a0, played, gaps, gap_s, need = arrivals[0], 0.0, 0, 0.0, 0.0
    play_end = a0
    for a, d in zip(arrivals, durations):
        need = max(need, a - (a0 + played))   # lateness vs. an uninterrupted schedule
        if a > play_end + 1e-9 and played > 0:
            gaps += 1
            gap_s += a - play_end
        play_end = max(play_end, a) + d
        played += d
    return gaps, gap_s, need


async def timed_stream(engine, decoder, text, chunk_cfg):
    from speechserve.codec import FrameAssembler, stream_audio

    asm = FrameAssembler()
    t0 = time.perf_counter()
    chunks = [c async for c in stream_audio(engine.stream(text, C.DEFAULT_VOICE), decoder, chunk_cfg, asm)]
    t_end = time.perf_counter()
    arrivals = [c.ready_s - t0 for c in chunks]
    durations = [len(c.pcm) / C.SAMPLE_RATE for c in chunks]
    gaps, gap_s, need = playback_metrics(arrivals, durations)
    audio_s = sum(durations)
    dec = [c.decode_ms for c in chunks]
    run = {
        "ttfa_ms": arrivals[0] * 1000,
        "gap_count": gaps,
        "gap_total_ms": gap_s * 1000,
        "buffer_needed_ms": need * 1000,
        "total_tokens": asm.n_tokens,
        "audio_frames": len(asm.frames),
        "audio_s": audio_s,
        "wall_s": t_end - t0,
        "rtf": (t_end - t0) / audio_s,
        "chunks": len(chunks),
        "snac_ms_mean": float(np.mean(dec)),
        "snac_ms_total": float(np.sum(dec)),
    }
    return run, chunks, asm


def summarize(runs):
    out = {}
    for k in runs[0]:
        v = np.array([r[k] for r in runs], dtype=float)
        out[k] = {"p50": round(float(np.percentile(v, 50)), 3), "p95": round(float(np.percentile(v, 95)), 3),
                  "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}
    return out


# ---- streaming vs whole-sequence check -----------------------------------------------------------------
def compare(x, ref):
    err = x - ref
    return {"max_abs_diff": float(np.abs(err).max()), "rms_diff": float(np.sqrt(np.mean(err**2))),
            "ref_rms": float(np.sqrt(np.mean(ref**2))),
            "snr_db": float(10 * np.log10(np.sum(ref**2) / max(np.sum(err**2), 1e-20))),
            "correlation": float(np.corrcoef(x, ref)[0, 1])}


def boundary_check(x, ref, boundaries, win=64):
    """Clicks show up as a sample-to-sample jump at a chunk boundary that the whole decode doesn't have."""
    dx, dref = np.abs(np.diff(x)), np.abs(np.diff(ref))
    jumps = np.array([dx[b - 1] for b in boundaries])        # x[b] - x[b-1]
    ref_jumps = np.array([dref[b - 1] for b in boundaries])
    near = np.zeros(len(x), bool)
    for b in boundaries:
        near[max(0, b - win):b + win] = True
    err = x - ref
    return {
        "n_boundaries": len(boundaries),
        "max_jump_at_boundary": float(jumps.max()),
        "max_jump_excess_vs_whole": float((jumps - ref_jumps).max()),
        "p99_9_jump_anywhere": float(np.percentile(dx, 99.9)),
        "rms_diff_near_boundaries": float(np.sqrt(np.mean(err[near] ** 2))),
        "rms_diff_elsewhere": float(np.sqrt(np.mean(err[~near] ** 2))),
    }


def replay_stream(decoder, frames, chunk_cfg):
    """Re-run the same chunk windows offline on the same frames (used for the noise-off comparison)."""
    from speechserve.codec import ChunkPlanner

    p, out = ChunkPlanner(chunk_cfg), []
    wins = [w for n in range(1, len(frames) + 1) for w in p.plan(n)] + p.plan(len(frames), final=True)
    for w in wins:
        audio, _ = decoder.decode(frames[w.decode_start:w.decode_end])
        out.append(audio[w.keep_samples])
    return np.concatenate(out)


def streaming_check(decoder, chunks, frames, chunk_cfg):
    spf = C.SAMPLES_PER_FRAME
    stitched = np.concatenate([c.pcm for c in chunks])
    start = chunks[0].window.emit_start                # 1: upstream never emits frame 0
    sl = slice(start * spf, start * spf + len(stitched))
    boundaries = list(np.cumsum([len(c.pcm) for c in chunks])[:-1])

    whole_a, _ = decoder.decode(frames)
    whole_b, _ = decoder.decode(frames)
    decoder.set_noise(False)
    whole_det, _ = decoder.decode(frames)
    stream_det = replay_stream(decoder, frames, chunk_cfg)
    decoder.set_noise(True)
    assert len(whole_a) == len(frames) * spf and len(stitched) == len(stream_det)
    return {
        "compared_range": f"whole-decode samples [{sl.start}:{sl.stop}] (frames {start}..{len(frames) - 1})",
        "noise_floor_whole_vs_whole": compare(whole_b[sl], whole_a[sl]),
        "stream_vs_whole_noise_on": {**compare(stitched, whole_a[sl]),
                                     **boundary_check(stitched, whole_a[sl], boundaries)},
        "stream_vs_whole_noise_off": {**compare(stream_det, whole_det[sl]),
                                      **boundary_check(stream_det, whole_det[sl], boundaries)},
    }


# ---- worker: one config, one process ------------------------------------------------------------------
async def worker(preset, out_path):
    import soundfile as sf
    import torch
    import vllm

    from speechserve.codec import SNACDecoder
    from speechserve.engine import TTSEngine

    sampling, chunk_cfg = C.SamplingConfig(), C.ChunkConfig()
    t0 = time.perf_counter()
    engine = TTSEngine(preset, sampling)
    load_s = time.perf_counter() - t0
    decoder = SNACDecoder()

    for text in TEXTS.values():                        # warmup: one full stream per text, same params
        await timed_stream(engine, decoder, text, chunk_cfg)

    per_text, check, wav = {}, None, None
    for name, text in TEXTS.items():
        runs, token_counts = [], set()
        for i in range(RUNS):
            run, chunks, asm = await timed_stream(engine, decoder, text, chunk_cfg)
            runs.append(run)
            token_counts.add(asm.n_tokens)
            print(f"[{preset}] {name} run {i}: TTFA {run['ttfa_ms']:.0f} ms, gaps {run['gap_count']}, "
                  f"buffer {run['buffer_needed_ms']:.0f} ms, {run['total_tokens']} tok, "
                  f"{run['audio_s']:.2f} s audio, RTF {run['rtf']:.3f}, SNAC {run['snac_ms_mean']:.1f} ms/chunk",
                  flush=True)
        per_text[name] = {"text": text, "distinct_token_counts": sorted(token_counts),
                          "summary": summarize(runs), "runs": [{k: round(v, 3) for k, v in r.items()} for r in runs]}
        if name == "paragraph":                        # last run: whole-decode check + saved WAV
            check = streaming_check(decoder, chunks, asm.frames, chunk_cfg)
            wav = ROOT / f"results/samples/step2_{preset}_stream.wav"
            wav.parent.mkdir(parents=True, exist_ok=True)
            sf.write(wav, np.concatenate([c.pcm for c in chunks]), C.SAMPLE_RATE, subtype="PCM_16")

    result = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gpu": torch.cuda.get_device_name(0),
        "model": engine.engine_args["model"],
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "config": {"preset": preset, "engine": engine.engine_args, "sampling": sampling.to_dict(),
                   "chunk": chunk_cfg.to_dict(), "voice": C.DEFAULT_VOICE, "snac_model": C.SNAC_MODEL,
                   "timed_runs_per_text": RUNS, "warmup": "one full stream per text, same params",
                   "snac_noise": "on (model default) for all timed runs"},
        "model_load_s": round(load_s, 1),
        "kv_cache_tokens": engine.kv_cache_tokens(),
        "texts": per_text,
        "streaming_vs_whole": check,
        "output_wav": str(wav.relative_to(ROOT)),
    }
    Path(out_path).write_text(json.dumps(result, indent=2) + "\n")
    engine.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=CONFIGS, choices=list(C.PRESETS))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        asyncio.run(worker(args.worker, args.out))
        return

    results = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for preset in args.configs:
        out = ROOT / f"results/.step2_{preset}.tmp.json"
        subprocess.run([sys.executable, __file__, "--worker", preset, "--out", str(out)], check=True)
        results[preset] = json.loads(out.read_text())
        out.unlink()
        OUT_JSON.write_text(json.dumps(results, indent=2) + "\n")
        s = results[preset]["texts"]
        for name in TEXTS:
            m = s[name]["summary"]
            print(f"RESULT {preset:13s} {name:9s} TTFA p50 {m['ttfa_ms']['p50']:.0f} / p95 {m['ttfa_ms']['p95']:.0f} ms"
                  f" | gaps p95 {m['gap_count']['p95']:.0f} | buffer p95 {m['buffer_needed_ms']['p95']:.0f} ms"
                  f" | RTF p50 {m['rtf']['p50']:.3f} | SNAC {m['snac_ms_mean']['p50']:.1f} ms/chunk")
    print(f"saved {OUT_JSON}")


if __name__ == "__main__":
    main()
