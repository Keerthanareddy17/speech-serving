"""Quality harness: synthesize a fixed sentence set through the serving path, transcribe it with Whisper,
and score WER/CER against the input text.

Three stages, each meant to run in its own process (the GPU holds either one vLLM engine or Whisper):
  1. `generate(preset, out_dir, seeds)`: TTSEngine + streaming SNAC decode (exactly the server's path),
     one clip at a time (batch 1, so seeded runs are reproducible), WAVs + manifest.json in out_dir.
  2. `transcribe(clip_dirs)`: faster-whisper on GPU, writes transcripts.json next to each manifest.
  3. `score(...)`: Whisper's English normaliser on both sides, jiwer WER/CER per clip, summaries, and a
     paired bootstrap over sentences for "is config X worse than bf16?".
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import numpy as np

from . import config as C

SENTENCES_FILE = Path(__file__).with_name("quality_sentences.txt")
DEFAULT_SEEDS = (0, 1)
VOICE = "tara"
WHISPER_MODEL = "large-v3"
WHISPER_ARGS = dict(language="en", beam_size=5, condition_on_previous_text=False)


def load_sentences() -> list[str]:
    lines = [line.strip() for line in SENTENCES_FILE.read_text().splitlines()]
    return [line for line in lines if line]


def clip_name(sid: int, seed: int) -> str:
    return f"s{sid:02d}_seed{seed}.wav"


# ---- 1. generation -----------------------------------------------------------------------------------
async def generate(preset: str, out_dir: Path, seeds=DEFAULT_SEEDS, decoder_kind: str = "batched") -> dict:
    import soundfile as sf
    import torch
    import vllm

    from .codec import BatchedChunkDecoder, FrameAssembler, SNACDecoder, stream_audio
    from .engine import TTSEngine

    out_dir.mkdir(parents=True, exist_ok=True)
    sentences = load_sentences()
    engine = TTSEngine(preset)
    decoder = BatchedChunkDecoder(SNACDecoder()) if decoder_kind == "batched" else SNACDecoder()
    chunk_cfg = C.ChunkConfig()

    async def synth(text, seed):
        asm, info = FrameAssembler(), {}
        t0 = time.perf_counter()
        pcm = [c.pcm async for c in stream_audio(engine.stream(text, VOICE, info=info, seed=seed),
                                                 decoder, chunk_cfg, asm)]
        return np.concatenate(pcm) if pcm else np.zeros(0, np.float32), asm, info, time.perf_counter() - t0

    await synth(sentences[0], 0)  # warmup: the first request after load samples differently (step 2)

    clips = []
    for sid, text in enumerate(sentences):
        for seed in seeds:
            audio, asm, info, wall = await synth(text, seed)
            name = clip_name(sid, seed)
            sf.write(out_dir / name, audio, C.SAMPLE_RATE, subtype="PCM_16")
            clips.append({"sid": sid, "seed": seed, "text": text, "file": name, "tokens": asm.n_tokens,
                          "audio_s": round(len(audio) / C.SAMPLE_RATE, 3), "wall_s": round(wall, 3),
                          "finish_reason": info.get("finish_reason")})
            print(f"[{preset}] s{sid:02d} seed{seed}: {asm.n_tokens} tok, {clips[-1]['audio_s']:.2f} s, "
                  f"{info.get('finish_reason')}", flush=True)

    manifest = {
        "preset": preset, "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": torch.cuda.get_device_name(0), "model": engine.engine_args["model"],
        "vllm_version": vllm.__version__, "torch_version": torch.__version__,
        "engine": engine.engine_args, "sampling": engine.sampling.to_dict(), "chunk": chunk_cfg.to_dict(),
        "voice": VOICE, "seeds": list(seeds),
        "path": f"TTSEngine.stream -> codec.stream_audio ({decoder_kind} SNAC decoder)",
        "clips": clips,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    engine.shutdown()
    return manifest


# ---- 2. transcription --------------------------------------------------------------------------------
def _preload_cuda12_cublas() -> None:
    """CTranslate2 (faster-whisper) is built against CUDA 12 and dlopens libcublas.so.12, which is
    installed (nvidia-cublas-cu12) but not on the loader path next to torch's CUDA 13 libraries."""
    import ctypes

    import nvidia

    lib = Path(list(nvidia.__path__)[0]) / "cublas" / "lib"
    for name in ("libcublasLt.so.12", "libcublas.so.12"):
        if (lib / name).exists():
            ctypes.CDLL(str(lib / name), mode=ctypes.RTLD_GLOBAL)


def load_audio_16k(path: Path) -> np.ndarray:
    """faster-whisper's own file reader breaks on the installed PyAV (19.x: unexpected `metadata_errors`
    argument), so read with soundfile and resample 24 kHz -> 16 kHz ourselves."""
    import soundfile as sf
    from scipy.signal import resample_poly

    audio, sr = sf.read(path, dtype="float32")
    assert sr == C.SAMPLE_RATE
    return resample_poly(audio, 2, 3).astype(np.float32)


def transcribe(clip_dirs: list[Path], model_name: str = WHISPER_MODEL) -> dict:
    _preload_cuda12_cublas()
    import faster_whisper
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device="cuda", compute_type="float16")
    for d in clip_dirs:
        manifest = json.loads((d / "manifest.json").read_text())
        out = {}
        t0 = time.perf_counter()
        for c in manifest["clips"]:
            segs, _ = model.transcribe(load_audio_16k(d / c["file"]), **WHISPER_ARGS)
            out[c["file"]] = " ".join(s.text.strip() for s in segs)
        meta = {"whisper_model": model_name, "faster_whisper_version": faster_whisper.__version__,
                "compute_type": "float16", "args": WHISPER_ARGS, "seconds": round(time.perf_counter() - t0, 1)}
        (d / "transcripts.json").write_text(json.dumps({"meta": meta, "transcripts": out}, indent=2) + "\n")
        print(f"transcribed {len(out)} clips in {d} ({meta['seconds']} s)", flush=True)
    return meta


# ---- 3. scoring ----------------------------------------------------------------------------------------
def make_normalizer():
    """Whisper's English normaliser (lowercase, punctuation, numbers -> digits, spelling variants),
    as shipped with the Whisper tokenizer in transformers."""
    from transformers import WhisperTokenizer

    tok = WhisperTokenizer.from_pretrained("openai/whisper-large-v3")
    return tok.normalize


def make_format_insensitive(normalize):
    """Supplementary normaliser that also ignores how numbers are *written*, which the Whisper normaliser
    handles unevenly (it turns "4:30" into "4 30" but keeps Whisper's "4.30", drops leading zeros in
    "6:05" -> "6 5", and keeps "3rd" on one side and "3" on the other). Applied to both sides:
      1. before normalising: drop : . - , between digits ("4:30", "4.30" -> "430"; "555-0142" -> "5550142")
      2. after: ordinals "3rd" -> "3"; "a m"/"p m" -> "am"/"pm"; join adjacent digit groups."""
    import re

    def f(text: str) -> str:
        text = re.sub(r"(?<=\d)[:.,\-](?=\d)", "", text)
        t = normalize(text)
        t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", t)
        t = re.sub(r"\b([ap]) m\b", r"\1m", t)
        while True:
            t2 = re.sub(r"\b(\d+) (\d+)\b", r"\1\2", t)
            if t2 == t:
                return t
            t = t2

    return f


def _wer_cer(ref: str, hyp: str):
    import jiwer

    if not hyp:
        return 1.0, 1.0, len(ref.split())
    m = jiwer.process_words(ref, hyp)
    return m.wer, jiwer.cer(ref, hyp), m.substitutions + m.deletions + m.insertions


def score_clips(manifest: dict, transcripts: dict, normalize, normalize_fmt=None) -> list[dict]:
    rows = []
    for c in manifest["clips"]:
        hyp_raw = transcripts[c["file"]]
        ref, hyp = normalize(c["text"]), normalize(hyp_raw)
        wer, cer, errs = _wer_cer(ref, hyp)
        row = {**{k: c[k] for k in ("sid", "seed", "file", "tokens", "audio_s", "finish_reason")},
               "preset": manifest["preset"], "expected": c["text"], "transcribed": hyp_raw,
               "ref_norm": ref, "hyp_norm": hyp, "wer": round(wer, 4), "cer": round(cer, 4),
               "ref_words": len(ref.split()), "word_errors": errs}
        if normalize_fmt:
            rf, hf = normalize_fmt(c["text"]), normalize_fmt(hyp_raw)
            wf, cf, ef = _wer_cer(rf, hf)
            row.update({"wer_fmt": round(wf, 4), "cer_fmt": round(cf, 4), "ref_fmt": rf, "hyp_fmt": hf,
                        "ref_words_fmt": len(rf.split()), "word_errors_fmt": ef})
        rows.append(row)
    return rows


def summarize(rows: list[dict], key: str = "wer") -> dict:
    sfx = key[3:]  # "" or "_fmt"
    w = np.array([r[key] for r in rows])
    return {
        "clips": len(rows),
        "wer_mean": round(float(w.mean()), 4),
        "wer_median": round(float(np.median(w)), 4),
        "wer_p90": round(float(np.percentile(w, 90)), 4),
        "clips_wer_gt_20pct": int((w > 0.20).sum()),
        "clips_wer_zero": int((w == 0).sum()),
        "cer_mean": round(float(np.mean([r["cer" + sfx] for r in rows])), 4),
        "corpus_wer": round(sum(r["word_errors" + sfx] for r in rows) / sum(r["ref_words" + sfx] for r in rows), 4),
        "not_stopped_clips": sum(r["finish_reason"] != "stop" for r in rows),
    }


def per_sentence_wer(rows: list[dict], key: str = "wer") -> dict[int, float]:
    """Mean WER per sentence id (averaged over seeds)."""
    by: dict[int, list[float]] = {}
    for r in rows:
        by.setdefault(r["sid"], []).append(r[key])
    return {sid: float(np.mean(v)) for sid, v in by.items()}


def paired_bootstrap(rows_a: list[dict], rows_b: list[dict], n: int = 10000, seed: int = 0,
                     key: str = "wer") -> dict:
    """Mean-WER difference (b - a), resampling the sentences (each with its seeds) with replacement.
    Pairing by sentence removes most of the sentence-difficulty variance."""
    a, b = per_sentence_wer(rows_a, key), per_sentence_wer(rows_b, key)
    sids = sorted(set(a) & set(b))
    d = np.array([b[s] - a[s] for s in sids])
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": round(float(d.mean()), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "sentences_worse": int((d > 0).sum()), "sentences_better": int((d < 0).sum()),
            "sentences_equal": int((d == 0).sum())}


def seed_noise(rows: list[dict], key: str = "wer") -> dict:
    """How much WER moves between two seeds of the same config and sentence: the noise floor."""
    by: dict[int, dict[int, float]] = {}
    for r in rows:
        by.setdefault(r["sid"], {})[r["seed"]] = r[key]
    pairs = [v for v in by.values() if len(v) >= 2]
    diffs = np.array([abs(list(v.values())[0] - list(v.values())[1]) for v in pairs])
    s0 = np.array([list(v.values())[0] for v in pairs])
    s1 = np.array([list(v.values())[1] for v in pairs])
    return {"mean_abs_seed_diff_per_sentence": round(float(diffs.mean()), 4),
            "mean_wer_seed_a": round(float(s0.mean()), 4), "mean_wer_seed_b": round(float(s1.mean()), 4)}


def run_generate(preset: str, out_dir: Path, seeds=DEFAULT_SEEDS, decoder_kind: str = "batched") -> dict:
    return asyncio.run(generate(preset, out_dir, seeds, decoder_kind))
