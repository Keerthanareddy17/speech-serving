"""Step 7: validate the token shards.

  python scripts/validate_shards.py [--shards /content/data/shards/full] [--n 5]

1. Every clip: n_tokens == 7 * n_frames, n_frames matches duration (within one 2048-sample frame), and each
   token id lies in its position's range [128266 + p*4096, 128266 + (p+1)*4096).
2. n random clips: original audio re-read from the parquet files and trimmed with the same VAD config,
   shard tokens decoded back with SNAC (fp32, noise off), compared to the original: waveform SNR (low for a
   ~1 kbps neural codec, which keeps how speech sounds, not the exact waveform) and log-mel SNR.
3. Same clips re-encoded alone with eager (uncompiled) SNAC: token agreement with the shard (effect of
   batching/padding + torch.compile) and that version's reconstruction quality.
4. One clip's original + reconstruction saved to results/samples/pipeline/.
Results are stored under "validation" in results/pipeline.json.
"""

import argparse
import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402
from speechserve.codec import FrameAssembler, SNACDecoder, frames_to_layers, layers_to_frames  # noqa: E402
from speechserve.pipeline.decode import decode_wav  # noqa: E402
from speechserve.pipeline.download import local_files  # noqa: E402
from speechserve.pipeline.vad import VadConfig, trim  # noqa: E402

SR, FRAME = 24_000, 2048


def snr_db(x, ref):
    n = min(len(x), len(ref))
    x, ref = x[:n], ref[:n]
    return float(10 * np.log10(np.sum(ref ** 2) / max(np.sum((x - ref) ** 2), 1e-20)))


def logmel(x):
    import librosa
    m = librosa.feature.melspectrogram(y=x, sr=SR, n_fft=1024, hop_length=256, n_mels=80)
    return np.log(m + 1e-5)


def mel_snr_db(x, ref):
    n = min(len(x), len(ref))
    a, b = logmel(x[:n]), logmel(ref[:n])
    return float(10 * np.log10(np.sum(b ** 2) / max(np.sum((a - b) ** 2), 1e-20)))


def mel_l1(x, ref):
    n = min(len(x), len(ref))
    return float(np.mean(np.abs(logmel(x[:n]) - logmel(ref[:n]))))


def tokens_to_layers(tokens):
    asm = FrameAssembler()
    asm.push([int(t) for t in tokens])
    return asm.frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", default="/content/data/shards/full")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import pyarrow.parquet as pq

    # ---- 1. every clip ----
    files = sorted(glob.glob(f"{args.shards}/shard-*.parquet"))
    offsets = C.AUDIO_TOKEN_BASE + np.arange(7) * C.CODEBOOK_SIZE
    total, bad_count, bad_dur, bad_range, all_ids = 0, [], [], [], []
    for f in files:
        t = pq.read_table(f, columns=["id", "duration_s", "n_frames", "n_tokens", "tokens"])
        for cid, dur, nf, nt, toks in zip(*(t[c].to_pylist() for c in ("id", "duration_s", "n_frames", "n_tokens")),
                                          t["tokens"].to_numpy(zero_copy_only=False)):
            total += 1
            all_ids.append((cid, f))
            toks = np.asarray(toks)
            if nt != 7 * nf or len(toks) != nt:
                bad_count.append(cid)
            if abs(nf * FRAME / SR - dur) > FRAME / SR + 1e-3 or nf != -(-round(dur * SR) // FRAME):
                bad_dur.append(cid)
            pos = np.arange(len(toks)) % 7
            codes = toks - offsets[pos]
            if codes.min() < 0 or codes.max() >= C.CODEBOOK_SIZE:
                bad_range.append(cid)
    per_clip = {"clips_checked": total, "token_count_vs_frames_failures": len(bad_count),
                "frames_vs_duration_failures": len(bad_dur), "token_range_failures": len(bad_range),
                "examples": {"count": bad_count[:5], "duration": bad_dur[:5], "range": bad_range[:5]}}
    print("per-clip checks:", per_clip)

    # ---- 2/3. random clips: reconstruct and compare ----
    rng = np.random.default_rng(args.seed)
    picks = [all_ids[i] for i in rng.choice(len(all_ids), size=args.n, replace=False)]
    import torch
    from snac import SNAC
    dec = SNACDecoder()
    dec.set_noise(False)
    enc_model = SNAC.from_pretrained(C.SNAC_MODEL).eval().cuda()
    vad_cfg = VadConfig()
    clips = []
    for cid, shard in picks:
        row = pq.read_table(shard, filters=[("id", "=", cid)]).to_pylist()[0]
        orig = None
        for pf in local_files():                                    # find the original WAV by id
            tab = pq.read_table(pf, columns=["id", "audio"], filters=[("id", "=", cid)])
            if tab.num_rows:
                orig = trim(decode_wav(tab["audio"][0].as_py()["bytes"])[0], SR, vad_cfg)
                break
        recon, _ = dec.decode(tokens_to_layers(row["tokens"]))
        with torch.inference_mode():                               # eager single-clip encode for comparison
            codes = enc_model.encode(torch.from_numpy(orig).view(1, 1, -1).cuda())
        eager = [c[0].cpu().numpy() for c in codes]
        shard_layers = frames_to_layers(tokens_to_layers(row["tokens"]))
        agree = [float((a == b).mean()) for a, b in zip(shard_layers, eager)]
        recon_eager, _ = dec.decode(layers_to_frames(*eager))
        clips.append({
            "id": cid, "text": row["text"], "duration_s": row["duration_s"], "n_tokens": row["n_tokens"],
            "len_samples_orig": len(orig), "len_samples_recon": len(recon),
            "waveform_snr_db": round(snr_db(recon, orig), 2), "logmel_snr_db": round(mel_snr_db(recon, orig), 2),
            "logmel_l1": round(mel_l1(recon, orig), 3),
            "eager_single_clip": {"token_agreement_L0_L1_L2": [round(a, 4) for a in agree],
                                  "waveform_snr_db": round(snr_db(recon_eager, orig), 2),
                                  "logmel_snr_db": round(mel_snr_db(recon_eager, orig), 2),
                                  "logmel_l1": round(mel_l1(recon_eager, orig), 3)},
            "_orig": orig, "_recon": recon})
        c = clips[-1]
        print(f"{cid}: {c['duration_s']:.2f} s, {c['n_tokens']} tokens | wave SNR {c['waveform_snr_db']} dB, "
              f"log-mel SNR {c['logmel_snr_db']} dB, L1 {c['logmel_l1']} | eager single-clip: agreement {c['eager_single_clip']['token_agreement_L0_L1_L2']}, "
              f"log-mel SNR {c['eager_single_clip']['logmel_snr_db']} dB")

    import soundfile as sf
    out = ROOT / "results/samples/pipeline"
    out.mkdir(parents=True, exist_ok=True)
    ex = max(clips, key=lambda c: c["duration_s"] if c["duration_s"] < 12 else 0)   # a mid-length clip
    sf.write(out / f"{ex['id']}_original.wav", ex["_orig"], SR, subtype="PCM_16")
    sf.write(out / f"{ex['id']}_reconstructed.wav", ex["_recon"], SR, subtype="PCM_16")
    for c in clips:
        c.pop("_orig"), c.pop("_recon")

    results = json.loads((ROOT / "results/pipeline.json").read_text())
    results["validation"] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "shards": args.shards,
        "per_clip_checks": per_clip, "random_clips": clips,
        "summary": {k: round(float(np.mean([c[k] for c in clips])), 2) for k in ("waveform_snr_db", "logmel_snr_db", "logmel_l1")}
                   | {"eager_logmel_snr_db": round(float(np.mean([c["eager_single_clip"]["logmel_snr_db"] for c in clips])), 2),
                      "eager_token_agreement_mean": [round(float(np.mean([c["eager_single_clip"]["token_agreement_L0_L1_L2"][i] for c in clips])), 4) for i in range(3)]},
        "listen": [str((out / f"{ex['id']}_{k}.wav").relative_to(ROOT)) for k in ("original", "reconstructed")],
        "decoder": "SNAC fp32, noise off", "vad": vars(vad_cfg)}
    (ROOT / "results/pipeline.json").write_text(json.dumps(results, indent=2) + "\n")
    print("summary:", results["validation"]["summary"], "| saved", results["validation"]["listen"])



if __name__ == "__main__":
    main()
