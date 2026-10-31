"""Send TTS request(s) to the websocket server, save the audio, and compare client vs server TTFA.

  python scripts/ws_client.py --text "Hello there." --out results/samples/step3_ws.wav
  python scripts/ws_client.py --preset-text paragraph
  python scripts/ws_client.py --preset-text sentence -n 3      # 3 clients at once, one connection each
"""

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from websockets.asyncio.client import connect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402
from speechserve.loadtest import TEXTS, ws_request  # noqa: E402


async def one(url, text, voice, idx, out):
    async with connect(url, max_size=None, ping_interval=None) as ws:
        r = await ws_request(ws, text, voice, keep_audio=out is not None)
    s = r["server"]
    if r["status"] != "done":
        print(f"[{idx}] ERROR: {r.get('error')}")
        return r
    print(f"[{idx}] TTFA client {r['client_ttfa_ms']:.0f} ms | server {s['ttfa_ms']:.0f} ms | "
          f"audio {r['audio_s']:.2f} s | RTF client {r['client_rtf']:.3f} server {s['rtf']:.3f} | "
          f"gaps {r['gap_count']} (buffer needed {r['buffer_needed_ms']:.0f} ms) | tokens {s['tokens']} | "
          f"queue {s['queue_ms']:.0f} ms | SNAC {s['snac_ms_mean']:.1f} ms + wait {s['snac_wait_ms_mean']:.1f} ms/chunk")
    if out is not None:
        path = Path(out) if idx == 0 else Path(out).with_stem(f"{Path(out).stem}_{idx}")
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.frombuffer(r["pcm16"], "<i2"), C.SAMPLE_RATE, subtype="PCM_16")
        print(f"[{idx}] saved {path}")
    return r


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/ws/tts")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--text")
    g.add_argument("--preset-text", choices=TEXTS, default="sentence")
    ap.add_argument("--voice", default=C.DEFAULT_VOICE)
    ap.add_argument("-n", type=int, default=1, help="concurrent clients")
    ap.add_argument("--out", help="save WAV here (client i>0 gets a _i suffix)")
    args = ap.parse_args()
    text = args.text or TEXTS[args.preset_text]
    rows = await asyncio.gather(*(one(args.url, text, args.voice, i, args.out) for i in range(args.n)))
    bad = [r for r in rows if r["status"] != "done" or r["gap_count"] > 0]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    asyncio.run(main())
