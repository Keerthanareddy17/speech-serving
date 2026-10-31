"""Start the speechserve websocket server with uvicorn (one worker: one engine owns the GPU).

  python scripts/run_server.py [--preset w4a16_lmhead] [--port 8000] [--log /content/logs/server.jsonl]
                               [--decoder per_stream|batched] [--snac-max-batch 64] [--snac-max-wait-ms 0]
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from speechserve import config as C  # noqa: E402
from speechserve.metrics import DEFAULT_LOG  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=C.DEFAULT_PRESET, choices=list(C.PRESETS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--log", default=DEFAULT_LOG, help="per-request JSON lines (keep outside the repo)")
    ap.add_argument("--decoder", default="per_stream", choices=["per_stream", "batched"],
                    help="per_stream: one SNAC call per window (step 4 baseline); batched: one call for all "
                         "streams' pending windows (step 6a)")
    ap.add_argument("--snac-max-batch", type=int, default=64, help="batched decoder: max windows per SNAC call")
    ap.add_argument("--snac-max-wait-ms", type=float, default=0.0,
                    help="batched decoder: how long an idle worker waits to gather more windows")
    args = ap.parse_args()
    os.environ["SPEECHSERVE_PRESET"] = args.preset
    os.environ["SPEECHSERVE_LOG"] = args.log
    os.environ["SPEECHSERVE_DECODER"] = args.decoder
    os.environ["SPEECHSERVE_SNAC_MAX_BATCH"] = str(args.snac_max_batch)
    os.environ["SPEECHSERVE_SNAC_MAX_WAIT_MS"] = str(args.snac_max_wait_ms)

    import uvicorn

    # workers=1: a second worker would load a second engine, which does not fit on the GPU.
    uvicorn.run("speechserve.server:app", host=args.host, port=args.port, workers=1, log_level="info",
                ws_max_size=1 << 20)


if __name__ == "__main__":
    main()
