"""Start the speechserve websocket server with uvicorn (one worker: one engine owns the GPU).

  python scripts/run_server.py [--preset w4a16_lmhead] [--port 8000] [--log /content/logs/server.jsonl]
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
    args = ap.parse_args()
    os.environ["SPEECHSERVE_PRESET"] = args.preset
    os.environ["SPEECHSERVE_LOG"] = args.log

    import uvicorn

    # workers=1: a second worker would load a second engine, which does not fit on the GPU.
    uvicorn.run("speechserve.server:app", host=args.host, port=args.port, workers=1, log_level="info",
                ws_max_size=1 << 20)


if __name__ == "__main__":
    main()
