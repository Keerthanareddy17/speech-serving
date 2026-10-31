#!/usr/bin/env bash
# One-command setup for a fresh Colab runtime: bash scripts/setup_colab.sh
# Assumes the launcher notebook already ran (Drive mounted, HF login, repo cloned).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== installing requirements (vllm brings its matching torch)"
pip install -q -r requirements.txt
pip install -q -e .

echo "== cuDNN fix: remove Colab's cu12 cuDNN, reinstall torch's cu13 cuDNN (see docs/DECISIONS.md)"
pip uninstall -y -q nvidia-cudnn-cu12 || true
pip install -q --force-reinstall --no-deps nvidia-cudnn-cu13==9.20.0.48

echo "== import check"
python - <<'EOF'
from importlib.metadata import version

import torch
import vllm
from snac import SNAC

assert torch.cuda.is_available(), "torch does not see the GPU"
print(f"gpu   : {torch.cuda.get_device_name(0)}")
print(f"torch : {torch.__version__} (cuda {torch.version.cuda}, cudnn {torch.backends.cudnn.version()})")
print(f"vllm  : {vllm.__version__}")
print(f"snac  : {version('snac')}")

# A real conv on the GPU: a cuDNN version mismatch only shows up here, not at import.
m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
codes = [torch.randint(0, 4096, (1, n), device="cuda") for n in (4, 8, 16)]
with torch.inference_mode():
    m.decode(codes)
print("snac GPU decode ok")
EOF
echo "== setup done"
