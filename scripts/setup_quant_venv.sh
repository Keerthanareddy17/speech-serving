#!/usr/bin/env bash
# Builds the isolated quantization venv: bash scripts/setup_quant_venv.sh
# Then run quantization with: /content/venvs/quant/bin/python scripts/quantize_w4a16.py
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=/content/venvs/quant

# Colab's python has no ensurepip, so `python -m venv` fails; uv creates venvs without it.
pip install -q uv
[ -x "$VENV/bin/python" ] || uv venv -q --python "$(command -v python3)" "$VENV"

# Colab sets UV_SYSTEM_PYTHON=true, which makes uv install into the system python (vLLM's env!)
# even with VIRTUAL_ENV set. Clear it and target the venv's interpreter explicitly.
uv_install() { env -u UV_SYSTEM_PYTHON uv pip install --python "$VENV/bin/python" -r requirements-quant.txt "$@"; }
# Dry run first: abort before touching anything if uv would target any environment other than the venv.
target=$(uv_install --dry-run 2>&1 | head -1)
echo "$target"
echo "$target" | grep -q "environment at: $VENV" || { echo "uv would not target $VENV, aborting"; exit 1; }
uv_install -q

"$VENV/bin/python" - <<'EOF'
import compressed_tensors, llmcompressor, torch, transformers
assert torch.cuda.is_available(), "venv torch does not see the GPU"
print(f"venv ok: torch {torch.__version__}, transformers {transformers.__version__}, "
      f"compressed-tensors {compressed_tensors.__version__}, llmcompressor {llmcompressor.__version__}")
EOF
