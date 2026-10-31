# Realtime TTS server (Orpheus 3B on vLLM, SNAC, websocket).
#
# NOT BUILT OR RUN in the environment where this repo was developed (no Docker daemon, no GPU there).
# Versions mirror the Colab stack in requirements.txt and docs/DECISIONS.md. Build and smoke-test on a
# machine with an NVIDIA driver supporting CUDA 13.0 (R580 or newer) and the NVIDIA Container Toolkit.
#
#   docker build -t speechserve .
#   docker run --gpus all -p 8000:8000 \
#     -e HF_TOKEN=...                                  `# gated tokenizer: orpheus-3b-0.1-pretrained` \
#     -v /path/to/models:/models:ro                    `# contains orpheus-3b-w4a16-lmhead/ (see README)` \
#     -v hf-cache:/hf-cache -v speechserve-logs:/logs \
#     speechserve
#
# No token or secret is baked into the image: HF_TOKEN is read from the environment at run time.

# The tag is the expected CUDA 13.0 runtime image; confirm it exists on Docker Hub before building.
ARG CUDA_IMAGE=nvidia/cuda:13.0.0-runtime-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl build-essential libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

# uv provides Python 3.13 (the version the results were measured on) without touching the system Python.
COPY --from=ghcr.io/astral-sh/uv:0.5.0 /uv /usr/local/bin/uv
RUN uv venv --python 3.13 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH

WORKDIR /app

# Pinned stack: vllm==0.32.0, torch==2.13.0 (cu130), snac==1.2.1, ... (requirements.txt)
COPY requirements.txt ./
RUN uv pip install -r requirements.txt

# cuDNN fix (docs/DECISIONS.md): a stray cu12 cuDNN next to torch's cu13 cuDNN makes SNAC's first conv fail
# with CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Remove cu12 if present and reinstall the cu13 build.
RUN (uv pip uninstall nvidia-cudnn-cu12 || true) \
 && uv pip install --reinstall --no-deps nvidia-cudnn-cu13==9.20.0.48

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY scripts ./scripts
RUN uv pip install -e .

# Paths and settings come from the environment.
ENV SPEECHSERVE_MODELS_DIR=/models \
    HF_HOME=/hf-cache \
    PRESET=w4a16_lmhead \
    PORT=8000
RUN useradd --create-home app && mkdir -p /models /hf-cache /logs && chown -R app /hf-cache /logs
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Single uvicorn worker by design: one worker owns the GPU and the vLLM engine.
CMD ["sh", "-c", "python scripts/run_server.py --host 0.0.0.0 --port ${PORT} --preset ${PRESET} --log /logs/server.jsonl"]
