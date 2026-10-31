"""CPU-only checks for speechserve.config (no torch or vllm imported)."""

import importlib

import pytest


def test_models_dir_can_be_set_from_env(monkeypatch):
    from speechserve import config as C

    monkeypatch.setenv("SPEECHSERVE_MODELS_DIR", "/models")
    try:
        importlib.reload(C)
        assert C.PRESETS["w4a16_lmhead"]["model"] == "/models/orpheus-3b-w4a16-lmhead"
        assert C.PRESETS["w4a16"]["model"] == "/models/orpheus-3b-w4a16"
        assert C.engine_args()["model"] == "/models/orpheus-3b-w4a16-lmhead"
    finally:
        monkeypatch.undo()
        importlib.reload(C)


def test_default_models_dir_is_the_colab_drive_path(monkeypatch):
    from speechserve import config as C

    monkeypatch.delenv("SPEECHSERVE_MODELS_DIR", raising=False)
    try:
        importlib.reload(C)
        assert C.MODELS_DIR == "/content/drive/MyDrive/speech-infra/models"
    finally:
        importlib.reload(C)


@pytest.mark.gpu
def test_gpu_marker_skips_without_cuda():
    """Runs only on a machine with a CUDA device; elsewhere it is reported as skipped with a reason."""
    import torch

    assert torch.cuda.is_available()
