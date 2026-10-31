"""Shared test setup: tests that need a GPU or vLLM skip cleanly, with a reason, on CPU-only machines.

Mark such a test with @pytest.mark.gpu (needs a CUDA device) or @pytest.mark.vllm (needs vllm installed).
Tests that only need optional libraries should use pytest.importorskip instead.
"""

import importlib.util

import pytest


def _has_cuda() -> bool:
    if importlib.util.find_spec("torch") is None:
        return False
    import torch

    return torch.cuda.is_available()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: test needs a CUDA GPU (skipped when none is available)")
    config.addinivalue_line("markers", "vllm: test needs vllm installed (skipped when it is missing)")


def pytest_collection_modifyitems(config, items):
    cuda = None
    for item in items:
        if "gpu" in item.keywords:
            cuda = _has_cuda() if cuda is None else cuda
            if not cuda:
                item.add_marker(pytest.mark.skip(reason="needs a CUDA GPU (none available on this machine)"))
        if "vllm" in item.keywords and importlib.util.find_spec("vllm") is None:
            item.add_marker(pytest.mark.skip(reason="needs vllm (not installed on this machine)"))
