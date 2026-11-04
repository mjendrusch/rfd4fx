"""Torch, when installed, must be the CPU-only build (never CUDA wheels).

The pyproject pins torch to the PyTorch CPU wheel index under uv; this test
fails loudly if a CUDA-bundled torch ever enters the environment. Skipped when
torch is not installed (torch-free inference environments)."""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed (torch-free inference)")


def test_torch_is_cpu_only():
    assert torch.version.cuda is None, (
        "CUDA-enabled torch detected — rfd4fx environments must use the "
        "CPU-only wheels (pytorch-cpu index / --index-url "
        "https://download.pytorch.org/whl/cpu)")
