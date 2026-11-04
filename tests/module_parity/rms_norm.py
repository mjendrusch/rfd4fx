from conftest import make_pair, run_parity

import torch.nn as tnn


def test_rms_norm_affine():
    # eps=1e-5 is what rfproteina's make_norm passes to RMSNorm variants.
    torch_mod, jax_mod = make_pair(tnn.RMSNorm, 16, eps=1e-5)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"rms_norm affine: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_rms_norm_no_affine():
    torch_mod, jax_mod = make_pair(tnn.RMSNorm, 16, eps=1e-5, elementwise_affine=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"rms_norm no-affine: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
