from conftest import make_pair, run_parity

import torch.nn as tnn


def test_layer_norm_affine():
    torch_mod, jax_mod = make_pair(tnn.LayerNorm, 16)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"layer_norm affine: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_layer_norm_no_affine():
    # AdaLN conditioning norms use elementwise_affine=False (weight/bias are None).
    torch_mod, jax_mod = make_pair(tnn.LayerNorm, 16, elementwise_affine=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"layer_norm no-affine: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_layer_norm_no_bias():
    torch_mod, jax_mod = make_pair(tnn.LayerNorm, 16, bias=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"layer_norm no-bias: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
