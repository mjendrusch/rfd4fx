from conftest import make_pair, run_parity

import torch.nn as tnn

from rfd4fx.nn.layers import Linear


def test_linear_2d():
    torch_mod, jax_mod = make_pair(tnn.Linear, 16, 32)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (8, 16))})
    print(f"linear 2d: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_linear_3d_bias():
    torch_mod, jax_mod = make_pair(tnn.Linear, 16, 16, bias=True)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"linear 3d: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_linear_no_bias():
    torch_mod, jax_mod = make_pair(tnn.Linear, 16, 16, bias=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"linear no-bias: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
