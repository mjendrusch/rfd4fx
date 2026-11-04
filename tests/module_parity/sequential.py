from conftest import make_pair, run_parity

import torch.nn as tnn

from rfproteina.nn.modules.swiglu import SwiGLU


def test_sequential_with_activations():
    # Sequential(Linear, SiLU, Sigmoid, Linear) exercises the string-keyed child
    # dict and the activation-wrapper singletons.
    torch_mod, jax_mod = make_pair(
        tnn.Sequential, tnn.Linear(16, 32), tnn.SiLU(), tnn.Sigmoid(), tnn.Linear(32, 16)
    )
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 16))})
    print(f"sequential: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_swiglu():
    torch_mod, jax_mod = make_pair(SwiGLU)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"x": ("float", (2, 8, 32))})
    print(f"swiglu: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
