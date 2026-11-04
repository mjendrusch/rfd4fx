from conftest import make_pair, run_parity

from rfproteina.nn.modules.openfold_modules.pair_transition import PairTransition
from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def test_triangle_multiplication_outgoing():
    torch_mod, jax_mod = make_pair(TriangleMultiplicationOutgoing, c_z=16, c_hidden=32, bias=True)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "z": ("float", (2, 8, 8, 16)),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"triangle_outgoing: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_triangle_multiplication_incoming():
    torch_mod, jax_mod = make_pair(TriangleMultiplicationIncoming, c_z=16, c_hidden=32, bias=True)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "z": ("float", (2, 8, 8, 16)),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"triangle_incoming: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_triangle_multiplication_no_bias():
    torch_mod, jax_mod = make_pair(TriangleMultiplicationOutgoing, c_z=16, c_hidden=32, bias=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "z": ("float", (2, 8, 8, 16)),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"triangle_outgoing no-bias: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_pair_transition():
    torch_mod, jax_mod = make_pair(PairTransition, c_z=16, n=4)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "z": ("float", (2, 8, 8, 16)),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"pair_transition: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_pair_transition_no_mask():
    torch_mod, jax_mod = make_pair(PairTransition, c_z=16, n=4)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "z": ("float", (2, 8, 8, 16)),
        "mask": (lambda key: None),
    })
    print(f"pair_transition no-mask: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
