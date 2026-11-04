from conftest import make_pair, run_parity

from rfproteina.nn.modules.seq_transition_af3 import Transition, TransitionADALN


def test_transition_adaln():
    torch_mod, jax_mod = make_pair(TransitionADALN, dim=16, dim_cond=32)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "cond": ("float", (2, 8, 32)),
        "mask": ("bool", (2, 8)),
    })
    print(f"transition_adaln: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_transition_with_layer_norm():
    torch_mod, jax_mod = make_pair(Transition, dim=16, layer_norm=True)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "mask": ("bool", (2, 8)),
    })
    print(f"transition +ln: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_transition_no_layer_norm():
    torch_mod, jax_mod = make_pair(Transition, dim=16, layer_norm=False)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "mask": ("bool", (2, 8)),
    })
    print(f"transition: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
