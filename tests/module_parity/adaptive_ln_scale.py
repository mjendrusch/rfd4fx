from conftest import make_pair, run_parity

from rfproteina.nn.modules.adaptive_ln_scale import AdaptiveLayerNorm, AdaptiveOutputScale


def test_adaptive_layer_norm_seq_cond():
    torch_mod, jax_mod = make_pair(AdaptiveLayerNorm, dim=16, dim_cond=32)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "cond": ("float", (2, 8, 32)),
        "mask": ("bool", (2, 8)),
    })
    print(f"adaptive_layer_norm seq cond: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_adaptive_layer_norm_broadcast_cond():
    # torch's cond.ndim == 2 special case: gamma/beta broadcast across spatial axes.
    torch_mod, jax_mod = make_pair(AdaptiveLayerNorm, dim=16, dim_cond=32)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "cond": ("float", (2, 32)),
        "mask": ("bool", (2, 8)),
    })
    print(f"adaptive_layer_norm broadcast cond: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_adaptive_output_scale():
    torch_mod, jax_mod = make_pair(AdaptiveOutputScale, dim=16, dim_cond=32)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "cond": ("float", (2, 8, 32)),
        "mask": ("bool", (2, 8)),
    })
    print(f"adaptive_output_scale: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
