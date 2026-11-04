from conftest import make_pair, run_parity

from rfproteina.nn.modules.pair_bias_attn import (
    CuEqMultiHeadBiasedAttentionADALN_MM,
    CuEqPairBiasAttention,
    MultiHeadBiasedAttentionADALN_MM,
    PairBiasAttention,
)


def test_pair_bias_attention_qkln():
    torch_mod, jax_mod = make_pair(
        PairBiasAttention, node_dim=16, dim_head=4, heads=4, bias=True, dim_out=16, qkln=True, pair_dim=8
    )
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "node_feats": ("float", (2, 8, 16)),
        "pair_feats": ("float", (2, 8, 8, 8)),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"pair_bias_attention qkln: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_pair_bias_attention_no_qkln_no_pair():
    # pair_dim=None: mask-only additive bias; qkln=False: Identity q/k norms.
    torch_mod, jax_mod = make_pair(
        PairBiasAttention, node_dim=16, dim_head=4, heads=4, bias=True, dim_out=16, qkln=False, pair_dim=None
    )
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "node_feats": ("float", (2, 8, 16)),
        "pair_feats": (lambda key: None),
        "mask": ("bool", (2, 8, 8)),
    })
    print(f"pair_bias_attention mask-only: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_cueq_pair_bias_attention():
    """The cuEq torch module cannot be constructed without the CUDA-only
    cuequivariance package, so no torch reference exists on macOS. The JAX module
    is shared with PairBiasAttention (identical weights/math, verified above);
    here we only verify the converter registration accepts the cuEq class."""
    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites

    register_base_types()
    composites.register()
    assert CuEqPairBiasAttention in from_torch.registry
    assert CuEqMultiHeadBiasedAttentionADALN_MM in from_torch.registry


def test_multihead_biased_attention_adaln_mm():
    torch_mod, jax_mod = make_pair(
        MultiHeadBiasedAttentionADALN_MM, dim_token=16, dim_pair=8, nheads=4, dim_cond=32, use_qkln=True
    )
    mean, std, gmax = run_parity(torch_mod, jax_mod, {
        "x": ("float", (2, 8, 16)),
        "pair_rep": ("float", (2, 8, 8, 8)),
        "cond": ("float", (2, 8, 32)),
        "mask": ("bool", (2, 8)),
    })
    print(f"multihead_biased_attention_adaln_mm: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
