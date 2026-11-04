"""Composite modules: AdaLN/AdaLN-zero, transitions, pair-bias attention,
triangle multiplicative updates. Field names match the torch modules exactly."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import rearrange
from jaxtyping import Array, Float

from ..backend import AbstractFromTorch, from_torch
from .layers import LayerNorm, Linear, RMSNorm, Sequential

# rfproteina pair_bias_attn.py: "not -inf to avoid gradient issues"
NEG_INF = -1e4


# --- Adaptive normalization (nn/modules/adaptive_ln_scale.py) ------------------


class AdaptiveLayerNorm(AbstractFromTorch):
    """Conditioned norm: norm(x) * sigmoid(W_c norm(cond)) + V_c norm(cond), then
    multiply by the spatial mask. torch's broadcast special case (cond.ndim == 2,
    x.ndim > 2) is static-shape logic: ndim is compile-time under jit."""

    norm: eqx.Module  # LayerNorm or RMSNorm, elementwise_affine=False
    norm_cond: eqx.Module
    to_gamma: Sequential
    to_beta: Linear
    preserve_dtype: bool = False

    def __call__(self, x, cond, mask):
        normed = self.norm(x)
        if self.preserve_dtype:
            normed = normed.astype(x.dtype)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        if cond.ndim == 2 and x.ndim > 2:
            # Broadcast [b, d] conditioning across all spatial axes.
            broadcast_shape = cond.shape[:1] + (1,) * (x.ndim - 2) + (x.shape[-1],)
            gamma = gamma.reshape(broadcast_shape)
            beta = beta.reshape(broadcast_shape)
        out = normed * gamma + beta
        return out * mask[..., None]


class AdaptiveOutputScale(AbstractFromTorch):
    """AdaLN-zero output scale: x * sigmoid(W_c cond) * mask. The torch Linear is
    zero-initialized with bias -2.0 at init; init is irrelevant post-conversion."""

    to_adaln_zero_gamma: Sequential

    def __call__(self, x, cond, mask):
        gamma = self.to_adaln_zero_gamma(cond)
        return x * gamma * mask[..., None]


# --- Transitions (nn/modules/seq_transition_af3.py) ----------------------------


class Transition(AbstractFromTorch):
    """SwiGLU transition: out_linear(swiglu(x)) * mask. ``ln`` exists only when the
    torch module was built with layer_norm=True (optional field covers both)."""

    swish_linear: Sequential
    linear_out: Linear
    ln: LayerNorm | None = None
    use_layer_norm: bool = False

    def __call__(self, x, mask):
        if self.ln is not None:
            x = self.ln(x)
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]


class TransitionADALN(AbstractFromTorch):
    adaln: AdaptiveLayerNorm
    transition: Transition
    scale_output: AdaptiveOutputScale

    def __call__(self, x, cond, mask):
        x = self.adaln(x, cond, mask)
        x = self.transition(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


# --- Pair-bias attention (nn/modules/pair_bias_attn.py) ------------------------


class PairBiasAttention(AbstractFromTorch):
    """Gated multi-head self-attention with additive pair bias + mask.

    One JAX implementation covers both torch backends (PairBiasAttention and
    CuEqPairBiasAttention): the cuEq fused kernel is mathematically identical and
    takes exactly these weights. Torch's (b, h, n, d) layout is kept and attention
    is computed with einsum — the same math as torch's CPU fallback path, so parity
    is exact without dot_product_attention axis swaps.
    """

    node_norm: eqx.Module
    to_qkv: Linear
    to_g: Linear
    q_layer_norm: eqx.Module  # LayerNorm/RMSNorm or identity wrapper
    k_layer_norm: eqx.Module
    to_out_node: Linear
    node_dim: int = 0
    pair_dim: int | None = None
    heads: int = 0
    dim_head: int = 0
    to_bias: Linear | None = None
    pair_norm: LayerNorm | None = None

    def __call__(self, node_feats, pair_feats, mask):
        h = self.heads

        node_feats = self.node_norm(node_feats)

        # torch chunk(3, dim=-1) == one split into three equal parts.
        q, k, v = jnp.split(self.to_qkv(node_feats), 3, axis=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)
        # Gate is computed from the *normalized* node features (torch reassigns
        # node_feats before to_g) — a classic gate-source bug if ported wrong.
        g = self.to_g(node_feats)

        # [b, n, (h d)] -> [b, h, n, d] (torch attention layout)
        q, k, v, g = (rearrange(t, "b n (h d) -> b h n d", h=h) for t in (q, k, v, g))

        attn_bias = None
        if pair_feats is not None and self.to_bias is not None:
            # [b, i, j, pair_dim] -> [b, h, i, j]
            attn_bias = rearrange(self.to_bias(self.pair_norm(pair_feats)), "b i j h -> b h i j")

        if mask is not None:
            # mask: [b, i, j] boolean; torch masked_fill(~mask, NEG_INF)
            mask_e = mask[:, None]
            if attn_bias is not None:
                attn_bias = jnp.where(mask_e, attn_bias, NEG_INF)
            else:
                attn_bias = jnp.where(mask_e, 0.0, NEG_INF)
            # Fully masked rows: torch resets bias to 0 so softmax is uniform
            # instead of NaN. jnp.where over the keepdim any() broadcasts cleanly.
            row_any = mask_e.any(axis=-1, keepdims=True)
            attn_bias = jnp.where(row_any, attn_bias, 0.0)

        scale = self.dim_head**-0.5
        sim = jnp.einsum("bhid,bhjd->bhij", q, k) * scale
        if attn_bias is not None:
            sim = sim + attn_bias
        attn = jax.nn.softmax(sim, axis=-1)
        o = jnp.einsum("bhij,bhjd->bhid", attn, v)

        o_gated = jax.nn.sigmoid(g) * o
        out = rearrange(o_gated, "b h n d -> b n (h d)", h=h)
        return self.to_out_node(out)


class MultiHeadBiasedAttentionADALN_MM(AbstractFromTorch):
    """AdaLN in -> pair-bias attention -> AdaLN-zero out, with mask multiplies."""

    adaln: AdaptiveLayerNorm
    mha: PairBiasAttention
    scale_output: AdaptiveOutputScale

    def __call__(self, x, pair_rep, cond, mask):
        # torch: mask[:, :, None] * mask[:, None, :] on a 0/1 int/float mask —
        # multiplication here (not logical and) matters only for non-binary masks.
        pair_mask = mask[:, :, None] * mask[:, None, :]
        x = self.adaln(x, cond, mask)
        x = self.mha(x, pair_rep, pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


# --- Triangle multiplication (openfold_modules/triangular_multiplicative_update.py)


class TriangleMultiplication(AbstractFromTorch):
    """AF3 algorithm 11/12, vanilla (non-cuEq) path.

    The torch contraction is a batched matmul over permuted dims; expressed as a
    single einsum over the shared k axis:
      outgoing: p[...,i,j,c] = sum_k a[...,i,k,c] * b[...,j,k,c]
      incoming: p[...,i,j,c] = sum_k a[...,k,i,c] * b[...,k,j,c]
    (a/b spatial axes are k-first in the incoming direction — that permutation is
    why the two directions have different einsum indices.)

    The torch no-grad "grouped GEMM" optimization in _gated_projections
    (F.linear on concatenated weights) is mathematically identical to separate
    projections; the plain form is used here.
    """

    linear_a_p: Linear
    linear_a_g: Linear
    linear_b_p: Linear
    linear_b_g: Linear
    linear_g: Linear
    linear_z: Linear
    layer_norm_in: LayerNorm
    layer_norm_out: LayerNorm
    sigmoid: eqx.Module
    c_z: int = 0
    c_hidden: int = 0
    direction: str = "outgoing"

    @classmethod
    def from_torch(cls, model):
        # _direction is a class attribute on the torch subclasses.
        mod = super().from_torch(model)
        return eqx.tree_at(lambda m: m.direction, mod, str(model._direction))

    def __call__(self, z, mask=None):
        if mask is None:
            mask = jnp.ones(z.shape[:-1], dtype=z.dtype)

        mask = mask[..., None]

        z = self.layer_norm_in(z)
        a = self.linear_a_p(z) * self.sigmoid(self.linear_a_g(z)) * mask
        b = self.linear_b_p(z) * self.sigmoid(self.linear_b_g(z)) * mask

        if self.direction == "outgoing":
            x = jnp.einsum("...ikc,...jkc->...ijc", a, b)
        else:
            x = jnp.einsum("...kic,...kjc->...ijc", a, b)

        x = self.linear_z(self.layer_norm_out(x))
        g = self.sigmoid(self.linear_g(z))
        return x * g


# --- Pair transition (openfold_modules/pair_transition.py) ---------------------


class PairTransition(AbstractFromTorch):
    """ReLU MLP over pair channels with mask; the torch chunk_size path is unused
    by the trunk (chunk_size=None) and is dropped."""

    layer_norm: LayerNorm
    linear_1: Linear
    relu: eqx.Module
    linear_2: Linear
    c_z: int = 0
    n: int = 0

    def __call__(self, z, mask=None):
        if mask is None:
            mask = jnp.ones(z.shape[:-1], dtype=z.dtype)
        z = self.layer_norm(z)
        z = self.linear_2(self.relu(self.linear_1(z)))
        return z * mask[..., None]


def register():
    from rfproteina.nn.modules.adaptive_ln_scale import (
        AdaptiveLayerNorm as TorchAdaptiveLayerNorm,
        AdaptiveOutputScale as TorchAdaptiveOutputScale,
    )
    from rfproteina.nn.modules.pair_bias_attn import (
        CuEqMultiHeadBiasedAttentionADALN_MM,
        CuEqPairBiasAttention,
        MultiHeadBiasedAttentionADALN_MM as TorchMHAADALN,
        PairBiasAttention as TorchPairBiasAttention,
    )
    from rfproteina.nn.modules.seq_transition_af3 import (
        Transition as TorchTransition,
        TransitionADALN as TorchTransitionADALN,
    )
    from rfproteina.nn.modules.openfold_modules.primitives import LayerNorm as OFLayerNorm
    from rfproteina.nn.modules.openfold_modules.pair_transition import PairTransition as TPairTransition
    from rfproteina.nn.modules.openfold_modules.triangular_multiplicative_update import (
        TriangleMultiplicationIncoming,
        TriangleMultiplicationOutgoing,
    )

    from_torch.register(TorchAdaptiveLayerNorm, AdaptiveLayerNorm.from_torch)
    from_torch.register(TorchAdaptiveOutputScale, AdaptiveOutputScale.from_torch)
    from_torch.register(TorchTransition, Transition.from_torch)
    from_torch.register(TorchTransitionADALN, TransitionADALN.from_torch)
    from_torch.register(TorchPairBiasAttention, PairBiasAttention.from_torch)
    # cuEq variant has identical weights and math -> same JAX module.
    from_torch.register(CuEqPairBiasAttention, PairBiasAttention.from_torch)
    from_torch.register(TorchMHAADALN, MultiHeadBiasedAttentionADALN_MM.from_torch)
    from_torch.register(CuEqMultiHeadBiasedAttentionADALN_MM, MultiHeadBiasedAttentionADALN_MM.from_torch)

    # openfold primitives that are *not* torch.nn subclasses need explicit entries
    # (its Linear is an nn.Linear subclass and dispatches via the base converter).
    from_torch.register(OFLayerNorm, lambda m: LayerNorm(
        weight=from_torch(m.weight),
        bias=from_torch(m.bias) if m.bias is not None else None,
        eps=m.eps,
    ))
    from_torch.register(TriangleMultiplicationOutgoing, TriangleMultiplication.from_torch)
    from_torch.register(TriangleMultiplicationIncoming, TriangleMultiplication.from_torch)
    from_torch.register(TPairTransition, PairTransition.from_torch)
