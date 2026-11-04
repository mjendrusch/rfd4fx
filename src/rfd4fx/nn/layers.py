"""Leaf modules: Linear, LayerNorm, RMSNorm, Embedding, Sequential, SwiGLU,
FourierEmbedding. Field names match the torch modules exactly."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from einops import einsum
from jaxtyping import Array, Float

from ..backend import AbstractFromTorch, from_torch, identity, silu, sigmoid


class Linear(AbstractFromTorch):
    """torch nn.Linear semantics: weight is (Out, In), out = x @ W.T + b."""

    weight: Float[Array, "Out In"]
    bias: Float[Array, "Out"] | None = None

    def __call__(self, x):
        o = einsum(x, self.weight, "... In, Out In -> ... Out")
        if self.bias is not None:
            o = o + self.bias
        return o


class LayerNorm(AbstractFromTorch):
    """torch nn.LayerNorm over the last axis; biased variance, optional affine.

    ``weight``/``bias`` are None when the torch layer was built with
    ``elementwise_affine=False`` (AdaLN conditioning norms) or ``bias=False``.
    """

    weight: Float[Array, "D"] | None = None
    bias: Float[Array, "D"] | None = None
    eps: float = 1e-5

    def __call__(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        # torch layer_norm uses the *biased* variance (mean of squared deviations).
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        x = (x - mean) * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x


class RMSNorm(AbstractFromTorch):
    """torch nn.RMSNorm / F.rms_norm semantics: x * rsqrt(mean(x^2) + eps) * weight."""

    weight: Float[Array, "D"] | None = None
    eps: float = 1e-5

    def __call__(self, x):
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(var + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x


class Embedding(AbstractFromTorch):
    weight: Float[Array, "Vocab D"]

    def __call__(self, indices):
        return self.weight[indices]


class Sequential(eqx.Module):
    """Equivalent of nn.Sequential; children kept in a string-keyed dict exactly
    like torch's ``_modules`` OrderedDict ("0", "1", ...)."""

    _modules: dict[str, eqx.Module]

    def __call__(self, x):
        for i in range(len(self._modules)):
            x = self._modules[str(i)](x)
        return x


class SwiGLU(eqx.Module):
    """rfproteina nn/modules/swiglu.py: split in half, silu(gates) * x. Stateless."""

    def __call__(self, x):
        x, gates = jnp.split(x, 2, axis=-1)
        return jax.nn.silu(gates) * x


# torch's blocks.py computes pi as float32 `2 * acos(0)` (= 3.14159274...), not the
# float64 pi. Match it exactly so the cosine arguments are bit-identical.
_TORCH_PI = float(jnp.float32(2.0) * jnp.acos(jnp.float32(0.0)))


class FourierEmbedding(AbstractFromTorch):
    """rfproteina nn/modules/blocks.py. Random-feature time embedding; ``w``/``b``
    arrive through ``named_buffers`` (persistent buffers, in the checkpoint)."""

    w: Float[Array, "C"]
    b: Float[Array, "C"]
    c: int = 0

    def __call__(self, t):
        return jnp.cos(2 * _TORCH_PI * (t[..., None] * self.w + self.b))


def _convert_linear(m):
    return Linear(
        weight=from_torch(m.weight),
        bias=from_torch(m.bias) if m.bias is not None else None,
    )


def _convert_layer_norm(m):
    return LayerNorm(
        weight=from_torch(m.weight) if m.weight is not None else None,
        bias=from_torch(m.bias) if getattr(m, "bias", None) is not None else None,
        eps=m.eps,
    )


def _convert_rms_norm(m):
    # torch RMSNorm stores eps=None when unset; CompiledRMSNorm.forward then uses
    # finfo(x.dtype).eps. Our inference path is fp32, so resolve that now.
    eps = m.eps if m.eps is not None else float(jnp.finfo(jnp.float32).eps)
    weight = getattr(m, "weight", None)
    return RMSNorm(
        weight=from_torch(weight) if weight is not None else None,
        eps=float(eps),
    )


def _convert_embedding(m):
    return Embedding(weight=from_torch(m.weight))


def _convert_sequential(m):
    return Sequential(_modules={name: from_torch(child) for name, child in m.named_children()})


def _convert_fourier(m):
    return FourierEmbedding(w=from_torch(m.w), b=from_torch(m.b), c=int(m.c))


def register():
    """Register converters for torch primitives. Call after register_base_types()."""
    import torch.nn as tnn

    from_torch.register(tnn.Linear, _convert_linear)
    from_torch.register(tnn.LayerNorm, _convert_layer_norm)
    if hasattr(tnn, "RMSNorm"):
        from_torch.register(tnn.RMSNorm, _convert_rms_norm)
    from_torch.register(tnn.Embedding, _convert_embedding)
    from_torch.register(tnn.Sequential, _convert_sequential)
    from_torch.register(tnn.Identity, lambda _: identity)
    from_torch.register(tnn.SiLU, lambda _: silu)
    from_torch.register(tnn.Sigmoid, lambda _: sigmoid)

    # rfproteina-specific stateless leaves
    from rfproteina.nn.modules.swiglu import SwiGLU as TorchSwiGLU

    from_torch.register(TorchSwiGLU, lambda _: SwiGLU())

    from rfproteina.nn.modules.blocks import FourierEmbedding as TorchFourier

    from_torch.register(TorchFourier, _convert_fourier)

    # CompiledRMSNorm subclasses nn.RMSNorm, so the tnn.RMSNorm registration above
    # dispatches to it as well.
    try:
        from rfproteina.nn.modules.norm import CompiledRMSNorm

        from_torch.register(CompiledRMSNorm, _convert_rms_norm)
    except ImportError:
        pass
