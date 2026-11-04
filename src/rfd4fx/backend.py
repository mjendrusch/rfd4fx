"""PyTorch -> Equinox conversion infrastructure.

``from_torch`` is a ``singledispatch`` over torch module types. Importing this
module requires torch; it is never used at inference time."""

from __future__ import annotations

import dataclasses
from functools import singledispatch
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp


@singledispatch
def from_torch(obj):
    raise TypeError(f"No from_torch converter registered for {type(obj)}")


def register_base_types():
    """Register Python/torch primitives. Call once, when torch is imported."""
    import torch

    @from_torch.register(torch.Tensor)
    def _tensor(t):
        t = t.detach()
        if t.dtype == torch.bfloat16:
            # Convert bf16 weights to fp32 (esmjfold2 convention; parity work is fp32).
            t = t.to(torch.float32)
        return jnp.asarray(t.cpu().numpy())

    from_torch.register(int, lambda x: x)
    from_torch.register(float, lambda x: x)
    from_torch.register(bool, lambda x: x)
    from_torch.register(str, lambda x: x)
    from_torch.register(type(None), lambda x: None)
    from_torch.register(tuple, lambda t: tuple(from_torch(v) for v in t))
    from_torch.register(list, lambda lst: [from_torch(v) for v in lst])
    from_torch.register(dict, lambda d: {k: from_torch(v) for k, v in d.items()})

    @from_torch.register(torch.nn.ModuleList)
    def _modulelist(m):
        return [from_torch(child) for child in m]

    # Activation modules -> prebuilt eqx wrapper singletons (pytree-safe).
    from_torch.register(torch.nn.Identity, lambda _: identity)
    from_torch.register(torch.nn.ReLU, lambda _: relu)
    from_torch.register(torch.nn.SiLU, lambda _: silu)
    from_torch.register(torch.nn.GELU, lambda _: gelu)  # nn.GELU default = exact erf form
    from_torch.register(torch.nn.Sigmoid, lambda _: sigmoid)


# --- Activation modules -------------------------------------------------------
# Bare jnp functions cannot be pytree leaves (they fail serialization), so every
# activation that can appear inside an nn.Sequential must be an eqx.Module.


class _Identity(eqx.Module):
    def __call__(self, x):
        return x


class _ReLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.relu(x)


class _SiLU(eqx.Module):
    def __call__(self, x):
        return jax.nn.silu(x)


class _GELU(eqx.Module):
    def __call__(self, x):
        # nn.GELU default is the exact erf form.
        return jax.nn.gelu(x, approximate=False)


class _Sigmoid(eqx.Module):
    def __call__(self, x):
        return jax.nn.sigmoid(x)


identity = _Identity()
relu = _ReLU()
silu = _SiLU()
gelu = _GELU()
sigmoid = _Sigmoid()


class AbstractFromTorch(eqx.Module):
    """Mixin: auto-implement ``from_torch`` by matching eqx dataclass fields against
    the torch module's ``named_children``, ``named_parameters(recurse=False)`` and
    ``named_buffers(recurse=False)``.

    Strict (per the pytorch-to-equinox skill): a torch child or direct parameter
    with no matching field raises — this surfaces checkpoint/architecture contract
    mismatches at conversion time instead of silently dropping weights.
    """

    @classmethod
    def from_torch(cls, model):
        field_to_field = {f.name: f for f in dataclasses.fields(cls)}
        kwargs: dict[str, Any] = {}

        for name, child in model.named_children():
            if name not in field_to_field:
                raise ValueError(
                    f"{cls.__name__}: torch module has child {name!r} with no matching "
                    f"eqx field. Fields: {sorted(field_to_field)}"
                )
            kwargs[name] = from_torch(child)

        for name, param in model.named_parameters(recurse=False):
            if name not in field_to_field:
                raise ValueError(
                    f"{cls.__name__}: torch module has parameter {name!r} with no "
                    f"matching eqx field. Fields: {sorted(field_to_field)}"
                )
            kwargs[name] = from_torch(param)

        for name, buf in model.named_buffers(recurse=False):
            if name in field_to_field:
                kwargs[name] = from_torch(buf)

        # Remaining fields: plain torch attributes (config scalars such as
        # ``heads``/``dim``/``preserve_dtype``) take priority over dataclass defaults.
        for name, field in field_to_field.items():
            if name in kwargs:
                continue
            if hasattr(model, name):
                attr = getattr(model, name)
                # Take scalars and non-callable state; never grab bound methods.
                if not callable(attr) or isinstance(attr, (int, float, bool, str)):
                    kwargs[name] = from_torch(attr)
                    continue
            if field.default is not dataclasses.MISSING:
                kwargs[name] = field.default
            elif field.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[name] = field.default_factory()
            elif _is_optional(field.type):
                kwargs[name] = None
            else:
                raise ValueError(
                    f"{cls.__name__}: field {name!r} is required but missing from "
                    f"torch module {type(model).__name__}"
                )

        return cls(**kwargs)


def _is_optional(field_type) -> bool:
    """Best-effort 'None is allowed for this annotation' check."""
    s = str(field_type)
    return "None" in s or "Optional" in s or "| None" in s


class TestModule:
    """Dev helper: run a torch module and its JAX conversion side by side and
    report max abs error + timing. Not a test fixture — an interactive tool."""

    def __init__(self, torch_module, jax_module=None):
        import torch

        self.torch_module = torch_module.eval()
        if jax_module is None:
            register_base_types()
            jax_module = from_torch(torch_module)
        self.jax_module = jax_module

    def __call__(self, *inputs):
        import numpy as np
        import torch

        torch_inputs = tuple(
            torch.from_numpy(np.asarray(x)) if isinstance(x, (jax.Array, np.ndarray)) else x
            for x in inputs
        )
        with torch.no_grad():
            expected = self.torch_module(*torch_inputs)
        actual = eqx.filter_jit(self.jax_module)(*inputs)
        expected = expected.numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
        err = np.abs(np.asarray(actual) - expected).max()
        return actual, err
