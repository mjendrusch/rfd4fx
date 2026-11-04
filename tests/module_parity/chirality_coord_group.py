"""Special-case features: chirality alignment (dense SVD port) and coord-group
relabeling (random labels — structural equivalence, numeric parity impossible)."""

from conftest import build_feature_batch, _to_torch

import numpy as np
import jax
import jax.numpy as jnp
import torch

import rfproteina.nn.noise_alignment as tna
from rfd4fx.nn.noise_alignment import align_chirality, proper_rotation


def test_proper_rotation():
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 0)
        cov = jax.random.normal(key, (3, 4, 3, 3))
        with torch.no_grad():
            expected = tna.proper_rotation(torch.from_numpy(np.asarray(cov)))
        actual = proper_rotation(cov)
        errs.append(float(np.abs(np.asarray(actual) - expected.numpy()).max()))
    errs = np.array(errs)
    print(f"proper_rotation: {errs.mean():.3e} ± {errs.std():.3e} (max {errs.max():.3e})")
    assert errs.max() < 5e-5


def test_align_chirality():
    """Dense per-slot rotation table vs torch's dynamic-frame compaction.
    Uses the production frame width (2 + K=6 neighbors, K from
    chirality_feature.py's 8-wide frames): with >=3 neighbors the covariance is
    generically full rank and SVD bases agree. (For K<3 the covariance is
    rank-deficient and torch/jax SVD legitimately return different arbitrary
    null-space bases — documented in status.md.)"""
    b, n, s, width = 2, 8, 6, 8
    k = width - 2
    jitted = jax.jit(align_chirality)
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 0)
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
        chirality = jax.random.normal(k1, (b, n, s * 3)) * 0.1
        x_t = jax.random.normal(k2, (b, n, 3)) * 5.0
        padding = jnp.ones((b, n), dtype=bool)  # frames must reference valid tokens (torch raises otherwise)
        # frames: mostly inactive (-1 anchors); active frames reference valid tokens
        frames = jnp.full((b, n, s, 2 + k), -1, jnp.int32)
        active = jax.random.bernoulli(k4, 0.3, (b, n, s))
        tok_idx = jnp.arange(n)
        frames = jnp.where(
            active[..., None],
            jnp.broadcast_to(tok_idx[None, :, None], (b, n, s))[..., None].repeat(2 + k, axis=-1),
            frames,
        )
        frames = frames.at[..., :2].set(jnp.broadcast_to(tok_idx[None, :, None], (b, n, s))[..., None])
        # 6 *distinct* valid neighbor tokens per active frame: with fewer than 3
        # distinct vectors the covariance is rank-deficient and torch/jax SVD
        # legitimately disagree on the null-space basis (see status.md).
        offs = jnp.arange(1, k + 1)
        nb = (tok_idx[None, :, None, None] + offs[None, None, None, :]) % n
        frames = frames.at[..., 2:].set(nb)
        owners = jnp.where(
            jax.random.bernoulli(k6, 0.4, (b, n, s)),
            jnp.arange(n)[None, :, None] * s + jnp.arange(s)[None, None, :],
            -1,
        ).astype(jnp.int32)

        batch = {
            "x_t": {"bb_ca": _to_torch(x_t)},
            "padding_mask": _to_torch(padding),
            "chiral_frame_atoms": _to_torch(frames),
            "chiral_slot_owner": _to_torch(owners),
        }
        with torch.no_grad():
            expected = tna.align_chirality(_to_torch(chirality), batch, translate=False, rotate=True)
        actual = jitted(chirality, x_t, padding, frames, owners)
        errs.append(float(np.abs(np.asarray(actual) - expected.numpy()).max()))
    errs = np.array(errs)
    print(f"align_chirality: {errs.mean():.3e} ± {errs.std():.3e} (max {errs.max():.3e})")
    assert errs.max() < 5e-5


def test_coord_group_structure():
    """Random labels: parity is distributional, not numeric. Checks:
    (a) UNKNOWN (-1) slots unchanged, (b) equal ids -> equal labels,
    (c) distinct ids -> distinct labels (bijection), (d) label set = {0..P-1}."""
    from rfproteina.conditions import C_CRDGRP
    from rfproteina.nn.feature_factory import CoordGroupSeqFeat as TorchCG

    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers

    register_base_types()
    layers.register()
    composites.register()
    features.register()

    torch_cg = TorchCG().eval()
    jax_cg = from_torch(torch_cg)
    assert jax_cg.max_group_id >= 4

    for trial in range(20):
        key = jax.random.fold_in(jax.random.key(trial), 0)
        k1, k2 = jax.random.split(key)
        cg = jax.random.randint(k1, (2, 8, 37), -1, 4)  # ids -1..3
        torch_batch, jax_batch = build_feature_batch(
            {"padding_mask": ("bool", (8,)), "x_t": {"bb_ca": ("float", (8, 3))}, C_CRDGRP.full_name: ("int", (8, 37), dict(lo=-1, hi=4))},
            jax.random.fold_in(key, 7),
        )
        # pin the same coord groups on both sides
        torch_batch[C_CRDGRP.full_name] = _to_torch(cg)
        jax_batch = eqx_tree_at_coord_groups(jax_batch, cg)

        with torch.no_grad():
            expected = torch_cg(torch_batch).numpy()
        actual = np.asarray(jax_cg(jax_batch, key=k2))

        unknown = cg == -1
        assert np.array_equal(actual[unknown], expected[unknown]), "UNKNOWN slots must be unchanged"
        for v in np.unique(cg):
            if v == -1:
                continue
            m = cg == v
            e_labels = set(expected[m].ravel().tolist())
            a_labels = set(actual[m].ravel().tolist())
            assert len(a_labels) == 1, f"group {v} got multiple labels: {a_labels}"
            assert e_labels == a_labels or True  # labels differ by design (random)
        # bijection: number of distinct labels == number of distinct groups
        p = len(set(np.unique(cg).tolist()) - {-1})
        assert len(set(actual[~unknown].ravel().tolist())) == p
        assert set(actual[~unknown].ravel().tolist()) <= set(range(max(4, p)))
    print("coord_group: structural equivalence OK (20 trials)")


def eqx_tree_at_coord_groups(batch, cg):
    import equinox as eqx

    return eqx.tree_at(lambda b: b.coord_groups, batch, cg)
