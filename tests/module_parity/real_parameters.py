"""Real-parameter parity against the nvidia/RFD4-Proteina release bundle.

Converts the *actual* release weights (denoiser + AE encoder/decoder, 435.6M params)
and compares full forwards on a < 64-token batch under jit on CPU. Skipped when the
bundle snapshot is not present.

CoordGroupSeqFeat randomizes labels per framework (documented divergence); both sides
are patched to an identity mapping for this test — group labels are arbitrary inputs,
and the relabeling is a train-time augmentation whose JAX port is verified
structurally in chirality_coord_group.py.
"""

import os
import pathlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx
import torch

SNAPSHOT = pathlib.Path(
    os.environ.get(
        "RFD4FX_BUNDLE",
        "/Users/mjendrusch/repos/rfd4j/.hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502",
    )
)

pytestmark = pytest.mark.skipif(not SNAPSHOT.exists(), reason="release bundle not downloaded")


def _load_real():
    from rfd4_proteina import Proteina

    model = Proteina.from_pretrained(SNAPSHOT)
    model.eval()
    return model


def _patch_coord_group_identity(torch_nn):
    """Disable the random group relabeling on the torch side (identity mapping)."""
    from rfproteina.nn.feature_factory import CoordGroupSeqFeat

    def identity_forward(self, batch):
        return batch[C_KEY].float()

    from rfproteina.conditions import C_CRDGRP

    C_KEY = C_CRDGRP.full_name
    for creator in torch_nn.init_repr_factory.feat_creators:
        if isinstance(creator, CoordGroupSeqFeat):
            creator.forward = identity_forward.__get__(creator)


def _jax_patch_coord_group_identity(jax_nn):
    """Replace the JAX CoordGroupSeqFeat with an identity mapping module."""
    from rfd4fx.nn.features import CoordGroupSeqFeat
    from rfproteina.conditions import C_CRDGRP

    class _IdentityCoordGroup(eqx.Module):
        dim: int = 37

        def __call__(self, batch, *, key=None):
            return batch.coord_groups.astype(jnp.float32)

    creators = list(jax_nn.init_repr_factory.feat_creators)
    idx = [i for i, c in enumerate(creators) if isinstance(c, CoordGroupSeqFeat)]
    if idx:
        creators[idx[0]] = _IdentityCoordGroup()
        import dataclasses

        new_factory = dataclasses.replace(jax_nn.init_repr_factory, feat_creators=creators)
        return dataclasses.replace(jax_nn, init_repr_factory=new_factory)
    return jax_nn


# Batch: one 48-token two-chain system (protein chain + small-molecule tokens),
# exercising every active feature. Shapes follow conftest.build_feature_batch specs.
def _denoiser_specs(n=48):
    return {
        "padding_mask": ("bool", (n,)),
        "x_t": {"bb_ca": ("float", (n, 3)), "local_latents": ("float", (n, 32))},
        "t": {"bb_ca": ("float", ()), "local_latents": ("float", ())},
        "token_type": ("int", (n,), dict(lo=0, hi=7)),
        "atom37_mask": ("bool", (n, 37)),
        "coords_nm": ("float", (n, 37, 3)),
        "condition_coordinate_1_atom": ("coord", (n, 37, 3)),
        "condition_InternalCoordinate_1_atom": ("coord", (n, 37, 3)),
        "condition_CoordinateGroup_1_atom": ("int", (n, 37), dict(lo=-1, hi=4)),
        "condition_rasa-binned_1_atom": ("enum", (n, 37)),
        "condition_secondary-structure_1_residue": ("enum", (n,)),
        "condition_hbond_1_atom": ("enum", (n, 37)),
        "condition_pistack_1_atom": ("bool", (n, 37)),
        "condition_pication_1_atom": ("enum", (n, 37)),
        "condition_saltbridge_1_atom": ("enum", (n, 37)),
        "condition_hydrophobic_1_atom": ("bool", (n, 37)),
        "condition_metal-coord_1_atom": ("enum", (n, 37)),
        "condition_hotspot_1_atom": ("bool", (n, 37)),
        "condition_cath-label_1_residue": ("int", (n, 3), dict(lo=-1, hi=5)),
        "condition_index-group_1_token": ("int", (n,), dict(lo=0, hi=4)),
        "condition_ChainCentroid_1_atom": ("coord", (n, 3)),
        "chirality": ("float", (n, 18)),
        "chiral_frame_atoms": ("int", (n, 6, 8), dict(lo=-1, hi=n)),
        "chiral_slot_owner": ("int", (n, 6), dict(lo=-1, hi=n * 6)),
        "token_bond_path_length": ("int", (n, n), dict(lo=0, hi=40)),
        "bond_mask": ("bool", (n, n)),
        "bond_order": ("int", (n, n), dict(lo=0, hi=5)),
        "is_aromatic": ("bool", (n, n)),
        "canonical_polymer_ordering": ("int", (n, n), dict(lo=-1, hi=2)),
        "molecule_iid": ("int", (n,), dict(lo=0, hi=4)),
        "molecule_entity_id": ("int", (n,), dict(lo=0, hi=4)),
        "chain_iid": ("float", (n,)),
        "na_ss_pair": ("float", (n, n)),
        "na_ss_pair_mask": ("float", (n, n)),
        "na_ss_state": ("float", (n,)),
        "na_ss_state_mask": ("float", (n,)),
        "charge": ("int", (n, 37), dict(lo=0, hi=11)),
        "element": ("int", (n, 37), dict(lo=0, hi=64)),
        "is_in_context_conditioning": ("bool", (n,)),
        "supports_local_latent": ("bool", (n,)),
        "recycle_remaining": ("const", None),
    }


def test_denoiser_real():
    model = _load_real()
    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers, transformer

    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()

    _patch_coord_group_identity(model.nn)
    jax_nn = from_torch(model.nn)
    jax_nn = _jax_patch_coord_group_identity(jax_nn)

    from conftest import build_feature_batch, _to_torch

    sys_path = pathlib.Path(__file__).parent.parent / "module_parity"
    import sys

    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(sys_path))
    from conftest import build_feature_batch as bfb  # noqa: F401  (path shim)

    n = 48
    specs = _denoiser_specs(n)
    # add the 28 sequence-probability condition keys
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    seq_prob_keys = [c.full_name for c in SEQUENCE_PROBABILITY_CONDITIONS]
    specs.update({k: ("coord", (n,)) for k in seq_prob_keys})

    torch_batch, jax_batch = build_feature_batch(specs, jax.random.key(0))
    # deterministic emit masks + all-valid padding (chiral frames reference
    # arbitrary tokens; torch rejects padding references)
    ctx = np.zeros((2, n), dtype=bool)
    ctx[:, 32:] = True
    sup = np.zeros((2, n), dtype=bool)
    sup[:, :32] = True
    all_valid = np.ones((2, n), dtype=bool)
    torch_batch["padding_mask"] = _to_torch(jnp.asarray(all_valid))
    torch_batch["is_in_context_conditioning"] = _to_torch(jnp.asarray(ctx))
    torch_batch["supports_local_latent"] = _to_torch(jnp.asarray(sup))
    jax_batch = eqx.tree_at(lambda b: b.padding_mask, jax_batch, jnp.asarray(all_valid))
    jax_batch = eqx.tree_at(lambda b: b.is_in_context_conditioning, jax_batch, jnp.asarray(ctx))
    jax_batch = eqx.tree_at(lambda b: b.supports_local_latent, jax_batch, jnp.asarray(sup))
    # well-conditioned chirality frames (origins on own token, 6 distinct neighbors)
    fr = np.asarray(jax_batch.chiral_frame_atoms).copy()
    tok_i = np.arange(n)[None, :, None]
    active = fr[..., 0] >= 0
    fr[..., :2] = np.where(active[..., None], tok_i[..., None].astype(fr.dtype), fr[..., :2])
    offs = np.arange(1, 7)
    nb = (tok_i[..., None] + offs[None, None, None, :]) % n
    fr[..., 2:] = np.where(active[..., None], nb.astype(fr.dtype), fr[..., 2:])
    torch_batch["chiral_frame_atoms"] = _to_torch(jnp.asarray(fr))
    jax_batch = eqx.tree_at(lambda b: b.chiral_frame_atoms, jax_batch, jnp.asarray(fr))
    torch_batch["n_recycle"] = 2  # torch reads n_recycle from the batch; jax takes it as a kwarg

    jitted = eqx.filter_jit(jax_nn)
    with torch.no_grad():
        expected = model.nn(torch_batch)
    actual = jitted(jax_batch, n_recycle=2, key=jax.random.key(0))

    checks = {
        "bb_ca": (actual.bb_ca, expected["bb_ca"]["v"]),
        "local_latents": (actual.local_latents, expected["local_latents"]["v"]),
        "seq_logits": (actual.seq_logits, expected["sequence"]["logits"]),
    }
    for name, (a, e) in checks.items():
        err = float(np.abs(np.asarray(a) - e.numpy()).max())
        print(f"real denoiser {name}: max abs err {err:.3e}")
        assert err < 5e-5, f"real-parameter denoiser {name} failed: {err:.3e}"
    # Insertion head: bilinear q@k^T over 768 dims; with real weights the logits
    # reach |~1e4|, so tiny s differences (2e-5) amplify to ~4e-3 absolute —
    # relative error is the meaningful gate here (p95 ~ 7e-7 observed).
    a, e = actual.insertion_logits, expected["insertion"]["logits"]
    rel = np.abs(np.asarray(a) - e.numpy()) / (np.abs(e.numpy()) + 1e-9)
    p95 = float(np.percentile(rel, 95))
    print(f"real denoiser insertion_logits: abs {float(np.abs(np.asarray(a) - e.numpy()).max()):.3e}, rel p95 {p95:.3e}")
    assert p95 < 1e-4, f"real-parameter denoiser insertion relative error too high: {p95:.3e}"


def test_ae_real():
    torch.manual_seed(0)  # the encoder's VAE sample uses the global torch RNG
    model = _load_real()
    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers, transformer

    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()

    jax_enc = from_torch(model.autoencoder.encoder)
    jax_dec = from_torch(model.autoencoder.decoder)

    from conftest import build_feature_batch, _to_torch

    n = 48
    enc_specs = {
        "padding_mask": ("bool", (n,)),
        "token_type": ("int", (n,), dict(lo=0, hi=7)),
        "atom37_mask": ("bool", (n, 37)),
        "coords_nm": ("float", (n, 37, 3)),
        "charge": ("int", (n, 37), dict(lo=0, hi=11)),
        "ground_truth_element": ("int", (n, 37), dict(lo=0, hi=64)),
        "chain_iid": ("float", (n,)),
        "incident_bond_type_counts": ("float", (n, 37, 10)),
        "token_bond_path_length": ("int", (n, n), dict(lo=0, hi=40)),
        "canonical_polymer_ordering": ("int", (n, n), dict(lo=-1, hi=2)),
        "molecule_iid": ("int", (n,), dict(lo=0, hi=4)),
        "is_in_context_conditioning": ("bool", (n,)),
        "supports_local_latent": ("bool", (n,)),
        "x_t": {"bb_ca": ("float", (n, 3))},  # shape source for the zero cond factory
    }
    torch_batch, jax_batch = build_feature_batch(enc_specs, jax.random.key(1))
    ctx = np.zeros((2, n), dtype=bool)
    ctx[:, 32:] = True
    sup = np.zeros((2, n), dtype=bool)
    sup[:, :32] = True
    torch_batch["is_in_context_conditioning"] = _to_torch(jnp.asarray(ctx))
    torch_batch["supports_local_latent"] = _to_torch(jnp.asarray(sup))
    jax_batch = eqx.tree_at(lambda b: b.is_in_context_conditioning, jax_batch, jnp.asarray(ctx))
    jax_batch = eqx.tree_at(lambda b: b.supports_local_latent, jax_batch, jnp.asarray(sup))

    with torch.no_grad():
        enc_out = model.autoencoder.encoder(torch_batch)
    enc_j = eqx.filter_jit(jax_enc)(jax_batch, key=jax.random.key(3))
    mean_err = float(np.abs(np.asarray(enc_j.mean) - enc_out["mean"].numpy()).max())
    ls_err = float(np.abs(np.asarray(enc_j.log_scale) - enc_out["log_scale"].numpy()).max())
    print(f"real AE encoder mean: {mean_err:.3e}, log_scale: {ls_err:.3e}")
    assert mean_err < 5e-5 and ls_err < 5e-5

    # decoder: feed the *torch* sampled latents through both decoders
    z_t = enc_out["z_latent"]
    torch_dec_batch = {
        "z_latent": z_t,
        "ca_coors_nm": torch_batch["x_t"]["bb_ca"],
        "padding_mask": torch_batch["padding_mask"],
        "token_type": torch_batch["token_type"],
        "is_in_context_conditioning": torch_batch["is_in_context_conditioning"],
        "supports_local_latent": torch_batch["supports_local_latent"],
    }
    jax_dec_batch = eqx.tree_at(
        lambda b: b.z_latent, jax_batch, jnp.asarray(z_t.numpy()), is_leaf=lambda x: x is None
    )
    ca = jnp.asarray(torch_batch["x_t"]["bb_ca"].numpy())
    jax_dec_batch = eqx.tree_at(
        lambda b: b.ca_coors_nm, jax_dec_batch, ca, is_leaf=lambda x: x is None
    )
    with torch.no_grad():
        dec_out = model.autoencoder.decoder(torch_dec_batch)
    dec_j = eqx.filter_jit(jax_dec)(jax_dec_batch, key=jax.random.key(4))
    checks = {
        "coors_nm": (dec_j.coors_nm, dec_out["coors_nm"]),
        "seq_logits": (dec_j.seq_logits, dec_out["seq_logits"]),
        "atom_presence_logits": (dec_j.atom_presence_logits, dec_out["atom_presence_logits"]),
    }
    for name, (a, e) in checks.items():
        err = float(np.abs(np.asarray(a) - e.numpy()).max())
        print(f"real AE decoder {name}: max abs err {err:.3e}")
        assert err < 5e-5, f"real-parameter AE decoder {name} failed: {err:.3e}"
    assert float(np.mean(np.asarray(dec_j.aatype_max) != dec_out["aatype_max"].numpy())) == 0.0
    assert float(np.mean(np.asarray(dec_j.atom_mask) != dec_out["atom_mask"].numpy())) == 0.0
