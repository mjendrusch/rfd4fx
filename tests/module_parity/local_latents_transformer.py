"""Trunk parity: full LocalLatentsTransformer forward (random weights, tiny config)
with the release checkpoint's feature subset, RMSNorm, hadamard pair writes,
triangle updates, recycle loop and all heads."""

from conftest import build_feature_batch, _to_torch

import numpy as np
import jax
import jax.numpy as jnp
import torch

from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer

PAD = ("bool", (8,))
XT = {"bb_ca": ("float", (8, 3)), "local_latents": ("float", (8, 4))}
T = {"bb_ca": ("float", ()), "local_latents": ("float", ())}

BATCH_SPECS = {
    "padding_mask": ("const", None),  # set below (all valid; see chirality note)
    "x_t": XT,
    "t": T,
    "token_type": ("enum", (8,)),
    "atom37_mask": ("bool", (8, 37)),
    "coords_nm": ("float", (8, 37, 3)),
    "condition_coordinate_1_atom": ("coord", (8, 37, 3)),
    "condition_InternalCoordinate_1_atom": ("coord", (8, 37, 3)),
    "condition_CoordinateGroup_1_atom": ("int", (8, 37), dict(lo=-1, hi=4)),
    "condition_rasa-binned_1_atom": ("enum", (8, 37)),
    "condition_hotspot_1_atom": ("bool", (8, 37)),
    "chirality": ("float", (8, 18)),
    "chiral_frame_atoms": ("int", (8, 6, 8), dict(lo=-1, hi=8)),
    "chiral_slot_owner": ("int", (8, 6), dict(lo=-1, hi=48)),
    "token_bond_path_length": ("int", (8, 8), dict(lo=0, hi=40)),
    "molecule_iid": ("int", (8,), dict(lo=0, hi=4)),
    "is_in_context_conditioning": ("bool", (8,)),
    "supports_local_latent": ("bool", (8,)),
    "recycle_remaining": ("const", None),
}

NN_KWARGS = dict(
    nlayers=4,
    token_dim=64,
    nheads=4,
    dim_cond=32,
    token_type_emb_dim=16,
    cath_emb_dim=8,
    latent_dim=4,
    seq_sep_dim=32,
    xt_pair_dist_dim=15,
    xt_pair_dist_min=0.1,
    xt_pair_dist_max=3.0,
    idx_emb_dim=32,
    t_emb_dim=32,
    strict_feats=False,
    noise_align_chirality=True,
    noise_align_chirality_translate=False,
    noise_align_internal_coords=False,
    noise_align_internal_coords_translate=False,
    use_atom37_factory=False,
    pair_repr_dim=32,
    update_pair_repr=True,
    update_pair_repr_every_n=2,
    use_tri_mult=True,
    tri_mult_c=16,
    tri_mult_bias=False,
    tri_mult_backend="torch",
    pair_write="hadamard",
    use_qkln=True,
    attention_type="torch",
    use_torch_compile=False,
    norm_type="rmsnorm_torch_fused",
    recycle={"enabled": True, "grad": "last", "condition_probability": 0.5},
    sequence_readout={"enabled": True, "detach": True, "num_classes": 32},
    output_parameterization={"bb_ca": "v", "local_latents": "v"},
    feats_seq=[
        "xt_bb_ca", "xt_local_latents", "token_type", "cond_a37coors_nm",
        "cond_a37coors_internal_nm", "rasa_binned", "hotspot", "chirality",
        # coord_group_seq excluded: its labels are random on both sides (verified
        # structurally in chirality_coord_group.py); a numeric trunk comparison
        # would fail by construction.
    ],
    feats_cond_seq=["time_emb_bb_ca", "time_emb_local_latents", "recycle_remaining"],
    feats_pair_repr=["xt_bb_ca_pair_dists", "token_bond_path_length", "molecule_idx_pair"],
    feats_pair_cond=["time_emb_bb_ca", "time_emb_local_latents"],
)

SEQ_PROB_KEYS = [f"condition_sequence-probability-{aa}" for aa in
                 ("ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS",
                  "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "A", "C", "G", "U",
                  "DA", "DC", "DG", "DT")]


def test_trunk_forward():
    torch.manual_seed(99)
    torch_mod = LocalLatentsTransformer(**NN_KWARGS).eval()

    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers, transformer

    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()
    jax_mod = from_torch(torch_mod)

    jitted = eqx_filter_jit(jax_mod)
    errs = {k: [] for k in ("bb_ca", "local_latents", "seq_logits", "insertion_logits")}
    for trial in range(20):
        key = jax.random.fold_in(jax.random.key(trial), 0)
        specs = {**BATCH_SPECS, **{k: ("coord", (8,)) for k in SEQ_PROB_KEYS}}
        specs["padding_mask"] = ("bool", (8,))
        torch_batch, jax_batch = build_feature_batch(specs, key)
        # all-valid padding: chiral frames reference arbitrary tokens, and torch
        # rejects frames that touch padding. Random-padding mask semantics are
        # verified at the layer/feature level.
        import equinox as eqx
        all_valid = jnp.ones((2, 8), dtype=bool)
        torch_batch["padding_mask"] = _to_torch(all_valid)
        jax_batch = eqx.tree_at(lambda b: b.padding_mask, jax_batch, all_valid)
        torch_batch["n_recycle"] = 2  # jax side passes n_recycle=2 below; keep both in sync
        # same coord groups on both sides (the feature randomizes labels itself)
        from rfproteina.conditions import C_CRDGRP
        cg = jax.random.randint(jax.random.key(500 + trial), (2, 8, 37), -1, 4)
        torch_batch[C_CRDGRP.full_name] = _to_torch(cg)
        import equinox as eqx
        jax_batch = eqx.tree_at(lambda b: b.coord_groups, jax_batch, cg)
        # chiral frames must reference valid tokens (torch raises on padding
        # references). The trunk test uses an all-valid padding mask (mask
        # semantics are covered at the layer/feature level); emit masking via
        # is_in_context/supports_local_latent still exercises head masking.
        fr = np.asarray(jax_batch.chiral_frame_atoms).copy()
        new_owner = jnp.asarray(jax_batch.chiral_slot_owner)
        # Active frames need valid origin entries and well-conditioned geometry:
        # real data puts both origin anchors on the frame's own token and uses
        # >=3 distinct bonded neighbors. With fewer distinct vectors the
        # covariance is rank-deficient and torch/jax SVD legitimately disagree
        # on the null-space basis (documented in status.md).
        tok_i = np.arange(8)[None, :, None]
        active = fr[..., 0] >= 0
        fr[..., :2] = np.where(active[..., None], tok_i[..., None].astype(fr.dtype), fr[..., :2])
        offs = np.arange(1, 7)
        nb = (tok_i[..., None] + offs[None, None, None, :]) % 8
        fr[..., 2:] = np.where(active[..., None], nb.astype(fr.dtype), fr[..., 2:])
        new_frames = jnp.asarray(fr)
        torch_batch["chiral_frame_atoms"] = _to_torch(new_frames)
        torch_batch["chiral_slot_owner"] = _to_torch(new_owner)
        jax_batch = eqx.tree_at(lambda b: b.chiral_frame_atoms, jax_batch, new_frames)
        jax_batch = eqx.tree_at(lambda b: b.chiral_slot_owner, jax_batch, new_owner)

        with torch.no_grad():
            expected = torch_mod(torch_batch)
        actual = jitted(jax_batch, n_recycle=2, key=jax.random.key(trial))

        errs["bb_ca"].append(float(np.abs(np.asarray(actual.bb_ca) - expected["bb_ca"]["v"].numpy()).max()))
        errs["local_latents"].append(float(np.abs(np.asarray(actual.local_latents) - expected["local_latents"]["v"].numpy()).max()))
        errs["seq_logits"].append(float(np.abs(np.asarray(actual.seq_logits) - expected["sequence"]["logits"].numpy()).max()))
        errs["insertion_logits"].append(float(np.abs(np.asarray(actual.insertion_logits) - expected["insertion"]["logits"].numpy()).max()))

    for name, e in errs.items():
        e = np.array(e)
        print(f"trunk {name}: {e.mean():.3e} ± {e.std():.3e} (max {e.max():.3e})")
        assert e.max() < 5e-5, f"trunk {name} parity failed: {e.max():.3e}"


def eqx_filter_jit(mod):
    import equinox as eqx

    return eqx.filter_jit(mod)
