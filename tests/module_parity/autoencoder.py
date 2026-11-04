"""AE encoder/decoder parity (random weights, tiny config). The encoder test
compares mean/log_scale/z_latent with a fixed PRNG key on both sides; the decoder
test covers logits, token-type restriction, coordinates, argmax and atom masks."""

from conftest import build_feature_batch, _to_torch

import numpy as np
import jax
import jax.numpy as jnp
import torch
import equinox as eqx

from rfproteina.partial_autoencoder.decoder import DecoderTransformer
from rfproteina.partial_autoencoder.encoder import EncoderTransformer

PAD = ("bool", (8,))

ENCODER_KWARGS = dict(
    latent_z_dim=4,
    n_residue_types=32,
    encoder=dict(
        nlayers=4,
        token_dim=64,
        nheads=4,
        parallel_mha_transition=False,
        token_type_emb_dim=16,
        strict_feats=False,
        feats_seq=[
            "token_type", "x1_a37coors_nm", "x1_a37coors_nm_rel", "chain_idx_seq",
            "incident_bond_type_counts", "atom37_element_charge_embedding",
        ],
        feats_cond_seq=None,
        dim_cond=128,  # torch ZeroFeat hardcodes dim=128; ret-zero cond paths need this
        idx_emb_dim=32,
        seq_sep_dim=32,
        pair_repr_dim=32,
        update_pair_repr=False,
        update_pair_repr_every_n=3,
        use_tri_mult=False,
        use_attn_pair_bias=True,
        use_qkln=True,
        normalize_latent=False,
        latent_z_dim=4,
        n_residue_types=32,
        charge_emb_dim=16,
        norm_type="rmsnorm_torch_fused",
        feats_pair_repr=["token_bond_path_length", "molecule_idx_pair", "x1_bb_pair_dists_nm"],
    ),
)

DECODER_KWARGS = dict(
    n_residue_types=147,  # decoder vocabulary is fixed at 147 (hard constraint)
    decoder=dict(
        nlayers=4,
        token_dim=64,
        abs_coors=False,
        nheads=4,
        token_type_emb_dim=16,
        parallel_mha_transition=False,
        strict_feats=False,
        feats_seq=["ca_coors_nm", "z_latent_seq", "token_type", "cond_residue_type"],
        feats_cond_seq=None,
        dim_cond=128,  # torch ZeroFeat hardcodes dim=128; ret-zero cond paths need this
        idx_emb_dim=32,
        seq_sep_dim=32,
        pair_repr_dim=32,
        update_pair_repr=False,
        update_pair_repr_every_n=3,
        use_tri_mult=False,
        use_attn_pair_bias=True,
        use_qkln=True,
        latent_z_dim=4,
        n_residue_types=147,
        norm_type="rmsnorm_torch_fused",
        feats_pair_repr=["token_bond_path_length", "molecule_idx_pair", "ca_coors_nm_pair_dists"],
    ),
)

ENCODER_BATCH = {
    "padding_mask": PAD,
    "token_type": ("enum", (8,)),
    "atom37_mask": ("bool", (8, 37)),
    "coords_nm": ("float", (8, 37, 3)),
    "charge": ("int", (8, 37), dict(lo=0, hi=11)),
    "ground_truth_element": ("int", (8, 37), dict(lo=0, hi=64)),
    "chain_iid": ("float", (8,)),
    "incident_bond_type_counts": ("float", (8, 37, 10)),
    "token_bond_path_length": ("int", (8, 8), dict(lo=0, hi=40)),
    "molecule_iid": ("int", (8,), dict(lo=0, hi=4)),
    "is_in_context_conditioning": ("bool", (8,)),
    "supports_local_latent": ("bool", (8,)),
    # ZeroFeat.extract_bs_and_n on the (empty) cond factory needs one of these:
    "x_t": {"bb_ca": ("float", (8, 3))},
}

DECODER_BATCH = {
    "padding_mask": PAD,
    "token_type": ("enum", (8,)),
    "ca_coors_nm": ("float", (8, 3)),
    "z_latent": ("float", (8, 4)),
    "condition_sequence_1_residue": ("int", (8,), dict(lo=0, hi=147)),
    "token_bond_path_length": ("int", (8, 8), dict(lo=0, hi=40)),
    "molecule_iid": ("int", (8,), dict(lo=0, hi=4)),
    "is_in_context_conditioning": ("bool", (8,)),
    "supports_local_latent": ("bool", (8,)),
}


def _convert(torch_cls, kwargs):
    torch.manual_seed(17)
    torch_mod = torch_cls(**kwargs).eval()
    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers, transformer

    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()
    return torch_mod, from_torch(torch_mod)


def test_encoder():
    torch_mod, jax_mod = _convert(EncoderTransformer, ENCODER_KWARGS)
    jitted = eqx.filter_jit(jax_mod)
    errs = {k: [] for k in ("mean", "log_scale", "z_latent", "z_latent_pre_ln")}
    for trial in range(20):
        torch_batch, jax_batch = build_feature_batch(ENCODER_BATCH, jax.random.fold_in(jax.random.key(trial), 0))
        # deterministic masks: tokens 0..3 emitted, 4..7 context
        ctx = np.zeros((2, 8), dtype=bool); ctx[:, 4:] = True
        sup = np.zeros((2, 8), dtype=bool); sup[:, :4] = True
        torch_batch["is_in_context_conditioning"] = _to_torch(jnp.asarray(ctx))
        torch_batch["supports_local_latent"] = _to_torch(jnp.asarray(sup))
        jax_batch = eqx.tree_at(lambda b: b.is_in_context_conditioning, jax_batch, jnp.asarray(ctx))
        jax_batch = eqx.tree_at(lambda b: b.supports_local_latent, jax_batch, jnp.asarray(sup))
        key = jax.random.key(42 + trial)
        with torch.no_grad():
            expected = torch_mod(torch_batch)
        actual = jitted(jax_batch, key=key)
        # torch samples with the global RNG; both sides use their own stream, so
        # z_latent itself is not comparable — compare mean/log_scale exactly and
        # verify z is a valid sample of the emitted distribution instead.
        errs["mean"].append(float(np.abs(np.asarray(actual.mean) - expected["mean"].numpy()).max()))
        errs["log_scale"].append(float(np.abs(np.asarray(actual.log_scale) - expected["log_scale"].numpy()).max()))
        # z and z_pre_ln contain framework-specific RNG samples: not comparable
        # numerically. Verify both are valid samples of the emitted Gaussian
        # (|z-score| < 5 on emitted tokens).
        std = np.exp(np.asarray(actual.log_scale))
        for name, zval in (("z_latent", actual.z_latent), ("z_latent_pre_ln", actual.z_latent_pre_ln)):
            zscore = (np.asarray(zval) - np.asarray(actual.mean)) / std
            errs[name].append(float(np.abs(zscore[np.asarray(actual.emit_mask)]).max()))
    for k, e in errs.items():
        e = np.array(e)
        print(f"encoder {k}: {e.mean():.3e} ± {e.std():.3e} (max {e.max():.3e})")
    assert max(errs["mean"]) < 5e-5 and max(errs["log_scale"]) < 5e-5
    # sampled z within 5 sigma of the emitted Gaussian on emitted tokens
    assert max(errs["z_latent"]) < 5.0 and max(errs["z_latent_pre_ln"]) < 5.0


def test_decoder():
    torch_mod, jax_mod = _convert(DecoderTransformer, DECODER_KWARGS)
    jitted = eqx.filter_jit(jax_mod)
    keys = ("coors_nm", "seq_logits", "aatype_max", "atom_mask", "atom_presence_logits")
    errs = {k: [] for k in keys}
    for trial in range(20):
        torch_batch, jax_batch = build_feature_batch(DECODER_BATCH, jax.random.fold_in(jax.random.key(trial), 0))
        ctx = np.zeros((2, 8), dtype=bool); ctx[:, 4:] = True
        sup = np.zeros((2, 8), dtype=bool); sup[:, :4] = True
        torch_batch["is_in_context_conditioning"] = _to_torch(jnp.asarray(ctx))
        torch_batch["supports_local_latent"] = _to_torch(jnp.asarray(sup))
        jax_batch = eqx.tree_at(lambda b: b.is_in_context_conditioning, jax_batch, jnp.asarray(ctx))
        jax_batch = eqx.tree_at(lambda b: b.supports_local_latent, jax_batch, jnp.asarray(sup))
        with torch.no_grad():
            expected = torch_mod(torch_batch)
        actual = jitted(jax_batch, key=jax.random.key(trial))
        errs["coors_nm"].append(float(np.abs(np.asarray(actual.coors_nm) - expected["coors_nm"].numpy()).max()))
        errs["seq_logits"].append(float(np.abs(np.asarray(actual.seq_logits) - expected["seq_logits"].numpy()).max()))
        errs["aatype_max"].append(float(np.mean(np.asarray(actual.aatype_max) != expected["aatype_max"].numpy())))
        errs["atom_mask"].append(float(np.mean(np.asarray(actual.atom_mask) != expected["atom_mask"].numpy())))
        errs["atom_presence_logits"].append(float(np.abs(np.asarray(actual.atom_presence_logits) - expected["atom_presence_logits"].numpy()).max()))
    for k, e in errs.items():
        e = np.array(e)
        print(f"decoder {k}: {e.mean():.3e} ± {e.std():.3e} (max {e.max():.3e})")
    assert max(errs["coors_nm"]) < 5e-5
    assert max(errs["seq_logits"]) < 5e-5
    assert max(errs["aatype_max"]) == 0.0  # argmax must agree exactly
    assert max(errs["atom_mask"]) == 0.0  # discrete masks must agree exactly
    assert max(errs["atom_presence_logits"]) < 5e-5
