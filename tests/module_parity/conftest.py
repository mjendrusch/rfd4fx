"""Shared parity-test harness.

Every module parity test: torch module with random (seeded) weights, converted via
from_torch; 100 seeded random input combinations; both sides run on CPU — torch
eager (no_grad, eval), JAX under eqx.filter_jit. Reports per-trial max abs error:
mean ± std over trials, plus the global max. Gates are uniform across modules and
never relaxed: max < 5e-5.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

try:
    import torch
except ImportError:  # native extra not installed — the parity suite is skipped
    torch = None

from rfd4fx.backend import from_torch, register_base_types
from rfd4fx.nn import composites, features, layers, transformer

if torch is not None:
    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()
else:
    collect_ignore_glob = ["*"]

# --- release checkpoint ------------------------------------------------------
# Loaded from the Hugging Face hub (pinned revision) on first use. Point
# RFD4FX_BUNDLE at a local bundle directory, or RFD4FX_HF_CACHE at a cache
# directory, to override.
SNAPSHOT = os.environ.get("RFD4FX_BUNDLE", "nvidia/RFD4-Proteina")
REV = "a06e2509c7033b010fc43614e9fd840303588502"
CACHE = os.environ.get("RFD4FX_HF_CACHE") or None


def release_bundle_available() -> bool:
    """True when the release checkpoint is loadable (local dir or HF cache)."""
    if pathlib.Path(SNAPSHOT).expanduser().is_dir():
        return True
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            SNAPSHOT,
            revision=REV,
            allow_patterns=["config.json", "model.safetensors"],
            cache_dir=CACHE,
            local_files_only=True,
        )
        return True
    except Exception:
        return False


MAX_ABS_GATE = 5e-5
N_TRIALS = 100


def _rand(kind, shape, key, lo=0, hi=16):
    k, key = jax.random.split(key)
    if kind == "float":
        return jax.random.normal(k, shape, jnp.float32)
    if kind == "bool":
        return jax.random.bernoulli(k, p=0.8, shape=shape)
    if kind == "int":
        return jax.random.randint(k, shape, lo, hi)
    if kind == "enum":  # small non-negative ints (annotation enums)
        return jax.random.randint(k, shape, 0, 4)
    if kind == "coord":  # atom37 coordinates in Å with NaN holes
        x = jax.random.normal(k, shape, jnp.float32) * 5.0
        km, _ = jax.random.split(key)
        holes = jax.random.bernoulli(km, p=0.3, shape=shape[:-1])
        return jnp.where(holes[..., None], jnp.nan, x)
    raise ValueError(kind)


def _to_torch(x):
    arr = np.asarray(x)
    t = torch.from_numpy(arr.copy())
    return t.long() if t.dtype == torch.int32 else (t.bool() if t.dtype == torch.bool else t)


# torch batch key -> FlowBatch field. Condition keys are resolved at import from
# the native condition objects so names never drift.
def _condition_key_map():
    import rfproteina.conditions as con
    from atomworks.ml.conditions import C_SEQ

    return {
        C_SEQ.full_name: "cond_seq",
        con.C_CRD.full_name: "cond_coords",
        con.C_CRDGRP.full_name: "coord_groups",
        con.C_INTCRD.full_name: "cond_internal_coords",
        con.C_RAS.full_name: "rasa",
        con.C_SEC.full_name: "secondary_structure",
        con.C_HB.full_name: "hbond",
        con.C_PISTACK.full_name: "pistack",
        con.C_PICATION.full_name: "pication",
        con.C_SALT.full_name: "saltbridge",
        con.C_HPHOB.full_name: "hydrophobic",
        con.C_MCOORD.full_name: "metal_coord",
        con.C_HOT.full_name: "hotspot",
        con.C_CENTROID.full_name: "chain_centroid",
        con.C_CATH.full_name: "cath_codes",
        con.C_IDXGRP.full_name: "idx_group",
        "chirality": "chirality",
        "chiral_frame_atoms": "chiral_frame_atoms",
        "chiral_slot_owner": "chiral_slot_owner",
        "token_type": "token_type",
        "atom37_mask": "atom37_mask",
        "coords_nm": "coords_nm",
        "charge": "charge",
        "element": "element",
        "ground_truth_element": "ground_truth_element",
        "incident_bond_type_counts": "incident_bond_type_counts",
        "ca_coors_nm": "ca_coors_nm",
        "z_latent": "z_latent",
        "token_bond_path_length": "token_bond_path_length",
        "bond_mask": "bond_mask",
        "bond_order": "bond_order",
        "is_aromatic": "is_aromatic",
        "canonical_polymer_ordering": "canonical_polymer_ordering",
        "molecule_iid": "molecule_iid",
        "molecule_entity_id": "molecule_entity_id",
        "chain_iid": "chain_iid",
        "na_ss_pair": "na_ss_pair",
        "na_ss_pair_mask": "na_ss_pair_mask",
        "na_ss_state": "na_ss_state",
        "na_ss_state_mask": "na_ss_state_mask",
    }


def build_feature_batch(specs: dict, key, b: int = 2, n: int = 8):
    """specs: torch batch key -> ("float"|"bool"|"int"|"enum"|"coord", shape-without-leading-b)
    or ("const", value). Special keys "x_t"/"t" take sub-dicts. Returns (torch dict, FlowBatch)."""
    from rfproteina.conditions import SEQUENCE_PROBABILITY_CONDITIONS

    key_map = _condition_key_map()
    torch_batch, fields = {}, {}
    for torch_key, spec in specs.items():
        if torch_key in ("x_t", "t"):
            torch_sub, field_prefix = {}, "x_t" if torch_key == "x_t" else "t"
            for sub_key, sub_spec in spec.items():
                v = _rand(sub_spec[0], (b, *sub_spec[1]), key, **(sub_spec[2] if len(sub_spec) > 2 else {}))
                torch_sub[sub_key] = _to_torch(v)
                fields[f"{field_prefix}_{sub_key}"] = v
            torch_batch[torch_key] = torch_sub
            continue
        if spec[0] == "const":
            torch_batch[torch_key] = spec[1]
            if torch_key == "recycle_remaining":
                fields["recycle_remaining"] = spec[1]
            continue
        field = key_map.get(torch_key, torch_key)
        if field.startswith("condition_sequence-probability"):
            # aggregated into seq_probabilities below; skip the field
            v = _rand(spec[0], (b, *spec[1]), key, **(spec[2] if len(spec) > 2 else {}))
            torch_batch[torch_key] = _to_torch(v)
            continue
        v = _rand(spec[0], (b, *spec[1]), key, **(spec[2] if len(spec) > 2 else {}))
        torch_batch[torch_key] = _to_torch(v)
        fields[field] = v

    # sequence probabilities: stack the 28 condition tensors with NaN holes
    if any(k.startswith("condition_sequence-probability") for k in specs):
        cols = []
        for c in SEQUENCE_PROBABILITY_CONDITIONS:
            v = _rand("coord", (b, specs["padding_mask"][1][0]), key)  # NaN holes
            torch_batch[c.full_name] = _to_torch(v)
            cols.append(v)
        fields["seq_probabilities"] = jnp.stack(cols, axis=-1)

    fields.setdefault("padding_mask", fields.get("padding_mask"))
    from rfd4fx.types import FlowBatch

    return torch_batch, FlowBatch(**fields)


def run_feature_parity(torch_feat, jax_feat, specs, n_trials: int = N_TRIALS, seed: int = 0, with_key: bool = False):
    """Compare a torch feature (dict batch) against its JAX port (FlowBatch)."""
    jitted = eqx.filter_jit(jax_feat)
    errs = []
    for trial in range(n_trials):
        key = jax.random.fold_in(jax.random.key(seed), trial)
        torch_batch, jax_batch = build_feature_batch(specs, key)
        with torch.no_grad():
            expected = torch_feat(torch_batch)
        actual = jitted(jax_batch, key=jax.random.key(1000 + trial)) if with_key else jitted(jax_batch)
        expected = expected.numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
        errs.append(float(np.abs(np.asarray(actual) - expected).max()))
    errs = np.array(errs)
    mean, std, gmax = float(errs.mean()), float(errs.std()), float(errs.max())
    assert gmax < MAX_ABS_GATE, f"feature parity failed: max abs err {gmax:.3e}"
    return mean, std, gmax


def make_pair(torch_cls, *args, seed: int = 1234, **kwargs):
    """Build a torch module with seeded random weights and its JAX conversion."""
    torch.manual_seed(seed)
    torch_mod = torch_cls(*args, **kwargs).eval()
    register_base_types()
    layers.register()
    composites.register()
    features.register()
    transformer.register()
    jax_mod = from_torch(torch_mod)
    return torch_mod, jax_mod


def random_inputs(specs: dict, key, dtype=jnp.float32):
    """specs: name -> ("float"|"bool"|"int", shape) or callable(key) -> pytree."""
    out = {}
    for name, spec in specs.items():
        if callable(spec):
            out[name] = spec(key)
        else:
            kind, shape = spec
            k, key = jax.random.split(key)
            if kind == "float":
                out[name] = jax.random.normal(k, shape, dtype)
            elif kind == "bool":
                # p=0.8 keeps ~20% of entries masked out, exercising mask paths.
                out[name] = jax.random.bernoulli(k, p=0.8, shape=shape)
            elif kind == "int":
                out[name] = jax.random.randint(k, shape, 0, 16)
            else:
                raise ValueError(kind)
    return out


def run_parity(torch_mod, jax_mod, specs, n_trials: int = N_TRIALS, seed: int = 0):
    """Run n_trials random input combinations; return (mean, std, global_max) of the
    per-trial max abs error, asserting the uniform gate."""
    jitted = eqx.filter_jit(jax_mod)
    errs = []
    for trial in range(n_trials):
        key = jax.random.fold_in(jax.random.key(seed), trial)
        inputs = random_inputs(specs, key)
        jax_inputs = [inputs[k] for k in specs]

        with torch.no_grad():
            torch_inputs = []
            for x in jax_inputs:
                if isinstance(x, jax.Array):
                    arr = np.asarray(x)
                    t = torch.from_numpy(arr)
                    torch_inputs.append(t.long() if t.dtype == torch.int32 else t)
                else:
                    torch_inputs.append(x)
            expected = torch_mod(*torch_inputs)

        actual = jitted(*jax_inputs)
        expected = expected.numpy() if isinstance(expected, torch.Tensor) else np.asarray(expected)
        errs.append(float(np.abs(np.asarray(actual) - expected).max()))

    errs = np.array(errs)
    mean, std, gmax = float(errs.mean()), float(errs.std()), float(errs.max())
    assert gmax < MAX_ABS_GATE, f"parity failed: max abs err {gmax:.3e} >= {MAX_ABS_GATE:.0e}"
    return mean, std, gmax
