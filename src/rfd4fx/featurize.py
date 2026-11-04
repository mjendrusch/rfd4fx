"""Host-tensor -> FlowBatch assembly in ``jax.numpy``.

Tokenization, the bond graph and chirality packing remain host-side (numpy);
this module assembles the model-ready batch channels."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .types import FlowBatch

# Host keys in the SEQUENCE_PROBABILITY_CONDITIONS order (dict ordering matters
# for the stacked [b, n, 28] tensor).
SEQ_PROB_ORDER: list[str] = [
    "condition_sequence-probability-ALA_1_residue", "condition_sequence-probability-ARG_1_residue",
    "condition_sequence-probability-ASN_1_residue", "condition_sequence-probability-ASP_1_residue",
    "condition_sequence-probability-CYS_1_residue", "condition_sequence-probability-GLN_1_residue",
    "condition_sequence-probability-GLU_1_residue", "condition_sequence-probability-GLY_1_residue",
    "condition_sequence-probability-HIS_1_residue", "condition_sequence-probability-ILE_1_residue",
    "condition_sequence-probability-LEU_1_residue", "condition_sequence-probability-LYS_1_residue",
    "condition_sequence-probability-MET_1_residue", "condition_sequence-probability-PHE_1_residue",
    "condition_sequence-probability-PRO_1_residue", "condition_sequence-probability-SER_1_residue",
    "condition_sequence-probability-THR_1_residue", "condition_sequence-probability-TRP_1_residue",
    "condition_sequence-probability-TYR_1_residue", "condition_sequence-probability-VAL_1_residue",
    "condition_sequence-probability-A_1_residue", "condition_sequence-probability-C_1_residue",
    "condition_sequence-probability-G_1_residue", "condition_sequence-probability-U_1_residue",
    "condition_sequence-probability-DA_1_residue", "condition_sequence-probability-DC_1_residue",
    "condition_sequence-probability-DG_1_residue", "condition_sequence-probability-DT_1_residue",
]

# Host tensor key -> FlowBatch field (identity unless noted).
_KEY_TO_FIELD = {
    "atom37_mask": "atom37_mask",
    "condition_coordinate_1_atom": "cond_coords",
    "condition_InternalCoordinate_1_atom": "cond_internal_coords",
    "condition_CoordinateGroup_1_atom": "coord_groups",
    "condition_rasa-binned_1_atom": "rasa",
    "condition_secondary-structure_1_residue": "secondary_structure",
    "condition_hbond_1_atom": "hbond",
    "condition_pistack_1_atom": "pistack",
    "condition_pication_1_atom": "pication",
    "condition_saltbridge_1_atom": "saltbridge",
    "condition_hydrophobic_1_atom": "hydrophobic",
    "condition_metal-coord_1_atom": "metal_coord",
    "condition_hotspot_1_atom": "hotspot",
    "condition_cath-label_1_residue": "cath_codes",
    "condition_index-group_1_token": "idx_group",
    "condition_ChainCentroid_1_atom": "chain_centroid",
    "condition_sequence_1_residue": "cond_seq",
    "token_type": "token_type",
    "chain_iid": "chain_iid",
    "molecule_iid": "molecule_iid",
    "molecule_entity_id": "molecule_entity_id",
    "token_bond_path_length": "token_bond_path_length",
    "bond_mask": "bond_mask",
    "bond_order": "bond_order",
    "is_aromatic": "is_aromatic",
    "canonical_polymer_ordering": "canonical_polymer_ordering",
    "na_ss_state": "na_ss_state",
    "na_ss_state_mask": "na_ss_state_mask",
    "na_ss_pair": "na_ss_pair",
    "na_ss_pair_mask": "na_ss_pair_mask",
    "charge": "charge",
    "element": "element",
    "ground_truth_element": "ground_truth_element",
    "incident_bond_type_counts": "incident_bond_type_counts",
    "chirality": "chirality",
    "chiral_frame_atoms": "chiral_frame_atoms",
    "chiral_slot_owner": "chiral_slot_owner",
    "is_in_context_conditioning": "is_in_context_conditioning",
    "supports_local_latent": "supports_local_latent",
}

# Schema defaults (proteina.py SPECS) for host tensors absent from a batch.
_DEFAULTS = {
    "condition_coordinate_1_atom": lambda b, n: jnp.full((b, n, 37, 3), jnp.nan, jnp.float32),
    "condition_InternalCoordinate_1_atom": lambda b, n: jnp.full((b, n, 37, 3), jnp.nan, jnp.float32),
    "condition_CoordinateGroup_1_atom": lambda b, n: jnp.full((b, n, 37), -1, jnp.int32),
    "condition_rasa-binned_1_atom": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "condition_secondary-structure_1_residue": lambda b, n: jnp.zeros((b, n), jnp.int32),
    "condition_hbond_1_atom": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "condition_pistack_1_atom": lambda b, n: jnp.zeros((b, n, 37), bool),
    "condition_pication_1_atom": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "condition_saltbridge_1_atom": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "condition_hydrophobic_1_atom": lambda b, n: jnp.zeros((b, n, 37), bool),
    "condition_metal-coord_1_atom": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "condition_hotspot_1_atom": lambda b, n: jnp.zeros((b, n, 37), bool),
    "condition_cath-label_1_residue": lambda b, n: jnp.full((b, n, 3), -1, jnp.int32),
    "condition_index-group_1_token": lambda b, n: jnp.full((b, n), -1, jnp.int32),
    "condition_ChainCentroid_1_atom": lambda b, n: jnp.zeros((b, n, 3), jnp.float32),
    "condition_sequence_1_residue": lambda b, n: jnp.zeros((b, n), jnp.int32),  # <M> = 0
    "na_ss_state": lambda b, n: jnp.zeros((b, n), jnp.float32),
    "na_ss_state_mask": lambda b, n: jnp.zeros((b, n), bool),
    "na_ss_pair": lambda b, n: jnp.zeros((b, n, n), jnp.float32),
    "na_ss_pair_mask": lambda b, n: jnp.zeros((b, n, n), bool),
    "bond_mask": lambda b, n: jnp.zeros((b, n, n), bool),
    "bond_order": lambda b, n: jnp.zeros((b, n, n), jnp.int32),
    "is_aromatic": lambda b, n: jnp.zeros((b, n, n), bool),
    "canonical_polymer_ordering": lambda b, n: jnp.zeros((b, n, n), jnp.int32),
    "charge": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "element": lambda b, n: jnp.full((b, n, 37), -1, jnp.int32),
    "ground_truth_element": lambda b, n: jnp.zeros((b, n, 37), jnp.int32),
    "incident_bond_type_counts": lambda b, n: jnp.zeros((b, n, 37, 10), jnp.float32),
    "chirality": lambda b, n: jnp.zeros((b, n, 18), jnp.float32),
    "chiral_frame_atoms": lambda b, n: jnp.full((b, n, 6, 8), -1, jnp.int32),
    "chiral_slot_owner": lambda b, n: jnp.full((b, n, 6), -1, jnp.int32),
    "molecule_entity_id": lambda b, n: jnp.full((b, n), -1, jnp.int32),
    "chain_iid": lambda b, n: jnp.full((b, n), -1, jnp.float32),
}


def _to_flowbatch_dtype(key: str, value: jax.Array) -> jax.Array:
    """Normalize dtypes: bool fields to bool, ints to int32, floats to float32.

    torch collate uses int16/int64 in places; the JAX model standardizes on
    int32 (documented convention)."""
    field = _KEY_TO_FIELD[key]
    bool_fields = {"atom37_mask", "pistack", "hydrophobic", "hotspot", "bond_mask",
                   "is_aromatic", "na_ss_state_mask", "na_ss_pair_mask",
                   "is_in_context_conditioning", "supports_local_latent"}
    int_fields = {"coord_groups", "rasa", "secondary_structure", "hbond", "pication",
                  "saltbridge", "metal_coord", "cath_codes", "idx_group", "cond_seq",
                  "token_type", "molecule_iid", "molecule_entity_id",
                  "token_bond_path_length", "bond_order", "canonical_polymer_ordering",
                  "charge", "element", "ground_truth_element", "chiral_frame_atoms",
                  "chiral_slot_owner"}
    if field in bool_fields:
        return value.astype(bool)
    if field in int_fields:
        return value.astype(jnp.int32)
    return value.astype(jnp.float32)


def assemble_flow_batch(host: dict, *, is_padding: jax.Array | None = None) -> tuple[FlowBatch, jax.Array]:
    """Assemble the model's FlowBatch from host-level (numpy/jnp) tensors.

    All transformations are jnp: unit conversion, dtype normalization, sequence-
    probability stacking, defaults for absent conditions. Returns the FlowBatch
    and the coordinate-conditioning mask (used by the sampler's stochastic
    centering; not a model input).

    `host` must contain `padding_mask` and `atom37_mask`; every other key falls
    back to its schema default when absent.
    """
    padding = jnp.asarray(host["padding_mask"]).astype(bool)
    b, n = padding.shape

    fields: dict = {"padding_mask": padding}

    # Coordinate conditioning mask for the sampler (torch batch["coord_cond_mask"]).
    coord_cond_mask = (
        jnp.asarray(host["coord_cond_mask"]).astype(bool)
        if "coord_cond_mask" in host
        else jnp.zeros((b, n), bool)
    )

    for key, field in _KEY_TO_FIELD.items():
        if key in host:
            value = jnp.asarray(host[key])
        elif key in _DEFAULTS:
            value = _DEFAULTS[key](b, n)
        else:
            raise KeyError(f"host batch missing required tensor {key!r}")
        fields[field] = _to_flowbatch_dtype(key, value)

    # Ground-truth coordinates in nm (model-side coordinate features).
    fields["coords_nm"] = jnp.asarray(host["coords"]).astype(jnp.float32) / 10.0

    # Sequence probabilities: stack the 28 condition channels in canonical order
    # (NaN leaves a channel unspecified — SequenceProbabilitiesSeqFeat semantics).
    probs = [jnp.asarray(host[k]).astype(jnp.float32) if k in host
             else jnp.full((b, n), jnp.nan, jnp.float32)
             for k in SEQ_PROB_ORDER]
    fields["seq_probabilities"] = jnp.stack(probs, axis=-1)

    return FlowBatch(**fields), coord_cond_mask


def make_unconditioned_batch(n: int, b: int = 1) -> tuple[dict, FlowBatch]:
    """Properly unconditioned batch at arbitrary length: every annotation at its
    'unspecified' default (what the featurization pipeline produces for a bare
    de-novo protein chain). Returns (host-dict-style torch-free dict, FlowBatch)."""
    defaults = dict(
        padding_mask=jnp.ones((b, n), dtype=bool),
        is_in_context_conditioning=jnp.zeros((b, n), dtype=bool),
        supports_local_latent=jnp.ones((b, n), dtype=bool),
        token_type=jnp.zeros((b, n), dtype=jnp.int32),  # protein
        # De-novo input carries no ground-truth atoms: the torch featurizer emits
        # an all-False atom37 mask (and zero coords) for fully-diffused chains.
        atom37_mask=jnp.zeros((b, n, 37), dtype=bool),
        cond_seq=jnp.zeros((b, n), dtype=jnp.int32),  # <M> everywhere (unconditioned)
        coords_nm=jnp.zeros((b, n, 37, 3), jnp.float32),
        cond_coords=jnp.full((b, n, 37, 3), jnp.nan),
        cond_internal_coords=jnp.full((b, n, 37, 3), jnp.nan),
        coord_groups=jnp.full((b, n, 37), -1, jnp.int32),  # UNKNOWN
        rasa=jnp.zeros((b, n, 37), jnp.int32),  # UNSPECIFIED
        secondary_structure=jnp.zeros((b, n), jnp.int32),
        hbond=jnp.zeros((b, n, 37), jnp.int32),
        pistack=jnp.zeros((b, n, 37), dtype=bool),
        pication=jnp.zeros((b, n, 37), jnp.int32),
        saltbridge=jnp.zeros((b, n, 37), jnp.int32),
        hydrophobic=jnp.zeros((b, n, 37), dtype=bool),
        metal_coord=jnp.zeros((b, n, 37), jnp.int32),
        hotspot=jnp.zeros((b, n, 37), dtype=bool),
        chain_centroid=jnp.full((b, n, 3), jnp.nan),
        cath_codes=jnp.full((b, n, 3), -1, jnp.int32),  # null fold class
        idx_group=jnp.zeros((b, n), jnp.int32),
        chirality=None,  # no chirality conditioning -> zero feature
        chiral_frame_atoms=None,
        chiral_slot_owner=None,
        # True hop counts, UNCLIPPED — the seq_sep clamp lives inside
        # TokenBondPathLengthPairFeat, not in the batch (torch carries |i-j| up
        # to n-1 for a linear chain).
        token_bond_path_length=jnp.abs(
            jnp.arange(n)[None, :] - jnp.arange(n)[:, None]
        )[None].repeat(b, 0).astype(jnp.int32),
        bond_mask=jnp.zeros((b, n, n), dtype=bool),
        bond_order=jnp.zeros((b, n, n), jnp.int32),
        is_aromatic=jnp.zeros((b, n, n), dtype=bool),
        # Signed adjacency of canonical backbone linkages (+1 at [downstream,
        # upstream], -1 at the transpose; bonds.py:429). For a linear chain this
        # is sign(i-j) on the first off-diagonal — a DIRECTIONAL pair feature the
        # trunk consumes verbatim (CanonicalPolymerOrderingPairFeat); zeros would
        # erase the N->C orientation signal.
        canonical_polymer_ordering=(
            jnp.sign(jnp.arange(n)[:, None] - jnp.arange(n)[None, :])
            * (jnp.abs(jnp.arange(n)[:, None] - jnp.arange(n)[None, :]) == 1)
        )[None].repeat(b, 0).astype(jnp.int32),
        molecule_iid=jnp.zeros((b, n), jnp.int32),
        # torch pipeline entity ids are 1-based; only the equality matrix is used
        # (MoleculeEntityPairFeat), but keep the raw value faithful.
        molecule_entity_id=jnp.ones((b, n), jnp.int32),
        # Unknown element (UNK sentinel atoms carry no atomic number); AtomicElement
        # features clamp -1->0 and gate on is_atomized, so this only affects raw-batch fidelity.
        element=jnp.full((b, n, 37), -1, jnp.int32),
        chain_iid=jnp.zeros((b, n), jnp.float32),
        na_ss_pair=jnp.zeros((b, n, n), jnp.float32),
        na_ss_pair_mask=jnp.zeros((b, n, n), jnp.float32),
        na_ss_state=jnp.zeros((b, n), jnp.float32),
        na_ss_state_mask=jnp.zeros((b, n), jnp.float32),
        charge=jnp.zeros((b, n, 37), jnp.int32),
        ground_truth_element=jnp.zeros((b, n, 37), jnp.int32),
        incident_bond_type_counts=jnp.zeros((b, n, 37, 10), jnp.float32),
        seq_probabilities=jnp.full((b, n, 28), jnp.nan),  # all unspecified
        recycle_remaining=None,
    )
    return defaults, FlowBatch(**defaults)
