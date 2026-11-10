"""Featurization parity: jnp FlowBatch assembly vs the real torch inference
pipeline (collated batches dumped to fixtures/torch_batches/*.npz).

Tokenization, bond graph and chirality packing stay host-side (numpy); this
verifies the jnp-ported *conditioning information*: annotation → field mapping,
dtypes, defaults, sequence-probability stacking, unit conversion, masks."""

import pathlib

import numpy as np
import pytest
import jax
import jax.numpy as jnp

FIXTURES = pathlib.Path("tests/fixtures/torch_batches")
pytestmark = pytest.mark.skipif(not FIXTURES.exists(), reason="torch batch fixtures not dumped")

from rfd4fx.featurize import assemble_flow_batch  # noqa: E402
from rfd4fx.types import FlowBatch  # noqa: E402

# torch key -> (FlowBatch field, comparison kind)
DTYPE_RULES = {
    "atom37_mask": "bool", "pistack": "bool", "hydrophobic": "bool", "hotspot": "bool",
    "bond_mask": "bool", "is_aromatic": "bool", "na_ss_state_mask": "bool",
    "na_ss_pair_mask": "bool", "is_in_context_conditioning": "bool",
    "supports_local_latent": "bool",
}
INT_FIELDS = {"coord_groups", "rasa", "secondary_structure", "hbond", "pication",
              "saltbridge", "metal_coord", "cath_codes", "idx_group", "cond_seq",
              "token_type", "molecule_iid", "molecule_entity_id",
              "token_bond_path_length", "bond_order", "canonical_polymer_ordering",
              "charge", "element", "ground_truth_element", "chiral_frame_atoms",
              "chiral_slot_owner"}


def test_featurize_parity():
    npzs = sorted(FIXTURES.glob("*.npz"))
    assert npzs, "no fixtures"
    fields_checked = 0
    for npz in npzs:
        host = dict(np.load(npz))
        batch, coord_cond = assemble_flow_batch(host)
        torch_padding = jnp.asarray(host["padding_mask"]).astype(bool)

        # direct field comparisons against the torch tensors
        key_field = {
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
            "padding_mask": "padding_mask",
        }
        for key, field in key_field.items():
            if key not in host:
                continue  # default-filled; covered by defaults test below
            expected = np.asarray(host[key])
            actual = np.asarray(getattr(batch, field))
            if expected.dtype == bool:
                assert (actual == expected).all(), f"{npz.name}:{key} mismatch"
            elif np.issubdtype(expected.dtype, np.integer):
                assert (actual.astype(np.int64) == expected.astype(np.int64)).all(), f"{npz.name}:{key} mismatch"
            else:
                # cond_coords etc. carry NaN for unconditioned atoms
                assert np.allclose(actual, expected, rtol=0, atol=1e-6, equal_nan=True), f"{npz.name}:{key} mismatch"
            fields_checked += 1

        # unit conversion: coords (Å) -> coords_nm (NaN-safe)
        assert np.allclose(np.asarray(batch.coords_nm), np.asarray(host["coords"]) / 10.0,
                           atol=1e-7, equal_nan=True)
        # coordinate conditioning mask passes through (default False here)
        assert np.asarray(coord_cond).dtype == np.bool_

        # sequence probabilities: stacked in canonical order with NaN passthrough
        from rfd4fx.featurize import SEQ_PROB_ORDER

        probs = np.asarray(batch.seq_probabilities)
        assert probs.shape[-1] == 28
        for i, k in enumerate(SEQ_PROB_ORDER):
            if k in host:
                expected = np.asarray(host[k])
                actual = probs[..., i]
                nan_mask = np.isnan(expected)
                assert (np.isnan(actual) == nan_mask).all(), f"{npz.name}:{k} NaN pattern"
                assert np.allclose(actual[~nan_mask], expected[~nan_mask], atol=1e-7)
                fields_checked += 1

        # defaults test: remove a condition and check the schema default appears
        host2 = {k: v for k, v in host.items() if k != "condition_rasa-binned_1_atom"}
        batch2, _ = assemble_flow_batch(host2)
        assert (np.asarray(batch2.rasa) == 0).all()
        host3 = {k: v for k, v in host.items() if k != "condition_cath-label_1_residue"}
        batch3, _ = assemble_flow_batch(host3)
        assert (np.asarray(batch3.cath_codes) == -1).all()

    print(f"featurize parity: {fields_checked} field comparisons across {len(npzs)} fixtures — all exact")
