"""Verify `annotated_cif_to_batch` against the ORIGINAL torch featurization.

For each example annotated CIF from the RFD4-Proteina repository (benchmarks +
tests/data), runs:
  (a) the torch reference: parse(ANNOTATED_CIF) -> build_design_validation_pipeline
      -> collate_batch (the exact inference featurization + collation), and
  (b) the ported jnp path: `rfd4fx.annotated_cif_to_batch` (torch-free),
and compares every model-relevant field (ints/bools exact, floats allclose with
NaN-pattern preservation, sequence-probability stacking in canonical order).

Both sides draw chain_iid/molecule_iid permutations from numpy's global RNG, so
both are seeded identically (np_seed=0) before the featurization pipeline.

Usage:
    python scripts/verify_annotated_cif_parity.py [file.cif ...]
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rfd4_upstream import ensure_benchmarks, ensure_upstream_tests_data

BENCH = ensure_benchmarks()

DEFAULT_FILES = [
    BENCH / "monomer/monomer-short.cif.zst",                       # de-novo expseg
    BENCH / "motif_scaffolding/motif_indexed/5TPN_AA.cif.zst",     # C_CRD/C_IDX motif conditioning
    BENCH / "small_molecule_binder/SAM_hbonds.cif.zst",            # ligand + C_HB
    BENCH / "adherence/interchain_interaction_conditioning/vitamin_b12_metalcoord_metal.cif.zst",  # C_MCOORD
    BENCH / "adherence/interchain_interaction_conditioning/sam_pistack_aromatic_ring.cif.zst",     # C_PISTACK/pication
    BENCH / "sequence_bias/v11_fixed225.cif.zst",                  # fixed sequence
    BENCH / "mcsa/mcsa_37_no_ligand/M0129_1os7.cif.zst",           # active-site motif
    BENCH / "ppi/ppi_all_atom/01_PD1.cif.zst",                     # multi-chain complex
]


def all_example_cifs() -> list[pathlib.Path]:
    """Every example annotated CIF in the RFD4 repository (benchmarks + tests/data)."""
    files: list[pathlib.Path] = []
    base = BENCH
    for pat in ("*.cif.zst", "*.cif.gz", "*.cif"):
        files.extend(sorted(base.rglob(pat)))
    files.extend(sorted(ensure_upstream_tests_data().glob("*.cif")))
    return files

KEY_FIELD = {
    "atom37_mask": "atom37_mask", "condition_coordinate_1_atom": "cond_coords",
    "condition_InternalCoordinate_1_atom": "cond_internal_coords",
    "condition_CoordinateGroup_1_atom": "coord_groups",
    "condition_rasa-binned_1_atom": "rasa",
    "condition_secondary-structure_1_residue": "secondary_structure",
    "condition_hbond_1_atom": "hbond", "condition_pistack_1_atom": "pistack",
    "condition_pication_1_atom": "pication", "condition_saltbridge_1_atom": "saltbridge",
    "condition_hydrophobic_1_atom": "hydrophobic", "condition_metal-coord_1_atom": "metal_coord",
    "condition_hotspot_1_atom": "hotspot", "condition_cath-label_1_residue": "cath_codes",
    "condition_index-group_1_token": "idx_group", "condition_ChainCentroid_1_atom": "chain_centroid",
    "condition_sequence_1_residue": "cond_seq", "token_type": "token_type",
    "chain_iid": "chain_iid", "molecule_iid": "molecule_iid",
    "molecule_entity_id": "molecule_entity_id", "token_bond_path_length": "token_bond_path_length",
    "bond_mask": "bond_mask", "bond_order": "bond_order", "is_aromatic": "is_aromatic",
    "canonical_polymer_ordering": "canonical_polymer_ordering", "na_ss_state": "na_ss_state",
    "na_ss_state_mask": "na_ss_state_mask", "na_ss_pair": "na_ss_pair",
    "na_ss_pair_mask": "na_ss_pair_mask", "charge": "charge", "element": "element",
    "ground_truth_element": "ground_truth_element",
    "incident_bond_type_counts": "incident_bond_type_counts", "chirality": "chirality",
    "chiral_frame_atoms": "chiral_frame_atoms", "chiral_slot_owner": "chiral_slot_owner",
    "is_in_context_conditioning": "is_in_context_conditioning",
    "supports_local_latent": "supports_local_latent", "padding_mask": "padding_mask",
}


def torch_reference(path: pathlib.Path) -> dict[str, np.ndarray]:
    import torch
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import DesignProblemDataset

    np.random.seed(0)
    dataset = DesignProblemDataset(path, transform=build_design_validation_pipeline(), num_replicates=1)
    sample = dataset[0]
    collated = collate_batch(
        [sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    )
    inner = collated.data_samples[0] if isinstance(collated.data_samples, list) else collated.data_samples
    tensors = {k: v.detach().cpu().numpy() for k, v in inner.tensors.items() if isinstance(v, torch.Tensor)}
    tensors["padding_mask"] = collated.padding_masks["I"].detach().cpu().numpy()
    return tensors


def compare(torch_t: dict[str, np.ndarray], batch, coord_cond) -> tuple[int, list[str]]:
    from rfd4fx.featurize import SEQ_PROB_ORDER

    errors: list[str] = []
    checked = 0

    def check(key, expected, actual):
        nonlocal checked
        expected, actual = np.asarray(expected), np.asarray(actual)
        if expected.shape != actual.shape:
            errors.append(f"{key}: SHAPE torch {expected.shape} vs jax {actual.shape}")
            return
        if expected.dtype == bool:
            if not (expected == actual).all():
                errors.append(f"{key}: bool mismatch {(expected != actual).sum()}")
        elif np.issubdtype(expected.dtype, np.integer) or np.issubdtype(actual.dtype, np.integer):
            if not (expected.astype(np.int64) == actual.astype(np.int64)).all():
                errors.append(f"{key}: int mismatch")
        else:
            nan_diff = int((np.isnan(expected) != np.isnan(actual)).sum())
            d = np.nanmax(np.abs(expected.astype(np.float64) - actual.astype(np.float64))) if expected.size else 0.0
            if nan_diff or d > 1e-6:
                errors.append(f"{key}: max|d|={d:.3g} nan-diff={nan_diff}")
        checked += 1

    for key, field in KEY_FIELD.items():
        if key not in torch_t:
            continue  # condition absent in this file; assembler default verified separately
        check(key, torch_t[key], getattr(batch, field))
    if "coord_cond_mask" in torch_t:
        check("coord_cond_mask", torch_t["coord_cond_mask"], coord_cond)
    # coords: compare the model-facing nm quantity (avoids the float32 Å->nm->Å
    # double-rounding of the round trip, which is ~2e-6 on large coordinates)
    check("coords (Å/10 nm)", torch_t["coords"] / 10.0, np.asarray(batch.coords_nm))
    # sequence-probability channels in canonical order
    probs = np.asarray(batch.seq_probabilities)
    for i, key in enumerate(SEQ_PROB_ORDER):
        if key in torch_t:
            expected = np.asarray(torch_t[key])
            actual = probs[..., i]
            nan_mask = np.isnan(expected)
            if not (np.isnan(actual) == nan_mask).all():
                errors.append(f"{key}: NaN pattern")
            elif not np.allclose(actual[~nan_mask], expected[~nan_mask], atol=1e-6):
                errors.append(f"{key}: value mismatch")
            checked += 1
    return checked, errors


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        files = all_example_cifs()
        print(f"sweeping ALL example annotated CIFs in the repository: {len(files)} files")
    else:
        files = [pathlib.Path(p) for p in args] or DEFAULT_FILES
    import rfd4fx.annotated_cif as ac

    total_fields = 0
    failures = 0
    for path in files:
        if not path.exists():
            print(f"SKIP (missing): {path}")
            continue
        name = "/".join(path.parts[-2:])
        try:
            batch, coord_cond = ac.annotated_cif_to_batch(path, np_seed=0)
            torch_t = torch_reference(path)
        except Exception as e:  # noqa: BLE001
            print(f"ERROR  {name}: {type(e).__name__}: {e}")
            failures += 1
            continue
        checked, errors = compare(torch_t, batch, coord_cond)
        total_fields += checked
        if errors:
            failures += 1
            print(f"FAIL   {name}: {checked} fields checked, {len(errors)} errors")
            for e in errors[:8]:
                print(f"         {e}")
        else:
            n = int(np.asarray(batch.padding_mask).sum())
            print(f"PASS   {name}: {checked} fields exact ({n} tokens)")

    print(f"\n{total_fields} field comparisons across {len(files)} files, "
          f"{failures} file(s) with errors")


if __name__ == "__main__":
    main()
