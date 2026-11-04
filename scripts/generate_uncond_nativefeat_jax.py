"""Unconditional generation driven by the ORIGINAL RFD4 torch input features.

Unlike `generate_uncond_jax.py` (which hand-builds the unconditioned batch via
the ported `make_unconditioned_batch`), this script:

1. Runs the **native torch featurization pipeline**
   (`build_design_validation_pipeline` + `DesignProblemDataset` + `collate_batch`)
   on the de-novo sentinel CIF (expseg 50) — exactly what RFD4 inference consumes.
2. Saves those tensors verbatim to `test-featuresets/rfd4_native.npz`.
3. Constructs the equivalent feature set with the **ported pipeline**
   (`make_unconditioned_batch`) and saves it to `test-featuresets/rfd4_port.npz`
   (same torch-style keys, so the two archives are directly comparable).
4. Compares the two archives field-by-field (dtype/NaN-aware) and reports.
5. Feeds the **native** torch features (`assemble_flow_batch`) into the JAX
   model and generates 5 proteins × 50 residues, writing validated PDBs.

Example:
    python scripts/generate_uncond_nativefeat_jax.py --n-residues 50 \
        --num-designs 5 --out-dir /Users/mjendrusch/repos/rfd4j/test-featuresets
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path("/Users/mjendrusch/repos/rfd4j")
SNAPSHOT = REPO / ".hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502"

# Sentinel CIF template (the exact pattern of the torch uncond-64 reference run,
# expanded to the requested chain length by string substitution).
SENTINEL_TEMPLATE = """data_uncond{n}
#
_entry.id       monomer-short.cif
_entry.author   rib7
_entry.date     2026-08-28
_entry.time     19:44:50
#
_atom_site.group_PDB                         HETATM
_atom_site.type_symbol                       X
_atom_site.label_atom_id                     UNK
_atom_site.label_alt_id                      .
_atom_site.label_comp_id                     {n}-{n}
_atom_site.label_asym_id                     A
_atom_site.label_entity_id                   0
_atom_site.label_seq_id                      1
_atom_site.pdbx_PDB_ins_code                 .
_atom_site.auth_seq_id                       1
_atom_site.auth_comp_id                      {n}-{n}
_atom_site.auth_asym_id                      A
_atom_site.auth_atom_id                      UNK
_atom_site.B_iso_or_equiv                    nan
_atom_site.occupancy                         0.0
_atom_site.pdbx_formal_charge                ?
_atom_site.Cartn_x                           nan
_atom_site.Cartn_y                           nan
_atom_site.Cartn_z                           nan
_atom_site.pdbx_PDB_model_num                1
_atom_site.id                                1
_atom_site.annotation_expsegmin_1_residue    {n}
_atom_site.annotation_expsegmax_1_residue    {n}
_atom_site.condition_sequence_1_residue      <M>
_atom_site.condition_index_1_residue         True
_atom_site.annotation_src-component_1_atom   .
_atom_site.condition_hbond_1_atom            0
_atom_site.condition_index-group_1_token     0
_atom_site.atom_array_index                  0
#
_condition_coordinate_1_atom.idx0   0
_condition_coordinate_1_atom.val0   nan
_condition_coordinate_1_atom.val1   nan
_condition_coordinate_1_atom.val2   nan
#
_entity.id                 0
_entity.type               polymer
_entity.pdbx_description   .
#
_entity_poly.entity_id                      0
_entity_poly.type                           polypeptide(L)
_entity_poly.nstd_linkage                   no
_entity_poly.nstd_monomer                   yes
_entity_poly.pdbx_seq_one_letter_code       X
_entity_poly.pdbx_seq_one_letter_code_can   X
_entity_poly.pdbx_strand_id                 A
_entity_poly.pdbx_target_identifier         ?
#
_entity_poly_seq.entity_id   0
_entity_poly_seq.hetero      n
_entity_poly_seq.mon_id      {n}-{n}
_entity_poly_seq.num         1
#
_chem_comp.id                        {n}-{n}
_chem_comp.name                      {n}-{n}
_chem_comp.type                      NON-POLYMER
_chem_comp.formula                   ?
_chem_comp.mon_nstd_parent_comp_id   ?
_chem_comp.pdbx_synonyms             ?
_chem_comp.formula_weight            ?
#
_rfd_design.sampled_contig   {n}-{n}
#
_chem_comp_atom.comp_id                    {n}-{n}
_chem_comp_atom.atom_id                    UNK
_chem_comp_atom.type_symbol                X
_chem_comp_atom.charge                     0
_chem_comp_atom.pdbx_aromatic_flag         N
_chem_comp_atom.pdbx_leaving_atom_flag     N
_chem_comp_atom.pdbx_model_Cartn_x_ideal   nan
_chem_comp_atom.pdbx_model_Cartn_y_ideal   nan
_chem_comp_atom.pdbx_model_Cartn_z_ideal   nan
#
"""


def native_torch_features(n_residues: int) -> dict[str, np.ndarray]:
    """Run the REAL torch inference featurization on the de-novo sentinel CIF."""
    sys.path.insert(0, str(REPO / "RFD4-Proteina/src"))
    import torch
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import DesignProblemDataset

    import tempfile

    cif = pathlib.Path(tempfile.gettempdir()) / f"uncond-{n_residues}.cif"
    cif.write_text(SENTINEL_TEMPLATE.format(n=n_residues))

    pipeline = build_design_validation_pipeline()
    dataset = DesignProblemDataset(cif, transform=pipeline, num_replicates=1)
    sample = dataset[0]
    collated = collate_batch(
        [sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample"
    )
    inner = (
        collated.data_samples[0]
        if isinstance(collated.data_samples, list)
        else collated.data_samples
    )
    tensors: dict[str, np.ndarray] = {}
    for key, value in inner.tensors.items():
        if isinstance(value, torch.Tensor):
            tensors[key] = value.detach().cpu().numpy()
    # padding lives in PaddedBatchedDataSample.padding_masks, keyed by dim name
    tensors["padding_mask"] = collated.padding_masks["I"].detach().cpu().numpy()
    return tensors


def ported_features(n_residues: int) -> dict[str, np.ndarray]:
    """The same feature set from the PORTED pipeline (make_unconditioned_batch),
    expressed under torch-style keys so the archives align key-for-key."""
    sys.path.insert(0, str(REPO / "rfd4fx/src"))
    from rfd4fx.featurize import (
        SEQ_PROB_ORDER,
        _KEY_TO_FIELD,
        make_unconditioned_batch,
    )

    _, fb = make_unconditioned_batch(n_residues, b=1)
    host: dict[str, np.ndarray] = {
        "padding_mask": np.asarray(fb.padding_mask),
        # torch carries coordinates in Å; the port keeps nm (model-side units)
        "coords": np.asarray(fb.coords_nm) * 10.0,
        # sampler's stochastic-centering mask (all-False: no coordinate conditioning)
        "coord_cond_mask": np.zeros((1, n_residues), dtype=bool),
    }
    for key, field in _KEY_TO_FIELD.items():
        value = getattr(fb, field)
        if value is None:
            # None -> the schema default the model actually consumes
            # (chirality zero / chiral frames -1; mirrors assemble_flow_batch defaults)
            from rfd4fx.featurize import _DEFAULTS

            b, n = host["padding_mask"].shape
            value = _DEFAULTS[key](b, n)
        host[key] = np.asarray(value)
    # unstack the sequence-probability channels into their torch keys
    probs = np.asarray(fb.seq_probabilities)
    for i, key in enumerate(SEQ_PROB_ORDER):
        host[key] = probs[..., i]
    return host


def compare(native: dict[str, np.ndarray], port: dict[str, np.ndarray]) -> list[tuple]:
    """Field-by-field comparison; returns rows of (key, status, detail).

    Native-only keys that are NOT consumed at inference are reported as INERT
    (with the reason) rather than failures:
    - is_resolved_mask / loss_weight / residue_type / insertion_source_idx:
      training supervision only (model_masks.py supervision masks, CE target,
      loss weighting, insertion bookkeeping).
    - tetrahedral_vertices: signed-volume aux loss only
      (product_space_flow_matcher.compute_aux_loss).
    - tok_idx: atom->token bookkeeping for the data pipeline.
    - condition_index_1_residue: CIF annotation passthrough, unread by nn/sampler.
    - centroid_of_atoms_without_coordinate_conditioning: read only when coordinate
      conditioning exists; all-zeros here (and equals the port default).
    """
    inert = {
        "is_resolved_mask": "training supervision mask only (model_masks.compute_*_supervision_masks)",
        "loss_weight": "training loss weighting",
        "residue_type": "training CE target",
        "insertion_source_idx": "insertion bookkeeping (training)",
        "tetrahedral_vertices": "signed-volume aux loss only (compute_aux_loss)",
        "tok_idx": "atom->token bookkeeping",
        "condition_index_1_residue": "CIF annotation passthrough, unread by nn/sampler",
        "centroid_of_atoms_without_coordinate_conditioning": "coordinate-conditioning centering only; zeros here",
    }
    rows = []
    for key in sorted(native):
        t = np.asarray(native[key])
        if key not in port:
            rows.append((key, "PORT-MISSING", "") if key not in inert else (key, "INERT", inert[key]))
            continue
        j = np.asarray(port[key])
        if t.shape != j.shape:
            rows.append((key, "SHAPE", f"torch {t.shape} vs port {j.shape}"))
            continue
        if t.dtype == bool or (t.dtype == object and t.size == 0):
            bad = int((t.astype(bool) != j.astype(bool)).sum())
            rows.append((key, "OK" if bad == 0 else "DIFF", f"{bad} bool mismatches"))
            continue
        tf, jf = t.astype(np.float64), j.astype(np.float64)
        nan_diff = int((np.isnan(tf) != np.isnan(jf)).sum())
        d = float(np.nanmax(np.abs(tf - jf))) if t.size and not np.isnan(tf).all() else 0.0
        ok = d == 0.0 and nan_diff == 0
        rows.append((key, "OK" if ok else "DIFF", f"max|d|={d:.4g} nan-diff={nan_diff}"))
    for key in sorted(port):
        if key not in native:
            rows.append((key, "NATIVE-MISSING", ""))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-residues", type=int, default=50)
    parser.add_argument("--num-designs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--out-dir", type=pathlib.Path, default=REPO / "test-featuresets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=SNAPSHOT)
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] native torch featurization (sentinel expseg {args.n_residues}) ...", flush=True)
    native = native_torch_features(args.n_residues)
    np.savez_compressed(args.out_dir / "rfd4_native.npz", **native)
    print(f"      {len(native)} tensors -> {args.out_dir / 'rfd4_native.npz'}")

    print("[2/4] ported featurization (make_unconditioned_batch) ...", flush=True)
    port = ported_features(args.n_residues)
    np.savez_compressed(args.out_dir / "rfd4_port.npz", **port)
    print(f"      {len(port)} tensors -> {args.out_dir / 'rfd4_port.npz'}")

    print("[3/4] field-by-field comparison ...", flush=True)
    rows = compare(native, port)
    n_ok = sum(1 for _, s, _ in rows if s == "OK")
    for key, status, detail in rows:
        marker = "" if status in ("OK", "INERT") else "   <<<"
        print(f"  {status:>14s}  {key:55s} {detail}{marker}")
    print(f"      {n_ok}/{len(rows)} fields exact ({len(rows) - n_ok} inert native-only)")
    bad = [r for r in rows if r[1] not in ("OK", "INERT")]
    if bad:
        raise SystemExit(f"feature comparison FAILED for {len(bad)} fields: {[r[0] for r in bad]}")

    if args.skip_generation:
        return

    print("[4/4] JAX generation from NATIVE torch features ...", flush=True)
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import torch
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.featurize import assemble_flow_batch
    from rfd4fx.model import RFD4

    torch.manual_seed(args.seed)
    print("      loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint)
    model.eval()
    jax_model = RFD4.from_torch(model)

    # native torch tensors -> FlowBatch (the parity-verified assembly path)
    batch1, coord_cond = assemble_flow_batch(native)
    assert not np.asarray(coord_cond).any(), "unexpected coordinate conditioning"
    # replicate the single native example across the design batch
    b = args.batch_size
    assert args.num_designs == b, "this script generates one batch of --num-designs"
    batch = jax.tree.map(
        lambda x: jnp.repeat(x, b, axis=0) if hasattr(x, "shape") and x.ndim >= 1 else x,
        batch1,
        is_leaf=lambda l: l is None,
    )

    designs_dir = args.out_dir / "designs_nativefeat"
    designs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha": "a06e2509c7033b010fc43614e9fd840303588502",
        "featurization": "native torch pipeline (build_design_validation_pipeline) -> assemble_flow_batch",
        "n_residues": args.n_residues,
        "num_designs": args.num_designs,
        "seed": args.seed,
        "nsteps": args.nsteps,
        "backend": "jax (rfd4fx)",
        "designs": [],
    }
    t0 = time.time()
    key = jax.random.fold_in(jax.random.key(args.seed), 0)
    x, x_1_pred, dec = jax_model.generate(batch, key=key, nsteps=args.nsteps)

    coors_a = np.asarray(dec.coors_nm) * 10.0  # nm -> Å
    aatype = np.asarray(dec.aatype_max)
    atom_mask = np.asarray(dec.atom_mask)
    emit = np.asarray(dec.emit_mask)
    for j in range(b):
        seq = aatype[j]
        mask_j = atom_mask[j] & emit[j][:, None]
        coords37 = coors_a[j].copy()
        coords37[~mask_j] = np.nan
        res_type = np.where(emit[j], seq, -1).astype(np.int64)
        arr = atom_array_from_encoding(
            encoded_coord=coords37,
            encoded_mask=mask_j,
            encoded_seq=res_type,
            encoding=UNIFIED_ATOM37_ENCODING,
            chain_id="A",
        )
        arr = arr[np.isfinite(arr.coord).all(axis=-1)]
        out = designs_dir / f"design_{j:03d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))
        back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
        ca = back[back.atom_name == "CA"]
        assert len(ca) == args.n_residues, f"design {j}: {len(ca)} CA atoms"
        assert bool(np.isfinite(back.coord).all()), f"design {j}: non-finite"
        d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=-1)
        manifest["designs"].append(
            {
                "id": f"design_{j:03d}",
                "seed": args.seed,
                "file": str(out),
                "n_atoms": int(back.array_length()),
                "ca_ca_mean": float(d.mean()),
                "ca_ca_max": float(d.max()),
            }
        )
        print(
            f"      design_{j:03d}.pdb: {back.array_length()} atoms, "
            f"CA-CA {d.mean():.2f}±{d.std():.2f} Å (max {d.max():.2f})  [{time.time() - t0:.0f}s]",
            flush=True,
        )
    (args.out_dir / "manifest_nativefeat.json").write_text(json.dumps(manifest, indent=1))
    print(
        f"done: {args.num_designs} designs in {time.time() - t0:.0f}s -> {designs_dir}"
    )


if __name__ == "__main__":
    main()
