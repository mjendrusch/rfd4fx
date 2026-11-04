"""Ligand + protein co-design sampling via `RFD4Input.from_chains` + the JAX port.

Builds a compiled input with one de-novo protein chain (DSSP-optional) plus one
or more ligands (SMILES via RDKit or CCD id via the gemmi-cached chemical
component dictionary — topology-only: elements, formal charges and bond features
condition the model, ligand positions are diffused design targets; `--ligand-pos`
optionally coordinate-conditions the ligand instead), samples with the JAX flow
model + autoencoder decode, and writes PDBs (protein chain A, ligands as HETATM
chains B, C, ...).

Example:
    python scripts/generate_ligand_jax.py --protein-len 72 \
        --smiles "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C" \
        --out-dir /Users/mjendrusch/repos/rfd4j/rfd4-test-ligands
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

REPO = pathlib.Path("/Users/mjendrusch/repos/rfd4j")
SNAPSHOT = REPO / ".hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502"

DFHBI_SMILES = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C"  # (Z)-DFHBI, PubChem CID 70808995


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-designs", type=int, default=5)
    parser.add_argument("--protein-len", type=int, default=72)
    parser.add_argument("--dssp", type=str, default=None,
                        help="optional DSSP condition for the protein chain")
    parser.add_argument("--smiles", type=str, default=None)
    parser.add_argument("--ccd", type=str, default=None)
    parser.add_argument("--ligand-pos", type=str, default=None,
                        help="optional path to a (num_atoms, 3) .npy in Å; "
                             "coordinate-conditions the ligand")
    parser.add_argument("--ligand-rasa", type=int, default=None,
                        help="optional binned-RASA value (1=buried, 2=intermediate, "
                             "3=exposed) applied to every ligand atom slot via "
                             "RFD4Input.add_rasa")
    parser.add_argument("--pad-to", type=int, default=None,
                        help="pad the compiled input to this TOTAL token count "
                             "(inert padded slots; never emitted into the PDBs)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=SNAPSHOT)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO / "rfd4-test-ligands")
    args = parser.parse_args()
    if (args.smiles is None) == (args.ccd is None):
        parser.error("pass exactly one of --smiles / --ccd")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.model import RFD4
    from rfd4fx.rfd4_input import RFD4Input
    from rfd4fx.spec import _atomic_number_from_symbol

    torch.manual_seed(args.seed)
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint)
    model.eval()
    jax_model = RFD4.from_torch(model)

    # ligand atom identities for the PDB output (element symbols; names per
    # element in RDKit/CCD atom order)
    ligand = {"smiles": args.smiles, "ccd": args.ccd}
    if args.ligand_pos is not None:
        ligand["ligand_pos"] = np.load(args.ligand_pos).astype(np.float32)
    probe = RFD4Input.from_chains(1, dict(ligand), np_seed=0)
    from rfd4fx.spec import RFD4Spec
    spec_probe = RFD4Spec()
    spec_probe.add_ligand(smiles=args.smiles, ccd=args.ccd)
    lig = spec_probe.ligands[0]
    n_lig_atoms = int(lig.elements.shape[0])
    Z_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S"}
    lig_syms = [Z_TO_SYM.get(int(z), str(z)) for z in lig.elements]
    counts: dict[str, int] = {}
    lig_names = []
    for sym in lig_syms:
        counts[sym] = counts.get(sym, 0) + 1
        lig_names.append(f"{sym}{counts[sym]}")
    print(f"ligand: {n_lig_atoms} atoms ({lig.label})", flush=True)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "featurization": "RFD4Input.from_chains -> direct featurizer (to_batch)",
        "seed": args.seed, "nsteps": args.nsteps,
        "backend": "jax (rfd4fx)",
        "protein_len": args.protein_len, "dssp": args.dssp,
        "ligand": {"smiles": args.smiles, "ccd": args.ccd,
                   "n_atoms": n_lig_atoms,
                   "ligand_pos": args.ligand_pos,
                   "ligand_rasa": args.ligand_rasa,
                   "atom_names": lig_names},
        "designs": [],
    }
    out_manifest = args.out_dir / "manifest_ligand.json"
    if out_manifest.exists():
        try:
            prev = json.loads(out_manifest.read_text())
            manifest["designs"].extend(prev.get("designs", []))
        except Exception:
            pass
    base_idx = len(manifest["designs"])

    # ---- compile the design input: protein chain + ligand
    chain_desc = {"num_residues": args.protein_len}
    if args.dssp is not None:
        chain_desc["dssp"] = args.dssp
    inp = RFD4Input.from_chains(args.protein_len if args.dssp is None else chain_desc,
                                dict(ligand), np_seed=args.seed, pad_to=args.pad_to)
    n_total = int(np.asarray(inp.padding_mask).sum())
    assert n_total == args.protein_len + n_lig_atoms
    if args.ligand_rasa is not None:
        # binned-RASA condition on every ligand atom slot (C_RAS = 1 → buried),
        # edited directly on the compiled input via the RFD4Input adder
        where = np.zeros((n_lig_atoms, 37), dtype=bool)
        where[:, 1] = True  # the atomized tokens' single atom slot
        inp = inp.add_rasa(int(args.ligand_rasa), aa=None, where=where,
                           start=args.protein_len)
        rasa = np.asarray(inp.rasa)[0]
        assert (rasa[args.protein_len:, 1] == int(args.ligand_rasa)).all(), \
            "ligand RASA condition did not land on the ligand slots"
        assert (rasa[:args.protein_len] == 0).all(), "protein RASA unexpectedly set"
        print(f"ligand RASA condition: all {n_lig_atoms} atoms -> "
              f"{args.ligand_rasa} (buried)", flush=True)
    batch1 = inp.to_batch()
    coord_cond = inp.coord_cond
    print(f"compiled: {n_total} tokens "
          f"({args.protein_len} protein + {n_lig_atoms} ligand), "
          f"coord-conditioned tokens: {int(np.asarray(coord_cond).sum())}", flush=True)

    b = args.num_designs
    batch = jax.tree.map(
        lambda x: jnp.repeat(x, b, axis=0) if hasattr(x, "shape") and x.ndim >= 1 else x,
        batch1, is_leaf=lambda l: l is None)

    t_all = time.time()
    t0 = t_all
    key = jax.random.fold_in(jax.random.key(args.seed), 12345)
    x, x_1_pred, dec = jax_model.generate(batch, key=key, nsteps=args.nsteps)

    coors_a = np.asarray(dec.coors_nm) * 10.0  # nm -> Å
    # atomized tokens are never AE-emitted (supports_local_latent=False ⇒ emit=0);
    # their coordinates are the flow's per-token positions (x["bb_ca"], the same
    # quantity the AE's decoded CA agrees with on residue tokens)
    x_bb = np.asarray(x["bb_ca"]) * 10.0  # (b, n, 3) Å
    aatype = np.asarray(dec.aatype_max)
    atom_mask = np.asarray(dec.atom_mask)
    emit = np.asarray(dec.emit_mask)
    # padding-aware token selection: padded slots (padding_mask False) are inert
    # (emit=0) but their flow-state rows still exist — never write them to PDBs
    real_tok = np.asarray(batch1.padding_mask)[0]
    n_width = real_tok.shape[0]  # padded width (>= real tokens)
    tok_is_ligand = np.zeros(n_width, dtype=bool)
    tok_is_ligand[args.protein_len:] = True
    tok_is_ligand &= real_tok
    tok_is_protein = real_tok & ~tok_is_ligand
    assert int(tok_is_ligand.sum()) == n_lig_atoms
    enc = UNIFIED_ATOM37_ENCODING

    for j in range(b):
        mask_j = atom_mask[j] & emit[j][:, None]
        coords37 = coors_a[j].copy()
        coords37[~mask_j] = np.nan
        res_type = np.where(emit[j], aatype[j], -1).astype(np.int64)
        # protein chain A (residue tokens; padded tokens excluded)
        arr_p = atom_array_from_encoding(
            encoded_coord=coords37[tok_is_protein],
            encoded_mask=mask_j[tok_is_protein],
            encoded_seq=res_type[tok_is_protein],
            encoding=enc,
            chain_id="A",
        )
        arr_p = arr_p[np.isfinite(arr_p.coord).all(axis=-1)]
        # ligand chain B: one atom per token — coordinates from the flow state
        lig_coords = x_bb[j][tok_is_ligand]
        assert np.isfinite(lig_coords).all(), f"design {j}: non-finite ligand"
        lig_rows = []
        for k in range(n_lig_atoms):
            lig_rows.append({
                "chain_id": "B", "res_id": 1, "res_name": lig.res_name,
                "atom_name": lig_names[k],
                "element": lig_syms[k],
                "coord": lig_coords[k], "hetero": True,
                "b_factor": 0.0, "occupancy": 1.0,
            })
        import biotite.structure as bs
        n_lig = len(lig_rows)
        lig_arr = bs.AtomArray(n_lig)
        lig_arr.add_annotation("hetero", dtype=bool)
        lig_arr.add_annotation("occupancy", dtype=float)
        lig_arr.add_annotation("b_factor", dtype=float)
        for k, row in enumerate(lig_rows):
            lig_arr.coord[k] = row["coord"]
            lig_arr.chain_id[k] = row["chain_id"]
            lig_arr.res_id[k] = row["res_id"]
            lig_arr.res_name[k] = row["res_name"]
            lig_arr.atom_name[k] = row["atom_name"]
            lig_arr.element[k] = row["element"]
            lig_arr.hetero[k] = True
            lig_arr.occupancy[k] = 1.0
            lig_arr.b_factor[k] = 0.0
        arr = arr_p + lig_arr if n_lig else arr_p

        out = args.out_dir / f"design_{base_idx + j:03d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))
        back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
        ca = back[back.atom_name == "CA"]
        lig_back = back[back.hetero]
        assert len(ca) == args.protein_len, f"design {j}: {len(ca)} CA atoms"
        assert len(lig_back) == n_lig, f"design {j}: {len(lig_back)} ligand atoms"
        assert bool(np.isfinite(back.coord).all()), f"design {j}: non-finite"
        d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=-1)
        # protein-ligand contact readout
        d_pl = np.linalg.norm(ca.coord[:, None] - lig_back.coord[None], axis=-1)
        # ligand internal geometry: bonded-pair distances (topology from the spec)
        lig_xy = lig_back.coord
        bl = np.linalg.norm(lig_xy[lig.bonds[:, 0]] - lig_xy[lig.bonds[:, 1]], axis=-1)
        run = {
            "id": f"design_{base_idx + j:03d}",
            "file": str(out),
            "n_atoms": int(back.array_length()),
            "n_protein_atoms": int((~back.hetero).sum()),
            "n_ligand_atoms": int(back.hetero.sum()),
            "ca_ca_mean": float(d.mean()), "ca_ca_max": float(d.max()),
            "ligand_bond_length_mean": float(bl.mean()),
            "ligand_bond_length_min": float(bl.min()),
            "ligand_bond_length_max": float(bl.max()),
            "min_protein_ligand_ca_distance": float(d_pl.min()),
            "protein_ligand_ca_pairs_within_10A": int((d_pl < 10).sum()),
        }
        manifest["designs"].append(run)
        print(
            f"      design_{base_idx + j:03d}.pdb: {back.array_length()} atoms "
            f"({len(lig_back)} ligand), CA-CA {d.mean():.2f}±{d.std():.2f} Å "
            f"(max {d.max():.2f}), ligand bonds {bl.mean():.2f} "
            f"[{bl.min():.2f}-{bl.max():.2f}] Å, min prot-lig CA {d_pl.min():.1f} Å, "
            f"{int((d_pl < 10).sum())} CA pairs < 10 Å  [{time.time() - t0:.0f}s]",
            flush=True,
        )
    out_manifest.write_text(json.dumps(manifest, indent=1))
    print(f"done: {b} designs in {time.time() - t_all:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
