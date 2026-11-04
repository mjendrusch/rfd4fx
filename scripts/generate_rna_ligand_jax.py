"""RNA + ligand co-design sampling via RFD4Input / add_<condition> + RFD4Sampler.

Follows generate_ligand_jax.py: one de-novo nucleic-acid chain (RNA by
default; --kind selects rna/dna) plus one ligand (SMILES via RDKit or CCD id
via the gemmi cache — topology-only: elements, formal charges and bond
features condition the model, ligand positions are diffused design targets).
Conditions are applied through the RFD4Input adder API: an optional
dot-bracket NA secondary structure via add_bp (C_NA_SS_STATE + C_BP) and an
optional binned-RASA value on the ligand atoms via add_rasa (C_RAS).
Sampling uses the RFD4/RFD4Sampler API (jitted step + decode, THINKING_HIGH
recycle schedule); PDBs are written through the shared native-parity export
helper (rfd4fx.pdb_io.decoded_to_atom_array), with the ligand appended
as a HETATM chain from the flow state.

Example:
    python scripts/generate_rna_ligand_jax.py --num-bases 40 \
        --smiles "Cn1c(=O)c2c(ncn2C)n(C)c1=O" --num-designs 5 \
        --out-dir /Users/mjendrusch/repos/rfd4j/rfd4-test-rna-ligands
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

REPO = pathlib.Path("/Users/mjendrusch/repos/rfd4j")
SNAPSHOT = (REPO / ".hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/"
            "a06e2509c7033b010fc43614e9fd840303588502")
CAFFEINE_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"  # caffeine, PubChem CID 2519


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-designs", type=int, default=5)
    parser.add_argument("--num-bases", type=int, default=40)
    parser.add_argument("--kind", default="rna", choices=["rna", "RNA", "dna", "DNA"],
                        help="polymer kind of the designed chain")
    parser.add_argument("--bp", type=str, default=None,
                        help="optional dot-bracket NA secondary structure "
                             "(. ( ) [ ] { } < > letters, x/_/X unknown); "
                             "applied via RFD4Input.add_bp")
    parser.add_argument("--smiles", type=str, default=CAFFEINE_SMILES)
    parser.add_argument("--ccd", type=str, default=None)
    parser.add_argument("--auto-condition", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="autofeaturize the ligand interaction conditions "
                             "(C_PISTACK on aromatic-ring atoms, C_HB donor/"
                             "acceptor, C_SALT charge sign) via from_chains "
                             "'auto_condition'")
    parser.add_argument("--buried", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="set C_RAS 1 (buried) on every ligand atom via "
                             "from_chains 'buried'")
    parser.add_argument("--ligand-rasa", type=int, default=None,
                        help="explicit binned-RASA value (1=buried, "
                             "2=intermediate, 3=exposed) on every ligand atom "
                             "slot via RFD4Input.add_rasa (conflicts with "
                             "--buried)")
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=SNAPSHOT)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO / "rfd4-test-rna-ligands")
    args = parser.parse_args()
    if args.buried and args.ligand_rasa is not None:
        parser.error("--buried and --ligand-rasa are mutually exclusive "
                     "(buried sets C_RAS 1 on every ligand atom)")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import torch
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
    from rfd4fx.model import RFD4
    from rfd4fx.pdb_io import decoded_to_atom_array
    from rfd4fx.rfd4_input import RFD4Input
    from rfd4fx.spec import RFD4Spec

    torch.manual_seed(args.seed)
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint)
    model.eval()
    jax_model = RFD4.from_torch(model)

    # ligand identities for the PDB output (element symbols; per-element numbering)
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
    roles = None  # autofeaturized roles, filled after compiling the input

    manifest = {
        "checkpoint": str(args.checkpoint),
        "featurization": "RFD4Input.from_chains -> direct featurizer (to_batch)",
        "sampler": "RFD4Sampler (jitted step + decode), THINKING_HIGH",
        "seed": args.seed, "nsteps": args.nsteps, "backend": "jax (rfd4fx)",
        "num_bases": args.num_bases, "kind": args.kind, "bp": args.bp,
        "auto_condition": args.auto_condition, "buried": args.buried,
        "ligand_roles": ({k: v.astype(int).tolist() for k, v in roles.items()}
                         if roles is not None else None),
        "ligand": {"smiles": args.smiles, "ccd": args.ccd,
                   "n_atoms": n_lig_atoms, "ligand_rasa": args.ligand_rasa,
                   "atom_names": lig_names},
        "designs": [],
    }
    out_manifest = args.out_dir / "manifest_rna_ligand.json"
    if out_manifest.exists():
        try:
            prev = json.loads(out_manifest.read_text())
            manifest["designs"].extend(prev.get("designs", []))
        except Exception:
            pass
    base_idx = len(manifest["designs"])

    # ---- compile the design input: NA chain + ligand (RFD4Input.from_chains)
    kind = args.kind.lower()
    chain_desc = {"kind": kind, "num_residues": args.num_bases}
    if args.bp is not None:
        chain_desc["bp"] = args.bp
    ligand_desc: dict = {"smiles": args.smiles, "ccd": args.ccd}
    if args.auto_condition:
        ligand_desc["auto_condition"] = True
    if args.buried:
        ligand_desc["buried"] = True
    inp = RFD4Input.from_chains(chain_desc, ligand_desc, np_seed=args.seed)
    n_total = int(np.asarray(inp.padding_mask).sum())
    assert n_total == args.num_bases + n_lig_atoms

    # ---- conditions via the RFD4Input adder API
    if args.bp is not None:
        inp = inp.add_bp(args.bp, start=0)
        st = np.asarray(inp.na_ss_state)[0]
        pm = np.asarray(inp.na_ss_pair_mask)[0]
        assert (st[:args.num_bases] != 0).any(), "add_bp did not land on the chain"
        n_pairs = int((np.asarray(inp.na_ss_pair)[0] > 0).sum() // 2)
        print(f"add_bp: NA-SS states on {int((st[:args.num_bases] != 0).sum())} bases, "
              f"{n_pairs} base pairs (C_NA_SS_STATE + C_BP)", flush=True)
    if args.ligand_rasa is not None:
        # binned-RASA condition on every ligand atom slot (C_RAS), edited on
        # the compiled input via the RFD4Input adder
        where = np.zeros((n_lig_atoms, 37), dtype=bool)
        where[:, 1] = True  # the atomized tokens' single atom slot
        inp = inp.add_rasa(int(args.ligand_rasa), aa=None, where=where,
                           start=args.num_bases)
        rasa = np.asarray(inp.rasa)[0]
        assert (rasa[args.num_bases:, 1] == int(args.ligand_rasa)).all(), \
            "ligand RASA condition did not land on the ligand slots"
        assert (rasa[:args.num_bases] == 0).all(), "chain RASA unexpectedly set"
        print(f"add_rasa: all {n_lig_atoms} ligand atoms -> {args.ligand_rasa}",
              flush=True)

    inp = inp.pad_to_nearest(args.pad_multiple)
    batch1 = inp.to_batch()
    n_width = int(np.asarray(batch1.padding_mask).shape[1])
    n_real = int(np.asarray(batch1.padding_mask).sum())
    tt = np.asarray(batch1.token_type)[0]
    assert (tt[:args.num_bases] == (1 if kind == "rna" else 2)).all()
    assert (tt[args.num_bases:n_real] == 3).all(), "ligand tokens must be atomized"
    # autofeaturized ligand roles (for the manifest + a compile sanity check)
    roles = None
    if args.auto_condition:
        from rfd4fx.ligand_conditions import ligand_roles

        roles = ligand_roles(lig.elements, lig.charges, lig.bonds)
        ps = np.asarray(batch1.pistack)[0]
        hb = np.asarray(batch1.hbond)[0]
        sa = np.asarray(batch1.saltbridge)[0]
        ra = np.asarray(batch1.rasa)[0]
        lig_sl = slice(args.num_bases, args.num_bases + n_lig_atoms)
        assert (ps[lig_sl, 1].astype(int) == roles["pistack"]).all()
        assert (hb[lig_sl, 1].astype(int) == roles["hbond"]).all()
        assert (sa[lig_sl, 1].astype(int) == roles["salt"]).all()
        assert (ra[lig_sl, 1] == (1 if args.buried else 0)).all()
        print(f"ligand auto_condition: {int(roles['pistack'].sum())} pistack "
              f"atoms, {int((roles['hbond'] == 2).sum())} acceptors, "
              f"{int((roles['hbond'] == 1).sum())} donors, "
              f"{int((roles['hbond'] == 3).sum())} both, "
              f"{int((roles['salt'] != 0).sum())} charged; "
              f"buried={'on' if args.buried else 'off'}", flush=True)
    print(f"compiled: width {n_width}, {n_real} real tokens "
          f"({args.num_bases} {kind} + {n_lig_atoms} ligand)", flush=True)

    # RFD4 / RFD4Sampler API: the model provides the step callable; the
    # sampler jits the step and the decode, integrates the flow, and decodes.
    sampler = RFD4Sampler(jax_model.step(), jax_model)

    t_all = time.time()
    for j in range(args.num_designs):
        t0 = time.time()
        key = jax.random.fold_in(jax.random.key(args.seed), 12345 + j)
        x, x_1_pred, data, dec = sampler.sample(batch1, key=key, nsteps=args.nsteps,
                                                recycle_schedule=THINKING_HIGH)

        # RNA chain A via the shared native-parity export (predicted atom
        # presence gating + OP3 dropped for the multi-residue NA chain)
        arr_p = decoded_to_atom_array(dec, batch1, aa_pred=data.aa_pred,
                                      chain_ids="A")
        # ligand chain B: one atom per token — coordinates from the flow state
        x_bb = np.asarray(x["bb_ca"])[0] * 10.0  # (n, 3) Å
        real_tok = np.asarray(batch1.padding_mask)[0]
        tok_is_ligand = np.zeros(n_width, dtype=bool)
        tok_is_ligand[args.num_bases:n_real] = True
        lig_coords = x_bb[tok_is_ligand]
        assert np.isfinite(lig_coords).all(), f"design {j}: non-finite ligand"
        n_lig = len(lig_names)
        lig_arr = bs.AtomArray(n_lig)
        lig_arr.add_annotation("hetero", dtype=bool)
        lig_arr.add_annotation("occupancy", dtype=float)
        lig_arr.add_annotation("b_factor", dtype=float)
        for k in range(n_lig):
            lig_arr.coord[k] = lig_coords[k]
            lig_arr.chain_id[k] = "B"
            lig_arr.res_id[k] = 1
            lig_arr.res_name[k] = lig.res_name
            lig_arr.atom_name[k] = lig_names[k]
            lig_arr.element[k] = lig_syms[k]
            lig_arr.hetero[k] = True
            lig_arr.occupancy[k] = 1.0
            lig_arr.b_factor[k] = 0.0
        arr = arr_p + lig_arr

        out = args.out_dir / f"design_{base_idx + j:03d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))
        back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
        p_atoms = back[~back.hetero]
        p_atoms = p_atoms[p_atoms.element != "H"]
        lig_back = back[back.hetero]
        assert len(np.unique(p_atoms.res_id)) == args.num_bases, \
            f"design {j}: {len(np.unique(p_atoms.res_id))} residues"
        assert len(lig_back) == n_lig, f"design {j}: {len(lig_back)} ligand atoms"
        assert bool(np.isfinite(back.coord).all()), f"design {j}: non-finite"
        assert "OP3" not in set(p_atoms.atom_name), \
            f"design {j}: spurious OP3 on a multi-residue NA chain"
        seq = "".join(p_atoms[p_atoms.res_id == r].res_name[0][0]
                      for r in range(1, args.num_bases + 1))
        # RNA backbone: consecutive phosphates
        phos = p_atoms[p_atoms.atom_name == "P"]
        d = (np.linalg.norm(np.diff(phos.coord, axis=0), axis=-1)
             if len(phos) == args.num_bases else np.array([np.nan]))
        # ligand internal geometry: bonded-pair distances (topology from the spec)
        bl = np.linalg.norm(lig_back.coord[lig.bonds[:, 0]]
                            - lig_back.coord[lig.bonds[:, 1]], axis=-1)
        # chain-ligand contact readout (heavy atoms)
        d_pl = np.linalg.norm(p_atoms.coord[:, None] - lig_back.coord[None], axis=-1)
        run = {
            "id": f"design_{base_idx + j:03d}", "file": str(out),
            "sequence": seq, "n_atoms": int(back.array_length()),
            "n_chain_atoms": int((~back.hetero).sum()),
            "n_ligand_atoms": int(back.hetero.sum()),
            "p_p_mean": float(d.mean()), "p_p_max": float(d.max()),
            "ligand_bond_length_mean": float(bl.mean()),
            "ligand_bond_length_min": float(bl.min()),
            "ligand_bond_length_max": float(bl.max()),
            "min_chain_ligand_distance": float(d_pl.min()),
            "chain_atoms_within_5A_of_ligand": int((d_pl < 5.0).any(axis=1).sum()),
        }
        manifest["designs"].append(run)
        print(f"      design_{base_idx + j:03d}.pdb: {seq} | P-P "
              f"{d.mean():.2f}±{d.std():.2f} Å (max {d.max():.2f}), ligand bonds "
              f"{bl.mean():.2f} [{bl.min():.2f}-{bl.max():.2f}] Å, min "
              f"chain-lig {d_pl.min():.1f} Å, "
              f"{run['chain_atoms_within_5A_of_ligand']} chain atoms < 5 Å  "
              f"[{time.time() - t0:.0f}s]", flush=True)
    out_manifest.write_text(json.dumps(manifest, indent=1))
    print(f"done: {args.num_designs} designs in {time.time() - t_all:.0f}s "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
