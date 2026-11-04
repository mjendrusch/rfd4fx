"""RNA structure generation from a dot-bracket base-pair condition (RFD4).

Builds a de-novo RNA design chain conditioned on the dot-bracket NA
secondary structure (C_NA_SS_STATE + C_BP — BasePairEnum 1=paired, 2=loop,
0=unspecified for x/_/X) via RFD4Input.from_chains({"kind": "rna", "bp": ...}),
then samples with the RFD4/RFD4Sampler API (jitted step + decode,
THINKING_HIGH recycle schedule). Per design, writes a PDB plus base-pair
geometry QC (C1'-C1' distances of the requested pairs, P-P backbone steps).
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bp", required=True,
                        help="dot-bracket string: . ( ) [ ] { } < > letters, "
                             "x/_/X = unknown; length = RNA length")
    parser.add_argument("--num-designs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--pad-multiple", type=int, default=64)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO / "rfd4-test-rna-bp")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import equinox as eqx  # noqa: F401  (sampler internals)
    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure.io.pdb as pdbio
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
    from rfd4fx.model import RFD4
    from rfd4fx.rfd4_input import RFD4Input
    from rfd4fx.spec import parse_dot_bracket

    states, pairs = parse_dot_bracket(args.bp)
    n = len(args.bp)
    print(f"bp: {args.bp}  ({n} residues, {len(pairs)} base pairs)", flush=True)

    # ------------------------------------------------------------------ input
    inp = RFD4Input.from_chains({"kind": "rna", "bp": args.bp}, np_seed=args.seed)
    inp = inp.pad_to_nearest(args.pad_multiple)
    batch = inp.to_batch()
    tt = np.asarray(batch.token_type)[0]
    assert (tt[:n] == 1).all(), "RNA chain tokens must carry token_type 1"
    st = np.asarray(batch.na_ss_state)[0]
    assert np.array_equal(st[:n].astype(int), states.astype(int))
    pm = np.asarray(batch.na_ss_pair_mask)[0]
    for i, j in pairs:
        assert pm[i, j] and pm[j, i]
    print(f"input compiled: width {np.asarray(batch.padding_mask).shape[1]}, "
          f"{len(pairs)} pairs conditioned", flush=True)

    # ------------------------------------------------------------------ model
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(SNAPSHOT)
    model.eval()
    jax_model = RFD4.from_torch(model)
    # RFD4 / RFD4Sampler API: the model provides the step callable; the
    # sampler jits the step and the decode, integrates the flow, and decodes.
    sampler = RFD4Sampler(jax_model.step(), jax_model)

    manifest = {
        "bp": args.bp, "n_residues": n, "pairs": [list(p) for p in pairs],
        "seed": args.seed, "nsteps": args.nsteps,
        "pad_multiple": args.pad_multiple, "recycle_schedule": "thinking_high",
        "backend": "jax (rfd4fx)", "designs": [],
    }
    t_all = time.time()
    for k in range(args.num_designs):
        t0 = time.time()
        key = jax.random.fold_in(jax.random.key(args.seed), 4242 + k)
        x, x_1, data, dec = sampler.sample(batch, key=key, nsteps=args.nsteps,
                                           recycle_schedule=THINKING_HIGH)

        # ------------------------------------------------------------- output
        # shared export helper: native structure_export semantics (atom_mask +
        # predicted-presence gating, OP3 dropped from multi-residue NA chains)
        from rfd4fx.pdb_io import decoded_to_atom_array

        arr = decoded_to_atom_array(dec, batch, aa_pred=data.aa_pred,
                                    chain_ids="A")
        n_atoms = len(arr)
        assert len(np.unique(arr.res_id)) == n, \
            f"design {k}: {len(np.unique(arr.res_id))} residues != {n}"
        out = args.out_dir / f"rna_bp_design_{k:02d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))

        # ------------------------------------------------------------ QC
        # consecutive phosphates (slot 0 = P) and requested-pair C1'-C1' (slot 1)
        p_atoms = arr[(arr.atom_name == "P")]
        pp = (np.linalg.norm(np.diff(p_atoms.coord, axis=0), axis=1)
              if len(p_atoms) == n else np.array([]))
        c1 = {int(r): arr[(arr.res_id == r) & (arr.atom_name == "C1'")].coord[0]
              for r in range(1, n + 1)
              if ((arr.res_id == r) & (arr.atom_name == "C1'")).any()}
        pair_d = [float(np.linalg.norm(c1[i + 1] - c1[j + 1]))
                  for i, j in pairs if (i + 1) in c1 and (j + 1) in c1]
        seq = "".join(arr[(arr.res_id == r)].res_name[0][0] for r in range(1, n + 1))
        design_rec = {
            "design": k, "pdb": out.name, "n_atoms": n_atoms, "sequence": seq,
            "p_p_mean": float(pp.mean()) if pp.size else None,
            "p_p_max": float(pp.max()) if pp.size else None,
            "pair_c1c1_mean": float(np.mean(pair_d)) if pair_d else None,
            "pair_c1c1_max": float(np.max(pair_d)) if pair_d else None,
            "seconds": round(time.time() - t0, 1),
        }
        manifest["designs"].append(design_rec)
        print(f"rna_bp_design_{k:02d}: {seq} | P-P {design_rec['p_p_mean']:.2f}"
              f"+-{pp.std():.2f} A (max {design_rec['p_p_max']:.2f}) | "
              f"pair C1'-C1' {design_rec['pair_c1c1_mean']:.1f} A "
              f"(max {design_rec['pair_c1c1_max']:.1f}, n={len(pair_d)}) "
              f"[{design_rec['seconds']}s]", flush=True)
    (args.out_dir / "manifest_rna_bp.json").write_text(json.dumps(manifest, indent=2))
    print(f"done: {args.num_designs} designs in {time.time() - t_all:.0f}s "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
