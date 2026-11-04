"""SS-conditioned protein sampling via `RFD4Spec.add_dssp` + the JAX port.

For each DSSP string, builds an `RFD4Spec` with one de-novo chain of
`len(dssp)` residues, conditions secondary structure with `add_dssp`
(strict 3-state: H=helix, E=strand, L=loop, X/_=unspecified), compiles the
batch through the direct featurizer, and samples with the JAX flow model +
autoencoder decode. Designs are written as PDBs with a manifest that also
records a backbone-dihedral secondary-structure sanity readout (helix/strand
propensity per residue region) so the conditioning effect is visible.

Example:
    python scripts/generate_dssp_jax.py \
        --out-dir /Users/mjendrusch/repos/rfd4j/rfd4-jax-test-dssp
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

DSSP_STRINGS = [
    "XXXHHHHHHHHHHXXXXXXHHHHHHHHHHXXXXXXHHHHHHHHHHXXX",
    "XXXEEEEEEXXXXXEEEEEEEXXXXXEEEEEEEXXXXXEEEEEEXXXXEEEXXX",
]


def ss_from_dihedrals(phi: np.ndarray, psi: np.ndarray) -> np.ndarray:
    """Rough per-residue SS class from backbone dihedrals (Sanity readout only).

    1 = helix region (phi ≈ -60±40, psi ≈ -45±40), 2 = strand region
    (phi ≈ -120±50, psi ≈ 120±50), else 0.
    """
    n = len(phi)
    out = np.zeros(n, dtype=np.int8)
    helix = (np.abs(((phi + 60) + 180) % 360 - 180) < 40) & \
            (np.abs(((psi + 45) + 180) % 360 - 180) < 40)
    strand = (np.abs(((phi + 120) + 180) % 360 - 180) < 50) & \
             (np.abs(((psi - 120) + 180) % 360 - 180) < 50)
    out[helix] = 1
    out[strand & ~helix] = 2
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-designs", type=int, default=3)
    parser.add_argument("--dssp", action="append", default=[],
                        help="DSSP string to condition on (repeatable; defaults to "
                             "the built-in helix/strand examples). With --two-chain, "
                             "strings are consumed in PAIRS: chain 1 gets dssp_1, "
                             "chain 2 gets dssp_2, and the condition is authored as "
                             "dssp_1 + dssp_2 over both chains.")
    parser.add_argument("--two-chain", action="store_true",
                        help="heterodimer mode: two chains, one per DSSP string pair")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=REPO / "rfd4-jax-test-dssp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=SNAPSHOT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import torch
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.model import RFD4
    from rfd4fx.spec import RFD4Spec

    torch.manual_seed(args.seed)
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint)
    model.eval()
    jax_model = RFD4.from_torch(model)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "featurization": "RFD4Spec.add_dssp -> direct featurizer (to_batch)",
        "seed": args.seed,
        "nsteps": args.nsteps,
        "backend": "jax (rfd4fx)",
        "runs": [],
    }
    manifest_path = args.out_dir / "manifest_dssp.json"
    if manifest_path.exists():
        try:
            prev = json.loads(manifest_path.read_text())
            manifest["runs"].extend(prev.get("runs", []))
        except Exception:
            pass  # unreadable previous manifest -> start fresh
    t_all = time.time()
    dssp_strings = args.dssp or DSSP_STRINGS
    base_idx = len(manifest["runs"])  # continue run numbering across invocations
    runs_spec = ([{"dssp": (dssp_strings[2 * k], dssp_strings[2 * k + 1]),
                   "two_chain": True}
                  for k in range(len(dssp_strings) // 2)]
                 if args.two_chain else
                 [{"dssp": d, "two_chain": False} for d in dssp_strings])
    for i, run_spec in enumerate(runs_spec):
        run_idx = base_idx + i
        dssp = run_spec["dssp"]
        run_dir = args.out_dir / f"run_{run_idx:02d}_{'heterodimer' if run_spec['two_chain'] else f'{sum(len(d) for d in ([dssp] if isinstance(dssp, str) else dssp))}res'}"
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run {run_idx}] DSSP: {dssp}", flush=True)

        spec = RFD4Spec()
        if run_spec["two_chain"]:
            d1, d2 = dssp
            spec.add_chain(len(d1))
            spec.add_chain(len(d2))
            spec.add_dssp(dssp=d1 + d2)  # chain=None: concatenated across both chains
        else:
            spec.add_chain(len(dssp))
            spec.add_dssp(chain=0, dssp=dssp)
        batch1, coord_cond = spec.to_batch(np_seed=args.seed)
        assert not np.asarray(coord_cond).any(), "unexpected coordinate conditioning"
        ss = np.asarray(batch1.secondary_structure)[0]
        dssp_all = dssp if isinstance(dssp, str) else dssp[0] + dssp[1]
        assert int((ss == 1).sum()) == dssp_all.count("H") \
            and int((ss == 2).sum()) == dssp_all.count("E"), \
            "C_SEC condition does not match the DSSP string"

        b = args.num_designs
        import jax.numpy as jnp
        batch = jax.tree.map(
            lambda x: jnp.repeat(x, b, axis=0) if hasattr(x, "shape") and x.ndim >= 1 else x,
            batch1,
            is_leaf=lambda l: l is None,
        )

        t0 = time.time()
        key = jax.random.fold_in(jax.random.key(args.seed), run_idx)
        x, x_1_pred, dec = jax_model.generate(batch, key=key, nsteps=args.nsteps)

        coors_a = np.asarray(dec.coors_nm) * 10.0  # nm -> Å
        aatype = np.asarray(dec.aatype_max)
        atom_mask = np.asarray(dec.atom_mask)
        emit = np.asarray(dec.emit_mask)
        # padding-aware token selection: padded slots (padding_mask False) are
        # inert (emit=0) but their chain_iid is 0.0 — exclude them from the
        # chain split so they can never reach a written PDB
        real_tok = np.asarray(batch1.padding_mask)[0]
        chain_ids_tok = np.asarray(batch1.chain_iid)[0]  # per-token chain iid (batch-replicated)
        dssp_per_chain = ([d for d in dssp] if isinstance(dssp, (list, tuple))
                          else [dssp])
        iid_order = list(dict.fromkeys(chain_ids_tok[real_tok].tolist()))
        n_residues_per_chain = ([len(d) for d in dssp_per_chain]
                                if run_spec["two_chain"] else [len(dssp)])
        run_info = {"dssp": dssp, "two_chain": run_spec["two_chain"],
                    "n_residues": n_residues_per_chain, "designs": []}

        def _chain_ss_readout(chain_arr, dssp_chain):
            """Backbone-dihedral SS sanity readout for one chain (degrees)."""
            ca = chain_arr[chain_arr.atom_name == "CA"]
            n_atoms = chain_arr[chain_arr.atom_name == "N"]
            c_atoms = chain_arr[chain_arr.atom_name == "C"]
            n = len(ca)

            def _dihedral(p0, p1, p2, p3):
                b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
                b1n = b1 / np.linalg.norm(b1)
                v = b0 - np.dot(b0, b1n) * b1n
                w = b2 - np.dot(b2, b1n) * b1n
                return float(np.degrees(np.arctan2(
                    np.dot(np.cross(b1n, v), w), np.dot(v, w))))

            phi = np.full(n, np.nan)
            psi = np.full(n, np.nan)
            for k in range(1, n):
                phi[k - 1] = _dihedral(c_atoms[k - 1].coord, n_atoms[k].coord,
                                       ca[k].coord, c_atoms[k].coord)
            for k in range(n - 1):
                psi[k] = _dihedral(n_atoms[k].coord, ca[k].coord,
                                   c_atoms[k].coord, n_atoms[k + 1].coord)
            pred = ss_from_dihedrals(np.nan_to_num(phi[:n - 1], nan=0.0),
                                     np.nan_to_num(psi[:n - 1], nan=0.0))
            cond = np.array([1 if c == "H" else 2 if c == "E" else 0
                             for c in dssp_chain])
            return {
                "helix": float(((cond[:n - 1] == 1) & (pred == 1)).sum()
                               / max(int((cond == 1).sum()), 1)),
                "strand": float(((cond[:n - 1] == 2) & (pred == 2)).sum()
                                / max(int((cond == 2).sum()), 1)),
            }

        for j in range(b):
            seq = aatype[j]
            mask_j = atom_mask[j] & emit[j][:, None]
            coords37 = coors_a[j].copy()
            coords37[~mask_j] = np.nan
            res_type = np.where(emit[j], seq, -1).astype(np.int64)
            # split the decoded tokens into chains by chain_iid (spec order)
            arrs = []
            for cid, ch_dssp in zip(iid_order, dssp_per_chain):
                tok_sel = (chain_ids_tok == cid) & real_tok
                arr_c = atom_array_from_encoding(
                    encoded_coord=coords37[tok_sel],
                    encoded_mask=mask_j[tok_sel],
                    encoded_seq=res_type[tok_sel],
                    encoding=UNIFIED_ATOM37_ENCODING,
                    chain_id=f"{'AB'[iid_order.index(cid)]}" if len(iid_order) <= 2
                    else f"chain_{cid}",
                )
                arrs.append(arr_c[np.isfinite(arr_c.coord).all(axis=-1)])
            arr = arrs[0]
            for extra in arrs[1:]:
                arr = arr + extra  # biotite concatenation (chain-A + chain-B)
            out = run_dir / f"design_{j:03d}.pdb"
            pdb = pdbio.PDBFile()
            pdb.set_structure(arr)
            pdb.write(str(out))
            back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
            ca = back[back.atom_name == "CA"]
            assert len(ca) == sum(n_residues_per_chain), \
                f"design {j}: {len(ca)} CA atoms"
            assert bool(np.isfinite(back.coord).all()), f"design {j}: non-finite"
            # CA-CA within chains only (no inter-chain peptide distance)
            d_all = []
            for cid in iid_order:
                ca_c = ca[ca.chain_id == "AB"[iid_order.index(cid)]] \
                    if len(iid_order) <= 2 else ca[ca.chain_id == f"chain_{cid}"]
                dd = np.linalg.norm(ca_c.coord[1:] - ca_c.coord[:-1], axis=-1)
                d_all.append(dd)
            d = np.concatenate(d_all)
            # per-chain SS readout
            readouts = {}
            for cid, ch_dssp in zip(iid_order, dssp_per_chain):
                ch_arr = back[back.chain_id == ("AB"[iid_order.index(cid)]
                                                if len(iid_order) <= 2 else f"chain_{cid}")]
                readouts[ch_dssp] = _chain_ss_readout(ch_arr, ch_dssp)
            hit_h = float(np.mean([r["helix"] for r in readouts.values()]))
            hit_e = float(np.mean([r["strand"] for r in readouts.values()]))
            run_info["designs"].append({
                "id": f"design_{j:03d}",
                "file": str(out),
                "n_atoms": int(back.array_length()),
                "ca_ca_mean": float(d.mean()),
                "ca_ca_max": float(d.max()),
                "per_chain_ss": {k: v for k, v in readouts.items()},
                "helix_basin_fraction_conditioned": hit_h,
                "strand_basin_fraction_conditioned": hit_e,
            })
            ss_str = ", ".join(
                f"chain {'AB'[i2]}: H {readouts[dd]['helix']:.0%}, E {readouts[dd]['strand']:.0%}"
                for i2, dd in enumerate(dssp_per_chain))
            print(
                f"      design_{j:03d}.pdb: {back.array_length()} atoms, "
                f"CA-CA {d.mean():.2f}±{d.std():.2f} Å (max {d.max():.2f}), "
                f"{ss_str}  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        manifest["runs"].append(run_info)
    (args.out_dir / "manifest_dssp.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {len(DSSP_STRINGS)} runs × {args.num_designs} designs "
          f"in {time.time() - t_all:.0f}s -> {args.out_dir}")


if __name__ == "__main__":
    main()
