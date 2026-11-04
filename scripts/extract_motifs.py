"""Extract ligand + contact-residue motifs from generated protein-ligand designs.

For each design PDB in the campaign directory (with `manifest_random.json` for
the SMILES): the ligand is the HETATM set; every protein residue gets its min
atom distance to any ligand atom; the `--num-residues` (default 5) closest
residues WITH at least one atom within `--cutoff` (default 5 A) are kept.
Writes one motif PDB per design (chain A = kept residues as ATOM records with
their original names/IDs, chain B = the ligand HETATM) plus `motifs.json`
(SMILES, kept residue names/IDs, per-residue min distances).

Example:
    python scripts/extract_motifs.py \
        --design-dir /Users/mjendrusch/repos/rfd4j/rfd4-test-ligands/random
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--design-dir", type=pathlib.Path,
                        default=pathlib.Path("/Users/mjendrusch/repos/rfd4j"
                                             "/rfd4-test-ligands/random"))
    parser.add_argument("--num-residues", type=int, default=5)
    parser.add_argument("--cutoff", type=float, default=5.0)
    args = parser.parse_args()

    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio

    manifest_path = args.design_dir / "manifest_random.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    smiles_by_id = {d["id"]: d.get("smiles") for d in manifest.get("designs", [])}

    motif_dir = args.design_dir / "motifs"
    motif_dir.mkdir(parents=True, exist_ok=True)
    index = []
    design_files = sorted(args.design_dir.glob("design_*.pdb"))
    print(f"{len(design_files)} design PDBs in {args.design_dir}")
    for path in design_files:
        design_id = path.stem
        back = pdbio.PDBFile.read(str(path)).get_structure(model=1)
        lig = back[back.hetero]
        prot = back[~back.hetero]
        if lig.array_length() == 0 or prot.array_length() == 0:
            print(f"  {design_id}: skipped (no ligand or no protein)")
            continue
        # per-residue min atom distance to any ligand atom
        res_ids = np.unique(prot.res_id)
        res_min_dist, res_names = [], []
        for rid in res_ids:
            res_atoms = prot[prot.res_id == rid]
            d = np.linalg.norm(res_atoms.coord[:, None] - lig.coord[None],
                               axis=-1)
            res_min_dist.append(float(d.min()))
            res_names.append(str(res_atoms.res_name[0]))
        order = np.argsort(res_min_dist)
        kept = []
        for idx in order:
            if len(kept) >= args.num_residues:
                break
            if res_min_dist[idx] <= args.cutoff:  # >=1 atom within the cutoff
                kept.append(int(idx))
        if not kept:
            print(f"  {design_id}: no residue within {args.cutoff} A — skipped")
            continue
        kept_ids = [int(res_ids[i]) for i in kept]

        # motif PDB: chain A = kept residues (original names/IDs), chain B = ligand
        res_arr = bs.AtomArray(0)
        for i in kept:
            sel = prot[prot.res_id == res_ids[i]]
            sel = sel.copy()
            sel.chain_id[:] = "A"
            sel.hetero[:] = False
            res_arr += sel
        lig_arr = lig.copy()
        lig_arr.chain_id[:] = "B"
        lig_arr.res_id[:] = 1
        lig_names = set(lig_arr.res_name.tolist())
        lig_arr.res_name[:] = lig_names.pop() if len(lig_names) == 1 else "LIG"
        motif = res_arr + lig_arr

        out = motif_dir / f"{design_id}_motif.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(motif)
        pdb.write(str(out))
        index.append({
            "design_id": design_id,
            "motif_pdb": str(out),
            "smiles": smiles_by_id.get(design_id),
            "kept_residues": [{"res_id": kept_ids[k],
                               "res_name": res_names[kept[k]],
                               "min_dist": res_min_dist[kept[k]]}
                              for k in range(len(kept))],
            "n_ligand_atoms": int(lig.array_length()),
        })
        dists = ", ".join(f"{res_names[kept[k]]}{kept_ids[k]}"
                          f"({res_min_dist[kept[k]]:.1f})" for k in range(len(kept)))
        print(f"  {design_id}: {len(kept)} residues [{dists}] + "
              f"{lig.array_length()} ligand atoms")
    (motif_dir / "motifs.json").write_text(json.dumps(index, indent=1))
    print(f"done: {len(index)} motifs -> {motif_dir}")


if __name__ == "__main__":
    main()
