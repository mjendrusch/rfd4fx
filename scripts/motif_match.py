"""Motif-matching search + aligned assembly for motif-scaffolding designs.

The trained conditioning semantics (and the native motif-scaffolding benchmark,
tag `motif_scaffold_refolding`, metric `design_motif_allatom_rmsd` computed per
coordinate group WITH Kabsch alignment) do NOT require the motif to appear at
its input-frame position: the design must recreate the motif's INTERNAL
geometry somewhere, found by alignment. This module implements that search
(non-consecutive residue matching: two-residue seeds, greedy extension) and
the aligned PDB assembly (motif + ligand reference chains transformed onto the
best-matching design segment).

All functions are pure numpy + gemmi/biotite at the file boundary.
"""

from __future__ import annotations

import pathlib

import numpy as np


# ---------------------------------------------------------------- parsing
def parse_design_chain(pdb_path: pathlib.Path, chain_id: str = "A"):
    """Designed polymer chain -> ({res_num: {atom_name: xyz}}, [res_num order])."""
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    design: dict[int, dict[str, np.ndarray]] = {}
    for c in model:
        if c.name != chain_id:
            continue
        for r in c:
            design[r.seqid.num] = {
                a.name.upper().strip(): np.array([a.pos.x, a.pos.y, a.pos.z])
                for a in r}
    return design, sorted(design)


def parse_motif_reference(pdb_path: pathlib.Path, res_chain: str = "M",
                          lig_chain: str = "B"):
    """Motif reference chains -> (motif residues, ligand atoms).

    Motif residues: [(res_id, res_name, {atom_name: xyz})] in file order.
    Ligand: {atom_name: xyz} (None when the motif has no ligand chain).

    The residue chain is `res_chain` (design-PDB convention, chain M) with a
    fallback to chain A (the extract_motifs.py convention: chain A = contact
    residues, chain B = ligand).
    """
    import gemmi

    st = gemmi.read_structure(str(pdb_path))
    model = st[0]
    chains = {c.name for c in model}
    if res_chain not in chains and "A" in chains:
        res_chain = "A"
    motif = []
    lig = None
    for c in model:
        if c.name == lig_chain:
            for r in c:
                lig = {a.name.upper().strip():
                       np.array([a.pos.x, a.pos.y, a.pos.z]) for a in r}
        elif c.name == res_chain:
            for r in c:
                motif.append((r.seqid.num, r.name, {
                    a.name.upper().strip():
                        np.array([a.pos.x, a.pos.y, a.pos.z]) for a in r}))
    return motif, lig


# ---------------------------------------------------------------- geometry
def kabsch_fit(P: np.ndarray, Q: np.ndarray):
    """Fit P onto Q: x' = (x - p_c) @ R + q_c. Returns (rmsd, R, p_c, q_c)."""
    p_c, q_c = P.mean(0), Q.mean(0)
    Pc, Qc = P - p_c, Q - q_c
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    R = V @ np.diag([1.0, 1.0, d]) @ Wt
    rmsd = float(np.sqrt(((Pc @ R - Qc) ** 2).sum(-1).mean()))
    return rmsd, R, p_c, q_c


def _pair_atoms(motif_atoms: dict, design_res: dict) -> tuple[list, list]:
    common = sorted(set(motif_atoms) & set(design_res))
    return ([motif_atoms[a] for a in common], [design_res[a] for a in common])


# ---------------------------------------------------------------- search
def find_motif_match(design: dict[int, dict], motif: list,
                     *, n_seeds: int = 100,
                     min_seed_atoms_per_res: int = 3):
    """Non-consecutive motif-matching search over the designed chain.

    Seeds: the two motif residues with the most atoms, matched against every
    ordered pair of design residues (Kabsch on the union of common atoms).
    Extension: greedily add the best (motif residue, unused design residue)
    pairing below `extend_cutoff` per-residue RMSD, refitting globally after
    each addition, until no residue can be added.

    Returns dict with:
      rmsd        final aligned all-atom RMSD over the matched correspondence
      n_atoms     number of atom pairs in the final fit
      mapping     list of (motif_res_id, motif_res_name, design_res_num)
      unmatched   motif residues that could not be placed
      R, p_c, q_c the rigid transform mapping motif frame -> design frame
      matched     whether the match is considered a recreation (rmsd < 2.5)
    """
    if len(motif) < 2 or len(design) < len(motif):
        return {"rmsd": float("inf"), "n_atoms": 0, "mapping": [],
                "unmatched": [m[:2] for m in motif], "R": None, "p_c": None,
                "q_c": None, "matched": False}
    nums = sorted(design)
    # seed with the two most atom-rich motif residues
    order = sorted(range(len(motif)),
                   key=lambda i: len(motif[i][2]), reverse=True)
    si, sj = order[0], order[1]
    seeds = []
    for di in nums:
        Pi, Qi = _pair_atoms(motif[si][2], design[di])
        if len(Pi) < min_seed_atoms_per_res:
            continue
        for dj in nums:
            if dj == di:
                continue
            Pj, Qj = _pair_atoms(motif[sj][2], design[dj])
            if len(Pj) < min_seed_atoms_per_res:
                continue
            P = np.array(Pi + Pj)
            Q = np.array(Qi + Qj)
            rmsd, R, p_c, q_c = kabsch_fit(P, Q)
            seeds.append((rmsd, di, dj, R, p_c, q_c))
    if not seeds:
        return {"rmsd": float("inf"), "n_atoms": 0, "mapping": [],
                "unmatched": [m[:2] for m in motif], "R": None, "p_c": None,
                "q_c": None, "matched": False}
    seeds.sort(key=lambda s: s[0])
    seeds = seeds[:n_seeds]

    def full_fit(mapping: dict[int, int]):
        P, Q = [], []
        for mi, di in mapping.items():
            p, q = _pair_atoms(motif[mi][2], design[di])
            P += p
            Q += q
        return kabsch_fit(np.array(P), np.array(Q)), len(P)

    best = None
    for _, di, dj, R0, p0, q0 in seeds:
        mapping = {si: di, sj: dj}
        (rmsd, R, p_c, q_c), n_atoms = full_fit(mapping)
        while len(mapping) < len(motif):
            # try every (unmatched motif residue, unused design residue)
            # addition, judged by the GLOBAL rmsd after refitting — a good
            # addition can look bad under the current 2-residue frame
            best_add = None
            for mi in range(len(motif)):
                if mi in mapping:
                    continue
                for dn in nums:
                    if dn in mapping.values():
                        continue
                    trial = dict(mapping)
                    trial[mi] = dn
                    (r, R2, p2, q2), na = full_fit(trial)
                    if best_add is None or (r, -na) < (best_add[0], -best_add[1]):
                        best_add = (r, na, mi, dn, R2, p2, q2)
            r, na, mi, dn, R2, p2, q2 = best_add
            # accept while the global fit keeps improving / stays sane
            if r > max(3.0, rmsd + 0.5) or r > 6.0:
                break
            mapping[mi] = dn
            rmsd, R, p_c, q_c, n_atoms = r, R2, p2, q2, na
        if best is None or (len(mapping), -rmsd) > (best["_n"], -best["rmsd"]):
            best = {
                "rmsd": rmsd, "n_atoms": n_atoms,
                "mapping": [(motif[mi][0], motif[mi][1], dn)
                            for mi, dn in sorted(mapping.items())],
                "unmatched": [motif[mi][:2] for mi in range(len(motif))
                              if mi not in mapping],
                "R": R, "p_c": p_c, "q_c": q_c,
                "matched": bool(rmsd < 2.5 and len(mapping) == len(motif)),
                "_n": len(mapping),
            }
    best.pop("_n", None)
    return best


# ---------------------------------------------------------------- assembly
def aligned_assembly(design_pdb: pathlib.Path, motif_pdb: pathlib.Path,
                     out_pdb: pathlib.Path, *, res_chain: str = "M",
                     lig_chain: str = "B") -> dict:
    """Search + write an aligned copy of a design PDB.

    The designed chain is copied as-is; the motif reference residues (chain M)
    and the ligand (chain B) are placed at the best-matching design segment
    (the rigid transform from find_motif_match). Returns the match report.
    """
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio
    import gemmi

    design, nums = parse_design_chain(design_pdb)
    motif, lig = parse_motif_reference(motif_pdb, res_chain, lig_chain)
    rep = find_motif_match(design, motif)
    st = gemmi.read_structure(str(design_pdb))
    model = st[0]

    motif_struct = bs.AtomArray(0)
    if rep["R"] is not None:
        R, p_c, q_c = rep["R"], rep["p_c"], rep["q_c"]
    else:  # no fit: identity
        R, p_c, q_c = np.eye(3), np.zeros(3), np.zeros(3)

    def xf(xyz):
        return (np.asarray(xyz) - p_c) @ R + q_c

    # motif residues (chain M) at the aligned position
    for res_id, res_name, atoms in motif:
        for an, xyz in sorted(atoms.items()):
            atom = bs.AtomArray(1)
            atom.coord[0] = xf(xyz)
            atom.chain_id[0] = res_chain
            atom.res_id[0] = int(res_id)
            atom.res_name[0] = str(res_name)
            atom.atom_name[0] = an
            atom.element[0] = an[0] if an[0].isalpha() else "C"
            atom.hetero[0] = True
            motif_struct += atom
    # ligand (chain B) at the aligned position
    if lig:
        for an, xyz in sorted(lig.items()):
            atom = bs.AtomArray(1)
            atom.coord[0] = xf(xyz)
            atom.chain_id[0] = lig_chain
            atom.res_id[0] = 1
            atom.res_name[0] = "LIG"
            atom.atom_name[0] = an
            atom.element[0] = an[0] if an[0].isalpha() else "C"
            atom.hetero[0] = True
            motif_struct += atom

    arr = gemmi_struct_to_biotite_protein(st) + motif_struct
    pdb = pdbio.PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(out_pdb))

    # ligand-packing metrics under the aligned placement
    if lig:
        lig_xyz = xf(np.array(list(lig.values())))
        ca = np.array([design[n]["CA"] for n in nums if "CA" in design[n]])
        d = np.linalg.norm(ca[:, None] - lig_xyz[None], axis=-1)
        rep["ligand_min_ca"] = float(d.min())
        rep["ligand_cas_within_10a"] = int((d.min(-1) < 10).sum())
    rep["design_pdb"] = str(design_pdb)
    rep["motif_pdb"] = str(motif_pdb)
    return rep


def gemmi_struct_to_biotite_protein(st):
    """The ATOM-record protein chains of a gemmi structure as a biotite array."""
    import biotite.structure as bs

    arr = bs.AtomArray(0)
    for chain in st[0]:
        for res in chain:
            if res.het_flag == "H":
                continue
            for a in res:
                atom = bs.AtomArray(1)
                atom.coord[0] = [a.pos.x, a.pos.y, a.pos.z]
                atom.chain_id[0] = chain.name
                atom.res_id[0] = res.seqid.num
                atom.res_name[0] = res.name
                atom.atom_name[0] = a.name
                atom.element[0] = a.element.name
                atom.hetero[0] = False
                arr += atom
    return arr


# ---------------------------------------------------------------- CLI
def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Aligned motif matching + assembly for design PDBs")
    parser.add_argument("--design-dir", type=pathlib.Path, required=True,
                        help="directory of design PDBs (chain A = designed)")
    parser.add_argument("--motifs", type=pathlib.Path, required=True,
                        help="directory of motif reference PDBs (chains M/B)")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--suffix", default="_scaffold_00",
                        help="design filename part to strip for the motif id")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for pdb in sorted(args.design_dir.glob("*.pdb")):
        mid = pdb.name.replace(".pdb", "").split(args.suffix)[0]
        motif_pdb = args.motifs / f"{mid}_motif.pdb"
        if not motif_pdb.exists():
            continue
        rep = aligned_assembly(pdb, motif_pdb,
                               args.out_dir / f"{pdb.stem}_aligned.pdb")
        rep_out = {k: v for k, v in rep.items()
                   if k not in ("R", "p_c", "q_c")}
        reports.append(rep_out)
        print(f"{mid}: rmsd {rep['rmsd']:.2f} A over {rep['n_atoms']} atoms | "
              f"matched {rep['matched']} | ligand min CA "
              f"{rep.get('ligand_min_ca', float('nan')):.2f} A")
    (args.out_dir / "motif_match_report.json").write_text(
        json.dumps(reports, indent=1, default=float))


if __name__ == "__main__":
    main()
