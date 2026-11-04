"""Backbone geometry validation for designed proteins.

Computes, per design PDB:
- bond distances between bonded backbone atoms, stored SEPARATELY for
  intra-residue bonds (N-CA, CA-C, C-O) and inter-residue bonds (C(i)-N(i+1));
- bond angles between bonded backbone atoms, likewise split into intra-residue
  (N-CA-C, CA-C-O) and inter-residue (CA(i)-C(i)-N(i+1), O(i)-C(i)-N(i+1),
  C(i)-N(i+1)-CA(i+1));
- all backbone dihedral angles: phi(i)=C(i-1)-N-CA-C, psi(i)=N-CA-C-N(i+1),
  omega(i)=CA(i)-C(i)-N(i+1)-CA(i+1).

Everything is cross-referenced against the literature:

Bond lengths / bond angles — Engh & Huber, "Accurate bond and angle parameters
for X-ray protein structure refinement", Acta Cryst. A47, 392-400 (1991), as
re-tabulated in Int. Tables for Crystallography Vol. F ch. 18.3 and at
https://www.ruppweb.org/Xray/tutorial/engtabl.html . A violation is |z| > 4
(sigma-level used by structure-validation packages, cf. WHAT_CHECK).

Dihedrals — omega must be planar: trans ~180 deg or cis ~0 deg (SWISS-MODEL
course ch. 1; Pauling & Corey 1951); violation when neither within 30 deg.
phi/psi are checked against Ramachandran-allowed regions (Ramachandran,
Ramakrishnan & Sasisekharan, J. Mol. Biol. 7, 95 (1963); generous bounds after
Lovell et al., Proteins 50, 437-450 (2003)) — a violation is a residue falling
in NO allowed basin (non-Gly; Gly/Pro get relaxed checks). Consecutive
left-handed (alphaL) basin occupancy is reported as INFO: a right-handed helix
mirrored/flipped shows up as an alphaL run.

Violations are reported per type (bond_length / bond_angle / dihedral_omega /
dihedral_phi_psi) with chain and residue index. Raw measurements are stored in
per-design .npz archives (intra- and inter-residue quantities separate).

Usage:
    python scripts/check_backbone_geometry.py design_dir_or_files... \
        --out-dir geometry_checks
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

# --- Literature reference values -------------------------------------------------
# Engh & Huber (1991) mean, sigma; gly variants where tabulated. Angles in deg.
BOND_LENGTHS = {  # (mean, sigma) in Angstrom
    ("N", "CA"): (1.458, 0.019),
    ("CA", "C"): (1.525, 0.021),
    ("C", "O"): (1.231, 0.020),
}
BOND_LENGTHS_GLY = {("CA", "C"): (1.516, 0.021)}
BOND_LENGTH_INTER = {("C", "N"): (1.329, 0.014)}

BOND_ANGLES = {  # (mean, sigma) in degrees; atom triple in bonded order
    ("N", "CA", "C"): (111.0, 2.7),
    ("CA", "C", "O"): (120.8, 2.1),
}
BOND_ANGLES_GLY = {("N", "CA", "C"): (112.5, 2.7)}
BOND_ANGLES_INTER = {  # triples spanning residue i -> i+1 (atoms of i, then i+1)
    ("CA", "C", "N"): (116.2, 2.0),
    ("O", "C", "N"): (123.0, 1.6),
    ("C", "N", "CA"): (121.7, 1.8),
}

Z_LIMIT = 4.0            # |z| threshold for length/angle violations
OMEGA_TOL_DEG = 30.0     # omega must be within this of 0 (cis) or 180 (trans)

# Clash detection (MolProbity convention): a heavy-atom pair clashes when
# distance < r_i + r_j - CLASH_OVERLAP (vdW overlap >= 0.4 A), cf. Word, Lovell
# et al., J. Mol. Biol. 285, 1735 (1999) / MolProbity (Chen et al. 2010).
CLASH_OVERLAP = 0.4
VDW_RADII = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
BACKBONE_ATOMS = {"N", "CA", "C", "O"}
# inter-residue pairs excluded from clash scoring: the peptide bond itself and
# its 1-3 neighbours (CA(i)-N(i+1), O(i)-N(i+1), C(i)-CA(i+1))
# inter-residue pairs excluded from clash scoring: all pairs across the peptide
# bond separated by <=3 covalent bonds — the bond itself (C-N), the 1-3 pairs
# (CA-N, O-N, C-CA) and the 1-4 pairs O-CA + CA-CA (fixed by omega planarity)
# and N-N + C-C (torsions psi(i)/phi(i+1), validated by the dihedral checks
# instead) — standard heavy-atom-only practice (1-4 pairs are torsion-validated,
# cf. MolProbity's all-atom-contact reduction).
PEPTIDE_13 = {frozenset(("C", "N")), frozenset(("CA", "N")),
              frozenset(("O", "N")), frozenset(("C", "CA")),
              frozenset(("O", "CA")), frozenset(("N", "N")),
              frozenset(("C", "C")), frozenset(("CA", "CA"))}

# Ramachandran allowed basins, generous rectangles after Lovell et al. 2003
# (general case, non-Gly, non-Pro), degrees.
RAMA_BASINS = {
    "alphaR": (-160.0, -35.0, -120.0, 30.0),   # phi_lo, phi_hi, psi_lo, psi_hi
    "beta": (-180.0, -50.0, 60.0, 180.0),
    "beta_wrap": (-180.0, -75.0, -180.0, -140.0),
    "alphaL": (30.0, 100.0, -30.0, 100.0),
}
RAMA_BASINS_PRO = {
    "alphaR": (-130.0, -25.0, -120.0, 30.0),
    "beta": (-180.0, -50.0, 60.0, 180.0),
    "beta_wrap": (-180.0, -75.0, -180.0, -140.0),
}
# Gly: mirrored regions allowed too; only the sterically forbidden (0,0) cluster flagged.
RAMA_FORBIDDEN_ALL = (-25.0, 25.0, -25.0, 25.0)


def _dihedral(p0, p1, p2, p3) -> float:
    """Torsion angle in degrees for four xyz points (standard IUPAC convention)."""
    b0 = p0 - p1
    b1 = p2 - p1
    b2 = p3 - p2
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def _angle(p0, p1, p2) -> float:
    """Bond angle p0-p1-p2 in degrees."""
    v0, v1 = p0 - p1, p2 - p1
    c = np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def measure(path: pathlib.Path) -> dict:
    """Measure all backbone quantities of one PDB (first model, every chain)."""
    import biotite.structure.io.pdb as pdbio

    arr = pdbio.PDBFile.read(str(path)).get_structure(model=1)
    backbones = {"N", "CA", "C", "O"}
    chains = {}
    for chain_id in np.unique(arr.chain_id):
        sub = arr[arr.chain_id == chain_id]
        residues = []
        for res_id in np.unique(sub.res_id):
            res = sub[sub.res_id == res_id]
            names = {a.atom_name: a.coord.astype(float) for a in res if a.atom_name in backbones}
            if "N" in names and "CA" in names and "C" in names:
                residues.append(
                    {
                        "res_id": int(res_id),
                        "res_name": str(res.res_name[0]),
                        **names,
                    }
                )
        # all heavy atoms for clash detection (backbone flag by atom name)
        atoms = [
            {
                "name": str(a.atom_name),
                "coord": a.coord.astype(float),
                "element": str(a.element).strip().upper() if isinstance(a.element, str) else "C",
                "res_id": int(a.res_id),
                "res_name": str(a.res_name),
                "backbone": str(a.atom_name) in BACKBONE_ATOMS,
            }
            for a in sub
            if not np.isnan(a.coord).any() and str(a.element).strip().upper() in VDW_RADII
        ]
        chains[str(chain_id)] = residues
        chains[str(chain_id) + "#atoms"] = atoms
    return {"file": str(path), "chains": chains}


def find_clashes(atoms: list[dict], chain: str, residues: list[dict]) -> list[dict]:
    """Heavy-atom clashes categorised as bb-bb / bb-sc / sc-sc.

    MolProbity convention: clash when distance < r_i + r_j - 0.4 A (vdW overlap
    >= 0.4 A; Word, Lovell et al., J. Mol. Biol. 285, 1735 (1999); Chen et al.,
    MolProbity 2010). Excludes same-residue pairs (all bonded or 1-3 through
    CA) and the peptide bond plus its 1-3 neighbours across consecutive
    residues (CA(i)-N(i+1), O(i)-N(i+1), C(i)-CA(i+1))."""
    n = len(atoms)
    if n < 2:
        return []
    coords = np.array([a["coord"] for a in atoms])
    radii = np.array([VDW_RADII[a["element"]] for a in atoms])
    rids = np.array([a["res_id"] for a in atoms])
    names = [a["name"] for a in atoms]
    # vectorized exclusions: same-residue pairs, and peptide-bond <=1-4 pairs
    # between residue-adjacent atoms (name-pair membership via lookup matrix)
    same_res = rids[:, None] == rids[None, :]
    adjacent = (rids[:, None] == rids[None, :] + 1) | (rids[:, None] + 1 == rids[None, :])
    uniq = sorted(set(names))
    nidx = {nm: i for i, nm in enumerate(uniq)}
    pair_ok = np.zeros((len(uniq), len(uniq)), dtype=bool)
    for s in PEPTIDE_13:
        for a in s:
            for b in s:
                if a != b or (a, b) == ("N", "N") or (a, b) == ("C", "C"):
                    if a in nidx and b in nidx:  # non-standard chains may carry other names
                        pair_ok[nidx[a], nidx[b]] = True
                        pair_ok[nidx[b], nidx[a]] = True
    nidx_arr = np.array([nidx[nm] for nm in names])
    excl = same_res | (adjacent & pair_ok[nidx_arr[:, None], nidx_arr[None, :]])
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    thresh = radii[:, None] + radii[None, :] - CLASH_OVERLAP
    clash = (d < thresh) & ~excl & ~np.eye(n, dtype=bool)
    pairs = np.argwhere(np.triu(clash, 1))
    out = []
    for i, j in pairs:
        a, b = atoms[int(i)], atoms[int(j)]
        cat = (
            "clash_bb_bb" if a["backbone"] and b["backbone"]
            else "clash_bb_sc" if a["backbone"] or b["backbone"]
            else "clash_sc_sc"
        )
        res_str = lambda x: f"{x['res_name']}{x['res_id']}"  # noqa: E731
        out.append(
            {
                "type": cat, "chain": chain, "res_i": a["res_id"],
                "atoms": f"{res_str(a)} {a['name']} / {res_str(b)} {b['name']}",
                "value": float(d[int(i), int(j)]), "ideal": float(thresh[int(i), int(j)]), "z": None,
                "detail": f"vdW overlap {thresh[int(i), int(j)] - d[int(i), int(j)]:.2f} A",
            }
        )
    out.sort(key=lambda c: c["value"])
    return out


def evaluate(meas: dict) -> dict:
    """Compute violations + store raw measurements for one design."""
    violations = []
    raw = {}
    info = []
    for chain, residues in meas["chains"].items():
        if chain.endswith("#atoms"):
            continue  # clash atom list, not residues
        n = len(residues)

        def R(i, atom):
            return residues[i][atom]

        # --- bond distances: intra- and inter-residue, stored separately
        intra_len = np.full((n, 3), np.nan)   # N-CA, CA-C, C-O
        inter_len = np.full(max(n - 1, 0), np.nan)  # C(i)-N(i+1)
        for i, res in enumerate(residues):
            gly = res["res_name"] == "GLY"
            for j, (a, b) in enumerate([("N", "CA"), ("CA", "C"), ("C", "O")]):
                if a in res and b in res:
                    table = BOND_LENGTHS_GLY if (gly and (a, b) in BOND_LENGTHS_GLY) else BOND_LENGTHS
                    if (a, b) not in table:
                        continue
                    d = float(np.linalg.norm(res[a] - res[b]))
                    intra_len[i, j] = d
                    mu, sig = table[(a, b)]
                    z = (d - mu) / sig
                    if abs(z) > Z_LIMIT:
                        violations.append(
                            {"type": "bond_length", "chain": chain, "res_i": res["res_id"],
                             "atoms": f"{a}-{b}", "value": d, "ideal": mu,
                             "z": z, "detail": f"{a}-{b} intra {res['res_name']}{res['res_id']}"}
                        )
        for i in range(n - 1):
            d = float(np.linalg.norm(R(i, "C") - R(i + 1, "N")))
            inter_len[i] = d
            mu, sig = BOND_LENGTH_INTER[("C", "N")]
            z = (d - mu) / sig
            if abs(z) > Z_LIMIT:
                violations.append(
                    {"type": "bond_length", "chain": chain,
                     "res_i": residues[i]["res_id"], "atoms": "C(i)-N(i+1)",
                     "value": d, "ideal": mu, "z": z,
                     "detail": f"peptide bond {residues[i]['res_name']}{residues[i]['res_id']}"
                               f"-{residues[i+1]['res_name']}{residues[i+1]['res_id']}"}
                )

        # --- bond angles: intra- and inter-residue, stored separately
        intra_ang = np.full((n, 2), np.nan)          # N-CA-C, CA-C-O
        inter_ang = np.full((max(n - 1, 0), 3), np.nan)  # CA-C-N, O-C-N, C-N-CA
        for i, res in enumerate(residues):
            gly = res["res_name"] == "GLY"
            for j, triple in enumerate([("N", "CA", "C"), ("CA", "C", "O")]):
                if all(a in res for a in triple):
                    table = BOND_ANGLES_GLY if (gly and triple in BOND_ANGLES_GLY) else BOND_ANGLES
                    a = _angle(res[triple[0]], res[triple[1]], res[triple[2]])
                    intra_ang[i, j] = a
                    mu, sig = table[triple]
                    z = (a - mu) / sig
                    if abs(z) > Z_LIMIT:
                        violations.append(
                            {"type": "bond_angle", "chain": chain, "res_i": res["res_id"],
                             "atoms": "-".join(triple), "value": a, "ideal": mu,
                             "z": z, "detail": f"{'-'.join(triple)} {res['res_name']}{res['res_id']}"}
                        )
        for i in range(n - 1):
            for j, triple in enumerate([("CA", "C", "N"), ("O", "C", "N")]):
                a, b, c = triple
                if a in residues[i] and b in residues[i] and c in residues[i + 1]:
                    ang = _angle(residues[i][a], residues[i][b], residues[i + 1][c])
                    inter_ang[i, j] = ang
                    mu, sig = BOND_ANGLES_INTER[triple]
                    z = (ang - mu) / sig
                    if abs(z) > Z_LIMIT:
                        violations.append(
                            {"type": "bond_angle", "chain": chain,
                             "res_i": residues[i]["res_id"], "atoms": f"{a}(i)-{b}(i)-{c}(i+1)",
                             "value": ang, "ideal": mu, "z": z,
                             "detail": f"{'-'.join(triple)} {residues[i]['res_name']}{residues[i]['res_id']}"}
                        )
            ang = _angle(R(i, "C"), R(i + 1, "N"), R(i + 1, "CA"))
            inter_ang[i, 2] = ang
            mu, sig = BOND_ANGLES_INTER[("C", "N", "CA")]
            z = (ang - mu) / sig
            if abs(z) > Z_LIMIT:
                violations.append(
                    {"type": "bond_angle", "chain": chain, "res_i": residues[i]["res_id"],
                     "atoms": "C(i)-N(i+1)-CA(i+1)", "value": ang, "ideal": mu, "z": z,
                     "detail": f"C-N-CA {residues[i+1]['res_name']}{residues[i+1]['res_id']}"}
                )

        # --- dihedral angles: phi, psi, omega
        phi = np.full(n, np.nan)
        psi = np.full(n, np.nan)
        omega = np.full(max(n - 1, 0), np.nan)
        for i in range(1, n):
            phi[i] = _dihedral(R(i - 1, "C"), R(i, "N"), R(i, "CA"), R(i, "C"))
        for i in range(n - 1):
            psi[i] = _dihedral(R(i, "N"), R(i, "CA"), R(i, "C"), R(i + 1, "N"))
        for i in range(n - 1):
            omega[i] = _dihedral(R(i, "CA"), R(i, "C"), R(i + 1, "N"), R(i + 1, "CA"))

        for i in range(n - 1):
            w = omega[i]
            # circular distance from cis(0) / trans(180): |w| and 180-|w|
            dev = min(abs(w), 180.0 - abs(w))
            if dev > OMEGA_TOL_DEG:
                violations.append(
                    {"type": "dihedral_omega", "chain": chain, "res_i": residues[i]["res_id"],
                     "atoms": f"omega {residues[i]['res_name']}{residues[i]['res_id']}"
                              f"-{residues[i+1]['res_name']}{residues[i+1]['res_id']}",
                     "value": w, "ideal": 180.0, "z": dev,
                     "detail": f"non-planar peptide bond, {dev:.1f} deg from cis/trans"}
                )
        for i in range(1, n):
            res = residues[i]
            if np.isnan(psi[i]):  # C-terminal residue: psi undefined
                continue
            name, p, q = res["res_name"], float(phi[i]), float(psi[i])
            if name == "GLY":
                lo_p, hi_p, lo_q, hi_q = RAMA_FORBIDDEN_ALL
                if lo_p < p < hi_p and lo_q < q < hi_q:
                    violations.append(
                        {"type": "dihedral_phi_psi", "chain": chain, "res_i": res["res_id"],
                         "atoms": f"phi/psi GLY{res['res_id']}", "value": (p, q),
                         "ideal": "allowed region", "z": None,
                         "detail": "Gly in sterically forbidden (0,0) cluster"}
                    )
                continue
            basins = RAMA_BASINS_PRO if name == "PRO" else RAMA_BASINS
            in_basin = None
            for bname, (plo, phi_hi, qlo, qhi) in basins.items():
                if plo <= p <= phi_hi and qlo <= q <= qhi:
                    in_basin = bname
                    break
            raw.setdefault("rama_basin", {})[f"{chain}:{res['res_id']}"] = in_basin
            if in_basin is None:
                violations.append(
                    {"type": "dihedral_phi_psi", "chain": chain, "res_i": res["res_id"],
                     "atoms": f"phi/psi {name}{res['res_id']}", "value": (p, q),
                     "ideal": "Ramachandran-allowed (Lovell 2003)", "z": None,
                     "detail": f"phi={p:.1f} psi={q:.1f}: no allowed basin"}
                )

        raw[f"{chain}_intra_bond_lengths"] = intra_len  # cols: N-CA, CA-C, C-O
        raw[f"{chain}_inter_bond_lengths"] = inter_len  # C(i)-N(i+1)
        raw[f"{chain}_intra_bond_angles"] = intra_ang   # cols: N-CA-C, CA-C-O
        raw[f"{chain}_inter_bond_angles"] = inter_ang   # cols: CA-C-N, O-C-N, C-N-CA
        raw[f"{chain}_phi"] = phi
        raw[f"{chain}_psi"] = psi
        raw[f"{chain}_omega"] = omega

        # INFO: runs of >=4 consecutive left-handed-basin residues (mirrored/flipped helix signature)
        rama = raw.get("rama_basin", {})
        run = []
        for i in range(1, n):
            b = rama.get(f"{chain}:{residues[i]['res_id']}")
            run.append(b == "alphaL")
        for i in range(len(run) - 3):
            if all(run[i:i + 4]):
                info.append(
                    f"chain {chain}: >=4 consecutive alphaL-basin residues starting at "
                    f"{residues[i+1]['res_id']} (left-handed/mirrored segment signature)"
                )
                break

        # --- clashes: bb-bb / bb-sc / sc-sc (MolProbity vdW-overlap criterion)
        violations.extend(find_clashes(meas["chains"][chain + "#atoms"], chain, residues))

    return {"file": meas["file"], "violations": violations, "info": info, "raw": raw}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=pathlib.Path, help="PDB files or directories")
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()

    pdbs = []
    for p in args.inputs:
        if p.is_dir():
            pdbs.extend(sorted(p.glob("*.pdb")))
        else:
            pdbs.append(p)
    assert pdbs, "no PDB files found"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    totals = {"bond_length": 0, "bond_angle": 0, "dihedral_omega": 0, "dihedral_phi_psi": 0,
              "clash_bb_bb": 0, "clash_bb_sc": 0, "clash_sc_sc": 0}
    for pdb in pdbs:
        result = evaluate(measure(pdb))
        stem = pdb.stem
        np.savez_compressed(args.out_dir / f"{stem}_geometry.npz", **result["raw"])
        (args.out_dir / f"{stem}_violations.json").write_text(
            json.dumps({"file": result["file"], "violations": result["violations"],
                        "info": result["info"]}, indent=1)
        )
        counts = {}
        for v in result["violations"]:
            counts[v["type"]] = counts.get(v["type"], 0) + 1
            totals[v["type"]] += 1
        summary[stem] = {
            "n_violations": len(result["violations"]),
            "by_type": counts,
            "info": result["info"],
        }
        line = f"{stem}: {len(result['violations'])} violations"
        if counts:
            line += " (" + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) + ")"
        if result["info"]:
            line += f"  [INFO: {'; '.join(result['info'])}]"
        print(line)
        for v in result["violations"][:15]:
            val = v["value"] if isinstance(v["value"], str) else (
                f"({v['value'][0]:.1f}, {v['value'][1]:.1f})" if isinstance(v["value"], tuple)
                else f"{v['value']:.2f}")
            print(f"    {v['type']:>16s}  chain {v['chain']} res {v['res_i']:>4}  "
                  f"{v['atoms']}: {val}" + (f" (vs {v['ideal']:.2f})" if isinstance(v['ideal'], float) else f" (vs {v['ideal']})")
                  + (f" (z={v['z']:.1f})" if isinstance(v['z'], float) else ""))

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"TOTALS across {len(pdbs)} designs: " + ", ".join(f"{k} {v}" for k, v in totals.items()))
    print(f"reports -> {args.out_dir}")


if __name__ == "__main__":
    main()
