"""Numpy-only featurization of real-atom annotated CIFs.

Reads structures with gemmi, resolves CCD chemistry, and builds the token,
bond-graph and chain/molecule bookkeeping for all-atom inputs."""

from __future__ import annotations

import pathlib

import warnings

import numpy as np

from .ccd import get_ccd_entries
from .featurize import SEQ_PROB_ORDER, assemble_flow_batch
from .types import FlowBatch

# biotite BondType ids used by incident_bond_type_counts (10 channels)
BT_SINGLE, BT_DOUBLE, BT_AROMATIC_SINGLE, BT_AROMATIC_DOUBLE, BT_AROMATIC = 1, 2, 5, 6, 9

MASKED_RES = "<M>"
UNK_ATOM = "UNK"
ELEMENT_X = "X"


# --------------------------------------------------------------------- CIF input
def _read_annotated_cif(path: str | pathlib.Path) -> dict:
    """gemmi read of an annotated CIF: atom rows + annotation columns + entity_poly."""
    import gemmi

    path = pathlib.Path(path)
    if path.suffix == ".zst":
        import zstandard

        dctx = zstandard.ZstdDecompressor()
        text = dctx.stream_reader(open(path, "rb")).read().decode()
        doc = gemmi.cif.read_string(text)
        block = doc.sole_block()
    else:
        doc = gemmi.cif.read(str(path))
        block = doc.sole_block()
    annotations: dict[str, list[str]] = {}
    atom_cols: dict[str, list[str]] = {}
    loop = None
    for item in block:
        if item.loop is not None and any(t.startswith("_atom_site.") for t in item.loop.tags):
            loop = item.loop
            break
    atom_cols: dict[str, list[str]] = {}
    if loop is None:
        # single-row atom_site authored as key-value pairs (sentinel-style CIFs)
        pairs = {}
        for item in block:
            if item.pair is not None:
                tag, value = item.pair
                if str(tag).startswith("_atom_site."):
                    pairs[str(tag).split(".", 1)[1]] = [str(value)]
        if "label_atom_id" not in pairs:
            raise ValueError("no _atom_site loop found")
        tags, width = list(pairs), 1
        atom_cols.update(pairs)
        for tag, values in pairs.items():
            if tag.startswith(("condition_", "annotation_")):
                annotations[tag] = values
    else:
        tags = [t.split(".", 1)[1] for t in loop.tags]
        width = len(loop.values) // len(loop.tags)
        for ci, tag in enumerate(tags):
            values = [loop.values[k * len(loop.tags) + ci] for k in range(width)]
            if tag.startswith(("condition_", "annotation_")):
                annotations[tag] = values
            else:
                atom_cols[tag] = values

    # separate _condition_coordinate_1_atom loop (per-atom conditioning coordinates;
    # idx0 indexes the atom_site rows, val0-2 the conditioned position or NaN)
    cond_coord_rows = np.full((width, 3), np.nan, np.float32)
    for item in block:
        item_loop = item.loop
        if item_loop is None or not any(
                t.startswith("_condition_coordinate_1_atom.") for t in item_loop.tags):
            continue
        ctags = [t.split(".", 1)[1] for t in item_loop.tags]
        ci0 = ctags.index("idx0")
        val_cols = {ctags.index(t): int(t.rsplit("val", 1)[-1])
                    for t in ctags if t.rsplit("val", 1)[-1] in ("0", "1", "2")}
        for k in range(item_loop.length()):
            atom_idx = int(float(item_loop.values[k * len(ctags) + ci0]))
            for j, comp in val_cols.items():
                v = item_loop.values[k * len(ctags) + j]
                if v not in (".", "?", "nan") and 0 <= atom_idx < width:
                    cond_coord_rows[atom_idx, comp] = float(v)
        break

    entity_poly: dict[str, list[str]] = {}
    for tag in ("entity_id", "type", "pdbx_strand_id"):
        col = [str(v) for v in block.find_loop(f"_entity_poly.{tag}")]
        entity_poly[tag] = col

    # covalent links (_struct_conn, conn_type_id covale): (chain1, res_id1, atom1,
    # chain2, res_id2, atom2, order) — matched against the atom_site rows
    covale: list[tuple[str, int, str, str, int, str, int]] = []
    sc_tab = block.find_mmcif_category('_struct_conn.')
    sc_tags = [t.split('.', 1)[1] for t in sc_tab.tags] if len(sc_tab.tags) else []
    if sc_tags:
        col_candidates = {
            "type": ["conn_type_id"],
            "c1": ["ptnr1_auth_asym_id", "ptnr1_label_asym_id"],
            "c2": ["ptnr2_auth_asym_id", "ptnr2_label_asym_id"],
            "r1": ["ptnr1_auth_seq_id", "ptnr1_label_seq_id"],
            "r2": ["ptnr2_auth_seq_id", "ptnr2_label_seq_id"],
            "a1": ["ptnr1_auth_atom_name", "ptnr1_label_atom_id"],
            "a2": ["ptnr2_auth_atom_name", "ptnr2_label_atom_id"],
            "ord": ["value_order", "pdbx_value_order"],
        }
        cols = {role: next((sc_tags.index(c) for c in cands if c in sc_tags), None)
                for role, cands in col_candidates.items()}
        if all(cols[r] is not None for r in ("type", "c1", "c2", "r1", "r2", "a1", "a2")):
            for k in range(len(sc_tab)):
                row = [sc_tab[k].str(j) for j in range(len(sc_tags))]
                if row[cols["type"]].lower() != "covale":
                    continue
                try:
                    order_str = row[cols["ord"]].lower() if cols["ord"] is not None else "sing"
                    order = {"sing": 1, "doub": 2, "trip": 3}.get(order_str, 1)
                    covale.append((row[cols["c1"]], int(float(row[cols["r1"]])), row[cols["a1"]],
                                   row[cols["c2"]], int(float(row[cols["r2"]])), row[cols["a2"]], order))
                except ValueError:
                    continue

    # multi-model files: keep the first model only (as the pipeline's parse does)
    if "pdbx_PDB_model_num" in atom_cols:
        models = atom_cols["pdbx_PDB_model_num"]
        first = models[0]
        rows_keep = [k for k in range(width) if models[k] == first]
        atom_cols = {k: [v[i] for i in rows_keep] for k, v in atom_cols.items()}
        annotations = {k: [v[i] for i in rows_keep] for k, v in annotations.items()}
        width = len(rows_keep)

    n = width

    def _unquote(v: str) -> str:
        # CIF quoting: strip matching surrounding quotes without dropping
        # significant characters (e.g. a trailing prime in O5')
        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("'", '"'):
            return v[1:-1]
        return v

    atoms = {
        "group": atom_cols["group_PDB"],
        "element": [_unquote(e).strip() for e in atom_cols["type_symbol"]],
        "atom_name": [_unquote(a).strip() for a in atom_cols["label_atom_id"]],
        "res_name": [_unquote(r).strip() for r in atom_cols["label_comp_id"]],
        "chain": [_unquote(c).strip() for c in atom_cols["label_asym_id"]],
        "entity": [int(e) if e not in (".", "?") else 0 for e in atom_cols["label_entity_id"]],
        "res_id": [int(s) if s not in (".", "?") else 0 for s in atom_cols["label_seq_id"]],
        "auth_res_id": [int(s) if s not in (".", "?") else 0 for s in atom_cols["auth_seq_id"]],
        "coords": np.array([[float(x) if x not in (".", "?", "nan") else np.nan
                             for x in (atom_cols["Cartn_x"][k], atom_cols["Cartn_y"][k],
                                       atom_cols["Cartn_z"][k])] for k in range(n)], dtype=np.float32),
        "occupancy": np.array([float(o) if o not in (".", "?") else 0.0
                               for o in atom_cols["occupancy"]], dtype=np.float32),
        "pdbx_formal_charge": [c.strip() for c in atom_cols.get("pdbx_formal_charge", ["?"] * n)],
    }
    return {"atoms": atoms, "annotations": annotations, "entity_poly": entity_poly,
            "cond_coord_rows": cond_coord_rows, "covale": covale,
            "n_atoms": n, "atom_tags": tags, "loop": loop}


# ------------------------------------------------------- expandable-segment realization
def _realize_expseg(atoms: dict, annotations: dict[str, list[str]]) -> tuple[dict, dict, int]:
    """Expand sentinel rows (annotation_expsegmin/max) into MASKED stubs.

    Returns (atoms, annotations, n_randint_draws) — the global numpy stream consumes
    one randint for the local rng, then per-sentinel draws from that local rng.
    """
    expseg = annotations.get("annotation_expsegmin_1_residue")
    expseg_max_col = annotations.get("annotation_expsegmax_1_residue")
    if expseg is None:
        atoms["is_stub"] = np.zeros(len(atoms["atom_name"]), dtype=bool)
        return atoms, annotations, 0
    rng = np.random.default_rng(np.random.randint(0, 2**32))  # global-stream draw #1
    new_atoms = {k: [] for k in atoms if k != "coords"}
    new_coords = []
    cond_rows = []
    is_stub = []
    src_cond = atoms.get("condition_coordinate_loop",
                         np.full((len(atoms["atom_name"]), 3), np.nan, np.float32))
    new_ann: dict[str, list[str]] = {k: [] for k in annotations}
    n_draws = 1

    def is_sentinel(i: int) -> bool:
        v = expseg[i].strip()
        return v not in (".", "?", "-1", "") and int(float(v)) >= 0

    # residue identity: each realized stub is its OWN residue; renumber per chain
    res_group: list[int] = []
    chain_of = atoms["chain"]
    last_key = None
    counter = -1
    for i in range(len(atoms["atom_name"])):
        key = chain_of[i]
        if is_sentinel(i):
            lo, hi = int(float(expseg[i])), int(float(expseg_max_col[i]))
            n = int(rng.integers(lo, hi + 1))
            for _ in range(n):
                counter += 1
                for k in new_atoms:
                    new_atoms[k].append(atoms[k][i])
                new_coords.append([np.nan, np.nan, np.nan])
                for k in new_ann:
                    new_ann[k].append(annotations[k][i])
                cond_rows.append([np.nan, np.nan, np.nan])
                is_stub.append(True)
                res_group.append(counter)
            continue
        auth_res = atoms["res_id"][i]
        if (key, auth_res) != last_key:
            counter += 1
            last_key = (key, auth_res)
        for k in new_atoms:
            new_atoms[k].append(atoms[k][i])
        new_coords.append(atoms["coords"][i].tolist())
        for k in new_ann:
            new_ann[k].append(annotations[k][i])
        cond_rows.append(src_cond[i].tolist())
        is_stub.append(False)
        res_group.append(counter)
    new_atoms["coords"] = np.array(new_coords, dtype=np.float32)
    new_atoms["condition_coordinate_loop"] = np.array(cond_rows, dtype=np.float32)
    new_atoms["is_stub"] = np.array(is_stub, dtype=bool)
    # renumber res_ids consecutively per chain in residue-group order
    chains = new_atoms["chain"]
    group_renumber: dict[int, int] = {}
    prev_chain = None
    for g, ch in zip(res_group, chains):
        if ch != prev_chain:
            prev_chain = ch
            next_id = 0
        if g not in group_renumber:
            group_renumber[g] = next_id
            next_id += 1
    new_atoms["res_id"] = [group_renumber[g] for g in res_group]
    return new_atoms, new_ann, n_draws


# ------------------------------------------------------------------ CCD bond graph
def _build_bonds(atoms: dict, ccd_entries: dict,
                 unit: np.ndarray | None = None,
                 covale_links: list | None = None,
                 token_type_atom: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Atom-level bond graph: (bonds [B,4] int32 (i, j, biotite type, aromatic), mask).

    Intra-residue bonds come from the CCD cache (Kekulé + aromatic flags mapped to
    biotite bond-type ids); inter-residue polymer links (consecutive residues in a
    chain) connect each residue's leaving atom to the next residue's entering atom
    (per-CCD polymerization atoms: protein C→N, DNA/RNA O3'→P); residue pairs
    without linking atoms (e.g. MASKED stubs) link their representative atoms.
    Covalent cross-component links come from the CIF's `_struct_conn` covale
    records.

    ``unit`` (from the inference unindexing): polymer and covale links are only
    formed when both residues share a unit (indexed region or the same unindexed
    index group) — the unindexed copies are per-group slices and lose cross-group
    links, matching the pipeline's BondList after UnindexFlaggedTokens.
    """
    bonds: list[tuple[int, int, int]] = []
    n = len(atoms["atom_name"])
    elements = [e.upper() for e in atoms["element"]]
    res_names = atoms["res_name"]
    chains = atoms["chain"]
    res_ids = atoms["res_id"]

    # group atoms per residue (chain, res_id) preserving file order
    residue_keys: list[tuple[str, int]] = []
    residue_atoms: dict[tuple[str, int], list[int]] = {}
    for i in range(n):
        key = (chains[i], res_ids[i])
        if key not in residue_atoms:
            residue_keys.append(key)
            residue_atoms[key] = []
        residue_atoms[key].append(i)

    for key, atom_idxs in residue_atoms.items():
        res_name = res_names[atom_idxs[0]]
        entry = ccd_entries.get(res_name)
        if entry is None:
            continue  # MASKED stubs etc. carry no CCD component
        name_to_local = {atoms["atom_name"][i]: i for i in atom_idxs}
        for a1, a2, order, arom in zip(entry.bond_i, entry.bond_j, entry.bond_order,
                                       entry.bond_aromatic):
            n1, n2 = entry.atom_names[a1].decode(), entry.atom_names[a2].decode()
            if n1 not in name_to_local or n2 not in name_to_local:
                continue
            e1, e2 = elements[name_to_local[n1]], elements[name_to_local[n2]]
            if e1 in ("H", "D") or e2 in ("H", "D"):
                continue
            if arom:
                bt = {1: BT_AROMATIC_SINGLE, 2: BT_AROMATIC_DOUBLE, 3: 7}.get(int(order), BT_AROMATIC)
            else:
                bt = int(order) if int(order) in (1, 2, 3) else BT_SINGLE
            bonds.append((name_to_local[n1], name_to_local[n2], bt, int(bool(arom))))

    # inter-residue polymer links: consecutive res_ids within a chain, connecting
    # the leaving atom of residue i to the entering atom of residue i+1 (per-CCD
    # polymerization atoms; protein C→N, DNA/RNA O3'→P)
    from atomworks.io.utils.ccd import get_polymerization_atoms as _poly_atoms
    _link_cache: dict[tuple[str, int], tuple[str, str]] = {}

    def _link(rn: str, tt: int) -> tuple[str, str]:
        key = (rn.strip().upper(), int(tt))
        if key not in _link_cache:
            leaving, entering = _poly_atoms(key[0])
            if leaving is None and entering is None:
                unknown = {0: "UNK", 1: "N", 2: "DN"}.get(int(tt))
                if unknown is not None:
                    leaving, entering = _poly_atoms(unknown)
            _link_cache[key] = (leaving or "", entering or "")
        return _link_cache[key]

    chain_first_idx: dict = {}
    if isinstance(chains, list):
        for idx_c, ch_c in enumerate(chains):
            chain_first_idx.setdefault(ch_c, idx_c)
    order_keys = sorted(residue_atoms.keys(), key=lambda k: (chain_first_idx.get(k[0], 0), k[1]))
    for (c1, r1), (c2, r2) in zip(order_keys[:-1], order_keys[1:]):
        if c1 != c2 or r2 != r1 + 1:
            continue
        ia, ib = residue_atoms[(c1, r1)], residue_atoms[(c2, r2)]
        names_a = {atoms["atom_name"][i].strip(): i for i in ia}
        names_b = {atoms["atom_name"][i].strip(): i for i in ib}
        tt_a = int(token_type_atom[ia[0]]) if token_type_atom is not None else 0
        tt_b = int(token_type_atom[ib[0]]) if token_type_atom is not None else 0
        leave_name, _ = _link(atoms["res_name"][ia[0]], tt_a)
        _, enter_name = _link(atoms["res_name"][ib[0]], tt_b)
        rep_a = names_a.get(leave_name, ia[0] if ia else None)   # leaving atom (fallback: representative)
        rep_b = names_b.get(enter_name, ib[0] if ib else None)   # entering atom (fallback: representative)
        if rep_a is not None and rep_b is not None:
            if unit is not None and unit[rep_a] != unit[rep_b]:
                continue  # link crosses the unindexing excision / group boundary
            bonds.append((rep_a, rep_b, BT_SINGLE, 0))

    # covalent cross-component links from _struct_conn (covale records)
    if covale_links:
        atom_lookup: dict[tuple[str, int, str], int] = {}
        for i in range(n):
            # struct_conn uses AUTH seq ids; index both auth and label res_ids
            for rid in {int(atoms["auth_res_id"][i]), int(res_ids[i])}:
                atom_lookup.setdefault((atoms["chain"][i], rid, atoms["atom_name"][i].strip()), i)
        existing_pairs = {(int(a), int(b)) for a, b, _t, _ar in bonds}
        for c1, r1, a1n, c2, r2, a2n, order in covale_links:
            ia = atom_lookup.get((c1, r1, a1n.strip()))
            ib = atom_lookup.get((c2, r2, a2n.strip()))
            if ia is None or ib is None:
                continue
            if ia == ib or (min(ia, ib), max(ia, ib)) in existing_pairs:
                continue  # duplicate of a CCD/peptide bond
            if elements[ia] in ("H", "D") or elements[ib] in ("H", "D"):
                continue
            if unit is not None and unit[ia] != unit[ib]:
                continue
            bonds.append((ia, ib, int(order), 0))
            existing_pairs.add((min(ia, ib), max(ia, ib)))

    bonds_arr = np.array(bonds, dtype=np.int32).reshape(-1, 4)
    return bonds_arr, np.ones(n, dtype=bool)


_CATH_MAPPING_CACHE: dict | None = None

# ── chirality (AddChiralityFeature semantics; numpy reimplementation) ──────

# Ideal polyhedra (rfproteina.utils.chirals.IDEAL_POLYHEDRA)
_IDEAL_POLYHEDRA = {
    "tetrahedral": np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], np.float32) / np.sqrt(3),
    "square_planar": np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], np.float32),
    "trigonal_bipyramidal": np.array(
        [[1, 0, 0], [-0.5, np.sqrt(3) / 2, 0], [-0.5, -np.sqrt(3) / 2, 0], [0, 0, 1], [0, 0, -1]], np.float32),
    "octahedral": np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32),
}
# Planar E/Z templates keyed by (side, n_left, n_right) (DOUBLE_BOND_TEMPLATES)
_DOUBLE_BOND_TEMPLATES = {
    (side, nl, nr): np.array(
        [[-0.5, 0, 0], [0.5, 0, 0]]
        + [[-1, np.sqrt(3) / 2, 0], [-1, -np.sqrt(3) / 2, 0]][:nl]
        + [[1, side * np.sqrt(3) / 2, 0], [1, -side * np.sqrt(3) / 2, 0]][:nr], np.float32)
    for side in (1, -1) for nl in (1, 2) for nr in (1, 2)
}
# One representative per proper-rotation orbit of all labelings (inference-stable
# subset: only used via classify_stereoisomer's argmin over rmsds)
import itertools as _it

_STEREOISOMER_TEMPLATES = {
    (geom, k): [np.asarray(t, np.float32) for t in
                dict.fromkeys(tuple(tuple(row) for row in p)
                              for p in [full[:k][list(perm)]
                                        for perm in _it.permutations(range(k))])]
    for geom, full in _IDEAL_POLYHEDRA.items() for k in range(3, len(full) + 1)
}

_STEREO_CACHE: dict[str, dict] = {}


def _kabsch_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation R minimizing ||target_i − R @ source_i|| (numpy Kabsch).

    Matches scipy's ``Rotation.align_vectors(target, source)`` + ``rot.apply``
    (row-vector convention: ``source @ R.T ≈ target``).
    """
    src = np.asarray(source, np.float64)
    tgt = np.asarray(target, np.float64)
    h = src.T @ tgt
    u, _s, vt = np.linalg.svd(h)
    det_s = float(np.sign(np.linalg.det(vt.T @ u.T)))
    d = np.eye(3) * (1.0 if det_s == 0.0 else det_s)
    r = (vt.T @ d @ u.T).astype(np.float64)
    return r


def _kabsch_rmsd(source: np.ndarray, target: np.ndarray) -> float:
    r = _kabsch_rotation(source, target)
    fitted = np.asarray(source, np.float64) @ r.T
    return float(np.sqrt(((fitted - np.asarray(target, np.float64)) ** 2).sum() / len(source)))


def _classify_stereoisomer(centered: np.ndarray, geometry: str) -> int | None:
    """Index of the best-matching stereoisomer template (argmin rmsd)."""
    templates = _STEREOISOMER_TEMPLATES.get((geometry, len(centered)))
    if templates is None or not np.isfinite(centered).all():
        return None
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    rmsds = np.array([_kabsch_rmsd(unit, t) for t in templates])
    return int(np.argmin(rmsds))


def _idealize_coords(centered: np.ndarray, geometry: str) -> np.ndarray:
    """Replace neighbor coords with the best-matching ideal polyhedron, rotated
    onto the input frame (inference: deterministic, no random rotation)."""
    idx = _classify_stereoisomer(centered, geometry)
    if idx is None:
        return centered.astype(np.float32)
    template = _STEREOISOMER_TEMPLATES[(geometry, len(centered))][idx]
    unit = centered / np.maximum(np.linalg.norm(centered, axis=1, keepdims=True), 1e-8)
    rot = _kabsch_rotation(template, unit)
    return (template @ rot.T).astype(np.float32)


def _classify_double_bond_side(coords: np.ndarray) -> int | None:
    """+1 same-side / −1 opposite-side substituents about a double bond.

    coords (4,3): left endpoint, right endpoint, left substituent, right substituent.
    """
    if not np.isfinite(coords).all():
        return None
    axis = coords[1] - coords[0]
    length = np.linalg.norm(axis)
    if length < 1e-6:
        return None
    axis = axis / length
    substituents = coords[2:] - coords[:2]
    projected = substituents - (substituents @ axis)[:, None] * axis
    lengths = np.linalg.norm(projected, axis=1)
    if np.any(lengths < 1e-6):
        return None
    cosine = np.dot(projected[0], projected[1]) / np.prod(lengths)
    if abs(cosine) < 0.5:
        return None
    return 1 if cosine > 0 else -1


_CCD_STEREO_LOOKUP = None


def _ccd_stereo(res_name: str) -> dict | None:
    """Reference stereochemistry for a CCD code (chiral centers + E/Z double bonds).

    Delegates to the reference implementation's RDKit-perceived, lru-cached lookup
    (`atomworks.ml.utils.annotator._get_stereochemistry_for_ccd_code`): it is keyed
    by CCD code only (structure-independent), like the CCD cache itself.
    """
    global _CCD_STEREO_LOOKUP
    if res_name in _STEREO_CACHE:
        return _STEREO_CACHE[res_name]
    try:
        if _CCD_STEREO_LOOKUP is None:
            from atomworks.ml.utils.annotator import _get_stereochemistry_for_ccd_code
            _CCD_STEREO_LOOKUP = _get_stereochemistry_for_ccd_code
        stereo = _CCD_STEREO_LOOKUP(res_name)
    except Exception:
        stereo = None
    _STEREO_CACHE[res_name] = stereo
    return stereo


def _residue_stereo_from_atoms(
    res_name: str, atom_names: list[str], elements: list[str], coords: np.ndarray,
    charges: list[int], bond_list: list[tuple[int, int, int, bool]],
) -> dict | None:
    """Per-residue stereo fallback: build an RDKit mol from the residue's atoms
    (reference: `atom_array_to_rdkit(atom_array[start:stop], set_coord=True)` when
    the CCD-code lookup fails), then perceive stereochemistry from 3D."""
    try:
        from rdkit import Chem
        from atomworks.ml.transforms.rdkit_utils import get_stereochemistry

        mol = Chem.RWMol()
        import gemmi
        for nm, el, q in zip(atom_names, elements, charges):
            rdatom = Chem.Atom(int(gemmi.Element(el.upper()).atomic_number))
            rdatom.SetFormalCharge(int(q))
            rdatom.SetProp("atom_name", nm)
            mol.AddAtom(rdatom)
        if np.isfinite(coords).all():
            conf = Chem.Conformer(len(atom_names))
            for k, xyz in enumerate(np.asarray(coords, np.float64)):
                conf.SetAtomPosition(k, xyz.tolist())
            mol.AddConformer(conf, assignId=True)
        for a1, a2, order, arom in bond_list:
            if arom:
                bt = Chem.BondType.AROMATIC
            else:
                bt = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE,
                      3: Chem.BondType.TRIPLE}.get(int(order), Chem.BondType.SINGLE)
            mol.AddBond(int(a1), int(a2), bt)
        m = mol.GetMol()
        Chem.SanitizeMol(m)
        return get_stereochemistry(m)
    except Exception:
        return None


def _build_chirality_features(
    atoms: dict, bonds: np.ndarray, token_of_atom: np.ndarray, atomize_atom: np.ndarray,
    cond_coord: np.ndarray, cond_groups_atom: np.ndarray, coords: np.ndarray,
    n_tokens: int, token_starts: np.ndarray, ccd_entries_ref: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-token chirality features for atomized residues (AddChiralityFeature).

    Chiral centers / stereogenic double bonds of ATOMIZED residues get idealized
    local-geometry frames written into 3-vector blocks of an 18-dim (6-block)
    per-atom feature, with cycling fill into empty blocks; token values are the
    first atom's row. Non-atomized residues are skipped by the reference, so
    their rows stay zero.
    Returns (chirality [n_tokens, 18], chiral_frame_atoms [n_tokens, 6, 8],
    chiral_slot_owner [n_tokens, 6]).
    """
    n = len(atoms["atom_name"])
    chir_at = np.zeros((n, 18), np.float32)
    frames_at = np.full((n, 6, 8), -1, np.int64)
    owners_at = np.full((n, 6), -1, np.int64)
    if not atomize_atom.any():
        return chir_at[token_starts], frames_at[token_starts], owners_at[token_starts]

    # undirected neighbor lists
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for a1, a2, _bt, _ar in bonds:
        neighbors[int(a1)].append(int(a2))
        neighbors[int(a2)].append(int(a1))
    neighbors = [sorted(set(nb)) for nb in neighbors]

    from atomworks.enums import ChiralType

    # residue groups (consecutive (chain, res_id) runs) for stereo lookups
    runs: list[tuple[str, int, int, int]] = []  # (res_name, start, stop)
    start = 0
    for i in range(1, n + 1):
        if i == n or (chains_key := (atoms["chain"][i], atoms["res_id"][i])) != \
                (atoms["chain"][start], atoms["res_id"][start]):
            runs.append((atoms["res_name"][start], start, i))
            start = i

    centers = []  # (cidx, bonded, geometry) globally ordered by atom index
    double_bonds = []  # (left, right, left_nb, right_nb)
    for res_name, rs, re in runs:
        if not atomize_atom[rs:re].any() or res_name == MASKED_RES:
            continue
        res_name_u = res_name.strip().upper()
        stereo = _ccd_stereo(res_name_u)
        if not stereo:
            # reference fallback: RDKit mol from the residue's own atoms
            atom_names = [atoms["atom_name"][i].strip() for i in range(rs, re)]
            charge_list = []
            for i in range(rs, re):
                entry = ccd_entries_ref.get(res_name_u)
                k = entry.atom_index(atom_names[i - rs]) if entry is not None else None
                charge_list.append(int(entry.charges[k]) if k is not None else 0)
            bond_list = [(int(a1) - rs, int(a2) - rs, int(t), bool(ar))
                         for a1, a2, t, ar in bonds.tolist()
                         if rs <= int(a1) < re and rs <= int(a2) < re]
            stereo = _residue_stereo_from_atoms(
                res_name_u, atom_names, [atoms["element"][i] for i in range(rs, re)],
                atoms["coords"][rs:re], charge_list, bond_list)
        if not stereo:
            continue
        name_to_idx = {atoms["atom_name"][i].strip(): i for i in range(rs, re)}
        for rec in stereo.get("chiral_centers", []):
            cname = rec.get("chiral_center_atom_name")
            if cname is None or cname not in name_to_idx:
                continue
            cidx = name_to_idx[cname]
            ct = ChiralType(int(rec["chiral_type"]))
            if not ct.is_stereogenic():
                continue
            geometry = ct.to_geometry()
            bonded = sorted(neighbors[cidx])
            if (geometry, len(bonded)) not in _STEREOISOMER_TEMPLATES:
                continue
            centers.append((cidx, bonded, geometry))
        # stereogenic double bonds: only assigned E/Z bonds, both endpoints atomized
        for rec in stereo.get("double_bonds", []):
            names = rec["atom_names"][:2]
            if any(nm not in name_to_idx for nm in names):
                continue
            left, right = sorted(name_to_idx[nm] for nm in names)
            if not (atomize_atom[left] and atomize_atom[right]):
                continue
            ln = sorted(i for i in neighbors[left] if i != right)
            rn_ = sorted(i for i in neighbors[right] if i != left)
            if not (1 <= len(ln) <= 2 and 1 <= len(rn_) <= 2):
                continue
            double_bonds.append((left, right, ln, rn_))

    def _resolve_coords(idxs: list[int]) -> np.ndarray | None:
        # prefer C_CRD, then C_INTCRD (absent here), else raw coordinates
        if np.isfinite(cond_coord[idxs]).all():
            return cond_coord
        return coords

    atom_blocks: dict[int, list[int]] = {}

    def _place(is_center: bool, cidx: int, bonded: list[int], idealized: np.ndarray) -> None:
        for block in range(6):
            offset = block * 3
            if not np.any(chir_at[bonded, offset:offset + 3]):
                origin = (cidx, cidx) if is_center else (int(bonded[0]), int(bonded[1]))
                if atomize_atom[[*origin, *bonded]].all():
                    frames_at[cidx, block, : 2 + len(bonded)] = token_of_atom[[*origin, *bonded]]
                    owners_at[bonded, block] = token_of_atom[cidx] * 6 + block
                chir_at[bonded, offset:offset + 3] = idealized
                for b in bonded:
                    atom_blocks.setdefault(b, []).append(block)
                break

    for cidx, bonded, geometry in centers:
        idxs = [cidx, *bonded]
        src = _resolve_coords(idxs)
        if src is None:
            continue
        centered = src[bonded] - src[cidx]
        _place(True, cidx, bonded, _idealize_coords(centered, geometry))

    for left, right, ln, rn_ in double_bonds:
        indices = [left, right, *ln, *rn_]
        src = _resolve_coords(indices)
        if src is None or not np.isfinite(src[indices]).all():
            continue
        side = _classify_double_bond_side(src[[left, right, ln[0], rn_[0]]])
        if side is None:
            continue
        template = _DOUBLE_BOND_TEMPLATES[(side, len(ln), len(rn_))]
        midpoint = (src[left] + src[right]) / 2
        centered = src[indices] - midpoint
        rot = _kabsch_rotation(template, centered)
        idealized = (template @ rot.T).astype(np.float32)
        _place(False, left, indices, idealized)

    # cycling fill: copy occupied blocks into empty ones, in order
    packed = chir_at.reshape(n, 6, 3)
    for atom_idx, blocks in atom_blocks.items():
        occupied = sorted(blocks)
        empty = [b for b in range(6) if b not in blocks]
        if not empty:
            continue
        for i, block in enumerate(empty):
            source_block = occupied[i % len(occupied)]
            packed[atom_idx, block] = packed[atom_idx, source_block]
            owners_at[atom_idx, block] = owners_at[atom_idx, source_block]

    return chir_at[token_starts], frames_at[token_starts], owners_at[token_starts]


_CATH_MAPPING_CACHE: dict | None = None
def _load_cath_mapping() -> dict:
    """The pipeline's CATH label → dense-id mapping (data/cath_label_mapping.pkl).

    Looked up relative to the current working directory (run from the repo root);
    ``RFD4FX_CATH_MAPPING`` overrides the path. Cached per process.
    """
    global _CATH_MAPPING_CACHE
    if _CATH_MAPPING_CACHE is None:
        import os
        import pickle
        default = pathlib.Path("data") / "cath_label_mapping.pkl"
        path = pathlib.Path(os.environ.get("RFD4FX_CATH_MAPPING", default))
        with open(path, "rb") as f:
            _CATH_MAPPING_CACHE = pickle.load(f)
    return _CATH_MAPPING_CACHE


def _remap_cath_labels(labels: list[str]) -> np.ndarray:
    """C_CATH label strings ('c.a.t.h') → [n, 3] dense (class, arch, topology) codes.

    Mirrors the pipeline: unique labels are remapped via the mapping pickle's
    C/A/T dicts (missing level → -1); 'x.x.x.x' (null) → (-1, -1, -1).
    """
    mapping = _load_cath_mapping()
    # level → number of dot-separated fields to drop from the right
    # (extract_cath_code_by_level: "2.60.40.x" --C--> "2", --A--> "2.60", --T--> "2.60.40")
    level_drop = {"C": 3, "A": 2, "T": 1}
    out = np.full((len(labels), 3), -1, dtype=np.int32)
    for i, label in enumerate(labels):
        label = label.strip()
        if label == "x.x.x.x":
            continue
        parts = label.split(".")
        for j, level in enumerate("CAT"):
            drop = level_drop[level]
            key = ".".join(parts[:-drop]) if len(parts) > drop else ""
            out[i, j] = mapping[level].get(key, -1)
    return out


def _try_ccd(ccd_id: str, *, force_cache_update: bool = False):
    try:
        from .ccd import get_ccd_entry
        return get_ccd_entry(ccd_id, force_cache_update=force_cache_update)
    except KeyError:
        return None


# --------------------------------------------------------------------- featurizer
def annotated_cif_to_batch_numpy(
    path: str | pathlib.Path,
    *,
    np_seed: int | None = 0,
    force_cache_update: bool = False,
    force_atomize=None,
) -> tuple[FlowBatch, np.ndarray]:
    """Annotated CIF → (FlowBatch, coord_cond_mask) via gemmi + CCD cache + numpy.

    `force_atomize`: iterable of CCD codes that are treated as atomized (one token
    per atom) even though the RFD4 autoencoder represents them as single residues
    (its 145-token latent alphabet: standard AA + UNK + <M> + A/C/G/U/N + DA/DC/DG/
    DT/DN + 100 NCAAs + 13 glycans). By default every component NOT in that
    alphabet is atomized, matching the reference pipeline's
    `AtomizeByCCDName(atomize_by_default=True, res_names_to_ignore=LOCAL_LATENT_RESIDUES)`.
    """
    import jax.numpy as jnp

    parsed = _read_annotated_cif(path)
    atoms, annotations = parsed["atoms"], parsed["annotations"]
    covale = parsed.get("covale", [])
    atoms["condition_coordinate_loop"] = parsed["cond_coord_rows"]
    if np_seed is not None:
        np.random.seed(np_seed)
    atoms, annotations, n_realize_draws = _realize_expseg(atoms, annotations)

    n = len(atoms["atom_name"])
    elements = [e.upper() for e in atoms["element"]]
    # drop hydrogens (RemoveHydrogens)
    # RemoveHydrogens + RemoveTerminalOxygen (multi-RESIDUE protein chains only)
    chain_res_ids: dict[str, set] = {}
    for i in range(n):
        chain_res_ids.setdefault(atoms["chain"][i], set()).add(atoms["res_id"][i])
    multi_res_chain = {c for c, rids in chain_res_ids.items() if len(rids) > 1}
    keep = [i for i in range(n)
            if elements[i] not in ("H", "D")
            and not (atoms["atom_name"][i] == "OXT" and atoms["chain"][i] in multi_res_chain)]
    atoms = {k: ([v[i] for i in keep] if isinstance(v, list) else v[keep]) for k, v in atoms.items()}
    annotations = {k: [v[i] for i in keep] for k, v in annotations.items()}
    n = len(keep)
    elements = [e.upper() for e in atoms["element"]]
    res_names = atoms["res_name"]
    chains = atoms["chain"]
    res_ids = atoms["res_id"]
    formal_charge = atoms.get("pdbx_formal_charge", ["?"] * n)

    # ---- UnindexFlaggedTokens (inference branch): tokens with any unindexed atom
    # (condition_index False) are removed; for each index group among the unindexed
    # atoms the group's atoms are copied and appended at the end (np.unique order).
    # Bonds survive only within the indexed region or within a single unindexed
    # group — the copies are per-group slices, so e.g. peptide bonds between motif
    # residues of different groups vanish. Copies keep C_CRD/C_SEQ (deep copy of
    # annotations); their index groups are shifted above the indexed range later.
    unit: np.ndarray | None = None
    idx_col = annotations.get("condition_index_1_residue")
    if idx_col is not None and any(v.strip() == "False" for v in idx_col):
        indexed_atom = np.array([v.strip() == "True" for v in idx_col])
        # C_IDX must be uniform across every token (pipeline raises otherwise)
        run_keys: list[tuple[str, int]] = []
        tok_of_run = np.zeros(n, dtype=np.int32)
        for i in range(n):
            key = (chains[i], res_ids[i])
            if not run_keys or run_keys[-1] != key:
                run_keys.append(key)
            tok_of_run[i] = len(run_keys) - 1
        _, run_starts = np.unique(tok_of_run, return_index=True)
        tok_indexed = indexed_atom[run_starts]
        if not np.array_equal(indexed_atom, tok_indexed[tok_of_run]):
            raise ValueError(
                "C_IDX must be uniform across every token; unindexed polymer tokens are full residues.")
        # effective index groups: authored values where present (>=0), else the
        # condition system's per-chain default (chain rank)
        chain_rank_a = {c: i for i, c in enumerate(sorted(dict.fromkeys(chains)))}
        gcol = annotations.get("condition_index-group_1_token")
        eff_group = np.array([
            (int(float(gcol[i]))
             if gcol is not None and gcol[i].strip() not in (".", "?", "")
             and int(float(gcol[i])) >= 0
             else chain_rank_a[chains[i]])
            for i in range(n)])
        unindexed_groups = np.unique(eff_group[~indexed_atom])
        perm = np.concatenate([
            np.flatnonzero(indexed_atom),
            np.concatenate([np.flatnonzero((~indexed_atom) & (eff_group == g))
                            for g in unindexed_groups])])
        atoms = {k: ([v[i] for i in perm] if isinstance(v, list) else v[perm])
                 for k, v in atoms.items()}
        annotations = {k: [v[i] for i in perm] for k, v in annotations.items()}
        # bond-survival unit per atom: -1 = indexed region, else its group id
        unit = np.where(indexed_atom[perm], -1, eff_group[perm])
        n = len(perm)
        elements = [e.upper() for e in atoms["element"]]
        res_names = atoms["res_name"]
        chains = atoms["chain"]
        res_ids = atoms["res_id"]
        formal_charge = atoms.get("pdbx_formal_charge", ["?"] * n)

    # CCD entries for every real component (sentinel rows like "10-40" are not CCD IDs)
    expseg_col = annotations.get("annotation_expsegmin_1_residue")

    def _is_sentinel(i: int) -> bool:
        if expseg_col is None:
            return False
        v = expseg_col[i].strip()
        return v not in (".", "?", "-1", "") and int(float(v)) >= 0

    real = [rn for i, rn in enumerate(res_names) if not _is_sentinel(i)]
    try:
        ccd_entries = get_ccd_entries(set(real), force_cache_update=force_cache_update)
    except KeyError:
        ccd_entries = {cid: rec for cid in set(real)
                       if (rec := _try_ccd(cid, force_cache_update=force_cache_update))}
    # covale records anchored on sentinel (expseg) rows are expansion bookkeeping,
    # not literal bonds — the sentinel realization provides the connectivity
    sentinel_keys = {(chains[i], int(res_ids[i])) for i in range(n) if _is_sentinel(i)}
    sentinel_keys |= {(chains[i], int(atoms["auth_res_id"][i])) for i in range(n) if _is_sentinel(i)}
    covale = [rec for rec in covale
              if (rec[0], int(rec[1])) not in sentinel_keys
              and (rec[3], int(rec[4])) not in sentinel_keys]

    # ---- AtomizeByCCDName (before bond building): every component outside the
    # AE's 145-token latent alphabet is atomized (one token per atom,
    # token_type=3, atom37 slot 1); `force_atomize` extends this to vocabulary
    # components.
    from rfproteina.utils.constants import LOCAL_LATENT_RESIDUES as _VOCAB
    from rfproteina.encoding import RESIDUES_BY_TOKEN_TYPE as _GROUPS
    _force_atomize = {str(s).strip().upper() for s in (force_atomize or [])}
    atomize_atom = np.zeros(n, dtype=bool)
    for i in range(n):
        rn = res_names[i].strip().upper()
        if rn in _force_atomize:
            atomize_atom[i] = True  # explicit override wins, also for sentinel stubs
        elif _is_sentinel(i):
            continue  # expseg sentinel rows realize as MASKED stubs (in vocabulary)
        elif rn not in _VOCAB:
            atomize_atom[i] = True
    # per-atom token types from the vocabulary groups (bonds.py/pipeline
    # semantics): PROTEIN 0 (AA/UNK/<M>), RNA 1, DNA 2, ATOMIZED 3, GLYCAN 5, NCAA 6
    token_type_atom = np.zeros(n, dtype=np.int32)
    res_names_arr = np.array([r.strip().upper() for r in res_names])
    for group, tt_id in (("dna", 2), ("rna", 1), ("ncaa", 6), ("glycan", 5)):
        gnames = [g.upper() for g in _GROUPS[group]]
        token_type_atom[np.isin(res_names_arr, gnames)] = tt_id
    token_type_atom[atomize_atom] = 3  # ATOMIZED overrides

    bonds, _ = _build_bonds(atoms, ccd_entries, unit=unit, covale_links=covale,
                            token_type_atom=token_type_atom)

    # ---- tokenization: (chain, res_id) groups in atom order; atomized residues
    # contribute one token per atom (move_atomized_part_to_end=False: atoms stay
    # in file order)
    token_of_atom = np.zeros(n, dtype=np.int32)
    token_keys: list[tuple[str, int]] = []
    for i in range(n):
        key = (chains[i], res_ids[i])
        if atomize_atom[i] or not token_keys or token_keys[-1] != key:
            token_keys.append(key if not atomize_atom[i] else (key[0], key[1], i))
        token_of_atom[i] = len(token_keys) - 1
    n_tokens = len(token_keys)
    _, token_starts = np.unique(token_of_atom, return_index=True)
    atomize_tok = atomize_atom[token_starts]
    token_type_tok = token_type_atom[token_starts]
    res_names_tok = res_names_arr[token_starts]

    # AddChiralityFeature consumes one global-stream draw for its local rng
    # (before ConvertToAtomLevelProteinaFeatures' permutations), even when the
    # structure carries no chiral centers (inference: deterministic feature)
    np.random.randint(0, 2**32)

    # ---- chain / molecule / entity ids (ConvertToAtomLevelProteinaFeatures semantics)
    authored_chains = sorted(dict.fromkeys(chains))
    chain_rank = {c: i for i, c in enumerate(authored_chains)}
    # molecule grouping (parse-time pn_unit semantics): non-polymer chains that are
    # covalently linked (directly or transitively) share one molecule unit;
    # polymer chains stay separate even when covale-linked to a ligand
    # (AnnotateCovalentModifications keeps modified polymer residues in their own
    # pn unit).
    chain_is_nonpolymer = {
        chain_rank[c]: bool(atomize_atom[np.array([i for i in range(n) if chains[i] == c])].all())
        for c in authored_chains}
    parent = list(range(len(authored_chains)))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for c1, _r1, _a1, c2, _r2, _a2, _o in (covale or []):
        if c1 in chain_rank and c2 in chain_rank:
            ra, rb = chain_rank[c1], chain_rank[c2]
            if chain_is_nonpolymer[ra] and chain_is_nonpolymer[rb]:
                parent[_find(ra)] = _find(rb)
    group_root_order: dict[int, int] = {}
    molecule_group = np.empty(len(authored_chains), dtype=np.int32)
    for ci in range(len(authored_chains)):
        root = _find(ci)
        if root not in group_root_order:
            group_root_order[root] = len(group_root_order)
        molecule_group[ci] = group_root_order[root]
    n_molecules = len(group_root_order)
    chain_perm = np.random.permutation(len(authored_chains)).astype(np.float32)
    molecule_perm = np.random.permutation(n_molecules).astype(np.int32)
    chain_iid = np.array([chain_perm[chain_rank[c]] for c in chains], dtype=np.float32)[token_starts]
    molecule_iid = np.array([molecule_perm[molecule_group[chain_rank[c]]] for c in chains],
                            dtype=np.int32)[token_starts]
    # masked-sequence chains get private entities (is_inference branch), 1-based.
    # A chain is "masked" if ANY of its atoms is an unconditioned non-atomized
    # atom (C_SEQ = <M>); atomized tokens are always sequence-conditioned, so
    # ligand-only chains keep their native molecule-group entity.
    entity = np.empty(n_tokens, dtype=np.int32)
    next_entity = n_molecules
    seq_col_ent = annotations.get("condition_sequence_1_residue")
    masked_tok = np.array([
        (seq_col_ent is None or seq_col_ent[token_starts[t]].strip() == "<M>")
        and not atomize_tok[t]
        for t in range(n_tokens)])
    for ci, c in enumerate(authored_chains):
        mask = np.array([chain_rank[chains[token_starts[t]]] == ci for t in range(n_tokens)])
        chain_masked = bool(masked_tok[mask].any())
        entity[mask] = next_entity if chain_masked else int(molecule_group[ci])
        if chain_masked:
            next_entity += 1

    # ---- atom37 slot lookup (encoding-derived; needed by condition scattering)
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    atom37_slot = np.full(n, -1, dtype=np.int32)
    key_to_slot: dict[str, int] = {}
    from rfproteina.utils.constants import LOCAL_LATENT_RESIDUES
    from atomworks.constants import STANDARD_AA
    for res_name in LOCAL_LATENT_RESIDUES:
        if res_name == MASKED_RES:
            continue
        for slot, an in enumerate(ENC.token_atoms[res_name]):
            an = an.strip()
            if an:
                key_to_slot[f"{res_name}|{an}"] = slot
                if res_name in STANDARD_AA:
                    key_to_slot[f"{UNK_ATOM}|{an}"] = slot
    for i in range(n):
        atom37_slot[i] = key_to_slot.get(f"{res_names[i]}|{atoms['atom_name'][i]}", -1)

    # ---- conditions
    # C_CRD: indexed atoms (condition_index True) with finite coordinates are
    # coordinate-conditioned on their own structure position (group 0) — the
    # semantics the pipeline applies to motif atoms; coordinates are centered on
    # the conditioned-atom centroid (CenterOnCoordinateConditionedAtoms).
    seq_col = annotations.get("condition_sequence_1_residue")
    indexed_atom = np.array([
        (seq_col is not None and seq_col[i].strip() == "True")
        if False else
        (annotations.get("condition_index_1_residue", ["False"] * n)[i].strip() == "True")
        for i in range(n)])
    loop_cond = atoms.get("condition_coordinate_loop",
                          np.full((n, 3), np.nan, np.float32))
    if np.isfinite(loop_cond).any():
        # the _condition_coordinate loop is the authoritative conditioning source
        cond_atom = np.isfinite(loop_cond).all(axis=-1)
        cond_source = loop_cond
    else:
        # no loop at all → NO coordinate conditioning (C_CRD mask all False),
        # regardless of condition_index
        cond_atom = np.zeros(n, dtype=bool)
        cond_source = atoms["coords"]
    # CenterOnCoordinateConditionedAtoms: mean of the STRUCTURE coordinates of the
    # conditioned atoms; with no conditioning the center is the mean over ALL
    # atoms with NaN coordinates counting as 0 (the atom37 zero-fill — verified
    # on A1EJ2: stub slot-1 zeros participate); conditioning values shift by
    # their own centroid
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if cond_atom.any():
            center = np.nanmean(atoms["coords"][cond_atom], axis=0)
        else:
            center = np.nan_to_num(atoms["coords"], nan=0.0).mean(axis=0)
    atoms["coords"] = atoms["coords"] - center
    cond_groups_atom = np.where(cond_atom, 0, -1)

    # conditioning VALUES are centered on their own centroid (the pipeline's
    # shifted C_CRD is exactly zero-mean, independent of the structure centering)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cond_center = (np.nanmean(cond_source[cond_atom], axis=0) if cond_atom.any()
                       else np.zeros(3, np.float32))
    condition_coordinate = np.where(
        cond_atom[:, None], cond_source - cond_center, np.nan).astype(np.float32)
    cond_coords = np.full((n_tokens, 37, 3), np.nan, np.float32)
    coord_cond_mask = np.zeros(n_tokens, dtype=bool)
    coord_groups = np.full((n_tokens, 37), -1, np.int32)

    def token_value(name: str, cast=float, default=None):
        col = annotations.get(name)
        return np.array([cast(col[token_starts[t]] if col is not None else default)
                         for t in range(n_tokens)])

    def atom_ann_like(name: str, default):
        col = annotations.get(name)
        if col is None:
            return np.full(n, default)
        return np.array([cast_or(col[i], default) for i in range(n)])

    def cast_or(v, default):
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return default

    # index groups: authored values where present (>=0); unauthored tokens get
    # their chain's rank (the condition system's per-chain default). Unindexed
    # copies' groups are shifted above the indexed range
    # (UnindexFlaggedTokens: += max_indexed - min_unindexed + 1).
    authored_idx_group = atom_ann_like("condition_index-group_1_token", -1)
    ig_auth = np.array([authored_idx_group[token_starts[t]] for t in range(n_tokens)])
    chain_rank_tok = np.array([chain_rank[chains[token_starts[t]]] for t in range(n_tokens)])
    idx_group = np.where(ig_auth >= 0, ig_auth, chain_rank_tok).astype(np.int32)
    if unit is not None:
        tok_unit = np.array([unit[token_starts[t]] for t in range(n_tokens)])
        is_copy = tok_unit >= 0
        if is_copy.any() and (~is_copy).any():
            shift = int(idx_group[~is_copy].max() - idx_group[is_copy].min() + 1)
            idx_group = np.where(is_copy, idx_group + shift, idx_group)
    # in-context conditioning: tokens whose atoms are unindexed (the appended copies)
    idx_col_tok = annotations.get("condition_index_1_residue")
    tok_in_context = np.array([
        (idx_col_tok is not None and idx_col_tok[token_starts[t]].strip() == "False")
        for t in range(n_tokens)])

    host: dict[str, np.ndarray] = {
        "padding_mask": np.ones((1, n_tokens), dtype=bool),
        "token_type": np.zeros((1, n_tokens), dtype=np.int32),
        "condition_coordinate_1_atom": cond_coords[None],
        "coord_cond_mask": coord_cond_mask,
        "condition_CoordinateGroup_1_atom": coord_groups[None],
        "condition_InternalCoordinate_1_atom": np.full((1, n_tokens, 37, 3), np.nan, np.float32),
        "condition_rasa-binned_1_atom": np.zeros((1, n_tokens, 37), np.int32),
        "condition_hbond_1_atom": np.zeros((1, n_tokens, 37), np.int32),
        "condition_hotspot_1_atom": np.zeros((1, n_tokens, 37), dtype=bool),
        "condition_metal-coord_1_atom": np.zeros((1, n_tokens, 37), np.int32),
        "condition_pistack_1_atom": np.zeros((1, n_tokens, 37), dtype=bool),
        "condition_pication_1_atom": np.zeros((1, n_tokens, 37), np.int32),
        "condition_saltbridge_1_atom": np.zeros((1, n_tokens, 37), np.int32),
        "condition_hydrophobic_1_atom": np.zeros((1, n_tokens, 37), dtype=bool),
        "condition_ChainCentroid_1_atom": np.full((1, n_tokens, 3), np.nan, np.float32),
        "condition_index-group_1_token": idx_group[None],
        "condition_secondary-structure_1_residue": token_value(
            "condition_secondary-structure_1_residue", int, 0).astype(np.int32)[None],
        "chain_iid": chain_iid[None],
        "molecule_iid": molecule_iid[None],
        "molecule_entity_id": entity[None],
        "is_in_context_conditioning": tok_in_context[None],
        "canonical_polymer_ordering": np.zeros((1, n_tokens, n_tokens), np.int32),
        "token_bond_path_length": np.zeros((1, n_tokens, n_tokens), np.int32),
        "bond_mask": np.zeros((1, n_tokens, n_tokens), dtype=bool),
        "bond_order": np.zeros((1, n_tokens, n_tokens), np.int32),
        "is_aromatic": np.zeros((1, n_tokens, n_tokens), dtype=bool),
        "na_ss_state": np.zeros((1, n_tokens), np.float32),
        "na_ss_state_mask": np.zeros((1, n_tokens), dtype=bool),
        "na_ss_pair": np.zeros((1, n_tokens, n_tokens), np.float32),
        "na_ss_pair_mask": np.zeros((1, n_tokens, n_tokens), dtype=bool),
        "charge": np.zeros((1, n_tokens, 37), np.int32),
        "element": np.full((1, n_tokens, 37), -1, np.int32),
        "ground_truth_element": np.zeros((1, n_tokens, 37), np.int32),
        "incident_bond_type_counts": np.zeros((1, n_tokens, 37, 10), np.float32),
        "chirality": np.zeros((1, n_tokens, 18), np.float32),
        "chiral_frame_atoms": np.full((1, n_tokens, 6, 8), -1, np.int32),
        "chiral_slot_owner": np.full((1, n_tokens, 6), -1, np.int32),
    }

    # C_CATH fold labels: 'x.x.x.x' is null (-1). Real labels are remapped to
    # dense ids through the pipeline's CATH mapping pickle (C/A/T level lookups
    # on the dot-separated hierarchy); token-level via first atom of token.
    host["token_type"] = token_type_tok[None]
    cath_col = annotations.get("condition_cath-label_1_residue")
    if cath_col is not None:
        host["condition_cath-label_1_residue"] = _remap_cath_labels(
            [cath_col[token_starts[t]] for t in range(n_tokens)])[None]

    # condition columns: atom37-channel conditions (spread per-atom values)
    _ATOM37_CONDITIONS = {
        "condition_rasa-binned_1_atom": "condition_rasa-binned_1_atom",
        "condition_hbond_1_atom": "condition_hbond_1_atom",
        "condition_hotspot_1_atom": "condition_hotspot_1_atom",
        "condition_metal-coord_1_atom": "condition_metal-coord_1_atom",
        "condition_pistack_1_atom": "condition_pistack_1_atom",
        "condition_pication_1_atom": "condition_pication_1_atom",
        "condition_saltbridge_1_atom": "condition_saltbridge_1_atom",
        "condition_hydrophobic_1_atom": "condition_hydrophobic_1_atom",
    }
    bool_conds = {"condition_hotspot_1_atom", "condition_pistack_1_atom", "condition_hydrophobic_1_atom"}
    for col_name in annotations:
        if col_name in _ATOM37_CONDITIONS:
            vals = annotations[col_name]
            arr = np.zeros((n, 37), dtype=bool if col_name in bool_conds else np.int32)
            for i in range(n):
                slot = 1 if atomize_atom[i] else atom37_slot[i]
                if slot < 0:
                    continue
                v = vals[i].strip()
                if v in (".", "?"):
                    continue
                arr[i, slot] = (v == "True") if col_name in bool_conds else int(float(v))
            tok_arr = (np.maximum.reduceat(arr, token_starts, axis=0)
                       if n_tokens > 0 else np.zeros((0, 37), dtype=arr.dtype))
            host[col_name] = tok_arr[None]
        elif col_name == "condition_sequence_1_residue":
            from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC  # numpy-only

            seq = np.zeros(n_tokens, dtype=np.int32)
            for t in range(n_tokens):
                v = annotations[col_name][token_starts[t]].strip()
                seq[t] = ENC.token_to_idx.get(v, 0)
            # identity-conditioned tokens keep their encoded residue name: atomized
            # (<A>), and DNA/RNA/NCAA/glycan tokens (the AE cannot generate their
            # identity, so C_SEQ.mask is True for them even without an authored
            # column); standard-AA/UNK/<M> tokens stay <M> unless authored
            identity_types = np.isin(token_type_tok, (1, 2, 3, 5, 6))
            authored = np.zeros(n_tokens, dtype=bool)
            for t in range(n_tokens):
                v = annotations[col_name][token_starts[t]].strip()
                authored[t] = v not in (".", "?", "")
            fix = identity_types & ~authored
            names = np.array([res_names[token_starts[t]].strip().upper() for t in range(n_tokens)])
            for t in np.flatnonzero(fix):
                seq[t] = ENC.token_to_idx.get(names[t], ENC.token_to_idx["<A>"])
            seq[atomize_tok] = ENC.token_to_idx["<A>"]
            host["condition_sequence_1_residue"] = seq[None]
    if "condition_sequence_1_residue" not in host:
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC  # numpy-only
        seq = np.zeros(n_tokens, dtype=np.int32)
        # identity-conditioned tokens keep their encoded residue name (no column)
        identity = np.isin(token_type_tok, (1, 2, 3, 5, 6))
        for t in np.flatnonzero(identity & ~atomize_tok):
            seq[t] = ENC.token_to_idx.get(res_names_tok[t], ENC.token_to_idx["<A>"])
        seq[atomize_tok] = ENC.token_to_idx["<A>"]
        host["condition_sequence_1_residue"] = seq[None]
        # condition_index absent -> all indexed -> no in-context conditioning
        host.setdefault("is_in_context_conditioning", np.zeros((1, n_tokens), dtype=bool))
    if "supports_local_latent" not in host:
        # supported residues & ~atomized & ~in-context (AddSupportsLocalLatentAnnotation)
        from rfproteina.utils.constants import LOCAL_LATENT_RESIDUES
        is_stub_tok = np.array([bool(atoms["is_stub"][token_starts[t]])
                                for t in range(n_tokens)])
        supported = np.array([(res_names[token_starts[t]] in LOCAL_LATENT_RESIDUES) | is_stub_tok[t]
                              for t in range(n_tokens)])
        supported &= ~atomize_tok  # atomized tokens never carry local latents
        in_context = np.asarray(host["is_in_context_conditioning"])[0]
        host["supports_local_latent"] = (supported & ~in_context)[None]

    # ---- CCD-derived atom metadata → atom37 (element/charge/ground_truth_element)
    coord_free_stub = np.array([
        _is_sentinel(i) and res_names[i] == MASKED_RES and atoms["atom_name"][i] == UNK_ATOM
        for i in range(n)])
    has_slot = atom37_slot >= 0
    # atomized atoms sit in atom37 slot 1 (one atom per token), outside the
    # vocabulary layouts
    slot_of_atom = np.where(atomize_atom, 1, atom37_slot)
    has_slot = has_slot | atomize_atom
    atom37_mask = np.zeros((n_tokens, 37), dtype=bool)
    element37 = np.full((n_tokens, 37), -1, np.int32)
    gt_element37 = np.zeros((n_tokens, 37), np.int32)
    charge37 = np.zeros((n_tokens, 37), np.int32)
    incident = np.zeros((n_tokens, 37, 10), np.int32)
    atom_element_id = {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "SE": 34, "X": -1}

    def _atomic_number(element_sym: str) -> int:
        # full element → Z via gemmi (the parse's atomic_number annotation covers
        # every element, e.g. Mg=12; the small hardcoded table above is only a
        # fast path for the common protein elements)
        sym = element_sym.upper()
        if sym in atom_element_id:
            return atom_element_id[sym]
        import gemmi
        z = int(gemmi.Element(sym).atomic_number)
        atom_element_id[sym] = z
        return z
    # element is provided ONLY for sequence-conditioned tokens (pipeline: element_mask
    # = seq_cond_mask); generated tokens pass -1
    seq_tok = np.asarray(host["condition_sequence_1_residue"])[0]
    elem_token = seq_tok != 0  # 0 = <M> (unconditioned)
    for i in range(n):
        if not has_slot[i]:
            continue
        t, s = int(token_of_atom[i]), int(slot_of_atom[i])
        atom37_mask[t, s] = True
        if elem_token[t]:
            element37[t, s] = _atomic_number(elements[i])
        gt_element37[t, s] = _atomic_number(elements[i]) if elements[i] != ELEMENT_X else 0
        if cond_atom[i]:
            cond_coords[t, s] = condition_coordinate[i]
            coord_groups[t, s] = int(cond_groups_atom[i])
        entry = ccd_entries.get(res_names[i])
        if entry is not None:
            k = entry.atom_index(atoms["atom_name"][i])
            if k is not None:
                charge37[t, s] = int(entry.charges[k])
        # formal charges are authored per-atom in the CIF (pdbx_formal_charge)
        if formal_charge[i] not in (".", "?", ""):
            charge37[t, s] = int(float(formal_charge[i]))
    coord_cond_mask = (np.logical_or.reduceat(cond_atom, token_starts)
                       if n_tokens > 0 else np.zeros(0, dtype=bool))
    host["atom37_mask"] = atom37_mask[None]
    host["element"] = element37[None]
    host["ground_truth_element"] = gt_element37[None]
    host["charge"] = charge37[None]
    host["condition_coordinate_1_atom"] = cond_coords[None]  # re-assign: filled above
    host["condition_CoordinateGroup_1_atom"] = coord_groups[None]
    host["coord_cond_mask"] = coord_cond_mask[None]
    # each bond endpoint is counted independently (ConvertToAtom37Format): an
    # endpoint without an atom37 slot (e.g. MASKED stubs) is skipped, its partner
    # still counts
    for endpoint in (0, 1):
        atoms_e = bonds[:, endpoint]
        bt_e = bonds[:, 2]
        valid = has_slot[atoms_e]
        ai_v = atoms_e[valid]
        np.add.at(incident, (token_of_atom[ai_v], slot_of_atom[ai_v], bt_e[valid]), 1)
    host["incident_bond_type_counts"] = incident[None]

    # coordinates: real atoms in slots; stubs zero (masked rows carry NaN coords → 0)
    coords37 = np.zeros((n_tokens, 37, 3), np.float32)
    valid_coords = has_slot & np.isfinite(atoms["coords"]).all(axis=-1)
    coords37[token_of_atom[valid_coords], slot_of_atom[valid_coords]] = atoms["coords"][valid_coords]
    host["coords"] = coords37[None]  # Å; assembler converts to nm

    # ---- token bond features (bonds.py semantics)
    tok1 = token_of_atom[bonds[:, 0]]
    tok2 = token_of_atom[bonds[:, 1]]
    inter = tok1 != tok2
    e1, e2, et, ear = tok1[inter], tok2[inter], bonds[:, 2][inter], bonds[:, 3][inter]
    lo, hi = np.minimum(e1, e2), np.maximum(e1, e2)

    order_lookup = np.zeros(10, np.int16)
    order_lookup[BT_SINGLE], order_lookup[BT_DOUBLE] = 1, 2
    order_lookup[BT_AROMATIC_SINGLE], order_lookup[BT_AROMATIC_DOUBLE] = 1, 2
    order_lookup[7], order_lookup[BT_AROMATIC] = 3, 1

    # canonical ordering BEFORE suppression (bonds.py semantics): token types from
    # the vocabulary groups (computed above); poly = ~atomize; NCAA remaps to
    # PROTEIN for the canonical test
    poly = ~atomize_tok
    poly_pair = np.outer(poly, poly)
    canon_type = np.where(token_type_tok == 6, 0, token_type_tok)  # NCAA → PROTEIN
    canonical = np.isin(canon_type, (0, 1, 2))
    same_type = canon_type[:, None] == canon_type[None, :]
    bond_pair = np.zeros((n_tokens, n_tokens), dtype=bool)
    bond_pair[lo, hi] = True
    bond_pair[hi, lo] = True
    same_grp_pair = idx_group[:, None] == idx_group[None, :]
    eligible = poly_pair & np.outer(canonical, canonical) & same_type & same_grp_pair & bond_pair
    # bond direction (bonds.py _upstream_direction): leaving atom of one residue
    # into the entering atom of the next (per-CCD polymerization atoms; protein
    # C→N, DNA/RNA O3'→P); no chemistry match (stub stand-ins) → chain-order
    # tie-break (lower token upstream)
    from atomworks.io.utils.ccd import get_polymerization_atoms as _poly_atoms
    _link_cache: dict[tuple[str, int], tuple[str, str]] = {}

    def _link_names(t: int) -> tuple[str, str]:
        key = (res_names[token_starts[t]].strip(), int(canon_type[t]))
        if key not in _link_cache:
            leaving, entering = _poly_atoms(key[0])
            if leaving is None and entering is None:
                unknown = {0: "UNK", 1: "N", 2: "DN"}.get(key[1])
                if unknown is not None:
                    leaving, entering = _poly_atoms(unknown)
            _link_cache[key] = (leaving or "", entering or "")
        return _link_cache[key]

    def _upstream(t1: int, a1n: str, t2: int, a2n: str) -> tuple[int, int] | None:
        l1, e1 = _link_names(t1)
        l2, e2 = _link_names(t2)
        if a1n == l1 and a2n == e2:
            return t1, t2
        if a2n == l2 and a1n == e1:
            return t2, t1
        return None  # no chemistry: chain-order tie-break

    # upstream = token contributing C (leaving); stub links: lower token upstream
    canon = np.zeros((n_tokens, n_tokens), np.int8)
    link_pairs: set[tuple[int, int]] = set()
    for a1, a2, bt, _ar in bonds:
        t1, t2 = token_of_atom[a1], token_of_atom[a2]
        if t1 == t2:
            continue
        a1n, a2n = atoms["atom_name"][a1].strip(), atoms["atom_name"][a2].strip()
        directed = _upstream(t1, a1n, t2, a2n)
        if directed is None:
            up, dn = (t1, t2) if t1 < t2 else (t2, t1)
        else:
            up, dn = directed
        if poly[t1] and poly[t2]:
            # backbone-linkage poly-poly bonds are suppressed from the bond
            # features (carried by path length + ordering)
            link_pairs.add((min(up, dn), max(up, dn)))
            if eligible[min(t1, t2), max(t1, t2)]:
                canon[dn, up] = 1
                canon[up, dn] = -1
    host["canonical_polymer_ordering"] = canon[None]
    ig_atoms = np.where(np.asarray(authored_idx_group)[bonds[:, 0]] >= 0,
                        np.asarray(authored_idx_group)[bonds[:, 0]],
                        np.array([chain_rank[chains[i]] for i in bonds[:, 0]]))
    ig_atoms2 = np.where(np.asarray(authored_idx_group)[bonds[:, 1]] >= 0,
                         np.asarray(authored_idx_group)[bonds[:, 1]],
                         np.array([chain_rank[chains[i]] for i in bonds[:, 1]]))
    keep_edge = [(a, b, t, ar) for a, b, t, ar, g1, g2 in
                 zip(lo, hi, et, ear, ig_atoms, ig_atoms2)
                 if (a, b) not in link_pairs and g1 == g2]
    for a, b, t, ar in keep_edge:
        host["bond_mask"][0, a, b] = host["bond_mask"][0, b, a] = True
        host["bond_order"][0, a, b] = host["bond_order"][0, b, a] = int(order_lookup[t])
        if ar:
            host["is_aromatic"][0, a, b] = host["is_aromatic"][0, b, a] = True

    # BFS path lengths over the full inter-token graph (cutoff None)
    if n_tokens > 0 and len(lo):
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        rows_g = np.concatenate([lo, hi])
        cols_g = np.concatenate([hi, lo])
        graph = csr_matrix(
            (np.ones(len(rows_g), dtype=np.int8), (rows_g, cols_g)),
            shape=(n_tokens, n_tokens),
        )
        sp = shortest_path(graph, directed=False, unweighted=True)
        path = np.where(np.isfinite(sp), sp, 0.0).astype(np.int32)
    else:
        path = np.zeros((n_tokens, n_tokens), np.int32)
    # paths across IndexGroups are unknown relative positions -> 0 (bonds.py)
    idx_group = np.asarray(host["condition_index-group_1_token"])[0]
    if np.unique(idx_group).size > 1:
        same_group = idx_group[:, None] == idx_group[None, :]
        path[~same_group] = 0
    host["token_bond_path_length"] = path[None]

    # sequence-probability channels: unspecified unless authored
    probs = np.full((1, n_tokens, 28), np.nan, np.float32)
    for ci, key in enumerate(SEQ_PROB_ORDER):
        if key in annotations:
            probs[0, :, ci] = [float(annotations[key][token_starts[t]]) for t in range(n_tokens)]
    # C_SEQ is authoritative: sequence-conditioned tokens get their one-hot distribution
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as _ENC
    seq_tokens = np.asarray(host["condition_sequence_1_residue"])[0]
    idx_to_tok = _ENC.idx_to_token if isinstance(_ENC.idx_to_token, dict) else dict(enumerate(_ENC.idx_to_token))
    aa_names = [idx_to_tok.get(int(t), "<M>") for t in seq_tokens]
    vocab_order = list(SEQ_PROB_ORDER)  # 20 aa + A/C/G/U + DA/DC/DG/DT, canonical order
    for t, name in enumerate(aa_names):
        if name == "<M>":
            continue
        if atomize_tok[t]:
            # atomized tokens carry C_SEQ=<A> and an all-zero (finite) distribution
            probs[0, t, :] = 0.0
            continue
        probs[0, t, :] = 0.0
        if f"condition_sequence-probability-{name}_1_residue" in vocab_order:
            probs[0, t, vocab_order.index(f"condition_sequence-probability-{name}_1_residue")] = 1.0
    # the assembler stacks from the individual channel keys, not the stacked tensor
    for ci, key in enumerate(SEQ_PROB_ORDER):
        host[key] = probs[0, :, ci][None]

    # ---- chirality (AddChiralityFeature): idealized stereo frames for atomized
    # residues; non-atomized residues are skipped by the reference (zeros)
    chir_tok, frames_tok, owners_tok = _build_chirality_features(
        atoms, bonds, token_of_atom, atomize_atom, condition_coordinate,
        cond_groups_atom, atoms["coords"], n_tokens, token_starts, ccd_entries)
    host["chirality"] = chir_tok[None]
    host["chiral_frame_atoms"] = frames_tok[None]
    host["chiral_slot_owner"] = owners_tok[None]

    batch, coord_cond = assemble_flow_batch(host)
    return batch, coord_cond
