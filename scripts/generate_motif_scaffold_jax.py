"""Unindexed motif scaffolding: generate `--num_aa`-residue proteins around
extracted ligand + contact-residue motifs.

For each motif PDB in `--motif_path` (chain A = contact residues, chain B =
ligand HETATM, as written by extract_motifs.py):
  - the ligand's heavy atoms become atomized motif tokens (aa = <A>, slot-1
    mask, coordinates from the file),
  - each contact residue becomes a motif residue token (atom name -> atom37
    slot via UNIFIED_ATOM37_ENCODING, aa = AE vocabulary index, author
    residue IDs as residue_index),
  - the motif mask keeps ONLY atoms within `--cutoff` (default 5 A) of any
    ligand atom — the binding-site contact atoms; everything farther away is
    left to the designer,
  - ligand chemistry (element / formal charge / bond graph) is decoded from
    the motif's SMILES (motifs.json) via RFD4Spec.add_ligand (RDKit) and passed
    explicitly to add_unindexed_motif — the campaign wrote ligand atoms in
    RDKit order, so the decoded atom order matches the PDB HETATM rows
    (asserted on the elements),
  - `RFD4Input.from_chains(--num_aa).add_unindexed_motif(..., fixed=--fixed)`
    with no other conditioning, then `pad_to_nearest(200)` — every trial is
    (1, 200), so the jitted model compiles exactly once. `fixed=true` ->
    absolute coordinate conditioning (C_CRD); `fixed=false` -> internal
    coordinate conditioning (C_INTCRD/C_CRDGRP, the trained
    TipAtomPartialLigandTask shape).

Designs are written as PDBs (designed protein chain A + the motif residues and
ligand as HETATM reference chains) with a resumable manifest. Because the
trained conditioning (and the native motif-scaffolding benchmark) evaluates
motif recreation by ALIGNED internal geometry rather than frame-level
placement, each design PDB is written twice:
  `<id>.pdb`           the flow-frame assembly (motif reference at input coords)
  `<id>_aligned.pdb`   motif + ligand reference placed at the best-matching
                       design segment (non-consecutive Kabsch search,
                       scripts/motif_match.py) with per-design match metrics
                       in the manifest.

Example:
    python scripts/generate_motif_scaffold_jax.py \
        --motif_path runs/rfd4-test-ligands/random/motifs \
        --num_aa 72 --num-designs 5
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

# Release weights, fetched from the Hugging Face hub on first use.
SNAPSHOT = "nvidia/RFD4-Proteina"


def parse_motif_pdb(path: pathlib.Path, cutoff: float,
                    include_ligand: bool = True,
                    residue_atom_mode: str = "contact"):
    """Parse a motif PDB -> (residue tokens, ligand tokens) arrays for
    add_unindexed_motif. Returns (pos, aa, mask, residue_index, ligand_meta,
    res_meta) with the residue tokens first, then the ligand atoms.

    With include_ligand=False the ligand still defines the 5 A site cutoff but
    contributes NO motif tokens (residues-only motif).

    residue_atom_mode: "contact" masks only the atoms within `cutoff` of the
    ligand; "all" conditions ALL mapped atoms of every residue that has at
    least one atom within the cutoff."""
    import gemmi
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    structure = gemmi.read_structure(str(path))
    model = structure[0]  # first model
    enc = ENC

    # ligand: HETATM chain(s), hydrogens dropped
    lig_atoms = []
    for chain in model:
        for res in chain:
            if res.het_flag != "H":
                continue
            for atom in res:
                if atom.element.name in ("H", "D"):
                    continue
                lig_atoms.append((atom.name, atom.element.name,
                                  [atom.pos.x, atom.pos.y, atom.pos.z]))
    if not lig_atoms:
        raise ValueError(f"{path}: no ligand (HETATM) atoms")
    lig_xyz = np.array([a[2] for a in lig_atoms], np.float32)

    # protein residues: chain A (non-HETATM)
    res_pos, res_mask, res_aa, res_ridx = [], [], [], []
    # residues with at least one atom within the cutoff (for residue_atom_mode="all")
    res_contact_residues = set()
    if residue_atom_mode == "all":
        for chain in model:
            for res in chain:
                if res.het_flag == "H":
                    continue
                for atom in res:
                    xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], np.float32)
                    if float(np.linalg.norm(lig_xyz - xyz, axis=-1).min()) <= cutoff:
                        res_contact_residues.add(id(res))
                        break

    for chain in model:
        for res in chain:
            if res.het_flag == "H":
                continue
            n37 = np.full((37, 3), np.nan, np.float32)
            mask37 = np.zeros(37, dtype=bool)
            seen = set()
            for atom in res:
                an = atom.name.strip()
                if an in seen:
                    continue
                seen.add(an)
                slot = None
                res_atoms = enc.token_atoms.get(res.name)
                if res_atoms is not None:
                    for s, slot_name in enumerate(res_atoms):
                        if str(slot_name).strip() == an:
                            slot = s
                            break
                if slot is None:
                    continue  # H / unmapped atoms
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], np.float32)
                # "contact": only atoms within the cutoff of ANY ligand atom;
                # "all": every mapped atom of a residue with >= 1 contact atom
                xyz = np.array([atom.pos.x, atom.pos.y, atom.pos.z], np.float32)
                # "contact": only atoms within the cutoff of ANY ligand atom;
                # "all": every mapped atom of a residue with >= 1 contact atom
                if (float(np.linalg.norm(lig_xyz - xyz, axis=-1).min()) <= cutoff
                        or (residue_atom_mode == "all" and id(res) in res_contact_residues)):
                    mask37[slot] = True
                    n37[slot] = xyz
            if not mask37.any():
                continue  # residue with no atom within the cutoff
            res_pos.append(n37)
            res_mask.append(mask37)
            res_aa.append(enc.token_to_idx.get(res.name, 0))
            res_ridx.append(res.seqid.num)
    if not res_pos:
        raise ValueError(f"{path}: no residue with an atom within {cutoff} A")

    # ligand -> atomized tokens (one per heavy atom, slot 1); with
    # include_ligand=False the ligand defines only the cutoff
    n_lig = len(lig_atoms) if include_ligand else 0
    if include_ligand:
        lig_pos = np.full((n_lig, 37, 3), np.nan, np.float32)
        lig_mask = np.zeros((n_lig, 37), dtype=bool)
        lig_pos[:, 1] = lig_xyz
        lig_mask[:, 1] = True
        lig_aa = np.full(n_lig, enc.token_to_idx["<A>"], np.int32)
        # ligand residue indices: fresh values far above the author IDs (no
        # accidental +-1 adjacency with the motif residues)
        max_ridx = max(res_ridx) if res_ridx else 0
        lig_ridx = max_ridx + 10 + np.arange(n_lig, dtype=np.int32)

    pos = np.concatenate([np.asarray(res_pos, np.float32)] +
                         ([lig_pos] if include_ligand else []), axis=0)
    mask = np.concatenate([np.asarray(res_mask, bool)] +
                          ([lig_mask] if include_ligand else []), axis=0)
    aa = np.asarray(res_aa + (lig_aa.tolist() if include_ligand else []),
                    np.int32)
    residue_index = np.asarray(
        res_ridx + (lig_ridx.tolist() if include_ligand else []), np.int32)
    meta = {
        "n_residue_tokens": len(res_pos),
        "n_ligand_atoms": n_lig,
        "ligand_atoms": [a[0] for a in lig_atoms],
        "ligand_elements": [a[1] for a in lig_atoms],
        "residues": res_meta_list(res_ridx, res_aa),
    }
    return pos, aa, mask, residue_index, meta


# biotite bond-type id -> raw bond order (for the add_unindexed_motif bonds
# kwarg, which takes [i, j, order, aromatic] with order in 1/2/3)
_BT_TO_ORDER = {1: 1, 2: 2, 3: 3, 5: 1, 6: 2, 7: 3, 9: 1}

# atomic number -> PDB element symbol (for the reference chain)
PERIODIC_TO_SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S",
                   17: "CL", 35: "BR", 53: "I"}


def ligand_chemistry_from_smiles(smiles: str, pdb_elements: list[str]):
    """Decode ligand chemistry from SMILES via RFD4Spec.add_ligand (RDKit) and
    return (element, charge, bonds) for add_unindexed_motif.

    The decoded atom order must match the motif PDB's HETATM row order (the
    campaign wrote ligand atoms in RDKit order) — asserted on the elements.
    """
    from rfd4fx.spec import RFD4Spec

    tmp = RFD4Spec()
    tmp.add_ligand(smiles=smiles)
    lig = tmp.ligands[0]
    decoded = [int(z) for z in lig.elements]
    from_pdb = [{"C": 6, "N": 7, "O": 8, "S": 16, "P": 15, "F": 9, "CL": 17,
                 "BR": 35, "I": 53}.get(e.upper(), -1) for e in pdb_elements]
    if decoded != from_pdb:
        raise ValueError(
            f"ligand atom order mismatch: SMILES decode {decoded} vs PDB rows "
            f"{from_pdb} (smiles={smiles!r})")
    bonds = np.stack([
        lig.bonds[:, 0], lig.bonds[:, 1],
        np.array([_BT_TO_ORDER[int(bt)] for bt in lig.bonds[:, 2]], np.int32),
        lig.bonds[:, 3].astype(np.int32)], axis=1).astype(np.int32)
    return lig.elements.astype(np.int32), lig.charges.astype(np.int32), bonds


def res_meta_list(ridx, aa):
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    idx_to_tok = (ENC.idx_to_token if isinstance(ENC.idx_to_token, dict)
                  else dict(enumerate(ENC.idx_to_token)))
    return [{"res_id": int(r), "res_name": str(idx_to_tok.get(int(a), "?"))}
            for r, a in zip(ridx, aa)]


def partial_ligand_crd(inp, n_design: int, n_res_tokens: int, n_lig_tokens: int,
                       hops: int, rng: np.random.Generator):
    """Replicate the trained TipAtomPartialLigandTask ligand masking
    (design_task.py): seed atom uniform over the ligand, grow along the bond
    graph by Geometric(p=1/(1+hops)) - 1 hops, C_CRD kept only on the grown
    region (the rest of the ligand stays indexed design targets with full
    chemistry). Returns a new RFD4Input."""
    import dataclasses

    import jax.numpy as jnp

    lo = n_design + n_res_tokens
    bm = np.asarray(inp.bond_mask)[0][lo:lo + n_lig_tokens, lo:lo + n_lig_tokens]
    adj = [set() for _ in range(n_lig_tokens)]
    for i, j in zip(*np.nonzero(bm)):
        if i < j:
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
    seed = int(rng.integers(n_lig_tokens))
    n_hops = int(rng.geometric(1 / (1 + hops))) - 1
    dist = {seed: 0}
    frontier = [seed]
    for _ in range(n_hops):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = 0
                    nxt.append(v)
        frontier = nxt
    keep = set(dist)

    out = {f: np.array(getattr(inp, f), copy=True) for f in
           inp.__dataclass_fields__ if getattr(inp, f) is not None}
    cc = out["cond_coords"]
    cg = out["coord_groups"]
    ccm = out["coord_cond"]
    for t in range(n_lig_tokens):
        if t in keep:
            continue
        cc[0, lo + t, 1] = np.nan
        cg[0, lo + t, 1] = -1
        # a token with no remaining conditioned atom is not coordinate-conditioned
        if not np.isfinite(cc[0, lo + t]).any():
            ccm[0, lo + t] = False
    fields = {f: jnp.asarray(v) for f, v in out.items()}
    print(f"    ligand C_CRD: {len(keep)}/{n_lig_tokens} atoms "
          f"(seed {seed}, hops {n_hops})", flush=True)
    return dataclasses.replace(inp, **fields)


def join_motif_into_design_chain(inp, n_design: int, n_res_tokens: int,
                                 n_lig_tokens: int):
    """Topology: re-author the motif RESIDUE tokens as members of the designed
    chain (the native unindexed-benchmark shape): shared chain_iid and molecule
    ids, while the tokens stay index-disconnected (per-token index groups, no
    bond paths, no polymer ordering — the native in-context semantics, which
    add_unindexed_motif now emits). The ligand tokens keep their own chain.
    """
    import dataclasses

    import jax.numpy as jnp
    import numpy as np

    n_design_end = n_design + n_res_tokens
    out = {f: np.array(getattr(inp, f), copy=True) for f in
           inp.__dataclass_fields__ if getattr(inp, f) is not None}
    out["chain_iid"][0, n_design:n_design_end] = out["chain_iid"][0, 0]
    out["molecule_iid"][0, n_design:n_design_end] = out["molecule_iid"][0, 0]
    out["molecule_entity_id"][0, n_design:n_design_end] = out["molecule_entity_id"][0, 0]
    fields = {f: jnp.asarray(v) for f, v in out.items()}
    return dataclasses.replace(inp, **fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motif_path", type=pathlib.Path, required=True,
                        help="directory of motif PDBs (from extract_motifs.py)")
    parser.add_argument("--num_aa", type=int, default=72,
                        help="number of designed amino-acid residues")
    parser.add_argument("--num-designs", type=int, default=1,
                        help="designs per motif PDB")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="motif mask cutoff (A) from the ligand")
    parser.add_argument("--fixed", choices=("true", "false"), default="true",
                        help="motif conditioning mode: true = absolute (C_CRD), "
                             "false = internal coordinates (C_INTCRD, the "
                             "trained TipAtomPartialLigandTask shape)")
    parser.add_argument("--num-motifs", type=int, default=None,
                        help="only the first N motif PDBs (sorted)")
    parser.add_argument("--motif-separate-chain", action="store_true",
                        help="give the motif residues their own fresh chain "
                             "(chain='new') instead of the default: joining "
                             "the designed chain (native unindexed shape)")
    parser.add_argument("--no-ligand", action="store_true",
                        help="motif = only the contact residues (atoms within "
                             "--cutoff of the ligand); the ligand itself is "
                             "NOT conditioned (it still defines the site)")
    parser.add_argument("--ligand-crd-hops", type=int, default=None,
                        help="theozyme/trained shape: condition only a subset "
                             "of the ligand atoms (seed + <=N bond-graph hops, "
                             "TipAtomPartialLigandTask masking); default = all "
                             "ligand atoms")
    parser.add_argument("--full-atom-residues", action="store_true",
                        help="condition ALL atoms of every contact residue "
                             "(default: only the atoms within --cutoff)")
    parser.add_argument("--pad-multiple", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default=SNAPSHOT)
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="HF cache dir for the release checkpoint "
                             "(default: the huggingface default cache)")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/rfd4-test-motifs"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
    from rfd4fx.model import RFD4
    from rfd4fx.rfd4_input import RFD4Input

    motif_files = sorted(args.motif_path.glob("*_motif.pdb"))
    if not motif_files:
        raise SystemExit(f"no motif PDBs in {args.motif_path}")
    if args.num_motifs is not None:
        motif_files = motif_files[:args.num_motifs]
    print(f"{len(motif_files)} motif PDBs in {args.motif_path}")

    # SMILES per motif (for ligand chemistry)
    smiles_path = args.motif_path / "motifs.json"
    smiles_by_id = {}
    if smiles_path.exists():
        for entry in json.loads(smiles_path.read_text()):
            mid = entry["motif_pdb"].rsplit("/", 1)[-1].replace("_motif.pdb", "")
            smiles_by_id[mid] = entry["smiles"]
    else:
        raise SystemExit(f"missing {smiles_path} (needed for ligand chemistry)")

    torch.manual_seed(args.seed)
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint, cache_dir=args.cache)
    model.eval()
    jax_model = RFD4.from_torch(model)
    enc = UNIFIED_ATOM37_ENCODING

    manifest = {
        "checkpoint": str(args.checkpoint),
        "featurization": "RFD4Input.from_chains + add_unindexed_motif -> "
                         "pad_to_nearest",
        "seed": args.seed, "nsteps": args.nsteps, "num_aa": args.num_aa,
        "cutoff": args.cutoff, "pad_multiple": args.pad_multiple,
        "fixed": args.fixed == "true",
        "recycle_schedule": "thinking_high",
        "motif_separate_chain": bool(args.motif_separate_chain),
        "backend": "jax (rfd4fx)",
        "designs": [],
    }
    out_manifest = args.out_dir / "manifest_scaffold.json"
    if out_manifest.exists():
        try:
            prev = json.loads(out_manifest.read_text())
            manifest["designs"].extend(prev.get("designs", []))
        except Exception:
            pass
    done_ids = {d["motif"] for d in manifest["designs"]}

    # RFD4 / RFD4Sampler API: the model provides the step callable; the
    # sampler filter_jits the step (and the decode) and integrates the flow.
    sampler = RFD4Sampler(jax_model.step(), jax_model)

    compiled = False
    t_all = time.time()
    for mi, motif_path in enumerate(motif_files):
        motif_id = motif_path.name.replace("_motif.pdb", "")
        for k in range(args.num_designs):
            design_id = f"{motif_id}_scaffold_{k:02d}"
            if design_id in done_ids:
                continue
            t0 = time.time()
            pos, aa, mask, residue_index, meta = parse_motif_pdb(
                motif_path, args.cutoff, include_ligand=not args.no_ligand,
                residue_atom_mode="all" if args.full_atom_residues else "contact")
            chem = {}
            if meta["n_ligand_atoms"]:
                element, charge, bonds = ligand_chemistry_from_smiles(
                    smiles_by_id[motif_id], meta["ligand_elements"])
                chem = {"element": element, "charge": charge, "bonds": bonds}
            inp = RFD4Input.from_chains(args.num_aa, np_seed=args.seed * 1000 + mi)
            # chain=None (default): the motif residues join the designed chain
            # (native unindexed shape); the ligand runs keep their own chains.
            # residue_index is a featurization no-op (in-context tokens carry
            # no index information) and is omitted.
            inp = inp.add_unindexed_motif(
                pos, aa, mask, fixed=args.fixed == "true",
                chain="new" if args.motif_separate_chain else None, **chem)
            if args.ligand_crd_hops is not None and meta["n_ligand_atoms"]:
                rng = np.random.default_rng(args.seed * 1000 + mi)
                inp = partial_ligand_crd(
                    inp, args.num_aa, meta["n_residue_tokens"],
                    meta["n_ligand_atoms"], args.ligand_crd_hops, rng)
            inp = inp.pad_to_nearest(args.pad_multiple)
            batch = inp.to_batch()
            n_width = int(np.asarray(batch.padding_mask).shape[1])
            n_real = int(np.asarray(batch.padding_mask).sum())
            if not compiled:
                print(f"compiling the jitted step once at (1, {n_width}) ...",
                      flush=True)
            key = jax.random.fold_in(jax.random.key(args.seed), 4242 + mi * 100 + k)
            # RFD4.step + RFD4Sampler: the sampler jits the step and the
            # decode, integrates the flow, and decodes.
            x, x_1, data, dec = sampler.sample(batch, key=key,
                                               nsteps=args.nsteps,
                                               recycle_schedule=THINKING_HIGH)
            compiled = True

            coors_a = np.asarray(dec.coors_nm) * 10.0
            aatype = np.asarray(dec.aatype_max)
            atom_mask = np.asarray(dec.atom_mask)
            emit = np.asarray(dec.emit_mask)
            real_tok = np.asarray(batch.padding_mask)[0]
            n_motif = meta["n_residue_tokens"] + meta["n_ligand_atoms"]
            tok_is_motif = np.zeros(n_width, dtype=bool)
            tok_is_motif[args.num_aa:args.num_aa + n_motif] = True
            tok_is_motif &= real_tok
            tok_is_protein = real_tok & ~tok_is_motif

            j = 0
            mask_j = atom_mask[j] & emit[j][:, None]
            coords37 = coors_a[j].copy()
            coords37[~mask_j] = np.nan
            res_type = np.where(emit[j], aatype[j], -1).astype(np.int64)
            arr_p = atom_array_from_encoding(
                encoded_coord=coords37[tok_is_protein],
                encoded_mask=mask_j[tok_is_protein],
                encoded_seq=res_type[tok_is_protein],
                encoding=enc, chain_id="A",
            )
            arr_p = arr_p[np.isfinite(arr_p.coord).all(axis=-1)]

            # motif reference chains: residues as HETATM chain M, ligand chain B
            motif_struct = bs.AtomArray(0)
            n_res_tok = meta["n_residue_tokens"]
            for t in range(n_res_tok):
                for s in np.flatnonzero(mask[t]):
                    if not np.isfinite(pos[t, s]).all():
                        continue
                    atom = bs.AtomArray(1)
                    atom.coord[0] = pos[t, s]
                    res_name = enc.idx_to_token.get(int(aa[t]), "UNK") \
                        if isinstance(enc.idx_to_token, dict) else enc.idx_to_token[int(aa[t])]
                    atom.chain_id[0] = "M"
                    atom.res_id[0] = int(residue_index[t])
                    atom.res_name[0] = str(res_name)
                    atom.atom_name[0] = str(enc.token_atoms[str(res_name)][s]).strip() or "X"
                    atom.element[0] = (atom.atom_name[0] or "X")[0]
                    atom.hetero[0] = True
                    motif_struct += atom
            for t in range(n_res_tok, n_motif):
                atom = bs.AtomArray(1)
                atom.coord[0] = pos[t, 1]
                atom.chain_id[0] = "B"
                atom.res_id[0] = 1
                atom.res_name[0] = "LIG"
                atom.atom_name[0] = f"A{t - n_res_tok + 1}"
                z = int(element[t - n_res_tok])
                atom.element[0] = PERIODIC_TO_SYM[z]
                atom.hetero[0] = True
                motif_struct += atom
            arr = arr_p + motif_struct

            out = args.out_dir / f"{design_id}.pdb"
            pdb = pdbio.PDBFile()
            pdb.set_structure(arr)
            pdb.write(str(out))
            back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
            ca = back[(back.atom_name == "CA") & (~back.hetero)]
            assert len(ca) == args.num_aa, f"{design_id}: {len(ca)} CA atoms"
            assert bool(np.isfinite(back.coord).all()), f"{design_id}: non-finite"
            d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=-1)

            # aligned motif matching + assembly (the native evaluation shape:
            # the motif must be recreated with correct internal geometry
            # somewhere in the design, found by alignment)
            from motif_match import aligned_assembly

            match = aligned_assembly(out, motif_path,
                                     args.out_dir / f"{design_id}_aligned.pdb")
            match = {k: v for k, v in match.items() if k not in ("R", "p_c", "q_c")}

            manifest["designs"].append({
                "id": design_id,
                "file": str(out),
                "aligned_file": str(args.out_dir / f"{design_id}_aligned.pdb"),
                "motif": motif_id,
                "motif_pdb": str(motif_path),
                "n_motif_tokens": n_motif,
                "n_residue_tokens": n_res_tok,
                "n_ligand_atoms": meta["n_ligand_atoms"],
                "padded_tokens": n_width,
                "ca_ca_mean": float(d.mean()), "ca_ca_max": float(d.max()),
                "motif_match": match,
            })
            out_manifest.write_text(json.dumps(manifest, indent=1))
            print(f"  {design_id}: {n_real} tokens ({args.num_aa} designed + "
                  f"{n_motif} motif) padded to {n_width}, CA-CA {d.mean():.2f} "
                  f"(max {d.max():.2f})  [{time.time() - t0:.0f}s]", flush=True)
    print(f"done: {len(manifest['designs'])} designs in {time.time() - t_all:.0f}s "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
