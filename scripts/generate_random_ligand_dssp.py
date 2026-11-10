"""Random protein-ligand co-design: salad-style random DSSP strings + ChEMBL drug-like ligands.

For each of `--num-pairs` trials:
  1. sample a random secondary-structure string of `--num-residues` (default 72)
     with the `random_dssp` sampler ported from mjendrusch/salad
     (salad/training/eval_confchange.py: random_dssp_mean / random_dssp —
     loop fraction <= 0.5, random H/E split, >= 6/4-residue elements, `_`
     unspecified flanks with probability p=0.1 per element),
  2. pick a random drug-like ligand from ChEMBL (MW 250-500, heavy atoms
     15-48, small molecule; largest fragment if the SMILES is a salt; ALL
     hydrogens dropped via RDKit RemoveHs) and resample until the TOTAL token
     count stays <= `--max-tokens` (default 120),
  3. compile `RFD4Input.from_chains(72, {"smiles": ...}, pad_to=...)`, set every
     ligand atom to buried via `add_rasa(1, ...)`, then `pad_to_nearest(120)` —
     every trial is (1, 120) so the jitted model compiles exactly once,
  4. sample with the JAX flow model + AE decode and write the PDB (protein
     chain A + HETATM ligand chain B) with a manifest entry.

Example:
    python scripts/generate_random_ligand_dssp.py --num-pairs 100 \
        --out-dir runs/rfd4-test-ligands/random
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import time
import urllib.parse
import urllib.request

import numpy as np

# Release weights, fetched from the Hugging Face hub on first use.
SNAPSHOT = "nvidia/RFD4-Proteina"
CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"


# ------------------------------------------------------------------ salad port
def random_dssp_mean(rng: np.random.Generator, max_loop: float = 0.5):
    """salad random_dssp_mean: loop fraction <= max_loop; the rest is a random
    helix/sheet split (uniform over the fraction of structured content)."""
    helix = rng.random()
    sheet = 1 - helix
    loop = rng.random() * max_loop
    return loop, (1 - loop) * helix, (1 - loop) * sheet


def random_dssp(count: int, rng: np.random.Generator, pyrng: random.Random,
                p: float = 0.1) -> str:
    """Port of salad/training/eval_confchange.py `random_dssp` (seeded).

    Returns a length-`count` string over {H, E, L, _} (RFD4Spec.add_dssp's
    strict 3-state alphabet; `_` = unspecified)."""
    loop, helix_f, sheet_f = random_dssp_mean(rng)
    helix = int(helix_f * count)
    if helix < 6:
        helix = 0
    min_helix_count = 1
    max_helix_count = helix // 6
    min_helix_count = min(min_helix_count, max_helix_count)
    if helix == 0:
        num_helices = 0
    elif min_helix_count == max_helix_count:
        num_helices = min_helix_count
    else:
        num_helices = int(rng.integers(min_helix_count, max_helix_count))
    sheet = int(sheet_f * count)
    if sheet < 8:
        sheet = 0
    loop = count - (helix + sheet)
    min_sheet_count = 2
    max_sheet_count = sheet // 4
    if sheet == 0:
        num_sheets = 0
    elif min_sheet_count == max_sheet_count:
        num_sheets = min_sheet_count
    else:
        num_sheets = int(rng.integers(min_sheet_count, max_sheet_count))
    helices = [6 for _ in range(num_helices)]
    sheets = [4 for _ in range(num_sheets)]
    loops = [0 for _ in range(num_helices + num_sheets + 1)]
    while sum(sheets) < sheet:
        index = int(rng.integers(num_sheets))
        sheets[index] += 1
    while sum(helices) < helix:
        index = int(rng.integers(num_helices))
        helices[index] += 1
    while sum(loops) < loop:
        index = int(rng.integers(0, len(loops)))
        loops[index] += 1
    helices = ["_" + "H" * (num - 2) + "_" if pyrng.random() > p else "_" * num
               for num in helices]
    sheets = ["_" + "E" * (num - 2) + "_" if pyrng.random() > p else "_" * num
              for num in sheets]
    loops = ["L" * num if pyrng.random() > p else "_" * num for num in loops]
    structured = helices + sheets
    pyrng.shuffle(structured)
    dssp = loops[0] + "".join(s + l for s, l in zip(structured, loops[1:]))
    assert len(dssp) == count, f"random_dssp length {len(dssp)} != {count}"
    return dssp


# ------------------------------------------------------------------- ChEMBL
def fetch_chembl_pool(size: int, seed: int, min_heavy: int, max_heavy: int,
                      cache: pathlib.Path) -> list[dict]:
    """Random drug-like ChEMBL molecules (MW 250-500, small molecule) via the
    public API at random offsets; cached to `cache` as JSON."""
    if cache.exists():
        return json.loads(cache.read_text())
    query = {"max_mw": 500, "min_mw": 250, "min_heavy_atoms": min_heavy,
             "max_heavy_atoms": max_heavy, "molecule_type": "Small molecule",
             "only": "molecule_chembl_id,molecule_structures,molecule_properties",
             "limit": 1}
    url = CHEMBL_BASE + "?" + urllib.parse.urlencode(query)
    with urllib.request.urlopen(url, timeout=60) as r:
        total = json.loads(r.read())["page_meta"]["total_count"]
    print(f"ChEMBL drug-like pool: {total} candidates", flush=True)
    rng = np.random.default_rng(seed)
    seen, pool = set(), []
    attempts = 0
    while len(pool) < size and attempts < 4 * size:
        offset = int(rng.integers(0, total - 1))
        page = dict(query, limit=min(200, size - len(pool)), offset=offset)
        url = CHEMBL_BASE + "?" + urllib.parse.urlencode(page)
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = json.loads(r.read())
        except Exception as e:  # transient API errors: keep going
            print(f"  chembl fetch failed at offset {offset}: {e}", flush=True)
            attempts += 1
            time.sleep(1.0)
            continue
        for mol in data.get("molecules", []):
            smiles = (mol.get("molecule_structures") or {}).get("canonical_smiles")
            if not smiles or smiles in seen:
                continue
            seen.add(smiles)
            pool.append({"chembl_id": mol.get("molecule_chembl_id"),
                         "smiles": smiles,
                         "heavy_atoms": int((mol.get("molecule_properties")
                                             or {}).get("heavy_atoms", -1))})
        attempts += 1
        if len(pool) % 200 < 200 and len(data.get("molecules", [])) == 0:
            attempts += 5  # empty page: offset beyond the filtered tail
    cache.write_text(json.dumps(pool, indent=0))
    print(f"  fetched {len(pool)} unique candidates", flush=True)
    return pool


def sample_ligand_smiles(pool: list[dict], rng: random.Random):
    """Pick a random candidate, decode with RDKit, drop ALL hydrogens; return
    (smiles_after_Hs, heavy_count, chembl_id, was_fragment_selected)."""
    from rdkit import Chem

    for _ in range(50):
        mol_rec = pool[rng.randrange(len(pool))]
        mol = Chem.MolFromSmiles(mol_rec["smiles"])
        if mol is None:
            continue
        frag = False
        if "." in mol_rec["smiles"]:  # salt/mixture: keep the largest fragment
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            mol = max(frags, key=lambda m: m.GetNumAtoms())
            frag = True
        mol = Chem.RemoveHs(mol)  # drop all hydrogens
        n = mol.GetNumAtoms()
        if n == 0:
            continue
        return Chem.MolToSmiles(mol), n, mol_rec["chembl_id"], frag
    raise RuntimeError("could not decode any ChEMBL candidate with RDKit")


# ----------------------------------------------------------------- sampling
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-pairs", type=int, default=100)
    parser.add_argument("--num-residues", type=int, default=72)
    parser.add_argument("--max-tokens", type=int, default=120)
    parser.add_argument("--pool-size", type=int, default=400)
    parser.add_argument("--p-unspecified", type=float, default=0.1,
                        help="salad random_dssp's per-element '_' probability")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default=SNAPSHOT)
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="HF cache dir for the release checkpoint "
                             "(default: the huggingface default cache)")
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/rfd4-test-ligands/random"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.model import RFD4
    from rfd4fx.rfd4_input import RFD4Input
    from rfd4fx.spec import RFD4Spec, _atomic_number_from_symbol

    pool = fetch_chembl_pool(args.pool_size, args.seed, min_heavy=15,
                             max_heavy=args.max_tokens - args.num_residues,
                             cache=args.out_dir / "chembl_pool.json")

    torch.manual_seed(args.seed)
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint, cache_dir=args.cache)
    model.eval()
    jax_model = RFD4.from_torch(model)
    enc = UNIFIED_ATOM37_ENCODING
    Z_TO_SYM = {6: "C", 7: "N", 8: "O", 9: "F", 15: "P", 16: "S", 17: "Cl",
                35: "Br", 53: "I", 5: "B"}

    manifest = {
        "checkpoint": str(args.checkpoint),
        "featurization": "RFD4Input.from_chains (random dssp + ChEMBL ligand) "
                         "-> add_rasa(buried) -> pad_to_nearest(max_tokens)",
        "seed": args.seed, "nsteps": args.nsteps,
        "num_residues": args.num_residues, "max_tokens": args.max_tokens,
        "backend": "jax (rfd4fx)",
        "dssp_sampler": "salad random_dssp (mjendrusch/salad "
                        "eval_confchange.py), p_unspecified="
                        f"{args.p_unspecified}",
        "designs": [],
    }
    out_manifest = args.out_dir / "manifest_random.json"
    if out_manifest.exists():
        try:
            prev = json.loads(out_manifest.read_text())
            manifest["designs"].extend(prev.get("designs", []))
        except Exception:
            pass

    # one-time jitted model: every trial is (1, pad_to_nearest -> max_tokens)
    compiled = False

    for trial in range(args.num_pairs):
        t0 = time.time()
        seed = args.seed * 1000 + trial
        rng = np.random.default_rng(seed)
        pyrng = random.Random(seed)
        dssp = random_dssp(args.num_residues, rng, pyrng,
                           p=args.p_unspecified)
        # resample the ligand until the TOTAL token count fits the budget
        while True:
            smiles, n_lig, chembl_id, frag = sample_ligand_smiles(pool, pyrng)
            if args.num_residues + n_lig <= args.max_tokens:
                break

        inp = RFD4Input.from_chains(
            {"num_residues": args.num_residues, "dssp": dssp},
            {"smiles": smiles}, np_seed=seed)
        n_real = int(np.asarray(inp.padding_mask).sum())
        assert n_real == args.num_residues + n_lig
        # buried: every ligand atom slot gets C_RAS = 1
        where = np.zeros((n_lig, 37), dtype=bool)
        where[:, 1] = True
        inp = inp.add_rasa(1, aa=None, where=where, start=args.num_residues)
        rasa = np.asarray(inp.rasa)[0]
        assert (rasa[args.num_residues:, 1] == 1).all()
        assert (rasa[:args.num_residues] == 0).all()
        inp = inp.pad_to_nearest(args.max_tokens)
        n_width = int(np.asarray(inp.padding_mask).shape[1])
        batch1 = inp.to_batch()
        if not compiled:
            print(f"compiling the jitted model once at (1, {n_width}) ...",
                  flush=True)

        batch = batch1  # batch size 1 per random pair
        key = jax.random.fold_in(jax.random.key(args.seed), 777 + trial)
        x, x_1_pred, dec = jax_model.generate(batch, key=key, nsteps=args.nsteps)
        compiled = True

        # ---- decode + write the PDB (padding-aware token selection)
        coors_a = np.asarray(dec.coors_nm) * 10.0
        x_bb = np.asarray(x["bb_ca"]) * 10.0
        aatype = np.asarray(dec.aatype_max)
        atom_mask = np.asarray(dec.atom_mask)
        emit = np.asarray(dec.emit_mask)
        real_tok = np.asarray(batch1.padding_mask)[0]
        tok_is_ligand = np.zeros(n_width, dtype=bool)
        tok_is_ligand[args.num_residues:] = True
        tok_is_ligand &= real_tok
        tok_is_protein = real_tok & ~tok_is_ligand

        # ligand atom identities from a fresh spec parse (atom order = RDKit's)
        spec_probe = RFD4Spec()
        spec_probe.add_ligand(smiles=smiles)
        lig = spec_probe.ligands[0]
        lig_syms = [Z_TO_SYM.get(int(z), str(z)) for z in lig.elements]
        counts: dict[str, int] = {}
        lig_names = []
        for sym in lig_syms:
            counts[sym] = counts.get(sym, 0) + 1
            lig_names.append(f"{sym}{counts[sym]}")

        j = 0  # batch size 1
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
        lig_coords = x_bb[j][tok_is_ligand]
        assert np.isfinite(lig_coords).all(), f"trial {trial}: non-finite ligand"
        n_lig_written = len(lig_coords)
        lig_arr = bs.AtomArray(n_lig_written)
        lig_arr.add_annotation("hetero", dtype=bool)
        lig_arr.add_annotation("occupancy", dtype=float)
        lig_arr.add_annotation("b_factor", dtype=float)
        for k in range(n_lig_written):
            lig_arr.coord[k] = lig_coords[k]
            lig_arr.chain_id[k] = "B"
            lig_arr.res_id[k] = 1
            lig_arr.res_name[k] = "LIG"
            lig_arr.atom_name[k] = lig_names[k]
            lig_arr.element[k] = lig_syms[k]
            lig_arr.hetero[k] = True
            lig_arr.occupancy[k] = 1.0
            lig_arr.b_factor[k] = 0.0
        arr = arr_p + lig_arr

        out = args.out_dir / f"design_{trial:03d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))
        back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
        ca = back[back.atom_name == "CA"]
        lig_back = back[back.hetero]
        assert len(ca) == args.num_residues, f"trial {trial}: {len(ca)} CA atoms"
        assert len(lig_back) == n_lig, f"trial {trial}: {len(lig_back)} ligand atoms"
        assert bool(np.isfinite(back.coord).all()), f"trial {trial}: non-finite"
        d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=-1)
        d_pl = np.linalg.norm(ca.coord[:, None] - lig_back.coord[None], axis=-1)
        bl = np.linalg.norm(lig_back.coord[lig.bonds[:, 0]]
                            - lig_back.coord[lig.bonds[:, 1]], axis=-1)
        manifest["designs"].append({
            "id": f"design_{trial:03d}",
            "file": str(out),
            "dssp": dssp,
            "dssp_counts": {"H": dssp.count("H"), "E": dssp.count("E"),
                            "L": dssp.count("L"), "_": dssp.count("_")},
            "smiles": smiles,
            "chembl_id": chembl_id,
            "fragment_selected": frag,
            "n_ligand_atoms": n_lig,
            "padded_tokens": n_width,
            "ca_ca_mean": float(d.mean()), "ca_ca_max": float(d.max()),
            "ligand_bond_length_mean": float(bl.mean()),
            "ligand_bond_length_min": float(bl.min()),
            "ligand_bond_length_max": float(bl.max()),
            "min_protein_ligand_ca_distance": float(d_pl.min()),
            "protein_ligand_ca_pairs_within_10A": int((d_pl < 10).sum()),
        })
        out_manifest.write_text(json.dumps(manifest, indent=1))
        print(
            f"  design_{trial:03d}: {chembl_id} ({n_lig} atoms) | "
            f"dssp H/E/L/_ = {dssp.count('H')}/{dssp.count('E')}/"
            f"{dssp.count('L')}/{dssp.count('_')} | "
            f"CA-CA {d.mean():.2f} (max {d.max():.2f}), bonds {bl.mean():.2f} "
            f"[{bl.min():.2f}-{bl.max():.2f}], min prot-lig {d_pl.min():.1f} A, "
            f"{int((d_pl < 10).sum())} pairs < 10 A  [{time.time() - t0:.0f}s]",
            flush=True,
        )
    print(f"done: {len(manifest['designs'])} designs -> {args.out_dir}")


if __name__ == "__main__":
    main()
