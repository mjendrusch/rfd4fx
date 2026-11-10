"""Side-by-side: native vs port on the SAME unindexed motif-scaffolding problem
(native benchmark 1QJG_AA_NATIVE: 113 designed stubs + 3 coordinate-conditioned
motif residues in one chain).

Port variants:
  A (separate chain): the current pipeline default — motif appended as its own
      fresh chain (no polymer linkage to the designed chain).
  B (joined): the motif residues re-authored INSIDE the designed chain
      (shared chain/molecule ids, continuous residue-index linkage — the
      native benchmark topology).

Outputs per variant: <tag>_design.pdb (chain A = designed stubs decoded
all-atom; chain M = motif residues at their input coordinates) + embedding
metrics + the aligned motif-match report, next to the native run's PDB.
"""

import json
import pathlib
import sys
from functools import partial

import numpy as np

REPO_ROOT = pathlib.Path("tests/fixtures/native/1qjg_run")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rfd4_upstream import ensure_benchmarks

OUT = REPO_ROOT / "side_by_side"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- native batch
from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.inference_utils.datasets import DesignProblemDataset
from rfproteina.datapipes.collate import collate_batch
from torch.utils.data import DataLoader

z = np.load(pathlib.Path("tests/fixtures/native/1qjg_run/raw_outputs.npz"))
WANT_N = z["padding_mask"].shape[0]
WANT_CCM = z["coord_cond_mask"]
p = (ensure_benchmarks()
     / "motif_scaffolding/motif_unindexed/1QJG_AA_NATIVE.cif.zst")

batch = None
for seed in range(500):
    pipe = build_design_validation_pipeline(
        metrics_tags=None, expandable_segment_rng=np.random.default_rng(seed))
    ds = DesignProblemDataset([str(p)], transform=pipe)
    b = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=partial(
        collate_batch, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
        fixed_dimensions=None))))
    if (b["padding_mask"].shape[1] == WANT_N
            and np.array_equal(b["coord_cond_mask"][0].numpy(), WANT_CCM)):
        batch = b
        print(f"native realization reproduced with seed {seed}: n={WANT_N}", flush=True)
        break
assert batch is not None

coords_in = batch["coords"][0].numpy()          # (116, 37, 3) A, centered
am = batch["atom37_mask"][0].numpy()
rt = batch["residue_type"][0].numpy()
ccm = batch["coord_cond_mask"][0].numpy()
motif_tok = np.flatnonzero(ccm)
n_stubs = int(np.flatnonzero(~ccm & batch["padding_mask"][0].numpy()).shape[0])
print(f"stubs {n_stubs} | motif tokens {motif_tok.tolist()} "
      f"residue types {rt[motif_tok].tolist()}", flush=True)

motif_pos = coords_in[motif_tok]                # (3, 37, 3)
motif_mask = am[motif_tok].astype(bool)
motif_aa = rt[motif_tok].astype(np.int32)
motif_ri = np.arange(1, len(motif_tok) + 1, dtype=np.int32)  # native seq 1..3

# ---------------------------------------------------------------- port runs
from rfd4fx.rfd4_input import RFD4Input
from generate_motif_scaffold_jax import join_motif_into_design_chain

import jax
import jax.numpy as jnp
import torch
from rfd4_proteina import Proteina
import rfd4fx.convert  # noqa: F401
from rfd4fx.model import RFD4
import biotite.structure as bs
import biotite.structure.io.pdb as pdbio

SNAP = "nvidia/RFD4-Proteina"
model = Proteina.from_pretrained(SNAP)
model.eval()
jax_model = RFD4.from_torch(model)


def write_pdb(coors_nm, aatype, real_mask, motif_reference, path):
    """chain A = designed residues (decoded sequence + proper atom names/
    elements, only the residue type's slots); chain M = motif residues at
    input coords (real atom names via the AE slot layouts)."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    enc = ENC
    idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
                  else dict(enumerate(enc.idx_to_token)))

    def element_of(atom_name):
        letters = "".join(c for c in atom_name if c.isalpha()).upper()
        if letters[:2] in ("CL", "BR", "MG", "ZN", "FE", "SE"):
            return letters[:2]
        return letters[:1] or "C"

    arr = bs.AtomArray(0)
    n_tok = coors_nm.shape[0]
    for t in range(n_stubs):  # designed stub tokens
        if not real_mask[t]:
            continue
        res_idx = int(aatype[t])
        res_name = str(idx_to_tok.get(res_idx, "UNK"))
        slot_names = enc.token_atoms.get(res_name)
        if slot_names is None:
            continue
        for s in range(37):
            xyz = coors_nm[t, s]
            an = str(slot_names[s]).strip()
            if not an or not np.isfinite(xyz).all() or np.abs(xyz).sum() == 0:
                continue
            atom = bs.AtomArray(1)
            atom.coord[0] = xyz
            atom.chain_id[0] = "A"
            atom.res_id[0] = t + 1
            atom.res_name[0] = res_name
            atom.atom_name[0] = an
            atom.element[0] = element_of(an)
            atom.hetero[0] = False
            arr += atom
    for j, tok in enumerate(motif_tok):  # motif reference at input coords
        res_name = str(idx_to_tok.get(int(cond_seq[tok]), "UNK"))
        slot_names = enc.token_atoms.get(res_name)
        if slot_names is None:
            slot_names = [f"A{s}" for s in range(37)]
        for s in np.flatnonzero(motif_mask[j] & np.isfinite(motif_reference[j]).all(-1)):
            an = str(slot_names[s]).strip() or f"A{s}"
            atom = bs.AtomArray(1)
            atom.coord[0] = motif_reference[j, s]
            atom.chain_id[0] = "M"
            atom.res_id[0] = j + 1
            atom.res_name[0] = res_name
            atom.atom_name[0] = an
            atom.element[0] = element_of(an)
            atom.hetero[0] = True
            arr += atom
    pdb = pdbio.PDBFile()
    pdb.set_structure(arr)
    pdb.write(str(path))


from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
idx_to_tok = (ENC.idx_to_token if isinstance(ENC.idx_to_token, dict)
              else dict(enumerate(ENC.idx_to_token)))
cond_seq = batch["condition_sequence_1_residue"][0].numpy()

reports = {}
for tag, chain in (("port_separate_chain", "new"), ("port_joined_chain", None)):
    t0 = __import__("time").time()
    inp = RFD4Input.from_chains(n_stubs, np_seed=0)
    # chain=None (default) joins the motif residues to the design chain;
    # NO residue_index for the motif residues (native: in-context tokens
    # carry no index information)
    inp = inp.add_unindexed_motif(motif_pos, motif_aa, motif_mask,
                                  fixed=True, chain=chain)
    inp = inp.pad_to_nearest(128)
    batch_j = inp.to_batch()
    key = jax.random.fold_in(jax.random.key(0), 4243)
    x, x_1_pred, dec = jax_model.generate(batch_j, key=key, nsteps=60)
    coors_nm = np.asarray(dec.coors_nm)[0] * 10.0  # A
    aatype = np.asarray(dec.aatype_max)[0]
    real_mask = np.asarray(batch_j.padding_mask)[0]
    out_pdb = OUT / f"{tag}_design.pdb"
    write_pdb(coors_nm, aatype, real_mask, coords_in[motif_tok], out_pdb)

    # embedding metrics in the flow frame (motif reference = input coords)
    stub_ca = np.array([coors_nm[t, 1] for t in range(n_stubs)])
    motif_cond = np.array([coords_in[t, 1] for t in motif_tok])
    d = np.linalg.norm(stub_ca[:, None] - motif_cond[None], axis=-1)
    # aligned motif match
    from motif_match import find_motif_match, parse_design_chain, parse_motif_reference
    design, _ = parse_design_chain(out_pdb)
    motif_ref, _ = parse_motif_reference(out_pdb)
    mm = find_motif_match(design, motif_ref)
    reports[tag] = {
        "min_stubCA_to_motifCA_A": float(d.min()),
        "stubCAs_within_10A": int((d.min(-1) < 10).sum()),
        "centroid_distance_A": float(np.linalg.norm(stub_ca.mean(0) - motif_cond.mean(0))),
        "motif_match_rmsd_A": mm["rmsd"], "motif_match_atoms": mm["n_atoms"],
        "motif_residues_placed": len(mm["mapping"]), "motif_matched": mm["matched"],
        "seconds": round(__import__("time").time() - t0),
    }
    print(tag, json.dumps(reports[tag], indent=1), flush=True)

reports["native_torch_reference"] = json.loads(
    pathlib.Path("tests/fixtures/native/1qjg_run/report.json").read_text())["motif_embedding"]
(OUT / "side_by_side_report.json").write_text(json.dumps(reports, indent=1))
print("done ->", OUT, flush=True)
