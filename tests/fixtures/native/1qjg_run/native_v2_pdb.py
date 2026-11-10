"""Write the proper native design PDB from native_v2.npz: designed sequence via
the token-type-restricted sequence-readout argmax (mirroring the port's
condition_decoder logic), real atom names/elements per residue type, motif
reference at input coordinates."""

import pathlib
import sys

import numpy as np

OUT = pathlib.Path("/Users/mjendrusch/repos/rfd4j/rfd4-test-native/1qjg_run")
sys.path.insert(0, "/Users/mjendrusch/repos/rfd4j/jrfproteina/src")

import biotite.structure as bs
import biotite.structure.io.pdb as pdbio
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

z = np.load(OUT / "native_v2.npz")
coors = z["coors"]
seq_logits = z["sequence_logits"]
pad = z["padding_mask"].astype(bool)
ccm = z["coord_cond_mask"].astype(bool)
cond_seq = z["cond_seq"]
token_type = z["token_type"]
coords_in = z["coords_in"]
atom37_mask = z["atom37_mask"]

enc = ENC
idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
              else dict(enumerate(enc.idx_to_token)))
allowed = np.load(OUT / "allowed_residue_types.npy")  # [token_type, vocab]

def restricted_argmax(logits, tt):
    l = logits.copy()
    l[~allowed[tt]] = -np.inf
    return l.argmax(-1)

designed = restricted_argmax(seq_logits[0], token_type) if seq_logits.size else cond_seq

def element_of(atom_name):
    letters = "".join(c for c in atom_name if c.isalpha()).upper()
    if letters[:2] in ("CL", "BR", "MG", "ZN", "FE", "SE"):
        return letters[:2]
    return letters[:1] or "C"

motif = np.flatnonzero(ccm)
motif_set = set(motif.tolist())
arr = bs.AtomArray(0)
for t in np.flatnonzero(pad):
    if t in motif_set:
        continue
    res_name = str(idx_to_tok.get(int(designed[t]), "UNK"))
    slot_names = enc.token_atoms.get(res_name)
    if slot_names is None:
        continue
    for s in range(37):
        xyz = coors[t, s]
        an = str(slot_names[s]).strip()
        if not an or not np.isfinite(xyz).all() or np.abs(xyz).sum() == 0:
            continue
        atom = bs.AtomArray(1)
        atom.coord[0] = xyz
        atom.chain_id[0] = "A"
        atom.res_id[0] = int(t) + 1
        atom.res_name[0] = res_name
        atom.atom_name[0] = an
        atom.element[0] = element_of(an)
        arr += atom
for j, t in enumerate(motif):
    res_name = str(idx_to_tok.get(int(cond_seq[t]), "UNK"))
    slot_names = enc.token_atoms.get(res_name)
    if slot_names is None:
        slot_names = [f"A{s}" for s in range(37)]
    for s in np.flatnonzero(atom37_mask[t] & np.isfinite(coords_in[t]).all(-1)):
        an = str(slot_names[s]).strip() or f"A{s}"
        atom = bs.AtomArray(1)
        atom.coord[0] = coords_in[t, s]
        atom.chain_id[0] = "M"
        atom.res_id[0] = j + 1
        atom.res_name[0] = res_name
        atom.atom_name[0] = an
        atom.element[0] = element_of(an)
        atom.hetero[0] = True
        arr += atom
pdb = pdbio.PDBFile()
pdb.set_structure(arr)
pdb.write(str(OUT / "native_1qjg_design.pdb"))
seq = [str(idx_to_tok.get(int(designed[t]), "?")) for t in np.flatnonzero(pad) if t not in motif_set]
print(f"wrote {OUT / 'native_1qjg_design.pdb'} | designed seq length {len(seq)}")
print("first 60 residues:", "".join({"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
                                     "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
                                     "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
                                     "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
                                    .get(r, "?") for r in seq[:60]))
