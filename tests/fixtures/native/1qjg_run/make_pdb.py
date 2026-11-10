"""Convert the saved native 1QJG run outputs (raw_outputs.npz) into an
inspectable PDB + a metrics README. Run after the native sampling finishes."""

import json
import pathlib

import numpy as np

OUT = pathlib.Path("/Users/mjendrusch/repos/rfd4j/rfd4-test-native/1qjg_run")
npz = np.load(OUT / "raw_outputs.npz")
coors = npz["coors"]            # (n, 37, 3) Å (designed + motif token slots)
pad = npz["padding_mask"]
ccm = npz["coord_cond_mask"]
rt = npz["residue_type"]
cond_seq = npz["cond_seq"]

# AE residue-type vocabulary for naming the motif tokens
import sys
sys.path.insert(0, "/Users/mjendrusch/repos/rfd4j/jrfproteina/src")
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC  # noqa: E402

idx_to_tok = (ENC.idx_to_token if isinstance(ENC.idx_to_token, dict)
              else dict(enumerate(ENC.idx_to_token)))

lines = []
serial = 0

def atom_line(serial, name, res, chain, resseq, xyz, element):
    return (f"{'ATOM':<6}{serial:>5} {name:^4}{' ':1}{res:>3} {chain:>1}"
            f"{resseq:>4}{' ':4}{xyz[0]:>8.3f}{xyz[1]:>8.3f}{xyz[2]:>8.3f}"
            f"{1.0:>6.2f}{0.0:>6.2f}          {element:>2}{'':2}")

real = np.flatnonzero(pad)
motif = np.flatnonzero(ccm)
for t in real:
    if t in motif:
        continue
    xyz = coors[t, 1]
    if not np.isfinite(xyz).all():
        continue
    serial += 1
    lines.append(atom_line(serial, "CA", "UNK", "A", int(t) + 1, xyz, "C"))
# motif tokens at their conditioned positions (x_t is emit-masked there)
cc = None  # conditioned CA coords are recovered from the cond field if saved
for t in motif:
    xyz = coors[t, 1]
    if not np.isfinite(xyz).all() or np.allclose(xyz, 0):
        continue
    serial += 1
    res = str(idx_to_tok.get(int(cond_seq[t]) if cond_seq[t] > 0 else int(rt[t]), "UNK"))
    lines.append(atom_line(serial, "CA", res[:3].upper(), "M", int(t) + 1, xyz, "C"))
(OUT / "native_1qjg_ca_trace.pdb").write_text("\n".join(lines) + "\n")

report = {
    "input": "data/benchmarks/motif_scaffolding/motif_unindexed/1QJG_AA_NATIVE.cif.zst"
             " (native pipeline, native torch model, release sampler args + recycle schedule)",
    "note": "CA trace of the generated design: chain A = designed stubs (UNK), "
            "chain M = motif residues at their predicted positions (in-context "
            "motif tokens are emit-masked in the flow state, so their plotted "
            "coordinates are the model's x_1_pred recovery, not exact pins).",
    "real_tokens": int(pad.sum()),
    "motif_tokens": motif.tolist(),
}
(OUT / "report.json").write_text(json.dumps(report, indent=1))
print("wrote", OUT / "native_1qjg_ca_trace.pdb", "with", serial, "CA atoms")
