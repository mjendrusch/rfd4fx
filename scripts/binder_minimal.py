"""Minimal binder design with RFD4: de-novo binder chain + target PDB motif.

The target is conditioned as a motif (--target_selection picks the loaded
chains/residues/atoms); --hotspots passes selection-grammar hotspot entries
(C_HOT on the selected residues/atoms, plus C_HB donor/acceptor conditioning
with --auto_condition).

Example:
    python binder_minimal.py --in_path target.pdb --target_selection "A" \
        --hotspots "A45;A47@N,CA,C" --num_designs 5 --num_aa 80 \
        --out_path out_binder/
"""

from __future__ import annotations

import argparse
import pathlib

import jax
import numpy as np

from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in_path", type=pathlib.Path, required=True)
    parser.add_argument("--target_selection", default="",
                        help="semicolon-separated selection entries "
                             "(chain/residue/atom) for the target")
    parser.add_argument("--hotspots", default="",
                        help="semicolon-separated hotspot selection entries "
                             "(C_HOT; + C_HB with --auto_condition)")
    parser.add_argument("--auto_condition", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="hbond donor/acceptor conditioning on the "
                             "hotspot residues")
    parser.add_argument("--num_designs", type=int, default=5)
    parser.add_argument("--num_aa", type=int, default=80)
    parser.add_argument("--out_path", type=pathlib.Path,
                        default=pathlib.Path("runs/rfd4-test-binder"))
    args = parser.parse_args()
    args.out_path.mkdir(parents=True, exist_ok=True)

    target = {"pdb": str(args.in_path)}
    if args.target_selection:
        target["select"] = args.target_selection.split(";")
    if args.hotspots:
        target["hotspot"] = args.hotspots
    target["auto_condition"] = args.auto_condition

    inp = RFD4Input.from_chains(args.num_aa, target)
    print(f"compiled: {int(np.asarray(inp.padding_mask).sum())} tokens",
          flush=True)

    jax_model = RFD4.from_pretrained()           # pure-jax checkpoint, torch-free
    sampler = RFD4Sampler(jax_model.step(), jax_model)
    batch = inp.pad_to_nearest(64).to_batch()

    for i in range(args.num_designs):
        result = sampler.sample(batch, key=jax.random.fold_in(jax.random.key(0), i),
                                recycle_schedule=THINKING_HIGH)
        out = args.out_path / f"binder_{i:02d}.pdb"
        result.save_pdb(out)
        print(f"binder_{i:02d}.pdb written", flush=True)


if __name__ == "__main__":
    main()
