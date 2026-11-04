"""Interface-motif scaffolding with RFD4.

Given a two-chain PDB: the residues of chain A with at least one heavy atom
within ``--cutoff`` of chain B are detected; their backbone atoms
(N, CA, C, O, OXT) are dropped and the remaining side-chain atoms condition
an UNINDEXED motif, while chain B conditions an INDEXED motif — both motifs
share ONE internal coordinate group (one rigid body: the library auto-merges
fixed=False motif pieces with the same coordinate group into a single frame
at to_batch time), alongside a designable ``--num_aa`` residue chain.

Example:
    python generate_interface_motif_jax.py --in_path complex.pdb \
        --num_aa 72 --num_designs 5 --out_path rfd4-test-interface/
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
    parser.add_argument("--chain_a", default="A",
                        help="donor chain (interface sidechain motif)")
    parser.add_argument("--chain_b", default="B",
                        help="partner chain (indexed backbone motif)")
    parser.add_argument("--cutoff", type=float, default=4.5)
    parser.add_argument("--fixed", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="condition the motif positions absolutely "
                             "(fixed=True; C_CRD, coordinate group 0) vs "
                             "internal coordinates (fixed=False; C_INTCRD/"
                             "C_CRDGRP, one shared rigid body)")
    parser.add_argument("--save_input", type=pathlib.Path, default=None,
                        help="save the compiled input batch (npz) for "
                             "inspection")
    parser.add_argument("--num_aa", type=int, required=True,
                        help="length of the designable chain")
    parser.add_argument("--num_designs", type=int, default=5)
    parser.add_argument("--out_path", type=pathlib.Path,
                        default=pathlib.Path("rfd4-test-interface"))
    args = parser.parse_args()
    args.out_path.mkdir(parents=True, exist_ok=True)

    inp = RFD4Input.from_chains(
        args.num_aa,
        {"pdb": str(args.in_path), "fixed": args.fixed,
         "motif": [
             # chain B: indexed motif (full atoms)
             {"select": [args.chain_b]},
             # chain A: interface residues only, sidechains-only unindexed
             # motif — same coordinate group as B (one rigid body for
             # fixed=False; absolute group 0 for fixed=True); the motif
             # residues join chain_iid 0 (target_chain "A") instead of a
             # fresh chain
             {"select": [args.chain_a], "indexed": False,
              "interface_to": args.chain_b, "cutoff": args.cutoff,
              "drop_backbone": True, "ordered": True,
              "target_chain": "A"},
         ]})
    n = int(np.asarray(inp.padding_mask).sum())
    mode = "absolute (C_CRD, group 0)" if args.fixed else \
        "internal (C_INTCRD/C_CRDGRP, one shared rigid body)"
    print(f"compiled: {n} tokens ({args.num_aa} design + {n - args.num_aa} "
          f"motif: {args.chain_b} indexed + {args.chain_a} interface "
          f"sidechains; coordinate conditioning {mode})", flush=True)
    if args.save_input is not None:
        args.save_input.parent.mkdir(parents=True, exist_ok=True)
        arrays = {k: np.asarray(v) for k, v in inp.to_dict().items()}
        arrays["coord_cond"] = np.asarray(inp.coord_cond)
        np.savez_compressed(args.save_input, **arrays)
        print(f"input batch saved to {args.save_input}", flush=True)

    jax_model = RFD4.from_pretrained()      # pure-jax checkpoint, torch-free
    sampler = RFD4Sampler(jax_model.step(), jax_model)
    batch = inp.pad_to_nearest(64).to_batch()

    for i in range(args.num_designs):
        result = sampler.sample(
            batch, key=jax.random.fold_in(jax.random.key(0), i),
            recycle_schedule=THINKING_HIGH)
        out = args.out_path / f"interface_{i:02d}.pdb"
        result.save_pdb(out)
        print(f"interface_{i:02d}.pdb written", flush=True)


if __name__ == "__main__":
    main()
