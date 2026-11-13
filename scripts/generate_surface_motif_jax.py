"""Surface-neighbour motif scaffolding with RFD4.

Given a structure and a residue selection (``--residues``, e.g. "A48"; the
selection grammar is the one parsed by ``RFD4Input.from_chains``), the
residues of the selection's chains with at least one sidechain heavy atom
within ``--cutoff`` of the selection are detected ("neighbours"); with
``--surface`` only neighbours whose CB exposes a Shrake-Rupley sphere
(``is_surface``) are kept. The selection plus its (surface) neighbours form
an UNINDEXED side-chain-only motif (backbone N/CA/C/O/OXT dropped, one
internal-coordinate rigid body for fixed=False) fused into the designable
chain (target_chain "A"), alongside ``--num_aa`` designable residues.

Example:
    python scripts/generate_surface_motif_jax.py --in_path data/8znl.pdb \
        --residues A48 --cutoff 5 --surface --num_aa 72 --num_designs 5
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
    parser.add_argument("--in_path", type=pathlib.Path,
                        default=pathlib.Path("data/8znl.pdb"))
    parser.add_argument("--residues", required=True,
                        help="comma-separated select entries seeding the "
                             "neighbour search (e.g. 'A48' or 'A48,A52')")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="sidechain-to-selection distance in A")
    parser.add_argument("--surface", type=lambda s: s.strip().lower()
                        in ("1", "true", "yes", "y"), default=True,
                        metavar="{True,False}",
                        help="keep only Shrake-Rupley-surface neighbours")
    parser.add_argument("--fixed", type=lambda s: s.strip().lower()
                        in ("1", "true", "yes", "y"), default=False,
                        metavar="{True,False}",
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
                        default=pathlib.Path("runs/surface-motif"))
    args = parser.parse_args()
    args.out_path.mkdir(parents=True, exist_ok=True)

    inp = RFD4Input.from_chains(
        args.num_aa,
        {"pdb": str(args.in_path), "fixed": args.fixed,
         "motif": [
             # selection + its (surface) neighbours: sidechain-only
             # unindexed motif fused into the designable chain (iid 0)
             {"select": [s.strip() for s in args.residues.split(",")
                         if s.strip()],
              "indexed": False, "neighbours": True, "surface": args.surface,
              "cutoff": args.cutoff, "drop_backbone": True, "ordered": True,
              "target_chain": "A"},
         ]})
    n = int(np.asarray(inp.padding_mask).sum())
    mode = "absolute (C_CRD, group 0)" if args.fixed else \
        "internal (C_INTCRD/C_CRDGRP, one shared rigid body)"
    print(f"compiled: {n} tokens ({args.num_aa} design + {n - args.num_aa} "
          f"motif: {args.residues} + "
          f"{'surface ' if args.surface else ''}neighbours "
          f"(cutoff {args.cutoff} A); coordinate conditioning {mode})",
          flush=True)
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
        out = args.out_path / f"surface_{i:02d}.pdb"
        result.save_pdb(out)
        print(f"surface_{i:02d}.pdb written", flush=True)


if __name__ == "__main__":
    main()
