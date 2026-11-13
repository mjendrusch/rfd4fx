"""Binder design against a disordered target sequence with RFD4.

The target is conditioned by SEQUENCE ONLY (`--target_sequence`, one-letter
codes; C_SEQ on every target residue) — no coordinates are given, so the
peptide's fold and pose are generated jointly with the binder. `--hotspots`
marks 1-indexed target residues ("1,5,6-10"; ranges inclusive) as hotspots
(C_HOT on their sidechain atoms), `--auto_condition` adds hydrogen-bond
donor/acceptor conditioning (C_HB) to all sidechain donors/acceptors of the
target, and the explicit interaction keys `--h_bond`, `--salt_bridge`,
`--pi_stack`, `--pi_cation` condition the eligible subset of the selected
residues/atoms (C_HB / C_SALTBRIDGE / C_PISTACK / C_PICATION).
`--auto_condition` is mutually exclusive with the explicit interaction keys.
`--kind` selects the target vocabulary (aa/dna/rna).

Example:
    python scripts/binder_disordered.py --target_sequence MERRYCHRISTMAS \
        --hotspots "1,5,6-10" --num_designs 5 --num_aa 80
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
    parser.add_argument("--target_sequence", required=True,
                        help="target sequence, one-letter codes")
    parser.add_argument("--kind", default="aa", choices=["aa", "dna", "rna"],
                        help="target vocabulary (default: aa)")
    parser.add_argument("--hotspots", default="",
                        help="comma-separated 1-indexed target residues "
                             "(e.g. '1,5,6-10'; C_HOT on their sidechains)")
    parser.add_argument("--auto_condition", type=lambda s: s.strip().lower()
                        in ("1", "true", "yes", "y"), default=False,
                        metavar="{True,False}",
                        help="C_HB donor/acceptor conditioning on all "
                             "sidechain donors/acceptors of the target "
                             "(mutually exclusive with the interaction keys)")
    for key, cond in (("h_bond", "C_HB donor/acceptor roles (1/2/3)"),
                      ("salt_bridge", "C_SALTBRIDGE charge sign (1/2)"),
                      ("pi_stack", "C_PISTACK on aromatic-ring atoms"),
                      ("pi_cation",
                       "C_PICATION aromatic (1) / cation (2) roles")):
        parser.add_argument(f"--{key}", default=None,
                            help=f"{cond} on the eligible subset of the "
                                 "selected target residues/atoms (mutually "
                                 "exclusive with --auto_condition)")
    parser.add_argument("--num_designs", type=int, default=5)
    parser.add_argument("--num_aa", type=int, default=80,
                        help="length of the designable binder chain")
    parser.add_argument("--out_path", type=pathlib.Path,
                        default=pathlib.Path("runs/binder-disordered"))
    args = parser.parse_args()
    interactions = {k: getattr(args, k) for k in
                    ("h_bond", "salt_bridge", "pi_stack", "pi_cation")
                    if getattr(args, k) is not None}
    if args.auto_condition and interactions:
        parser.error("--auto_condition is mutually exclusive with the "
                     "explicit interaction keys (--h_bond/--salt_bridge/"
                     "--pi_stack/--pi_cation)")
    args.out_path.mkdir(parents=True, exist_ok=True)

    target = {"kind": args.kind, "sequence": args.target_sequence}
    if args.auto_condition:
        target["auto_condition"] = True
    if args.hotspots:
        target["hotspot"] = args.hotspots
    target.update(interactions)

    inp = RFD4Input.from_chains(args.num_aa, target)
    n_target = int(np.asarray(inp.padding_mask).sum()) - args.num_aa
    channels = ["C_HOT"] if args.hotspots else []
    if args.auto_condition:
        channels.append("C_HB")
    channels += [{"h_bond": "C_HB", "salt_bridge": "C_SALTBRIDGE",
                  "pi_stack": "C_PISTACK", "pi_cation": "C_PICATION"}[k]
                 for k in interactions]
    print(f"compiled: {args.num_aa} design + {n_target} target tokens "
          f"({args.kind} sequence-only conditioning: {' + '.join(['C_SEQ'] + channels)}"
          f"; no coordinates)", flush=True)

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
