"""Unconditional protein generation with the JAX port of RFD4-Proteina.

Samples `--num-designs` proteins of `--n-residues` residues (batched) through
the full JAX pipeline — flow matching (60 vf_tsr steps, 180 trunk passes) +
AE decode — and writes one validated PDB per design.

Example:
    python scripts/generate_uncond_jax.py --n-residues 100 --num-designs 100 \
        --batch-size 10 --out-dir runs/rfd4-jax-testruns
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-residues", type=int, default=100)
    parser.add_argument("--num-designs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--checkpoint", type=str, default=SNAPSHOT)
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="HF cache dir for the release checkpoint "
                             "(default: the huggingface default cache)")
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from atomworks.ml.transforms.encoding import atom_array_from_encoding
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.featurize import make_unconditioned_batch
    from rfd4fx.model import RFD4

    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint)
    model.eval()
    jax_model = RFD4.from_torch(model)

    _, template = make_unconditioned_batch(args.n_residues, b=args.batch_size)

    manifest = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha": "a06e2509c7033b010fc43614e9fd840303588502",
        "n_residues": args.n_residues,
        "num_designs": args.num_designs,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "nsteps": args.nsteps,
        "backend": "jax (rfd4fx)",
        "designs": [],
    }

    n_batches = (args.num_designs + args.batch_size - 1) // args.batch_size
    design_idx = 0
    t0 = time.time()
    for batch_i in range(n_batches):
        b = min(args.batch_size, args.num_designs - design_idx)
        keys = [jax.random.fold_in(jax.random.key(args.seed), design_idx + j) for j in range(b)]
        batch = jax.tree.map(
            lambda x: x[:b] if hasattr(x, "shape") and x.shape and x.shape[0] == args.batch_size else x,
            template,
            is_leaf=lambda l: eqx_leaf(l),
        )
        key = jax.random.fold_in(jax.random.key(args.seed), 10_000 + batch_i)
        x, x_1_pred, dec = jax_model.generate(batch, key=key, nsteps=args.nsteps)

        coors_a = np.asarray(dec.coors_nm) * 10.0  # nm -> Å
        aatype = np.asarray(dec.aatype_max)
        atom_mask = np.asarray(dec.atom_mask)
        emit = np.asarray(dec.emit_mask)

        for j in range(b):
            seq = aatype[j]
            mask_j = atom_mask[j] & emit[j][:, None]
            coords37 = coors_a[j].copy()
            coords37[~mask_j] = np.nan
            # tokens outside the emit set: -1 marks them (none for uncond generation)
            res_type = np.where(emit[j], seq, -1).astype(np.int64)
            arr = atom_array_from_encoding(
                encoded_coord=coords37,
                encoded_mask=mask_j,
                encoded_seq=res_type,
                encoding=UNIFIED_ATOM37_ENCODING,
                chain_id="A",
            )
            # drop NaN atoms (absent slots)
            arr = arr[np.isfinite(arr.coord).all(axis=-1)]
            out = args.out_dir / f"design_{design_idx + j:03d}.pdb"
            pdb = pdbio.PDBFile()
            pdb.set_structure(arr)
            pdb.write(str(out))
            # validate by re-reading
            back = pdbio.PDBFile.read(str(out)).get_structure(model=1)
            ca = back[back.atom_name == "CA"]
            assert len(ca) == args.n_residues, f"design {design_idx + j}: {len(ca)} CA atoms"
            d = np.linalg.norm(ca.coord[1:] - ca.coord[:-1], axis=-1)
            assert bool(np.isfinite(back.coord).all()), f"design {design_idx + j}: non-finite"
            manifest["designs"].append(
                {
                    "id": f"design_{design_idx + j:03d}",
                    "seed": args.seed,
                    "fold_in": design_idx + j,
                    "file": out.name,
                    "n_atoms": int(back.array_length()),
                    "ca_ca_mean": float(d.mean()),
                    "ca_ca_max": float(d.max()),
                }
            )
            print(
                f"[{design_idx + j + 1}/{args.num_designs}] {out.name}: "
                f"{back.array_length()} atoms, CA-CA {d.mean():.2f}±{d.std():.2f} Å "
                f"(max {d.max():.2f})  [{time.time() - t0:.0f}s]",
                flush=True,
            )
        design_idx += b

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {args.num_designs} designs in {time.time() - t0:.0f}s -> {args.out_dir}")


def eqx_leaf(l):
    import equinox as eqx

    return eqx.is_array(l)


if __name__ == "__main__":
    main()
