"""Dump torch ground-truth batches: run the REAL inference pipeline
(build_design_validation_pipeline + ConditionSpecDataset + collate_batch) on the
SS-diversity condition spec and save the collated tensors as npz fixtures.

These fixtures are the numerical-parity targets for rfd4fx.featurize — the
jnp reimplementation of the batch assembly (conditioning information, masks,
defaults) — while tokenization, the bond graph and chirality packing remain
host-side (numpy) by design.
"""

from __future__ import annotations

import pathlib

import numpy as np
import torch

OUT = pathlib.Path("tests/fixtures/torch_batches")
DATA = pathlib.Path("data/ss_diversity.yaml")


def main() -> None:
    from rfproteina.datapipes.collate import collate_batch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.inference_utils.datasets import ConditionSpecDataset

    pipeline = build_design_validation_pipeline()
    dataset = ConditionSpecDataset(
        data=str(DATA),
        transform=pipeline,
        name="ss_diversity",
        num_replicates=1,
        prevalidate=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    print(f"examples: {n}")
    for idx in range(n):
        sample = dataset[idx]
        collated = collate_batch([sample], schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample")
        inner = collated.data_samples[0] if isinstance(collated.data_samples, list) else collated.data_samples
        tensors = {}
        for key, value in inner.tensors.items():
            if isinstance(value, torch.Tensor):
                tensors[key] = value.detach().cpu()
        # padding lives in PaddedBatchedDataSample.padding_masks, keyed by dim name
        for dim, mask in collated.padding_masks.items():
            tensors[f"padding_mask_{dim}"] = mask.detach().cpu()
        tensors["padding_mask"] = collated.padding_masks["I"].detach().cpu()
        example_id = inner.example_id
        safe = str(example_id[0] if isinstance(example_id, list) else example_id).replace("/", "_")
        np.savez_compressed(OUT / f"{safe}.npz", **{k: v.numpy() for k, v in tensors.items()})
        if idx == 0:
            print(f"{len(tensors)} tensors; sample keys:")
            for k in sorted(tensors)[:12]:
                print(f"  {k:45s} {tuple(tensors[k].shape)}")
    print(f"wrote {n} fixtures to {OUT}")


if __name__ == "__main__":
    main()
