"""Torch-free featurization of RFD4 annotated-CIF design files.

``annotated_cif_to_batch(path)`` reads an RFD4-spec annotated CIF and returns a
model-ready FlowBatch plus the coordinate-conditioning mask."""

from __future__ import annotations

import pathlib

import numpy as np

from .featurize import SEQ_PROB_ORDER, _KEY_TO_FIELD, assemble_flow_batch
from .types import FlowBatch


def _load_pipeline(np_seed: int | None):
    """Parse + the numpy-only featurization pipeline (torch-free)."""
    from atomworks.io.parser import parse
    from atomworks.ml.transforms.base import Compose
    from atomworks.ml.transforms.base import ConvertToTorch
    from rfproteina.datapipes.pipeline import build_design_validation_pipeline
    from rfproteina.transforms.invariants import ValidateBatchInvariants
    from rfproteina.utils.parse_presets import ANNOTATED_CIF

    def load_and_featurize(path: str | pathlib.Path) -> dict:
        data = parse(str(path), config=ANNOTATED_CIF)
        # extract the first assembly model (exactly as DesignProblemDataset does)
        data["atom_array"] = data["assemblies"]["1"][0]
        data.pop("asym_unit")
        data.pop("assemblies")
        if np_seed is not None:
            np.random.seed(np_seed)  # chain_iid/molecule_iid permutations
        pipeline = build_design_validation_pipeline()
        transforms = [
            t for t in pipeline.transforms
            # ConvertToTorch is the only torch transform; ValidateBatchInvariants
            # assumes torch tensors (a checks-only transform — the parity harness
            # against the torch pipeline is the stronger validation anyway)
            if not isinstance(t, (ConvertToTorch, ValidateBatchInvariants))
        ]
        return Compose(transforms)(data)

    return load_and_featurize


def annotated_cif_to_batch(
    path: str | pathlib.Path,
    *,
    np_seed: int | None = 0,
) -> tuple[FlowBatch, np.ndarray]:
    """Annotated CIF file → (FlowBatch, coord_cond_mask) of jax.numpy features.

    Returns the batch (b=1, n=padded token count — unpadded: every token real) and
    the coordinate-conditioning mask used by the sampler's stochastic centering.
    """
    load_and_featurize = _load_pipeline(np_seed)
    data = load_and_featurize(path)
    feats = data["feats"]

    host: dict[str, np.ndarray] = {}
    for key, value in feats.items():
        if key in _KEY_TO_FIELD or key in SEQ_PROB_ORDER or key in ("coords", "coord_cond_mask"):
            host[key] = np.asarray(value)[None]  # token-level [n, ...] -> [1, n, ...]
    n = host["token_type"].shape[1]
    host["padding_mask"] = np.ones((1, n), dtype=bool)
    # sanity: every kept feature must be token-length
    for key, value in host.items():
        if key == "padding_mask":
            continue
        assert value.shape[1] == n, f"{key}: {value.shape} vs n={n}"

    return assemble_flow_batch(host)
