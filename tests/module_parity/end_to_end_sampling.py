"""End-to-end JAX sampling smoke test on the real release weights.

Samples an unconditioned 64-residue protein entirely in JAX (flow integration +
AE decode) on CPU and validates the output geometry. This is the Step 8
end-to-end check; trajectory-level equality with the torch sampler is not
expected (RNG streams differ — documented esmjfold2 convention).
"""

import pathlib
import sys

import numpy as np
import pytest
import jax
import jax.numpy as jnp
import equinox as eqx

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from conftest import (  # noqa: E402
    CACHE,
    REV,
    SNAPSHOT,
    build_feature_batch,
    release_bundle_available,
)

pytestmark = pytest.mark.skipif(not release_bundle_available(),
                                reason="release bundle not in the local HF cache")


def _uncond_batch(n=64, b=1):
    """Properly unconditioned batch: every annotation at its 'unspecified'
    default (the featurization pipeline's output for a bare de-novo chain)."""
    import torch

    from rfd4fx.types import FlowBatch

    def zeros(*s, dtype=jnp.float32):
        return jnp.zeros((b, *s), dtype)

    defaults = dict(
        padding_mask=jnp.ones((b, n), dtype=bool),
        token_type=jnp.zeros((b, n), dtype=jnp.int32),  # protein
        atom37_mask=jnp.zeros((b, n, 37), dtype=bool).at[..., :5].set(True),  # N,CA,C,O,CB
        coords_nm=zeros(n, 37, 3),  # unused (no C_CRD)
        cond_coords=jnp.full((b, n, 37, 3), jnp.nan),  # no coordinate conditioning
        cond_internal_coords=jnp.full((b, n, 37, 3), jnp.nan),
        coord_groups=jnp.full((b, n, 37), -1, jnp.int32),  # UNKNOWN
        rasa=jnp.zeros((b, n, 37), jnp.int32),  # UNSPECIFIED
        secondary_structure=jnp.zeros((b, n), jnp.int32),
        hbond=jnp.zeros((b, n, 37), jnp.int32),
        pistack=jnp.zeros((b, n, 37), dtype=bool),
        pication=jnp.zeros((b, n, 37), jnp.int32),
        saltbridge=jnp.zeros((b, n, 37), jnp.int32),
        hydrophobic=jnp.zeros((b, n, 37), dtype=bool),
        metal_coord=jnp.zeros((b, n, 37), jnp.int32),
        hotspot=jnp.zeros((b, n, 37), dtype=bool),
        chain_centroid=jnp.full((b, n, 3), jnp.nan),
        cath_codes=jnp.full((b, n, 3), -1, jnp.int32),  # null fold class
        idx_group=jnp.zeros((b, n), jnp.int32),
        chirality=None,  # no chirality conditioning -> zero feature
        chiral_frame_atoms=None,
        chiral_slot_owner=None,
        token_bond_path_length=jnp.clip(
            jnp.abs(jnp.arange(n)[None, :] - jnp.arange(n)[:, None])[None], 0, 63
        ).repeat(b, 0).astype(jnp.int32),
        bond_mask=jnp.zeros((b, n, n), dtype=bool),
        bond_order=jnp.zeros((b, n, n), jnp.int32),
        is_aromatic=jnp.zeros((b, n, n), dtype=bool),
        canonical_polymer_ordering=jnp.zeros((b, n, n), jnp.int32),
        molecule_iid=jnp.zeros((b, n), jnp.int32),
        molecule_entity_id=jnp.zeros((b, n), jnp.int32),
        chain_iid=jnp.zeros((b, n), jnp.float32),
        na_ss_pair=jnp.zeros((b, n, n), jnp.float32),
        na_ss_pair_mask=jnp.zeros((b, n, n), jnp.float32),
        na_ss_state=jnp.zeros((b, n), jnp.float32),
        na_ss_state_mask=jnp.zeros((b, n), jnp.float32),
        charge=jnp.zeros((b, n, 37), jnp.int32),
        element=jnp.zeros((b, n, 37), jnp.int32),
        ground_truth_element=jnp.zeros((b, n, 37), jnp.int32),
        incident_bond_type_counts=jnp.zeros((b, n, 37, 10), jnp.float32),
        seq_probabilities=jnp.full((b, n, 28), jnp.nan),  # all unspecified
        is_in_context_conditioning=jnp.zeros((b, n), dtype=bool),
        supports_local_latent=jnp.ones((b, n), dtype=bool),
        recycle_remaining=None,
    )
    jax_batch = FlowBatch(**defaults)
    torch_batch = {k: torch.from_numpy(np.asarray(v).copy()) for k, v in defaults.items() if v is not None}
    torch_batch["charge"] = torch_batch["charge"].long()
    torch_batch["element"] = torch_batch["element"].long()
    torch_batch["ground_truth_element"] = torch_batch["ground_truth_element"].long()
    torch_batch["cath_codes"] = torch_batch["cath_codes"].long()
    torch_batch["coord_groups"] = torch_batch["coord_groups"].long()
    torch_batch["idx_group"] = torch_batch["idx_group"].long()
    torch_batch["molecule_iid"] = torch_batch["molecule_iid"].long()
    torch_batch["molecule_entity_id"] = torch_batch["molecule_entity_id"].long()
    torch_batch["token_type"] = torch_batch["token_type"].long()
    torch_batch["secondary_structure"] = torch_batch["secondary_structure"].long()
    torch_batch["hbond"] = torch_batch["hbond"].long()
    torch_batch["pication"] = torch_batch["pication"].long()
    torch_batch["saltbridge"] = torch_batch["saltbridge"].long()
    torch_batch["metal_coord"] = torch_batch["metal_coord"].long()
    torch_batch["token_bond_path_length"] = torch_batch["token_bond_path_length"].long()
    return torch_batch, jax_batch


def test_end_to_end_sampling(tmp_path):
    from rfd4_proteina import Proteina

    model = Proteina.from_pretrained(SNAPSHOT, revision=REV, cache_dir=CACHE)
    model.eval()

    from rfd4fx.model import RFD4, save_model, load_model

    jax_model = RFD4.from_torch(model)

    # serialization round-trip (Step 7) with the real weights
    ckpt = tmp_path / "release"
    save_model(jax_model, ckpt)
    restored = load_model(ckpt)
    assert eqx.tree_equal(jax_model, restored)

    torch_batch, jax_batch = _uncond_batch(64, 1)

    x, x_1_pred, dec = jax_model.generate(jax_batch, key=jax.random.key(0), nsteps=60)

    # finiteness of the trajectory end state and decode
    for name, arr in (("bb_ca", x["bb_ca"]), ("local_latents", x["local_latents"]),
                      ("coors_nm", dec.coors_nm), ("seq_logits", dec.seq_logits)):
        assert bool(np.isfinite(np.asarray(arr)).all()), f"{name} non-finite"

    # geometry of the decoded structure
    ca_nm = np.asarray(dec.coors_nm)[0, :, 1, :]
    d = np.linalg.norm(ca_nm[1:] - ca_nm[:-1], axis=-1) * 10.0  # Å
    print(f"JAX sample: CA-CA {d.mean():.2f}±{d.std():.2f} Å (max {d.max():.2f}), "
          f"Rg {np.sqrt(((ca_nm - ca_nm.mean(0)) ** 2).sum(-1).mean()) * 10:.1f} Å")
    assert 3.0 < d.mean() < 4.5 and d.max() < 7.0, "unphysical CA chain"
    aatype = np.asarray(dec.aatype_max)[0]
    emitted = np.asarray(dec.emit_mask)[0]
    print(f"sequence: {''.join('ACDEFGHIKLMNPQRSTVWYX'[min(a, 19)] if a < 20 else '?' for a in aatype[emitted])[:64]}")
    assert int(emitted.sum()) == 64

    # decoded CA must agree with the sampled bb_ca at emitted tokens
    err = float(np.abs(np.asarray(dec.coors_nm)[0, :, 1, :] - np.asarray(x_1_pred["bb_ca"])[0]).max())
    print(f"decoded CA vs sampled bb_ca: {err:.3e}")
    assert err < 1e-4
