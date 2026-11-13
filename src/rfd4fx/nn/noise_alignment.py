"""Dense, jit-able ports of the native conditioning-to-noisy-state alignment
(chirality frames, coordinate groups)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float


def proper_rotation(covariance: Float[Array, "... 3 3"]) -> Float[Array, "... 3 3"]:
    """Kabsch proper-rotation fit from row-vector covariances (utils/align.py).

    torch: u, s, vh = svd(C); det-correct the last singular vector's sign so the
    result is a rotation (not a reflection); return (u * signs) @ vh.
    """
    u, _, vh = jnp.linalg.svd(covariance.astype(jnp.float32))
    det = jnp.linalg.det(u @ vh)
    det_sign = jnp.where(det < 0.0, -1.0, 1.0)
    signs = jnp.ones(u.shape[:-1], dtype=u.dtype).at[..., -1].set(det_sign)
    return (u * signs[..., None, :]) @ vh


def align_chirality(
    chirality_vectors: Float[Array, "b n d"],
    x_t_bb_ca: Float[Array, "b n 3"],
    padding_mask: Bool[Array, "b n"],
    chiral_frame_atoms: Float[Array, "b n s k"],
    chiral_slot_owner: Float[Array, "b n s"],
    *,
    rotate: bool = True,
) -> Float[Array, "b n d"]:
    """Align packed stereochemical vectors (d = 3*S) to the noisy state.

    translate=False only (release config: noise_align_chirality_translate=False);
    the torch translate branch would add the frame origin — omitted, documented
    in status.md.
    """
    b, n, d = chirality_vectors.shape
    n_blocks = d // 3
    valid_frame = (chiral_frame_atoms[..., 0] >= 0) & padding_mask[..., None]  # [B,N,S]

    def _passthrough(_):
        return jnp.where(padding_mask[..., None], chirality_vectors.astype(jnp.float32), 0.0)

    def _align_active(_):
        packed = chirality_vectors.reshape(b, n, n_blocks, 3)
        origins = chiral_frame_atoms[..., :2]  # [B,N,S,2]
        neighbors = chiral_frame_atoms[..., 2:]  # [B,N,S,K]
        present = neighbors >= 0
        neighbors_c = jnp.maximum(neighbors, 0)

        # Advanced-indexing gather: packed[b, neighbor_token, slot] for every frame.
        b_idx = jnp.arange(b)[:, None, None, None]
        s_idx = jnp.arange(n_blocks)[None, None, :, None]
        supplied = packed[b_idx, neighbors_c, s_idx].astype(jnp.float32)  # [B,N,S,K,3]
        noisy_origin = x_t_bb_ca[b_idx, origins].mean(axis=-2, keepdims=True)  # [B,N,S,1,3]
        noisy_vec = x_t_bb_ca[b_idx, neighbors_c] - noisy_origin  # [B,N,S,K,3]

        supplied = jnp.where(present[..., None], supplied, 0.0)
        noisy_vec = jnp.where(present[..., None], noisy_vec, 0.0)
        norm_sq = jnp.sum(jnp.square(noisy_vec), axis=-1, keepdims=True)
        norms = jnp.sqrt(jnp.where(norm_sq > 0.0, norm_sq, 1.0))
        noisy_unit = jnp.where(norm_sq > 0.0, noisy_vec / jnp.clip(norms, min=1e-8), 0.0)

        # Per-address rotation; inactive frames get identity before and after SVD.
        eye = jnp.eye(3, dtype=jnp.float32)
        cov = supplied.transpose(0, 1, 2, 4, 3) @ noisy_unit
        cov = jnp.where(valid_frame[..., None, None], cov, eye)
        rot = proper_rotation(cov)
        rotations = jnp.where(valid_frame[..., None, None], rot, eye)  # [B,N,S,3,3]

        # Slots reference their owner's frame (owner = token * S + block, -1 = none).
        own = jnp.maximum(chiral_slot_owner, 0)
        own_tok = own // n_blocks  # [B,N,S]
        own_blk = own % n_blocks
        b2 = jnp.arange(b)[:, None, None]
        slot_rot = rotations[b2, own_tok, own_blk]  # [B,N,S,3,3]

        aligned = jnp.einsum("bnsx,bnsxy->bnsy", packed.astype(jnp.float32), slot_rot)
        aligned = jnp.where((chiral_slot_owner >= 0)[..., None], aligned, packed.astype(jnp.float32))
        return jnp.where(padding_mask[..., None], aligned.reshape(b, n, d), 0.0)

    if not rotate:
        return _passthrough(None)
    return jax.lax.cond(jnp.any(valid_frame), _align_active, _passthrough, None)
