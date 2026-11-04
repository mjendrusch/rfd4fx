"""Per-token feature modules for the active checkpoint (jit-able).

Each class mirrors its torch counterpart in ``rfproteina.nn.feature_factory``."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int

from ..backend import AbstractFromTorch, from_torch
from ..types import FlowBatch
from .layers import Embedding, LayerNorm, Linear, RMSNorm, Sequential
from .noise_alignment import align_chirality

# --- constants baked from the torch-side enums/vocabulary (queried at port time;
# inference must be torch-free, so they are literals here) ---
_RASA_UNSPECIFIED = 0  # RasaEnum.UNSPECIFIED
_HB_DONOR, _HB_ACCEPTOR, _HB_BOTH = 1, 2, 3  # HBondEnum
_PC_AROMATIC, _PC_CATION, _PC_BOTH = 1, 2, 3  # PiCationEnum
_SALT_NONE = 0  # SaltBridgeEnum.NONE
_MCOORD_NONE = 0  # MetalCoordEnum.NONE
_CRDGRP_UNKNOWN = -1  # CoordinateGroup.UNKNOWN_COORDS
_CRDGRP_ABSOLUTE = 0  # CoordinateGroup.KNOWN_ABSOLUTE_COORDS
_TOKEN_TYPE_ATOMIZED = 3  # TokenType.ATOMIZED
_SEQ_MASK_TOKEN = 0  # UNIFIED_ATOM37_ENCODING.token_to_idx["<M>"]
_N_SEQ_PROB_CONDITIONS = 28  # len(SEQUENCE_PROBABILITY_CONDITIONS)


# --- jit-able helpers (mirror module-level torch functions) -------------------


def get_time_embedding(t: Float[Array, "b"], edim: int, max_positions: int = 2000) -> Float[Array, "b edim"]:
    """FrameFlow-style sinusoidal time embedding (feature_factory.get_time_embedding)."""
    t = t * max_positions
    half_dim = edim // 2
    # math.log on a Python constant is compile-time (allowed; not numpy on traced data).
    emb = jnp.exp(jnp.arange(half_dim, dtype=jnp.float32) * -(math.log(max_positions) / (half_dim - 1)))
    emb = t[:, None].astype(jnp.float32) * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=1)
    if edim % 2 == 1:
        emb = jnp.pad(emb, ((0, 0), (0, 1)))
    return emb


def bin_and_one_hot(tensor: Float[Array, "..."], bin_limits: Float[Array, " l"]) -> Float[Array, "... l1"]:
    """torch.bucketize(tensor, bin_limits) with default right=False equals
    jnp.searchsorted(..., side='left') (count of limits strictly below the value);
    verified numerically by the feature parity tests."""
    bin_indices = jnp.searchsorted(bin_limits, tensor, side="left")
    return jax.nn.one_hot(bin_indices, len(bin_limits) + 1)


def bin_pairwise_distances(x: Float[Array, "b n 3"], min_dist: float, max_dist: float, dim: int):
    pair_dists = jnp.linalg.norm(x[:, :, None, :] - x[:, None, :, :], axis=-1)
    bin_limits = jnp.linspace(min_dist, max_dist, dim - 1)
    return bin_and_one_hot(pair_dists, bin_limits)


def embed_charge(embedding: Embedding, charges: Int[Array, "..."]) -> Float[Array, "... d"]:
    # AtomChargeSeqFeat.embed_charge: saturated charge vocabulary -4..+6 -> 0..10.
    indices = jnp.clip(charges.astype(jnp.int32) + 4, 0, 10)
    return embedding(indices)


# --- batch / time features (no parameters) ------------------------------------


class TimeEmbeddingBatchFeat(eqx.Module):
    dim: int = 0
    data_mode_use: str = "bb_ca"  # static

    def __call__(self, batch: FlowBatch, *, key=None):
        t = batch.t_bb_ca if self.data_mode_use == "bb_ca" else batch.t_local_latents
        return get_time_embedding(t, self.dim)


class RecycleRemainingBatchFeat(eqx.Module):
    dim: int = 0

    def __call__(self, batch: FlowBatch, *, key=None):
        remaining = batch.recycle_remaining
        if remaining is None:
            return jnp.zeros((batch.padding_mask.shape[0], self.dim))
        # remaining may be a Python int (trace-time) or a traced 0-d array.
        indices = jnp.full((batch.padding_mask.shape[0],), jnp.minimum(remaining, self.dim - 1), dtype=jnp.int32)
        return jax.nn.one_hot(indices, self.dim)


# --- sequence features ---------------------------------------------------------


class XtBBCASeqFeat(eqx.Module):
    dim: int = 3

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.x_t_bb_ca


class XtLocalLatentsSeqFeat(eqx.Module):
    dim: int = 0

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.x_t_local_latents


class CaCoorsNanometersSeqFeat(eqx.Module):
    dim: int = 3

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.ca_coors_nm is not None:
            return batch.ca_coors_nm
        return batch.coords_nm[:, :, 1, :]


class LatentVariableSeqFeat(eqx.Module):
    dim: int = 0

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.z_latent


class ChainIdxSeqFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.chain_iid[..., None].astype(jnp.float32)


class TokenTypeSeqFeat(AbstractFromTorch):
    embedding: Embedding
    token_type_emb_dim: int = 32

    def __call__(self, batch: FlowBatch, *, key=None):
        return self.embedding(batch.token_type)


class SequenceProbabilitiesSeqFeat(eqx.Module):
    dim: int = 2 * _N_SEQ_PROB_CONDITIONS

    def __call__(self, batch: FlowBatch, *, key=None):
        # batch.seq_probabilities stacks the 28 per-residue conditions with NaN
        # where unspecified (torch stacks them by name at forward time).
        probabilities = batch.seq_probabilities.astype(jnp.float32)
        masks = ~jnp.isnan(probabilities)
        return jnp.concatenate((jnp.where(masks, probabilities, 0.0), masks.astype(jnp.float32)), axis=-1)


class Atom37NanometersCoorsSeqFeat(eqx.Module):
    dim: int = 37 * 4
    rel: bool = False

    def __call__(self, batch: FlowBatch, *, key=None):
        coors = batch.coords_nm * batch.atom37_mask[..., None]
        if self.rel:
            ca = coors[:, :, 1, :]
            coors = (coors - ca[:, :, None, :]) * batch.atom37_mask[..., None]
        coors_flat = coors.reshape(*coors.shape[:-2], -1)
        return jnp.concatenate([coors_flat, batch.atom37_mask.astype(jnp.float32)], axis=-1)


class ConditionalAtom37NanometersCoorsSeqFeat(eqx.Module):
    dim: int = 37 * 4
    rel: bool = False

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.cond_coords is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        conditioned = batch.cond_coords
        # Finite coords mark conditioned atoms; unconditioned entries are NaN.
        coors_mask = jnp.isfinite(conditioned).any(axis=-1)
        coors = jnp.nan_to_num(conditioned, nan=0.0) / 10.0  # Å -> nm
        coors = coors * coors_mask[..., None]
        if self.rel:
            ca = coors[:, :, 1, :]
            coors = (coors - ca[:, :, None, :]) * coors_mask[..., None]
        coors_flat = coors.reshape(*coors.shape[:-2], -1)
        return jnp.concatenate([coors_flat, coors_mask.astype(jnp.float32)], axis=-1)


class ConditionalAtom37NanometersInternalCoorsSeqFeat(eqx.Module):
    """Internal-coordinate conditioning. The release config has
    noise_align_internal_coords=False, so only the pass-through path is live;
    the align_internal_coords rewrite (batch/group Python loops) is not ported."""

    dim: int = 37 * 4
    noise_align: bool = False
    translate: bool = False

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.cond_internal_coords is None or batch.coord_groups is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        assert not (self.noise_align or self.translate), "internal-coord alignment not ported (unused by config)"
        mask = (batch.coord_groups != _CRDGRP_UNKNOWN) & (batch.coord_groups != _CRDGRP_ABSOLUTE)
        internal = jnp.nan_to_num(batch.cond_internal_coords, nan=0.0) / 10.0
        internal = internal * mask[..., None]
        flat = internal.reshape(*internal.shape[:-2], -1)
        return jnp.concatenate([flat, mask.astype(jnp.float32)], axis=-1)


class ChiralitySeqFeat(eqx.Module):
    dim: int = 18  # MAX_OVERLAPPING_CENTERS * 3
    noise_align: bool = False
    translate: bool = False

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.chirality is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        if self.noise_align or self.translate:
            assert not self.translate, "chirality translate not ported (unused by config)"
            return align_chirality(
                batch.chirality,
                batch.x_t_bb_ca,
                batch.padding_mask,
                batch.chiral_frame_atoms,
                batch.chiral_slot_owner,
                rotate=self.noise_align,
            )
        return batch.chirality.astype(jnp.float32)


class CoordGroupSeqFeat(eqx.Module):
    """Random relabeling of coordinate groups. The torch version uses
    ``torch.unique`` + ``torch.randperm`` (dynamic shapes); this is an exact
    distributional equivalent: present group ids get a uniform random bijection
    onto {0..P-1} via a static argsort-rank over the id range (see status.md).
    Numeric parity is impossible in principle (labels are random); the parity
    test checks the bijection/structure instead."""

    dim: int = 37
    max_group_id: int = 64  # static bound; annotation group ids are small ints

    def __call__(self, batch: FlowBatch, *, key=None):
        assert key is not None, "CoordGroupSeqFeat requires an RNG key"
        cg = batch.coord_groups.astype(jnp.int32)
        non_padding = cg != _CRDGRP_UNKNOWN
        ids = jnp.arange(self.max_group_id + 1)
        # present[v] = does group id v occur anywhere in the batch (torch.unique semantics)
        present = jnp.any(cg[..., None] == ids, axis=(0, 1, 2))
        # Random bijection present-ids -> {0..P-1}: sort present ids by a random
        # key, absent ids get +inf and sort last; rank = argsort(argsort(keys)).
        keys = jnp.where(present, jax.random.uniform(key, (self.max_group_id + 1,)), jnp.inf)
        order = jnp.argsort(keys)
        labels = jnp.empty_like(order).at[order].set(jnp.arange(self.max_group_id + 1))
        new_cg = jnp.where(non_padding, labels[jnp.clip(cg, 0, self.max_group_id)], cg)
        return new_cg.astype(jnp.float32)


class RasaBinnedSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.rasa is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, 37))
        # torch: rasa_atom37[~atom37_mask] = UNSPECIFIED (0), in-place.
        rasa = jnp.where(batch.atom37_mask, batch.rasa, _RASA_UNSPECIFIED)
        return rasa.astype(jnp.float32)


class ChainCentroidXyzSeqFeat(eqx.Module):
    dim: int = 3

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.chain_centroid is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        included = jnp.isfinite(batch.chain_centroid).all(axis=-1)
        return jnp.where(included[..., None], batch.chain_centroid / 10.0, 0.0)


class SecondaryStructureSeqFeat(AbstractFromTorch):
    ss_embedding: Embedding
    embed_dim: int = 16

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.secondary_structure is None:
            b, n = batch.padding_mask.shape
            # torch adds 0.0 * weight.sum() for DDP; numerically zeros.
            return jnp.zeros((b, n, self.embed_dim))
        return self.ss_embedding(batch.secondary_structure.astype(jnp.int32))


class ModernizedHydrogenBondSeqFeat(eqx.Module):
    dim: int = 74

    def __call__(self, batch: FlowBatch, *, key=None):
        hb = batch.hbond
        atom37_mask = batch.atom37_mask.astype(jnp.float32)
        is_donor = ((hb == _HB_DONOR) | (hb == _HB_BOTH)).astype(jnp.float32) * atom37_mask
        is_acceptor = ((hb == _HB_ACCEPTOR) | (hb == _HB_BOTH)).astype(jnp.float32) * atom37_mask
        return jnp.concatenate([is_donor, is_acceptor], axis=-1)


class PiStackingSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.pistack.astype(jnp.float32) * batch.atom37_mask.astype(jnp.float32)


class PiCationSeqFeat(eqx.Module):
    dim: int = 74

    def __call__(self, batch: FlowBatch, *, key=None):
        pc = batch.pication
        atom37_mask = batch.atom37_mask.astype(jnp.float32)
        is_aromatic = ((pc == _PC_AROMATIC) | (pc == _PC_BOTH)).astype(jnp.float32) * atom37_mask
        is_cation = ((pc == _PC_CATION) | (pc == _PC_BOTH)).astype(jnp.float32) * atom37_mask
        return jnp.concatenate([is_aromatic, is_cation], axis=-1)


class SaltBridgeSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        salt = batch.saltbridge.astype(jnp.int32)
        return (salt != _SALT_NONE).astype(jnp.float32) * batch.atom37_mask.astype(jnp.float32)


class HydrophobicSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        return batch.hydrophobic.astype(jnp.float32) * batch.atom37_mask.astype(jnp.float32)


class MetalCoordinationSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        mcoord = batch.metal_coord.astype(jnp.int32)
        return (mcoord != _MCOORD_NONE).astype(jnp.float32) * batch.atom37_mask.astype(jnp.float32)


class HotspotSeqFeat(eqx.Module):
    dim: int = 37

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.hotspot is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, 37))
        return batch.hotspot.astype(jnp.float32) * batch.atom37_mask.astype(jnp.float32)


class NucleicSecondaryStructureStateSeqFeat(eqx.Module):
    dim: int = 2

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.na_ss_state is None or batch.na_ss_state_mask is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        return jnp.concatenate(
            [batch.na_ss_state[..., None], batch.na_ss_state_mask[..., None]], axis=-1
        ).astype(jnp.float32)


class ConditionalResidueTypeSeqFeat(AbstractFromTorch):
    """Frozen identity embedding = one-hot lookup (weight requires_grad=False in
    torch; converted weights keep the identity)."""

    embedding: Embedding
    num_classes: int = 0

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.cond_seq is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.num_classes))
        rtype = batch.cond_seq.astype(jnp.int32)
        seq_cond_mask = rtype != _SEQ_MASK_TOKEN
        rtype = rtype * seq_cond_mask  # unconditioned tokens -> row 0
        embedded = self.embedding(rtype)
        return embedded * seq_cond_mask[..., None]


class AtomChargeSeqFeat(AbstractFromTorch):
    embedding: Embedding
    dim: int = 16

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.charge is None or batch.token_type is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        # Atomized tokens store their per-atom scalars in atom37 slot 1.
        charge_slot1 = batch.charge[:, :, 1].astype(jnp.int32)
        emb = embed_charge(self.embedding, charge_slot1)
        is_atomized = (batch.token_type == _TOKEN_TYPE_ATOMIZED)[..., None]
        return emb * is_atomized


class AtomicElementSeqFeat(AbstractFromTorch):
    embedding: Embedding
    dim: int = 16
    max_atomic_num: int = 128

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.element is None or batch.token_type is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, self.dim))
        element_slot1 = jnp.clip(batch.element[:, :, 1].astype(jnp.int32), 0, self.max_atomic_num - 1)
        emb = self.embedding(element_slot1)
        is_atomized = (batch.token_type == _TOKEN_TYPE_ATOMIZED)[..., None]
        return emb * is_atomized


class Atom37ElementChargeEmbeddingSeqFeat(AbstractFromTorch):
    element_embedding: Embedding
    charge_embedding: Embedding
    dim: int = 37 * 24

    def __call__(self, batch: FlowBatch, *, key=None):
        mask = batch.atom37_mask
        elements = jnp.where(mask, batch.ground_truth_element, 0).astype(jnp.int32)
        charges = jnp.where(mask, batch.charge, 0).astype(jnp.int32)
        encoded = jnp.concatenate(
            (self.element_embedding(elements), embed_charge(self.charge_embedding, charges)), axis=-1
        )
        return (encoded * mask[..., None]).reshape(*encoded.shape[:-2], -1)


class IncidentBondTypeCountsSeqFeat(eqx.Module):
    dim: int = 37 * 10

    def __call__(self, batch: FlowBatch, *, key=None):
        counts = batch.incident_bond_type_counts.astype(jnp.float32)
        return (counts * batch.atom37_mask[..., None]).reshape(*counts.shape[:-2], -1)


class FoldEmbeddingSeqFeat(AbstractFromTorch):
    embedding_C: Embedding
    embedding_A: Embedding
    embedding_T: Embedding
    dim: int = 96

    def __call__(self, batch: FlowBatch, *, key=None):
        # torch: cath_codes + 1 because padding/null are -1.
        cath = batch.cath_codes.astype(jnp.int32) + 1
        return jnp.concatenate(
            [self.embedding_C(cath[:, :, 0]), self.embedding_A(cath[:, :, 1]), self.embedding_T(cath[:, :, 2])],
            axis=-1,
        )


# --- pair features -------------------------------------------------------------


class TokenBondPathLengthPairFeat(eqx.Module):
    dim: int = 0  # seq_sep_dim, set at construction

    def __call__(self, batch: FlowBatch, *, key=None):
        path = batch.token_bond_path_length.astype(jnp.int32)
        clipped = jnp.clip(path, max=self.dim - 1)  # overflow bin
        return jax.nn.one_hot(clipped, self.dim)


class XtBBCAPairwiseDistancesPairFeat(eqx.Module):
    dim: int = 30
    min_dist: float = 0.1
    max_dist: float = 3.0

    def __call__(self, batch: FlowBatch, *, key=None):
        return bin_pairwise_distances(batch.x_t_bb_ca, self.min_dist, self.max_dist, self.dim)


class CaCoorsNanometersPairwiseDistancesPairFeat(eqx.Module):
    dim: int = 30
    min_dist: float = 0.1
    max_dist: float = 3.0

    def __call__(self, batch: FlowBatch, *, key=None):
        ca = batch.ca_coors_nm if batch.ca_coors_nm is not None else batch.coords_nm[:, :, 1, :]
        return bin_pairwise_distances(ca, self.min_dist, self.max_dist, self.dim)


class BackbonePairDistancesNanometerPairFeat(eqx.Module):
    dim: int = 84

    def __call__(self, batch: FlowBatch, *, key=None):
        coords = batch.coords_nm
        atom_mask = batch.atom37_mask
        mask = atom_mask[:, :, 1]  # CA present
        pair_mask = mask[:, None, :] * mask[:, :, None]
        has_cb = atom_mask[:, :, 3]

        N, CA, C, CB = (coords[:, :, i, :] for i in range(4))
        CA_i = CA[:, :, None, :]
        ca_n, ca_ca, ca_c, ca_cb = (
            jnp.linalg.norm(CA_i - v[:, None, :, :], axis=-1) for v in (N, CA, C, CB)
        )
        # Glycine etc. have no CB: zero those distances (torch broadcasts has_cb
        # as [b, 1, n] — the j-in-axis mask).
        ca_cb = ca_cb * has_cb[:, None, :]
        ca_n, ca_ca, ca_c, ca_cb = (v * pair_mask for v in (ca_n, ca_ca, ca_c, ca_cb))
        bin_limits = jnp.linspace(0.1, 2.0, 20)
        feats = [bin_and_one_hot(v, bin_limits) for v in (ca_n, ca_ca, ca_c, ca_cb)]
        feat = jnp.concatenate(feats, axis=-1)
        return feat * pair_mask[..., None]


class MoleculeIdxPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        m = batch.molecule_iid.astype(jnp.int32)
        return (m[:, :, None] == m[:, None, :])[..., None].astype(jnp.float32)


class MoleculeEntityPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        e = batch.molecule_entity_id.astype(jnp.int32)
        return (e[:, :, None] == e[:, None, :])[..., None].astype(jnp.float32)


class IndexGroupPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        g = batch.idx_group.astype(jnp.int32)
        return (g[:, :, None] == g[:, None, :])[..., None].astype(jnp.float32)


class BondMaskPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.bond_mask is None:
            b, n, _ = batch.padding_mask.shape[0], batch.padding_mask.shape[1], batch.padding_mask.shape[1]
            return jnp.zeros((b, n, n, 1))
        return batch.bond_mask[..., None].astype(jnp.float32)


class BondOrderPairFeat(eqx.Module):
    dim: int = 5
    num_bond_orders: int = 5

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.bond_order is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, n, self.num_bond_orders))
        return jax.nn.one_hot(batch.bond_order.astype(jnp.int32), self.num_bond_orders)


class IsAromaticPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.is_aromatic is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, n, 1))
        return batch.is_aromatic[..., None].astype(jnp.float32)


class CanonicalPolymerOrderingPairFeat(eqx.Module):
    dim: int = 1

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.canonical_polymer_ordering is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, n, 1))
        return batch.canonical_polymer_ordering[..., None].astype(jnp.float32)


class NucleicSecondaryStructurePairFeat(eqx.Module):
    dim: int = 2

    def __call__(self, batch: FlowBatch, *, key=None):
        if batch.na_ss_pair is None or batch.na_ss_pair_mask is None:
            b, n = batch.padding_mask.shape
            return jnp.zeros((b, n, n, self.dim))
        return jnp.concatenate(
            [batch.na_ss_pair[..., None], batch.na_ss_pair_mask[..., None]], axis=-1
        ).astype(jnp.float32)


# --- factory -------------------------------------------------------------------


class ZeroFeat(eqx.Module):
    dim: int = 0
    mode: str = "seq"

    def __call__(self, batch: FlowBatch, *, key=None):
        b, n = batch.padding_mask.shape
        assert self.mode == "seq", "pair-mode ret_zero unused by torch (its forward returns None)"
        return jnp.zeros((b, n, self.dim))


class FeatureFactory(AbstractFromTorch):
    """Project the selected features to ``dim_feats_out`` channels.

    Torch children convert 1:1: ``feat_creators`` (ModuleList -> list),
    ``ln_out`` (norm or Identity), ``linear_out``, optional ``zero_creator``.
    """

    # Children are absent when ret_zero=True (torch returns early in __init__),
    # so all three are optional.
    feat_creators: list | None = None
    ln_out: eqx.Module | None = None
    linear_out: Linear | None = None
    mode: str = "seq"
    compact: bool = False
    ret_zero: bool = False
    zero_creator: ZeroFeat | None = None

    def _apply_padding_mask(self, feature, mask):
        if feature.ndim == 2:
            # Compact per-example features: zero if the example has no tokens.
            return feature * mask.any(axis=-1, keepdims=True)
        if self.mode == "seq":
            return feature * mask[..., None]
        pair_mask = mask[:, None, :] * mask[:, :, None]
        return feature * pair_mask[..., None]

    def __call__(self, batch: FlowBatch, *, key=None):
        if self.ret_zero:
            return self.zero_creator(batch)
        feature_tensors = [fc(batch, key=key) for fc in self.feat_creators]

        if not (self.compact and all(f.ndim == 2 for f in feature_tensors)):
            n = batch.padding_mask.shape[-1]
            spatial = (n,) if self.mode == "seq" else (n, n)
            feature_tensors = [
                jnp.broadcast_to(f.reshape(f.shape[0], *((1,) * len(spatial)), -1), (f.shape[0], *spatial, f.shape[-1]))
                if f.ndim == 2
                else f
                for f in feature_tensors
            ]

        features = jnp.concatenate(feature_tensors, axis=-1)
        features = self._apply_padding_mask(features, batch.padding_mask)
        projected = self.linear_out(features)
        out = self.ln_out(projected)
        return self._apply_padding_mask(out, batch.padding_mask)


def register():
    import dataclasses

    import rfproteina.nn.feature_factory as ff

    def _convert_stateless(jax_cls, m):
        """Stateless features: pull static scalars (dim, flags) from torch attrs."""
        kwargs = {}
        for f in dataclasses.fields(jax_cls):
            if hasattr(m, f.name):
                v = getattr(m, f.name)
                if v is None or isinstance(v, (int, float, bool, str)):
                    kwargs[f.name] = v
        return jax_cls(**kwargs)

    T = {
        ff.TimeEmbeddingBatchFeat: TimeEmbeddingBatchFeat,
        ff.RecycleRemainingBatchFeat: RecycleRemainingBatchFeat,
        ff.XtBBCASeqFeat: XtBBCASeqFeat,
        ff.XtLocalLatentsSeqFeat: XtLocalLatentsSeqFeat,
        ff.CaCoorsNanometersSeqFeat: CaCoorsNanometersSeqFeat,
        ff.LatentVariableSeqFeat: LatentVariableSeqFeat,
        ff.ChainIdxSeqFeat: ChainIdxSeqFeat,
        ff.TokenTypeSeqFeat: TokenTypeSeqFeat,
        ff.SequenceProbabilitiesSeqFeat: SequenceProbabilitiesSeqFeat,
        ff.Atom37NanometersCoorsSeqFeat: Atom37NanometersCoorsSeqFeat,
        ff.ConditionalAtom37NanometersCoorsSeqFeat: ConditionalAtom37NanometersCoorsSeqFeat,
        ff.ConditionalAtom37NanometersInternalCoorsSeqFeat: ConditionalAtom37NanometersInternalCoorsSeqFeat,
        ff.ChiralitySeqFeat: ChiralitySeqFeat,
        ff.CoordGroupSeqFeat: CoordGroupSeqFeat,
        ff.RasaBinnedSeqFeat: RasaBinnedSeqFeat,
        ff.ChainCentroidXyzSeqFeat: ChainCentroidXyzSeqFeat,
        ff.SecondaryStructureSeqFeat: SecondaryStructureSeqFeat,
        ff.ModernizedHydrogenBondSeqFeat: ModernizedHydrogenBondSeqFeat,
        ff.PiStackingSeqFeat: PiStackingSeqFeat,
        ff.PiCationSeqFeat: PiCationSeqFeat,
        ff.SaltBridgeSeqFeat: SaltBridgeSeqFeat,
        ff.HydrophobicSeqFeat: HydrophobicSeqFeat,
        ff.MetalCoordinationSeqFeat: MetalCoordinationSeqFeat,
        ff.HotspotSeqFeat: HotspotSeqFeat,
        ff.NucleicSecondaryStructureStateSeqFeat: NucleicSecondaryStructureStateSeqFeat,
        ff.ConditionalResidueTypeSeqFeat: ConditionalResidueTypeSeqFeat,
        ff.AtomChargeSeqFeat: AtomChargeSeqFeat,
        ff.AtomicElementSeqFeat: AtomicElementSeqFeat,
        ff.Atom37ElementChargeEmbeddingSeqFeat: Atom37ElementChargeEmbeddingSeqFeat,
        ff.IncidentBondTypeCountsSeqFeat: IncidentBondTypeCountsSeqFeat,
        ff.FoldEmbeddingSeqFeat: FoldEmbeddingSeqFeat,
        ff.TokenBondPathLengthPairFeat: TokenBondPathLengthPairFeat,
        ff.XtBBCAPairwiseDistancesPairFeat: XtBBCAPairwiseDistancesPairFeat,
        ff.CaCoorsNanometersPairwiseDistancesPairFeat: CaCoorsNanometersPairwiseDistancesPairFeat,
        ff.BackbonePairDistancesNanometerPairFeat: BackbonePairDistancesNanometerPairFeat,
        ff.MoleculeIdxPairFeat: MoleculeIdxPairFeat,
        ff.MoleculeEntityPairFeat: MoleculeEntityPairFeat,
        ff.IndexGroupPairFeat: IndexGroupPairFeat,
        ff.BondMaskPairFeat: BondMaskPairFeat,
        ff.BondOrderPairFeat: BondOrderPairFeat,
        ff.IsAromaticPairFeat: IsAromaticPairFeat,
        ff.CanonicalPolymerOrderingPairFeat: CanonicalPolymerOrderingPairFeat,
        ff.NucleicSecondaryStructurePairFeat: NucleicSecondaryStructurePairFeat,
        ff.ZeroFeat: ZeroFeat,
        ff.FeatureFactory: FeatureFactory,
    }
    for torch_cls, jax_cls in T.items():
        if hasattr(jax_cls, "from_torch") and issubclass(jax_cls, AbstractFromTorch):
            from_torch.register(torch_cls, jax_cls.from_torch)
        else:
            from_torch.register(torch_cls, lambda m, _j=jax_cls: _convert_stateless(_j, m))
