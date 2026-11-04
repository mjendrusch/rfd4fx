"""Typed batch containers (equinox modules) shared by denoiser and autoencoder.

Absent optional inputs are ``None`` (static under jit); shapes mirror the
native torch pipeline field-for-field."""

from __future__ import annotations

import equinox as eqx
from jaxtyping import Array, Bool, Float, Int


class FlowBatch(eqx.Module):
    """All fields optional unless noted. Shapes for batch b, tokens n."""

    # Core (always present at inference)
    padding_mask: Bool[Array, "b n"]
    is_in_context_conditioning: Bool[Array, "b n"] | None = None
    supports_local_latent: Bool[Array, "b n"] | None = None

    # Noisy state / times (denoiser)
    x_t_bb_ca: Float[Array, "b n 3"] | None = None
    x_t_local_latents: Float[Array, "b n lat"] | None = None
    t_bb_ca: Float[Array, "b"] | None = None
    t_local_latents: Float[Array, "b"] | None = None

    # Recycle conditioning (None on the first trunk pass; a traced scalar array
    # (0-d int) afterwards so the recycle loop can update it without retrace)
    recycle_remaining: "int | Array | None" = None

    # Traced recycle depth for the current step (RFD4.step reads it; the
    # sampler writes it per step — one compile per token shape)
    n_recycle: "int | Array | None" = None

    # Per-step predictions written by RFD4.step (padded width; emit-masked
    # where noted). v_pred/s_pred are per-modality dicts.
    aa_pred: Int[Array, "b n"] | None = None
    x_pred: Float[Array, "b n 3"] | None = None
    z_pred: Float[Array, "b n lat"] | None = None
    v_pred: "dict[str, Array] | None" = None
    s_pred: "dict[str, Array] | None" = None

    # Token-level annotation (denoiser + AE)
    token_type: Int[Array, "b n"] | None = None
    seq_probabilities: Float[Array, "b n 28"] | None = None  # NaN = unspecified
    atom37_mask: Bool[Array, "b n 37"] | None = None
    coords_nm: Float[Array, "b n 37 3"] | None = None
    charge: Int[Array, "b n 37"] | None = None
    element: Int[Array, "b n 37"] | None = None
    ground_truth_element: Int[Array, "b n 37"] | None = None
    incident_bond_type_counts: Float[Array, "b n 37 10"] | None = None

    # AE encoder/decoder inputs
    ca_coors_nm: Float[Array, "b n 3"] | None = None
    z_latent: Float[Array, "b n lat"] | None = None

    # Conditioning annotations (None = not requested)
    cond_seq: Int[Array, "b n"] | None = None  # atomworks C_SEQ
    cond_coords: Float[Array, "b n 37 3"] | None = None  # C_CRD (Å, NaN = unconditioned)
    cond_internal_coords: Float[Array, "b n 37 3"] | None = None  # C_INTCRD
    coord_groups: Int[Array, "b n 37"] | None = None  # C_CRDGRP
    rasa: Int[Array, "b n 37"] | None = None  # C_RAS
    secondary_structure: Int[Array, "b n"] | None = None  # C_SEC
    hbond: Int[Array, "b n 37"] | None = None  # C_HB
    pistack: Bool[Array, "b n 37"] | None = None
    pication: Int[Array, "b n 37"] | None = None
    saltbridge: Int[Array, "b n 37"] | None = None
    hydrophobic: Bool[Array, "b n 37"] | None = None
    metal_coord: Int[Array, "b n 37"] | None = None
    hotspot: Bool[Array, "b n 37"] | None = None
    chain_centroid: Float[Array, "b n 3"] | None = None  # C_CENTROID (Å, NaN)
    cath_codes: Int[Array, "b n 3"] | None = None  # C_CATH (values shifted +1 at use)
    idx_group: Int[Array, "b n"] | None = None  # C_IDXGRP
    chirality: Float[Array, "b n d"] | None = None
    chiral_frame_atoms: Int[Array, "b n s k2"] | None = None
    chiral_slot_owner: Int[Array, "b n s"] | None = None

    # Pair-level annotations
    token_bond_path_length: Int[Array, "b n n"] | None = None
    bond_mask: Bool[Array, "b n n"] | None = None
    bond_order: Int[Array, "b n n"] | None = None
    is_aromatic: Bool[Array, "b n n"] | None = None
    canonical_polymer_ordering: Int[Array, "b n n"] | None = None
    molecule_iid: Int[Array, "b n"] | None = None
    molecule_entity_id: Int[Array, "b n"] | None = None
    chain_iid: Float[Array, "b n"] | None = None
    na_ss_pair: Float[Array, "b n n"] | None = None
    na_ss_pair_mask: Float[Array, "b n n"] | None = None
    na_ss_state: Float[Array, "b n"] | None = None
    na_ss_state_mask: Float[Array, "b n"] | None = None

    def with_recycle_remaining(self, value: int | None) -> "FlowBatch":
        """Functional update used per recycle cycle (recycle_remaining changes).
        is_leaf lets us replace the leaf with None (first trunk pass)."""
        return eqx.tree_at(
            lambda b: b.recycle_remaining, self, value, is_leaf=lambda x: x is None
        )


class TrunkOutput(eqx.Module):
    """Denoiser outputs (velocity parameterization per the checkpoint's output_parameterization)."""

    bb_ca: Float[Array, "b n 3"]
    local_latents: Float[Array, "b n lat"]
    seq_logits: Float[Array, "b n classes"] | None = None
    insertion_logits: Float[Array, "b n n"] | None = None


class EncoderOutput(eqx.Module):
    """AE encoder outputs (VAE)."""

    mean: Float[Array, "b n lat"]
    log_scale: Float[Array, "b n lat"]
    z_latent: Float[Array, "b n lat"]
    z_latent_pre_ln: Float[Array, "b n lat"]
    emit_mask: Bool[Array, "b n"]


class DecoderOutput(eqx.Module):
    """AE decoder outputs."""

    coors_nm: Float[Array, "b n 37 3"]
    seq_logits: Float[Array, "b n classes"]
    emit_mask: Bool[Array, "b n"]
    aatype_max: Int[Array, "b n"]
    atom_mask: Bool[Array, "b n 37"]
    atom_presence_logits: Float[Array, "b n 37"]
