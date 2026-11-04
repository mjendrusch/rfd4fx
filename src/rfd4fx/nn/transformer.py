"""The denoiser trunk: attention+transition blocks, pair-representation
update/builder, masks, and the LocalLatentsTransformer with a scan-based
trunk and recycle loop."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from ..types import DecoderOutput, EncoderOutput, FlowBatch, TrunkOutput
from ..backend import AbstractFromTorch, from_torch
from .composites import MultiHeadBiasedAttentionADALN_MM, PairTransition, Transition, TransitionADALN
from .composites import TriangleMultiplication
from .features import FeatureFactory
from .layers import LayerNorm, Linear, RMSNorm, Sequential


# --- masks (model_masks.py) ----------------------------------------------------


def compute_flow_emit_masks(batch: FlowBatch) -> dict[str, Bool[Array, "b n"]]:
    bb_ca = batch.padding_mask & ~batch.is_in_context_conditioning
    return {"bb_ca": bb_ca, "local_latents": bb_ca & batch.supports_local_latent}


# --- trunk layers ---------------------------------------------------------------


class MultiheadAttnAndTransition(AbstractFromTorch):
    """One trunk layer: masked x -> (mha, transition) sequentially or in parallel,
    with per-branch residual flags (nn/modules/attn_n_transition.py)."""

    mhba: MultiHeadBiasedAttentionADALN_MM
    transition: TransitionADALN
    parallel: bool = False
    use_attn_pair_bias: bool = True
    residual_mha: bool = True
    residual_transition: bool = True

    def _apply_mha(self, x, pair_rep, cond, mask):
        x_attn = self.mhba(x, pair_rep if self.use_attn_pair_bias else None, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def __call__(self, x, pair_rep, cond, mask):
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(x, cond, mask)
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask)
        return x * mask[..., None]


class PairReprUpdate(AbstractFromTorch):
    """Single-to-pair write (sum or hadamard) + optional triangle updates +
    pair transition, all residual and masked (nn/modules/pair_update.py)."""

    layer_norm_in: LayerNorm
    linear_x: Linear
    transition_out: PairTransition
    tri_mult_out: TriangleMultiplication | None = None
    tri_mult_in: TriangleMultiplication | None = None
    use_tri_mult: bool = False
    pair_write: str = "sum"

    def __call__(self, x, pair_rep, mask):
        pair_mask = mask[:, None, :] * mask[:, :, None]
        x = x * mask[..., None]

        # torch chunk(2, dim=-1): pi is written at j-index? No — pi indexes the
        # first spatial axis (i) via pi[:, None, :, :] and pj the second (j).
        pi, pj = jnp.split(self.linear_x(self.layer_norm_in(x)), 2, axis=-1)
        if self.pair_write == "sum":
            pair_rep = pair_rep + pi[:, None, :, :] + pj[:, :, None, :]
        else:
            pair_rep = pair_rep + pi[:, None, :, :] * pj[:, :, None, :]
        pair_rep = pair_rep * pair_mask[..., None]

        # torch checkpoint(use_reentrant=False) -> plain calls here (inference);
        # a standalone block would take eqx.filter_checkpoint.
        if self.use_tri_mult:
            pair_rep = pair_rep + self.tri_mult_out(pair_rep, pair_mask)
            pair_rep = pair_rep * pair_mask[..., None]
            pair_rep = pair_rep + self.tri_mult_in(pair_rep, pair_mask)
            pair_rep = pair_rep * pair_mask[..., None]
        pair_rep = pair_rep + self.transition_out(pair_rep, pair_mask)
        pair_rep = pair_rep * pair_mask[..., None]
        return pair_rep


class PairReprBuilder(AbstractFromTorch):
    """Pair features + optional broadcast AdaLN from compact per-example cond."""

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory | None = None
    adaln: eqx.Module | None = None

    def __call__(self, batch: FlowBatch, *, key=None):
        mask = batch.padding_mask
        pair_mask = mask[:, :, None] * mask[:, None, :]
        rep = self.init_repr_factory(batch, key=key)
        if self.cond_factory is not None:
            cond = self.cond_factory(batch, key=key)  # [b, dim_cond_pair]
            rep = self.adaln(rep, cond, pair_mask)
        return rep


# --- trunk scan blocks ----------------------------------------------------------


class _TrunkBlock(eqx.Module):
    """A group of consecutive attention layers followed by an optional pair
    update. Attention layers are scan-stacked (equal structure); the pair update
    runs once after the group (nn/trunk placement: after every n-th layer)."""

    attn_params: MultiheadAttnAndTransition  # array leaves stacked along axis 0
    attn_static: MultiheadAttnAndTransition
    pair_update: PairReprUpdate | None = None

    def __call__(self, s, z, c, mask):
        def body(state, params):
            layer = eqx.combine(self.attn_static, params)
            return layer(state, z, c, mask), None

        # scan over the group's attention layers (same weights reused would be
        # fori_loop; here each layer has its own params -> scan).
        s, _ = jax.lax.scan(body, s, self.attn_params)
        if self.pair_update is not None:
            z = self.pair_update(s, z, mask)
        return s, z


def _group_layers(nlayers: int, pair_layers: list | None):
    """Split layer indices into groups closed by a pair update (torch
    _apply_trunk order: attn i, then pair update i if present)."""
    groups, cur = [], []
    for i in range(nlayers):
        cur.append(i)
        if pair_layers and i < nlayers - 1 and pair_layers[i] is not None:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def _build_trunk_blocks(attn_layers: list, pair_layers: list | None) -> list[_TrunkBlock]:
    blocks = []
    for idxs in _group_layers(len(attn_layers), pair_layers):
        mods = [attn_layers[i] for i in idxs]
        _, static = eqx.partition(mods[0], eqx.is_inexact_array)
        stacked = jax.tree.map(
            lambda *v: jnp.stack(v, 0), *[eqx.filter(m, eqx.is_inexact_array) for m in mods]
        )
        pair = None
        if pair_layers is not None:
            last = idxs[-1]
            if last < len(pair_layers) and pair_layers[last] is not None:
                pair = pair_layers[last]
        blocks.append(_TrunkBlock(attn_params=stacked, attn_static=static, pair_update=pair))
    return blocks


# --- the denoiser ---------------------------------------------------------------


class LocalLatentsTransformer(AbstractFromTorch):
    """RFD4 denoiser trunk (nn/local_latents_transformer.py), inference path.

    Custom ``from_torch``: ``transformer_layers``/``pair_update_layers`` are
    restructured into scan blocks instead of being stored flat (torch.compile
    machinery, dynamo caches and checkpointing flags are torch-only and dropped).
    """

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory
    transition_c_1: Transition
    transition_c_2: Transition
    pair_repr_builder: PairReprBuilder
    local_latents_linear: Sequential
    ca_linear: Sequential
    insertion_q_proj: Linear
    insertion_k_proj: Linear
    trunk_blocks: list  # built by from_torch
    nlayers: int = 0
    token_dim: int = 0
    pair_repr_dim: int = 0
    update_pair_repr: bool = False
    update_pair_repr_every_n: int = 1
    use_tri_mult: bool = False
    condition_on_recycle_remaining: bool = False
    recycle_enabled: bool = False
    recycle_condition_probability: float = 0.5
    sequence_readout_enabled: bool = False
    sequence_readout_detach: bool = True
    insertion_head_scale: float = 0.0
    s_cycle_norm: eqx.Module | None = None
    s_cycle_proj: Linear | None = None
    z_cycle_norm: eqx.Module | None = None
    z_cycle_proj: Linear | None = None
    sequence_readout: Sequential | None = None

    @classmethod
    def from_torch(cls, model):
        import dataclasses

        kwargs = {}
        for name, child in model.named_children():
            if name in ("transformer_layers", "pair_update_layers"):
                continue
            kwargs[name] = from_torch(child)
        for name, param in model.named_parameters(recurse=False):
            kwargs[name] = from_torch(param)
        for f in dataclasses.fields(cls):
            if f.name in kwargs or f.name == "trunk_blocks":
                continue
            if hasattr(model, f.name):
                attr = getattr(model, f.name)
                if not callable(attr) or isinstance(attr, (int, float, bool, str)):
                    kwargs[f.name] = from_torch(attr)
                    continue
            if f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[f.name] = f.default_factory()
            else:
                raise ValueError(f"{cls.__name__}: missing field {f.name!r}")

        attn = [from_torch(layer) for layer in model.transformer_layers]
        pair = None
        if model.update_pair_repr:
            pair = [from_torch(layer) if layer is not None else None for layer in model.pair_update_layers]
        kwargs["trunk_blocks"] = _build_trunk_blocks(attn, pair)
        # torch attr name has a leading underscore; rename to the field.
        kwargs["insertion_head_scale"] = float(model._insertion_head_scale)
        return cls(**kwargs)

    def _trunk_pass(self, s, z, c, seq_mask):
        for block in self.trunk_blocks:
            s, z = block(s, z, c, seq_mask)
        return s, z

    def _run_trunk(self, batch, s_init, z_init, c, seq_mask, *, n_recycle=1, condition_remaining=True):
        if not self.recycle_enabled:
            return self._trunk_pass(s_init, z_init, c, seq_mask)

        # Dynamic-depth recycle loop: `n_recycle` may be a traced scalar (the
        # sampler passes per-step depths as traced values), so ONE compiled
        # executable serves every recycle depth. The per-cycle conditioning is
        # recomputed inside the loop with a traced recycle_remaining (the same
        # values the unrolled version produced: depth - cycle - 1); cycle 0
        # keeps the first-pass conditioning (recycle_remaining=None).
        n_rec = jnp.maximum(jnp.asarray(n_recycle), 1)

        def body(cycle, carry):
            s, z = carry
            z_in = z_init + self.z_cycle_proj(self.z_cycle_norm(z))
            s_in = s_init + self.s_cycle_proj(self.s_cycle_norm(s))
            if condition_remaining and self.cond_factory is not None:
                c_cyc = self.cond_factory(
                    batch.with_recycle_remaining(n_rec - cycle - 1))
                c_cyc = self.transition_c_2(self.transition_c_1(c_cyc, seq_mask), seq_mask)
                c_i = jnp.where(cycle == 0, c, c_cyc)
            else:
                c_i = c
            s, z = self._trunk_pass(s_in, z_in, c_i, seq_mask)
            # inference-only port: the unrolled version's inter-cycle
            # stop_gradient is a no-op numerically
            return s, z

        s, z = jax.lax.fori_loop(0, n_rec, body, (jnp.zeros_like(s_init),
                                                  jnp.zeros_like(z_init)))
        return s, z

    def __call__(
        self,
        batch: FlowBatch,
        *,
        n_recycle: int = 1,
        condition_on_recycle_remaining: bool = True,
        key=None,
    ) -> TrunkOutput:
        seq_mask = batch.padding_mask

        # Conditioning (first pass: recycle_remaining=None).
        c = self.cond_factory(batch.with_recycle_remaining(None), key=key)
        c = self.transition_c_2(self.transition_c_1(c, seq_mask), seq_mask)

        s_init = self.init_repr_factory(batch, key=key) * seq_mask[..., None]
        z_init = self.pair_repr_builder(batch, key=key)

        # static flag only — the depth-1 case is handled inside the loop (the
        # single cycle keeps the first-pass conditioning)
        cond_remaining = (self.condition_on_recycle_remaining
                          and condition_on_recycle_remaining)
        s, _z = self._run_trunk(
            batch, s_init, z_init, c, seq_mask, n_recycle=n_recycle, condition_remaining=cond_remaining
        )

        emit = compute_flow_emit_masks(batch)
        local_latents_out = self.local_latents_linear(s) * emit["local_latents"][..., None]
        ca_nm_out = self.ca_linear(s) * emit["bb_ca"][..., None]

        seq_logits = None
        if self.sequence_readout_enabled:
            seqs = jax.lax.stop_gradient(s) if self.sequence_readout_detach else s
            seq_logits = self.sequence_readout(seqs) * seq_mask[..., None]

        # Insertion head operates on detached states (torch s.detach()).
        s_ins = jax.lax.stop_gradient(s)
        q = self.insertion_q_proj(s_ins)
        k = self.insertion_k_proj(s_ins)
        # torch.bmm(q, k.transpose(1, 2)) * d^-0.5
        insertion_logits = jnp.einsum("bnd,bmd->bnm", q, k) * self.insertion_head_scale

        return TrunkOutput(
            bb_ca=ca_nm_out, local_latents=local_latents_out, seq_logits=seq_logits, insertion_logits=insertion_logits
        )


def register():
    from rfproteina.nn.local_latents_transformer import LocalLatentsTransformer as TorchLLT
    from rfproteina.nn.modules.attn_n_transition import MultiheadAttnAndTransition as TorchMAT
    from rfproteina.nn.modules.pair_rep_initial import PairReprBuilder as TorchPRB
    from rfproteina.nn.modules.pair_update import PairReprUpdate as TorchPRU
    from rfproteina.partial_autoencoder.decoder import DecoderTransformer as TorchDecoder
    from rfproteina.partial_autoencoder.encoder import EncoderTransformer as TorchEncoder

    from_torch.register(TorchMAT, MultiheadAttnAndTransition.from_torch)
    from_torch.register(TorchPRU, PairReprUpdate.from_torch)
    from_torch.register(TorchPRB, PairReprBuilder.from_torch)
    from_torch.register(TorchLLT, LocalLatentsTransformer.from_torch)
    from_torch.register(TorchEncoder, EncoderTransformer.from_torch)
    from_torch.register(TorchDecoder, DecoderTransformer.from_torch)


# --- autoencoder encoder / decoder ---------------------------------------------


def _ae_masks(batch: FlowBatch):
    """model_masks.compute_ae_masks: attention = P & ~C, emit = attention & L."""
    attn_mask = batch.padding_mask & ~batch.is_in_context_conditioning
    return attn_mask, attn_mask & batch.supports_local_latent


class _AETransformerMixin:
    """Shared conversion helper for the AE encoder/decoder (same trunk machinery
    as the denoiser; differs in factories, masks and heads)."""

    @staticmethod
    def _convert_ae(cls, model):
        import dataclasses

        kwargs = {}
        for name, child in model.named_children():
            if name in ("transformer_layers", "pair_update_layers"):
                continue
            kwargs[name] = from_torch(child)
        for f in dataclasses.fields(cls):
            if f.name in kwargs or f.name == "trunk_blocks":
                continue
            if hasattr(model, f.name):
                attr = getattr(model, f.name)
                if not callable(attr) or isinstance(attr, (int, float, bool, str)):
                    kwargs[f.name] = from_torch(attr)
                    continue
            if f.default is not dataclasses.MISSING:
                kwargs[f.name] = f.default
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                kwargs[f.name] = f.default_factory()
            else:
                raise ValueError(f"{cls.__name__}: missing field {f.name!r}")
        attn = [from_torch(layer) for layer in model.transformer_layers]
        pair = None
        if getattr(model, "update_pair_repr", False):
            pair = [from_torch(layer) if layer is not None else None for layer in model.pair_update_layers]
        kwargs["trunk_blocks"] = _build_trunk_blocks(attn, pair)
        return cls(**kwargs)


class EncoderTransformer(_AETransformerMixin, AbstractFromTorch):
    """AE encoder (partial_autoencoder/encoder.py): trunk over P & ~C tokens,
    VAE head emitting mean/log_scale and a sampled z (explicit PRNG key)."""

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory
    transition_c_1: Transition
    transition_c_2: Transition
    latent_decoder_mean_n_log_scale: Sequential
    trunk_blocks: list
    ln_z: eqx.Module | None = None
    pair_rep_factory: FeatureFactory | None = None
    nlayers: int = 0
    token_dim: int = 0
    pair_repr_dim: int = 0
    update_pair_repr: bool = False
    use_attn_pair_bias: bool = True
    normalize_latent: bool = False

    from_torch = classmethod(_AETransformerMixin._convert_ae)

    def __call__(self, batch: FlowBatch, *, key=None) -> EncoderOutput:
        attn_mask, emit_mask = _ae_masks(batch)

        c = self.cond_factory(batch, key=key)
        c = self.transition_c_2(self.transition_c_1(c, attn_mask), attn_mask)

        seqs = self.init_repr_factory(batch, key=key) * attn_mask[..., None]
        pair_rep = self.pair_rep_factory(batch, key=key) if self.pair_rep_factory is not None else None

        for block in self.trunk_blocks:
            seqs, pair_rep = block(seqs, pair_rep, c, attn_mask)

        flat_out = self.latent_decoder_mean_n_log_scale(seqs) * emit_mask[..., None]
        mean, log_scale = jnp.split(flat_out, 2, axis=-1)
        if key is None:
            raise ValueError("EncoderTransformer requires an RNG key for VAE sampling")
        z = mean + jax.random.normal(key, log_scale.shape) * jnp.exp(log_scale)

        z_pre_ln = z * emit_mask[..., None]
        z_out = (self.ln_z(z) if self.ln_z is not None else z) * emit_mask[..., None]
        return EncoderOutput(mean=mean, log_scale=log_scale, z_latent=z_out, z_latent_pre_ln=z_pre_ln, emit_mask=emit_mask)


class DecoderTransformer(_AETransformerMixin, AbstractFromTorch):
    """AE decoder (partial_autoencoder/decoder.py): z_latent + CA frames ->
    sequence logits (token-type-restricted), atom37 coordinates, atom presence.

    ``allowed_residue_types`` / ``residue_atom_mask`` are non-persistent derived
    buffers: converted from the torch module at conversion time and serialized
    with the weights (they are static vocabulary tables)."""

    init_repr_factory: FeatureFactory
    cond_factory: FeatureFactory
    transition_c_1: Transition
    transition_c_2: Transition
    logit_linear: Sequential
    struct_linear: Sequential
    atom_presence_linear: Sequential
    trunk_blocks: list
    allowed_residue_types: Bool[Array, "7 classes"] | None = None
    residue_atom_mask: Bool[Array, "classes 37"] | None = None
    pair_rep_factory: FeatureFactory | None = None
    n_residue_types: int = 0
    nlayers: int = 0
    token_dim: int = 0
    pair_repr_dim: int = 0
    update_pair_repr: bool = False
    use_attn_pair_bias: bool = True
    abs_coors: bool = True

    from_torch = classmethod(_AETransformerMixin._convert_ae)

    def __call__(self, batch: FlowBatch, *, key=None) -> DecoderOutput:
        ca_coors_nm = batch.ca_coors_nm
        attn_mask, emit_mask = _ae_masks(batch)

        c = self.cond_factory(batch, key=key)
        c = self.transition_c_2(self.transition_c_1(c, attn_mask), attn_mask)

        seqs = self.init_repr_factory(batch, key=key) * attn_mask[..., None]
        pair_rep = self.pair_rep_factory(batch, key=key) if self.pair_rep_factory is not None else None

        for block in self.trunk_blocks:
            seqs, pair_rep = block(seqs, pair_rep, c, attn_mask)

        logits_out = self.logit_linear(seqs) * emit_mask[..., None]

        if batch.token_type is not None:
            # Restrict residue classes to the token's type (UNKNOWN row = any).
            allowed_per_token = self.allowed_residue_types[batch.token_type] | ~emit_mask[..., None]
            logits_out = jnp.where(allowed_per_token, logits_out, jnp.finfo(logits_out.dtype).min)

        coors_flat_nm = self.struct_linear(seqs) * emit_mask[..., None]
        coors_a37_nm = coors_flat_nm.reshape(*coors_flat_nm.shape[:-1], 37, 3)

        if self.abs_coors:
            # torch: coors[..., 1, :] * 0.0 + ca (in-place on a fresh tensor)
            coors_a37_nm = coors_a37_nm.at[..., 1, :].set(ca_coors_nm)
        else:
            coors_a37_nm = coors_a37_nm.at[..., 1, :].set(0.0)
            coors_a37_nm = coors_a37_nm + ca_coors_nm[:, :, None, :]
        coors_a37_nm = coors_a37_nm * emit_mask[..., None, None]

        aatype_max = jnp.argmax(logits_out, axis=-1) * emit_mask

        atom_mask = self.residue_atom_mask[aatype_max] & emit_mask[..., None]
        atom_presence_logits = self.atom_presence_linear(seqs)
        predicted_mask = atom_presence_logits > 0
        # torch sets slot 1 True in place: every emitted token keeps its ref atom.
        predicted_mask = predicted_mask.at[..., 1].set(True)
        atom_mask = atom_mask & predicted_mask

        return DecoderOutput(
            coors_nm=coors_a37_nm,
            seq_logits=logits_out,
            emit_mask=emit_mask,
            aatype_max=aatype_max,
            atom_mask=atom_mask,
            atom_presence_logits=atom_presence_logits,
        )
