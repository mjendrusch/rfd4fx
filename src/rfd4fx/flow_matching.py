"""Product-space flow matching: the sampler, results, and structure export.

Ports the native flow-matching samplers to Equinox. Key objects:
:class:`RFD4Sampler` (ODE sampling with recycle schedules),
:class:`RFD4Result` (per-design output with ``save_pdb``/``save_cif``)."""

from __future__ import annotations

from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float

from .atom24 import Atoms, Atom24, nonpolymer_atoms, to_atom24
from .types import DecoderOutput, FlowBatch

# --- static configs ------------------------------------------------------------


class MatcherConfig(eqx.Module):
    """RDNFlowMatcher configuration (one per modality)."""

    zero_com_noise: bool = False
    guidance_enabled: bool = False
    dim: int = 3
    stochastic_centering_mode: str = "none"  # none | x_0 | x_1 | x_t
    stochastic_centering_scale: float | None = None
    stochastic_centering_scope: str = "without_absolute_coords"  # all | without_absolute_coords | per_group
    max_groups: int = 64  # static bound for per-group noise tables

    def _apply_stochastic_offset(self, x, emit_mask, offset_mask, group_ids, key):
        if offset_mask is None:
            return x
        zero = jnp.zeros((), dtype=x.dtype)
        if group_ids is None:
            noise = jax.random.normal(key, x.shape[:-2] + x.shape[-1:])
            delta = jnp.where(offset_mask[..., None], self.stochastic_centering_scale * noise[..., None, :], zero)
            x = x + delta
        else:
            # Per-group offsets without torch.unique: a static noise table indexed
            # by the (small-int) group id gives every group its own independent
            # normal draw, identical across the group's tokens — same
            # distribution as torch's per-unique-group loop, no dynamic shapes.
            g = jnp.clip(group_ids.astype(jnp.int32), 0, self.max_groups - 1)
            table = jax.random.normal(key, (self.max_groups, x.shape[-1]))
            delta = jnp.where(offset_mask[..., None], self.stochastic_centering_scale * table[g], zero)
            x = x + delta
        if emit_mask is not None:
            x = _mask(x, emit_mask)
        return x

    def _offset_mask(self, emit_mask, coord_cond_mask):
        return (emit_mask & ~coord_cond_mask) if coord_cond_mask is not None else emit_mask

    def add_stochastic_noise(self, x, emit_mask, coord_cond_mask, group_ids, key):
        offset_mask = self._offset_mask(emit_mask, coord_cond_mask)
        if self.stochastic_centering_scope == "all":
            return self._apply_stochastic_offset(x, emit_mask, emit_mask, None, key)
        if self.stochastic_centering_scope == "without_absolute_coords":
            return self._apply_stochastic_offset(x, emit_mask, offset_mask, None, key)
        if self.stochastic_centering_scope == "per_group":
            return self._apply_stochastic_offset(x, emit_mask, offset_mask, group_ids, key)
        raise ValueError(f"Unknown stochastic_centering_scope: {self.stochastic_centering_scope!r}")


class StepParams(eqx.Module):
    """simulation_step_params (per modality)."""

    sampling_mode: str = "vf"  # vf | sc | vf_tsr
    sc_scale_noise: float = 1.0
    t_lim_ode: float = 0.95
    t_lim_ode_below: float = 0.02
    center_every_step: bool = False
    tsr_k: float = 1.0
    tsr_sigma: float = 1.0


def _mask(x, emit_mask):
    return jnp.where(emit_mask[..., None], x, jnp.zeros((), dtype=x.dtype))


def force_zero_com(x: Float[Array, "* n d"], emit_mask=None) -> Float[Array, "* n d"]:
    """Center over the token axis (utils/align.mean_w_mask semantics when masked)."""
    if emit_mask is None:
        return x - jnp.mean(x, axis=-2, keepdims=True)
    mask_e = emit_mask[..., None]
    zero = jnp.zeros((), dtype=x.dtype)
    x_masked = jnp.where(mask_e, x, zero)
    count = mask_e.sum(axis=-2, keepdims=True).astype(x.dtype)
    safe_count = jnp.maximum(count, 1.0)
    mean = jnp.where(count > 0, x_masked.sum(axis=-2, keepdims=True) / safe_count, zero)
    return jnp.where(mask_e, x - mean, zero)


# --- interpolation / conversions ------------------------------------------------


def sample_noise(
    key,
    shape: tuple,
    n: int,
    cfg: MatcherConfig,
    emit_mask=None,
    *,
    coord_cond_mask=None,
    group_ids=None,
    inference: bool = True,
) -> Float[Array, "* n d"]:
    noise = jax.random.normal(key, (*shape, n, cfg.dim))
    if emit_mask is not None:
        noise = _mask(noise, emit_mask)
    if cfg.zero_com_noise:
        noise = force_zero_com(noise, emit_mask)
    # x_0 stochastic centering at inference time (torch: `not training`).
    if cfg.stochastic_centering_mode == "x_0" and inference:
        noise = cfg.add_stochastic_noise(noise, emit_mask, coord_cond_mask, group_ids, key)
    return noise


def interpolate(
    x_0, x_1, t, cfg: MatcherConfig, emit_mask=None, *, coord_cond_mask=None, group_ids=None, key=None
) -> Float[Array, "* n d"]:
    x_0 = _mask(x_0, emit_mask) if emit_mask is not None else x_0
    x_1 = _mask(x_1, emit_mask) if emit_mask is not None else x_1
    if cfg.stochastic_centering_mode == "x_0":
        assert key is not None
        x_0 = cfg.add_stochastic_noise(x_0, emit_mask, coord_cond_mask, group_ids, key)
    elif cfg.stochastic_centering_mode == "x_1":
        assert key is not None
        x_1 = cfg.add_stochastic_noise(x_1, emit_mask, coord_cond_mask, group_ids, key)
    t = t[..., None, None]
    x_t = (1.0 - t) * x_0 + t * x_1
    if cfg.stochastic_centering_mode == "x_t":
        assert key is not None
        x_t = cfg.add_stochastic_noise(x_t, emit_mask, coord_cond_mask, group_ids, key)
    return x_t


def predict_x1_from_v(x_t, v, t):
    """x_1 = x_t + (1 - t) * v."""
    t = t[..., None, None]
    return x_t + (1.0 - t) * v


def vf_to_score(x_t, v, t):
    """score = (t * v - x_t) / (1 - t); boundary-safe at t = 1."""
    t = jnp.asarray(t, dtype=x_t.dtype)[..., None, None]
    at_one = t >= 1.0
    safe_denom = jnp.where(at_one, 1.0, 1.0 - t)
    return jnp.where(at_one, jnp.zeros_like(x_t), (t * v - x_t) / safe_denom)


def score_to_vf(x_t, score, t):
    """v = (x_t + (1 - t) * score) / t; boundary-safe at t = 0."""
    t = jnp.asarray(t, dtype=x_t.dtype)[..., None, None]
    at_zero = t <= 0.0
    safe_t = jnp.where(at_zero, 1.0, t)
    return jnp.where(at_zero, jnp.zeros_like(x_t), (x_t + (1.0 - t) * score) / safe_t)


def compute_tsr_ratio(t, tsr_k: float, tsr_sigma: float):
    t = jnp.asarray(t)
    at_one = t >= 1.0
    safe_one_minus_t = jnp.where(at_one, 1.0, 1.0 - t)
    snr = t**2 / safe_one_minus_t**2
    ratio = (snr * tsr_sigma**2 + 1) / (snr * tsr_sigma**2 / tsr_k + 1)
    return jnp.where(at_one, jnp.asarray(tsr_k, dtype=ratio.dtype), ratio)


def guided_velocity(v, v_ucond, guidance_w: float, guidance_enabled: bool):
    """CFG blend: v_guided = w * v + (1 - w) * v_ucond (no-op when disabled)."""
    if not guidance_enabled:
        return v
    v_u = jnp.zeros_like(v) if v_ucond is None else v_ucond
    return guidance_w * v + (1 - guidance_w) * v_u


# --- simulation step ------------------------------------------------------------


def simulation_step(
    x_t,
    v,
    t,  # scalar or [*] — all samples share the same time (torch asserts this)
    dt: float,
    gt: float,
    emit_mask,
    params: StepParams,
    key=None,
) -> Float[Array, "* n d"]:
    """One Euler step of the ODE/SDE.

    ``params.sampling_mode`` is a static string on :class:`StepParams`, so we
    dispatch in Python on the active mode while using ``jnp.where`` for
    time-dependent ODE/SDE windowing inside each mode.
    """
    t_scalar = jnp.asarray(t).reshape(())
    mode = params.sampling_mode

    if mode == "vf":
        dx = v * dt
    elif mode == "sc":
        score = vf_to_score(x_t, v, t_scalar)
        endpoint_score_scale = 1.5  # torch: low-temperature ODE endpoint near t=1
        v_sc_scaled = score_to_vf(x_t, score * endpoint_score_scale, t_scalar)
        dx_sc_ode = v_sc_scaled * dt
        eps = jax.random.normal(key, x_t.shape) if key is not None else 0.0
        std_eps = jnp.sqrt(jnp.maximum(jnp.asarray(2 * gt * params.sc_scale_noise * dt), 0.0))
        dx_sc_sde = (v + gt * score) * dt + std_eps * eps
        at_zero = (t_scalar <= 0.0)[..., None, None]
        use_ode_sc = (t_scalar > params.t_lim_ode)[..., None, None]
        dx = jnp.where(use_ode_sc, dx_sc_ode, dx_sc_sde)
        dx = jnp.where(at_zero, 0.0, dx)
    elif mode == "vf_tsr":
        score = vf_to_score(x_t, v, t_scalar)
        tsr = compute_tsr_ratio(t_scalar, params.tsr_k, params.tsr_sigma)
        scaled_score = score * tsr
        v_tsr = score_to_vf(x_t, scaled_score, t_scalar)
        dx_tsr_ode = v_tsr * dt
        eps = jax.random.normal(key, x_t.shape) if key is not None else 0.0
        std_tsr = jnp.sqrt(jnp.maximum(jnp.asarray(2 * gt * params.sc_scale_noise * dt), 0.0))
        dx_tsr_sde = (v_tsr + gt * scaled_score) * dt + std_tsr * eps
        # t == 0: zero update; t outside (below, lim): deterministic
        at_zero = (t_scalar <= 0.0)[..., None, None]
        deterministic = ((t_scalar > params.t_lim_ode) | (t_scalar < params.t_lim_ode_below))[..., None, None]
        dx = jnp.where(deterministic, dx_tsr_ode, dx_tsr_sde)
        dx = jnp.where(at_zero, 0.0, dx)
    else:
        raise ValueError(f"Unknown sampling_mode: {mode!r}")

    x_next = x_t + dx
    x_next = _mask(x_next, emit_mask)
    if params.center_every_step:
        x_next = force_zero_com(x_next, emit_mask)
    return x_next


# --- schedules (host-side, precomputed per run) ----------------------------------


def get_schedule(mode: str, nsteps: int, *, p1: float | None = None, eps: float = 1e-5,
                 tail_steps: int = 0, tail_step_multiplier: int = 1) -> jnp.ndarray:
    """Time partition of [0, 1]; torch-equivalent (host-side numpy-free jnp)."""
    if tail_steps:
        if tail_step_multiplier < 2:
            raise ValueError("tail_step_multiplier must be an integer >= 2")
        base_steps = nsteps - (tail_step_multiplier - 1) * tail_steps
        if base_steps < tail_steps:
            raise ValueError("nsteps must be at least tail_steps * tail_step_multiplier")
        fine = base_steps * tail_step_multiplier
        tail_start = (base_steps - tail_steps) * tail_step_multiplier
        indices = [*range(0, tail_start, tail_step_multiplier), *range(tail_start, fine + 1)]
        return get_schedule(mode, fine, p1=p1, eps=eps)[jnp.asarray(indices)]
    if mode == "uniform":
        return jnp.linspace(0.0, 1.0, nsteps + 1)
    if mode == "power":
        assert p1 is not None
        return jnp.linspace(0.0, 1.0, nsteps + 1) ** p1
    if mode == "log":
        assert p1 is not None and p1 > 0
        t = 1.0 - jnp.logspace(-p1, 0, nsteps + 1)[::-1]
        t = t - t.min()
        return t / t.max()
    raise ValueError(f"Schedule mode not recognized {mode}")


def get_gt(t, mode: str, param: float, clamp_val: float | None = None, eps: float = 1e-2):
    """Noise-injection schedule g(t); torch get_gt equivalent."""
    if mode == "ode":
        return jnp.zeros_like(t)

    def transform_gt(gt, f_pow):
        if f_pow == 1.0:
            return gt
        log_gt = jnp.log(gt)
        mean_log_gt = jnp.mean(log_gt)
        centered = log_gt - mean_log_gt
        normalized = jax.nn.sigmoid(centered)
        normalized = normalized**f_pow
        rec = jax.scipy.special.logit(jnp.clip(normalized, 1e-6, 1 - 1e-6))
        return jnp.exp(rec + mean_log_gt)

    t = jnp.clip(t, 0, 1 - 1e-5)
    if mode == "1-t/t":
        gt = (1.0 - t) / (t + eps)
    elif mode == "tan":
        # sin((1 - t) * pi/2) / cos((1 - t) * pi/2) == cos(t * pi/2) / sin(t * pi/2);
        # direct t * pi/2 avoids catastrophic 1 - t cancellation near t = 0.
        half_pi = jnp.pi / 2.0
        gt = half_pi * jnp.cos(t * half_pi) / (jnp.sin(t * half_pi) + eps)
    elif mode == "1/t":
        gt = 1.0 / (t + eps)
    else:
        raise NotImplementedError(f"gt not implemented {mode}")
    gt = transform_gt(gt, f_pow=param)
    if clamp_val is not None:
        gt = jnp.clip(gt, 0, clamp_val)
    return gt


# --- the sampling loop -----------------------------------------------------------


def _set_x_t(batch: FlowBatch, x_bb, x_lat, t_bb, t_lat) -> FlowBatch:
    is_none = lambda l: l is None  # fields may hold None before the first step
    batch = eqx.tree_at(lambda b: b.x_t_bb_ca, batch, x_bb, is_leaf=is_none)
    batch = eqx.tree_at(lambda b: b.x_t_local_latents, batch, x_lat, is_leaf=is_none)
    batch = eqx.tree_at(lambda b: b.t_bb_ca, batch, t_bb, is_leaf=is_none)
    batch = eqx.tree_at(lambda b: b.t_local_latents, batch, t_lat, is_leaf=is_none)
    return batch


def full_simulation(
    model,
    batch: FlowBatch,
    *,
    key,
    nsteps: int,
    recycle_depths: tuple[int, ...],
    matcher_cfgs: dict[str, MatcherConfig],
    step_params: dict[str, StepParams],
    sampler_args: dict[str, dict],
    n: int,
    guidance_w: float = 1.0,
    coord_cond_mask=None,
    group_ids=None,
    emit_masks: dict[str, Bool[Array, "b n"]] | None = None,
    guidance_model: Callable | None = None,
    jit_model: bool = True,
):
    """Integrate both modalities from t=0 to t=1 (product_space full_simulation).

    The step loop stays a Python loop (like torch); the model call and the
    stepper are jitted. `recycle_depths` entries are static ints — a schedule
    with few unique depths compiles one trunk per depth.
    """
    if len(recycle_depths) != nsteps:
        raise ValueError(f"expected {nsteps} recycle depths, got {len(recycle_depths)}")
    from .nn.transformer import compute_flow_emit_masks

    if emit_masks is None:
        emit_masks = compute_flow_emit_masks(batch)

    ts, gts = {}, {}
    for dm, cfg_args in sampler_args.items():
        ssp = step_params[dm]
        sched_fn = get_schedule_tsr_safe if ssp.sampling_mode == "vf_tsr" else get_schedule
        sargs = cfg_args["schedule"]
        ts[dm] = sched_fn(sargs["mode"], nsteps, p1=sargs["p"],
                          tail_steps=sargs.get("tail_steps", 0),
                          tail_step_multiplier=sargs.get("tail_step_multiplier", 1))
        gargs = cfg_args["gt"]
        g = get_gt(ts[dm][:-1], gargs["mode"], gargs["p"], gargs.get("clamp_val"))
        if ssp.sampling_mode == "vf_tsr":
            g = g / compute_tsr_ratio(ts[dm][:-1], ssp.tsr_k, ssp.tsr_sigma) ** 2
        gts[dm] = g

    bsz = batch.padding_mask.shape[0]
    ts_np = {dm: np.asarray(v, dtype=np.float32) for dm, v in ts.items()}
    gts_np = {dm: np.asarray(v, dtype=np.float32) for dm, v in gts.items()}

    k_init, key_state = jax.random.split(key)
    x = {
        "bb_ca": sample_noise(k_init, (bsz,), n, matcher_cfgs["bb_ca"],
                              emit_masks["bb_ca"], coord_cond_mask=coord_cond_mask, group_ids=group_ids),
        "local_latents": sample_noise(k_init, (bsz,), n, matcher_cfgs["local_latents"],
                                      emit_masks["local_latents"], coord_cond_mask=coord_cond_mask,
                                      group_ids=group_ids),
    }

    model_call = eqx.filter_jit(model) if jit_model else model

    x_1_pred, final_logits = None, {}
    for step in range(nsteps):
        k_model, k_bb, k_lat, key_state = jax.random.split(key_state, 4)
        t_bb_s = jnp.asarray(ts_np["bb_ca"][step], dtype=jnp.float32)
        t_lat_s = jnp.asarray(ts_np["local_latents"][step], dtype=jnp.float32)
        t_bb = jnp.full((bsz,), ts_np["bb_ca"][step], dtype=jnp.float32)
        t_lat = jnp.full((bsz,), ts_np["local_latents"][step], dtype=jnp.float32)
        dt = {"bb_ca": float(ts_np["bb_ca"][step + 1] - ts_np["bb_ca"][step]),
              "local_latents": float(ts_np["local_latents"][step + 1] - ts_np["local_latents"][step])}

        batch = _set_x_t(batch, x["bb_ca"], x["local_latents"], t_bb, t_lat)
        # traced depth: a dynamic fori_loop trip count -> one compile serves all depths
        out = model_call(batch, n_recycle=jnp.asarray(recycle_depths[step], dtype=jnp.int32), key=k_model)
        v = {"bb_ca": out.bb_ca, "local_latents": out.local_latents}

        if guidance_w != 1.0 and matcher_cfgs["bb_ca"].guidance_enabled:
            # torch CFG: drop cath_code/x_motif from the batch -> features fall
            # back to zeros. Only cath_codes exist in the FlowBatch (no motif yet).
            ucond_batch = eqx.tree_at(lambda b: b.cath_codes, batch, None, is_leaf=lambda x: x is None)
            out_u = model_call(ucond_batch, n_recycle=jnp.asarray(recycle_depths[step], dtype=jnp.int32), key=k_model)
            v = {dm: guided_velocity(v[dm], {"bb_ca": out_u.bb_ca, "local_latents": out_u.local_latents}[dm],
                                     guidance_w, matcher_cfgs[dm].guidance_enabled)
                 for dm in v}
        else:
            v = {dm: guided_velocity(v[dm], None, 1.0, False) for dm in v}

        t_step = {"bb_ca": t_bb_s, "local_latents": t_lat_s}
        x_1_pred = {dm: _mask(predict_x1_from_v(x[dm], v[dm], t_step[dm]), emit_masks[dm]) for dm in x}

        # fresh keys per step (split once, one per modality)
        k_bb, k_lat, key_state = jax.random.split(key_state, 3)
        x["bb_ca"] = simulation_step(x["bb_ca"], v["bb_ca"], t_bb_s, dt["bb_ca"],
                                     float(gts_np["bb_ca"][step]), emit_masks["bb_ca"], step_params["bb_ca"], key=k_bb)
        x["local_latents"] = simulation_step(x["local_latents"], v["local_latents"], t_lat_s,
                                             dt["local_latents"], float(gts_np["local_latents"][step]),
                                             emit_masks["local_latents"], step_params["local_latents"], key=k_lat)
        if guidance_model is not None:
            delta = guidance_model(x_1_pred=x_1_pred)
            for dm, d in delta.items():
                x[dm] = x[dm] + d.astype(x[dm].dtype)

    final_logits = {"insertion_logits": out.insertion_logits, "seq_logits": out.seq_logits}
    return x, x_1_pred, final_logits


# --- RFD4 sampler ----------------------------------------------------------------

#: Native configs/recycle_strategy/thinking_high.yaml (high/front_loaded):
#: 60 sampling steps, 300 total trunk passes including the first.
THINKING_HIGH: tuple[tuple[int, int], ...] = ((10, 9), (10, 8), (10, 5), (20, 3), (10, 2))


class RFD4Result(eqx.Module):
    """Sampling result of :meth:`RFD4Sampler.sample`.

    Attributes:
        x: Final flow states per modality.
        x_1: Last step's clean predictions (per-modality dict + ``aa_pred``).
        data: The final FlowBatch carrying ``aa_pred/x_pred/z_pred/v_pred/s_pred``.
        dec: DecoderOutput, or ``None`` when sampled with ``decode=False``.

    All fields keep the batch dimension exactly as sampled; derived outputs
    (:meth:`to_atom24` / :meth:`to_atoms`) are unbatched for single-sample
    results and carry a leading batch dim otherwise. Iterating yields
    ``(x, x_1, data, dec)`` so tuple unpacking keeps working.
    """

    x: dict
    x_1: dict | None
    data: FlowBatch
    dec: DecoderOutput | None

    def __iter__(self):
        yield self.x
        yield self.x_1
        yield self.data
        yield self.dec

    @property
    def is_single_sample(self) -> bool:
        """True iff the result carries no batch dimension (b == 1)."""
        return int(np.asarray(self.data.padding_mask).shape[0]) == 1

    def _slice(self, index: int):
        """Single-sample (keepdim) views of (dec, data, x, x_1) at `index`.

        Only leaves whose leading dimension equals the batch size are sliced —
        static tables (e.g. the decoder's allowed_residue_types) pass through
        untouched."""
        b = int(np.asarray(self.data.padding_mask).shape[0])

        def take(a):
            if a is None:
                return None

            def slicer(leaf):
                if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                        and int(leaf.shape[0]) == b:
                    return leaf[index:index + 1]
                return leaf

            return jax.tree.map(slicer, a)

        dec_i = take(self.dec)
        data_i = take(self.data)
        x_i = {k: take(v) for k, v in self.x.items()}
        x1_i = None if self.x_1 is None else {k: take(v) for k, v in self.x_1.items()}
        aa_i = data_i.aa_pred if data_i.aa_pred is not None else None
        return dec_i, data_i, x_i, x1_i, aa_i

    # ------------------------------------------------------------- saving
    def _structure(self, index: int = 0, metadata: dict | None = None):
        """biotite AtomArray for one sample: decoded polymer chains via the
        native-parity export + non-polymer (atomized) atoms from the flow
        state (element-count names + res_name "LIG" unless `metadata`
        overrides with the keys "residue_name" / "atom_name"). Chains are
        split into A, B, C, ... by the batch's chain_iid (unique values in
        first-appearance order over the real tokens)."""
        import biotite.structure as bs

        from .pdb_io import decoded_to_atom_array

        dec_i, data_i, x_i, _x1_i, aa_i = self._slice(index)
        assert dec_i is not None, \
            "save_pdb/save_cif require decode=True at sampling time"
        letters = _chain_letters(data_i)
        arr = decoded_to_atom_array(dec_i, data_i, aa_pred=aa_i,
                                    chain_ids=letters)
        # non-polymer (atomized) tokens: one atom each, from the flow state,
        # under their own chain letter
        x_bb = np.asarray(x_i["bb_ca"])[0] * 10.0  # nm -> Å
        coords_np, elements, _restype = nonpolymer_atoms(data_i, x_bb)
        if len(coords_np):
            metadata = metadata or {}
            res_name = str(metadata.get("residue_name", "LIG"))
            atom_names = metadata.get("atom_name")
            if atom_names is None:
                counts: dict[int, int] = {}
                atom_names = []
                for z in elements:
                    counts[int(z)] = counts.get(int(z), 0) + 1
                    atom_names.append(f"{_Z_SYMBOL.get(int(z), str(z))}"
                                      f"{counts[int(z)]}")
            tt = np.asarray(data_i.token_type)[0]
            real = np.asarray(data_i.padding_mask)[0]
            atom_toks = np.flatnonzero(real & (tt == 3))
            lig_letter = str(letters[atom_toks[0]]) if len(atom_toks) else "B"
            n_lig = len(coords_np)
            lig = bs.AtomArray(n_lig)
            lig.add_annotation("hetero", dtype=bool)
            lig.add_annotation("occupancy", dtype=float)
            lig.add_annotation("b_factor", dtype=float)
            lig.coord[:] = coords_np
            lig.chain_id[:] = lig_letter
            lig.res_id[:] = 1
            lig.res_name[:] = res_name
            lig.atom_name[:] = np.asarray(atom_names, dtype="U6")[:len(coords_np)]
            lig.element[:] = np.asarray(
                [_Z_SYMBOL.get(int(z), str(z)) for z in elements], dtype="U4")
            lig.hetero[:] = True
            lig.occupancy[:] = 1.0
            lig.b_factor[:] = 0.0
            arr = arr + lig
        return arr

    def save_pdb(self, path, *, index: int = 0,
                 metadata: dict | None = None) -> None:
        """Write one sample as a PDB file (`index` selects the batch element;
        ignored for single-sample results)."""
        import biotite.structure.io.pdb as pdbio

        arr = self._structure(index=index, metadata=metadata)
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(path))

    def save_cif(self, path, *, index: int = 0,
                 metadata: dict | None = None) -> None:
        """Write one sample as an mmCIF file (`index` selects the batch
        element; ignored for single-sample results)."""
        from biotite.structure.io.pdbx import CIFFile, set_structure

        arr = self._structure(index=index, metadata=metadata)
        cif = CIFFile()
        set_structure(cif, arr)
        cif.write(str(path))

    # ------------------------------------------------------- atom formats
    def to_atom24(self):
        """All decoded coordinates/masks in atom24 format (per-residue 24
        slots in CCD order; absent atoms 0.0/False) — unbatched for
        single-sample results, leading batch dim for b > 1. Requires
        decode=True."""
        assert self.dec is not None, \
            "to_atom24 requires decode=True at sampling time"
        b = int(np.asarray(self.data.padding_mask).shape[0])
        out = []
        for i in range(b):
            dec_i, data_i, _x_i, _x1_i, aa_i = self._slice(i)
            out.append(to_atom24(dec_i, data_i, aa_pred=aa_i))
        if self.is_single_sample:
            return out[0]
        return Atom24(
            coords=np.stack([a.coords for a in out]),
            mask=np.stack([a.mask for a in out]),
            restype=np.stack([a.restype for a in out]),
            elements=np.stack([a.elements for a in out]))

    def to_atoms(self):
        """Flat per-atom coordinates/masks, length exactly
        24 * n_polymer_residues + n_nonpolymer_atoms (absent atoms keep
        their 0.0/False slots; non-polymer atoms come from the flow state) —
        unbatched for single-sample results, leading batch dim for b > 1.
        Requires decode=True."""
        a24 = self.to_atom24()
        assert self.dec is not None
        b = int(np.asarray(self.data.padding_mask).shape[0])

        def one(i: int) -> Atoms:
            atoms = a24[i] if b > 1 else a24
            flat = atoms.to_atoms()
            dec_i, data_i, x_i, _x1_i, _aa_i = self._slice(i)
            x_bb = np.asarray(x_i["bb_ca"])[0] * 10.0
            coords_np, elements, restype = nonpolymer_atoms(data_i, x_bb)
            m = len(coords_np)
            pad = (0, m)
            coords = np.concatenate([flat.coords, coords_np], axis=-2)
            mask = np.pad(flat.mask, (*[(0, 0)] * (flat.mask.ndim - 1), pad))
            mask[..., flat.mask.shape[-1]:] = True   # ligand atoms are real
            r = np.pad(flat.restype, (*[(0, 0)] * (flat.restype.ndim - 1), pad),
                       constant_values=0)
            r[..., flat.restype.shape[-1]:] = restype
            e = np.pad(flat.elements, (*[(0, 0)] * (flat.elements.ndim - 1), pad),
                       constant_values=0)
            e[..., flat.elements.shape[-1]:] = elements
            return Atoms(coords=coords, mask=mask, restype=r, elements=e)

        if self.is_single_sample:
            return one(0)
        return Atoms(
            coords=np.stack([one(i).coords for i in range(b)]),
            mask=np.stack([one(i).mask for i in range(b)]),
            restype=np.stack([one(i).restype for i in range(b)]),
            elements=np.stack([one(i).elements for i in range(b)]))


_Z_SYMBOL = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 11: "NA", 12: "MG",
             15: "P", 16: "S", 17: "CL", 20: "CA", 25: "MN", 26: "FE",
             30: "ZN", 34: "SE"}


def _chain_letter(i: int) -> str:
    """Spreadsheet-style chain letters: A..Z, AA, AB, ..."""
    s = ""
    while True:
        s = chr(ord("A") + i % 26) + s
        i = i // 26 - 1
        if i < 0:
            return s


def _chain_letters(data_i) -> np.ndarray:
    """Per-token chain letters (A, B, C, ...) from the batch's chain_iid:
    unique values in first-appearance order over the real tokens."""
    ciid = np.asarray(data_i.chain_iid)
    if ciid.ndim == 2:
        ciid = ciid[0]
    real = np.asarray(data_i.padding_mask)
    if real.ndim == 2:
        real = real[0]
    letters = np.full(ciid.shape, "A", dtype="<U4")
    order: list[float] = []
    for t in np.flatnonzero(real):
        v = float(ciid[t])
        if v not in order:
            order.append(v)
        letters[t] = _chain_letter(order.index(v))
    return letters


def _decode_predictions(model, data: FlowBatch, x_bb, x_lat, aa_pred):
    return model.decode(data, x_bb, x_lat, aa_pred=aa_pred)


class RFD4Sampler:
    """Flow sampler driving a jitted step callable (``RFD4.step()``).

    Implements the release product-space sampling loop (noise init, CFG,
    vf_tsr Euler integration of both modalities) as a Python for-loop over the
    jitted ``step(model, key, data)``, reading the per-step predictions from
    the returned batch. The recycle depth is a TRACED input on the batch
    (``data.n_recycle``), so one compile serves all depths per token shape.
    """

    def __init__(self, step, model, *, matcher_cfgs=None, step_params=None,
                 sampler_args=None, guidance_w: float = 1.0, jit_step: bool = True):
        if matcher_cfgs is None or step_params is None or sampler_args is None:
            from .model import MATCHER_CFGS as _mc, SAMPLER_ARGS as _sa, STEP_PARAMS as _sp

            matcher_cfgs = matcher_cfgs or _mc
            step_params = step_params or _sp
            sampler_args = sampler_args or _sa
        self.step = step
        self.model = model
        self.matcher_cfgs = matcher_cfgs
        self.step_params = step_params
        self.sampler_args = sampler_args
        self.guidance_w = guidance_w
        self._step = eqx.filter_jit(step) if jit_step else step
        self._decode = eqx.filter_jit(_decode_predictions)

    def sample(self, data: FlowBatch, key, *, nsteps: int = 60,
               recycle_schedule: tuple[tuple[int, int], ...] = THINKING_HIGH,
               decode: bool = True, jit_decode: bool = True):
        """Integrate the flow-matching ODE; optionally AE-decode at the end.

        Args:
            data: The model-ready batch (``RFD4Input.to_batch()``).
            key: JAX random key seeding the noise init and per-step noise.
            nsteps: Number of Euler integration steps.
            recycle_schedule: Per-step recycle depths; a ``(width, depth)``
                schedule is expanded to ``nsteps`` depths (default
                ``THINKING_HIGH``).
            decode: Run the autoencoder decoder on the final clean
                prediction (needed for ``save_pdb``/``save_cif``).
            jit_decode: Jit the decode call.

        Returns:
            An :class:`RFD4Result` bundling ``(x, x_1, data, dec)`` — final
            flow states, the last step's clean predictions, the final batch
            (with ``aa_pred/x_pred/z_pred/v_pred/s_pred``) and the decoded
            structures (``None`` when ``decode=False``).

        Raises:
            ValueError: If the recycle schedule does not expand to ``nsteps``.
        """
        if len(recycle_schedule) != nsteps:
            depths = [d for width, depth in recycle_schedule for d in [depth] * width]
            if len(depths) != nsteps:
                raise ValueError(
                    f"recycle schedule expands to {len(depths)} steps; expected {nsteps}")
            recycle_schedule = tuple(depths)

        from .nn.transformer import compute_flow_emit_masks

        emit_masks = compute_flow_emit_masks(data)
        ts, gts = {}, {}
        for dm, cfg_args in self.sampler_args.items():
            ssp = self.step_params[dm]
            sched_fn = get_schedule_tsr_safe if ssp.sampling_mode == "vf_tsr" else get_schedule
            sargs = cfg_args["schedule"]
            ts[dm] = sched_fn(sargs["mode"], nsteps, p1=sargs["p"],
                              tail_steps=sargs.get("tail_steps", 0),
                              tail_step_multiplier=sargs.get("tail_step_multiplier", 1))
            gargs = cfg_args["gt"]
            g = get_gt(ts[dm][:-1], gargs["mode"], gargs["p"], gargs.get("clamp_val"))
            if ssp.sampling_mode == "vf_tsr":
                g = g / compute_tsr_ratio(ts[dm][:-1], ssp.tsr_k, ssp.tsr_sigma) ** 2
            gts[dm] = g

        bsz, n = data.padding_mask.shape
        ts_np = {dm: np.asarray(v, dtype=np.float32) for dm, v in ts.items()}
        gts_np = {dm: np.asarray(v, dtype=np.float32) for dm, v in gts.items()}

        k_init, key_state = jax.random.split(key)
        x = {
            "bb_ca": sample_noise(k_init, (bsz,), n,
                                  self.matcher_cfgs["bb_ca"], emit_masks["bb_ca"]),
            "local_latents": sample_noise(k_init, (bsz,), n,
                                          self.matcher_cfgs["local_latents"],
                                          emit_masks["local_latents"]),
        }

        x_1, data = None, eqx.tree_at(lambda d: d.n_recycle, data, None,
                                      is_leaf=lambda l: l is None)
        for i in range(nsteps):
            k_model, _, _, key_state = jax.random.split(key_state, 4)
            t_bb_s = jnp.asarray(ts_np["bb_ca"][i], dtype=jnp.float32)
            t_lat_s = jnp.asarray(ts_np["local_latents"][i], dtype=jnp.float32)
            t_bb = jnp.full((bsz,), ts_np["bb_ca"][i], dtype=jnp.float32)
            t_lat = jnp.full((bsz,), ts_np["local_latents"][i], dtype=jnp.float32)
            data = _set_x_t(data, x["bb_ca"], x["local_latents"], t_bb, t_lat)
            data = eqx.tree_at(lambda d: d.n_recycle, data,
                               jnp.asarray(recycle_schedule[i], dtype=jnp.int32),
                               is_leaf=lambda l: l is None)
            data = self._step(self.model, k_model, data)
            v = dict(data.v_pred)
            if self.guidance_w != 1.0 and self.matcher_cfgs["bb_ca"].guidance_enabled:
                # torch CFG: drop cath_code from the batch -> features fall back
                ucond = eqx.tree_at(lambda b: b.cath_codes, data, None,
                                    is_leaf=lambda x: x is None)
                data_u = self._step(self.model, k_model, ucond)
                v = {dm: guided_velocity(v[dm], data_u.v_pred[dm], self.guidance_w,
                                         self.matcher_cfgs[dm].guidance_enabled)
                     for dm in v}
            t = {"bb_ca": t_bb, "local_latents": t_lat}
            x_1 = {dm: _mask(predict_x1_from_v(x[dm], v[dm], t[dm]), emit_masks[dm])
                   for dm in x}

            # fresh keys per step (split once, one per modality) — same key
            # discipline as full_simulation
            k_bb, k_lat, key_state = jax.random.split(key_state, 3)
            x["bb_ca"] = simulation_step(x["bb_ca"], v["bb_ca"], t_bb_s,
                                         float(ts_np["bb_ca"][i + 1] - ts_np["bb_ca"][i]),
                                         float(gts_np["bb_ca"][i]), emit_masks["bb_ca"],
                                         self.step_params["bb_ca"], key=k_bb)
            x["local_latents"] = simulation_step(
                x["local_latents"], v["local_latents"], t_lat_s,
                float(ts_np["local_latents"][i + 1] - ts_np["local_latents"][i]),
                float(gts_np["local_latents"][i]), emit_masks["local_latents"],
                self.step_params["local_latents"], key=k_lat)

        dec = None
        if decode:
            dec = (self._decode(self.model, data, x_1["bb_ca"],
                                x_1["local_latents"], data.aa_pred)
                   if jit_decode else
                   _decode_predictions(self.model, data, x_1["bb_ca"],
                                       x_1["local_latents"], data.aa_pred))
        return RFD4Result(x=x, x_1=x_1, data=data, dec=dec)


def get_schedule_tsr_safe(mode: str, nsteps: int, *, p1: float | None = None, eps: float = 1e-3,
                          tail_steps: int = 0, tail_step_multiplier: int = 1) -> jnp.ndarray:
    """Schedule starting at eps (safe for the TSR t=0 handling)."""
    if tail_steps:
        if tail_step_multiplier < 2:
            raise ValueError("tail_step_multiplier must be an integer >= 2")
        base_steps = nsteps - (tail_step_multiplier - 1) * tail_steps
        fine = base_steps * tail_step_multiplier
        tail_start = (base_steps - tail_steps) * tail_step_multiplier
        indices = [*range(0, tail_start, tail_step_multiplier), *range(tail_start, fine + 1)]
        return get_schedule_tsr_safe(mode, fine, p1=p1, eps=eps)[jnp.asarray(indices)]
    if mode == "uniform":
        return jnp.linspace(eps, 1.0, nsteps + 1)
    if mode == "power":
        assert p1 is not None
        return jnp.linspace(eps ** (1 / p1), 1.0, nsteps + 1) ** p1
    if mode == "log":
        assert p1 is not None and p1 > 0
        t = 1.0 - jnp.logspace(-p1, eps, nsteps + 1)[::-1]
        t = t - (t.min() - eps)
        return t / t.max()
    raise ValueError(f"Schedule mode not recognized {mode}")


def decode(model_ae_decoder, batch: FlowBatch, x_bb, x_lat, *, key=None) -> DecoderOutput:
    """Decode sampled (bb_ca, local_latents) into sequence + atom37 coordinates."""
    batch = eqx.tree_at(lambda b: b.z_latent, batch, x_lat, is_leaf=lambda x: x is None)
    batch = eqx.tree_at(lambda b: b.ca_coors_nm, batch, x_bb, is_leaf=lambda x: x is None)
    return model_ae_decoder(batch, key=key)
