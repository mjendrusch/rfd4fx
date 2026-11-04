"""Trajectory-level beam search over flow-matching rollouts.

Each beam element is a full latent state; branching runs independent partial
ODE continuations from the same state, scored by a user ``evaluate`` callable
on decoded :class:`RFD4Result` objects."""

from __future__ import annotations

import dataclasses
import os
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .flow_matching import (RFD4Result, THINKING_HIGH, _mask, _set_x_t,
                            compute_tsr_ratio, get_gt, get_schedule,
                            get_schedule_tsr_safe, guided_velocity,
                            predict_x1_from_v, sample_noise, simulation_step)
from .types import FlowBatch


@dataclasses.dataclass
class SearchCheckpoint:
    """Per-checkpoint configuration (keyed by positive schedule step).

    evaluate: optional callable overriding the search default for this
      checkpoint; signature ``evaluate(result: RFD4Result) -> dict`` with
      keys "score" (float32 scalar, higher better), "pass" (bool), "stop"
      (bool), "metrics" (auxiliary dict — not used by the search itself).
    n_branch: optional children-per-state override for the interval ending
      at this checkpoint.
    use_clean_prediction: score the branch's decoded clean prediction
      instead of a full rollout to t=1.
    """

    evaluate: Callable | None = None
    n_branch: int | None = None
    use_clean_prediction: bool = False


class RFD4SearchResults(eqx.Module):
    """Ranked search output.

    Attributes:
        results: One batched :class:`RFD4Result` whose batch axis holds the
            candidates, sorted by score descending (rank 0 = best).
        score: float32 (N,) — the sorting key.
        metrics: PyTree with leading shape (N,) — the auxiliary metrics
            dicts returned by the evaluate callable.
    is_intermediate: bool (N,) — True for checkpoint-scored candidates,
      False for the final survivors.
    """

    results: RFD4Result
    score: jnp.ndarray
    metrics: dict
    is_intermediate: jnp.ndarray

    def __len__(self) -> int:
        return int(np.asarray(self.score).shape[0])

    def __getitem__(self, index):
        return (_result_take(self.results, index), self.score[index],
                jax.tree.map(lambda l: l[index], self.metrics),
                self.is_intermediate[index])


# ---------------------------------------------------------------- helpers
def _batch_size(result: RFD4Result) -> int:
    return int(np.asarray(result.data.padding_mask).shape[0])


def _result_take(result: RFD4Result, index: int) -> RFD4Result:
    """One candidate from a batched RFD4Result, batch dim removed."""

    def take(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                and int(leaf.shape[0]) == _batch_size(result):
            return leaf[index]
        return leaf

    return jax.tree.map(take, result)


def stack_results(results: list[RFD4Result]) -> RFD4Result:
    """Stack single-sample RFD4Results into one batched RFD4Result.

    Only leaves whose leading dim is 1 (truly batched) are stacked; static
    tables (e.g. the decoder's allowed_residue_types) are taken from
    candidate 0.
    """

    def merge(*leaves):
        first = leaves[0]
        if hasattr(first, "shape") and getattr(first, "ndim", 0) >= 1 \
                and int(first.shape[0]) == 1:
            return jnp.stack([np.asarray(l) for l in leaves], axis=0)
        return first

    return jax.tree.map(merge, *results)


def _unbatched(result: RFD4Result) -> RFD4Result:
    """Remove a leading batch dim of size 1 (for user evaluate callables)."""

    def squeeze(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                and int(leaf.shape[0]) == 1:
            return leaf[0]
        return leaf

    return jax.tree.map(squeeze, result)


def _tile_beam(batch, widths: int, b: int):
    """Tile a batched container to the interleaved beam layout
    ([inp0 w0, inp1 w0, ..., inp0 w1, ...] — native repeat(width, 1))."""

    def tile(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                and int(leaf.shape[0]) == b:
            return jnp.tile(leaf, (widths,) + (1,) * (leaf.ndim - 1))
        return leaf

    return jax.tree.map(tile, batch)


def _gather(result: RFD4Result, indices: list[int]) -> RFD4Result:
    """Gather candidates (keepdim) from a batched RFD4Result.

    Only leaves whose leading dim equals the batch size are gathered; static
    tables (e.g. the decoder's allowed_residue_types) pass through."""

    def take(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                and int(leaf.shape[0]) == _batch_size(result):
            return leaf[jnp.asarray(indices)]
        return leaf

    return jax.tree.map(take, result)


def _stack_results(pairs: list[tuple[RFD4Result, list[int]]]) -> RFD4Result:
    """Concatenate gathered candidate groups into one batched RFD4Result.

    `pairs` holds (batched source result, candidate indices into it) in the
    desired output order. Leaves whose leading dim equals the group's gather
    size are concatenated (candidate data); other leaves (static tables) are
    taken from the first group.
    """
    gathered = [(_gather(res, idxs), len(idxs)) for res, idxs in pairs]

    def merge(*leaves):
        parts, static = [], None
        for leaf, (_g, k) in zip(leaves, gathered):
            if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                    and int(leaf.shape[0]) == k:
                parts.append(np.asarray(leaf))
            elif static is None:
                static = leaf
        if parts:
            return jnp.concatenate(parts, axis=0)
        return static

    return jax.tree.map(merge, *[g for g, _ in gathered])


def _stack_metrics_tree(trees: list[tuple[dict, list[int]]]) -> dict:
    """Concatenate per-source metric PyTrees (leading candidate axes) in the
    given (tree, indices) order."""
    trees = [(t, i) for t, i in trees if t]
    if not trees:
        return {}
    keys: list[str] = []
    for t, _ in trees:
        for k in t:
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        out[k] = jnp.concatenate([
            jnp.asarray(t[k])[jnp.asarray(idxs)] for t, idxs in trees], axis=0)
    return out


def _take_batch(batch, start: int, end: int, indices=None):
    """Slice a batched container along its batch dim (keepdim), or gather
    explicit indices."""

    def take(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1:
            if indices is not None:
                return leaf[jnp.asarray(indices)]
            return leaf[start:end]
        return leaf

    return jax.tree.map(take, batch)


def _select_inputs(batch, keep_inp, b: int, widths: int):
    """Keep only the given inputs (each still carrying its width replicas)."""

    def take(leaf):
        if hasattr(leaf, "shape") and getattr(leaf, "ndim", 0) >= 1 \
                and int(leaf.shape[0]) == b * widths:
            return jnp.concatenate(
                [leaf[i * widths:(i + 1) * widths] for i in keep_inp], axis=0)
        return leaf

    return jax.tree.map(take, batch)


def _stack_metrics(metric_dicts: list[dict]) -> dict:
    """Stack a list of auxiliary metric dicts into a PyTree with leading
    shape (N,)."""
    if not metric_dicts or all(not m for m in metric_dicts):
        return {}
    keys: list[str] = []
    for m in metric_dicts:
        for k in m:
            if k not in keys:
                keys.append(k)
    out = {}
    for k in keys:
        vals = [m.get(k, np.nan) for m in metric_dicts]
        if all(hasattr(v, "shape") or np.isscalar(v) for v in vals):
            try:
                out[k] = np.stack([np.asarray(v, np.float32) for v in vals])
                continue
            except (TypeError, ValueError):
                pass
        out[k] = vals
    return out


def _save_state(directory, x, step: int, beam_width: int, n_branch: int,
                nsteps: int, b: int) -> str:
    """Save the selected beam states after an interior checkpoint (npz)."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"states_step{step}.npz")
    np.savez(path,
             step=np.int64(step), beam_width=np.int64(beam_width),
             n_branch=np.int64(n_branch), nsteps=np.int64(nsteps),
             batch_size=np.int64(b),
             x_bb_ca=np.asarray(x["bb_ca"]),
             x_local_latents=np.asarray(x["local_latents"]))
    return path


def _load_state(path: str, *, n: int, beam_width: int, b: int, nsteps: int):
    """Load a saved interior beam state; validates shapes and configuration."""
    z = np.load(path)
    step = int(z["step"])
    if not 0 < step < nsteps:
        raise ValueError(f"resume: saved step {step} outside (0, {nsteps})")
    if int(z["beam_width"]) != beam_width:
        raise ValueError(f"resume: saved beam_width {int(z['beam_width'])} != "
                         f"{beam_width}")
    if int(z["nsteps"]) != nsteps:
        raise ValueError(f"resume: saved nsteps {int(z['nsteps'])} != {nsteps}")
    if int(z["batch_size"]) != b:
        raise ValueError(f"resume: saved batch_size {int(z['batch_size'])} != {b}")
    x = {"bb_ca": jnp.asarray(z["x_bb_ca"]),
         "local_latents": jnp.asarray(z["x_local_latents"])}
    for dm, v in x.items():
        if v.shape[:2] != (b * beam_width, n):
            raise ValueError(f"resume: saved {dm} shape {v.shape} does not "
                             f"match the beam ({b * beam_width}, {n}, *)")
        if not np.isfinite(np.asarray(v)).all():
            raise ValueError(f"resume: saved {dm} contains non-finite values")
    return x, step


def batch_evaluate_default(evaluate: Callable) -> Callable:
    """Serial batched evaluation (the default): loop over the candidate
    batch, call `evaluate` per single-sample result, stack the outputs over
    axis 0."""

    def batched(results: RFD4Result) -> dict:
        n = _batch_size(results)
        scores, passes, stops, metrics = [], [], [], []
        for j in range(n):
            out = evaluate(_result_take(results, j))
            missing = [k for k in ("score", "pass", "stop") if k not in out]
            if missing:
                raise ValueError(
                    f"evaluate callable is missing key(s) {missing} (needs "
                    "'score', 'pass', 'stop'; 'metrics' optional)")
            scores.append(np.float32(jnp.asarray(out["score"]).reshape(())))
            passes.append(bool(out["pass"]))
            stops.append(bool(out["stop"]))
            metrics.append(dict(out.get("metrics") or {}))
        return {"score": jnp.asarray(scores, jnp.float32),
                "pass": jnp.asarray(passes, bool),
                "stop": jnp.asarray(stops, bool),
                "metrics": _stack_metrics(metrics)}

    return batched


def batch_evaluate_vmap(evaluate: Callable) -> Callable:
    """Batched evaluation via ``jax.vmap`` — mapped over the candidate batch
    axis. `evaluate` must be jax-pure (no numpy conversions or host side
    effects) and return jax values."""

    def batched(results: RFD4Result) -> dict:
        out = jax.vmap(evaluate)(results)
        missing = [k for k in ("score", "pass", "stop") if k not in out]
        if missing:
            raise ValueError(
                f"evaluate callable is missing key(s) {missing} (needs "
                "'score', 'pass', 'stop'; 'metrics' optional)")
        return out

    return batched


def batch_evaluate_map(evaluate: Callable) -> Callable:
    """Batched evaluation via ``jax.lax.map`` — serial (loop-based) jax
    evaluation over the candidate batch axis; `evaluate` must be jax-pure."""

    def batched(results: RFD4Result) -> dict:
        out = jax.lax.map(evaluate, results)
        missing = [k for k in ("score", "pass", "stop") if k not in out]
        if missing:
            raise ValueError(
                f"evaluate callable is missing key(s) {missing} (needs "
                "'score', 'pass', 'stop'; 'metrics' optional)")
        return out

    return batched


class RFD4Search(eqx.Module):
    """Beam search over flow trajectories (flexcraft-style interface).

    Args:
      step: the model step callable ``step(model, key, data) -> data``
        (``RFD4.step()``), as for :class:`RFD4Sampler`.
      model: the RFD4 module (for AE decode of scored candidates).
      evaluate: default scoring callable — **unbatched**:
        ``evaluate(result: RFD4Result) -> dict`` with "score" (float32,
        higher better), "pass" (bool), "stop" (bool), "metrics" (auxiliary
        dict); receives a single-sample result.
      batch_evaluate: optional callable taking the evaluate function and
        returning its batched version: ``batch_evaluate(evaluate) ->
        batched_fn`` with ``batched_fn(result: RFD4Result) -> dict`` whose
        values carry a leading candidate axis (score (N,), pass (N,),
        stop (N,), metrics PyTree leading (N,)). None (default) uses
        :func:`batch_evaluate_default` (a serial per-candidate loop); see
        :func:`batch_evaluate_vmap` / :func:`batch_evaluate_map` for
        jax-native alternatives.
      checkpoints: dict of positive schedule step -> SearchCheckpoint (or
        dicts with the same fields). Step 0 is implicit; a checkpoint at the
        schedule end makes its survivors the final designs; without one,
        survivors of the last checkpoint finish without selection.
      n_branch: default children per beam state per checkpoint interval.
      beam_width: candidates retained per input after each checkpoint.
      guidance_w: classifier-free guidance weight (the sampler's semantics).
    """

    step: Callable
    model: Any
    evaluate: Callable
    batch_evaluate: Callable | None
    checkpoints: dict
    n_branch: int
    beam_width: int
    matcher_cfgs: dict
    step_params: dict
    sampler_args: dict
    guidance_w: float
    _step: Callable

    def __init__(self, step, model, evaluate: Callable, *,
                 batch_evaluate: Callable | None = None,
                 checkpoints: dict[int, SearchCheckpoint | dict],
                 n_branch: int = 4, beam_width: int = 4,
                 matcher_cfgs=None, step_params=None, sampler_args=None,
                 guidance_w: float = 1.0, jit_step: bool = True):
        if n_branch < 1 or beam_width < 1:
            raise ValueError("n_branch and beam_width must be >= 1")
        norm: dict[int, SearchCheckpoint] = {}
        for step_i, cfg in checkpoints.items():
            step_i = int(step_i)
            if step_i <= 0:
                raise ValueError(
                    f"checkpoint steps must be positive, got {step_i}")
            if isinstance(cfg, SearchCheckpoint):
                norm[step_i] = cfg
            elif isinstance(cfg, dict):
                unknown = set(cfg) - {"evaluate", "n_branch",
                                      "use_clean_prediction"}
                if unknown:
                    raise ValueError(f"unknown checkpoint keys {sorted(unknown)}")
                norm[step_i] = SearchCheckpoint(**cfg)
            else:
                raise TypeError(f"checkpoint config must be a SearchCheckpoint "
                                f"or dict, got {type(cfg).__name__}")
        if not norm:
            raise ValueError("checkpoints must be non-empty")
        if matcher_cfgs is None or step_params is None or sampler_args is None:
            from .model import MATCHER_CFGS as _mc, SAMPLER_ARGS as _sa, STEP_PARAMS as _sp

            matcher_cfgs = matcher_cfgs or _mc
            step_params = step_params or _sp
            sampler_args = sampler_args or _sa
        self.step = step
        self.model = model
        self.evaluate = evaluate
        self.batch_evaluate = batch_evaluate
        self.checkpoints = norm
        self.n_branch = int(n_branch)
        self.beam_width = int(beam_width)
        self.matcher_cfgs = matcher_cfgs
        self.step_params = step_params
        self.sampler_args = sampler_args
        self.guidance_w = guidance_w
        self._step = eqx.filter_jit(step) if jit_step else step

    # ----------------------------------------------------------- schedules
    def _schedules(self, nsteps: int):
        ts, gts = {}, {}
        for dm, cfg_args in self.sampler_args.items():
            ssp = self.step_params[dm]
            sched_fn = (get_schedule_tsr_safe if ssp.sampling_mode == "vf_tsr"
                        else get_schedule)
            sargs = cfg_args["schedule"]
            ts[dm] = sched_fn(sargs["mode"], nsteps, p1=sargs["p"],
                              tail_steps=sargs.get("tail_steps", 0),
                              tail_step_multiplier=sargs.get("tail_step_multiplier", 1))
            gargs = cfg_args["gt"]
            g = get_gt(ts[dm][:-1], gargs["mode"], gargs["p"], gargs.get("clamp_val"))
            if ssp.sampling_mode == "vf_tsr":
                g = g / compute_tsr_ratio(ts[dm][:-1], ssp.tsr_k, ssp.tsr_sigma) ** 2
            gts[dm] = g
        return ts, gts

    def _depths(self, recycle_schedule, nsteps: int) -> tuple[int, ...]:
        if len(recycle_schedule) != nsteps:
            depths = [d for width, depth in recycle_schedule
                      for d in [depth] * width]
            if len(depths) != nsteps:
                raise ValueError(
                    f"recycle schedule expands to {len(depths)} steps; "
                    f"expected {nsteps}")
            return tuple(depths)
        return tuple(d for _, d in recycle_schedule)

    # ------------------------------------------------- partial simulation
    def _partial_sim(self, batch, x, *, emit_masks, ts, gts, start: int,
                     end: int, depths, key):
        """Integrate the schedule segment [start, end) from state `x`.

        Mirrors the sampler loop (jitted step, traced recycle depth, per-step
        key splits). Returns (x_end, x_1_last, batch) where x_1_last is the
        emit-masked clean prediction of the segment's last step (the decode
        input, as in RFD4Sampler)."""
        from .nn.transformer import compute_flow_emit_masks

        emit_masks = compute_flow_emit_masks(batch)  # match the batch's batch dim
        x = dict(x)
        x_1_last = None
        for step_i in range(start, end):
            k_model, k_bb, k_lat, key = jax.random.split(key, 4)
            t_bb = ts["bb_ca"][step_i] * jnp.ones(batch.padding_mask.shape[0])
            t_lat = ts["local_latents"][step_i] * jnp.ones(batch.padding_mask.shape[0])
            batch = _set_x_t(batch, x["bb_ca"], x["local_latents"], t_bb, t_lat)
            batch = eqx.tree_at(lambda d: d.n_recycle, batch,
                                jnp.asarray(depths[step_i], dtype=jnp.int32),
                                is_leaf=lambda l: l is None)
            out = self._step(self.model, k_model, batch)
            v = dict(out.v_pred)
            if self.guidance_w != 1.0 and self.matcher_cfgs["bb_ca"].guidance_enabled:
                ucond = eqx.tree_at(lambda b_: b_.cath_codes, batch, None,
                                    is_leaf=lambda l: l is None)
                out_u = self._step(self.model, k_model, ucond)
                v_u = dict(out_u.v_pred)
                v = {dm: guided_velocity(v[dm], v_u[dm], self.guidance_w,
                                         self.matcher_cfgs[dm].guidance_enabled)
                     for dm in v}
            else:
                v = {dm: guided_velocity(v[dm], None, 1.0, False) for dm in v}
            x_1_last = {
                "bb_ca": _mask(predict_x1_from_v(x["bb_ca"], v["bb_ca"],
                                                 ts["bb_ca"][step_i]),
                               emit_masks["bb_ca"]),
                "local_latents": _mask(
                    predict_x1_from_v(x["local_latents"], v["local_latents"],
                                      ts["local_latents"][step_i]),
                    emit_masks["local_latents"]),
            }
            k_bb, k_lat, key = jax.random.split(key, 3)
            x["bb_ca"] = simulation_step(
                x["bb_ca"], v["bb_ca"], ts["bb_ca"][step_i],
                float(ts["bb_ca"][step_i + 1] - ts["bb_ca"][step_i]),
                float(gts["bb_ca"][step_i]), emit_masks["bb_ca"],
                self.step_params["bb_ca"], key=k_bb)
            x["local_latents"] = simulation_step(
                x["local_latents"], v["local_latents"], ts["local_latents"][step_i],
                float(ts["local_latents"][step_i + 1] - ts["local_latents"][step_i]),
                float(gts["local_latents"][step_i]), emit_masks["local_latents"],
                self.step_params["local_latents"], key=k_lat)
        return x, x_1_last, batch

    def _decode(self, batch, x_1) -> RFD4Result:
        """AE-decode clean predictions into a batched RFD4Result."""
        decode_fn = eqx.filter_jit(
            lambda m, d: m.decode(d, x_1["bb_ca"], x_1["local_latents"],
                                  aa_pred=d.aa_pred))
        dec = decode_fn(self.model, batch)
        return RFD4Result(x={}, x_1=x_1, data=batch, dec=dec)

    # ------------------------------------------------------------- search
    def search(self, data, key, *, nsteps: int = 60,
               recycle_schedule: tuple[tuple[int, int], ...] = THINKING_HIGH,
               resume_from: str | None = None, save_states_dir=None,
               batch_size: int | None = None) -> RFD4SearchResults:
        """Run the beam search.

        Args:
          data: RFD4Input or FlowBatch — `b` designs searched simultaneously
            (one independent beam per input, native-style). With b == 1 the
            beam carries no extra input axis beyond the single design.
          key: jax random key seeding all branching noise.
          nsteps / recycle_schedule: the sampling protocol (as for
            RFD4Sampler.sample).
          resume_from: path to a saved interior state (npz written via
            ``save_states_dir``) — restarts the beam at its saved step.
          save_states_dir: when given, the selected beam states are saved to
            ``<dir>/states_step{end}.npz`` after every interior checkpoint.
          batch_size: expected number of independent inputs (validated
            against `data`).

        Returns RFD4SearchResults ranked score-descending.
        """
        if hasattr(data, "to_batch"):
            data = data.to_batch()
        b = int(np.asarray(data.padding_mask).shape[0])
        if batch_size is not None and batch_size != b:
            raise ValueError(f"batch_size={batch_size} but the input batch "
                             f"holds {b} design(s)")
        n = int(np.asarray(data.padding_mask).shape[1])
        widths = self.beam_width
        ts, gts = self._schedules(nsteps)
        depths = self._depths(recycle_schedule, nsteps)
        checkpoint_steps = sorted(self.checkpoints)
        if max(checkpoint_steps) > nsteps:
            raise ValueError(f"checkpoint step {max(checkpoint_steps)} > "
                             f"nsteps {nsteps}")

        from .nn.transformer import compute_flow_emit_masks

        # ---- init: beam_width interleaved replicas of the input batch
        beam_data = _tile_beam(data, widths, b)
        if resume_from is not None:
            x, cursor = _load_state(resume_from, n=n, beam_width=widths, b=b,
                                    nsteps=nsteps)
            if cursor not in self.checkpoints:
                raise ValueError(f"resume: saved step {cursor} is not a "
                                 "configured checkpoint step")
        else:
            k_init, key = jax.random.split(key)
            emit_masks = compute_flow_emit_masks(beam_data)
            x = {
                "bb_ca": sample_noise(k_init, (b * widths,), n,
                                      self.matcher_cfgs["bb_ca"],
                                      emit_masks["bb_ca"]),
                "local_latents": sample_noise(k_init, (b * widths,), n,
                                              self.matcher_cfgs["local_latents"],
                                              emit_masks["local_latents"]),
            }
            cursor = 0

        alive = np.ones(b, bool)                 # per-input continuation
        survivor = np.zeros((b, widths), bool)   # final survivors (end step)
        last_score = np.zeros((b, widths), np.float32)
        # retained candidates: (source batched results, candidate index,
        # score, metrics source tree or None, is_intermediate)
        kept: list[tuple[RFD4Result, int, float, dict | None, bool]] = []
        sel_x1_fin = None                        # clean preds of an end checkpoint
        x_1_fin = None                           # clean preds of the final segment

        seg_ends = [s for s in checkpoint_steps if s > cursor] + [nsteps]
        for seg_end in seg_ends:
            seg_end = int(min(seg_end, nsteps))
            is_checkpoint = seg_end in self.checkpoints
            ckpt = self.checkpoints.get(seg_end, SearchCheckpoint())
            n_branch = int(ckpt.n_branch) if (is_checkpoint and
                                              ckpt.n_branch is not None) \
                else self.n_branch
            emit_masks = compute_flow_emit_masks(beam_data)

            if is_checkpoint:
                # ---- branch: n_branch children per (input, beam slot);
                # pool order: index p = branch_i * W + replica_r
                children_x, children_x1, children_batch = [], [], []
                for r in range(widths):
                    replica_x = {dm: v[r * b:(r + 1) * b] for dm, v in x.items()}
                    for _br in range(n_branch):
                        k_branch, key = jax.random.split(key)
                        cx, cx1, cbatch = self._partial_sim(
                            _take_batch(beam_data, r * b, (r + 1) * b),
                            replica_x, emit_masks=emit_masks, ts=ts, gts=gts,
                            start=cursor, end=seg_end, depths=depths,
                            key=k_branch)
                        children_x.append(cx)
                        children_x1.append(cx1)
                        children_batch.append(cbatch)
                pool_x = {dm: jnp.concatenate([c[dm] for c in children_x], axis=0)
                          for dm in x}
                pool_x1 = {dm: jnp.concatenate([c[dm] for c in children_x1], axis=0)
                           for dm in x}
                pool_batch = jax.tree.map(
                    lambda *ls: (ls[0] if getattr(ls[0], "ndim", 1) == 0
                                 else jnp.concatenate(ls, axis=0)), *children_batch)

                # ---- score: rollout to t=1 + decode (default), or the
                # children's own clean predictions (use_clean_prediction)
                if ckpt.use_clean_prediction:
                    results = self._decode(pool_batch, pool_x1)
                else:
                    roll_x, roll_x1, _rb = self._partial_sim(
                        pool_batch, pool_x, emit_masks=emit_masks, ts=ts,
                        gts=gts, start=seg_end, end=nsteps, depths=depths,
                        key=key)
                    results = self._decode(pool_batch, roll_x1)

                # ---- evaluate the candidate pool (unbatched evaluate, per
                # example, stacked — or the user's batch_evaluate wrapper)
                eval_fn = ckpt.evaluate if ckpt.evaluate is not None \
                    else self.evaluate
                batched_fn = (self.batch_evaluate if self.batch_evaluate
                              is not None else batch_evaluate_default)(eval_fn)
                out = batched_fn(results)
                missing = [k for k in ("score", "pass", "stop") if k not in out]
                if missing:
                    raise ValueError(
                        f"the batched evaluation is missing key(s) {missing} "
                        "(needs 'score', 'pass', 'stop'; 'metrics' optional)")
                rewards = np.asarray(out["score"], np.float32).reshape(-1)
                passes = np.asarray(out["pass"], bool).reshape(-1)
                stops = np.asarray(out["stop"], bool).reshape(-1)
                metrics_tree = out.get("metrics") or {}
                pool_size = int(rewards.size)
                if pool_size != n_branch * widths * b:
                    raise ValueError(
                        f"the batched evaluation returned {pool_size} "
                        f"candidates; expected {n_branch * widths * b}")
                if not np.isfinite(rewards).all():
                    raise ValueError(
                        "non-finite score(s) from the evaluate callable — the "
                        "native search aborts on -inf rewards; map them to a "
                        "finite value inside the callable")

                # ---- retain intermediates: scored candidates with pass=True
                for j in range(pool_size):
                    if passes[j]:
                        kept.append((results, j, float(rewards[j]),
                                     metrics_tree, True))

                # ---- per-input top-beam_width selection (stable, desc);
                # rewards reshaped (n_branch, widths, b): flat = br * W + r
                rew = rewards.reshape(n_branch, widths, b)
                pass_m = passes.reshape(n_branch, widths, b)
                stop_m = stops.reshape(n_branch, widths, b)
                sel_children = np.zeros((b, widths), np.int64)
                new_alive = np.zeros(b, bool)
                for inp in range(b):
                    if not alive[inp]:
                        continue
                    r_sel = rew[:, :, inp].reshape(-1)
                    order = np.argsort(-r_sel, kind="stable")[:widths]
                    for w_i, flat in enumerate(order):
                        br, rr = int(flat) // widths, int(flat) % widths
                        sel_children[inp, w_i] = br * widths + rr
                        last_score[inp, w_i] = r_sel[flat]
                        if seg_end == nsteps:
                            survivor[inp, w_i] = True
                    sel_pass = pass_m[:, :, inp].reshape(-1)[order]
                    sel_stop = stop_m[:, :, inp].reshape(-1)[order]
                    # continue iff any selected candidate passes and does not stop
                    new_alive[inp] = bool((sel_pass & ~sel_stop).any())

                # ---- prune the beam to the selected children, preserving the
                # [replica-major] layout: slot (inp, w) -> pool candidate
                # sel_children[inp, w] * b + inp
                pool_np = {dm: np.asarray(pool_x[dm]) for dm in x}
                new_x = {dm: np.zeros((b * widths,) + pool_np[dm].shape[1:],
                                      dtype=pool_np[dm].dtype) for dm in x}
                for inp in range(b):
                    for w_i in range(widths):
                        src = int(sel_children[inp, w_i]) * b + inp
                        dst = w_i * b + inp
                        for dm in x:
                            new_x[dm][dst] = pool_np[dm][src]
                x = {dm: jnp.asarray(new_x[dm]) for dm in x}
                if seg_end == nsteps:
                    # survivors of an end-of-schedule checkpoint ARE the
                    # finals: keep their clean predictions for the decode
                    pool_x1_np = {dm: np.asarray(pool_x1[dm]) for dm in pool_x1}
                    sel_x1_fin = {dm: np.zeros((b * widths,)
                                               + pool_x1_np[dm].shape[1:],
                                               dtype=pool_x1_np[dm].dtype)
                                  for dm in pool_x1_np}
                    for inp in range(b):
                        for w_i in range(widths):
                            src = int(sel_children[inp, w_i]) * b + inp
                            dst = w_i * b + inp
                            for dm in sel_x1_fin:
                                sel_x1_fin[dm][dst] = pool_x1_np[dm][src]
                    sel_x1_fin = {dm: jnp.asarray(v) for dm, v in sel_x1_fin.items()}
                if save_states_dir is not None and seg_end < nsteps:
                    _save_state(save_states_dir, x, seg_end, widths, n_branch,
                                nsteps, b)
                alive = new_alive
                cursor = seg_end
                if not alive.any():
                    break
                if (~alive).any():
                    keep_inp = np.flatnonzero(alive)
                    beam_data = _select_inputs(beam_data, keep_inp, b, widths)
                    x = {dm: _select_inputs(v, keep_inp, b, widths)
                         for dm, v in x.items()}
                    survivor = survivor[keep_inp]
                    last_score = last_score[keep_inp]
                    b = int(keep_inp.size)
            else:
                # segment without a checkpoint: plain continuation
                x, x_1_fin, beam_data = self._partial_sim(
                    beam_data, x, emit_masks=emit_masks, ts=ts, gts=gts,
                    start=cursor, end=seg_end, depths=depths, key=key)
                cursor = seg_end

        # ---- finish: decode the survivors. If the schedule ended on a
        # checkpoint, their clean predictions come from that checkpoint's
        # selected children; otherwise from the final plain continuation.
        finals: list[tuple[RFD4Result, int, float, dict | None, bool]] = []
        if alive.any():
            if cursor < nsteps:
                x, x_1_fin, beam_data = self._partial_sim(
                    beam_data, x, emit_masks=emit_masks, ts=ts, gts=gts,
                    start=cursor, end=nsteps, depths=depths, key=key)
            x_1_fin = sel_x1_fin if sel_x1_fin is not None else x_1_fin
            if x_1_fin is None:
                raise RuntimeError(
                    "no clean predictions available for the final decode")
            live_idx = [inp * widths + w_i for inp in range(b)
                        for w_i in range(widths) if survivor[inp, w_i]]
            if live_idx:
                live_batch = _take_batch(beam_data, 0, 0, indices=live_idx)
                live_x1 = {dm: jnp.concatenate(
                    [x_1_fin[dm][i:i + 1] for i in live_idx], axis=0)
                    for dm in x}
                fin_results = self._decode(live_batch, live_x1)
                for j, idx in enumerate(live_idx):
                    inp, w_i = divmod(idx, widths)
                    finals.append((fin_results, j, float(last_score[inp, w_i]),
                                   None, False))
        if not kept and not finals:
            raise RuntimeError("no surviving candidates — every input stopped "
                               "early and no scored candidate passed")
        entries = kept + finals
        entries.sort(key=lambda e: -float(e[2]))
        # group the sorted entries by their batched source for the gather
        pairs: list[tuple[RFD4Result, list[int]]] = []
        metric_pairs: list[tuple[dict, list[int]]] = []
        scores, is_inter = [], []
        for source, j, score, mtree, inter in entries:
            if pairs and pairs[-1][0] is source:
                pairs[-1][1].append(j)
                metric_pairs[-1][1].append(j)
            else:
                pairs.append((source, [j]))
                metric_pairs.append((mtree or {}, [j]))
            scores.append(score)
            is_inter.append(inter)
        return RFD4SearchResults(
            results=_stack_results(pairs),
            score=jnp.asarray(scores, np.float32),
            metrics=_stack_metrics_tree(metric_pairs),
            is_intermediate=jnp.asarray(is_inter, bool))
