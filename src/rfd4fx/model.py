"""The RFD4 model: trunk denoiser, autoencoder decoder, and checkpoint loading.

``RFD4.from_pretrained()`` loads a converted (torch-free) checkpoint;
``RFD4.step()`` is the jitted flow-matching update used by the sampler."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp

from .flow_matching import (MatcherConfig, StepParams, THINKING_HIGH,
                            compute_tsr_ratio, full_simulation, get_gt,
                            get_schedule, get_schedule_tsr_safe,
                            predict_x1_from_v, vf_to_score)
from .flow_matching import RFD4Sampler, _mask, _set_x_t, guided_velocity  # noqa: F401
from .nn.transformer import DecoderTransformer, LocalLatentsTransformer, compute_flow_emit_masks
from .types import FlowBatch

# Release sampler configuration (configs/generation/inference_base.yaml + the
# bundle's product_flowmatcher section).
RECYCLE_SCHEDULE = [(5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)]  # [width, depth]
N_STEPS = 60

MATCHER_CFGS = {
    "bb_ca": MatcherConfig(zero_com_noise=True, guidance_enabled=True, dim=3,
                           stochastic_centering_mode="x_0", stochastic_centering_scale=0.2,
                           stochastic_centering_scope="per_group"),
    "local_latents": MatcherConfig(zero_com_noise=False, guidance_enabled=True, dim=32,
                                   stochastic_centering_mode="none", stochastic_centering_scale=None),
}
STEP_PARAMS = {
    "bb_ca": StepParams(sampling_mode="vf_tsr", sc_scale_noise=1.0, t_lim_ode=0.95,
                        t_lim_ode_below=0.02, center_every_step=False, tsr_k=16.0, tsr_sigma=3.0),
    "local_latents": StepParams(sampling_mode="vf_tsr", sc_scale_noise=1.0, t_lim_ode=0.95,
                                t_lim_ode_below=0.02, center_every_step=False, tsr_k=4.0, tsr_sigma=12.0),
}
SAMPLER_ARGS = {
    "bb_ca": {"schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
              "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None}},
    "local_latents": {"schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
                      "gt": {"mode": "tan", "p": 1.0, "clamp_val": None}},
}


def recycle_depths(nsteps: int = N_STEPS) -> tuple[int, ...]:
    depths: list[int] = []
    for width, depth in RECYCLE_SCHEDULE:
        depths.extend([depth] * width)
    if len(depths) != nsteps:
        raise ValueError(f"recycle schedule expands to {len(depths)} steps; expected {nsteps}")
    return tuple(depths)


class RFD4(eqx.Module):
    """Converted RFD4-Proteina: flow denoiser + autoencoder decoder."""

    nn: LocalLatentsTransformer
    decoder: DecoderTransformer

    @classmethod
    def from_torch(cls, proteina):
        from .backend import from_torch

        return cls(nn=from_torch(proteina.nn), decoder=from_torch(proteina.autoencoder.decoder))

    @classmethod
    def from_pretrained(cls, path: str | None = None, model: str | None = None,
                        hf: str | None = None):
        """Load a pure-jax (Equinox) RFD4 checkpoint — torch-free.

        Exactly one source must be given. With no arguments, the default
        local checkpoint ``"rfd4_release"`` (produced by
        ``scripts/translate_rfd4.py``) is loaded.

        Args:
            path: Base path of a local two-file checkpoint (``<base>.eqx``
                arrays + ``<base>.skeleton.pkl`` module skeleton), resolved
                like :func:`load_model` (explicit path first, then
                ``$RFD4FX_MODEL_CACHE``).
            model: Checkpoint name. Resolved locally like ``path`` when
                ``hf`` is not given; with ``hf``, the name of the checkpoint
                to fetch from ``checkpoints/`` inside the HF repo.
            hf: Hugging Face repo id. Downloads
                ``checkpoints/<model>.eqx`` + ``checkpoints/<model>.skeleton.pkl``
                into the local cache and loads them. Requires ``model``.

        Returns:
            An :class:`RFD4` module with the loaded parameters.

        Raises:
            ValueError: If more than one source (or ``hf`` without ``model``)
                is given, or the checkpoint cannot be resolved.
        """
        if path is not None and (model is not None or hf is not None):
            raise ValueError(
                "from_pretrained: pass exactly one source — path= (local "
                "two-file checkpoint) or model= (checkpoint name, local or "
                "within hf='s checkpoints/)")
        if hf is not None and model is None:
            raise ValueError(
                "from_pretrained: hf= requires model= (the checkpoint name "
                "to load from the repo's checkpoints/ directory)")
        if hf is not None:
            model = _download_jax_checkpoint(hf, model)
        return load_model(path if path is not None
                          else (model if model is not None else "rfd4_release"))

    def step(self):
        """Return the default step callable ``step(model, key, data) -> data``.

        One denoising pass: runs the trunk on ``data`` (``data.n_recycle`` as
        a traced depth) and writes the predictions into the returned batch:
        ``aa_pred`` (token-type-restricted sequence-readout argmax), ``x_pred``
        (anchor clean prediction, emit-masked), ``z_pred`` (denoised local
        latents, emit-masked) and per-modality ``v_pred``/``s_pred`` (raw
        velocity / probability-flow score).

        Returns:
            A callable ``step(model, key, data) -> FlowBatch``. Clean-prediction
            conversions follow the native flow-matcher semantics for
            ``t in (0, 1)``: ``x1 = x_t + (1 - t) * v`` and
            ``s = (t * v - x_t) / (1 - t)``.
        """

        def step(model: "RFD4", key, data: FlowBatch) -> FlowBatch:
            out = model.nn(data, n_recycle=data.n_recycle, key=key)
            v = {"bb_ca": out.bb_ca, "local_latents": out.local_latents}
            x_t = {"bb_ca": data.x_t_bb_ca, "local_latents": data.x_t_local_latents}
            t = {"bb_ca": data.t_bb_ca, "local_latents": data.t_local_latents}
            emit = compute_flow_emit_masks(data)
            x1 = {dm: _mask(predict_x1_from_v(x_t[dm], v[dm], t[dm]), emit[dm])
                  for dm in v}
            s = {dm: vf_to_score(x_t[dm], v[dm], t[dm]) for dm in v}
            aa_pred = _restricted_sequence_argmax(
                out.seq_logits, data.token_type, model.decoder.allowed_residue_types
            )
            return eqx.tree_at(
                lambda b: (b.aa_pred, b.x_pred, b.z_pred, b.v_pred, b.s_pred),
                data,
                (aa_pred, x1["bb_ca"], x1["local_latents"], v, s),
                is_leaf=lambda x: x is None,
            )

        return step

    def decode(self, data: FlowBatch, x1_bb, z1, aa_pred=None):
        """Autoencoder decode of the clean predictions (release
        condition_decoder_on_sequence_prediction: fill C_SEQ on emitted,
        sequence-unconditioned tokens with ``aa_pred`` first)."""
        seq_cond = data.cond_seq
        if seq_cond is None:
            seq_cond = jnp.zeros_like(data.padding_mask, dtype=jnp.int32)  # <M>
        if aa_pred is not None:
            seq_cond = jnp.where(
                (seq_cond == 0) & data.padding_mask, aa_pred, seq_cond)
        return self.decoder(
            eqx.tree_at(lambda b: b.cond_seq,
                        eqx.tree_at(lambda b: b.ca_coors_nm,
                                    eqx.tree_at(lambda b: b.z_latent, data, z1,
                                                is_leaf=lambda l: l is None),
                                    x1_bb, is_leaf=lambda l: l is None),
                        seq_cond, is_leaf=lambda l: l is None),
        )

    def generate(
        self,
        batch: FlowBatch,
        *,
        key,
        nsteps: int = N_STEPS,
        guidance_w: float = 1.0,
        jit_model: bool = True,
    ):
        """Unconditional/conditioned generation via :class:`RFD4Sampler`
        (back-compat wrapper: the release `medium/front_loaded` recycle
        schedule). Returns (x, x_1_pred, DecoderOutput)."""
        sampler = RFD4Sampler(self.step(), self,
                              matcher_cfgs=MATCHER_CFGS, step_params=STEP_PARAMS,
                              sampler_args=SAMPLER_ARGS, guidance_w=guidance_w,
                              jit_step=jit_model)
        x, x_1, data, _ = sampler.sample(
            batch, key, nsteps=nsteps, recycle_schedule=RECYCLE_SCHEDULE,
            decode=False)
        x_1_pred = {"bb_ca": x_1["bb_ca"], "local_latents": x_1["local_latents"],
                    "aa_pred": data.aa_pred, "v_pred": data.v_pred,
                    "s_pred": data.s_pred}
        dec = self.decode(batch, x_1["bb_ca"], x_1["local_latents"],
                          aa_pred=data.aa_pred)
        return x, x_1_pred, dec


def _restricted_sequence_argmax(seq_logits, token_type, allowed_residue_types):
    """Token-type-restricted argmax of the trunk sequence readout."""
    allowed = allowed_residue_types[token_type]
    logits = jnp.where(allowed, seq_logits, jnp.finfo(seq_logits.dtype).min)
    return jnp.argmax(logits, axis=-1)


ProteinaJAX = RFD4  # back-compat alias (one transition cycle)


def _download_jax_checkpoint(repo_id: str, name: str) -> str:
    """Download a two-file jax checkpoint (``checkpoints/<name>.eqx`` +
    ``checkpoints/<name>.skeleton.pkl``) from a Hugging Face model repo and
    return the local base path for :func:`load_model`.

    A repo may hold multiple checkpoints side by side under ``checkpoints/``;
    `name` selects one. Files are fetched through ``huggingface_hub``
    (respecting HF_HOME / HF_TOKEN) into the local cache; no torch is
    involved at any point.
    """
    import os
    import shutil

    from huggingface_hub import hf_hub_download

    base_dir = os.path.join(_model_cache_dir(), "hub", repo_id, "checkpoints")
    os.makedirs(base_dir, exist_ok=True)
    base = os.path.join(base_dir, name)
    for ext in ("eqx", "skeleton.pkl"):
        local = f"{base}.{ext}"
        if os.path.exists(local):
            continue
        try:
            got = hf_hub_download(repo_id=repo_id,
                                  filename=f"checkpoints/{name}.{ext}")
        except Exception as e:
            raise FileNotFoundError(
                f"no jax checkpoint {name!r} for HF model {repo_id!r}: "
                f"expected 'checkpoints/{name}.{ext}' in the repo. Jax "
                f"checkpoints are uploaded separately from the torch "
                f"weights; until then, translate locally with "
                f"scripts/translate_rfd4.py and pass path= instead. "
                f"Original error: {e}") from e
        shutil.copyfile(got, local)
    return base


def save_model(model, path):
    """Two-file serialization (pytorch-to-equinox skill): .eqx arrays + skeleton."""
    import pickle

    with open(f"{path}.eqx", "wb") as f:
        eqx.tree_serialise_leaves(f, model)
    skeleton = jax.tree.map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype) if eqx.is_array(x) else x,
        model,
        is_leaf=eqx.is_array,
    )
    with open(f"{path}.skeleton.pkl", "wb") as f:
        pickle.dump(skeleton, f)


def _model_cache_dir() -> str:
    import os

    return os.path.expanduser(
        os.environ.get("RFD4FX_MODEL_CACHE", "~/.rfd4fx/models"))


def resolve_model_path(name_or_path) -> str:
    """Resolve a model name/path to the base path of a two-file model
    (``<base>.eqx`` + ``<base>.skeleton.pkl``): an explicit path first, then
    the local cache directory (``$RFD4FX_MODEL_CACHE``)."""
    import os

    path = os.path.expanduser(str(name_or_path))
    if os.path.exists(f"{path}.eqx"):
        return path
    base = os.path.join(_model_cache_dir(), str(name_or_path))
    if os.path.exists(f"{base}.eqx"):
        return base
    raise FileNotFoundError(
        f"no model {name_or_path!r}: looked for {path}.eqx and {base}.eqx "
        f"(convert one with scripts/translate_rfd4.py)")


def load_model(name_or_path):
    """Load an Equinox RFD4 model — completely torch-free
    (skeleton pickle + .eqx arrays; protenij/esmjfold2 convention)."""
    import pickle

    base = resolve_model_path(name_or_path)
    with open(f"{base}.skeleton.pkl", "rb") as f:
        skeleton = pickle.load(f)
    with open(f"{base}.eqx", "rb") as f:
        return eqx.tree_deserialise_leaves(f, skeleton)
