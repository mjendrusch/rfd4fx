"""RFD4.step + RFD4Sampler vs the reference full_simulation path.

The sampler must reproduce full_simulation + decode bit-exactly on identical
keys: same trunk call (module as a traced eqx arg), same key discipline, same
integration. Runs on the release weights (same env as end_to_end_sampling).
"""

import pathlib

import numpy as np
import pytest

REPO = pathlib.Path("/Users/mjendrusch/repos/rfd4j")
SNAPSHOT = (REPO / ".hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/"
            "a06e2509c7033b010fc43614e9fd840303588502")


def _close(a, b, atol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    if not np.array_equal(a, b):
        np.testing.assert_allclose(a, b, atol=atol, rtol=1e-6)


def test_sampler_matches_full_simulation(tmp_path):
    if not SNAPSHOT.exists():
        pytest.skip("release snapshot not present")

    import jax
    import jax.numpy as jnp

    from rfd4fx.model import RFD4, MATCHER_CFGS, STEP_PARAMS, SAMPLER_ARGS
    from rfd4fx.flow_matching import RFD4Sampler, full_simulation
    from rfd4fx.rfd4_input import RFD4Input

    import torch
    from rfd4_proteina import Proteina
    import rfd4fx.convert  # noqa: F401

    model = Proteina.from_pretrained(SNAPSHOT)
    model.eval()
    jax_model = RFD4.from_torch(model)

    # torch-free serialization round trip with the real weights
    from rfd4fx.model import load_model, save_model

    ckpt = tmp_path / "release"
    save_model(jax_model, ckpt)
    reloaded = load_model(ckpt)
    import equinox as eqx

    assert eqx.tree_equal(jax_model, reloaded)

    inp = RFD4Input.from_chains(24, np_seed=0)
    batch = inp.pad_to_nearest(64).to_batch()
    key = jax.random.key(0)
    nsteps = 20  # >= tail_steps(10) + 1 for the release schedule math
    schedule = ((10, 2), (10, 1))  # expands to 20 steps

    # reference integration (the generate core)
    x_ref, x1_ref, _ = full_simulation(
        jax_model.nn, batch, key=key, nsteps=nsteps,
        recycle_depths=tuple([2] * 10 + [1] * 10),
        matcher_cfgs=MATCHER_CFGS, step_params=STEP_PARAMS,
        sampler_args=SAMPLER_ARGS, n=64)

    # sampler over the jitted step
    sampler = RFD4Sampler(jax_model.step(), jax_model)
    x, x_1, data, dec = sampler.sample(batch, key, nsteps=nsteps,
                                       recycle_schedule=schedule)

    for dm in x_ref:
        _close(x_ref[dm], x[dm])
        _close(x1_ref[dm], x_1[dm])

    # the step's per-step predictions are present and self-consistent
    assert data.aa_pred is not None and data.x_pred is not None
    assert data.z_pred is not None and data.v_pred is not None
    assert data.s_pred is not None and data.n_recycle is not None
    t = float(np.asarray(data.t_bb_ca)[0])
    x1_check = np.asarray(data.x_t_bb_ca) + (1.0 - t) * np.asarray(data.v_pred["bb_ca"])
    np.testing.assert_allclose(np.asarray(data.x_pred), x1_check, atol=1e-5)
    s_check = (t * np.asarray(data.v_pred["bb_ca"]) - np.asarray(data.x_t_bb_ca)) / (1.0 - t)
    np.testing.assert_allclose(np.asarray(data.s_pred["bb_ca"]), s_check, atol=1e-4)

    # decode: the sampler's decode equals RFD4.decode on the same predictions
    dec_manual = jax_model.decode(data, x_1["bb_ca"], x_1["local_latents"],
                                  aa_pred=data.aa_pred)
    _close(np.asarray(dec.coors_nm), np.asarray(dec_manual.coors_nm))
    _close(np.asarray(dec.aatype_max), np.asarray(dec_manual.aatype_max))
