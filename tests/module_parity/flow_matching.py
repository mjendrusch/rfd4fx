"""Flow-matching parity: pure math (interpolant, v<->x1/score, TSR, schedules,
gt, loss) is compared numerically against torch; RNG-driven branches (SDE noise,
stochastic centering) are verified structurally/distributionally since torch and
JAX RNG streams legitimately differ."""

from conftest import make_pair  # noqa: F401  (registers converters)

import numpy as np
import jax
import jax.numpy as jnp
import torch

from rfproteina.flow_matching.rdn_flow_matcher import (
    RDNFlowMatcher as TorchRDN,
    compute_tsr_ratio as torch_tsr,
    score_to_vf as torch_s2v,
    vf_to_score as torch_v2s,
)
from rfproteina.flow_matching.product_space_flow_matcher import (
    get_gt as torch_get_gt,
    get_schedule as torch_schedule,
    get_schedule_tsr_safe as torch_schedule_tsr,
)
from rfd4fx.flow_matching import (
    MatcherConfig,
    StepParams,
    compute_tsr_ratio,
    force_zero_com,
    get_gt,
    get_schedule,
    get_schedule_tsr_safe,
    interpolate,
    predict_x1_from_v,
    score_to_vf,
    simulation_step,
    vf_to_score,
)


def _rel_err(a, e):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(e))))


def test_conversions():
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 0)
        x_t = jax.random.normal(key, (2, 16, 3))
        v = jax.random.normal(key, (2, 16, 3))
        t = jax.random.uniform(key, (2,), minval=0.05, maxval=0.95)
        e1 = torch_v2s(torch.from_numpy(np.asarray(x_t)), torch.from_numpy(np.asarray(v)), torch.from_numpy(np.asarray(t)))
        e2 = torch_s2v(torch.from_numpy(np.asarray(x_t)), torch.from_numpy(np.asarray(e1.numpy())), torch.from_numpy(np.asarray(t)))
        errs.append(_rel_err(vf_to_score(x_t, v, t), e1.numpy()))
        errs.append(_rel_err(score_to_vf(x_t, jnp.asarray(e1.numpy()), t), e2.numpy()))
        errs.append(_rel_err(predict_x1_from_v(x_t, v, t),
                             (torch.from_numpy(np.asarray(x_t)) + (1 - torch.from_numpy(np.asarray(t)))[:, None, None] * torch.from_numpy(np.asarray(v))).numpy()))
    print(f"v<->score/x1 conversions: max {_rel_err(errs, errs):.1e} (max trial err {max(errs):.3e})")
    assert max(errs) < 5e-5


def test_tsr_ratio():
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 1)
        t = jax.random.uniform(key, (2,), maxval=0.98)
        e = torch_tsr(torch.from_numpy(np.asarray(t)), torch.tensor(16.0), torch.tensor(3.0))
        errs.append(_rel_err(compute_tsr_ratio(t, 16.0, 3.0), e.numpy()))
    print(f"tsr_ratio: max {max(errs):.3e}")
    assert max(errs) < 5e-5


def test_schedules():
    for mode, p in (("log", 2.0), ("power", 2.0), ("uniform", None)):
        for tsr_safe in (False, True):
            fn_j = get_schedule_tsr_safe if tsr_safe else get_schedule
            fn_t = torch_schedule_tsr if tsr_safe else torch_schedule
            jt = fn_j(mode, 60, p1=p, tail_steps=10, tail_step_multiplier=2)
            tt = fn_t(mode, 60, p1=p, tail_steps=10, tail_step_multiplier=2)
            d = _rel_err(jt, tt.numpy())
            print(f"schedule {mode} tsr_safe={tsr_safe}: max diff {d:.3e}")
            assert d < 1e-6
    for mode, p in (("1/t", 1.0), ("tan", 1.0), ("ode", 1.0)):
        t = jnp.linspace(0.01, 0.95, 60)
        jg = get_gt(t, mode, p)
        tg = torch_get_gt(torch.from_numpy(np.asarray(t)), mode, p)
        d = _rel_err(jg, tg.numpy())
        print(f"gt {mode}: max diff {d:.3e}")
        assert d < 1e-6


def test_interpolate_and_centering():
    # matcher without stochastic centering: numeric parity
    cfg_none = MatcherConfig(zero_com_noise=False, dim=3, stochastic_centering_mode="none")
    torch_none = TorchRDN(zero_com_noise=False, dim=3, stochastic_centering_mode="none")
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 2)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x0 = jax.random.normal(k1, (2, 16, 3))
        x1 = jax.random.normal(k2, (2, 16, 3))
        t = jax.random.uniform(k3, (2,))
        em = jax.random.bernoulli(k4, 0.8, (2, 16))
        e = torch_none.interpolate(
            torch.from_numpy(np.asarray(x0)), torch.from_numpy(np.asarray(x1)),
            torch.from_numpy(np.asarray(t)), emit_mask=torch.from_numpy(np.asarray(em)),
        )
        errs.append(_rel_err(interpolate(x0, x1, t, cfg_none, em), e.numpy()))
    print(f"interpolate (no centering): max {max(errs):.3e}")
    assert max(errs) < 5e-5


def test_simulation_step_vf():
    """vf mode: deterministic, numeric parity."""
    params = StepParams(sampling_mode="vf")
    torch_matcher = TorchRDN(dim=3)
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 3)
        k1, k2, k3 = jax.random.split(key, 3)
        x_t = jax.random.normal(k1, (2, 16, 3))
        v = jax.random.normal(k2, (2, 16, 3))
        t = jnp.asarray(0.3 + 0.1 * jax.random.uniform(k3))
        em = jnp.ones((2, 16), dtype=bool)
        nn_out = {"v": torch.from_numpy(np.asarray(v))}
        e = torch_matcher.simulation_step(
            torch.from_numpy(np.asarray(x_t)), nn_out, torch.full((2,), float(t)), dt=0.1, gt=0.0,
            emit_mask=torch.from_numpy(np.asarray(em)),
            simulation_step_params={"sampling_mode": "vf", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
                                    "t_lim_ode_below": 0.02, "center_every_step": False},
        )
        errs.append(_rel_err(simulation_step(x_t, v, t, 0.1, 0.0, em, params, key=k3), e.numpy()))
    print(f"simulation_step vf: max {max(errs):.3e}")
    assert max(errs) < 5e-5


def test_simulation_step_deterministic_branches():
    """sc (t > t_lim_ode) and vf_tsr (t outside window): no RNG draw, numeric parity."""
    torch_matcher = TorchRDN(dim=3)
    ssp_sc = {"sampling_mode": "sc", "sc_scale_noise": 1.0, "t_lim_ode": 0.95,
              "t_lim_ode_below": 0.02, "center_every_step": False}
    ssp_tsr = {**ssp_sc, "sampling_mode": "vf_tsr", "tsr_k": 16.0, "tsr_sigma": 3.0}
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 4)
        k1, k2, k3, k4 = jax.random.split(key, 4)
        x_t = jax.random.normal(k1, (2, 16, 3))
        v = jax.random.normal(k2, (2, 16, 3))
        em = jnp.ones((2, 16), dtype=bool)
        # sc at t=0.97 > t_lim_ode: deterministic low-temp ODE branch
        t = jnp.asarray(0.97)
        e = torch_matcher.simulation_step(torch.from_numpy(np.asarray(x_t)),
                                          {"v": torch.from_numpy(np.asarray(v))},
                                          torch.full((2,), 0.97), dt=0.01, gt=0.5,
                                          emit_mask=torch.from_numpy(np.asarray(em)),
                                          simulation_step_params=ssp_sc)
        errs.append(_rel_err(simulation_step(x_t, v, t, 0.01, 0.5, em, StepParams(sampling_mode="sc"), key=k3), e.numpy()))
        # vf_tsr at t=0.98 > t_lim: deterministic branch
        t_tsr = jnp.asarray(0.98)
        e = torch_matcher.simulation_step(torch.from_numpy(np.asarray(x_t)),
                                          {"v": torch.from_numpy(np.asarray(v))},
                                          torch.full((2,), 0.98), dt=0.01, gt=0.5,
                                          emit_mask=torch.from_numpy(np.asarray(em)),
                                          simulation_step_params=ssp_tsr)
        errs.append(_rel_err(simulation_step(x_t, v, t_tsr, 0.01, 0.5, em,
                                             StepParams(sampling_mode="vf_tsr", tsr_k=16.0, tsr_sigma=3.0), key=k4),
                             e.numpy()))
    print(f"simulation_step deterministic branches: max {max(errs):.3e}")
    assert max(errs) < 5e-5


def test_stochastic_centering_structure():
    """per_group x_0 centering: same-group tokens move together, distinct groups
    get distinct offsets, coord-conditioned and non-emitted tokens never move."""
    cfg = MatcherConfig(zero_com_noise=False, dim=3, stochastic_centering_mode="x_0",
                        stochastic_centering_scale=0.2, stochastic_centering_scope="per_group")
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    x = jax.random.normal(k1, (2, 16, 3))
    em = jnp.ones((2, 16), dtype=bool)
    em = em.at[0, 15].set(False)
    coord_cond = jnp.zeros((2, 16), dtype=bool).at[1, 0].set(True)
    groups = jnp.concatenate([jnp.zeros(8, jnp.int32), jnp.ones(8, jnp.int32)])[None].repeat(2, 0)

    out = cfg.add_stochastic_noise(x, em, coord_cond, groups, k2)
    moved = np.asarray(out) - np.asarray(x)
    # non-emitted token zeroed (torch: offset then emit-multiply)
    assert np.asarray(out)[0, 15].max() == 0.0
    # coord-conditioned but emitted token: no offset (torch keeps it in place)
    assert moved[1, 0].max() == 0.0
    # same group (tokens 0..7, batch 0) share one offset
    offs = np.asarray(out)[0, :8] - np.asarray(x)[0, :8]
    assert np.allclose(offs, offs[0][None], atol=1e-12)
    # distinct groups get distinct offsets
    offs2 = np.asarray(out)[0, 8:] - np.asarray(x)[0, 8:]
    assert not np.allclose(offs[0], offs2[0], atol=1e-6)
    # offset scale ~ 0.2 * N(0,1)
    assert 0.05 < float(np.abs(offs).mean()) < 1.0
    print("stochastic centering structure: OK")


def test_zero_com():
    cfg = MatcherConfig(zero_com_noise=True, dim=3)
    errs = []
    for trial in range(100):
        key = jax.random.fold_in(jax.random.key(trial), 5)
        x = jax.random.normal(key, (2, 16, 3)) * 5
        torch_matcher = TorchRDN(zero_com_noise=True, dim=3)
        # torch _force_zero_com without mask
        e = torch_matcher._force_zero_com(torch.from_numpy(np.asarray(x)))
        errs.append(_rel_err(force_zero_com(x), e.numpy()))
        em = jnp.ones((2, 16), dtype=bool).at[0, :8].set(False)
        e = torch_matcher._force_zero_com(torch.from_numpy(np.asarray(x)),
                                          emit_mask=torch.from_numpy(np.asarray(em)))
        errs.append(_rel_err(force_zero_com(x, em), e.numpy()))
    print(f"force_zero_com: max {max(errs):.3e}")
    assert max(errs) < 5e-5
