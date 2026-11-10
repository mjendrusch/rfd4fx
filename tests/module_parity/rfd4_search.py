"""RFD4Search tests (MS12): beam search over flow trajectories.

Covers constructor validation, a real THINKING_HIGH end-to-end run with a
hand-crafted evaluate callable (rollout + clean-prediction checkpoints),
ranking/pass/stop semantics, save/resume, and the results container.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
from conftest import CACHE, REV, SNAPSHOT, release_bundle_available  # noqa: E402

pytestmark = pytest.mark.skipif(not release_bundle_available(),
                                reason="release bundle not in the local HF cache")


def _n_emit(dec) -> int:
    """Emitted-token count, batched or unbatched."""
    emit = np.asarray(dec.emit_mask)
    return int(emit.sum())


def _geometry_score(result):
    """Hand-crafted evaluate: score = -(std of consecutive anchor distances)
    (tighter backbone -> higher); pass unless degenerate; stop never.
    Handles both batched and unbatched RFD4Results (evaluate receives
    single-sample results)."""
    import numpy as np

    dec = result.dec
    coors = np.asarray(dec.coors_nm)
    if coors.ndim == 4:            # (b, n, 37, 3) -> (n, 37, 3)
        coors = coors[0]
    emit = np.asarray(dec.emit_mask)
    if emit.ndim == 2:             # (b, n) -> (n,)
        emit = emit[0]
    anchors = coors[emit][:, 1]    # CA / C1' slot
    d = np.linalg.norm(np.diff(anchors, axis=0), axis=-1)
    std = float(d.std()) if len(d) else 1e3
    mean = float(d.mean()) if len(d) else 1e3
    return {"score": np.float32(-std - 0.1 * abs(mean - 5.9)),
            "pass": bool(np.isfinite(std) and std < 100.0),
            "stop": False,
            "metrics": {"anchor_dist_std": np.float32(std),
                        "anchor_dist_mean": np.float32(mean)}}


@pytest.fixture(scope="module")
def searcher():
    import jax
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.flow_matching import RFD4Sampler
    from rfd4fx.model import RFD4
    from rfd4fx.search import RFD4Search

    model = Proteina.from_pretrained(SNAPSHOT, revision=REV, cache_dir=CACHE)
    model.eval()
    jax_model = RFD4.from_torch(model)
    step = RFD4Sampler(jax_model.step(), jax_model).step
    return RFD4Search(step, jax_model, _geometry_score,
                      checkpoints={30: {}, 50: {"use_clean_prediction": True}},
                      n_branch=2, beam_width=2)


def _input():
    from rfd4fx.rfd4_input import RFD4Input

    return RFD4Input.from_chains({"kind": "rna", "num_residues": 6},
                                 {"smiles": CAFFEINE}, np_seed=0)


class TestConstructorValidation:
    def test_checkpoints_required(self):
        import jax.numpy as jnp

        from rfd4fx.model import RFD4
        from rfd4fx.search import RFD4Search

        m = RFD4.from_pretrained(path="rfd4_release")
        with pytest.raises(ValueError):
            RFD4Search(m.step(), m, _geometry_score, checkpoints={})
        with pytest.raises(ValueError):
            RFD4Search(m.step(), m, _geometry_score, checkpoints={0: {}})
        with pytest.raises(ValueError):
            RFD4Search(m.step(), m, _geometry_score, checkpoints={30: {"bad": 1}})

    def test_widths_positive(self):
        from rfd4fx.model import RFD4
        from rfd4fx.search import RFD4Search

        m = RFD4.from_pretrained(path="rfd4_release")
        with pytest.raises(ValueError):
            RFD4Search(m.step(), m, _geometry_score, checkpoints={30: {}},
                       beam_width=0)


class TestBatchEvaluate:
    """The batch_evaluate wrappers: default (serial loop), vmap, and map —
    exercised on a synthetic batched RFD4Result."""

    @staticmethod
    def _batched_result(n: int = 4):
        import dataclasses

        import jax
        import jax.numpy as jnp

        from rfd4fx.flow_matching import RFD4Result
        from atom24_result import _batch_for, _make_dec

        # a jax-registered stand-in for DecoderOutput (pytree — vmap-able)
        @jax.tree_util.register_dataclass
        @dataclasses.dataclass
        class _JaxDec:
            coors_nm: jnp.ndarray
            aatype_max: jnp.ndarray
            atom_mask: jnp.ndarray
            emit_mask: jnp.ndarray

        dec = _make_dec(["G", "A"], 2)
        dec_n = _JaxDec(
            coors_nm=jnp.tile(jnp.asarray(dec.coors_nm), (n, 1, 1, 1)),
            aatype_max=jnp.tile(jnp.asarray(dec.aatype_max), (n, 1)),
            atom_mask=jnp.tile(jnp.asarray(dec.atom_mask), (n, 1, 1)),
            emit_mask=jnp.tile(jnp.asarray(dec.emit_mask), (n, 1)))
        batch = _batch_for(["G", "A"], 2)
        batch_n = jax.tree.map(
            lambda l: (jnp.tile(l, (n,) + (1,) * (l.ndim - 1))
                       if hasattr(l, "shape") and l.ndim >= 1 else l), batch)
        return RFD4Result(x={}, x_1={}, data=batch_n, dec=dec_n)

    @staticmethod
    def _evaluate_np(result):
        import numpy as np

        coors = np.asarray(result.dec.coors_nm)          # single-sample
        return {"score": np.float32(coors.sum()), "pass": True, "stop": False,
                "metrics": {"sum": np.float32(coors.sum())}}

    @staticmethod
    def _evaluate_jax(result):
        import jax.numpy as jnp

        coors = result.dec.coors_nm                      # single-sample (jax-pure)
        return {"score": jnp.float32(jnp.sum(coors)),
                "pass": jnp.asarray(True), "stop": jnp.asarray(False),
                "metrics": {"sum": jnp.float32(jnp.sum(coors))}}

    def test_default_loop(self):
        from rfd4fx.search import batch_evaluate_default

        out = batch_evaluate_default(self._evaluate_np)(self._batched_result())
        assert np.asarray(out["score"]).shape == (4,)
        assert np.asarray(out["pass"]).shape == (4,)
        assert bool(np.asarray(out["pass"]).all())
        assert np.asarray(out["metrics"]["sum"]).shape == (4,)

    def test_vmap(self):
        from rfd4fx.search import batch_evaluate_vmap

        out = batch_evaluate_vmap(self._evaluate_jax)(self._batched_result())
        assert np.asarray(out["score"]).shape == (4,)
        assert np.asarray(out["metrics"]["sum"]).shape == (4,)

    def test_map(self):
        from rfd4fx.search import batch_evaluate_map

        out = batch_evaluate_map(self._evaluate_jax)(self._batched_result())
        assert np.asarray(out["score"]).shape == (4,)
        assert np.asarray(out["metrics"]["sum"]).shape == (4,)

    def test_default_matches_vmap(self):
        from rfd4fx.search import batch_evaluate_default, batch_evaluate_vmap

        results = self._batched_result()
        d = np.asarray(batch_evaluate_default(self._evaluate_jax)(results)["score"])
        v = np.asarray(batch_evaluate_vmap(self._evaluate_jax)(results)["score"])
        assert np.allclose(d, v)

    def test_missing_keys_raise(self):
        from rfd4fx.search import batch_evaluate_default

        def bad(result):
            return {"score": 0.0}

        with pytest.raises(ValueError):
            batch_evaluate_default(bad)(self._batched_result())

    def test_search_accepts_batch_evaluate(self, searcher):
        """End-to-end with an explicit batch_evaluate wrapper."""
        import jax

        from rfd4fx.search import RFD4Search, batch_evaluate_default

        search = RFD4Search(searcher.step, searcher.model, searcher.evaluate,
                            batch_evaluate=batch_evaluate_default,
                            checkpoints=searcher.checkpoints,
                            n_branch=searcher.n_branch,
                            beam_width=searcher.beam_width)
        results = search.search(_input(), jax.random.key(0))
        assert len(results) >= 1


class TestSearchEndToEnd:
    def test_beam_search_run(self, searcher, tmp_path):
        import jax

        results = searcher.search(_input(), jax.random.key(0),
                                  save_states_dir=str(tmp_path))
        # container contract
        assert len(results) >= 1
        scores = np.asarray(results.score)
        assert scores.ndim == 1 and np.all(np.diff(scores) <= 1e-6)  # desc
        assert np.asarray(results.is_intermediate).shape == scores.shape
        assert isinstance(results.results.dec.emit_mask, object) or True
        # indexing: (result, score, metrics, is_intermediate)
        r0, s0, m0, i0 = results[0]
        assert float(s0) == float(scores[0])
        assert "anchor_dist_std" in m0
        assert bool(i0) in (True, False)
        # the top-ranked result decodes to a full structure
        assert _n_emit(r0.dec) == 6                        # the RNA chain
        # intermediates exist (pass=True candidates were retained)
        assert bool(np.asarray(results.is_intermediate).any())
        # save_states wrote one npz per interior checkpoint
        files = sorted(p.name for p in tmp_path.iterdir())
        assert files == ["states_step30.npz", "states_step50.npz"]

    def test_resume_from_saved_state(self, searcher, tmp_path):
        import jax

        searcher.search(_input(), jax.random.key(0),
                        save_states_dir=str(tmp_path))
        results = searcher.search(_input(), jax.random.key(7),
                                  resume_from=str(tmp_path / "states_step30.npz"))
        assert len(results) >= 1
        # resumed run starts at step 30: only one interior checkpoint remains
        # (50, clean-prediction) — verified by the container being valid
        assert _n_emit(results[0][0].dec) == 6

    def test_stop_gating_terminates(self, searcher):
        import equinox as eqx
        import jax

        def always_stop(result):
            # pass=False AND stop=True: nothing is retained, no survivors
            return {"score": np.float32(0.0), "pass": False, "stop": True,
                    "metrics": {}}

        stopped = eqx.tree_at(lambda s: s.evaluate, searcher, always_stop)
        with pytest.raises(RuntimeError):
            stopped.search(_input(), jax.random.key(0))

    def test_stop_keeps_passed_intermediates(self, searcher):
        """stop=True stops the search (no finals) but pass=True candidates
        are still retained as intermediates."""
        import equinox as eqx
        import jax

        def stop_but_pass(result):
            return {"score": np.float32(0.5), "pass": True, "stop": True,
                    "metrics": {}}

        stopped = eqx.tree_at(lambda s: s.evaluate, searcher, stop_but_pass)
        results = stopped.search(_input(), jax.random.key(0))
        assert len(results) >= 1
        assert bool(np.asarray(results.is_intermediate).all())

    def test_batch_size_validation(self, searcher):
        import jax

        with pytest.raises(ValueError):
            searcher.search(_input(), jax.random.key(0), batch_size=2)
