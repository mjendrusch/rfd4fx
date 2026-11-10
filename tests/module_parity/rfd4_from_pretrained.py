"""RFD4.from_pretrained tests: pure-jax checkpoint loading (MS11).

Two-file convention (<base>.eqx + <base>.skeleton.pkl), local-cache
resolution, HF-model download hook (monkeypatched — no network), and
parameter parity against the torch conversion path.
"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pytest

from conftest import CACHE, REV, SNAPSHOT, release_bundle_available  # noqa: E402

pytestmark = pytest.mark.skipif(not release_bundle_available(),
                                reason="release bundle not in the local HF cache")


def _tree_paths(module):
    """(path-tuple, array) for every array leaf of an eqx module."""
    import equinox as eqx

    out = []

    def walk(node, path):
        if eqx.is_array(node):
            out.append((path, node))
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, path + (k,))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, path + (i,))
        else:
            for f in getattr(node, "__dataclass_fields__", None) or []:
                walk(getattr(node, f), path + (f,))

    walk(module, ())
    assert len(out) > 0
    return out


def _assert_same(a, b):
    pa, pb = _tree_paths(a), _tree_paths(b)
    assert [p for p, _ in pa] == [p for p, _ in pb]
    for (_, xa), (_, xb) in zip(pa, pb):
        assert xa.shape == xb.shape and xa.dtype == xb.dtype
        assert np.array_equal(np.asarray(xa), np.asarray(xb))


def test_from_pretrained_path_and_default():
    from rfd4fx.model import RFD4, load_model

    m = RFD4.from_pretrained(path="rfd4_release")
    assert isinstance(m, RFD4)
    _assert_same(m, load_model("rfd4_release"))
    # model= names a local checkpoint when hf= is absent
    _assert_same(RFD4.from_pretrained(model="rfd4_release"), m)
    # default (no args) = the same local release checkpoint
    _assert_same(RFD4.from_pretrained(), m)


def test_from_pretrained_source_validation():
    from rfd4fx.model import RFD4

    # path is exclusive
    with pytest.raises(ValueError):
        RFD4.from_pretrained(path="rfd4_release", model="other")
    with pytest.raises(ValueError):
        RFD4.from_pretrained(path="rfd4_release", hf="org/repo")
    # hf requires a checkpoint name
    with pytest.raises(ValueError):
        RFD4.from_pretrained(hf="org/repo")


def test_from_pretrained_model_download(monkeypatch, tmp_path):
    """model= + hf= downloads checkpoints/<model>.{eqx,skeleton.pkl} from the
    repo (multiple checkpoints side by side); monkeypatched — no network."""
    import huggingface_hub

    from rfd4fx.model import RFD4, _model_cache_dir

    release = os.path.join(_model_cache_dir(), "rfd4_release")

    def fake_download(repo_id, filename):
        assert filename.startswith("checkpoints/")
        name = filename.split("/", 1)[1]
        # pretend the repo holds the release files as checkpoints/release.*
        return f"{os.path.join(_model_cache_dir(), name)}"

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    m = RFD4.from_pretrained(model="rfd4_release", hf="fake-org/fake-jax-model")
    assert isinstance(m, RFD4)
    from rfd4fx.model import load_model
    _assert_same(m, load_model("rfd4_release"))
    # the downloaded files are cached under the cache hub dir
    hub_dir = os.path.join(_model_cache_dir(), "hub", "fake-org/fake-jax-model",
                           "checkpoints")
    assert os.path.exists(os.path.join(hub_dir, "rfd4_release.eqx"))
    assert os.path.exists(os.path.join(hub_dir, "rfd4_release.skeleton.pkl"))
    # a second call uses the cache (download not invoked again)
    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename: calls.append(filename) or "")
    m2 = RFD4.from_pretrained(model="rfd4_release", hf="fake-org/fake-jax-model")
    assert calls == []
    _assert_same(m2, m)
    shutil.rmtree(os.path.dirname(os.path.dirname(hub_dir)), ignore_errors=True)


def test_from_pretrained_matches_torch_conversion():
    """The pure-jax checkpoint is parameter-identical to the torch
    conversion path (translate round-trip through the new API)."""
    import torch

    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.model import RFD4

    model = Proteina.from_pretrained(SNAPSHOT, revision=REV, cache_dir=CACHE)
    model.eval()
    m_torch = RFD4.from_torch(model)
    m_jax = RFD4.from_pretrained(path="rfd4_release")
    _assert_same(m_jax, m_torch)
    del torch
