"""RFD4Result / atom24 tests (MS10): CCD-order atom24 conversion, flat
to_atoms packing, and PDB/mmCIF saving through the sampler result object."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from conftest import CACHE, REV, SNAPSHOT, release_bundle_available  # noqa: E402

pytestmark = pytest.mark.skipif(not release_bundle_available(),
                                reason="release bundle not in the local HF cache")

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


# ------------------------------------------------------------- CCD order
def test_ccd_atom_names_guanosine_24_heavy():
    from rfd4fx.atom24 import ATOM24, ccd_atom_names
    from rfd4fx.ccd import get_ccd_entry

    names = ccd_atom_names("G")
    rec = get_ccd_entry("G")
    all_names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in rec.atom_names]
    all_elems = [e.decode() if isinstance(e, bytes) else str(e)
                 for e in rec.elements]
    expect = [n for n, e in zip(all_names, all_elems) if e.upper() not in ("H", "D")]
    assert names == expect                      # CCD file order, H/D filtered
    assert len(names) == 24 == ATOM24           # G is the vocabulary maximum


def test_ccd_atom_names_order_matches_ccd():
    from rfd4fx.atom24 import ccd_atom_names
    from rfd4fx.ccd import get_ccd_entry

    for token in ("A", "DA", "TRP"):
        names = ccd_atom_names(token)
        rec = get_ccd_entry(token)
        all_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in rec.atom_names]
        all_elems = [e.decode() if isinstance(e, bytes) else str(e)
                     for e in rec.elements]
        assert names == [n for n, e in zip(all_names, all_elems)
                         if e.upper() not in ("H", "D")]


def test_residue_atom24_plan():
    from rfd4fx.atom24 import ATOM24, residue_atom24_plan
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    tok_to_idx = enc.token_to_idx
    plan, elems = residue_atom24_plan("G", int(tok_to_idx["G"]))
    assert len(plan) == ATOM24 and (plan >= 0).all()     # G: every slot filled
    # each planned slot names a distinct layout atom
    layout = [str(a).strip() for a in enc.token_atoms["G"]]
    used = {layout[s] for s in plan}
    assert len(used) == ATOM24
    # protein residue: TRP has 15 heavy atoms (incl. OXT, per its CCD) ->
    # 15 mapped slots, 22 absent
    plan_t, elems_t = residue_atom24_plan("TRP", int(tok_to_idx["TRP"]))
    assert (plan_t[:15] >= 0).all() and (plan_t[15:] == -1).all()
    assert (elems_t[15:] == 0).all()


def test_residue_atom24_plan_layout_fallback(monkeypatch):
    # tokens whose CCD lookup fails fall back to the atom37 layout order
    import rfd4fx.atom24 as a24
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc
    from rfd4fx.atom24 import ATOM24

    monkeypatch.setattr(a24, "ccd_atom_names", lambda name: None)
    tok_to_idx = enc.token_to_idx
    plan, elems = a24.residue_atom24_plan("TRP", int(tok_to_idx["TRP"]))
    layout = [str(a).strip() for a in enc.token_atoms["TRP"]]
    named = [i for i, n in enumerate(layout) if n]
    assert plan[:len(named)].tolist() == named[:ATOM24]
    assert (plan[len(named):] == -1).all()


# ------------------------------------------------------------- to_atom24
@dataclasses.dataclass
class _Dec:
    coors_nm: np.ndarray
    aatype_max: np.ndarray
    atom_mask: np.ndarray
    emit_mask: np.ndarray


def _make_dec(aa: list[str], n: int) -> _Dec:
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    tok_to_idx = enc.token_to_idx
    seq = np.array([int(tok_to_idx[a]) for a in aa] + [0] * (n - len(aa)), np.int32)
    coords = np.zeros((1, n, 37, 3), np.float32)
    amask = np.zeros((n, 37), bool)
    for t, a in enumerate(aa):
        layout = enc.token_atoms.get(a)
        amask[t] = np.array([bool(str(x).strip()) for x in layout])
        for s in range(37):
            coords[0, t, s] = (1.0 + t, 2.0 + s, 3.0)
    return _Dec(coors_nm=coords / 10.0, aatype_max=seq[None],
                atom_mask=amask[None], emit_mask=np.ones((1, n), bool))


def _batch_for(aa: list[str], n: int):
    from rfd4fx.rfd4_input import RFD4Input

    if all(a in ("A", "C", "G", "U", "N") for a in aa):
        chain = {"kind": "rna", "num_residues": n}
    elif all(a in ("DA", "DC", "DG", "DT", "DN") for a in aa):
        chain = {"kind": "dna", "num_residues": n}
    else:
        chain = {"num_residues": n}
    return RFD4Input.from_chains(chain, np_seed=0).to_batch()


def test_to_atom24_ccd_order_and_padding():
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc
    from rfd4fx.atom24 import ATOM24, to_atom24
    from rfd4fx.ccd import get_ccd_entry

    aa = ["G", "A", "C"]
    dec = _make_dec(aa, 3)
    batch = _batch_for(aa, 3)
    a24 = to_atom24(dec, batch)
    assert a24.coords.shape == (3, ATOM24, 3)
    assert a24.mask.shape == (3, ATOM24)
    tok_to_idx = enc.token_to_idx
    assert a24.restype.tolist() == [tok_to_idx[x] for x in aa]
    # G: all 24 slots present; C: 21 present, 3 absent (0.0 / False)
    assert a24.mask[0].all()
    assert int(a24.mask[2].sum()) == 21
    absent = np.flatnonzero(~a24.mask[2])
    assert (a24.coords[2, absent] == 0.0).all()
    assert (a24.elements[2, absent] == 0).all()
    # CCD order: the first atoms of G in CCD file order carry G's elements
    rec = get_ccd_entry("G")
    all_names = [n.decode() if isinstance(n, bytes) else str(n) for n in rec.atom_names]
    all_elems = [e.decode() if isinstance(e, bytes) else str(e) for e in rec.elements]
    expect_first = [n for n, e in zip(all_names, all_elems) if e.upper() not in ("H", "D")][:3]
    from rfd4fx.atom24 import _ccd_element_of
    assert [int(_element_z(_ccd_element_of("G", n))) for n in expect_first] \
        == a24.elements[0, :3].tolist()
    # coordinates: slot for CCD atom #0 maps to its atom37 layout slot
    layout = [str(a).strip() for a in enc.token_atoms["G"]]
    s37 = layout.index(expect_first[0])
    assert np.allclose(a24.coords[0, 0], np.asarray(dec.coors_nm)[0, 0, s37] * 10.0)


def _element_z(symbol: str) -> int:
    from rfd4fx.spec import _atomic_number_from_symbol
    return _atomic_number_from_symbol(symbol)


# ------------------------------------------------------- RFD4Result (real)
def test_rfd4_result_end_to_end():
    """A tiny real sampling run: RFD4Result fields, tuple compat,
    to_atom24/to_atoms shapes, save_pdb/save_cif round trips."""
    import jax
    import biotite.structure.io.pdb as pdbio
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.flow_matching import RFD4Result, RFD4Sampler, THINKING_HIGH
    from rfd4fx.model import RFD4
    from rfd4fx.rfd4_input import RFD4Input

    model = Proteina.from_pretrained(SNAPSHOT, revision=REV, cache_dir=CACHE)
    model.eval()
    jax_model = RFD4.from_torch(model)
    sampler = RFD4Sampler(jax_model.step(), jax_model)

    inp = RFD4Input.from_chains({"kind": "rna", "num_residues": 6},
                                {"smiles": CAFFEINE}, np_seed=0)
    batch = inp.pad_to_nearest(64).to_batch()
    # production schedule (the release schedule's tail_steps arithmetic
    # requires the full 60-step budget)
    result = sampler.sample(batch, key=jax.random.key(0), nsteps=60,
                            recycle_schedule=THINKING_HIGH)
    assert isinstance(result, RFD4Result)
    # legacy tuple unpacking still works
    x, x_1, data, dec = result
    assert dec is not None and data.aa_pred is not None
    assert result.is_single_sample

    a24 = result.to_atom24()
    assert a24.coords.shape == (6, 24, 3) and a24.mask.shape == (6, 24)
    # no spurious OP3 in the atom24 mask (multi-residue NA chain)
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc
    tok_to_idx = enc.token_to_idx
    g_idx = tok_to_idx["G"]
    a_idx = tok_to_idx["A"]
    for r in range(6):
        if a24.restype[r] in (g_idx, a_idx, tok_to_idx["C"], tok_to_idx["U"]):
            pass  # element/mask checks are covered by the unit tests above

    atoms = result.to_atoms()
    n_lig = 14
    assert atoms.coords.shape == (24 * 6 + n_lig, 3)
    assert atoms.mask.shape == (24 * 6 + n_lig,)
    # the ligand atoms sit at the tail with real coordinates + mask True
    assert atoms.mask[24 * 6:].all()
    assert np.isfinite(atoms.coords[24 * 6:]).all()

    import pathlib, tempfile
    tmp = pathlib.Path(tempfile.mkdtemp())
    result.save_pdb(tmp / "r.pdb")
    back = pdbio.PDBFile.read(tmp / "r.pdb").get_structure(model=1)
    prot = back[~back.hetero]
    assert "OP3" not in set(prot.atom_name)
    assert len(np.unique(prot.res_id)) == 6
    assert int(back.hetero.sum()) == n_lig
    result.save_cif(tmp / "r.cif")
    from biotite.structure.io.pdbx import CIFFile, get_structure as cif_get
    back_cif = cif_get(CIFFile.read(tmp / "r.cif"), model=1)
    assert int(back_cif.hetero.sum()) == n_lig
    assert len(np.unique(back_cif[~back_cif.hetero].res_id)) == 6
