"""RFD4Spec parity: direct de-novo featurizer (pure numpy, biotite/atomworks/torch-free).

Gates:
1. `RFD4Spec().add_chain(50).to_batch()` reproduces the native torch featurization
   archives (tests/fixtures-level ground truth in test-featuresets/rfd4_native.npz)
   field-exactly — 15 model-relevant fields.
2. The direct featurizer matches the parity-verified pipeline path on multi-chain
   and conditioned specs (chain/molecule-iid permutations drawn in the pipeline's
   numpy-stream order).
3. Validation behavior: scalar-only on de-novo chains, SSEnum ranges, unknown
   condition names rejected, fixed sequences/atom-level conditions deferred.
"""

import dataclasses
import pathlib

import numpy as np
import pytest

REPO = pathlib.Path("/Users/mjendrusch/repos/rfd4j")
NATIVE = REPO / "test-featuresets/rfd4_native.npz"
pytestmark = pytest.mark.skipif(not NATIVE.exists(), reason="native feature archive not present")

import jax.numpy as jnp  # noqa: E402

import rfd4fx.spec as sp  # noqa: E402
from rfd4fx.annotated_cif import annotated_cif_to_batch  # noqa: E402
from rfd4fx.types import FlowBatch  # noqa: E402

ARCHIVE_FIELDS = {
    "token_type": "token_type", "atom37_mask": "atom37_mask", "coords": "coords_nm",
    "condition_sequence_1_residue": "cond_seq",
    "token_bond_path_length": "token_bond_path_length",
    "canonical_polymer_ordering": "canonical_polymer_ordering",
    "bond_mask": "bond_mask", "is_in_context_conditioning": "is_in_context_conditioning",
    "chirality": "chirality", "chiral_frame_atoms": "chiral_frame_atoms",
    "molecule_entity_id": "molecule_entity_id", "element": "element",
    "chain_iid": "chain_iid", "molecule_iid": "molecule_iid",
    "condition_ChainCentroid_1_atom": "chain_centroid",
}


def _assert_batch_field(t, j, key):
    t, j = np.asarray(t), np.asarray(j)
    if key == "coords":
        j = j * 10.0  # archive in Å, batch in nm
    if t.dtype == bool:
        assert (t == j).all(), key
    elif np.issubdtype(t.dtype, np.integer) or np.issubdtype(j.dtype, np.integer):
        assert (t.astype(np.int64) == j.astype(np.int64)).all(), key
    else:
        assert np.allclose(t, j, atol=1e-6, equal_nan=True), key


def test_direct_vs_native_archive():
    spec = sp.RFD4Spec()
    spec.add_chain(50)
    batch, coord_cond = spec.to_batch(np_seed=0)
    native = np.load(NATIVE)
    for key, field in ARCHIVE_FIELDS.items():
        _assert_batch_field(native[key], getattr(batch, field), key)


def _diff(a: FlowBatch, b: FlowBatch) -> list[str]:
    errs = []
    for f in [x.name for x in dataclasses.fields(FlowBatch)]:
        va, vb = getattr(a, f), getattr(b, f)
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            errs.append(f)
            continue
        va, vb = np.asarray(va), np.asarray(vb)
        if va.shape != vb.shape:
            errs.append(f)
            continue
        if va.dtype == bool:
            if not (va == vb).all():
                errs.append(f)
        elif np.issubdtype(va.dtype, np.integer) or np.issubdtype(vb.dtype, np.integer):
            if not (va.astype(np.int64) == vb.astype(np.int64)).all():
                errs.append(f)
        else:
            if (np.isnan(va) != np.isnan(vb)).any():
                errs.append(f)
            elif va.size and not np.allclose(va, vb, atol=1e-6, equal_nan=True):
                errs.append(f)
    return errs


def _pipeline_batch(spec, np_seed=0):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "spec.cif"
        path.write_text(spec.to_annotated_cif())
        return annotated_cif_to_batch(path, np_seed=np_seed)


@pytest.mark.parametrize(
    "build",
    [
        lambda s: s.add_chain(50),
        lambda s: (s.add_chain(20), s.add_dssp(chain=0, dssp='H')),
        lambda s: (s.add_chain(30), s.add_chain(40), s.add_dssp(chain=0, dssp='H'),
                   s.add_dssp(chain=1, dssp='E')),
        lambda s: (s.add_chain(10), s.add_chain(15), s.add_chain(12)),
        lambda s: (s.add_chain(8), s.add_chain(9), s.add_chain(7), s.add_chain(11)),
    ],
)
def test_direct_vs_pipeline(build):
    spec = sp.RFD4Spec()
    build(spec)
    b_direct, cc_direct = spec.to_batch(np_seed=0)
    b_pipe, cc_pipe = _pipeline_batch(spec, np_seed=0)
    assert _diff(b_direct, b_pipe) == []
    assert np.array_equal(np.asarray(cc_direct), np.asarray(cc_pipe))


def test_validation_behavior():
    spec = sp.RFD4Spec()
    spec.add_chain(10)
    with pytest.raises(ValueError):  # wrong per-residue length
        spec.add_dssp(chain=0, dssp=np.array([1, 2, 1, 0, 0, 0, 0, 0], dtype=np.int32))
    with pytest.raises(AssertionError):  # outside the SSEnum domain (per-residue)
        spec.add_dssp(chain=0, dssp=np.array([7] * 10, dtype=np.int32))
    with pytest.raises(AssertionError):  # outside the SSEnum domain (scalar)
        spec.add_dssp(chain=0, dssp=np.array([7]))
    with pytest.raises(ValueError):  # atom-level conditions need aa/where materialization
        spec.add_condition("C_RAS", 3, chain=0)
    with pytest.raises(NotImplementedError):  # fixed sequences deferred on add_chain
        spec.add_chain(5, sequence="AGLST")
    with pytest.raises(KeyError):  # unknown condition
        spec.add_condition("C_BOGUS", 1)
    spec.add_dssp(chain=0, dssp='H')
    assert spec.validate() == []


def test_add_dssp():
    """add_dssp: string → SSEnum ints (strict 3-state), arrays pass through,
    None clears, chain=None concatenates over chains in add order."""
    spec = sp.RFD4Spec()
    spec.add_chain(4)
    spec.add_chain(6)
    spec.add_dssp(dssp="HHELXXLLEE")          # whole system, concatenated
    spec.add_dssp(chain=1, dssp="HHEELL")     # per-chain replaces
    b, _ = spec.to_batch(np_seed=0)
    ss = np.asarray(b.secondary_structure)[0]
    assert ss[:4].tolist() == [1, 1, 2, 3]
    assert ss[4:].tolist() == [1, 1, 2, 2, 3, 3]
    with pytest.raises(ValueError):  # unsupported character (strict 3-state)
        spec.add_dssp(chain=0, dssp="HGBT")
    with pytest.raises(ValueError):  # wrong system length
        spec.add_dssp(dssp="HHH")
    with pytest.raises(ValueError):  # wrong per-chain length
        spec.add_dssp(chain=0, dssp="HH")
    spec.add_dssp(dssp=None)                  # clears the whole system
    b2, _ = spec.to_batch(np_seed=0)
    assert (np.asarray(b2.secondary_structure)[0] == 0).all()
    # int32 ndarray form
    spec.add_dssp(chain=0, dssp=np.array([0, 1, 2, 3], dtype=np.int32))
    b3, _ = spec.to_batch(np_seed=0)
    assert np.asarray(b3.secondary_structure)[0][:4].tolist() == [0, 1, 2, 3]


def test_per_residue_sequence_pos_and_atom_conditions():
    """The approved combination test: per-residue C_SEQ (add_sequence) +
    materialized slots with positions (add_pos) + an atom-level condition
    (add_hbond) on the materialized slots — compiled by the direct featurizer."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    spec = sp.RFD4Spec()
    spec.add_chain(6)
    spec.add_sequence("AGLSTP")                       # per-residue C_SEQ
    where = np.zeros((6, 37), dtype=bool)
    where[:, 0] = True                                # N slots everywhere
    where[2, 1] = True                                # + CA on residue 2
    pos = np.full((6, 37, 3), np.nan, np.float32)
    pos[:, 0] = np.arange(6 * 3, dtype=np.float32).reshape(6, 3)
    pos[2, 1] = [1.0, 1.0, 1.0]
    spec.add_pos(pos, where)
    hb = np.zeros(6, dtype=np.int32)
    hb[2] = 2                                         # ACCEPTOR on residue 2
    spec.add_hbond(hb, aa=np.array([0, 0, ENC.token_to_idx["GLY"], 0, 0, 0], np.int32),
                   where=where)
    b, cc = spec.to_batch(np_seed=0)
    # sequence conditioning: add_sequence sets AGLSTP → ALA,GLY,LEU,SER,THR,PRO;
    # the atom-level adder's `aa` then overrides the conditioned position 2 (GLY)
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC2
    expected = [ENC2.token_to_idx[n] for n in ("ALA", "GLY", "GLY", "SER", "THR", "PRO")]
    assert np.asarray(b.cond_seq)[0].tolist() == expected
    # materialized slots + positions
    m37 = np.asarray(b.atom37_mask)[0]
    assert m37[:, 0].all() and m37[2, 1] and m37[0, 1] == False
    coords = np.asarray(b.coords_nm)[0] * 10.0
    assert np.isfinite(coords[2, 1]).all()
    assert np.isfinite(coords[:, 0]).all()
    # coordinate conditioning on positioned atoms (C_CRD, group 0)
    assert cc[0].tolist() == [True] * 6
    assert (np.asarray(b.coord_groups)[0][:, 0] == 0).all()
    assert np.isfinite(np.asarray(b.cond_coords)[0][:, 0]).all()
    # hbond condition scattered into the materialized slots only
    hbond = np.asarray(b.hbond)[0]
    assert (hbond[:, 0] == [0, 0, 2, 0, 0, 0]).all()
    assert hbond[2, 1] == 2 and hbond.sum() == 4      # residue 2's two slots
    # element filled at materialized slots of sequence-conditioned tokens
    element = np.asarray(b.element)[0]
    assert element[:, 0].tolist() == [7, 7, 7, 7, 7, 7]     # N
    assert element[2, 1] == 6                                # CA → carbon
    # chain with conditioned sequence keeps its native entity
    assert np.asarray(b.molecule_entity_id)[0].tolist() == [0] * 6


def test_to_annotated_cif_rejects_per_residue():
    spec = sp.RFD4Spec()
    spec.add_chain(3)
    spec.add_dssp(chain=0, dssp=np.array([1, 2, 3], dtype=np.int32))
    with pytest.raises(NotImplementedError):
        spec.to_annotated_cif()
    spec2 = sp.RFD4Spec()
    spec2.add_chain(3)
    spec2.add_sequence("AAA")
    with pytest.raises(NotImplementedError):
        spec2.to_annotated_cif()


def test_atomized_chain_spec():
    """Explicit per-chain atomization: ATOMIZED tokens (token_type=3, C_SEQ=<A>,
    atom37 slot 1, no local latent, all-zero finite seq probabilities, no
    canonical linkage) — the programmatic counterpart of featurize_atoms'
    force_atomize / the reference AtomizeByCCDName."""
    spec = sp.RFD4Spec()
    spec.add_chain(10)
    spec.add_chain(3, atomize=True)
    batch, _ = spec.to_batch(np_seed=0)
    tt = np.asarray(batch.token_type)[0]
    atom = tt == 3
    assert atom.sum() == 3 and tt[:10].tolist() == [0] * 10
    assert (np.asarray(batch.cond_seq)[0][atom] == 32).all()  # <A>
    m37 = np.asarray(batch.atom37_mask)[0]
    assert m37[atom][:, 1].all() and m37[atom][:, [0] + list(range(2, 37))].sum() == 0
    assert (np.asarray(batch.supports_local_latent)[0][atom] == False).all()
    probs = np.asarray(batch.seq_probabilities)[0][atom]
    assert np.isfinite(probs).all() and (probs == 0).all()
    ordm = np.asarray(batch.canonical_polymer_ordering)[0]
    assert (ordm[atom, :] == 0).all() and (ordm[:, atom] == 0).all()
    # normal chain retains the verified conventions
    assert (np.asarray(batch.cond_seq)[0][:10] == 0).all()
    assert np.asarray(batch.supports_local_latent)[0][:10].all()
    # chain with the same build but no atomize flag: default behavior unchanged
    spec2 = sp.RFD4Spec()
    spec2.add_chain(10)
    spec2.add_chain(3)
    b2, _ = spec2.to_batch(np_seed=0)
    assert (np.asarray(b2.token_type)[0] == 0).all()
    assert np.asarray(b2.supports_local_latent)[0].all()


def test_unindexed_motif():
    """add_unindexed_motif: in-context conditioning tokens appended after the
    chains — is_in_context=True, supports_local=False, C_SEQ=restype, centered
    C_CRD at masked slots, explicit index group, no polymer linkage (canonical
    ordering / bond path 0), one private entity per motif block."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    spec = sp.RFD4Spec()
    spec.add_chain(5)
    pos = np.zeros((2, 37, 3), np.float32)
    pos[:, 0] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]     # N slots of two motif residues
    mask = np.zeros((2, 37), dtype=bool)
    mask[:, 0] = True
    mask[0, 1] = True                                   # residue 0 also has CA
    pos[0, 1] = [1.5, 0.5, 0.0]
    spec.add_unindexed_motif(pos, np.array([ENC.token_to_idx["CYS"]] * 2, np.int32),
                             mask, residue_index=np.array([0, 3], np.int32))
    b, cc = spec.to_batch(np_seed=0)
    assert np.asarray(b.padding_mask).shape == (1, 7)   # 5 chain + 2 motif tokens
    ic = np.asarray(b.is_in_context_conditioning)[0]
    sup = np.asarray(b.supports_local_latent)[0]
    assert ic.tolist() == [False] * 5 + [True, True]
    assert sup.tolist() == [True] * 5 + [False, False]
    assert np.asarray(b.cond_seq)[0][5:].tolist() == [ENC.token_to_idx["CYS"]] * 2
    m37 = np.asarray(b.atom37_mask)[0]
    assert m37[5, 0] and m37[5, 1] and m37[6, 0] and not m37[6, 1]
    assert np.asarray(cc)[0][5:].all() and not np.asarray(cc)[0][:5].any()
    assert np.asarray(b.coord_groups)[0][5:, 0].tolist() == [0, 0]
    # one shared FRESH index group for the whole motif (not used by the chains)
    ig = np.asarray(b.idx_group)[0]
    assert ig[5] == ig[6] and ig[5] not in ig[:5].tolist() and ig[5] > 0
    assert ig[:5].tolist() == [0] * 5
    # no linkage between the motif and the chains; within the motif, linkage and
    # sequence distance follow the relative residue_index distances: indices [0, 3]
    # → no ±1 linkage (gap), path = |0 - 3| = 3 (the C–[2–2]–C pattern)
    ordm = np.asarray(b.canonical_polymer_ordering)[0]
    assert (ordm[5:, :5] == 0).all() and (ordm[:5, 5:] == 0).all()
    assert ordm[6, 5] == 0 and ordm[5, 6] == 0      # index gap ≠ 1 → no linkage
    pl = np.asarray(b.token_bond_path_length)[0]
    assert (pl[5:, :5] == 0).all() and (pl[:5, 5:] == 0).all() and pl[6, 5] == 3
    # consecutive residue_index values → ±1 linkage, path 1
    spec1 = sp.RFD4Spec()
    spec1.add_chain(2)
    spec1.add_unindexed_motif(pos, np.array([ENC.token_to_idx["CYS"]] * 2, np.int32),
                              mask, residue_index=np.array([0, 1], np.int32))
    b1, _ = spec1.to_batch(np_seed=0)
    ordm1 = np.asarray(b1.canonical_polymer_ordering)[0]
    pl1 = np.asarray(b1.token_bond_path_length)[0]
    assert ordm1[3, 2] == 1 and ordm1[2, 3] == -1 and pl1[3, 2] == 1

    # full-matrix cross-reference vs the native featurization (see
    # test-featuresets/motif_bond_features/REPORT.md): for same-index-group
    # pairs, path == |Δresidue_index| exactly and ordering is ±1 exactly at
    # |Δresidue_index| == 1 (+1 at [downstream, upstream]); cross-group pairs
    # carry path 0; bond_mask stays empty (all polymer-linkage bonds suppressed)
    spec3 = sp.RFD4Spec()
    spec3.add_chain(3)
    pos3 = np.zeros((3, 37, 3), np.float32)
    mask3 = np.zeros((3, 37), dtype=bool)
    mask3[:, 0] = True
    spec3.add_unindexed_motif(pos3, np.array([ENC.token_to_idx["CYS"]] * 3, np.int32),
                              mask3, residue_index=np.array([0, 1, 2], np.int32))
    b3, _ = spec3.to_batch(np_seed=0)
    mot3 = slice(3, 6)
    ri3 = np.array([0, 1, 2])
    pl3 = np.asarray(b3.token_bond_path_length)[0][mot3, mot3]
    assert (pl3 == np.abs(ri3[:, None] - ri3[None, :])).all()
    ordm3 = np.asarray(b3.canonical_polymer_ordering)[0][mot3, mot3]
    expected_ord = np.where(
        np.abs(ri3[:, None] - ri3[None, :]) == 1,
        np.sign(ri3[:, None] - ri3[None, :]),  # +1 at [downstream, upstream]
        0)
    assert (ordm3 == expected_ord).all()
    assert not np.asarray(b3.bond_mask)[0].any()
    # residue_index=None → every motif residue its own fresh group, no linkage
    spec0 = sp.RFD4Spec()
    spec0.add_chain(2)
    spec0.add_unindexed_motif(pos, np.array([ENC.token_to_idx["CYS"]] * 2, np.int32),
                              mask)
    b0, _ = spec0.to_batch(np_seed=0)
    ig0 = np.asarray(b0.idx_group)[0]
    assert ig0[2] != ig0[3] and ig0[2] > 0 and ig0[3] > 0
    ordm0 = np.asarray(b0.canonical_polymer_ordering)[0]
    pl0 = np.asarray(b0.token_bond_path_length)[0]
    assert ordm0[3, 2] == 0 and pl0[3, 2] == 0
    # motif tokens: own iids/entity, chain untouched
    assert len(set(np.asarray(b.molecule_iid)[0][5:].tolist())) == 1
    assert np.asarray(b.molecule_entity_id)[0][5].item() not in \
        np.asarray(b.molecule_entity_id)[0][:5].tolist()
    # element at masked slots of the CYS motif tokens
    el = np.asarray(b.element)[0]
    assert el[5, 0] == 7 and el[5, 1] == 6 and el[6, 0] == 7
    # coordinates: ground truth + conditioning, centered jointly
    coords = np.asarray(b.coords_nm)[0] * 10.0
    assert np.isfinite(coords[5, 0]).all() and np.isfinite(coords[6, 0]).all()
    with pytest.raises(NotImplementedError):
        spec.to_annotated_cif()
    with pytest.raises(ValueError):  # bad restype
        spec.add_unindexed_motif(pos, np.array([999], np.int32), mask[:1])
    with pytest.raises(ValueError):  # bad shapes
        spec.add_unindexed_motif(np.zeros((2, 3)), np.array([1, 1], np.int32), mask)
    with pytest.raises(ValueError):  # residue_index length mismatch
        spec.add_unindexed_motif(pos, np.array([1, 1], np.int32), mask,
                                 residue_index=np.array([0], np.int32))


def test_indexed_motif():
    """add_indexed_motif: an overlay on the existing system — conditioned rows
    become fixed indexed motif residues (C_SEQ=restype, materialized slots with
    C_CRD, index group), unconditioned rows untouched; polymer semantics
    (ordering ±1, bond path, supports_local) preserved."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    spec = sp.RFD4Spec()
    spec.add_chain(4)
    pos = np.full((4, 37, 3), np.nan, np.float32)
    where = np.zeros((4, 37), dtype=bool)
    where[1, 0] = True                                   # N of residue 1 only
    where[2, 0] = True                                   # N of residue 2
    where[2, 1] = True                                   # + CA
    pos[1, 0] = [0.5, 0.5, 0.5]
    pos[2, 0] = [1.5, 0.5, 0.5]
    pos[2, 1] = [1.5, 1.5, 0.5]
    restype = np.zeros(4, dtype=np.int32)                # ignored at unconditioned rows
    restype[1] = ENC.token_to_idx["CYS"]
    restype[2] = ENC.token_to_idx["CYS"]
    spec.add_indexed_motif(pos, restype, where)
    b, cc = spec.to_batch(np_seed=0)
    assert np.asarray(b.padding_mask).shape == (1, 4)    # overlay, no new tokens
    seq = np.asarray(b.cond_seq)[0]
    assert seq.tolist() == [0, ENC.token_to_idx["CYS"], ENC.token_to_idx["CYS"], 0]
    m37 = np.asarray(b.atom37_mask)[0]
    assert m37[1, 0] and m37[2, 0] and m37[2, 1] and not m37[0, 0] and not m37[3, 0]
    assert np.asarray(cc)[0].tolist() == [False, True, True, False]
    # all conditioned residues share the underlying spec's index group (0)
    assert np.asarray(b.idx_group)[0].tolist() == [0, 0, 0, 0]
    # polymer semantics preserved (native convention: +1 at [downstream, upstream])
    ordm = np.asarray(b.canonical_polymer_ordering)[0]
    assert ordm[1, 0] == 1 and ordm[0, 1] == -1 and ordm[2, 1] == 1
    # full-matrix cross-reference vs the native featurization: same-group path
    # == |Δresidue_index| and ordering ±1 exactly on adjacent pairs
    pl = np.asarray(b.token_bond_path_length)[0]
    idx4 = np.arange(4)
    assert (pl == np.abs(idx4[:, None] - idx4[None, :])).all()
    expected_ord4 = np.where(
        np.abs(idx4[:, None] - idx4[None, :]) == 1,
        np.sign(idx4[:, None] - idx4[None, :]), 0)  # +1 at [downstream, upstream]
    assert (ordm == expected_ord4).all()
    assert not np.asarray(b.bond_mask)[0].any()
    assert np.asarray(b.supports_local_latent)[0].all()
    assert not np.asarray(b.is_in_context_conditioning)[0].any()
    # coordinates conditioned at the masked slots
    assert np.isfinite(np.asarray(b.cond_coords)[0][2, 1]).all()
    with pytest.raises(ValueError):  # wrong system size
        spec.add_indexed_motif(np.zeros((3, 37, 3), np.float32),
                               np.zeros(3, np.int32), np.zeros((3, 37), bool))
    with pytest.raises(NotImplementedError):
        spec.to_annotated_cif()


def test_secondary_structure_method_removed():
    """add_secondary_structure is removed; add_dssp is the SS API."""
    spec = sp.RFD4Spec()
    assert not hasattr(spec, "add_secondary_structure")
    spec.add_chain(3)
    spec.add_dssp(chain=0, dssp="HEE")
    b, _ = spec.to_batch(np_seed=0)
    assert np.asarray(b.secondary_structure)[0].tolist() == [1, 2, 2]
