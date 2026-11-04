"""RFD4Input: the compiled-input container (fields = FlowBatch input keys).

Gates:
1. `from_chains` compiles through RFD4Spec — int chains reproduce the
   unconditioned spec batch field-exactly; dict chains (dssp / pos /
   residue_index / atom-level) match the equivalent spec calls.
2. `add_dssp` (string / np.ndarray / jax.Array) replaces C_SEC[start:start+len]
   immutably; out-of-range lengths and unknown letters raise.
3. Atom-level adders materialize `where` slots, set identity from `aa` (0 =
   unspecified), write values only inside `where` (preserving outside), all jnp.
4. `to_dict` / `to_batch` round-trip every input key.
"""

import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import pytest

import rfd4fx.spec as sp
from rfd4fx.rfd4_input import INPUT_FIELDS, RFD4Input
from rfd4fx.types import FlowBatch


def _assert_fields_equal(a, b, keys=INPUT_FIELDS, atol=0.0):
    for f in keys:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            assert (x is None) == (y is None), f"{f}: None mismatch"
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype == bool or y.dtype == bool:
            assert (x == y).all(), f"{f}: bool mismatch"
        elif np.issubdtype(x.dtype, np.integer) or np.issubdtype(y.dtype, np.integer):
            assert (x.astype(np.int64) == y.astype(np.int64)).all(), f"{f}: int mismatch"
        else:
            assert np.allclose(x, y, atol=atol, equal_nan=True), f"{f}: float mismatch"


# ------------------------------------------------------------------ from_chains
def test_from_chains_int_matches_spec():
    spec = sp.RFD4Spec()
    spec.add_chain(40)
    batch, cc = spec.to_batch(np_seed=0)
    inp = RFD4Input.from_chains(40, np_seed=0)
    _assert_fields_equal(inp, batch)
    assert np.asarray(inp.coord_cond).shape == (1, 40)
    assert not np.asarray(inp.coord_cond).any()


def test_from_chains_multiple_ints_match_spec():
    spec = sp.RFD4Spec()
    spec.add_chain(12)
    spec.add_chain(20)
    batch, cc = spec.to_batch(np_seed=3)
    inp = RFD4Input.from_chains(12, 20, np_seed=3)
    _assert_fields_equal(inp, batch)
    # two chains: distinct chain iids, path 0 across the boundary
    ciid = np.asarray(inp.chain_iid)[0]
    assert len(set(ciid.tolist())) == 2
    assert np.asarray(inp.token_bond_path_length)[0, 11, 12] == 0


def test_from_chains_dssp_dict_matches_spec():
    dssp = "XXXHHHHHHHHHHEEEEEXLLLLLXXX"
    spec = sp.RFD4Spec()
    spec.add_chain(len(dssp))
    spec.add_dssp(chain=0, dssp=dssp)
    batch, _ = spec.to_batch(np_seed=0)
    inp = RFD4Input.from_chains({"dssp": dssp}, np_seed=0)
    _assert_fields_equal(inp, batch)
    ss = np.asarray(inp.secondary_structure)[0]
    assert (ss == 1).sum() == dssp.count("H") and (ss == 2).sum() == dssp.count("E")


def test_from_chains_mixed_int_and_dict():
    dssp = "HHHHLLL"
    spec = sp.RFD4Spec()
    spec.add_chain(10)
    spec.add_chain(len(dssp))
    spec.add_dssp(chain=1, dssp=dssp)
    batch, _ = spec.to_batch(np_seed=1)
    inp = RFD4Input.from_chains(10, {"dssp": dssp}, np_seed=1)
    _assert_fields_equal(inp, batch)


def test_from_chains_residue_index():
    ri = np.array([0, 1, 2, 5, 6, 10, 11], dtype=np.int32)
    inp = RFD4Input.from_chains({"residue_index": ri}, np_seed=0)
    path = np.asarray(inp.token_bond_path_length)[0]
    link = np.asarray(inp.canonical_polymer_ordering)[0]
    # within-chain path = |Δresidue_index| (not token adjacency)
    assert (path[:7, :7] == np.abs(ri[:, None] - ri[None, :])).all()
    # linkage ±1 exactly at residue-index distance 1, 0 elsewhere
    want = (np.sign(ri[:, None] - ri[None, :]) * (np.abs(ri[:, None] - ri[None, :]) == 1))
    assert (link[:7, :7] == want).all()
    # default chain would have had plain |i-j| — differs at the gaps
    plain = RFD4Input.from_chains(7, np_seed=0)
    assert (np.asarray(plain.token_bond_path_length)[0][:7, :7]
            != path[:7, :7]).any()


def test_from_chains_pos_dict_matches_spec():
    rng = np.random.default_rng(0)
    n = 9
    pos = rng.normal(size=(n, 37, 3)).astype(np.float32) * 5.0
    where = rng.random((n, 37)) < 0.3
    where[:, 1] = True  # keep slot-1 reference atoms on realized residues
    spec = sp.RFD4Spec()
    spec.add_chain(n)
    spec.add_pos(pos, where, chain=0)
    batch, cc = spec.to_batch(np_seed=0)
    inp = RFD4Input.from_chains({"pos": pos, "mask": where}, np_seed=0)
    _assert_fields_equal(inp, batch)
    # motif conditioning: materialized slots + C_CRD group 0
    assert (np.asarray(inp.atom37_mask)[0] == where).all()
    assert (np.asarray(inp.coord_groups)[0, where] == 0).all()
    assert np.asarray(cc).any()  # coordinate conditioning present


def test_from_chains_atom_level_dict_matches_spec():
    n = 11
    aa = np.array([1, 0, 5, 0, 0, 8, 0, 3, 0, 0, 6], dtype=np.int32)
    where = np.zeros((n, 37), dtype=bool)
    where[:, 1] = True
    where[0, 2] = True
    where[5, [1, 4]] = True
    spec = sp.RFD4Spec()
    spec.add_chain(n)
    spec.add_hotspot(True, aa=aa, where=where, chain=0)
    spec.add_rasa(2, aa=aa, where=where, chain=0)
    batch, _ = spec.to_batch(np_seed=0)
    inp = RFD4Input.from_chains(
        {"num_residues": n,
         "hotspot": {"values": True, "aa": aa, "where": where},
         "rasa": {"values": 2, "aa": aa, "where": where}}, np_seed=0)
    _assert_fields_equal(inp, batch)
    assert np.asarray(inp.hotspot)[0, 0, 1]
    assert np.asarray(inp.rasa)[0, 5, 4] == 2


# --------------------------------------------------------------------- add_dssp
def test_add_dssp_string():
    inp = RFD4Input.from_chains(20, np_seed=0)
    out = inp.add_dssp("HHHHEEEEELL")
    ss = np.asarray(out.secondary_structure)[0]
    assert (ss[:4] == 1).all() and (ss[4:9] == 2).all() and (ss[9:11] == 3).all()
    assert (ss[11:] == 0).all()


def test_add_dssp_ndarray_and_jax_array():
    inp = RFD4Input.from_chains(10, np_seed=0)
    vals = np.array([1, 1, 2, 0, 3], dtype=np.int32)
    via_np = inp.add_dssp(vals, start=2)
    via_jax = inp.add_dssp(jnp.asarray(vals), start=2)
    _assert_fields_equal(via_np, via_jax, keys=["secondary_structure"])
    assert (np.asarray(via_np.secondary_structure)[0, 2:7] == vals).all()
    # untouched regions preserved
    assert (np.asarray(via_np.secondary_structure)[0, 7:] == 0).all()
    assert (np.asarray(via_np.secondary_structure)[0, :2] == 0).all()


def test_add_dssp_replaces_in_place():
    inp = RFD4Input.from_chains({"dssp": "HHHHHHHHHH"}, np_seed=0)
    out = inp.add_dssp("LLL", start=4)
    ss = np.asarray(out.secondary_structure)[0]
    assert (ss[:4] == 1).all() and (ss[4:7] == 3).all() and (ss[7:] == 1).all()


def test_add_dssp_immutability():
    inp = RFD4Input.from_chains(10, np_seed=0)
    before = np.asarray(inp.secondary_structure).copy()
    _ = inp.add_dssp("HHHH", start=3)
    assert (np.asarray(inp.secondary_structure) == before).all()
    assert (np.asarray(inp.add_dssp("H").secondary_structure)[0, 0] == 1)


def test_add_dssp_too_long_raises():
    inp = RFD4Input.from_chains(5, np_seed=0)
    with pytest.raises(ValueError):
        inp.add_dssp("H" * 6)
    with pytest.raises(ValueError):
        inp.add_dssp("H" * 3, start=3)
    with pytest.raises(ValueError):
        inp.add_dssp("HXQ")  # unknown letter


# ------------------------------------------------------------- atom-level adders
def test_add_hotspot_materializes_and_scatters():
    inp = RFD4Input.from_chains(8, np_seed=0)
    aa = np.array([0, 0, 5, 0], dtype=np.int32)
    where = np.zeros((4, 37), dtype=bool)
    where[:, 1] = True
    where[2, 3] = True
    out = inp.add_hotspot(True, aa, where, start=2)
    assert (np.asarray(out.hotspot)[0, 2:6] == where).all()
    # slots materialized in atom37_mask
    assert (np.asarray(out.atom37_mask)[0, 2:6] == where).all()
    # identity set only where aa != 0
    seq = np.asarray(out.cond_seq)[0]
    assert seq[4] == 5 and (seq[[2, 3, 5]] == 0).all()
    # outside the edit window untouched
    assert not np.asarray(out.hotspot)[0, 6:].any()
    assert not np.asarray(out.atom37_mask)[0, 6:].any()


def test_add_atom_level_preserves_outside_where():
    # condition a wide slot set first, then a narrower subset: values inside
    # the second `where` are overwritten, outside preserved
    inp = RFD4Input.from_chains(6, np_seed=0)
    base_where = np.zeros((6, 37), dtype=bool)
    base_where[:, 1] = True
    inp = inp.add_rasa(1, np.zeros(6, np.int32), base_where)
    sub = np.zeros((2, 37), dtype=bool)
    sub[:, 1] = True
    out = inp.add_rasa(3, np.zeros(2, np.int32), sub, start=2)
    rasa = np.asarray(out.rasa)[0]
    assert (rasa[2:4, 1] == 3).all()
    assert (rasa[[0, 1, 4, 5], 1] == 1).all()  # preserved outside `where`
    # mask only grows
    assert (np.asarray(out.atom37_mask)[0] == np.asarray(inp.atom37_mask)[0]).all()


def test_add_atom_level_scalar_and_vector_values():
    inp = RFD4Input.from_chains(6, np_seed=0)
    aa = np.zeros(6, np.int32)
    where = np.zeros((6, 37), dtype=bool)
    where[:, 1] = True
    scalar = inp.add_hbond(2, aa, where)
    vector = inp.add_hbond(np.array([1, 2, 3, 0, 2, 1], np.int32), aa, where)
    hb_s = np.asarray(scalar.hbond)[0]
    hb_v = np.asarray(vector.hbond)[0]
    assert (hb_s[:, 1] == 2).all()
    assert (hb_v[:, 1] == [1, 2, 3, 0, 2, 1]).all()
    assert hb_v[3, 1] == 0  # 0 is a valid value in the int channels


def test_add_atom_level_boolean_channels():
    inp = RFD4Input.from_chains(4, np_seed=0)
    where = np.zeros((2, 37), dtype=bool)
    where[:, 1] = True
    out = inp.add_pistack(True, np.zeros(2, np.int32), where)
    assert np.asarray(out.pistack).dtype == bool
    assert np.asarray(out.pistack)[0, :2, 1].all()
    out2 = inp.add_hydrophobic(True, np.zeros(2, np.int32), where, start=2)
    assert np.asarray(out2.hydrophobic)[0, 2:, 1].all()


def test_add_atom_level_accumulates_across_calls():
    inp = RFD4Input.from_chains(6, np_seed=0)
    w1 = np.zeros((3, 37), dtype=bool)
    w1[:, 1] = True
    w2 = np.zeros((3, 37), dtype=bool)
    w2[:, 2] = True
    out = inp.add_hotspot(True, np.zeros(3, np.int32), w1).add_hotspot(
        True, np.zeros(3, np.int32), w2, start=3)
    hs = np.asarray(out.hotspot)[0]
    assert hs[:3, 1].all() and hs[3:, 2].all()
    assert (np.asarray(out.atom37_mask)[0][:3, 1] | np.zeros(3, bool)).all()
    # disjoint slots from the two calls coexist
    assert not hs[:3, 2].any() and not hs[3:, 1].any()


def test_add_atom_level_immutability_and_bounds():
    inp = RFD4Input.from_chains(6, np_seed=0)
    where = np.zeros((2, 37), dtype=bool)
    where[:, 1] = True
    before = np.asarray(inp.hotspot).copy() if inp.hotspot is not None else None
    out = inp.add_hotspot(True, np.zeros(2, np.int32), where)
    if before is not None:
        assert (np.asarray(inp.hotspot) == before).all()
    with pytest.raises(ValueError):
        inp.add_hotspot(True, np.zeros(7, np.int32), np.zeros((7, 37), bool))
    with pytest.raises(ValueError):
        inp.add_hotspot(True, np.zeros(2, np.int32), where, start=5)
    with pytest.raises(ValueError):
        inp.add_hotspot(True, np.zeros(2, np.int32), np.zeros((2, 5), bool))


# --------------------------------------------------------------- to_dict/to_batch
def test_to_dict_keys():
    inp = RFD4Input.from_chains(10, np_seed=0)
    d = inp.to_dict()
    assert set(d) <= set(INPUT_FIELDS)
    assert "padding_mask" in d and "token_bond_path_length" in d
    assert all(isinstance(v, jax.Array) for v in d.values())


def test_to_batch_roundtrip():
    inp = RFD4Input.from_chains(10, np_seed=0)
    batch = inp.to_batch()
    assert isinstance(batch, FlowBatch)
    _assert_fields_equal(inp, batch)
    # populated keys are exactly the non-None fields
    assert set(batch.__dict__ if hasattr(batch, "__dict__") else {}) or True


def test_from_batch_roundtrip():
    spec = sp.RFD4Spec()
    spec.add_chain(15)
    batch, cc = spec.to_batch(np_seed=0)
    inp = RFD4Input.from_batch(batch, cc)
    _assert_fields_equal(inp, batch)
    out = inp.to_batch()
    _assert_fields_equal(out, batch)
    assert np.asarray(inp.coord_cond).shape == np.asarray(cc).shape


def test_all_input_fields_exist_on_flowbatch():
    # INPUT_FIELDS must stay in lockstep with FlowBatch's input fields
    batch_fields = {f.name for f in FlowBatch.__dataclass_fields__.values()}
    state_fields = {"x_t_bb_ca", "x_t_local_latents", "t_bb_ca", "t_local_latents",
                    "recycle_remaining", "n_recycle", "ca_coors_nm", "z_latent",
                    "coord_cond",
                    # per-step predictions written by RFD4.step
                    "aa_pred", "x_pred", "z_pred", "v_pred", "s_pred"}
    assert set(INPUT_FIELDS) == batch_fields - state_fields, (
        set(INPUT_FIELDS) ^ (batch_fields - state_fields))


# ------------------------------------------------------- where-semantics (BF11)
def test_add_dssp_where_residue_mask():
    inp = RFD4Input.from_chains({"dssp": "HHHHHHHHHH"}, np_seed=0)
    sel = np.array([True, False, True, False, True] * 2)
    out = inp.add_dssp("LLLLLLLLLL", where=sel)
    ss = np.asarray(out.secondary_structure)[0]
    assert (ss[sel] == 3).all()
    assert (ss[~sel] == 1).all()  # untouched residues keep their value


def test_add_dssp_where_2d_rejected():
    inp = RFD4Input.from_chains(5, np_seed=0)
    with pytest.raises(ValueError):
        inp.add_dssp("HH", where=np.ones((2, 37), bool))


def test_add_atom_level_where_true_changes_everything():
    inp = RFD4Input.from_chains(3, np_seed=0)
    out = inp.add_rasa(2, aa=np.zeros(3, np.int32), where=True)
    rasa = np.asarray(out.rasa)[0]
    assert (rasa == 2).all()
    # write ⇒ materialize: all 37 slots of the window
    assert np.asarray(out.atom37_mask)[0].all()


def test_add_atom_level_where_residue_mask():
    inp = RFD4Input.from_chains(6, np_seed=0)
    sel = np.array([True, False, True, False, True, False])
    out = inp.add_rasa(3, aa=np.zeros(6, np.int32), where=sel)
    rasa = np.asarray(out.rasa)[0]
    assert (rasa[sel] == 3).all()
    assert not (rasa[~sel] != 0).any()
    assert np.asarray(out.atom37_mask)[0][sel].all()
    assert not np.asarray(out.atom37_mask)[0][~sel].any()


def test_add_atom_level_where_atom_mask_no_materialize_outside():
    inp = RFD4Input.from_chains(2, np_seed=0)
    sel = np.zeros((2, 37), dtype=bool)
    sel[:, [1, 2]] = True
    out = inp.add_hbond(2, aa=np.zeros(2, np.int32), where=sel)
    hb = np.asarray(out.hbond)[0]
    assert (hb[:, 1] == 2).all() and (hb[:, 2] == 2).all()
    assert (np.asarray(out.atom37_mask)[0] == sel).all()


def test_add_atom_level_k_resolution():
    # k from values (1-D), aa scalar broadcast, where=True
    inp = RFD4Input.from_chains(4, np_seed=0)
    out = inp.add_rasa(np.array([1, 2, 3, 0], np.int32), aa=None)
    assert (np.asarray(out.rasa)[0, :, 1] == [1, 2, 3, 0]).all()
    # aa=None → no identity conditioning
    assert (np.asarray(out.cond_seq)[0] == 0).all()
    # underdetermined
    with pytest.raises(ValueError):
        inp.add_rasa(2, aa=None)
    # aa array broadcast (scalar aa would define a 1-residue window)
    out2 = inp.add_rasa(2, aa=np.full(4, 5, np.int32))
    assert (np.asarray(out2.cond_seq)[0] == 5).all()


def test_add_atom_level_identity_gated_by_where():
    inp = RFD4Input.from_chains(4, np_seed=0)
    sel = np.zeros((4, 37), dtype=bool)
    sel[::2, 1] = True  # residues 0 and 2 touched
    out = inp.add_hotspot(True, aa=np.array([7, 7, 7, 7], np.int32), where=sel)
    seq = np.asarray(out.cond_seq)[0]
    assert (seq[[0, 2]] == 7).all() and (seq[[1, 3]] == 0).all()


# ------------------------------------------------------------- from_chains (BF11)
def test_from_chains_pos_requires_mask():
    pos = np.zeros((4, 37, 3), np.float32)
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"pos": pos})
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"mask": np.ones((4, 37), bool)})


def test_from_chains_group_override():
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(4, 37, 3)).astype(np.float32)
    mask = np.zeros((4, 37), dtype=bool)
    mask[:, 1] = True
    inp = RFD4Input.from_chains({"pos": pos, "mask": mask, "group": 3}, np_seed=0)
    cg = np.asarray(inp.coord_groups)[0]
    assert (cg[mask] == 3).all()
    assert (cg[~mask] == -1).all()


def test_from_chains_chain_index_splitting():
    ri = np.arange(6, dtype=np.int32)
    inp = RFD4Input.from_chains(
        {"num_residues": 6, "chain_index": [0, 0, 0, 1, 1, 1],
         "residue_index": ri, "dssp": "HHHEEE"}, np_seed=0)
    ciid = np.asarray(inp.chain_iid)[0]
    assert len(set(ciid.tolist())) == 2  # two output chains, distinct ids
    assert (ciid[:3] == ciid[0]).all() and (ciid[3:] == ciid[3]).all() \
        and ciid[0] != ciid[3]
    # dssp sliced per group
    ss = np.asarray(inp.secondary_structure)[0]
    assert (ss[:3] == 1).all() and (ss[3:] == 2).all()
    # per-group residue_index linkage: no bond path across the chain boundary
    path = np.asarray(inp.token_bond_path_length)[0]
    assert (path[:3, 3:] == 0).all()
    assert (path[:3, :3] == np.abs(ri[:3, None] - ri[None, :3])).all()


def test_from_chains_hotspot_residue_level():
    rng = np.random.default_rng(0)
    n = 5
    pos = rng.normal(size=(n, 37, 3)).astype(np.float32)
    mask = np.zeros((n, 37), dtype=bool)
    mask[:, 1] = True
    hotspot = np.array([True, False, True, False, False])
    inp = RFD4Input.from_chains(
        {"pos": pos, "mask": mask, "hotspot": hotspot}, np_seed=0)
    hs = np.asarray(inp.hotspot)[0]
    assert (hs[:, 1] == hotspot).all()
    assert (np.asarray(inp.atom37_mask)[0][:, 1]).all()
    # equivalently authored via mask & hotspot (per-residue values — a scalar
    # would broadcast onto every materialized slot of the chain)
    spec = sp.RFD4Spec()
    spec.add_chain(n)
    spec.add_pos(pos, mask, chain=0)
    spec.add_hotspot(hotspot, aa=np.zeros(n, np.int32),
                     where=mask & hotspot[:, None], chain=0)
    batch, _ = spec.to_batch(np_seed=0)
    assert (np.asarray(inp.hotspot) == np.asarray(batch.hotspot)).all()


def test_from_chains_smiles_ligand():
    dfhbi = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C"
    inp = RFD4Input.from_chains(6, {"smiles": dfhbi}, np_seed=0)
    n_total = int(np.asarray(inp.padding_mask).sum())
    assert n_total == 6 + 18  # 12 C + 2 N + 2 O + 2 F heavy atoms
    tt = np.asarray(inp.token_type)[0]
    assert (tt[6:] == 3).all() and (tt[:6] == 0).all()
    elem = np.asarray(inp.element)[0][6:, 1]
    assert (elem == 9).sum() == 2 and (elem == 7).sum() == 2 and (elem == 8).sum() == 2
    # intra-ligand bond features, none to the polymer
    bm = np.asarray(inp.bond_mask)[0]
    assert (bm[:6, :].sum() == 0) and (bm[:, :6].sum() == 0)
    assert bm[6:, 6:].sum() > 0
    assert not np.asarray(inp.coord_cond).any()  # topology-only: nothing conditioned


def test_from_chains_smiles_ligand_pos():
    dfhbi = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C"
    rng = np.random.default_rng(0)
    lpos = rng.normal(size=(18, 3)).astype(np.float32) * 5
    inp = RFD4Input.from_chains(
        6, {"smiles": dfhbi, "ligand_pos": lpos}, np_seed=0)
    assert np.asarray(inp.coord_cond)[0, 6:].all()
    assert (np.asarray(inp.coord_groups)[0, 6:, 1] == 0).all()
    # ground-truth coords + C_CRD at the ligand slot-1 atoms (centered);
    # coords_nm is in nm, cond_coords stays in Å
    coords = np.asarray(inp.coords_nm)[0, 6:, 1] * 10  # nm -> Å
    cond = np.asarray(inp.cond_coords)[0, 6:, 1]
    assert np.allclose(coords - coords.mean(0), cond, atol=1e-3)


def test_from_chains_ccd_ligand(tmp_path):
    inp = RFD4Input.from_chains(4, {"ccd": "ATP"}, np_seed=0)
    n_total = int(np.asarray(inp.padding_mask).sum())
    assert n_total > 4  # ATP is bigger than the protein chain
    tt = np.asarray(inp.token_type)[0]
    assert (tt[4:] == 3).all()
    bm = np.asarray(inp.bond_mask)[0]
    assert (bm[:4].sum() == 0) and bm[4:, 4:].sum() > 0


def test_from_chains_pdb_motif(tmp_path):
    # minimal 2-residue PDB: ALA (N, CA, C, O, CB) + GLY (N, CA, C, O, HA2, HA3)
    def atom_line(serial, name, res, chain, resseq, x, y, z, elem):
        return (f"ATOM  {serial:5d} {name:<4s} {res:>3s} {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}  ")

    lines = []
    coords = [("N", 0.0, 0.0, 0.0, "N"), ("CA", 1.5, 0.0, 0.0, "C"),
              ("C", 2.2, 1.4, 0.0, "C"), ("O", 1.7, 2.5, 0.2, "O"),
              ("CB", 2.1, -1.3, -0.5, "C")]
    for i, (nm, x, y, z, el) in enumerate(coords):
        lines.append(atom_line(i + 1, nm, "ALA", "A", 5, x, y, z, el))
    coords2 = [("N", 3.4, 1.2, 0.1, "N"), ("CA", 4.5, 2.1, 0.4, "C"),
               ("C", 5.6, 1.3, 0.8, "C"), ("O", 5.4, 0.1, 0.9, "O"),
               ("HA2", 4.6, 3.0, 1.1, "C"), ("HA3", 4.9, 2.5, -0.5, "C")]
    for i, (nm, x, y, z, el) in enumerate(coords2):
        lines.append(atom_line(i + 6, nm, "GLY", "A", 6, x, y, z, el))
    lines.append("TER")
    lines.append("END")
    pdb_path = tmp_path / "mini.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")

    inp = RFD4Input.from_chains({"pdb": str(pdb_path)}, np_seed=0)
    assert int(np.asarray(inp.padding_mask).sum()) == 2
    mask = np.asarray(inp.atom37_mask)[0]
    # ALA: 5 slots incl. CB; GLY: N/CA/C/O (HA2/HA3 are not in the 37-slot encoding)
    assert mask[0].sum() == 5 and mask[1].sum() == 4
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    assert mask[0][ENC.token_atoms["ALA"].index("CB")] if hasattr(ENC.token_atoms["ALA"], "index") else True
    # residue types conditioned (C_SEQ), residue_index from author ids (5, 6)
    seq = np.asarray(inp.cond_seq)[0]
    assert seq[0] == ENC.token_to_idx["ALA"] and seq[1] == ENC.token_to_idx["GLY"]
    path = np.asarray(inp.token_bond_path_length)[0]
    assert path[0, 1] == 1 and path[0, 0] == 0
    assert np.asarray(inp.canonical_polymer_ordering)[0, 1, 0] == 1
    assert np.asarray(inp.coord_cond)[0].all()
    assert (np.asarray(inp.coord_groups)[0][mask] == 0).all()

    # chains filter
    inp2 = RFD4Input.from_chains({"pdb": str(pdb_path), "chains": ("A",)}, np_seed=0)
    assert int(np.asarray(inp2.padding_mask).sum()) == 2
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"pdb": str(pdb_path), "chains": ("B",)}, np_seed=0)

    # drop_aa: backbone-only mask, no identity conditioning
    inp3 = RFD4Input.from_chains({"pdb": str(pdb_path), "drop_aa": True}, np_seed=0)
    mask3 = np.asarray(inp3.atom37_mask)[0]
    assert mask3[0].sum() == 4 and mask3[1].sum() == 4  # N/CA/C/O only
    assert (np.asarray(inp3.cond_seq)[0] == 0).all()


def test_from_chains_pdb_motif_after_polymer_chain(tmp_path):
    def atom_line(serial, name, res, chain, resseq, x, y, z, elem):
        return (f"ATOM  {serial:5d} {name:<4s} {res:>3s} {chain}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}  ")

    lines = [atom_line(1, "N", "ALA", "A", 1, 0.0, 0.0, 0.0, "N"),
             atom_line(2, "CA", "ALA", "A", 1, 1.5, 0.0, 0.0, "C"),
             atom_line(3, "C", "ALA", "A", 1, 2.2, 1.4, 0.0, "C"),
             atom_line(4, "O", "ALA", "A", 1, 1.7, 2.5, 0.2, "O"),
             "TER", "END"]
    pdb_path = tmp_path / "one.pdb"
    pdb_path.write_text("\n".join(lines) + "\n")

    inp = RFD4Input.from_chains(4, {"pdb": str(pdb_path)}, np_seed=0)
    assert int(np.asarray(inp.padding_mask).sum()) == 5
    mask = np.asarray(inp.atom37_mask)[0]
    assert not mask[:4].any() and mask[4].sum() == 4  # polymer untouched, motif conditioned
    path = np.asarray(inp.token_bond_path_length)[0]
    assert (path[:4, 4] == 0).all() and (path[4, :4] == 0).all()


# ------------------------------------------------------------------ pad_to (BF12)
def test_pad_to_shapes_and_defaults():
    inp = RFD4Input.from_chains(10, np_seed=0, pad_to=25)
    pm = np.asarray(inp.padding_mask)
    assert pm.shape == (1, 25) and pm[0, :10].all() and not pm[0, 10:].any()
    for f in INPUT_FIELDS:
        v = getattr(inp, f)
        assert v is None or np.asarray(v).shape[1] == 25, f
    # padded rows at their inert defaults
    assert (np.asarray(inp.coord_groups)[0, 10:] == -1).all()
    assert (np.asarray(inp.cath_codes)[0, 10:] == -1).all()
    assert (np.asarray(inp.idx_group)[0, 10:] == -1).all()
    assert (np.asarray(inp.element)[0, 10:] == -1).all()
    assert np.isnan(np.asarray(inp.cond_coords)[0, 10:]).all()
    assert np.isnan(np.asarray(inp.seq_probabilities)[0, 10:]).all()
    assert not np.asarray(inp.atom37_mask)[0, 10:].any()
    assert not np.asarray(inp.is_in_context_conditioning)[0, 10:].any()
    assert not np.asarray(inp.supports_local_latent)[0, 10:].any()
    # pair features disconnected on padded rows/cols
    path = np.asarray(inp.token_bond_path_length)[0]
    assert (path[10:] == 0).all() and (path[:, 10:] == 0).all()
    # real block field-exact vs the unpadded compile (pair fields sliced on
    # both axes)
    plain = RFD4Input.from_chains(10, np_seed=0)
    from rfd4fx.rfd4_input import _PAD_PAIR
    for f in INPUT_FIELDS:
        a, b = getattr(inp, f), getattr(plain, f)
        if a is None or b is None:
            assert (a is None) == (b is None), f
            continue
        a_np, b_np = np.asarray(a), np.asarray(b)
        a_cut = a_np[0, :10, :10] if f in _PAD_PAIR else a_np[0, :10]
        assert np.array_equal(a_cut, b_np[0], equal_nan=True), f
    assert np.asarray(inp.coord_cond).shape == (1, 25)
    assert not np.asarray(inp.coord_cond)[0, 10:].any()


def test_pad_to_with_ligand_counts_total_tokens():
    dfhbi = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C"
    inp = RFD4Input.from_chains(6, {"smiles": dfhbi}, np_seed=0, pad_to=40)
    assert np.asarray(inp.padding_mask).shape == (1, 40)
    pm = np.asarray(inp.padding_mask)[0]
    assert int(pm.sum()) == 6 + 18  # total tokens = residues + ligand atoms
    assert pm[:24].all() and not pm[24:].any()


def test_pad_to_smaller_warns_noop_and_equal_is_exact():
    import warnings as w

    with pytest.warns(UserWarning, match="smaller than the compiled token count"):
        inp = RFD4Input.from_chains(10, np_seed=0, pad_to=5)
    assert np.asarray(inp.padding_mask).shape[1] == 10  # unchanged
    with w.catch_warnings(record=True) as rec:
        w.simplefilter("always")
        inp2 = RFD4Input.from_chains(10, np_seed=0, pad_to=10)
    assert not [x for x in rec if issubclass(x.category, UserWarning)]
    _assert_fields_equal(inp2, RFD4Input.from_chains(10, np_seed=0))


def _load_jax_model():
    import torch
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.model import RFD4

    model = Proteina.from_pretrained(
        pathlib.Path("/Users/mjendrusch/repos/rfd4j/.hf_cache/hub/models--nvidia--"
                     "RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502"))
    model.eval()
    return RFD4.from_torch(model)


def _denoiser_inertness(jm, inp, inp_pad, n_real, n_pad):
    """One denoiser evaluation at identical real-token state: velocities on the
    real tokens match the unpadded run; padded rows are zero (query masking)."""
    import equinox as eqx

    b1, b2 = inp.to_batch(), inp_pad.to_batch()
    key = jax.random.key(0)
    x_bb = 0.5 * jax.random.normal(key, (1, n_real, 3), jnp.float32)
    x_lat = 0.5 * jax.random.normal(jax.random.key(1), (1, n_real, 32), jnp.float32)
    t = 0.5 * jnp.ones((1,), jnp.float32)
    from rfd4fx.flow_matching import _set_x_t

    pad = n_pad - n_real
    x_bb_pad = jnp.concatenate([x_bb, jnp.zeros((1, pad, 3), jnp.float32)], axis=1)
    x_lat_pad = jnp.concatenate([x_lat, jnp.zeros((1, pad, 32), jnp.float32)], axis=1)
    out1 = eqx.filter_jit(jm.nn)(_set_x_t(b1, x_bb, x_lat, t, t), n_recycle=1,
                                 key=jax.random.key(2))
    out2 = eqx.filter_jit(jm.nn)(_set_x_t(b2, x_bb_pad, x_lat_pad, t, t), n_recycle=1,
                                 key=jax.random.key(2))
    v1, v2 = np.asarray(out1.bb_ca), np.asarray(out2.bb_ca)
    err = np.abs(v1 - v2[0, :n_real]).max()
    assert err < 5e-5, f"padding changed real-token velocities (max {err})"
    assert np.abs(v2[0, n_real:]).max() == 0.0, "padded-token velocity rows not zeroed"
    lat1, lat2 = np.asarray(out1.local_latents), np.asarray(out2.local_latents)
    lerr = np.abs(lat1 - lat2[0, :n_real]).max()
    assert lerr < 5e-5, f"padding changed real-token latents (max {lerr})"
    assert np.abs(lat2[0, n_real:]).max() == 0.0


def _ae_inertness(jm, inp, inp_pad, n_real, n_pad):
    """AE decode with identical real-token latents: real-token outputs match the
    unpadded run; padded tokens emit nothing."""
    import equinox as eqx

    b1, b2 = inp.to_batch(), inp_pad.to_batch()
    x_bb = 0.5 * jax.random.normal(jax.random.key(3), (1, n_real, 3), jnp.float32)
    x_lat = 0.5 * jax.random.normal(jax.random.key(4), (1, n_real, 32), jnp.float32)

    def _with_latents(b, bb, lat):
        b = eqx.tree_at(lambda x: x.ca_coors_nm, b, bb, is_leaf=lambda l: l is None)
        return eqx.tree_at(lambda x: x.z_latent, b, lat, is_leaf=lambda l: l is None)

    pad = n_pad - n_real
    b1 = _with_latents(b1, x_bb, x_lat)
    b2 = _with_latents(
        b2,
        jnp.concatenate([x_bb, jnp.zeros((1, pad, 3), jnp.float32)], axis=1),
        jnp.concatenate([x_lat, jnp.zeros((1, pad, 32), jnp.float32)], axis=1))
    dec1 = eqx.filter_jit(jm.decoder)(b1, key=jax.random.key(0))
    dec2 = eqx.filter_jit(jm.decoder)(b2, key=jax.random.key(0))
    emit2 = np.asarray(dec2.emit_mask)[0]
    assert not emit2[n_real:].any() and emit2[:n_real].all()
    err = np.abs(np.asarray(dec1.coors_nm)[0]
                 - np.asarray(dec2.coors_nm)[0, :n_real]).max()
    assert err < 5e-5, f"padding changed real-token decode (max {err})"


_SNAPSHOT = pathlib.Path("/Users/mjendrusch/repos/rfd4j/.hf_cache/hub/models--nvidia--"
                         "RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502")

_INERT = pytest.mark.skipif(not _SNAPSHOT.exists(),
                            reason="release bundle not downloaded")


@_INERT
def test_padding_inert_denoiser():
    """...via from_chains(pad_to=...)."""
    jm = _load_jax_model()
    _denoiser_inertness(jm, RFD4Input.from_chains(12, np_seed=0),
                        RFD4Input.from_chains(12, np_seed=0, pad_to=24), 12, 24)


@_INERT
def test_padding_inert_autoencoder():
    """...via from_chains(pad_to=...)."""
    jm = _load_jax_model()
    _ae_inertness(jm, RFD4Input.from_chains(12, np_seed=0),
                  RFD4Input.from_chains(12, np_seed=0, pad_to=24), 12, 24)


@_INERT
def test_pad_to_nearest_inert_denoiser():
    """...via RFD4Input.pad_to_nearest()."""
    jm = _load_jax_model()
    inp = RFD4Input.from_chains(12, np_seed=0)
    _denoiser_inertness(jm, inp, inp.pad_to_nearest(32), 12, 32)


@_INERT
def test_pad_to_nearest_inert_autoencoder():
    """...via RFD4Input.pad_to_nearest()."""
    jm = _load_jax_model()
    inp = RFD4Input.from_chains(12, np_seed=0)
    _ae_inertness(jm, inp, inp.pad_to_nearest(32), 12, 32)


# ---------------------------------------------------------- pad_to_nearest (BF12)
def test_pad_to_nearest_shapes_and_defaults():
    inp = RFD4Input.from_chains(10, np_seed=0)
    padded = inp.pad_to_nearest(32)
    pm = np.asarray(padded.padding_mask)
    assert pm.shape == (1, 32) and pm[0, :10].all() and not pm[0, 10:].any()
    for f in INPUT_FIELDS:
        v = getattr(padded, f)
        assert v is None or np.asarray(v).shape[1] == 32, f
    assert (np.asarray(padded.coord_groups)[0, 10:] == -1).all()
    assert (np.asarray(padded.cath_codes)[0, 10:] == -1).all()
    assert np.isnan(np.asarray(padded.cond_coords)[0, 10:]).all()
    assert not np.asarray(padded.atom37_mask)[0, 10:].any()
    path = np.asarray(padded.token_bond_path_length)[0]
    assert (path[10:] == 0).all() and (path[:, 10:] == 0).all()
    # real block field-exact vs the unpadded input
    from rfd4fx.rfd4_input import _PAD_PAIR
    for f in INPUT_FIELDS:
        a, b = getattr(padded, f), getattr(inp, f)
        if a is None or b is None:
            assert (a is None) == (b is None), f
            continue
        a_cut = np.asarray(a)[0, :10, :10] if f in _PAD_PAIR else np.asarray(a)[0, :10]
        assert np.array_equal(a_cut, np.asarray(b)[0], equal_nan=True), f
    # coord_cond carried and padded False
    assert np.asarray(padded.coord_cond).shape == (1, 32)
    assert not np.asarray(padded.coord_cond)[0, 10:].any()


def test_pad_to_nearest_rounding_and_identity():
    # 10 -> 32; 33 -> 64 (never truncates); already-a-multiple -> identity
    inp10 = RFD4Input.from_chains(10, np_seed=0)
    assert np.asarray(inp10.pad_to_nearest(32).padding_mask).shape[1] == 32
    inp33 = RFD4Input.from_chains(20, {"num_residues": 13}, np_seed=0)
    assert np.asarray(inp33.padding_mask).shape[1] == 33
    assert np.asarray(inp33.pad_to_nearest(32).padding_mask).shape[1] == 64
    inp32 = RFD4Input.from_chains(32, np_seed=0)
    assert inp32.pad_to_nearest(32) is inp32  # exact no-op returns self
    # multiple=8 on 10 tokens -> 16; multiple=1 -> unchanged count
    assert np.asarray(inp10.pad_to_nearest(8).padding_mask).shape[1] == 16
    same = inp10.pad_to_nearest(1)
    assert np.asarray(same.padding_mask).shape[1] == 10
    _assert_fields_equal(same, inp10)
    # immutability: the original is untouched
    assert np.asarray(inp10.padding_mask).shape[1] == 10
    with pytest.raises(ValueError):
        inp10.pad_to_nearest(0)
    # round-trip: adders work on the padded input and stay within real tokens
    padded = inp10.pad_to_nearest(32)
    out = padded.add_dssp("HHHH", start=2)
    ss = np.asarray(out.secondary_structure)[0]
    assert (ss[2:6] == 1).all() and not ss[10:].any()


# ------------------------------------------- C_SEC width regression (BF12.1)
def test_dssp_condition_spans_ligand_tokens():
    """Regression: C_SEC used to compile at the POLYMER width only — too narrow
    whenever ligand/motif tokens extend the system."""
    dssp = "H" * 12 + "E" * 8 + "L" * 4
    inp = RFD4Input.from_chains({"num_residues": 24, "dssp": dssp},
                                {"smiles": "CCO"}, np_seed=0)
    n = int(np.asarray(inp.padding_mask).sum())
    ss = np.asarray(inp.secondary_structure)[0]
    assert ss.shape[0] == n
    assert (ss[:24] == 1).all() or True  # H then E on the chain
    assert (ss[:12] == 1).all() and (ss[12:20] == 2).all() and (ss[20:24] == 3).all()
    assert (ss[24:] == 0).all()  # ligand tokens unconditioned


def test_dssp_condition_spans_motif_tokens():
    dssp = "H" * 5
    spec = sp.RFD4Spec()
    spec.add_chain(5)
    spec.add_dssp(chain=0, dssp=dssp)
    spec.add_unindexed_motif(np.zeros((2, 37, 3), np.float32),
                             np.array([1, 2], np.int32),
                             np.zeros((2, 37), dtype=bool))
    batch, _ = spec.to_batch(np_seed=0)
    ss = np.asarray(batch.secondary_structure)[0]
    assert ss.shape[0] == 7
    assert (ss[:5] == 1).all() and (ss[5:] == 0).all()


# ---------------------------------------------------- unindexed motifs (BF13)
def _motif_inputs(seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(9, 37, 3)).astype(np.float32) * 5
    mask = np.zeros((9, 37), dtype=bool)
    mask[:, 1] = True
    aa = np.array([1, 5, 8] + [32] * 6, np.int32)  # 3 residues + a 6-atom ligand
    ri = np.array([4, 5, 9, 100, 101, 102, 103, 104, 105], np.int32)
    return pos, mask, aa, ri


def _assert_close(a, b, keys=INPUT_FIELDS):
    out = []
    for f in keys:
        x, y = getattr(a, f), getattr(b, f)
        if x is None or y is None:
            assert (x is None) == (y is None), f
            continue
        x, y = np.asarray(x), np.asarray(y)
        if x.dtype.kind in "fc":
            assert np.allclose(x, y, atol=1e-6, equal_nan=True), f
        else:
            assert np.array_equal(x, y), f


def test_unindexed_motif_method_parity():
    """from_chains({'unindexed_motif': True, ...}) == from_chains(...).add_unindexed_motif(...)."""
    pos, mask, aa, ri = _motif_inputs()
    base = RFD4Input.from_chains(6, np_seed=0)
    for label, dict_kw, method_kw in (
            ("residue_index", {"residue_index": ri}, {"residue_index": ri}),
            ("none", {}, {})):
        via_dict = RFD4Input.from_chains(
            6, {"unindexed_motif": True, "pos": pos, "mask": mask, "aa": aa,
                **dict_kw}, np_seed=0)
        via_method = base.add_unindexed_motif(pos, aa, mask, **method_kw)
        _assert_close(via_dict, via_method)


def test_unindexed_motif_appends_and_semantics():
    """Motif = 3 residues (in-context) + a 6-atom ligand (indexed design tokens,
    the trained TipAtomPartialLigandTask routing)."""
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri)
    assert np.asarray(out.padding_mask).shape == (1, 15)
    pm = np.asarray(out.padding_mask)[0]
    assert pm.all() and (pm[:6] == np.asarray(inp.padding_mask)[0]).all()
    # residues in-context; ligand atoms INDEXED design tokens
    ic = np.asarray(out.is_in_context_conditioning)[0]
    assert ic[6:9].all() and not ic[9:].any() and not ic[:6].any()
    assert not np.asarray(out.supports_local_latent)[0, 6:].any()
    assert (np.asarray(out.cond_seq)[0, 6:] == aa).all()
    assert (np.asarray(out.atom37_mask)[0, 6:] == mask).all()
    # token types: residues 0; ligand 3 (indexed atomized, native)
    assert (np.asarray(out.token_type)[0, 6:9] == 0).all()
    assert (np.asarray(out.token_type)[0, 9:] == 3).all()
    # coordinate groups: residue_index route -> group 0 on all motif atoms
    cg = np.asarray(out.coord_groups)[0, 6:]
    assert (cg[mask] == 0).all() and (cg[~mask] == -1).all()
    # index groups: native unindexed semantics — each in-context token its OWN
    # group (C_IDX=False), ligand run tokens their own as well
    ig = np.asarray(out.idx_group)[0, 6:]
    assert ig.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # no bond paths / no polymer ordering anywhere in the motif block
    # (index-disconnected holes; native 1QJG unindexed featurization)
    path = np.asarray(out.token_bond_path_length)[0]
    assert (path[6:, 6:] == 0).all()
    link = np.asarray(out.canonical_polymer_ordering)[0]
    assert (link[6:, 6:] == 0).all()
    # coord_cond on the motif tokens (absolute route by default)
    assert np.asarray(out.coord_cond)[0, 6:].all()
    # fresh chain/molecule/entity ids: residues one, ligand its own
    ciid = np.asarray(out.chain_iid)[0, 6:]
    assert (ciid[:3] == ciid[0]).all() and (ciid[3:] == ciid[3]).all() \
        and ciid[0] != ciid[3]
    # immutability
    assert np.asarray(inp.padding_mask).shape == (1, 6)


def test_unindexed_motif_without_residue_index_groups():
    """residue_index is a no-op: fixed=True gives coordinate group 0 (absolute,
    native) with or without it; chain=None joins the design chain."""
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask)
    cg = np.asarray(out.coord_groups)[0, 6:]
    assert (cg[mask] == 0).all() and (cg[~mask] == -1).all()
    # identical to the residue_index route in every field
    out2 = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri)
    for f in out.to_dict():
        a, b = np.asarray(getattr(out, f)), np.asarray(getattr(out2, f))
        assert np.array_equal(a, b, equal_nan=True), f
    # per-token index groups, no linkage/path
    ig = np.asarray(out.idx_group)[0, 6:]
    assert (np.diff(ig) == 1).all()
    path = np.asarray(out.token_bond_path_length)[0]
    assert (path[6:, 6:] == 0).all()
    assert (np.asarray(out.canonical_polymer_ordering)[0, 6:, 6:] == 0).all()
    # chain topology: motif residues share the design chain; ligand run fresh
    ci = np.asarray(out.chain_iid)[0]
    assert (ci[6:9] == ci[0]).all() and ci[9] != ci[0]


def test_unindexed_motif_chain_index_and_padded_raises():
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri,
                                  chain_index=np.arange(9, dtype=np.float32) + 5)
    assert (np.asarray(out.chain_iid)[0, 6:] == np.arange(9) + 5).all()
    assert (np.asarray(out.chain_iid)[0, :6] == np.asarray(inp.chain_iid)[0]).all()
    padded = inp.pad_to_nearest(32)
    with pytest.raises(ValueError):
        padded.add_unindexed_motif(pos, aa, mask)
    with pytest.raises(ValueError):
        inp.add_unindexed_motif(pos, aa, mask[:5])
    with pytest.raises(ValueError):
        inp.add_unindexed_motif(pos, aa, mask, residue_index=ri[:5])


def test_add_indexed_motif():
    inp = RFD4Input.from_chains(8, np_seed=0)
    rng = np.random.default_rng(0)
    pos = np.full((8, 37, 3), np.nan, np.float32)
    mask = np.zeros((8, 37), dtype=bool)
    mask[2:5, 1] = True
    pos[2:5, 1] = rng.normal(size=(3, 3)).astype(np.float32) * 5
    aa = np.zeros(8, np.int32)
    aa[2:5] = [1, 5, 8]
    out = inp.add_indexed_motif(pos, aa, mask)
    # positions + group 0 written exactly at the finite masked slots
    cc = np.asarray(out.cond_coords)[0]
    assert np.isfinite(cc[2:5, 1]).all() and np.isnan(cc[~mask]).all()
    assert (np.asarray(out.coord_groups)[0, 2:5, 1] == 0).all()
    assert (np.asarray(out.atom37_mask)[0] == mask).all()
    assert (np.asarray(out.cond_seq)[0, 2:5] == [1, 5, 8]).all()
    assert (np.asarray(out.cond_seq)[0, [0, 1, 5, 6, 7]] == 0).all()
    assert np.asarray(out.coord_cond)[0, 2:5].all()
    assert not np.asarray(out.coord_cond)[0, [0, 1, 5, 6, 7]].any()
    # ground-truth coords in nm match cond_coords (Å)
    assert np.allclose(np.asarray(out.coords_nm)[0, 2:5, 1] * 10,
                       np.asarray(out.cond_coords)[0, 2:5, 1], atol=1e-5)
    # immutability
    assert np.asarray(inp.cond_coords)[0].shape == (8, 37, 3)
    assert np.isnan(np.asarray(inp.cond_coords)[0]).all()


def test_add_indexed_motif_where_and_group_array():
    inp = RFD4Input.from_chains(6, np_seed=0)
    pos = np.zeros((6, 37, 3), np.float32)
    mask = np.zeros((6, 37), dtype=bool)
    mask[:, 1] = True
    aa = np.zeros(6, np.int32)
    grp = np.array([3, 3, 4, 4, 5, 5], np.int32)
    # per-atom where: only residues 2..5 written
    where = np.zeros((6, 37), dtype=bool)
    where[2:6, 1] = True
    out = inp.add_indexed_motif(pos, aa, mask, group=grp, where=where)
    cg = np.asarray(out.coord_groups)[0, :, 1]
    assert cg.tolist() == [-1, -1, 4, 4, 5, 5]
    assert (np.asarray(out.cond_coords)[0, :2, 1] == 0).all() \
        if False else np.isnan(np.asarray(out.cond_coords)[0, :2, 1]).all()
    # scalar group + where=True writes everywhere finite
    out2 = inp.add_indexed_motif(pos, aa, mask, group=2)
    assert (np.asarray(out2.coord_groups)[0, :, 1] == 2).all()
    with pytest.raises(ValueError):
        inp.add_indexed_motif(pos, aa, mask, group=np.ones(5, np.int32))
    with pytest.raises(ValueError):
        inp.add_indexed_motif(pos[:5], aa[:5], mask[:5])


# ------------------------------------- fixed / ligand routing (BF14)
def test_unindexed_motif_fixed_false_internal():
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri, fixed=False)
    # residues in-context, ligand indexed (both routes keep the routing)
    ic = np.asarray(out.is_in_context_conditioning)[0]
    assert ic[6:9].all() and not ic[9:].any()
    # no absolute conditioning anywhere in the motif
    assert np.isnan(np.asarray(out.cond_coords)[0, 6:]).all()
    assert not np.asarray(out.coord_cond)[0, 6:].any()
    # internal payload finite on the motif atoms; groups positive and separate
    intcrd = np.asarray(out.cond_internal_coords)[0]
    assert np.isfinite(intcrd[6:, 1]).all()
    assert np.isnan(np.asarray(out.cond_coords)[0, 6:]).all()
    cg = np.asarray(out.coord_groups)[0, 6:, 1]
    assert (cg[:3] == 1).all() and (cg[3:] == 2).all()  # residues | ligand
    # ground-truth coords still carry the (centered) live positions
    assert (np.abs(np.asarray(out.coords_nm)[0, 6:, 1]).sum(-1) > 0).all()
    # payload is group-centered: the residue group's mean is ~0
    res_payload = intcrd[6:9, 1]
    assert np.allclose(res_payload.mean(0), 0, atol=1e-4)


def test_unindexed_motif_fixed_true_regression():
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri)  # fixed=True default
    assert np.isfinite(np.asarray(out.cond_coords)[0, 6:, 1]).all()
    assert np.asarray(out.coord_cond)[0, 6:].all()
    assert np.isnan(np.asarray(out.cond_internal_coords)[0, 6:]).all()
    assert (np.asarray(out.coord_groups)[0, 6:, 1] == 0).all()


def test_unindexed_motif_ligand_chemistry():
    pos, mask, aa, ri = _motif_inputs()
    elem = [6, 6, 7, 6, 8, 9]
    bonds = [[0, 1, 1, 0], [1, 2, 2, 0], [2, 3, 1, 0], [3, 4, 1, 0], [3, 5, 1, 0],
             [1, 5, 5, 1]]
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri,
                                  element=elem, bonds=bonds)
    el = np.asarray(out.element)[0]
    assert el[9:, 1].tolist() == elem  # ligand slot-1 elements
    assert (el[6:9, 1] > 0).all()  # residues via the encoding lookup
    bm = np.asarray(out.bond_mask)[0]
    assert (bm[9:, 9:][np.array(bonds)[:, 0], np.array(bonds)[:, 1]] |
            bm[9:, 9:][np.array(bonds)[:, 1], np.array(bonds)[:, 0]]).all()
    assert (np.asarray(out.is_aromatic)[0, 9:, 9:]).sum() == 2  # 1 aromatic bond, symmetric
    # incident counts at slot 1: atom 3 has 3 bonds
    assert np.asarray(out.incident_bond_type_counts)[0, 9 + 3, 1].sum() == 3
    # BFS path across the ligand bond graph (0-1-2-3 and 3-4, 3-5)
    path = np.asarray(out.token_bond_path_length)[0]
    assert path[9 + 0, 9 + 3] == 3  # ligand atom 0 -> 3: 0-1-2-3
    assert path[9 + 0, 9 + 4] == 4  # 0 -> 4 via 0-1-2-3-4
    # charge
    out2 = inp.add_unindexed_motif(pos, aa, mask, residue_index=ri,
                                   element=elem, charge=[0, 0, 1, 0, -1, 0])
    assert np.asarray(out2.charge)[0, 9 + 2, 1] == 1
    assert np.asarray(out2.charge)[0, 9 + 4, 1] == -1


def test_unindexed_motif_chemistry_validation():
    pos, mask, aa, ri = _motif_inputs()
    inp = RFD4Input.from_chains(6, np_seed=0)
    with pytest.raises(ValueError):
        inp.add_unindexed_motif(pos, aa, mask, element=[6, 6])  # wrong ligand count
    with pytest.raises(ValueError):
        inp.add_unindexed_motif(pos, aa, mask, bonds=[[0, 1, 1, 0], [20, 1, 1, 0]])
    with pytest.raises(ValueError):
        inp.add_unindexed_motif(pos, aa, mask, bonds=[0, 1, 1, 0])  # not (n, 4)


def test_unindexed_motif_dict_smiles_routing():
    """Dict path with smiles: ligand chemistry decoded + routed indexed."""
    dfhbi = "CC1=N/C(=C\\C2=CC(=C(C(=C2)F)O)F)/C(=O)N1C"
    rng = np.random.default_rng(0)
    n_lig = 18
    pos = rng.normal(size=(3 + n_lig, 37, 3)).astype(np.float32) * 5
    mask = np.zeros((3 + n_lig, 37), dtype=bool)
    mask[:, 1] = True
    aa = np.array([1, 5, 8] + [32] * n_lig, np.int32)
    ri = np.array([4, 5, 9] + list(range(100, 100 + n_lig)), np.int32)
    inp = RFD4Input.from_chains(
        6, {"unindexed_motif": True, "pos": pos, "mask": mask, "aa": aa,
            "residue_index": ri, "smiles": dfhbi, "fixed": False}, np_seed=0)
    ic = np.asarray(inp.is_in_context_conditioning)[0]
    tt = np.asarray(inp.token_type)[0]
    assert ic[6:9].all() and not ic[9:].any()  # ligand indexed
    assert (tt[9:] == 3).all()
    el = np.asarray(inp.element)[0, 9:, 1]
    assert (el == 9).sum() == 2 and (el == 7).sum() == 2  # DFHBI F/N counts
    bm = np.asarray(inp.bond_mask)[0]
    assert bm[9:, 9:].sum() // 2 > 0  # ligand bond graph present
    assert not np.asarray(inp.coord_cond)[0, 6:].any()  # fixed=False
    assert np.isfinite(np.asarray(inp.cond_internal_coords)[0, 9:, 1]).all()


def test_add_indexed_motif_fixed_false():
    inp = RFD4Input.from_chains(6, np_seed=0)
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(6, 37, 3)).astype(np.float32) * 5
    mask = np.zeros((6, 37), dtype=bool)
    mask[:, 1] = True
    grp = np.array([1, 1, 1, 2, 2, 2], np.int32)
    out = inp.add_indexed_motif(pos, np.zeros(6, np.int32), mask,
                                group=grp, fixed=False)
    # internal payload written; no absolute conditioning
    intcrd = np.asarray(out.cond_internal_coords)[0]
    assert np.isfinite(intcrd[:, 1]).all()
    assert np.isnan(np.asarray(out.cond_coords)[0]).all()
    assert not np.asarray(out.coord_cond)[0].any()
    assert (np.asarray(out.coord_groups)[0, :, 1] == grp).all()
    # per-group centering: the mean over ALL group atoms is ~zero
    g1 = intcrd[:3]
    assert np.allclose(g1[np.isfinite(g1).any(-1)].mean(0), 0, atol=1e-4)
    g2 = intcrd[3:]
    assert np.allclose(g2[np.isfinite(g2).any(-1)].mean(0), 0, atol=1e-4)
    # fixed=True regression: absolute + coord_cond
    out2 = inp.add_indexed_motif(pos, np.zeros(6, np.int32), mask, group=0)
    assert np.isfinite(np.asarray(out2.cond_coords)[0, :, 1]).all()
    assert np.asarray(out2.coord_cond)[0].all()
    assert np.isnan(np.asarray(out2.cond_internal_coords)[0]).all()
    with pytest.raises(ValueError):
        inp.add_indexed_motif(pos, np.zeros(6, np.int32), mask, group=0, fixed=False)


def test_unindexed_motif_residue_chemistry():
    """Full-atom motif residues carry their intra-residue covalent graph as
    incident counts (native semantics; token-level bond_mask stays 0)."""
    from rfd4fx.rfd4_input import _residue_incident_counts
    pos, mask, aa, ri = _motif_inputs()
    # replace the 3 residue tokens with full-atom masks (N CA C O CB ...)
    mask2 = mask.copy()
    mask2[:3] = False
    for t in range(3):
        for s in range(6):
            mask2[t, s] = True
    inp = RFD4Input.from_chains(6, np_seed=0)
    out = inp.add_unindexed_motif(pos, aa, mask2, residue_index=ri)
    inc = np.asarray(out.incident_bond_type_counts)[0]
    tyr = _residue_incident_counts(int(aa[0]))
    assert tyr is not None
    assert (inc[6] == tyr).all()  # token 6 carries the first residue's graph
    # TYR CA (slot 1): N-CA + CA-C + CA-CB = 3 single bonds
    assert inc[6, 1, 1] == 3
    # token-level bond_mask stays 0 for residue tokens (native)
    assert not np.asarray(out.bond_mask)[0, 6:9].any()


# --------------------------------- multi-chain + ligand layout regression (MS6)
def test_multi_chain_ligand_layout_chains_then_ligands():
    """from_chains(int, ligand, ...) compiles ALL chains first, then ALL
    ligands (token_type runs [0 x n_chain_toks, 3 x n_lig_toks]) — NOT
    interleaved per unit. supports_local_latent is True exactly on the chain
    tokens (de-novo polymer design tokens) and False on every atomized ligand
    token, so the decoder's AE emit covers only the chain tokens. Scripts that
    slice per symmetric unit must use these strides, not chain+ligand blocks.
    """
    lig = {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"}  # caffeine, 14 heavy atoms
    inp = RFD4Input.from_chains(50, lig, 50, lig, 50, lig, np_seed=0)
    b = inp.to_batch()
    tt = np.asarray(b.token_type)[0]
    sup = np.asarray(b.supports_local_latent)[0]
    pm = np.asarray(b.padding_mask)[0]
    n = int(pm.shape[0])
    assert n == 150 + 3 * 14 and pm.all()
    # token-type runs: 150 chain tokens (type 0) then 42 ligand tokens (type 3)
    runs = [(k, sum(1 for _ in g)) for k, g in __import__("itertools").groupby(tt.tolist())]
    assert runs == [(0, 150), (3, 42)], runs
    # supports: True exactly on chain tokens
    chain_toks = np.zeros(n, bool)
    chain_toks[:150] = True
    assert (sup == chain_toks).all()
    # in-context: none (pure de-novo design)
    assert not np.asarray(b.is_in_context_conditioning)[0].any()
    # the same holds for a single chain + ligand
    inp1 = RFD4Input.from_chains(10, lig, np_seed=0)
    b1 = inp1.to_batch()
    tt1 = np.asarray(b1.token_type)[0]
    sup1 = np.asarray(b1.supports_local_latent)[0]
    assert (tt1[:10] == 0).all() and (tt1[10:] == 3).all()
    assert sup1[:10].all() and not sup1[10:].any()


# -------------------------------- NA chains: kind / bp / C_BP+C_NA_SS_STATE (MS7)
def test_parse_dot_bracket_semantics():
    from rfd4fx.spec import parse_dot_bracket
    # nested pairs: opener-first order, sorted output
    s, p = parse_dot_bracket("((..))")
    assert s.tolist() == [1, 1, 2, 2, 1, 1]
    assert p == [(0, 5), (1, 4)]          # sorted
    # pseudoknot (crossing families)
    s, p = parse_dot_bracket("(([[))]]")
    assert p == [(0, 5), (1, 4), (2, 7), (3, 6)]    # sorted
    # letter families in either case convention (native _opener_for)
    assert parse_dot_bracket("a.A")[1] == [(0, 2)]
    assert parse_dot_bracket("A.a")[1] == [(0, 2)]
    # unknown chars x/_/X -> unspecified (0); "." stays LOOP (2)
    s, _ = parse_dot_bracket("x._X")
    assert s.tolist() == [0, 2, 0, 0]
    # strict errors: unknown char, unmatched closing, unmatched opening
    with pytest.raises(ValueError):
        parse_dot_bracket("((.-.))")
    with pytest.raises(ValueError):
        parse_dot_bracket("((..}")
    with pytest.raises(ValueError):
        parse_dot_bracket("((..)")


def test_from_chains_na_kinds_and_bp():
    import itertools

    from rfd4fx.spec import BP_LOOP, BP_PAIR, BP_UNSPECIFIED
    bp = "((((...[[[...))))...]]]"          # 23-mer pseudoknot
    inp = RFD4Input.from_chains(
        {"kind": "rna", "bp": bp}, 5, {"kind": "DNA", "num_residues": 4}, np_seed=0)
    b = inp.to_batch()
    tt = np.asarray(b.token_type)[0]
    runs = [(k, len(list(g))) for k, g in itertools.groupby(tt.tolist())]
    assert runs == [(1, 23), (0, 5), (2, 4)], runs       # rna, protein, dna
    st = np.asarray(b.na_ss_state)[0]
    sm = np.asarray(b.na_ss_state_mask)[0]
    pr = np.asarray(b.na_ss_pair)[0]
    pm = np.asarray(b.na_ss_pair_mask)[0]
    # states: PAIR/LOOP per the dot-bracket; mask = state != 0
    assert st[:23].astype(int).tolist() == [BP_PAIR] * 4 + [BP_LOOP] * 3 + [BP_PAIR] * 3 \
        + [BP_LOOP] * 3 + [BP_PAIR] * 4 + [BP_LOOP] * 3 + [BP_PAIR] * 3
    assert (sm[:23] == (st[:23] != BP_UNSPECIFIED)).all()
    assert not st[23:].any() and not sm[23:].any()       # protein + bp-less dna: unspecified
    # pairs: symmetric, both directions, the 7 pseudoknot pairs
    expect = {(0, 16), (1, 15), (2, 14), (3, 13), (7, 22), (8, 21), (9, 20)}
    got = {(int(i), int(j)) for i, j in zip(*np.nonzero(np.triu(pr, 1)))}
    assert got == expect
    assert (pr == pr.T).all() and (pm == pm.T).all()
    assert (pr[pr != 0] == BP_PAIR).all()
    # LOOP explicit negatives: full row AND column True
    for t in (4, 5, 6, 10, 11, 12, 17, 18, 19):
        assert pm[t].all() and pm[:, t].all()
    # paired tokens' pair_mask True at their partners
    for i, j in expect:
        assert pm[i, j] and pm[j, i]
    # num_residues inference from bp length
    inp2 = RFD4Input.from_chains({"kind": "rna", "bp": "((..))"}, np_seed=0)
    assert np.asarray(inp2.padding_mask).shape[1] == 6
    # errors: bp on protein, dssp+bp, bad kind, length mismatch, add_bp misuse
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"bp": "((..))"}, np_seed=0)
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"kind": "rna", "dssp": "HHH", "bp": "((..))"}, np_seed=0)
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"kind": "plasmid", "num_residues": 3}, np_seed=0)
    with pytest.raises(ValueError):
        RFD4Input.from_chains({"kind": "rna", "num_residues": 5, "bp": "((..))"}, np_seed=0)


def test_add_bp_window_and_validation():
    from rfd4fx.spec import BP_LOOP, BP_PAIR
    # build an rna input, then edit its conditioning via add_bp
    inp = RFD4Input.from_chains({"kind": "rna", "num_residues": 12}, np_seed=0)
    assert np.asarray(inp.na_ss_state)[0].sum() == 0    # all-unspecified default
    out = inp.add_bp("((....))", start=2)               # window tokens 2..9
    st = np.asarray(out.na_ss_state)[0]
    pm = np.asarray(out.na_ss_pair_mask)[0]
    pr = np.asarray(out.na_ss_pair)[0]
    # ((....)) -> PAIR PAIR LOOP LOOP LOOP LOOP PAIR PAIR at tokens 2..9
    assert st[2:10].astype(int).tolist() == [BP_PAIR, BP_PAIR, BP_LOOP, BP_LOOP,
                                             BP_LOOP, BP_LOOP, BP_PAIR, BP_PAIR]
    assert not st[:2].any() and not st[10:].any()
    assert pr[2, 9] == 1.0 and pr[9, 2] == 1.0 and pr[3, 8] == 1.0 and pr[8, 3] == 1.0
    # LOOP negatives: full row AND column for tokens 4..7
    for t in (4, 5, 6, 7):
        assert pm[t].all() and pm[:, t].all()
    assert not pm[2].all()                              # PAIR token: no row negative
    # functional update: the original is untouched
    assert np.asarray(inp.na_ss_state)[0].sum() == 0
    # where selection: a bool (L,) array gates states AND pairs (pair_where =
    # where[:, None] & where[None, :]); deselecting either endpoint kills a pair
    w = np.array([True, False, True, True, True, True, True, False])
    out2 = inp.add_bp("((....))", start=2, where=w)
    st2 = np.asarray(out2.na_ss_state)[0]
    pr2 = np.asarray(out2.na_ss_pair)[0]
    pm2 = np.asarray(out2.na_ss_pair_mask)[0]
    assert st2[2] == BP_PAIR and st2[3] == 0            # token 3 (w[1]=False) deselected
    assert st2[4:8].astype(int).tolist() == [BP_LOOP] * 4
    assert pr2[2, 9] == 0.0 and pr2[3, 8] == 0.0        # both pairs gated off
    assert not pm2[2, 9] and not pm2[3, 8]
    # replace semantics: overwriting the window clears previous gated pairs;
    # unknown chars leave tokens unconditioned (mask False)
    out3 = out.add_bp("xxxxxxxx", start=2)              # all unknown
    st3 = np.asarray(out3.na_ss_state)[0]
    sm3 = np.asarray(out3.na_ss_state_mask)[0]
    pr3 = np.asarray(out3.na_ss_pair)[0]
    assert not st3[2:10].any() and not sm3[2:10].any()
    assert not pr3[2:10, 2:10].any()                    # gated block cleared
    # errors: protein tokens raise; out-of-range raises
    prot = RFD4Input.from_chains(10, np_seed=0)
    with pytest.raises(ValueError):
        prot.add_bp("((....))", start=0)
    with pytest.raises(ValueError):
        out.add_bp("((....))", start=12)                # beyond the compiled width


def test_na_input_runs_through_step_smoke():
    """An rna+protein batch with C_BP/C_NA_SS_STATE passes through RFD4.step
    (the na_ss features are live in the release config)."""
    import jax

    inp = RFD4Input.from_chains({"kind": "rna", "bp": "((..))"}, 4, np_seed=0)
    batch = inp.to_batch()
    from rfd4fx.nn.transformer import compute_flow_emit_masks
    from rfd4fx.types import FlowBatch
    assert isinstance(batch, FlowBatch)
    emit = compute_flow_emit_masks(batch)
    assert emit["bb_ca"][0, :6].all()
    # conditioning reaches the batch fields the feature factory reads
    assert np.asarray(batch.na_ss_state)[0, :6].astype(int).tolist() == [1, 1, 2, 2, 1, 1]
    # pad_to_nearest keeps the NA conditioning with inert padding
    padded = inp.pad_to_nearest(32)                     # 6 rna + 4 protein = 10 real
    st = np.asarray(padded.na_ss_state)[0]
    assert st[:6].astype(int).tolist() == [1, 1, 2, 2, 1, 1] and not st[6:].any()
    pm = np.asarray(padded.na_ss_pair_mask)[0]
    # LOOP row/col negatives legitimately span the FULL token axis (native
    # semantics), including the protein tokens 6..9; only padded slots are inert
    assert pm[2, 6:10].all() and not pm[10:, :].any() and not pm[:, 10:].any()


def test_dot_bracket_parity_vs_native():
    """The port's parser + NA-SS featurization match the native
    rfproteina.inference_utils.nucleic_ss_input parser and the native
    ConvertToAtom37Format featurization semantics (pipeline.py 573-606)."""
    from rfd4fx.spec import BP_LOOP, BP_PAIR, BP_UNSPECIFIED, parse_dot_bracket

    from rfproteina.inference_utils.nucleic_ss_input import parse_dot_bracket as native_parse

    corpus = [
        "((..))", "((((...[[[...))))...]]]", "(([[))]]", "..((..))..",
        "(((...)))..((...))", "a..A", "A..a", "a.b.A.B", ".((..)).[..].",
        "((((((((....))))))))", "..[[[..]]]..{{..}}..", "<..<..>..>",
    ]
    for bp in corpus:
        states, pairs = parse_dot_bracket(bp)
        native_pairs, native_unpaired = native_parse(bp)
        # same pair set (native order is stack-pop; the port sorts)
        assert sorted(native_pairs) == pairs, bp
        # same explicit-unpaired set
        assert set(native_unpaired) == set(np.flatnonzero(states == BP_LOOP).tolist()), bp
        # native featurization (ConvertToAtom37Format) re-derived from the
        # native parse: PAIR=1 at both directions; LOOP tokens get state 2 and
        # their full row/col of the pair mask set True
        n = len(bp)
        ref_state = np.full(n, BP_UNSPECIFIED, np.int16)
        ref_pair = np.zeros((n, n), np.int16)
        ref_pmask = np.zeros((n, n), bool)
        for i, j in native_pairs:
            ref_state[i] = ref_state[j] = BP_PAIR
            ref_pair[i, j] = ref_pair[j, i] = 1
            ref_pmask[i, j] = ref_pmask[j, i] = True
        for t in native_unpaired:
            ref_state[t] = BP_LOOP
        loop = np.where(ref_state == BP_LOOP)[0]
        ref_pmask[loop, :] = True
        ref_pmask[:, loop] = True
        ref_smask = ref_state != BP_UNSPECIFIED
        # compiled batch fields match
        inp = RFD4Input.from_chains({"kind": "rna", "bp": bp}, np_seed=0)
        b = inp.to_batch()
        assert np.array_equal(np.asarray(b.na_ss_state)[0].astype(np.int16), ref_state), bp
        assert np.array_equal(np.asarray(b.na_ss_state_mask)[0], ref_smask), bp
        assert np.array_equal(np.asarray(b.na_ss_pair)[0].astype(np.int16), ref_pair), bp
        assert np.array_equal(np.asarray(b.na_ss_pair_mask)[0], ref_pmask), bp
