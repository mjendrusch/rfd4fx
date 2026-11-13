"""Cyclic peptide / covalent-bond featurization tests.

Authors the three CP-SEA cyclization chemistries (the native fixtures in
RFD4-Proteina `tests/data/cyclic_peptides/`: head-to-tail `4djs`, disulfide
`7bob`, isopeptide `6rlv`) through both rfd4fx authoring APIs —
`RFD4Input.from_chains` (`"bonds"`/`"cyclic"` keys) and the post-construction
`add_bonds`/`cyclize` adders — and pins the native bond-feature semantics
documented in `info.md` / native `tests/pipeline/test_canonical_polymer_ordering.py`:

* a head-to-tail closure is a backbone linkage: `canonical_polymer_ordering`
  ±1, suppressed from `bond_mask`, path lengths are ring hop counts;
* a crosslink is a visible pair-track bond (`bond_mask`/`bond_order`, ordering 0);
* both enter `incident_bond_type_counts` at every endpoint slot.
"""

import numpy as np
import pytest
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

from rfd4fx.rfd4_input import RFD4Input


def _slot(res_name: str, atom: str) -> int:
    return [s for s, nm in enumerate(ENC.token_atoms[res_name])
            if str(nm).strip() == atom][0]


def _field(inp, name):
    return np.asarray(getattr(inp, name))[0]


# ---------------------------------------------------------------- head-to-tail
# native fixture cpsea_head_tail_4djs: ring NIMRT, closure THR(last).C -> ASN(first).N

def test_head_tail_from_chains():
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "NIMRT",
                                     "cyclic": True})
    f, l = 80, 84                                   # first/last ring tokens
    cpo, bm = _field(inp, "canonical_polymer_ordering"), _field(inp, "bond_mask")
    assert cpo[f, l] == 1 and cpo[l, f] == -1       # first = downstream (N)
    assert not bm[f, l] and not bm[l, f]            # backbone linkage: suppressed
    pl = _field(inp, "token_bond_path_length")
    assert pl[f, l] == 1 and pl[81, 83] == 2        # ring hop counts, not |i-j|=4/2
    inc = _field(inp, "incident_bond_type_counts")
    assert inc[l, _slot("THR", "C"), 1] == 1        # closure C gains a single bond
    assert inc[f, _slot("ASN", "N"), 1] == 1        # closure N gains a single bond


def test_head_tail_cyclize_api():
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "NIMRT"}).cyclize(1)
    cpo, bm = _field(inp, "canonical_polymer_ordering"), _field(inp, "bond_mask")
    assert cpo[80, 84] == 1 and cpo[84, 80] == -1
    assert not bm[80, 84]
    assert _field(inp, "token_bond_path_length")[81, 83] == 2
    # feature equality with the from_chains route
    ref = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "NIMRT",
                                     "cyclic": True})
    for f in ("canonical_polymer_ordering", "bond_mask",
              "token_bond_path_length", "incident_bond_type_counts"):
        assert np.array_equal(_field(inp, f), _field(ref, f)), f


# ------------------------------------------------------------------- disulfide
# native fixture cpsea_disulfide_7bob: ring CNLFSC (+ACE/NME caps), CYS.SG-SG crosslink

def test_disulfide_from_chains():
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "CNLFSC",
        "bonds": [{"atoms": ["1@SG", "6@SG"], "order": 1}]})
    u, d = 80, 85                                   # CYS1 / CYS6 tokens
    bm, bo = _field(inp, "bond_mask"), _field(inp, "bond_order")
    cpo = _field(inp, "canonical_polymer_ordering")
    assert bm[u, d] and bm[d, u]                    # crosslink: visible
    assert bo[u, d] == 1                            # single bond
    assert cpo[u, d] == 0 and cpo[d, u] == 0        # no backbone ordering
    assert _field(inp, "token_bond_path_length")[u, d] == 1
    sg = _slot("CYS", "SG")
    inc = _field(inp, "incident_bond_type_counts")
    # the authored crosslink enters both cysteines at the SG slot (native
    # collated batches additionally carry the intra-residue CB-SG count,
    # which the de-novo spec route does not populate)
    assert inc[u, sg, 1] == 1 and inc[d, sg, 1] == 1


def test_disulfide_add_bonds_api():
    sg = _slot("CYS", "SG")
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "CNLFSC"})
    inp = inp.add_bonds((80, sg, 85, sg, 1))
    assert _field(inp, "bond_mask")[80, 85]
    assert _field(inp, "bond_order")[80, 85] == 1
    assert _field(inp, "canonical_polymer_ordering")[80, 85] == 0
    assert _field(inp, "incident_bond_type_counts")[80, sg, 1] == 1
    ref = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "CNLFSC",
        "bonds": [{"atoms": ["1@SG", "6@SG"], "order": 1}]})
    for f in ("canonical_polymer_ordering", "bond_mask", "bond_order",
              "token_bond_path_length", "incident_bond_type_counts"):
        assert np.array_equal(_field(inp, f), _field(ref, f)), f


# ------------------------------------------------------------------ isopeptide
# native fixture cpsea_isopeptide_6rlv: ring KYMMD (+caps), LYS.NZ-ASP.CG amide

def test_isopeptide_both_apis():
    nz, cg = _slot("LYS", "NZ"), _slot("ASP", "CG")
    ref = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "KYMMD",
        "bonds": [{"atoms": ["1@NZ", "5@CG"], "order": 1}]})
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "KYMMD"}).add_bonds(
        (80, nz, 84, cg, 1))
    bm, cpo = _field(inp, "bond_mask"), _field(inp, "canonical_polymer_ordering")
    assert bm[80, 84] and _field(inp, "bond_order")[80, 84] == 1
    assert cpo[80, 84] == 0
    assert (_field(inp, "incident_bond_type_counts")[80, nz, 1] == 1
            and _field(inp, "incident_bond_type_counts")[84, cg, 1] == 1)
    for f in ("canonical_polymer_ordering", "bond_mask", "bond_order",
              "token_bond_path_length", "incident_bond_type_counts"):
        assert np.array_equal(_field(inp, f), _field(ref, f)), f


# --------------------------------------------------------- pdb-dict route

def test_pdb_dict_bonds_and_cyclic():
    inp = RFD4Input.from_chains(40, {"pdb": "data/8znl.pdb",
        "select": ["A48-52"],
        "bonds": [{"atoms": ["A48@NH1", "A51@OE1"], "order": 1}],
        "cyclic": True})
    bm, cpo = _field(inp, "bond_mask"), _field(inp, "canonical_polymer_ordering")
    assert bm[40, 43] and _field(inp, "bond_order")[40, 43] == 1
    assert cpo[40, 44] == 1 and cpo[44, 40] == -1 and not bm[40, 44]
    assert _field(inp, "token_bond_path_length")[41, 43] == 2  # through closure

    inp2 = RFD4Input.from_chains(40, {"pdb": "data/8znl.pdb",
        "select": ["A48-52"], "unindexed_motif": True, "ordered": True,
        "bonds": [{"atoms": ["A48@NH1", "A51@OE1"], "order": 2}]})
    assert _field(inp2, "bond_mask")[40, 43]
    assert _field(inp2, "bond_order")[40, 43] == 2


# ------------------------------------------------------------------- errors

def test_bond_errors():
    # same-token bonds are supported: counts only, no pair-track entry
    inp = RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK",
        "bonds": [{"atoms": ["1@N", "1@C"], "order": 1}]})
    assert _field(inp, "incident_bond_type_counts")[80, 0, 1] == 1
    assert not _field(inp, "bond_mask")[80, 80]
    with pytest.raises(ValueError, match="exactly one atom"):
        RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK",
            "bonds": [{"atoms": ["1@N", "2@XX"], "order": 1}]})
    with pytest.raises(ValueError, match="outside 1..3"):
        RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK",
            "bonds": [{"atoms": ["1@N", "9@C"], "order": 1}]})
    with pytest.raises(ValueError, match="order must be 1, 2 or 3"):
        RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK"}).add_bonds(
            (0, 0, 1, 2, 4))
    with pytest.raises(ValueError, match="outside 0.."):
        RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK"}).add_bonds(
            (0, 0, 99, 2, 1))
    with pytest.raises(ValueError, match="cannot be combined with explicit"):
        RFD4Input.from_chains(80, {"kind": "aa", "sequence": "MEK",
            "auto_condition": True, "h_bond": "1"})
