"""pdb/cif selection ("select"), fixed, ordered, and "ligands" chemistry
tests (MS14). Fixture: the theozyme scaffold (chains A=72 residues, M=5,
B=1 'LIG' HETATM)."""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

FIXTURES = pathlib.Path("tests/fixtures/pdb")
PDB = str(FIXTURES / "design_000_scaffold_00.pdb")


def _n_real(inp) -> int:
    return int(np.asarray(inp.padding_mask).sum())


class TestSelectionParser:
    def test_grammar(self):
        import gemmi

        from rfd4fx.rfd4_input import _parse_pdb_selection

        model = gemmi.read_structure(PDB)[0]
        sel = _parse_pdb_selection(["A", "M1-3", "B1@N,CA,C"], model)
        assert set(sel) == {"A", "M", "B"}
        assert sel["A"] == {"*": None}                       # whole chain
        assert sel["M"][1] is None and sel["M"][2] is None and sel["M"][3] is None
        assert 4 not in sel["M"]
        assert sel["B"][1] == frozenset({"N", "CA", "C"})

    def test_longest_prefix_and_errors(self):
        import gemmi

        from rfd4fx.rfd4_input import _parse_pdb_selection

        model = gemmi.read_structure(PDB)[0]
        with pytest.raises(ValueError):
            _parse_pdb_selection(["X5"], model)
        with pytest.raises(ValueError):
            _parse_pdb_selection(["A5@" ], model)
        with pytest.raises(ValueError):
            _parse_pdb_selection(["A5-"], model)


class TestIndexedSelection:
    def test_select_filters_chains_and_residues(self):
        from rfd4fx.rfd4_input import RFD4Input

        # binder(10) + chain A whole: 10 + 72 = 82
        inp = RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A"]})
        assert _n_real(inp) == 82
        # residues A1-10 only: 10 + 10 = 20
        inp2 = RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1-10"]})
        assert _n_real(inp2) == 20
        # atom selection: slots restricted to the named atoms
        inp3 = RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1@N,CA,C"]})
        mask = np.asarray(inp3.atom37_mask)[0, 10:20]
        assert int(mask[0].sum()) == 3
        # select overrides chains
        inp4 = RFD4Input.from_chains(
            10, {"pdb": PDB, "chains": ["M"], "select": ["A1-5"]})
        assert _n_real(inp4) == 15

    def test_fixed_false_internal_coords(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1-10"],
                                         "fixed": False})
        cc = np.asarray(inp.coord_cond)[0]
        intcrd = np.asarray(inp.cond_internal_coords)[0]
        cg = np.asarray(inp.coord_groups)[0]
        # absolute conditioning absent on the motif; internal payload present
        assert not cc[10:20].any()
        assert np.isfinite(intcrd[10:20][np.asarray(inp.atom37_mask)[0, 10:20]]).all()
        # ONE positive coordinate group for the whole selection
        groups = np.unique(cg[10:20][np.asarray(inp.atom37_mask)[0, 10:20]])
        assert len(groups) == 1 and groups[0] > 0
        # the binder chain keeps default groups (-1 = unconditioned)
        assert (cg[:10] == -1).all()

    def test_fixed_default_unchanged(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1-10"]})
        cc = np.asarray(inp.coord_cond)[0]
        assert cc[10:20].all()                               # absolute
        assert np.isfinite(np.asarray(inp.cond_coords)[0, 10:20]).any()
        assert (np.asarray(inp.coord_groups)[0, 10:20]
                [np.asarray(inp.atom37_mask)[0, 10:20]] == 0).all()


class TestUnindexedPdbMotif:
    def test_residue_tokens_and_ligand_fallback(self):
        """The 'LIG' HETATM (chain B, 33 atoms, no CCD entry) gets the
        element fallback: elements parsed per atom, zero charges, no bonds."""
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5", "B1"], "unindexed_motif": True})
        b = inp.to_batch()
        tt = np.asarray(b.token_type)[0]
        lig = np.flatnonzero(tt == 3)
        assert len(lig) == 33                                # the LIG atoms
        elem = np.asarray(b.element)[0][lig, 1]
        assert (elem > 0).any()                              # elements present
        charge = np.asarray(b.charge)[0][lig, 1]
        assert (charge == 0).all()                           # fallback: no charges
        bm = np.asarray(b.bond_mask)[0]
        assert not bm[np.ix_(lig, lig)].any()                # fallback: no bonds
        # motif residues are in-context; ligand tokens indexed
        ic = np.asarray(b.is_in_context_conditioning)[0]
        assert ic[10:15].all() and not ic[lig].any()

    def test_ligands_smiles_map(self, tmp_path):
        """The 'ligands' SMILES map takes precedence: bonds/charges recovered
        from the RDKit molecule via coordinate-perceived graph isomorphism."""
        import gemmi
        import biotite.structure.io.pdb as pdbio

        from rfd4fx.rfd4_input import RFD4Input

        # synthetic ethanol HETATM: C-C-O chain, resname EOH (no CCD entry)
        st = gemmi.Structure()
        st.add_model(gemmi.Model(0))
        chain = gemmi.Chain("Z")
        res = gemmi.Residue()
        res.name = "EOH"
        res.seqid = gemmi.SeqId(1, ' ')
        for name, elem, xyz in (("C1", "C", (0.0, 0.0, 0.0)),
                                ("C2", "C", (1.5, 0.0, 0.0)),
                                ("O1", "O", (2.4, 1.1, 0.0))):
            atom = gemmi.Atom()
            atom.name = name
            atom.element = gemmi.Element(elem)
            atom.pos = gemmi.Position(*xyz)
            res.add_atom(atom)
        chain.add_residue(res)
        st[0].add_chain(chain)
        st.setup_entities()
        pdb_path = tmp_path / "ethanol.pdb"
        (tmp_path / "ethanol.cif").write_text(
            st.make_mmcif_document().as_string())

        inp = RFD4Input.from_chains(
            10, {"cif": str(tmp_path / "ethanol.cif"), "select": ["Z1"],
                 "unindexed_motif": True, "ligands": {"EOH": "CCO"}})
        b = inp.to_batch()
        tt = np.asarray(b.token_type)[0]
        lig = np.flatnonzero(tt == 3)
        assert len(lig) == 3
        bm = np.asarray(b.bond_mask)[0]
        assert bm[np.ix_(lig, lig)].sum() == 4               # C-C + C-O (x2 sym)
        bo = np.asarray(b.bond_order)[0]
        assert (bo[np.ix_(lig, lig)][bm[np.ix_(lig, lig)]] == 1).all()
        charge = np.asarray(b.charge)[0][lig, 1]
        assert (charge == 0).all()

    def test_ordered_index_groups(self):
        from rfd4fx.rfd4_input import RFD4Input

        # ordered=True: one index group per contiguous same-chain run
        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "unindexed_motif": True,
                 "ordered": True})
        ig = np.asarray(inp.idx_group)[0, 10:15]
        assert len(np.unique(ig)) == 1                       # A1-5 contiguous
        # ordered=False: per-token groups
        inp2 = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "unindexed_motif": True,
                 "ordered": False})
        ig2 = np.asarray(inp2.idx_group)[0, 10:15]
        assert len(np.unique(ig2)) == 5
        # ligand runs always isolated: residue + LIG in one motif
        inp3 = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5", "B1"], "unindexed_motif": True,
                 "ordered": True})
        ig3 = np.asarray(inp3.idx_group)[0]
        tt = np.asarray(inp3.to_batch().token_type)[0]
        lig_local = np.flatnonzero(tt[10:] == 3) + 10
        res_local = np.flatnonzero(tt[10:] != 3) + 10
        assert not set(ig3[lig_local]) & set(ig3[res_local])

    def test_fixed_false_unindexed(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "unindexed_motif": True,
                 "fixed": False})
        cc = np.asarray(inp.coord_cond)[0]
        intcrd = np.asarray(inp.cond_internal_coords)[0]
        assert not cc[10:15].any()
        assert np.isfinite(intcrd[10:15]).any()


class TestStrHotspot:
    def test_indexed_route_str_hotspot(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-10"], "hotspot": "A3;A5@N,CA"})
        b = inp.to_batch()
        hot = np.asarray(b.hotspot)[0]
        # A3: all present atoms; A5: only N,CA
        mask = np.asarray(b.atom37_mask)[0]
        assert (hot[12] == mask[12]).all() and hot[12].any()   # token 12 = A3
        n5 = int(hot[14].sum())
        assert 0 < n5 <= 2 and hot[14, 0] and hot[14, 1]       # N + CA slots
        assert not hot[10:12].any() and not hot[15:20].any()   # non-hotspot

    def test_unindexed_route_str_hotspot(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "unindexed_motif": True,
                 "hotspot": "A2;A4"})
        b = inp.to_batch()
        hot = np.asarray(b.hotspot)[0, 10:15]
        mask = np.asarray(b.atom37_mask)[0, 10:15]
        assert (hot[1] == mask[1]).all() and (hot[3] == mask[3]).all()
        assert not hot[0].any() and not hot[2].any() and not hot[4].any()

    def test_out_of_motif_raises(self):
        from rfd4fx.rfd4_input import RFD4Input

        with pytest.raises(ValueError):
            RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1-10"],
                                       "hotspot": "A15"})
        with pytest.raises(ValueError):
            RFD4Input.from_chains(10, {"pdb": PDB, "select": ["A1-10"],
                                       "hotspot": "M1"})

    def test_non_str_hotspot_atom_level(self):
        from rfd4fx.rfd4_input import RFD4Input

        # (n,) values: per-residue hotspot over the motif window
        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-3"], "hotspot": [1, 0, 1]})
        hot = np.asarray(inp.hotspot)[0, 10:13]
        mask = np.asarray(inp.atom37_mask)[0, 10:13]
        assert (hot[0] == mask[0]).all() and (hot[2] == mask[2]).all()
        assert not hot[1].any()

    def test_auto_condition_hbond(self):
        """C_HB donor/acceptor roles on the hotspot residues' atoms, per the
        native tables (HIS ND1/NE2 = both; ARG NH1/NH2 = donor...)."""
        from rfd4fx.rfd4_input import RFD4Input

        # A3 = HIS (per the fixture: ALA1 HIS2 LEU3...): hotspot A3
        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "hotspot": "A3",
                 "auto_condition": True})
        hb = np.asarray(inp.hbond)[0, 12]
        mask = np.asarray(inp.atom37_mask)[0, 12]
        donors = {1, 3}
        acceptors = {2, 3}
        hit = {int(v) for v in hb[mask] if v > 0}
        assert hit and hit <= donors | acceptors
        # non-hotspot motif residues: no hbond conditioning
        assert not np.asarray(inp.hbond)[0, 10:12].any()

    def test_no_auto_condition_without_hotspot(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "auto_condition": True})
        assert not np.asarray(inp.hbond)[0].any()


# ------------------------------------- partially-indexed motifs (MS16)

BINDER = str(FIXTURES / "binder_01.pdb")
BACKBONE_ATOMS = ("N", "CA", "C", "O", "OXT")
_BACKBONE_SLOTS = {"N": 0, "CA": 1, "C": 2, "O": 4, "OXT": 36}


def _slot_of(atom_name):
    return _BACKBONE_SLOTS[atom_name]


def _atom_name_at(token, slot, resnum, pdb_path=BINDER, chain=None):
    """Atom name occupying (token, slot) — resolved from the file residue."""
    import gemmi
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
    model = gemmi.read_structure(pdb_path)[0]
    res = next(r for c in model for r in c
               if r.seqid.num == resnum and r.name not in ("HOH", "DOD", "WAT"))
    return str(ENC.token_atoms[res.name][slot]).strip()


def _residue_order(pdb_path, chain):
    """Residue objects of one chain in file order (solvent skipped)."""
    import gemmi
    return [r for r in gemmi.read_structure(pdb_path)[0][chain]
            if r.name not in ("HOH", "DOD", "WAT")]


def _interface_reference(pdb_path, donor="A", acceptor="B", cutoff=4.5):
    """Brute-force interface-residue set (heavy atoms, solvent skipped)."""
    import gemmi
    model = gemmi.read_structure(pdb_path)[0]
    acc = np.asarray([[a.pos.x, a.pos.y, a.pos.z]
                      for r in model[acceptor] if r.name not in ("HOH", "DOD", "WAT")
                      for a in r if a.element.name.upper() not in ("H", "D")])
    hits = set()
    for res in model[donor]:
        if res.name in ("HOH", "DOD", "WAT"):
            continue
        for a in res:
            if a.element.name.upper() in ("H", "D"):
                continue
            d2 = ((a.pos.x - acc[:, 0]) ** 2 + (a.pos.y - acc[:, 1]) ** 2
                  + (a.pos.z - acc[:, 2]) ** 2)
            if (d2 < cutoff ** 2).any():
                hits.add(res.seqid.num)
                break
    return hits


def _pdb_atom_positions(pdb_path, chain):
    """{(resnum, atom_name): xyz} for the heavy atoms of one chain."""
    import gemmi
    model = gemmi.read_structure(pdb_path)[0]
    return {(r.seqid.num, a.name.strip()): np.array([a.pos.x, a.pos.y, a.pos.z])
            for r in model[chain] if r.name not in ("HOH", "DOD", "WAT")
            for a in r if a.element.name.upper() not in ("H", "D")}


class TestDropBackbone:
    def test_sidechains_only(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-6"], "drop_backbone": True})
        mask = np.asarray(inp.atom37_mask)[0, 10:16]
        # no backbone slots on any motif token
        for slot_name in BACKBONE_ATOMS:
            assert not mask[:, _slot_of(slot_name)].any()
        assert mask.any()

    def test_glycine_skipped(self):
        """A1-8 contains GLY7: with the backbone dropped GLY has no atoms left
        and gets no token (A1-6 = 6 tokens, A7 skipped, A8 kept -> 7)."""
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-8"], "drop_backbone": True})
        n = int(np.asarray(inp.padding_mask).sum())
        assert n == 10 + 7


class TestInterfaceSelection:
    def test_matches_brute_force(self):
        from rfd4fx.rfd4_input import RFD4Input

        ref = _interface_reference(BINDER)
        assert len(ref) == 16
        inp = RFD4Input.from_chains(
            10, {"pdb": BINDER, "select": ["A"], "interface_to": "B"})
        n = int(np.asarray(inp.padding_mask).sum())
        assert n == 10 + len(ref)

    def test_without_select_uses_chains_key(self):
        from rfd4fx.rfd4_input import RFD4Input

        ref = _interface_reference(BINDER)
        inp = RFD4Input.from_chains(
            10, {"pdb": BINDER, "chains": ["A"], "interface_to": "B"})
        assert int(np.asarray(inp.padding_mask).sum()) == 10 + len(ref)

    def test_missing_interface_chain_raises(self):
        from rfd4fx.rfd4_input import RFD4Input

        with pytest.raises(ValueError):
            RFD4Input.from_chains(
                10, {"pdb": BINDER, "select": ["A"], "interface_to": "Z"})


class TestMotifSubSpecs:
    def test_shared_group_merged_frame(self):
        """Indexed B + unindexed A sidechains in ONE rigid body: after
        to_batch both motif ranges share a coordinate-group label and the
        internal-coordinate payload preserves pairwise distances ACROSS the
        two pieces (separate frames would not)."""
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(10, {
            "pdb": BINDER, "fixed": False,
            "motif": [
                {"select": ["B"]},
                {"select": ["A"], "interface_to": "B", "drop_backbone": True,
                 "indexed": False, "ordered": True},
            ]})
        n = int(np.asarray(inp.padding_mask).sum())
        assert n == 10 + 114 + 16
        b = inp.to_batch()
        cg = np.asarray(b.coord_groups)[0]
        mask = np.asarray(b.atom37_mask)[0]
        g_b = np.unique(cg[10:124][mask[10:124]])
        g_a = np.unique(cg[124:][mask[124:]])
        assert len(g_b) == 1 and len(g_a) == 1 and g_b[0] == g_a[0]
        # cross-piece distance preservation (file coordinates vs payload)
        ic = np.asarray(b.cond_internal_coords)[0]
        ref_b = _pdb_atom_positions(BINDER, "B")
        ref_a = _pdb_atom_positions(BINDER, "A")
        # token layout: [10, 124) = B residues in file order; [124, 140) = the
        # 16 interface residues in file order
        b_res = [r.seqid.num for r in _residue_order(BINDER, "B")]
        iface = _interface_reference(BINDER)
        a_iface = [r.seqid.num for r in _residue_order(BINDER, "A")
                   if r.seqid.num in iface]
        tb, ta = 10 + 40, 124 + 2
        sb = int(np.flatnonzero(mask[tb])[0])   # B: any conditioned atom
        sa = int(np.flatnonzero(mask[ta])[0])   # A: first SIDECHAIN atom
        an_b = _atom_name_at(tb, sb, b_res[40])
        an_a = _atom_name_at(ta, sa, a_iface[2])
        pb = ref_b[(b_res[40], an_b)]
        pa = ref_a[(a_iface[2], an_a)]
        d_intcrd = np.linalg.norm(ic[tb, sb] - ic[ta, sa])
        d_true = np.linalg.norm(pb - pa)
        assert abs(d_intcrd - d_true) < 1e-3


class TestRawAdderExplicitGroup:
    def test_unindexed_explicit_group(self):
        from rfd4fx.rfd4_input import RFD4Input

        pos = np.zeros((3, 37, 3), np.float32)
        pos[:, 1] = np.arange(3)[:, None] * [3.8, 0.0, 0.0]
        mask = np.zeros((3, 37), bool)
        mask[:, 1] = True
        inp = RFD4Input.from_chains(6)
        out = inp.add_unindexed_motif(pos, np.array([5, 5, 5]), mask,
                                      fixed=False, group=3)
        assert (np.asarray(out.coord_groups)[0, 6:9, 1] == 3).all()

    def test_shared_group_raw_adders_merged(self):
        from rfd4fx.rfd4_input import RFD4Input

        pos = np.zeros((3, 37, 3), np.float32)
        pos[:, 1] = np.arange(3)[:, None] * [3.8, 0.0, 0.0]
        mask = np.zeros((3, 37), bool)
        mask[:, 1] = True
        inp = RFD4Input.from_chains(6)
        out = inp.add_unindexed_motif(pos, np.array([5, 5, 5]), mask,
                                      fixed=False, group=3)
        # an indexed motif piece on the SAME group
        pos2 = np.zeros((9, 37, 3), np.float32)
        pos2[2, 1] = [10.0, 5.0, 1.0]
        mask2 = np.zeros((9, 37), bool)
        mask2[2, 1] = True
        where = np.zeros(9, bool)
        where[2] = True
        out2 = out.add_indexed_motif(pos2, np.zeros(9, np.int32), mask2,
                                     group=3, fixed=False, where=where)
        b = out2.to_batch()
        ic = np.asarray(b.cond_internal_coords)[0]
        # cross-piece pair: indexed token 2 vs motif token 7
        d_intcrd = np.linalg.norm(ic[2, 1] - ic[7, 1])
        d_true = np.linalg.norm(np.array([10.0, 5.0, 1.0]) - pos[1, 1])
        assert abs(d_intcrd - d_true) < 1e-3

    def test_negative_group_raises(self):
        from rfd4fx.rfd4_input import RFD4Input

        pos = np.zeros((2, 37, 3), np.float32)
        pos[:, 1] = 1.0
        mask = np.zeros((2, 37), bool)
        mask[:, 1] = True
        inp = RFD4Input.from_chains(6)
        with pytest.raises(ValueError):
            inp.add_unindexed_motif(pos, np.array([5, 5]), mask,
                                    fixed=False, group=-1)


class TestSeqProbabilitiesOnOverlays:
    def test_fixed_false_motif_identity_reaches_trunk(self):
        """The fixed=False pdb overlay bypasses the spec — its identity
        conditioning must still land in seq_probabilities (the trunk's C_SEQ
        feature), one-hot at the motif tokens."""
        from rfd4fx.rfd4_input import RFD4Input
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
        from rfd4fx.featurize import SEQ_PROB_ORDER

        inp = RFD4Input.from_chains(
            10, {"pdb": PDB, "select": ["A1-5"], "fixed": False, "group": 1})
        sp = np.asarray(inp.seq_probabilities)[0, 10:15]
        restype = [ENC.token_to_idx.get(r.name, 0)
                   for r in _residue_order(PDB, "A")[:5]]
        for t, v in enumerate(restype):
            name = str(ENC.idx_to_token[int(v)])
            key = f"condition_sequence-probability-{name}_1_residue"
            assert sp[t, SEQ_PROB_ORDER.index(key)] == 1.0
        # designable (de-novo) tokens stay unspecified
        assert np.isnan(np.asarray(inp.seq_probabilities)[0, :10]).all()

    def test_atom_level_aa_identity_reaches_trunk(self):
        from rfd4fx.rfd4_input import RFD4Input
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
        from rfd4fx.featurize import SEQ_PROB_ORDER

        inp = RFD4Input.from_chains(10)
        inp = inp.add_hotspot(1, aa=[5, 0, 0, 0, 0], start=2)
        sp = np.asarray(inp.seq_probabilities)[0]
        name = str(ENC.idx_to_token[5])
        key = f"condition_sequence-probability-{name}_1_residue"
        assert sp[2, SEQ_PROB_ORDER.index(key)] == 1.0
        assert np.isnan(sp[3]).all()
