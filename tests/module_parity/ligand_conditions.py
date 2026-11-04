"""Ligand autofeaturization tests: C_PISTACK / C_HB / C_SALT + "buried" (MS9).

Native reference (atomworks AnnotateInteractions ligand heuristics, PLIP v3):
acceptor = ANY N/O/S ligand atom; donor = N/O with >=1 H (implicit valence
here); pistack = all atoms of RDKit-aromatic 5/6-membered rings; salt =
formal-charge sign (1=POSITIVE / 2=NEGATIVE); buried = C_RAS 1 on every
ligand atom slot.
"""

from __future__ import annotations

import numpy as np
import pytest

CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def _roles(smiles: str) -> dict[str, np.ndarray]:
    from rfd4fx.ligand_conditions import ligand_roles
    from rfd4fx.spec import RFD4Spec

    spec = RFD4Spec()
    spec.add_ligand(smiles=smiles)
    lig = spec.ligands[0]
    return ligand_roles(lig.elements, lig.charges, lig.bonds)


class TestLigandRoles:
    def test_caffeine(self):
        roles = _roles(CAFFEINE)
        # fused 5+6 aromatic rings share 2 atoms -> 9 ring atoms
        assert int(roles["pistack"].sum()) == 9
        # broad native rule: every N/O/S is an acceptor (caffeine: 4 N + 2 O)
        assert int((roles["hbond"] == 2).sum()) == 6
        assert int((roles["hbond"] & 1).any()) == 0        # no donors (no N-H/O-H)
        assert not (roles["salt"] != 0).any()              # no formal charges

    def test_benzene(self):
        roles = _roles("c1ccccc1")
        assert int(roles["pistack"].sum()) == 6
        assert not roles["hbond"].any()
        assert not (roles["salt"] != 0).any()

    def test_acetate_salt_negative(self):
        roles = _roles("CC(=O)[O-]")
        # formal-charge rule: only the [O-] oxygen carries the -1 (the CCD/
        # RDKit convention; the carbonyl O stays neutral)
        assert sorted(np.flatnonzero(roles["salt"] == 2).tolist()) == [3]
        assert int((roles["salt"] == 1).sum()) == 0
        # carbonyl + carboxylate O are acceptors (the carboxyl C is not); no donors
        assert set(np.flatnonzero(roles["hbond"] == 2).tolist()) == {2, 3}

    def test_protonated_amine_positive_and_donor(self):
        roles = _roles("C[NH3+]")
        n_idx = 1
        assert roles["salt"][n_idx] == 1                   # POSITIVE
        assert roles["hbond"][n_idx] == 3                  # DONOR | ACCEPTOR = BOTH
        assert int((roles["hbond"] == 1).sum()) == 0       # C is neither
        assert not roles["pistack"].any()

    def test_purine_ring_sizes(self):
        # a non-aromatic 4-membered ring must not be pistack-conditioned
        roles = _roles("C1CC1")                            # cyclopropane: no aromatic rings
        assert not roles["pistack"].any()


class TestFromChainsIntegration:
    def test_channels_land_on_ligand_slots(self):
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            {"kind": "rna", "num_residues": 4},
            {"smiles": CAFFEINE, "auto_condition": True, "buried": True}, np_seed=0)
        b = inp.to_batch()
        lig = slice(4, 4 + 14)
        ps = np.asarray(b.pistack)[0]
        hb = np.asarray(b.hbond)[0]
        sa = np.asarray(b.saltbridge)[0]
        ra = np.asarray(b.rasa)[0]
        assert ps[lig, 1].sum() == 9 and ps.sum() == 9
        assert (hb[lig, 1] == 2).sum() == 6 and not hb[:4].any()
        assert not sa.any()                                # caffeine uncharged
        assert (ra[lig, 1] == 1).all() and not ra[:4].any()
        # every conditioned slot is materialized (atom37_mask)
        assert np.asarray(b.atom37_mask)[0][lig, 1].all()

    def test_defaults_stay_zero(self):
        from rfd4fx.rfd4_input import RFD4Input

        b = RFD4Input.from_chains(4, {"smiles": CAFFEINE}, np_seed=0).to_batch()
        for f in ("pistack", "hbond", "saltbridge", "rasa"):
            assert not np.asarray(getattr(b, f))[0].any(), f

    def test_unknown_ligand_key_raises(self):
        from rfd4fx.rfd4_input import RFD4Input

        with pytest.raises(ValueError):
            RFD4Input.from_chains({"smiles": CAFFEINE, "autocoondition": True},
                                  np_seed=0)

    def test_ccd_ligand_autofeaturization(self):
        # unified graph path: a CCD ligand gets roles through the same code as
        # SMILES ligands. AMP: adenine ring(s) pistack, N/O acceptors — and
        # NO salt roles (the CCD stores all-zero formal charges for AMP; the
        # native rule respects the charge annotation as-is)
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(4, {"ccd": "AMP", "auto_condition": True},
                                    np_seed=0)
        b = inp.to_batch()
        ps = np.asarray(b.pistack)[0]
        hb = np.asarray(b.hbond)[0]
        sa = np.asarray(b.saltbridge)[0]
        n_amp = int(np.asarray(b.padding_mask).sum()) - 4
        lig = slice(4, 4 + n_amp)
        assert ps[lig, 1].sum() > 0                        # adenine ring(s)
        assert (hb[lig, 1] != 0).sum() > 0                 # base N + phosphate O
        assert not sa.any()                                # CCD charges all zero
        assert not ps[:4].any() and not hb[:4].any()

    def test_ccd_ligand_salt_negative(self):
        # a CCD component WITH per-atom formal charges: SO4 carries -1 on two
        # oxygens -> C_SALT NEGATIVE on exactly those atoms
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(2, {"ccd": "SO4", "auto_condition": True},
                                    np_seed=0)
        b = inp.to_batch()
        sa = np.asarray(b.saltbridge)[0]
        n_so4 = int(np.asarray(b.padding_mask).sum()) - 2
        lig = slice(2, 2 + n_so4)
        assert (sa[lig, 1] == 2).sum() == 2                # O3, O4 per the CCD
        assert (sa[lig, 1] == 1).sum() == 0

    def test_salt_and_pistack_together(self):
        # a ligand with both an aromatic ring and a charged group
        from rfd4fx.rfd4_input import RFD4Input

        inp = RFD4Input.from_chains(
            2, {"smiles": "C1=CC=C(C=C1)C[NH3+]", "auto_condition": True},
            np_seed=0)
        b = inp.to_batch()
        ps = np.asarray(b.pistack)[0]
        sa = np.asarray(b.saltbridge)[0]
        hb = np.asarray(b.hbond)[0]
        lig = slice(2, 2 + 8)
        assert ps[lig, 1].sum() == 6                       # benzene ring
        assert (sa[lig, 1] == 1).sum() == 1                # the NH3+ nitrogen
        assert (hb[lig, 1] == 3).sum() == 1                # POSITIVE N is donor+acceptor
