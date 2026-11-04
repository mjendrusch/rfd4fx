"""PDB-writing regression tests: OP3 / OXT terminal-oxygen handling (MS7.1).

Native export semantics (rfproteina/utils/structure_export.py +
atomworks polymer_terminal_oxygen_mask): the unified vocabulary places OP3
at atom37 slot 36 of EVERY canonical NA token (CCD-faithful: the 5'-mono-
phosphate components carry OP3 and O3'), and the decoder's residue_atom_mask
marks it legal — so the DECODER may emit slot 36, but the EXPORT must drop
OP3 from multi-residue nucleic-acid chains (and OXT from multi-residue
protein chains), keeping it only on free single-residue nucleotides and
atomized tokens. No renaming ever occurs; O3' (slot 5) is always kept.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest


@dataclasses.dataclass
class _Dec:
    """Minimal DecoderOutput stand-alone: all-ones emit, atom_mask = the
    vocabulary residue_atom_mask for the given identities, coordinates with
    finite values everywhere so ONLY the masks decide what gets written."""
    coors_nm: np.ndarray
    aatype_max: np.ndarray
    atom_mask: np.ndarray
    emit_mask: np.ndarray


def _make_dec(aa: list[str], n: int, op3_coord: float = 5.0) -> _Dec:
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
                  else dict(enumerate(enc.idx_to_token)))
    tok_to_idx = enc.token_to_idx
    seq = np.array([int(tok_to_idx[a]) for a in aa] + [0] * (n - len(aa)), np.int32)
    names = np.array([str(x).strip() for x in
                      (enc.token_atoms.get(a) for a in aa)], dtype=object)
    # vocabulary residue mask: slot legal iff token_atoms names it
    amask = np.zeros((n, 37), bool)
    for t, a in enumerate(aa):
        layout = enc.token_atoms.get(a)
        amask[t] = np.array([bool(str(x).strip()) for x in layout])
    coords = np.zeros((1, n, 37, 3), np.float32)
    for t in range(n):
        for s in range(37):
            coords[0, t, s] = (1.0 + t, 2.0 + s, op3_coord if s == 36 else 3.0)
    return _Dec(coors_nm=coords / 10.0, aatype_max=seq[None],
                atom_mask=amask[None], emit_mask=np.ones((1, n), bool))


def _batch_for(aa: list[str], n: int):
    """A real compiled batch carrying token_type for the given identities."""
    from rfd4fx.rfd4_input import RFD4Input

    chain: dict = {"kind": "rna"}
    if aa and all(a in ("A", "C", "G", "U", "N") for a in aa):
        chain = {"kind": "rna", "num_residues": n}
    elif aa and all(a in ("DA", "DC", "DG", "DT", "DN") for a in aa):
        chain = {"kind": "dna", "num_residues": n}
    else:
        chain = {"num_residues": n}
    return RFD4Input.from_chains(chain, np_seed=0).to_batch()


class TestTerminalOxygenCleanup:
    def test_rna_multimer_drops_op3_keeps_o3p(self):
        from rfd4fx.pdb_io import decoded_to_atom_array

        aa = ["A", "C", "G", "U"]
        arr = decoded_to_atom_array(_make_dec(aa, 4), _batch_for(aa, 4))
        assert "OP3" not in set(arr.atom_name)          # the spurious atom
        assert (arr.atom_name == "O3'").sum() == 4      # slot 5 kept everywhere
        assert (arr.atom_name == "P").sum() == 4

    def test_dna_multimer_drops_op3_keeps_o3p(self):
        # native applies the SAME rule to DNA (no OP3 writing for polymers,
        # O3' kept — deoxyribose carries a 3'-OH)
        from rfd4fx.pdb_io import decoded_to_atom_array

        aa = ["DA", "DC", "DG", "DT"]
        arr = decoded_to_atom_array(_make_dec(aa, 4), _batch_for(aa, 4))
        assert "OP3" not in set(arr.atom_name)
        assert (arr.atom_name == "O3'").sum() == 4
        # DNA layout: no O2' (slot 3 is empty) — RNA/DNA layouts differ
        assert "O2'" not in set(arr.atom_name)

    def test_free_nucleotide_keeps_op3(self):
        from rfd4fx.pdb_io import decoded_to_atom_array

        aa = ["A"]
        arr = decoded_to_atom_array(_make_dec(aa, 1), _batch_for(aa, 1))
        assert (arr.atom_name == "OP3").sum() == 1      # single-residue chain
        assert (arr.atom_name == "O3'").sum() == 1

    def test_protein_multimer_drops_oxt(self):
        import biotite.structure as bs

        from rfd4fx.pdb_io import drop_terminal_oxygen

        arr = bs.AtomArray(6)
        arr.chain_id[:] = "A"
        arr.res_id[:] = [1, 1, 2, 2, 3, 3]
        arr.res_name[:] = ["ALA", "ALA", "ARG", "ARG", "ASN", "ASN"]
        arr.atom_name[:] = ["N", "OXT", "N", "OXT", "N", "OXT"]
        arr.element[:] = ["N", "O", "N", "O", "N", "O"]
        arr2 = drop_terminal_oxygen(arr, {"A": "protein"})
        assert "OXT" not in set(arr2.atom_name)
        assert len(arr2) == 3
        # single-residue protein keeps OXT (free amino acid)
        arr1 = arr[arr.res_id == 1]
        arr1.res_id[:] = 1
        assert (drop_terminal_oxygen(arr1, {"A": "protein"}).atom_name == "OXT").sum() == 1

    def test_aa_pred_identity_overrides_argmax(self):
        from rfd4fx.pdb_io import decoded_to_atom_array

        dec = _make_dec(["A", "C"], 2)
        dec.aatype_max = np.zeros((1, 2), np.int32)     # <M> argmax (layout-less)
        aa_pred = None
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc
        tok_to_idx = enc.token_to_idx
        aa_pred = np.array([[tok_to_idx["A"], tok_to_idx["C"]]], np.int32)
        arr = decoded_to_atom_array(dec, _batch_for(["A", "C"], 2), aa_pred=aa_pred)
        assert set(arr.res_name) == {"A", "C"}
        assert len(arr) > 0

    def test_native_parity(self):
        """The helper matches the native export path exactly:
        atomworks atom_array_from_encoding + polymer_terminal_oxygen_mask."""
        from atomworks.enums import ChainType
        from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc
        from atomworks.ml.transforms.encoding import atom_array_from_encoding
        from atomworks.ml.transforms.filters import polymer_terminal_oxygen_mask

        from rfd4fx.pdb_io import decoded_to_atom_array

        for aa, ctype in ((["A", "C", "G", "U"], ChainType.RNA),
                          (["DA", "DC", "DG"], ChainType.DNA),
                          (["A"], ChainType.RNA)):
            n = len(aa)
            dec = _make_dec(aa, n)
            batch = _batch_for(aa, n)
            ours = decoded_to_atom_array(dec, batch)
            # native path: coords to Å, vocabulary names, mask gating, cleanup
            tok_to_idx = enc.token_to_idx
            seq = np.array([int(tok_to_idx[a]) for a in aa], np.int32)
            enc_mask = np.asarray(dec.atom_mask)[0] & np.asarray(dec.emit_mask)[0][:, None]
            native = atom_array_from_encoding(
                encoded_coord=np.asarray(dec.coors_nm)[0] * 10.0,
                encoded_seq=seq,
                encoding=enc,
                chain_id="A",
                encoded_mask=enc_mask,
                chain_type=np.full(n, ctype, dtype=object),
            )
            native = native[native.occupancy > 0]
            native = native[~polymer_terminal_oxygen_mask(native)]
            ours_sorted = sorted(zip(ours.res_id, ours.atom_name))
            native_sorted = sorted(zip(native.res_id, native.atom_name))
            assert ours_sorted == native_sorted, aa
            assert np.allclose(np.asarray(ours.coord), np.asarray(native.coord),
                               atol=1e-5), aa
