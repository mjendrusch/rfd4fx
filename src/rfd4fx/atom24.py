"""Atom24: fixed-size per-residue atom arrays in CCD order.

Decoded structures are converted to a (num_residues, 24, 3) layout matching
each residue's PDB chemical component order, which drives the PDB/mmCIF
export at :mod:`rfd4fx.pdb_io`."""

from __future__ import annotations

import functools

import equinox as eqx
import numpy as np

ATOM24 = 24
_H_ELEMENTS = frozenset(("H", "D"))
_POLYMER_TOKEN_TYPES = (0, 1, 2)  # aa, rna, dna (TOKEN_TYPE_IDS)
_NONPOLYMER_TOKEN_TYPE = 3        # atomized


@functools.lru_cache(maxsize=None)
def ccd_atom_names(res_name: str) -> list[str] | None:
    """Heavy-atom names of one residue type in CCD file order (cached).

    Returns None when the token has no CCD entry — the caller falls back to
    the atom37 layout order.
    """
    try:
        from .ccd import get_ccd_entry

        rec = get_ccd_entry(res_name)
    except Exception:
        return None
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in rec.atom_names]
    elems = [e.decode() if isinstance(e, bytes) else str(e)
             for e in rec.elements]
    return [n for n, e in zip(names, elems) if e.upper() not in _H_ELEMENTS]


def _layout_and_elements(res_name: str, vocab_index: int):
    """(atom37 layout names, per-slot atomic numbers) for one residue type."""
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    layout = enc.token_atoms.get(res_name)
    if layout is None:
        return [], np.zeros(0, np.int32)
    names = [str(a).strip() for a in layout]
    if hasattr(enc, "idx_to_element"):
        elem_row = np.asarray(enc.idx_to_element)[vocab_index]
        elements = np.array([
            _element_to_z(str(e).strip()) if str(e).strip() else 0
            for e in elem_row], np.int32)
    else:  # pragma: no cover - enc always carries idx_to_element
        elements = np.zeros(len(names), np.int32)
    return names, elements


_Z_CACHE: dict[str, int] = {}


def _element_to_z(symbol: str) -> int:
    if symbol not in _Z_CACHE:
        from .spec import _atomic_number_from_symbol

        _Z_CACHE[symbol] = _atomic_number_from_symbol(symbol)
    return _Z_CACHE[symbol]


def residue_atom24_plan(res_name: str, vocab_index: int):
    """((24,) atom-order slot plan, (24,) atomic numbers) for one residue.

    The plan maps each atom24 position to its atom37 layout slot (-1 when the
    atom is absent from the layout). CCD order where available, layout order
    otherwise; layout atoms missing from the CCD are appended at the end.
    """
    layout, layout_elements = _layout_and_elements(res_name, vocab_index)
    name_to_slot = {n: i for i, n in enumerate(layout) if n}
    ccd = ccd_atom_names(res_name)
    if ccd is None:
        order = [i for i, n in enumerate(layout) if n][:ATOM24]
        plan = np.array(order + [-1] * (ATOM24 - len(order)), np.int32)
        elems = np.zeros(ATOM24, np.int32)
        elems[:len(order)] = layout_elements[order]
        return plan, elems
    seen = set()
    plan, elems = [], []
    for n in ccd:
        plan.append(name_to_slot.get(n, -1))
        elems.append(_element_to_z(_ccd_element_of(res_name, n)))
        seen.add(n)
    for i, n in enumerate(layout):  # layout-only atoms appended at the end
        if n and n not in seen and len(plan) < ATOM24:
            plan.append(i)
            elems.append(int(layout_elements[i]))
            seen.add(n)
    plan = np.array(plan + [-1] * (ATOM24 - len(plan)), np.int32)[:ATOM24]
    elems = np.array(elems + [0] * (ATOM24 - len(elems)), np.int32)[:ATOM24]
    return plan, elems


@functools.lru_cache(maxsize=None)
def _ccd_element_of(res_name: str, atom_name: str) -> str:
    try:
        from .ccd import get_ccd_entry

        rec = get_ccd_entry(res_name)
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in rec.atom_names]
        elems = [e.decode() if isinstance(e, bytes) else str(e)
                 for e in rec.elements]
        return elems[names.index(atom_name)]
    except Exception:
        return atom_name[:1].upper()


class Atom24(eqx.Module):
    """Per-residue fixed-size atom arrays in CCD order (eqx.Module).

    coords (n_res, 24, 3) float32 Å; mask (n_res, 24) bool; restype (n_res,)
    int32 unified-vocabulary class indices; elements (n_res, 24) int32
    atomic numbers (0 = absent). Unbatched for single-sample results,
    leading batch dim of size b otherwise.
    """

    coords: np.ndarray
    mask: np.ndarray
    restype: np.ndarray
    elements: np.ndarray

    def __init__(self, coords: np.ndarray, mask: np.ndarray,
                 restype: np.ndarray, elements: np.ndarray):
        self.coords = np.asarray(coords, np.float32)
        self.mask = np.asarray(mask, bool)
        self.restype = np.asarray(restype, np.int32)
        self.elements = np.asarray(elements, np.int32)
        n_res = self.restype.shape[-1]
        assert self.coords.shape[-3:] == (n_res, ATOM24, 3), self.coords.shape
        assert self.mask.shape[-2:] == (n_res, ATOM24), self.mask.shape
        assert self.elements.shape[-2:] == (n_res, ATOM24)

    def to_atoms(self) -> "Atoms":
        """Flat per-atom view: 24 slots per residue, order preserved; absent
        atoms keep their (0.0, False) slots. Batched inputs keep the batch
        dim."""
        coords = self.coords.reshape(*self.coords.shape[:-3],
                                     self.restype.shape[-1] * ATOM24, 3)
        mask = self.mask.reshape(*self.mask.shape[:-2],
                                 self.restype.shape[-1] * ATOM24)
        elements = self.elements.reshape(*self.elements.shape[:-2],
                                         self.restype.shape[-1] * ATOM24)
        restype = np.repeat(self.restype, ATOM24, axis=-1)
        return Atoms(coords=coords, mask=mask, restype=restype,
                     elements=elements)


class Atoms(eqx.Module):
    """Flat per-atom arrays (eqx.Module): coords (..., N, 3) Å,
    mask (..., N), restype (..., N) per-atom residue-type integers,
    elements (..., N) atomic numbers (0 = absent)."""

    coords: np.ndarray
    mask: np.ndarray
    restype: np.ndarray
    elements: np.ndarray

    def __init__(self, coords: np.ndarray, mask: np.ndarray,
                 restype: np.ndarray, elements: np.ndarray):
        self.coords = np.asarray(coords, np.float32)
        self.mask = np.asarray(mask, bool)
        self.restype = np.asarray(restype, np.int32)
        self.elements = np.asarray(elements, np.int32)
        n = self.mask.shape[-1]
        assert self.coords.shape[-2:] == (n, 3), self.coords.shape
        assert self.restype.shape[-1] == n and self.elements.shape[-1] == n


def to_atom24(dec, batch, aa_pred=None) -> Atom24:
    """Convert a single-sample decode into the Atom24 format.

    Residues = the emitted real polymer tokens (token_type 0/1/2), in token
    order. Residue identity = `aa_pred` (the trunk's token-type-restricted
    readout) falling back to the decoder's argmax head. Each residue's atoms
    are placed in CCD order (name-matched into the atom37 layout); absent
    atoms (no layout slot, not predicted present, non-finite or degenerate
    coordinates) are 0.0 / False.
    """
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
                  else dict(enumerate(enc.idx_to_token)))
    coors = np.asarray(dec.coors_nm)[0] * 10.0                 # nm -> Å
    emit = np.asarray(dec.emit_mask)[0]
    gate = np.asarray(dec.atom_mask)[0] & emit[:, None]
    real = np.asarray(batch.padding_mask)[0]
    tt = np.asarray(batch.token_type)[0]
    src = (np.asarray(aa_pred)[0] if aa_pred is not None and np.asarray(aa_pred).ndim == 2
           else (np.asarray(aa_pred) if aa_pred is not None
                 else np.asarray(dec.aatype_max)[0]))
    is_polymer = real & emit & np.isin(tt, _POLYMER_TOKEN_TYPES)
    res_tokens = np.flatnonzero(is_polymer)

    coords = np.zeros((len(res_tokens), ATOM24, 3), np.float32)
    mask = np.zeros((len(res_tokens), ATOM24), bool)
    restype = np.zeros(len(res_tokens), np.int32)
    elements = np.zeros((len(res_tokens), ATOM24), np.int32)
    for r, t in enumerate(res_tokens):
        v = int(src[t])
        restype[r] = v
        res_name = str(idx_to_tok.get(v, ""))
        plan, elems = residue_atom24_plan(res_name, v)
        elements[r] = elems
        for s24, s37 in enumerate(plan):
            if s37 < 0:
                continue
            xyz = coors[t, s37]
            if gate[t, s37] and np.isfinite(xyz).all() and np.abs(xyz).sum() > 0:
                coords[r, s24] = xyz
                mask[r, s24] = True
    return Atom24(coords=coords, mask=mask, restype=restype, elements=elements)


def nonpolymer_atoms(batch, x_bb: np.ndarray):
    """Per-atom arrays for the non-polymer (atomized) tokens of one sample.

    Returns (coords (m, 3) Å, elements (m,) int32, restype (m,) int32) —
    coordinates from the flow state, elements from the compiled batch's
    element channel, residue type = the token's conditioned sequence value
    (<A>, the atomized-vocabulary index).
    """
    real = np.asarray(batch.padding_mask)[0]
    tt = np.asarray(batch.token_type)[0]
    toks = np.flatnonzero(real & (tt == _NONPOLYMER_TOKEN_TYPE))
    if toks.size == 0:
        z = np.zeros(0, np.int32)
        return np.zeros((0, 3), np.float32), z, z
    element37 = np.asarray(batch.element)[0]
    elements = element37[toks, 1].astype(np.int32)
    cond_seq = np.asarray(batch.cond_seq)[0] if getattr(batch, "cond_seq", None) is not None \
        else np.zeros(len(toks), np.int32)
    restype = cond_seq[toks].astype(np.int32)
    return np.asarray(x_bb)[toks].astype(np.float32), elements, restype
