"""Structure export at the numpy/biotite file-I/O boundary.

Mirrors the native export path, including the OP3/O3' conventions for
multi-residue nucleic-acid chains."""

from __future__ import annotations

import functools

import numpy as np

_PROTEIN_TOKEN_TYPES = (0,)
_NA_TOKEN_TYPES = (1, 2)  # rna, dna (rfproteina TOKEN_TYPE_IDS)
_TWO_CHAR_ELEMS = frozenset(("CL", "BR", "SE", "MG", "ZN", "FE"))


@functools.lru_cache(maxsize=1)
def _atom37_vocab_tables():
    """Precompute (V+1, 37) atom-name / element lookup tables from UNIFIED_ATOM37_ENCODING."""
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING as enc

    idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
                  else dict(enumerate(enc.idx_to_token)))
    v_size = max(idx_to_tok) + 1 if idx_to_tok else 0
    vocab_names = np.full(v_size + 1, "", dtype="U8")
    slot_names = np.full((v_size + 1, 37), "", dtype="U6")
    slot_elems = np.full((v_size + 1, 37), "", dtype="U4")
    tok_to_idx: dict[str, int] = {}

    for idx, tok in idx_to_tok.items():
        res_name = str(tok)
        vocab_names[idx] = res_name
        tok_to_idx[res_name] = idx
        atoms = enc.token_atoms.get(res_name)
        if atoms is None:
            continue
        for s in range(37):
            an = str(atoms[s]).strip()
            if not an:
                continue
            slot_names[idx, s] = an
            letters = "".join(c for c in an if c.isalpha()).upper()
            slot_elems[idx, s] = letters[:2] if letters[:2] in _TWO_CHAR_ELEMS else letters[:1]
    return vocab_names, slot_names, slot_elems, tok_to_idx


def _chain_assignments(token_type: np.ndarray, real: np.ndarray,
                       chain_ids) -> np.ndarray:
    """Per-token chain-id strings. `chain_ids` may be None (single "A"),
    a list/str per unique token-type-respecting chain, or a (n,) array."""
    n = token_type.shape[0]
    if chain_ids is not None:
        arr = np.asarray(chain_ids)
        if arr.ndim == 0:
            return np.full(n, str(arr))
        if arr.dtype.kind in ("U", "S", "O") and arr.shape == (n,):
            return arr.astype(str)
    # derive letters from first-appearance order of real tokens (one chain
    # per maximal run is NOT assumed — callers pass explicit ids for
    # multi-unit layouts; the default assumes one polymer chain)
    return np.full(n, "A")


def drop_terminal_oxygen(arr, chain_kind: dict[str, str]):
    """Native terminal-oxygen cleanup on an assembled AtomArray.

    `chain_kind` maps chain_id -> "na" | "protein". Mirrors atomworks
    `polymer_terminal_oxygen_mask` (filters.py:555): OP3 is dropped from
    MULTI-RESIDUE nucleic-acid chains and OXT from multi-residue protein
    chains; free single-residue chains (free nucleotides / amino acids) and
    atomized tokens keep their terminal atoms. No renaming — O3' (slot 5)
    is always kept.
    """
    if len(arr) == 0:
        return arr
    keep = np.ones(len(arr), bool)
    for cid, kind in chain_kind.items():
        sel = np.asarray(arr.chain_id == cid)
        if not sel.any() or len(np.unique(arr.res_id[sel])) <= 1:
            continue  # free nucleotides / single residues keep their OP3
        if kind == "na":
            keep[sel] &= np.asarray(arr.atom_name[sel]) != "OP3"
        elif kind == "protein":
            keep[sel] &= np.asarray(arr.atom_name[sel]) != "OXT"
    return arr[keep]


def decoded_to_atom_array(dec, batch=None, *, aa_pred=None,
                          chain_ids=None, identity=None) -> "object":
    """Decode a DecoderOutput into a biotite AtomArray (coordinates in Å).

    Args:
      dec: decoder output with `coors_nm` (1, n, 37, 3), `aatype_max`
        (1, n), `atom_mask` (1, n, 37) and `emit_mask` (1, n).
      batch: optional FlowBatch/RFD4Input-batch carrying `padding_mask` and
        `token_type` (used for padding + the per-chain kind of the terminal-
        oxygen cleanup). May be None.
      aa_pred: optional (1, n) / (n,) trunk sequence readout used as the
        residue identity instead of the decoder's own argmax head (the
        decoder can degenerate to layout-less tokens under asymmetric
        latents; aa_pred is the token-type-restricted readout the decode
        was conditioned on).
      chain_ids: optional per-token chain ids ((n,) array of str, or a
        single str). Defaults to "A" for every token.
      identity: optional (n,) token-name array overriding both aa_pred and
        aatype_max (explicit residue identities).

    Native semantics mirrored (structure_export.py:112-138):
      - identity -> UNIFIED_ATOM37_ENCODING.token_atoms atom names;
      - a slot is written iff dec.atom_mask (residue_atom_mask & predicted
        presence) & dec.emit_mask (& padding) AND its coordinates are finite
        and non-degenerate;
      - terminal-oxygen cleanup: OP3 dropped from multi-residue nucleic-acid
        chains, OXT dropped from multi-residue protein chains (atomworks
        polymer_terminal_oxygen_mask; no renaming — O3' is slot 5 and is
        always kept).
    """
    import biotite.structure as bs

    vocab_names, slot_names_tbl, slot_elems_tbl, tok_to_idx = _atom37_vocab_tables()
    sentinel = vocab_names.shape[0] - 1

    coors = np.asarray(dec.coors_nm)[0] * 10.0          # nm -> Å
    emit = np.asarray(dec.emit_mask)[0]
    gate = np.asarray(dec.atom_mask)[0] & emit[:, None]
    n = coors.shape[0]
    real = (np.asarray(batch.padding_mask)[0] if batch is not None
            else np.ones(n, bool))
    token_type = (np.asarray(batch.token_type)[0] if batch is not None
                  else np.zeros(n, np.int32))
    if identity is not None:
        res_names = np.asarray(identity).astype(str)
        v_idx = np.fromiter(
            (tok_to_idx.get(str(nm), sentinel) for nm in res_names),
            dtype=np.int32, count=n,
        )
    else:
        src = (np.asarray(aa_pred)[0] if aa_pred is not None and np.asarray(aa_pred).ndim == 2
               else (np.asarray(aa_pred) if aa_pred is not None
                     else np.asarray(dec.aatype_max)[0])).astype(np.int32)
        v_idx = np.where((src >= 0) & (src < sentinel), src, sentinel)
        res_names = vocab_names[v_idx]
    chain_of = _chain_assignments(token_type, real, chain_ids)

    an_grid = slot_names_tbl[v_idx]
    valid_slot = (
        (real & emit)[:, None]
        & gate
        & (an_grid != "")
        & np.isfinite(coors).all(axis=-1)
        & (np.abs(coors).sum(axis=-1) > 0)
    )
    tok_idx, slot_idx = np.nonzero(valid_slot)
    arr = bs.AtomArray(len(tok_idx))
    if len(tok_idx):
        arr.coord[:] = coors[tok_idx, slot_idx]
        arr.chain_id[:] = chain_of[tok_idx]
        arr.res_id[:] = tok_idx + 1
        arr.res_name[:] = res_names[tok_idx]
        arr.atom_name[:] = an_grid[tok_idx, slot_idx]
        arr.element[:] = slot_elems_tbl[v_idx[tok_idx], slot_idx]

    # native terminal-oxygen cleanup: per chain, multi-residue only; kind
    # from the compiled token_type (na if any rna/dna token, else protein)
    chain_kind: dict[str, str] = {}
    for cid in np.unique(chain_of[real]):
        toks = np.flatnonzero(real & (chain_of == cid))
        tt = set(np.unique(token_type[toks]).tolist())
        if tt & set(_NA_TOKEN_TYPES):
            chain_kind[str(cid)] = "na"
        elif tt & set(_PROTEIN_TOKEN_TYPES):
            chain_kind[str(cid)] = "protein"
    return drop_terminal_oxygen(arr, chain_kind)
