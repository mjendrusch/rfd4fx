"""RFD4Spec: programmatic authoring of design specifications.

Holds annotated-CIF-style annotations per chain (no biotite at authoring
time) and featurizes them into model-ready host tensors."""

from __future__ import annotations

import dataclasses
import pathlib
import tempfile
from collections.abc import Sequence

import numpy as np

# SSEnum values (atomworks.ml.transforms.dssp.SSEnum), mirrored to keep the class import-light
SS_NONE, SS_HELIX, SS_STRAND, SS_LOOP = 0, 1, 2, 3

# BasePairEnum (rfproteina.conditions.BasePairEnum): 0=unspecified, 1=paired, 2=loop
BP_UNSPECIFIED, BP_PAIR, BP_LOOP = 0, 1, 2
# TOKEN_TYPE_IDS (rfproteina.encoding): aa=0, rna=1, dna=2
CHAIN_KIND_TOKEN_TYPE = {"aa": 0, "rna": 1, "dna": 2}

_ELEMENT_Z_CACHE: dict[str, int] = {}


def sequence_to_names(seq: str, kind: str = "aa") -> list[str]:
    """Parse a one-letter polymer string into 3-letter residue names.

    `kind` selects the vocabulary: "aa" (protein), "dna", "rna". The
    ambiguity code "X" (any kind) and "N" (dna/rna) are accepted as UNKNOWN
    positions (atomworks maps them to the UNK/DN/N sentinels); any other
    unresolvable letter raises ValueError."""
    from atomworks.enums import ChainType
    from atomworks.io.utils.sequence import get_3_from_1_letter_code

    chain_types = {"aa": ChainType.POLYPEPTIDE_L, "dna": ChainType.DNA,
                   "rna": ChainType.RNA}
    ct = chain_types.get(kind)
    if ct is None:
        raise ValueError(
            f"sequence: kind must be one of {sorted(chain_types)}, got {kind!r}")
    valid = (set("ACDEFGHIKLMNPQRSTVWY") | {"X"} if kind == "aa"
             else set("ACGT" if kind == "dna" else "ACGU") | {"N", "X"})
    names = []
    for ch1 in seq.upper():
        if ch1 not in valid:
            raise ValueError(f"sequence: unknown {kind} letter {ch1!r}")
        names.append(str(get_3_from_1_letter_code(ch1, ct)))
    return names


# atomworks' sentinel names for ambiguous letters — treated as "unknown":
# C_SEQ is left unset (the residue is designed) and they have no sidechain
_AMBIGUOUS_NAMES = frozenset({"UNK", "DN", "N"})


def sequence_to_indices(seq: str, kind: str = "aa") -> np.ndarray:
    """One-letter polymer string -> int32 array of RFD4 encoding indices.

    Ambiguous letters (X for aa; N/X for dna/rna) map to 0 = <M> = unset
    C_SEQ at that position."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    return np.array([0 if name in _AMBIGUOUS_NAMES
                     else ENC.token_to_idx.get(name, 0)
                     for name in sequence_to_names(seq, kind)], dtype=np.int32)


def parse_dot_bracket(dot_bracket: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Parse a dot-bracket string into per-position NA-SS states + base pairs.

    Port of rfproteina.inference_utils.nucleic_ss_input.parse_dot_bracket (numpy
    only), extended with the unknown-state characters ``x``/``_``/``X`` (user
    decision 2026-11-02): ``.`` → BP_LOOP (2, explicitly unpaired), bracket
    characters → BP_PAIR (1), ``x``/``_``/``X`` → BP_UNSPECIFIED (0, no
    statement). Bracket families ``()[]{}<>`` plus letter families ``Aa``/``Bb``
    (either case convention, auto-resolved per string — native ``_opener_for``)
    encode nested and crossing (pseudoknot) pairs; pairs are unordered and
    symmetric (both ends carry BP_PAIR).

    Returns ``(states, pairs)``: ``states`` is an int8 (L,) array of
    BasePairEnum values; ``pairs`` a list of 0-based ``(i, j)`` tuples with
    ``i < j`` (sorted). Raises ValueError on unmatched brackets and
    unrecognized characters — malformed conditioning is never dropped silently.
    """
    dot_bracket = str(dot_bracket)
    # native _opener_for: fixed-shape families + letter-case auto-resolution
    symbol_pairs = {"(": ")", "[": "]", "{": "}", "<": ">"}
    opener_for = {close: open_ for open_, close in symbol_pairs.items()}
    for lower in {c.lower() for c in dot_bracket if c.isalpha()}:
        upper = lower.upper()
        i_lower, i_upper = dot_bracket.find(lower), dot_bracket.find(upper)
        # a missing case leaves an unbalanced string; either assignment surfaces it
        if i_upper >= 0 and (i_lower < 0 or i_upper < i_lower):
            opener_for[lower] = upper
        else:
            opener_for[upper] = lower
    openers = set(opener_for.values())

    unknown = {"x", "_", "X"}
    states = np.full(len(dot_bracket), BP_UNSPECIFIED, np.int8)
    stack: dict[str, list[int]] = {}
    pairs: list[tuple[int, int]] = []
    for i, ch in enumerate(dot_bracket):
        if ch == ".":
            states[i] = BP_LOOP
        elif ch in unknown:
            states[i] = BP_UNSPECIFIED
        elif ch in openers:
            stack.setdefault(ch, []).append(i)
        elif ch in opener_for:
            open_ = opener_for[ch]
            if not stack.get(open_):
                raise ValueError(
                    f"unmatched closing bracket {ch!r} at position {i} in dot-bracket string")
            open_pos = stack[open_].pop()
            pairs.append((open_pos, i))
            states[open_pos] = BP_PAIR
            states[i] = BP_PAIR
        else:
            raise ValueError(f"unrecognized dot-bracket character {ch!r} at position {i} "
                             "(supported: . ( ) [ ] { } < > letters, x/_/X unknown)")
    dangling = {open_: idxs for open_, idxs in stack.items() if idxs}
    if dangling:
        detail = ", ".join(f"{open_!r} at {idxs}" for open_, idxs in sorted(dangling.items()))
        raise ValueError(f"unmatched opening bracket(s) in dot-bracket string: {detail}")
    return states, sorted(pairs)


def _atomic_number_from_symbol(symbol: str) -> int:
    """Element symbol -> atomic number (gemmi; cached; X/unknown -> 0)."""
    sym = symbol.strip().upper()
    if sym in ("X", ""):
        return 0
    if sym not in _ELEMENT_Z_CACHE:
        import gemmi

        _ELEMENT_Z_CACHE[sym] = int(gemmi.Element(sym).atomic_number)
    return _ELEMENT_Z_CACHE[sym]


@dataclasses.dataclass
class ChainSpec:
    """One polymer chain: `num_residues` residues plus per-chain conditions.

    Conditions are either a scalar (broadcast to every realized residue — the
    sentinel-CIF route) or a per-residue ``np.ndarray`` of length ``num_residues``
    (direct-featurization route). Atom-level conditioning materializes atom37
    slots: `seq_array` (int32 per-residue RFD4 encoding indices, 0 = <M>),
    `slot_mask` (bool num_residues × 37), and `pos` (float32 num_residues × 37 × 3,
    NaN where a slot carries no position).
    """

    num_residues: int
    name: str = "UNK"                 # residue CCD code for realized (de-novo) residues
    kind: str = "aa"                  # polymer kind: "aa" | "rna" | "dna" (token_type)
    indexed: bool = True              # condition_index_1_residue (token has a sequential index)
    atomize: bool = False             # emit one ATOMIZED token per residue (token_type=3)
    # dot-bracket NA secondary structure (C_NA_SS_STATE / C_BP); RNA/DNA chains
    # only (an all-"x" dummy string is materialized when absent)
    bp: str | None = None
    # condition full_name -> scalar per-chain value or per-residue array
    conditions: dict = dataclasses.field(default_factory=dict)
    # materialized atom37 conditioning (atom-level adders / add_pos / add_sequence)
    seq_array: np.ndarray | None = None    # int32 (num_residues,) encoding indices
    slot_mask: np.ndarray | None = None    # bool (num_residues, 37)
    pos: np.ndarray | None = None          # float32 (num_residues, 37 × 3, NaN = absent
    # coordinate group (C_CRDGRP) for this chain's conditioned slots (0 = absolute)
    coord_group: int = 0


@dataclasses.dataclass
class LigandSpec:
    """One ligand: atomized tokens (one per atom), appended after the chains.

    `elements` (int32 num_atoms,) atomic numbers; `charges` (int32 num_atoms,)
    formal charges; `bonds` (int32 B × 4: i, j, biotite bond-type id, aromatic —
    the featurize_atoms._build_bonds convention); `pos` None (topology-only:
    ligand positions are diffused design targets) or float32 (num_atoms, 3) Å
    (slot-1 coordinate conditioning, C_CRD group 0); `atom_names` optional S8
    array for PDB output; `res_name` the residue code written to PDB files.
    """

    label: str
    elements: np.ndarray
    charges: np.ndarray
    bonds: np.ndarray
    pos: np.ndarray | None = None
    atom_names: np.ndarray | None = None
    res_name: str = "LIG"
    # autofeaturized interaction roles (from_chains "auto_condition": True):
    # {"hbond": int8 (k,) 0/1/2/3, "pistack": bool (k,), "salt": int8 (k,)
    # 0/1/2} per ligand atom — written at slot 1 of the ligand's atomized
    # tokens at compile time (native AnnotateInteractions ligand semantics)
    interactions: dict | None = None
    # from_chains "buried": True → C_RAS 1 (buried) at slot 1 of every token
    buried: bool = False


@dataclasses.dataclass
class UnindexedMotifSpec:
    """One unindexed motif: in-context conditioning tokens (C_IDX False).

    `pos` (float32 num_motif_residues × 37 × 3) are the conditioning positions;
    `restype` (int32 num_motif_residues,) the AE residue-vocabulary indices (what
    C_SEQ carries); `mask` (bool num_motif_residues × 37) marks the specified
    atoms. `residue_index` None → each motif residue gets its own index group;
    an int array → all residues share one fresh index group with linkage and
    residue sequence distance from the relative residue_index distances.
    """

    pos: np.ndarray
    restype: np.ndarray
    mask: np.ndarray
    residue_index: np.ndarray | None = None


class RFD4Spec:
    """Programmatic RFD4 design specification (purely Python; gemmi planned for CIF I/O).

    Empty constructor; add chains, then conditions; compile with `to_batch`.
    """

    def __init__(self) -> None:
        self.chains: list[ChainSpec] = []
        self.motifs_unindexed: list[UnindexedMotifSpec] = []
        self.ligands: list[LigandSpec] = []

    # ------------------------------------------------------------------ chains
    def add_chain(
        self,
        num_residues: int,
        *,
        kind: str = "aa",
        sequence: str | None = None,
        indexed: bool = True,
        name: str = "UNK",
        atomize: bool = False,
    ) -> int:
        """Add a separate de-novo polymer chain with `num_residues` residues.

        `kind` selects the polymer vocabulary group: "aa" (protein,
        token_type=0), "rna" (token_type=1) or "dna" (token_type=2). RNA/DNA
        design chains may carry dot-bracket NA secondary-structure conditioning
        via :meth:`add_bp`.

        The chain is authored as an expandable segment (the RFD4 sentinel pattern):
        unconditioned sequence (<M>), every realized residue inheriting the chain's
        uniform condition values. `condition_index` (C_IDX) is True — the token HAS
        a sequential index; generation is governed by the masked sequence, and
        `indexed=False` selects the unindexed motif/insertion path.

        `atomize=True` emits the chain's residues as ATOMIZED tokens (one token per
        residue atom, token_type=3, C_SEQ=<A>, atom37 slot 1) — the explicit
        programmatic counterpart of the featurizer's `force_atomize` override.
        Note that atomized tokens require coordinates for actual generation (the
        pipeline rejects atomized tokens without a slot-1 reference atom).

        Fixed-sequence de-novo chains (`sequence=`) are NOT expressible in the
        single-row CIF route: real residue tokens must carry their atom37 slot-1
        reference atom (RaiseOnCondition), and expseg stubs replicate the row's
        single C_SEQ value. They require the condition-spec/contig route or the
        direct featurizer (planning.md Part 5) — see the class docstring.
        Returns the chain index (use as `chain=` in condition methods).
        """
        assert num_residues >= 1
        if kind not in CHAIN_KIND_TOKEN_TYPE:
            raise ValueError(
                f"add_chain: kind must be one of {sorted(CHAIN_KIND_TOKEN_TYPE)}, got {kind!r}")
        if sequence is not None:
            raise NotImplementedError(
                "Fixed-sequence de-novo chains need the condition-spec/contig route or "
                "the direct featurizer (planning.md Part 5): real residue tokens require "
                "their slot-1 reference atom, and expandable-segment stubs replicate the "
                "single authored C_SEQ value per residue.")
        self.chains.append(
            ChainSpec(num_residues=num_residues, kind=kind, name=name,
                      indexed=indexed, atomize=atomize)
        )
        return len(self.chains) - 1

    def add_bp(self, bp: str, *, chain: int) -> None:
        """Attach dot-bracket NA secondary-structure conditioning (C_NA_SS_STATE
        / C_BP) to a nucleic-acid chain. Only "rna"/"dna" chains accept `bp`
        (protein chains raise); the string length must equal the chain's
        residue count. At compile time the bp strings of all nucleic-acid
        chains are concatenated in add order (a chain without `bp` contributes
        an all-"x" dummy string of its length) and parsed together, so pairs
        may cross chain boundaries.
        """
        ch = self.chains[self._resolve_chain(chain)]
        if ch.kind not in ("rna", "dna"):
            raise ValueError(
                f"add_bp: chain {chain} has kind {ch.kind!r} — dot-bracket NA "
                "secondary-structure conditioning requires an rna/dna chain")
        if len(bp) != ch.num_residues:
            raise ValueError(
                f"add_bp: bp string length {len(bp)} != chain {chain} residue count "
                f"{ch.num_residues}")
        ch.bp = str(bp)

    # -------------------------------------------------------------- conditions
    def _split_target(self, values, chain: int | None, what: str) -> list[tuple[int, np.ndarray]]:
        """Resolve `chain=None` concatenation semantics: an array/string covers the
        whole system (split across chains in add order); a scalar broadcasts."""
        arr = np.asarray(values)
        if arr.ndim == 0:
            return [(ci, arr.item()) for ci in range(len(self.chains))]
        sizes = [c.num_residues for c in self.chains]
        if arr.shape[0] != sum(sizes):
            raise ValueError(
                f"{what}: with chain=None the values must cover the whole system "
                f"({sum(sizes)} residues in add order), got {arr.shape[0]}. "
                "Pass chain=<index> to target a single chain.")
        out, off = [], 0
        for ci, size in enumerate(sizes):
            out.append((ci, arr[off:off + size]))
            off += size
        return out

    def add_condition(self, name: str, values, *, chain: int | None = None) -> None:
        """Add a condition by RFD4 condition name or alias (e.g. "C_SEC", "hbond").

        `values` is a scalar (broadcast to every realized residue of the chain —
        the sentinel-CIF route) or a 1-D array of per-residue values (the direct-
        featurization route). With `chain=None`, a scalar applies to every chain
        and an array must cover all chains concatenated in add order. The
        condition's full name, level and value domain are validated against
        rfproteina's condition registry (enum ranges checked where defined).
        """
        full_name = _resolve_condition(name)
        if full_name in _ATOM_LEVEL_DIRECT_CONDITIONS:
            raise ValueError(
                f"{full_name} is atom-level: use its typed adder (add_rasa/add_hbond/"
                "add_hotspot/...) with `aa` (per-residue sequence indices) and `where` "
                "(num_residues × 37 atom37 slot mask) so the condition has atoms to "
                "attach to — unrealized de-novo residues carry no atom37 slots.")
        targets = (self._split_target(values, chain, full_name) if chain is None
                   else [(self._resolve_chain(chain), values)])
        for ci, vals in targets:
            ch = self.chains[ci]
            ch.conditions[full_name] = _validate_values(full_name, vals, ch.num_residues)

    def add_dssp(self, chain: int | None = None, dssp: None | str | np.ndarray = None) -> None:
        """Secondary-structure conditioning (C_SEC) from a DSSP-style string.

        `dssp=None` clears the SS condition on the targeted chain(s) (native
        `set_annotation` replace semantics). A string is converted per character —
        strictly `H` → 1 (helix), `E` → 2 (strand), `L` → 3 (loop), `X`/`_` → 0
        (unspecified); any other character raises. An `np.ndarray` must be int32
        (or safely castable) already containing RFD4's SSEnum encoding
        (0=none/unspecified, 1=helix, 2=sheet, 3=loop, 4=non-protein).

        `chain=None` conditions the entire system: a string/array covers all
        chains concatenated in add order; a scalar broadcast is not meaningful for
        DSSP strings and only `dssp=None` (clear) or a length-1 value are accepted.
        """
        if dssp is None:
            targets = range(len(self.chains)) if chain is None else [self._resolve_chain(chain)]
            for ci in targets:
                self.chains[ci].conditions.pop("condition_secondary-structure_1_residue", None)
            return
        if isinstance(dssp, str):
            letter_map = {"H": 1, "E": 2, "L": 3, "X": 0, "_": 0}
            bad = sorted(set(dssp) - set(letter_map))
            if bad:
                raise ValueError(
                    f"add_dssp: unsupported DSSP character(s) {bad!r} — this parser is "
                    "strictly 3-state: H (helix), E (strand), L (loop), X/_ (unspecified).")
            dssp = np.array([letter_map[ch] for ch in dssp], dtype=np.int32)
        dssp = np.asarray(dssp)
        if dssp.dtype != np.int32:
            dssp = dssp.astype(np.int32)
        if dssp.ndim == 1 and dssp.shape[0] == 1:
            dssp = dssp[0]  # single value → uniform (scalar broadcast)
        self.add_condition("C_SEC", dssp, chain=chain)

    def add_index_group(self, values, *, chain: int | None = None) -> None:
        """Index groups (C_IDXGRP): scalar or per-residue int array. Tokens sharing
        a group have known relative positions; across groups they do not."""
        self.add_condition("C_IDXGRP", values, chain=chain)

    def _overlay_idx_group(self, ci: int, rows: np.ndarray, group) -> None:
        """Set C_IDXGRP on the selected rows of one chain, preserving other rows."""
        ch = self.chains[ci]
        n = ch.num_residues
        existing = ch.conditions.get("condition_index-group_1_token")
        if existing is None:
            base = np.zeros(n, dtype=np.int32)
        elif np.asarray(existing).ndim == 0:
            base = np.full(n, int(existing), dtype=np.int32)
        else:
            base = np.asarray(existing, dtype=np.int32).copy()
        group_arr = np.asarray(group)
        base[rows] = (int(group_arr) if group_arr.ndim == 0
                      else group_arr.astype(np.int32)[rows])
        ch.conditions["condition_index-group_1_token"] = base

    def add_unindexed_motif(self, pos, restype, mask,
                            residue_index: np.ndarray | None = None) -> None:
        """Add an unindexed motif: in-context conditioning tokens (C_IDX False).

        The motif's residues are NOT part of the generated polymer — at featurization
        they become in-context conditioning tokens appended after the de-novo chains
        (is_in_context=True, supports_local_latent=False), exactly what the reference
        pipeline's UnindexFlaggedTokens produces at inference. The model never emits
        them; it recreates motif-equivalent residues elsewhere, which are matched to
        the conditioning afterwards.

        Args:
          pos: float32 (num_motif_residues, 37, 3) — the conditioning positions (Å).
          restype: int32 (num_motif_residues,) — RFD4 autoencoder residue-vocabulary
            indices (what C_SEQ carries; e.g. ALA=1, GLY=8, NCAAs per the encoding).
          mask: bool (num_motif_residues, 37) — the specified atoms; ONLY these
            atom37 slots are materialized on the motif tokens.
          residue_index: None or int array (num_motif_residues,). If None, every
            motif residue is placed in its OWN index group (no linkage, no known
            relative positions). If given, all motif residues share ONE fresh index
            group (never used elsewhere in the spec); canonical-polymer linkage is
            set between residues whose residue_index differs by exactly 1, and the
            residue sequence distance (token_bond_path_length) between motif
            residues is their |residue_index| difference — the "C–[2–2]–C"
            relative-index pattern of the reference's unindexed motifs
            (docs/concepts/conditioning.md).
        """
        pos = np.asarray(pos, dtype=np.float32)
        restype = np.asarray(restype, dtype=np.int32)
        mask = np.asarray(mask, dtype=bool)
        n = restype.shape[0]
        if pos.shape != (n, 37, 3) or mask.shape != (n, 37):
            raise ValueError(
                f"add_unindexed_motif: pos must be (num_motif_residues, 37, 3) and "
                f"mask (num_motif_residues, 37) with num_motif_residues={n}, got "
                f"{pos.shape} and {mask.shape}")
        if restype.ndim != 1:
            raise ValueError(f"add_unindexed_motif: restype must be 1-D, got {restype.shape}")
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC
        if len(ENC.token_to_idx) == 0 or restype.min() < 0 or restype.max() >= 147:
            raise ValueError(
                "add_unindexed_motif: restype must contain RFD4 autoencoder "
                "residue-vocabulary indices in [0, 147).")
        if residue_index is not None:
            residue_index = np.asarray(residue_index, dtype=np.int32)
            if residue_index.shape != (n,):
                raise ValueError(
                    f"add_unindexed_motif: residue_index must have length {n}")
        self.motifs_unindexed.append(
            UnindexedMotifSpec(pos=pos, restype=restype, mask=mask,
                               residue_index=residue_index))

    def add_indexed_motif(self, pos, restype, where) -> None:
        """Add an indexed motif: fixed residues INSIDE the generated polymer.

        An overlay on the existing system: residues with at least one conditioned
        atom (`where.any(axis=1)`) become indexed, conditioned motif residues —
        generated positions held at their conditioned geometry (C_CRD) and identity
        (C_SEQ = restype), remaining polymer members (canonical ordering, bond path,
        supports_local_latent all preserved). `restype` only takes effect at
        conditioned rows.

        All conditioned residues share ONE index group — the underlying spec's group
        (chain default 0) — and linkage follows the underlying residue indices of
        the spec (the de-novo chains' sequential numbering): motif residues keep
        their chain's canonical ordering and bond-path distances to their neighbors.

        Args:
          pos: float32 (num_total_residues, 37, 3) — positions over the WHOLE system
            (all chains concatenated in add order; NaN where a masked slot has no
            position).
          restype: int32 (num_total_residues,) — AE residue-vocabulary indices; only
            read at rows where `where.any(axis=1)`.
          where: bool (num_total_residues, 37) — conditioning only for atoms where
            True.
        """
        pos = np.asarray(pos, dtype=np.float32)
        restype = np.asarray(restype, dtype=np.int32)
        where = np.asarray(where, dtype=bool)
        num_total = sum(c.num_residues for c in self.chains)
        if not self.chains:
            raise ValueError("add_indexed_motif: add at least one chain first")
        if pos.shape != (num_total, 37, 3) or restype.shape != (num_total,) \
                or where.shape != (num_total, 37):
            raise ValueError(
                f"add_indexed_motif: the system has {num_total} residues — pos must be "
                f"(num_total, 37, 3), restype (num_total,), where (num_total, 37); got "
                f"{pos.shape}, {restype.shape}, {where.shape}")
        rows = where.any(axis=1)
        off = 0
        for ci, ch in enumerate(self.chains):
            n = ch.num_residues
            ch_rows = rows[off:off + n]
            ch_where = where[off:off + n]
            ch_pos = pos[off:off + n]
            ch_restype = restype[off:off + n]
            # sequence identity at conditioned rows (atom37 reveals identity)
            if ch.seq_array is None:
                ch.seq_array = np.zeros(n, dtype=np.int32)
            ch.seq_array = np.where(ch_rows, ch_restype, ch.seq_array)
            # materialized slots + positions (merge with any existing state)
            ch.slot_mask = (ch_where.copy() if ch.slot_mask is None
                            else (ch.slot_mask | ch_where))
            if ch.pos is None:
                ch.pos = np.full((n, 37, 3), np.nan, np.float32)
            fresh = ch_where & np.isfinite(ch_pos).all(axis=-1)
            ch.pos[fresh] = ch_pos[fresh]
            # all conditioned residues share the underlying spec's index group:
            # normalize the chain's C_IDXGRP to a per-residue array carrying the
            # chain's group value (default 0) so the motif rows are explicit
            if ch_rows.any():
                self._overlay_idx_group(
                    ci, ch_rows,
                    ch.conditions.get("condition_index-group_1_token", 0))
            off += n

    def add_sequence(self, values, *, chain: int | None = None) -> None:
        """Per-residue sequence conditioning (C_SEQ) on de-novo chains.

        `values` is a one-letter polymer string (one character per residue,
        e.g. "AGLST"; DNA/RNA chains use their own 1-letter codes — A/C/G/T
        and A/C/G/U) or an int32 array of RFD4 encoding indices. This fixes the
        identity of the targeted residues inside the designed chain — a capability
        the sentinel-CIF route cannot express (its expseg row carries one C_SEQ for
        the whole segment), so it compiles through the direct featurizer only and
        is self-consistency-tested (no native parity route).

        NOTE: this is a custom extension, not a native RFD4 input pattern.
        """
        targets = (self._split_target(values, chain, "C_SEQ") if chain is None
                   else [(self._resolve_chain(chain), values)])
        for ci, vals in targets:
            ch = self.chains[ci]
            if isinstance(vals, str):
                vals = sequence_to_indices(vals, ch.kind)
            vals = np.asarray(vals, dtype=np.int32)
            if vals.shape != (ch.num_residues,):
                raise ValueError(
                    f"add_sequence: values must have length {ch.num_residues}, got {vals.shape}")
            ch.seq_array = vals

    def add_pos(self, pos, where, *, chain: int | None = None,
                coord_group: int = 0) -> None:
        """Materialize atom37 slots with position information on de-novo chains.

        `pos`: float32 (num_residues, 37, 3) — per-slot atom positions in Ångström
        (NaN where a masked slot carries no position). `where`: bool
        (num_residues, 37) — True where the slot is materialized. With
        `chain=None` the arrays cover the whole system (concatenated in add
        order). Materialized slots enter `atom37_mask` and — where `pos` is
        finite — the ground-truth coordinates and coordinate conditioning
        (C_CRD, group `coord_group` — default 0 = absolute conditioning —
        centered per the pipeline's
        CenterOnCoordinateConditionedAtoms semantics).

        NOTE: coordinate-free or position-free materialized slots are outside the
        reference pipeline's validated envelope (its inference checks reject
        residue tokens without slot-1 reference atoms); this is a direct-
        featurizer-only extension.
        """
        pos = np.asarray(pos, dtype=np.float32)
        where = np.asarray(where, dtype=bool)
        if pos.shape[-2:] != (37, 3) or where.shape[-1:] != (37,):
            raise ValueError(
                f"add_pos: pos must be (N, 37, 3) and where (N, 37), got "
                f"{pos.shape} and {where.shape}")
        targets = (self._split_target(np.zeros(where.shape[0]), chain, "add_pos")
                   if chain is None else [(self._resolve_chain(chain), None)])
        off = 0
        for ci, _ in targets:
            ch = self.chains[ci]
            n = ch.num_residues
            ch_pos = pos[off:off + n]
            ch_where = where[off:off + n]
            off += n
            if ch_where.shape != (n, 37) or ch_pos.shape != (n, 37, 3):
                raise ValueError(
                    f"add_pos: chain {ci} has {n} residues; got where {ch_where.shape}, "
                    f"pos {ch_pos.shape}")
            # merge across calls: slots accumulate, positions/values override
            ch.slot_mask = ch_where.copy() if ch.slot_mask is None else (ch.slot_mask | ch_where)
            if ch.pos is None:
                ch.pos = np.full((n, 37, 3), np.nan, np.float32)
            fresh = np.isfinite(ch_pos)
            ch.pos[fresh] = ch_pos[fresh]
            ch.coord_group = int(coord_group)

    def add_ligand(self, *, smiles: str | None = None, ccd: str | None = None,
                   ligand_pos=None, res_name: str | None = None) -> int:
        """Add a ligand: atomized tokens (one per atom) appended after the chains.

        Topology-only by default — elements (atomic numbers), formal charges and
        the bond graph (bond order + aromatic flags) condition the model, ligand
        positions are diffused design targets. `ligand_pos` (num_atoms, 3, Å)
        additionally coordinate-conditions every atom (C_CRD, group 0, ground-
        truth coordinates; the pipeline centering includes them).

        `smiles` is decoded with RDKit (heavy atoms only, implicit H excluded;
        bond orders Kekulized for the aromatic-flag convention; atom order =
        RDKit atom order). `ccd` looks up the cached chemical component
        dictionary (gemmi-extracted; atom order = CCD file order). Pass exactly
        one of the two. `res_name` overrides the residue code written to PDB
        files (default: "LIG" for smiles, the CCD id). Returns the ligand index.
        """
        if (smiles is None) == (ccd is None):
            raise ValueError("add_ligand: pass exactly one of smiles= / ccd=")
        from .featurize_atoms import BT_AROMATIC, BT_AROMATIC_DOUBLE, BT_AROMATIC_SINGLE

        def _bt(order: int, arom: bool) -> int:
            if arom:
                return {1: BT_AROMATIC_SINGLE, 2: BT_AROMATIC_DOUBLE, 3: 7}.get(
                    int(order), BT_AROMATIC)
            return int(order) if int(order) in (1, 2, 3) else 1

        if ccd is not None:
            from .ccd import get_ccd_entry

            rec = get_ccd_entry(str(ccd))
            elements = np.array(
                [_atomic_number_from_symbol(e.decode()) for e in rec.elements], np.int32)
            charges = rec.charges.astype(np.int32)
            bonds = np.stack([
                rec.bond_i, rec.bond_j,
                np.array([_bt(o, a) for o, a in zip(rec.bond_order, rec.bond_aromatic)],
                         np.int32),
                rec.bond_aromatic.astype(np.int32)], axis=1).astype(np.int32)
            atom_names, rname = rec.atom_names, str(rec.ccd_id)
            label = f"ccd:{rname}"
            n_atoms = len(elements)
        else:
            from rdkit import Chem

            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                raise ValueError(f"add_ligand: RDKit failed to parse SMILES {smiles!r}")
            # Kekulize a copy for bond orders (aromatic flags stay True on the
            # original mol, matching the CCD convention of order + aromatic flag)
            mol_kek = Chem.Mol(mol)
            Chem.Kekulize(mol_kek, clearAromaticFlags=True)
            elements = np.array([a.GetAtomicNum() for a in mol.GetAtoms()], np.int32)
            charges = np.array([a.GetFormalCharge() for a in mol.GetAtoms()], np.int32)
            bi, bj, bts, bar = [], [], [], []
            for b in mol.GetBonds():
                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                arom = bool(b.GetIsAromatic())
                order = int(mol_kek.GetBondWithIdx(b.GetIdx()).GetBondTypeAsDouble()) \
                    if arom else int(b.GetBondTypeAsDouble())
                bi.append(i)
                bj.append(j)
                bts.append(_bt(order, arom))
                bar.append(int(arom))
            bonds = np.stack([bi, bj, bts, bar], axis=1).astype(np.int32)
            atom_names, rname = None, "LIG"
            label = f"smiles:{smiles}"
            n_atoms = len(elements)
        pos = None
        if ligand_pos is not None:
            pos = np.asarray(ligand_pos, dtype=np.float32)
            if pos.shape != (n_atoms, 3) or not np.isfinite(pos).all():
                raise ValueError(
                    f"add_ligand: ligand_pos must be a finite ({n_atoms}, 3) array, "
                    f"got {pos.shape}")
        self.ligands.append(LigandSpec(
            label=label, elements=elements, charges=charges, bonds=bonds, pos=pos,
            atom_names=atom_names,
            res_name=res_name.upper()[:3] if res_name else rname))
        return len(self.ligands) - 1

    def _add_atom_level(self, full_name: str, values, aa, where, chain: int | None) -> None:
        """Shared implementation of the atom-level adders (rasa, hbond, hotspot,
        pistack, pication, saltbridge, hydrophobic, metal_coord).

        `aa` (int32, num_residues) gives the per-residue sequence for conditioned
        positions (0 = <M>); `where` (bool, num_residues × 37) selects the atom37
        slots to materialize — the condition value lands on every materialized
        slot of the residue. `chain=None` shapes over the whole system.
        """
        if aa is None or where is None:
            raise ValueError(
                f"{full_name} on de-novo chains needs `aa` (per-residue sequence "
                "indices) and `where` (num_residues × 37 atom37 slot mask): unrealized "
                "residues carry no atom37 slots, so the condition must materialize "
                "the slots it conditions.")
        aa = np.asarray(aa, dtype=np.int32)
        where = np.asarray(where, dtype=bool)
        if aa.ndim != 1 or where.ndim != 2 or where.shape != (aa.shape[0], 37):
            raise ValueError(
                f"{full_name}: aa must be (num_residues,) int32 and where "
                f"(num_residues, 37) bool, got {aa.shape} and {where.shape}")
        targets = (self._split_target(np.zeros(len(aa)), chain, full_name)
                   if chain is None else [(self._resolve_chain(chain), None)])
        off = 0
        for ci, _ in targets:
            ch = self.chains[ci]
            n = ch.num_residues
            ch_aa, ch_where = aa[off:off + n], where[off:off + n]
            ch_vals = (np.asarray(values) if np.asarray(values).ndim == 0
                       else np.asarray(values)[off:off + n])
            off += n
            # merge materialization across calls; sequence values override where
            # this call specifies a conditioned identity (aa == 0 = <M> = "not
            # specified by this call", so earlier add_sequence values survive)
            ch.slot_mask = (ch_where.copy() if ch.slot_mask is None
                            else (ch.slot_mask | ch_where))
            if ch.seq_array is None:
                ch.seq_array = np.zeros(n, dtype=np.int32)
            ch.seq_array = np.where(ch_aa != 0, ch_aa, ch.seq_array)
            ch.conditions[full_name] = _validate_values(full_name, ch_vals, n)

    def add_rasa(self, values, *, aa, where, chain: int | None = None) -> None:
        """Binned RASA (C_RAS): 0=unspecified, 1=buried, 2=intermediate, 3=exposed.
        Atom-level: conditions the materialized slots (see _add_atom_level)."""
        self._add_atom_level("condition_rasa-binned_1_atom", values, aa, where, chain)

    def add_hbond(self, values, *, aa, where, chain: int | None = None) -> None:
        """Hydrogen-bond roles (C_HB): 0=none, 1=donor, 2=acceptor, 3=both.
        Atom-level: conditions the materialized slots (see _add_atom_level)."""
        self._add_atom_level("condition_hbond_1_atom", values, aa, where, chain)

    def add_hotspot(self, values, *, aa, where, chain: int | None = None) -> None:
        """Hotspot residues (C_HOT, boolean). Atom-level: conditions the
        materialized slots (see _add_atom_level)."""
        self._add_atom_level("condition_hotspot_1_atom", values, aa, where, chain)

    def add_pistack(self, values, *, aa, where, chain: int | None = None) -> None:
        """Pi-stacking (C_PISTACK, boolean). Atom-level: conditions the
        materialized slots (see _add_atom_level)."""
        self._add_atom_level("condition_pistack_1_atom", values, aa, where, chain)

    def add_pication(self, values, *, aa, where, chain: int | None = None) -> None:
        """Pi-cation (C_PICATION). Atom-level: conditions the materialized slots
        (see _add_atom_level)."""
        self._add_atom_level("condition_pication_1_atom", values, aa, where, chain)

    def add_saltbridge(self, values, *, aa, where, chain: int | None = None) -> None:
        """Salt bridges (C_SALT). Atom-level: conditions the materialized slots
        (see _add_atom_level)."""
        self._add_atom_level("condition_saltbridge_1_atom", values, aa, where, chain)

    def add_hydrophobic(self, values, *, aa, where, chain: int | None = None) -> None:
        """Hydrophobic contacts (C_HPHOB, boolean). Atom-level: conditions the
        materialized slots (see _add_atom_level)."""
        self._add_atom_level("condition_hydrophobic_1_atom", values, aa, where, chain)

    def add_metal_coord(self, values, *, aa, where, chain: int | None = None) -> None:
        """Metal coordination (C_MCOORD). Atom-level: conditions the materialized
        slots (see _add_atom_level)."""
        self._add_atom_level("condition_metal-coord_1_atom", values, aa, where, chain)

    # ----------------------------------------------------------------- compile
    def to_annotated_cif(self) -> str:
        """Render the spec as annotated-CIF text (purely Python; no structure I/O deps).

        A single `atom_site` loop with one HETATM/UNK row per chain carrying the
        expandable segment and that chain's condition columns (verified to
        propagate to all realized residues of the chain). Condition columns absent
        for a chain are emitted with their registry default.

        Raises for per-residue condition arrays, per-residue sequence
        conditioning, and materialized atom37 slots/positions: the sentinel-CIF
        route carries ONE row (one scalar value) per chain — such specs compile
        through `to_batch`'s direct featurizer instead.
        """
        for ci, chain in enumerate(self.chains):
            if chain.seq_array is not None or chain.slot_mask is not None or chain.pos is not None:
                raise NotImplementedError(
                    f"chain {ci}: per-residue sequence conditioning / materialized "
                    "atom37 slots / positions are direct-featurizer-only "
                    "(to_batch), not expressible in the sentinel-CIF route.")
            for full_name, value in chain.conditions.items():
                if np.asarray(value).ndim > 0:
                    raise NotImplementedError(
                        f"chain {ci}: per-residue values for {full_name!r} are "
                        "direct-featurizer-only (to_batch); the sentinel-CIF route "
                        "carries one scalar per chain.")
        if self.motifs_unindexed:
            raise NotImplementedError(
                "unindexed motifs (add_unindexed_motif) are direct-featurizer-only "
                "(to_batch): the sentinel-CIF route cannot express in-context "
                "conditioning tokens.")
        if self.ligands:
            raise NotImplementedError(
                "ligands (add_ligand) are direct-featurizer-only (to_batch): the "
                "sentinel-CIF route cannot express atomized ligand tokens.")
        from rfproteina.conditions import (
            C_BP, C_CATH, C_CENTROID, C_CRDGRP, C_HB, C_HPHOB, C_HOT, C_IDXGRP,
            C_INTCRD, C_MCOORD, C_NA_SS_STATE, C_PICATION, C_PISTACK, C_RAS, C_SALT, C_SEC,
        )
        registry = {c.full_name: c for c in (
            C_HOT, C_RAS, C_SEC, C_HB, C_PISTACK, C_PICATION, C_SALT, C_HPHOB,
            C_MCOORD, C_IDXGRP, C_CATH, C_CENTROID, C_BP, C_NA_SS_STATE,
        )}
        # union of condition columns across chains (residue/token-level only;
        # atom-level conditions need realized atoms — see Part 5 plan)
        cond_columns: dict[str, str] = {}
        for chain in self.chains:
            for full_name in chain.conditions:
                if not full_name.endswith("_1_atom"):
                    cond_columns[full_name] = getattr(registry.get(full_name), "default_value", 0)
        # every registry condition present on ANY chain joins the column set;
        # chains without it get the registry default (same as an absent annotation)
        for full_name in list(cond_columns):
            for cond_full, cond in registry.items():
                if cond_full not in cond_columns and any(
                    cond_full in ch.conditions for ch in self.chains
                ):
                    cond_columns[cond_full] = cond.default_value

        header_cols = [
            "group_PDB", "type_symbol", "label_atom_id", "label_alt_id", "label_comp_id",
            "label_asym_id", "label_entity_id", "label_seq_id", "pdbx_PDB_ins_code",
            "auth_seq_id", "auth_comp_id", "auth_asym_id", "auth_atom_id",
            "B_iso_or_equiv", "occupancy", "pdbx_formal_charge",
            "Cartn_x", "Cartn_y", "Cartn_z", "pdbx_PDB_model_num", "id",
            "annotation_expsegmin_1_residue", "annotation_expsegmax_1_residue",
            "condition_sequence_1_residue", "condition_index_1_residue",
            "condition_hbond_1_atom", "condition_index-group_1_token",
            "annotation_src-component_1_atom", "atom_array_index",
        ]
        all_cols = header_cols[:-2] + sorted(cond_columns) + header_cols[-2:]
        rows = []
        for ci, chain in enumerate(self.chains):
            n = chain.num_residues
            asym = chr(ord("A") + ci) if ci < 26 else f"Z{ci}"
            condition_seq = "<M>"
            values = {
                "group_PDB": "HETATM", "type_symbol": "X", "label_atom_id": "UNK",
                "label_alt_id": ".", "label_comp_id": f"{n}-{n}", "label_asym_id": asym,
                "label_entity_id": str(ci), "label_seq_id": "1", "pdbx_PDB_ins_code": ".",
                "auth_seq_id": "1", "auth_comp_id": f"{n}-{n}", "auth_asym_id": asym,
                "auth_atom_id": "UNK", "B_iso_or_equiv": "nan", "occupancy": "0.0",
                "pdbx_formal_charge": "?", "Cartn_x": "nan", "Cartn_y": "nan",
                "Cartn_z": "nan", "pdbx_PDB_model_num": "1", "id": str(ci + 1),
                "annotation_expsegmin_1_residue": str(n),
                "annotation_expsegmax_1_residue": str(n),
                "condition_sequence_1_residue": condition_seq,
                "condition_index_1_residue": str(bool(chain.indexed)),
                "condition_hbond_1_atom": "0",
                "condition_index-group_1_token": "0",
                # chain's own value when set, else the column default
                **{k: _format_cif_value(chain.conditions.get(k, v))
                   for k, v in cond_columns.items()},
                "annotation_src-component_1_atom": ".",
                "atom_array_index": str(ci),
            }
            rows.append(" ".join(values[c] for c in all_cols))

        n0 = self.chains[0].num_residues
        col_lines = "\n".join(f"_atom_site.{c}" for c in all_cols)
        # per-chain loops: entity, entity_poly (+seq), chem_comp (+atom) — the chain_type
        # (POLYPEPTIDE_L) is derived from _entity_poly, so every chain must be listed
        entity_rows = []
        entity_poly_rows = []
        entity_poly_seq_rows = []
        chem_comp_rows = []
        chem_comp_atom_rows = []
        contigs = []
        for ci, chain in enumerate(self.chains):
            n = chain.num_residues
            asym = chr(ord("A") + ci) if ci < 26 else f"Z{ci}"
            entity_rows.append(f"{ci} polymer .")
            entity_poly_rows.append(
                f"{ci} polypeptide(L) no yes X X {asym} ?")
            entity_poly_seq_rows.append(f"{ci} n {n}-{n} 1")
            chem_comp_rows.append(
                f"{n}-{n} {n}-{n} NON-POLYMER ? ? ? ?")
            chem_comp_atom_rows.append(
                f"{n}-{n} UNK X 0 N N nan nan nan")
            contigs.append(f"{n}-{n}")
        return f"""data_rfd4spec
#
loop_
{col_lines}
{chr(10).join(rows)}
#
_condition_coordinate_1_atom.idx0   0
_condition_coordinate_1_atom.val0   nan
_condition_coordinate_1_atom.val1   nan
_condition_coordinate_1_atom.val2   nan
#
loop_
_entity.id
_entity.type
_entity.pdbx_description
{chr(10).join(entity_rows)}
#
loop_
_entity_poly.entity_id
_entity_poly.type
_entity_poly.nstd_linkage
_entity_poly.nstd_monomer
_entity_poly.pdbx_seq_one_letter_code
_entity_poly.pdbx_seq_one_letter_code_can
_entity_poly.pdbx_strand_id
_entity_poly.pdbx_target_identifier
{chr(10).join(entity_poly_rows)}
#
loop_
_entity_poly_seq.entity_id
_entity_poly_seq.hetero
_entity_poly_seq.mon_id
_entity_poly_seq.num
{chr(10).join(entity_poly_seq_rows)}
#
loop_
_chem_comp.id
_chem_comp.name
_chem_comp.type
_chem_comp.formula
_chem_comp.mon_nstd_parent_comp_id
_chem_comp.pdbx_synonyms
_chem_comp.formula_weight
{chr(10).join(chem_comp_rows)}
#
_rfd_design.sampled_contig   {' '.join(contigs)}
#
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
_chem_comp_atom.pdbx_aromatic_flag
_chem_comp_atom.pdbx_leaving_atom_flag
_chem_comp_atom.pdbx_model_Cartn_x_ideal
_chem_comp_atom.pdbx_model_Cartn_y_ideal
_chem_comp_atom.pdbx_model_Cartn_z_ideal
{chr(10).join(chem_comp_atom_rows)}
#"""

    def to_batch(self, *, np_seed: int | None = 0):
        """Compile to (FlowBatch, coord_cond_mask) — jax.numpy features for the JAX port.

        De-novo-only specs (expandable-segment chains, no authored atoms) use the
        **direct featurizer** — pure numpy, no biotite/atomworks/torch — verified
        field-exact against the torch pipeline and the native feature archives,
        and extended for per-residue conditions, per-residue sequence
        conditioning, materialized atom37 slots/positions, and atom-level
        conditions on materialized slots. Other specs fall back to the
        parity-verified annotated-CIF path (scalar conditions only).
        """
        if self.chains and all(
            set(c.conditions) <= set(_DIRECT_CONDITIONS) for c in self.chains
        ):
            return self._to_batch_direct(np_seed=np_seed)
        from .annotated_cif import annotated_cif_to_batch

        if not self.chains:
            raise ValueError("empty spec: add at least one chain with add_chain()")
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "spec.cif"
            path.write_text(self.to_annotated_cif())
            return annotated_cif_to_batch(path, np_seed=np_seed)

    # ----------------------------------------------------- direct featurizer
    def _to_batch_direct(self, *, np_seed: int | None = 0):
        """De-novo specs → FlowBatch in pure numpy (no structure-I/O dependencies).

        Mirrors the torch featurization for expandable-segment chains: MASKED stubs
        realize to coordinate-free protein tokens (atom37_mask all-False, zero
        coords), intra-chain bond-path |i-j| (unclipped; cross-chain 0 = disconnec-
        ted), signed linkage adjacency within chains, per-chain conditions, and the
        seeded chain/molecule-iid permutations drawn in the same order as
        ConvertToAtomLevelProteinaFeatures.
        """
        import jax.numpy as jnp

        from .featurize import _DEFAULTS, assemble_flow_batch

        if np_seed is not None:
            np.random.seed(np_seed)  # same draws, same order as the pipeline
        sizes = [c.num_residues for c in self.chains]
        n_chains, n = len(sizes), sum(sizes)
        # unindexed motifs: in-context conditioning tokens appended after the chains
        # (UnindexFlaggedTokens inference semantics); each motif token gets its own
        # pseudo-chain id so no polymer adjacency/path arises between motif tokens
        motif_sizes = [int(m.restype.shape[0]) for m in self.motifs_unindexed]
        n_motif = sum(motif_sizes)
        # ligands: atomized tokens (one per atom), one pseudo-chain per ligand
        lig_sizes = [int(l.elements.shape[0]) for l in self.ligands]
        n_lig = sum(lig_sizes)
        lig_offsets = np.concatenate([[0], np.cumsum(lig_sizes)]).astype(np.int64)
        token_chain = np.repeat(np.arange(n_chains), sizes)
        motif_token_chain = np.concatenate(
            [np.full(k, n_chains + bi, dtype=np.int64)
             for bi, k in enumerate(motif_sizes)]) if n_motif else np.zeros(0, np.int64)
        # every motif token gets its OWN pseudo-chain: no generic polymer adjacency
        # or bond path arises between motif tokens; residue_index-based linkage and
        # sequence distances are written explicitly below
        motif_token_chain = n_chains + np.arange(n_motif, dtype=np.int64)
        # ligand tokens share one pseudo-chain per ligand (a bonded molecule —
        # bond-graph BFS path lengths are written explicitly below)
        lig_token_chain = np.concatenate(
            [np.full(k, n_chains + n_motif + li, dtype=np.int64)
             for li, k in enumerate(lig_sizes)]) if n_lig else np.zeros(0, np.int64)
        token_chain = np.concatenate([token_chain, motif_token_chain, lig_token_chain])
        n_total = n + n_motif + n_lig
        # atomized chains: one ATOMIZED token per residue (token_type=3, C_SEQ=<A>,
        # atom37 slot 1, no local latent) — the explicit programmatic atomization
        # option (cf. featurize_atoms' force_atomize / AtomizeByCCDName). Ligand
        # tokens are atomized by construction.
        atomize_tok = np.repeat(
            np.array([bool(c.atomize) for c in self.chains]), sizes)
        atomize_tok = np.concatenate(
            [atomize_tok, np.zeros(n_motif, dtype=bool), np.ones(n_lig, dtype=bool)])
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as _ENC
        _atomize_seq_idx = int(_ENC.token_to_idx["<A>"])
        _atomize_slot_mask = np.zeros((1, n_total, 37), dtype=bool)
        _atomize_slot_mask[0, atomize_tok, 1] = True

        # RealizeExpandableSegments and AddChiralityFeature each consume one
        # global-stream draw (randint for their per-call rngs) before the feature
        # permutations — replicate to stay aligned with the pipeline's stream.
        np.random.randint(0, 2**32)
        np.random.randint(0, 2**32)
        # chain_iid / molecule_iid: random permutations over unique chains/molecules
        # (one molecule per de-novo chain; each ligand is its own chain/molecule),
        # drawn in the pipeline's order
        n_perm = n_chains + len(self.ligands)
        chain_perm = np.random.permutation(n_perm).astype(np.float32)
        molecule_perm = np.random.permutation(n_perm).astype(np.int32)

        # entity ids: contiguous from the molecule grouping, then each masked-sequence
        # chain gets a private entity (is_inference branch), 1-based. Fully atomized
        # (sequence-conditioned) chains and chains with any conditioned sequence
        # position keep their native molecule-group entity. Unindexed motifs get one
        # private entity per motif block.
        has_seq_cond = np.zeros(n, dtype=bool)
        for ci, chain in enumerate(self.chains):
            if chain.seq_array is not None:
                has_seq_cond[token_chain[:n] == ci] = chain.seq_array != 0
        entity = np.empty(n_total, dtype=np.int32)
        next_entity = n_chains  # after np.unique over k molecules: 0..k-1, next = k
        for ci in range(n_chains):
            mask = token_chain[:n] == ci
            if atomize_tok[:n][mask].all() or has_seq_cond[mask].any():
                entity[:n][mask] = ci
            else:
                entity[:n][mask] = next_entity
                next_entity += 1
        for bi, k in enumerate(motif_sizes):
            entity[n + sum(motif_sizes[:bi]):n + sum(motif_sizes[:bi + 1])] = next_entity
            next_entity += 1
        # ligands: one entity per ligand molecule (mirrors atomized chains keeping
        # their chain-index entity)
        lig_start = n + n_motif
        for li, k in enumerate(lig_sizes):
            entity[lig_start + lig_offsets[li]:lig_start + lig_offsets[li + 1]] = \
                n_chains + li

        # materialized atom37 slots: mask from the atom-level adders / add_pos /
        # add_indexed_motif (chains) and add_unindexed_motif (in-context tokens);
        # positions (where finite) become ground-truth coordinates AND coordinate
        # conditioning (C_CRD, per-chain coord_group — default 0 = absolute),
        # centered per the pipeline's CenterOnCoordinateConditionedAtoms semantics
        # (conditioning values on their own centroid) — one global center over all
        # conditioned atoms
        slot_mask = np.zeros((n_total, 37), dtype=bool)
        pos = np.full((n_total, 37, 3), np.nan, np.float32)
        chain_group_of_token = np.zeros(n_total, dtype=np.int32)
        for ci, chain in enumerate(self.chains):
            mask = token_chain[:n] == ci
            if chain.slot_mask is not None:
                slot_mask[mask] = chain.slot_mask
            if chain.pos is not None:
                pos[mask] = chain.pos
            chain_group_of_token[:n][mask] = chain.coord_group
        motif_pos = np.full((n_motif, 37, 3), np.nan, np.float32)
        motif_mask = np.zeros((n_motif, 37), dtype=bool)
        cur = 0
        for m in self.motifs_unindexed:
            k = int(m.restype.shape[0])
            motif_mask[cur:cur + k] = m.mask
            motif_pos[cur:cur + k] = m.pos
            cur += k
        slot_mask[n:n + n_motif] = motif_mask
        pos[n:n + n_motif] = motif_pos
        # ligand atoms: slot-1 positions (coordinate conditioning only when
        # ligand_pos was provided — topology-only ligands diffuse their positions)
        for li, lig in enumerate(self.ligands):
            if lig.pos is not None:
                lo = lig_start + lig_offsets[li]
                pos[lo:lo + lig_sizes[li], 1] = lig.pos
        # the full atom37 mask (incl. atomized tokens' slot-1) gates which
        # materialized positions count as coordinate-conditioned
        full_slot = slot_mask | (atomize_tok[:, None] & (np.arange(37)[None, :] == 1))
        pos_atoms = full_slot & np.isfinite(pos).all(axis=-1)
        if pos_atoms.any():
            center = pos[pos_atoms].mean(axis=0)
            cond_center = pos[pos_atoms].mean(axis=0)
        else:
            center = np.zeros(3, np.float32)
            cond_center = np.zeros(3, np.float32)
        _atomize_slot_mask = _atomize_slot_mask | slot_mask[None]
        coords37 = np.zeros((1, n_total, 37, 3), dtype=np.float32)
        coords37[0, pos_atoms] = pos[pos_atoms] - center
        cond_coords = np.full((1, n_total, 37, 3), np.nan, np.float32)
        cond_coords[0, pos_atoms] = pos[pos_atoms] - cond_center
        coord_groups = np.full((1, n_total, 37), -1, np.int32)
        coord_groups[0, pos_atoms] = np.broadcast_to(
            chain_group_of_token[:, None], (n_total, 37))[pos_atoms]
        # motif coordinate groups: residue_index given -> group 0 (absolute
        # conditioning); None -> fresh groups, one per motif residue, with all
        # tokens of a small molecule (a run of consecutive atomized <A> tokens)
        # sharing one group
        next_cg = int(max(coord_groups.max(), 0)) + 1
        cur = n
        for m in self.motifs_unindexed:
            k = int(m.restype.shape[0])
            if m.residue_index is not None:
                cg_tok = np.zeros(k, np.int32)
            else:
                atomized = (m.restype == _atomize_seq_idx)
                cg_tok = np.empty(k, np.int32)
                i = 0
                while i < k:
                    if atomized[i]:
                        j = i
                        while j < k and atomized[j]:
                            j += 1
                        cg_tok[i:j] = next_cg
                        next_cg += 1
                        i = j
                    else:
                        cg_tok[i] = next_cg
                        next_cg += 1
                        i += 1
            region = pos_atoms[cur:cur + k]
            block = np.broadcast_to(cg_tok[:, None], (k, 37))
            coord_groups[0, cur:cur + k][region] = block[region]
            cur += k
        coord_cond_mask = pos_atoms.any(axis=1)
        is_in_context = np.zeros(n_total, dtype=bool)
        is_in_context[n:n + n_motif] = n_motif > 0
        supports = ~atomize_tok
        supports[n:n + n_motif] = False  # in-context tokens never carry local latents
        # token types: vocabulary groups for chains (computed per chain below) and
        # for motif tokens from their restype's group
        host: dict[str, np.ndarray] = {
            "padding_mask": np.ones((1, n_total), dtype=bool),
            "token_type": np.zeros((1, n_total), dtype=np.int32),  # filled below
            "atom37_mask": _atomize_slot_mask,
            "coords": coords37,  # Å; ground truth only where positions were provided
            "condition_coordinate_1_atom": cond_coords,
            "cond_internal_coords": np.full((1, n_total, 37, 3), np.nan, np.float32),
            "condition_CoordinateGroup_1_atom": coord_groups,
            "condition_ChainCentroid_1_atom": np.full((1, n_total, 3), np.nan, np.float32),
            "coord_cond_mask": coord_cond_mask[None],
            "condition_sequence_1_residue": (
                np.where(atomize_tok, _atomize_seq_idx, 0).astype(np.int32)[None]),
            "condition_index-group_1_token": np.zeros((1, n_total), dtype=np.int32),
            "chain_iid": np.zeros((1, n_total), dtype=np.float32),  # filled below
            "molecule_iid": np.zeros((1, n_total), dtype=np.int32),  # filled below
            "molecule_entity_id": entity[None],
            "is_in_context_conditioning": is_in_context[None],
            "supports_local_latent": supports[None],
            "element": np.full((1, n_total, 37), -1, np.int32),
            "ground_truth_element": np.zeros((1, n_total, 37), np.int32),
            "charge": np.zeros((1, n_total, 37), np.int32),
            "incident_bond_type_counts": np.zeros((1, n_total, 37, 10), np.float32),
            "chirality": np.zeros((1, n_total, 18), np.float32),
            "chiral_frame_atoms": np.full((1, n_total, 6, 8), -1, np.int32),
            "chiral_slot_owner": np.full((1, n_total, 6), -1, np.int32),
            "canonical_polymer_ordering": self._polymer_ordering(
                token_chain, n_total, atomize_tok).astype(np.int32)[None],
            "token_bond_path_length": self._bond_path_lengths(
                token_chain, n_total).astype(np.int32)[None],
            # (chain centroid of a coordinate-free de-novo chain is NaN — all stub coords NaN)
            "bond_mask": np.zeros((1, n_total, n_total), dtype=bool),
            "bond_order": np.zeros((1, n_total, n_total), np.int32),
            "is_aromatic": np.zeros((1, n_total, n_total), dtype=bool),
            "na_ss_state": np.zeros((1, n_total), np.float32),
            "na_ss_state_mask": np.zeros((1, n_total), dtype=bool),
            "na_ss_pair": np.zeros((1, n_total, n_total), np.float32),
            "na_ss_pair_mask": np.zeros((1, n_total, n_total), dtype=bool),
        }
        # chain/molecule iids: pipeline permutations over the chains (de-novo
        # chains + ligands); unindexed motif blocks get fresh ids beyond the
        # permuted range (chain_iid is a float feature and molecule_iid an
        # equality feature — no embedding bounds)
        motif_block = np.concatenate(
            [np.full(k, bi, dtype=np.int64) for bi, k in enumerate(motif_sizes)]) \
            if n_motif else np.zeros(0, np.int64)
        lig_idx_of_token = np.concatenate(
            [np.full(k, li, dtype=np.int64) for li, k in enumerate(lig_sizes)]) \
            if n_lig else np.zeros(0, np.int64)
        chain_iid = np.zeros(n_total, dtype=np.float32)
        chain_iid[:n] = chain_perm[token_chain[:n]]
        chain_iid[n:n + n_motif] = n_perm + motif_block
        chain_iid[n + n_motif:] = chain_perm[n_chains + lig_idx_of_token]
        molecule_iid = np.zeros(n_total, dtype=np.int32)
        molecule_iid[:n] = molecule_perm[token_chain[:n]]
        molecule_iid[n:n + n_motif] = n_perm + motif_block
        molecule_iid[n + n_motif:] = molecule_perm[n_chains + lig_idx_of_token]
        host["chain_iid"] = chain_iid[None]
        host["molecule_iid"] = molecule_iid[None]
        # per-residue sequence conditioning (add_sequence / atom-level `aa` /
        # add_indexed_motif / add_unindexed_motif restype)
        seq_tok = np.asarray(host["condition_sequence_1_residue"])[0]
        for ci, chain in enumerate(self.chains):
            if chain.seq_array is not None:
                seq_tok[:n][token_chain[:n] == ci] = chain.seq_array
        seq_tok[n:n + n_motif] = (np.concatenate([m.restype for m in self.motifs_unindexed])
                                  if n_motif else seq_tok[n:n + n_motif])
        host["condition_sequence_1_residue"] = seq_tok[None]
        # token types: vocabulary groups per chain (add_condition/atomize semantics)
        # and per motif token from restype's group
        from rfproteina.encoding import RESIDUES_BY_TOKEN_TYPE as _GROUPS
        token_type_tok = np.zeros(n_total, dtype=np.int32)
        idx_to_tok = (_ENC.idx_to_token if isinstance(_ENC.idx_to_token, dict)
                      else dict(enumerate(_ENC.idx_to_token)))
        for group, tt_id in (("dna", 2), ("rna", 1), ("ncaa", 6), ("glycan", 5)):
            gnames = {g.upper() for g in _GROUPS[group]}
            for t in range(n_total):
                name = str(idx_to_tok.get(int(seq_tok[t]), "")).upper()
                if name in gnames:
                    token_type_tok[t] = tt_id
        # de-novo chain kind overrides the name-derived default (design stubs
        # carry <M> and would otherwise stay 0); atomized tokens stay 3
        for ci, chain in enumerate(self.chains):
            sel = (token_chain[:n] == ci) & ~atomize_tok[:n]
            token_type_tok[:n][sel] = CHAIN_KIND_TOKEN_TYPE[chain.kind]
        token_type_tok[atomize_tok] = 3
        host["token_type"] = token_type_tok[None]
        # ---- NA secondary structure (C_NA_SS_STATE / C_BP) from the chain bp
        # strings: concatenated over the NUCLEIC-ACID chains in add order (a
        # chain without bp contributes an all-"x" dummy string of its length;
        # non-NA chains are skipped entirely), parsed once, then scattered onto
        # the chain-residue token block [0, n) — chains precede motifs/ligands,
        # and the concatenation offsets skip non-NA chains by construction.
        # Native featurization (ConvertToAtom37Format): na_ss_state in
        # {0,1,2} with mask = state != 0; na_ss_pair[i,j] = na_ss_pair[j,i] = 1
        # with mask True in both directions; every LOOP token's ENTIRE row and
        # column of na_ss_pair_mask set True (explicit "definitely not paired"
        # negatives).
        na_states, na_pairs = [], []
        na_offset = 0
        for chain in self.chains:
            if chain.kind not in ("rna", "dna"):
                continue
            bp = chain.bp if chain.bp is not None else "x" * chain.num_residues
            if len(bp) != chain.num_residues:
                raise ValueError(
                    f"compile: bp string length {len(bp)} != chain residue count "
                    f"{chain.num_residues}")
            states, pairs = parse_dot_bracket(bp)
            na_states.append(states)
            na_pairs.extend((na_offset + i, na_offset + j) for i, j in pairs)
            na_offset += chain.num_residues
        if na_offset:
            states = np.concatenate(na_states)
            na_ss_state = np.zeros(n_total, np.float32)
            na_ss_state_mask = np.zeros(n_total, bool)
            na_ss_pair = np.zeros((n_total, n_total), np.float32)
            na_ss_pair_mask = np.zeros((n_total, n_total), bool)
            na_ss_state[:na_offset] = states
            na_ss_state_mask[:na_offset] = states != BP_UNSPECIFIED
            for i, j in na_pairs:
                na_ss_pair[i, j] = na_ss_pair[j, i] = float(BP_PAIR)
                na_ss_pair_mask[i, j] = na_ss_pair_mask[j, i] = True
            loop_toks = np.flatnonzero(na_ss_state == BP_LOOP)
            na_ss_pair_mask[loop_toks, :] = True
            na_ss_pair_mask[:, loop_toks] = True
            host["na_ss_state"] = na_ss_state[None]
            host["na_ss_state_mask"] = na_ss_state_mask[None]
            host["na_ss_pair"] = na_ss_pair[None]
            host["na_ss_pair_mask"] = na_ss_pair_mask[None]
        # index groups: chains (scalar or per-residue conditions) + motif groups.
        # Motif allocation: residue_index=None → every motif residue its own fresh
        # group; residue_index given → the whole block shares one fresh group
        # (never used elsewhere in the spec).
        ig = np.zeros(n_total, dtype=np.int32)
        for ci, chain in enumerate(self.chains):
            value = chain.conditions.get("condition_index-group_1_token")
            if value is not None:
                arr = np.asarray(value)
                ig[:n][token_chain[:n] == ci] = (int(arr) if arr.ndim == 0
                                             else arr.astype(np.int32))
        next_group = int(ig[:n].max()) + 1 if n else 1
        cur = n
        for m in self.motifs_unindexed:
            k = int(m.restype.shape[0])
            if m.residue_index is None:
                ig[cur:cur + k] = next_group + np.arange(k, dtype=np.int32)
                next_group += k
            else:
                ig[cur:cur + k] = next_group  # one shared fresh group
                next_group += 1
            cur += k
        host["condition_index-group_1_token"] = ig[None]
        # residue_index-based linkage and sequence distances for unindexed motifs:
        # within a block sharing one index group, canonical ordering ±1 connects
        # residues whose residue_index differs by exactly 1 (native convention:
        # +1 at [downstream, upstream], downstream = larger index), and the bond
        # path between motif residues is their |residue_index| difference
        # (the C–[2–2]–C relative-index pattern of the reference docs)
        cur = n
        for m in self.motifs_unindexed:
            k = int(m.restype.shape[0])
            if m.residue_index is not None and k > 1:
                ri = np.asarray(m.residue_index, dtype=np.int64)
                rows = np.arange(cur, cur + k)
                d_ri = ri[:, None] - ri[None, :]
                linked = np.abs(d_ri) == 1
                ordering = np.asarray(host["canonical_polymer_ordering"])[0]
                ordering[np.ix_(rows, rows)] = np.where(
                    linked, np.sign(d_ri), 0).astype(np.int32)
                path = np.asarray(host["token_bond_path_length"])[0]
                path[np.ix_(rows, rows)] = np.abs(d_ri).astype(np.int32)
            cur += k
        # element at materialized slots of sequence-conditioned tokens (the
        # pipeline fills element wherever C_SEQ is conditioned); the 37-slot
        # encoding only carries C/N/O/S/P/SE, so a leading-letter heuristic is
        # exact for its atom names (CA→C, N→N, OD1→O, SE→Se, ...)
        if slot_mask.any():
            from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as _ENC2
            idx_to_tok = (_ENC2.idx_to_token if isinstance(_ENC2.idx_to_token, dict)
                          else dict(enumerate(_ENC2.idx_to_token)))
            element37 = np.asarray(host["element"])[0]
            for t in np.flatnonzero(slot_mask.any(axis=1) & (seq_tok != 0)):
                res_name = idx_to_tok.get(int(seq_tok[t]), "<M>")
                if res_name == "<M>" or res_name not in _ENC2.token_atoms:
                    continue
                for s in np.flatnonzero(slot_mask[t]):
                    an = str(_ENC2.token_atoms[res_name][s]).strip()
                    if not an:
                        continue
                    letters = "".join(c for c in an if c.isalpha()).upper()
                    two = letters[:2]
                    element37[t, s] = (34 if two == "SE" else
                                       {"C": 6, "N": 7, "O": 8, "S": 16, "P": 15}.get(letters[:1], -1))
            host["element"] = element37[None]
        # per-chain conditions (scalar broadcast or per-residue array). C_IDXGRP is
        # already compiled above (scalar/per-residue/motif groups); atom-level
        # conditions scatter onto the materialized slots
        for ci, chain in enumerate(self.chains):
            mask = token_chain[:n] == ci
            # conditions span the FULL compiled width (incl. motif/ligand tokens):
            # a polymer-only mask would build a too-narrow condition array
            mask_full = np.zeros(n_total, dtype=bool)
            mask_full[:n] = mask
            for full_name, value in chain.conditions.items():
                if full_name == "condition_secondary-structure_1_residue":
                    host["condition_secondary-structure_1_residue"] = _broadcast_chain(
                        host.get("condition_secondary-structure_1_residue"), value,
                        mask_full, n_total)
                elif full_name == "condition_index-group_1_token":
                    pass  # compiled above (chains + motif groups)
                elif full_name in _ATOM_LEVEL_DIRECT_CONDITIONS:
                    if not slot_mask.any():
                        raise ValueError(
                            f"{full_name} on de-novo chains requires materialized "
                            "atom37 slots — pass `aa`/`where` to the adder (or call "
                            "add_pos) so the condition has atoms to attach to.")
                    arr = np.asarray(value)
                    vals = (np.full(n, int(arr) if arr.ndim == 0 else 0, np.int32)
                            if arr.ndim == 0 else arr.astype(np.int32))
                    channel = host.setdefault(
                        full_name,
                        np.zeros((1, n_total, 37), dtype=bool
                                 if full_name in ("condition_hotspot_1_atom",
                                                  "condition_pistack_1_atom",
                                                  "condition_hydrophobic_1_atom")
                                 else np.int32))
                    for t in np.flatnonzero(mask):
                        channel[0, t, slot_mask[t]] = vals[t]
                else:
                    raise NotImplementedError(
                        f"{full_name} on de-novo chains: not supported by the direct "
                        "featurizer (planning.md Part 5)")
        # ---- ligand atom features: element/ground-truth-element/formal charge at
        # slot 1 of every atomized ligand token (native real-structure semantics:
        # element wherever C_SEQ is conditioned — atomized tokens carry <A>)
        # plus the autofeaturized interaction conditions (from_chains
        # "auto_condition" / "buried"): C_HB roles, C_PISTACK, C_SALT charge
        # sign and C_RAS buried, all at the atomized tokens' single slot
        for li, lig in enumerate(self.ligands):
            lo = lig_start + lig_offsets[li]
            k = lig_sizes[li]
            element37 = np.asarray(host["element"])[0]
            element37[lo:lo + k, 1] = lig.elements
            host["element"] = element37[None]
            gt37 = np.asarray(host["ground_truth_element"])[0]
            gt37[lo:lo + k, 1] = lig.elements
            host["ground_truth_element"] = gt37[None]
            charge37 = np.asarray(host["charge"])[0]
            charge37[lo:lo + k, 1] = lig.charges
            host["charge"] = charge37[None]
            if lig.interactions is not None:
                hbond = np.asarray(lig.interactions["hbond"], np.int32)
                salt = np.asarray(lig.interactions["salt"], np.int32)
                pistack = np.asarray(lig.interactions["pistack"], bool)
                assert hbond.shape == (k,) and salt.shape == (k,) \
                    and pistack.shape == (k,)
                ch = host.setdefault("condition_hbond_1_atom",
                                     np.zeros((1, n_total, 37), np.int32))
                ch[0, lo:lo + k, 1] = hbond
                host["condition_hbond_1_atom"] = ch
                ch = host.setdefault("condition_saltbridge_1_atom",
                                     np.zeros((1, n_total, 37), np.int32))
                ch[0, lo:lo + k, 1] = salt
                host["condition_saltbridge_1_atom"] = ch
                ch = host.setdefault("condition_pistack_1_atom",
                                     np.zeros((1, n_total, 37), bool))
                ch[0, lo:lo + k, 1] = pistack
                host["condition_pistack_1_atom"] = ch
                pication = lig.interactions.get("pication")
                if pication is not None:
                    ch = host.setdefault("condition_pication_1_atom",
                                         np.zeros((1, n_total, 37), np.int32))
                    ch[0, lo:lo + k, 1] = np.asarray(pication, np.int32)
                    host["condition_pication_1_atom"] = ch
            if lig.buried:
                ch = host.setdefault("condition_rasa-binned_1_atom",
                                     np.zeros((1, n_total, 37), np.int32))
                ch[0, lo:lo + k, 1] = 1  # RasaEnum buried
                host["condition_rasa-binned_1_atom"] = ch
        # ---- ligand bond features (bonds.py semantics for atomized tokens): the
        # ligand's bond graph → bond_mask/bond_order/is_aromatic (symmetric,
        # order_lookup per biotite type), incident_bond_type_counts at slot 1
        # (each endpoint counted independently), and token_bond_path_length =
        # BFS hop count over the ligand graph (within the ligand's pseudo-chain;
        # disconnected pairs stay 0; cross-ligand/cross-chain pairs stay 0)
        if n_lig:
            order_lookup = np.zeros(10, np.int16)
            order_lookup[1], order_lookup[2] = 1, 2  # SINGLE, DOUBLE
            order_lookup[5], order_lookup[6] = 1, 2  # AROMATIC_SINGLE/DOUBLE
            order_lookup[7], order_lookup[9] = 3, 1  # TRIPLE(arom), AROMATIC
            bond_mask = np.asarray(host["bond_mask"])[0]
            bond_order = np.asarray(host["bond_order"])[0]
            is_aromatic = np.asarray(host["is_aromatic"])[0]
            incident = np.asarray(host["incident_bond_type_counts"])[0]
            path = np.asarray(host["token_bond_path_length"])[0]
            for li, lig in enumerate(self.ligands):
                lo = lig_start + lig_offsets[li]
                k = lig_sizes[li]
                if k == 0 or len(lig.bonds) == 0:
                    continue
                from scipy.sparse import csr_matrix
                from scipy.sparse.csgraph import shortest_path

                bi = lig.bonds[:, 0]
                bj = lig.bonds[:, 1]
                for a, b, bt, ar in zip(bi, bj, lig.bonds[:, 2], lig.bonds[:, 3]):
                    ta, tb = lo + int(a), lo + int(b)
                    bond_mask[ta, tb] = bond_mask[tb, ta] = True
                    bond_order[ta, tb] = bond_order[tb, ta] = int(order_lookup[int(bt)])
                    if ar:
                        is_aromatic[ta, tb] = is_aromatic[tb, ta] = True
                    incident[ta, 1, int(bt)] += 1
                    incident[tb, 1, int(bt)] += 1
                rows_g = np.concatenate([bi, bj])
                cols_g = np.concatenate([bj, bi])
                graph = csr_matrix(
                    (np.ones(len(rows_g), dtype=np.int8), (rows_g, cols_g)),
                    shape=(k, k),
                )
                sp = shortest_path(graph, directed=False, unweighted=True)
                path[lo:lo + k, lo:lo + k] = np.where(np.isfinite(sp), sp, 0.0).astype(np.int32)
            host["bond_mask"] = bond_mask[None]
            host["bond_order"] = bond_order[None]
            host["is_aromatic"] = is_aromatic[None]
            host["incident_bond_type_counts"] = incident[None]
            host["token_bond_path_length"] = path[None]
        # sequence-probability channels: unspecified (NaN) for unconditioned tokens;
        # atomized tokens and identity-conditioned tokens (motifs, add_sequence,
        # add_indexed_motif) carry all-zero finite rows, one-hot where the residue
        # name maps to a channel (the assembler stacks the individual keys)
        identity_tok = (atomize_tok | (seq_tok != 0))
        if identity_tok.any():
            from .featurize import SEQ_PROB_ORDER
            probs = np.full((n_total, 28), np.nan, np.float32)
            probs[identity_tok] = 0.0
            for t in np.flatnonzero(identity_tok & ~atomize_tok):
                name = str(idx_to_tok.get(int(seq_tok[t]), ""))
                key = f"condition_sequence-probability-{name}_1_residue"
                if key in SEQ_PROB_ORDER:
                    probs[t, SEQ_PROB_ORDER.index(key)] = 1.0
            for ci, ckey in enumerate(SEQ_PROB_ORDER):
                host[ckey] = probs[:, ci][None]
        else:
            host.setdefault("seq_probabilities", None)  # assembler stacks NaN defaults
        host = {k: v for k, v in host.items() if v is not None}
        return assemble_flow_batch(host)

    @staticmethod
    def _polymer_ordering(token_chain: np.ndarray, n: int,
                          atomize_tok: np.ndarray | None = None) -> np.ndarray:
        """Signed linkage adjacency (+1 at [downstream, upstream]) within chains.

        Pairs involving atomized tokens carry 0 (non-polymer, no canonical linkage).
        """
        idx = np.arange(n)
        same = token_chain[:, None] == token_chain[None, :]
        adjacent = np.abs(idx[:, None] - idx[None, :]) == 1
        sign = np.sign(idx[:, None] - idx[None, :])
        out = np.where(same & adjacent, sign, 0)
        if atomize_tok is not None and atomize_tok.any():
            # non-polymer tokens have no canonical linkage entries
            out[atomize_tok, :] = 0
            out[:, atomize_tok] = 0
        return out

    @staticmethod
    def _bond_path_lengths(token_chain: np.ndarray, n: int) -> np.ndarray:
        """Unclipped |i-j| hop counts within chains; 0 across chains (disconnected)."""
        idx = np.arange(n)
        same = token_chain[:, None] == token_chain[None, :]
        return np.where(same, np.abs(idx[:, None] - idx[None, :]), 0)

    def validate(self) -> list[str]:
        """Structural self-checks (condition names/domains checked on add)."""
        problems = []
        for ci, chain in enumerate(self.chains):
            for full_name in chain.conditions:
                if not full_name.startswith("condition_"):
                    problems.append(f"chain {ci}: non-condition key {full_name!r}")
        return problems

    # --------------------------------------------------------------- internals
    def _resolve_chain(self, chain: int | None) -> int:
        idx = len(self.chains) - 1 if chain is None else chain
        assert 0 <= idx < len(self.chains), f"chain {idx} out of range ({len(self.chains)} chains)"
        return idx

    def _chain(self, chain: int | None) -> ChainSpec:
        return self.chains[self._resolve_chain(chain)]


# --------------------------------------------------------------------- helpers
def _resolve_condition(name: str) -> str:
    """Map a condition name/alias to its full_name via the rfproteina registry."""
    from rfproteina.conditions import (
        C_BP, C_CATH, C_CENTROID, C_CRDGRP, C_HB, C_HPHOB, C_HOT, C_IDXGRP,
        C_INTCRD, C_MCOORD, C_NA_SS_STATE, C_PICATION, C_PISTACK, C_RAS, C_SALT, C_SEC,
    )
    from atomworks.ml.conditions import C_CRD, C_IDX, C_SEQ

    registry = {c.full_name: c for c in (
        C_HOT, C_RAS, C_SEC, C_HB, C_PISTACK, C_PICATION, C_SALT, C_HPHOB,
        C_MCOORD, C_IDXGRP, C_CATH, C_INTCRD, C_CRDGRP, C_CENTROID, C_BP,
        C_NA_SS_STATE, C_CRD, C_SEQ, C_IDX,
    )}
    for full, cond in registry.items():
        if name == full or name in getattr(cond, "aliases", ()):  # alias or full name
            return full
    raise KeyError(f"unknown condition {name!r}; known: {sorted(registry)} + aliases")


_ENUM_RANGES = {
    "condition_rasa-binned_1_atom": (0, 3),
    "condition_hbond_1_atom": (0, 3),
    "condition_pication_1_atom": (0, 3),
    "condition_saltbridge_1_atom": (0, 2),
    "condition_secondary-structure_1_residue": (0, 4),  # SSEnum: 0=none 1=helix 2=sheet 3=loop 4=non-protein
}

# residue/token-level conditions the direct de-novo featurizer supports
_RESIDUE_LEVEL_CONDITIONS = {
    "condition_secondary-structure_1_residue",
    "condition_index-group_1_token",
}

# atom-level conditions supported by the direct featurizer on MATERIALIZED
# atom37 slots (see RFD4Spec._add_atom_level / add_pos)
_ATOM_LEVEL_DIRECT_CONDITIONS = {
    "condition_rasa-binned_1_atom",
    "condition_hbond_1_atom",
    "condition_hotspot_1_atom",
    "condition_pistack_1_atom",
    "condition_pication_1_atom",
    "condition_saltbridge_1_atom",
    "condition_hydrophobic_1_atom",
    "condition_metal-coord_1_atom",
}
_DIRECT_CONDITIONS = _RESIDUE_LEVEL_CONDITIONS | _ATOM_LEVEL_DIRECT_CONDITIONS

# C_CATH class id -> (class, architecture, topology) one-hot triple (cath-label mapping)
# (C_CATH on de-novo chains is deferred: the pipeline expects authored CATH strings)


def _broadcast_chain(existing, value, mask: np.ndarray, n: int):
    """Set a condition on one chain's tokens (scalar, per-residue array, or
    per-token vector such as a CATH triple)."""
    value = np.asarray(value)
    if existing is None:
        if value.ndim == 0:
            existing = np.zeros((1, n), np.int32)
        elif value.shape[0] == int(mask.sum()):
            existing = np.zeros((1, n), np.int32)   # per-residue array
        else:
            existing = np.full((1, n, value.shape[-1]), -1, np.int32)  # per-token vector
    existing = np.asarray(existing)
    existing[0][mask] = value
    return existing


def _validate_values(full_name: str, values, num_residues: int):
    """Values: a scalar (broadcast to every realized residue — sentinel-CIF route)
    or an int/bool array of length `num_residues` (per-residue — direct-
    featurization route). Enum domains are checked against `_ENUM_RANGES`."""
    arr = np.asarray(values)
    if arr.ndim == 0:
        flat = [arr.item()]
    elif arr.ndim == 1:
        if arr.shape[0] != num_residues:
            raise ValueError(
                f"{full_name}: per-residue values must have length {num_residues} "
                f"(the chain's residue count), got {arr.shape[0]}.")
        flat = arr.tolist()
    else:
        raise ValueError(f"{full_name}: values must be a scalar or a 1-D per-residue array")
    if full_name in _ENUM_RANGES:
        lo, hi = _ENUM_RANGES[full_name]
        for v in flat:
            assert lo <= int(v) <= hi, f"{full_name}: value {v} outside [{lo}, {hi}]"
    if full_name == "condition_hotspot_1_atom":
        assert all(v in (True, False, 0, 1) for v in flat), "C_HOT is boolean"
    return values


def _format_cif_value(value) -> str:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return " ".join(_format_cif_value(v) for v in value)
    if isinstance(value, (bool, np.bool_)):
        return str(bool(value))
    return str(value)
