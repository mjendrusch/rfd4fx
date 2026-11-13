"""RFD4Input: every model input channel as fields of an immutable module.

Authoring entry point: :meth:`RFD4Input.from_chains` compiles designable
chains, indexed/unindexed motifs, ligands and interaction conditions from
plain dictionaries. Adders (``add_dssp``, ``add_hotspot``, ``add_indexed_motif``, ...)
edit channels functionally; see ``docs/from_chains.md`` for the full format."""

from __future__ import annotations

import dataclasses
from functools import lru_cache

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

# SSEnum values (atomworks.ml.transforms.dssp.SSEnum), mirrored to stay import-light
SS_NONE, SS_HELIX, SS_STRAND, SS_LOOP = 0, 1, 2, 3
_DSSP_LETTERS = {"H": 1, "E": 2, "L": 3, "X": 0, "_": 0}

#: Every RFD4 input key (the FlowBatch conditioning/annotation fields — the noisy
#: state ``x_t_*``/``t_*``, recycle counter and AE latents are sampler state, not
#: inputs). ``to_dict`` emits exactly these keys.
INPUT_FIELDS: tuple[str, ...] = (
    "padding_mask", "is_in_context_conditioning", "supports_local_latent",
    "token_type", "seq_probabilities", "atom37_mask", "coords_nm",
    "charge", "element", "ground_truth_element", "incident_bond_type_counts",
    "cond_seq", "cond_coords", "cond_internal_coords", "coord_groups",
    "rasa", "secondary_structure", "hbond", "pistack", "pication",
    "saltbridge", "hydrophobic", "metal_coord", "hotspot",
    "chain_centroid", "cath_codes", "idx_group",
    "chirality", "chiral_frame_atoms", "chiral_slot_owner",
    "token_bond_path_length", "bond_mask", "bond_order", "is_aromatic",
    "canonical_polymer_ordering", "molecule_iid", "molecule_entity_id",
    "chain_iid", "na_ss_pair", "na_ss_pair_mask", "na_ss_state",
    "na_ss_state_mask",
)

#: atom-level adder name -> whether the channel is boolean (vs int).
_ATOM_LEVEL: dict[str, bool] = {
    "rasa": False, "hbond": False, "hotspot": True, "pistack": True,
    "pication": False, "saltbridge": False, "hydrophobic": True,
    "metal_coord": False,
}


class RFD4Input(eqx.Module):
    """All RFD4 input features as (optional) jnp array fields, batch dim b, n tokens.

    Shapes mirror :class:`~rfd4fx.types.FlowBatch` field-for-field. ``None``
    means the key is absent from the input (static under jit, matching torch's
    ``if key in batch`` branching).
    """

    padding_mask: Bool[Array, "b n"] | None = None
    is_in_context_conditioning: Bool[Array, "b n"] | None = None
    supports_local_latent: Bool[Array, "b n"] | None = None
    token_type: Int[Array, "b n"] | None = None
    seq_probabilities: Float[Array, "b n 28"] | None = None
    atom37_mask: Bool[Array, "b n 37"] | None = None
    coords_nm: Float[Array, "b n 37 3"] | None = None
    charge: Int[Array, "b n 37"] | None = None
    element: Int[Array, "b n 37"] | None = None
    ground_truth_element: Int[Array, "b n 37"] | None = None
    incident_bond_type_counts: Float[Array, "b n 37 10"] | None = None
    cond_seq: Int[Array, "b n"] | None = None
    cond_coords: Float[Array, "b n 37 3"] | None = None
    cond_internal_coords: Float[Array, "b n 37 3"] | None = None
    coord_groups: Int[Array, "b n 37"] | None = None
    rasa: Int[Array, "b n 37"] | None = None
    secondary_structure: Int[Array, "b n"] | None = None
    hbond: Int[Array, "b n 37"] | None = None
    pistack: Bool[Array, "b n 37"] | None = None
    pication: Int[Array, "b n 37"] | None = None
    saltbridge: Int[Array, "b n 37"] | None = None
    hydrophobic: Bool[Array, "b n 37"] | None = None
    metal_coord: Int[Array, "b n 37"] | None = None
    hotspot: Bool[Array, "b n 37"] | None = None
    chain_centroid: Float[Array, "b n 3"] | None = None
    cath_codes: Int[Array, "b n 3"] | None = None
    idx_group: Int[Array, "b n"] | None = None
    chirality: Float[Array, "b n d"] | None = None
    chiral_frame_atoms: Int[Array, "b n s k2"] | None = None
    chiral_slot_owner: Int[Array, "b n s"] | None = None
    # metadata (not an input key): (start, end) token ranges appended by
    # add_unindexed_motif, used to resolve "last non-motif chain" defaults
    motif_token_ranges: tuple = ()
    # metadata (not an input key): per-piece internal-coordinate sources of
    # fixed=False motifs — tuples (coord_group, positions (k, 3) A, flat token
    # slots (k,)). Pieces sharing a coordinate group are merged into ONE rigid
    # body (single centroid + rotation over the union) at to_batch time;
    # single-piece groups keep their per-call payload. Cleared after merging.
    _intcrd_sources: tuple = ()
    # metadata (not an input key): authored bond edges (tok_i, slot_i, tok_j,
    # slot_j, order, backbone_linkage) — the path-length graph includes
    # backbone-linkage closures even though they are suppressed from bond_mask
    _bond_edges: tuple = ()
    token_bond_path_length: Int[Array, "b n n"] | None = None
    bond_mask: Bool[Array, "b n n"] | None = None
    bond_order: Int[Array, "b n n"] | None = None
    is_aromatic: Bool[Array, "b n n"] | None = None
    canonical_polymer_ordering: Int[Array, "b n n"] | None = None
    molecule_iid: Int[Array, "b n"] | None = None
    molecule_entity_id: Int[Array, "b n"] | None = None
    chain_iid: Float[Array, "b n"] | None = None
    na_ss_pair: Float[Array, "b n n"] | None = None
    na_ss_pair_mask: Float[Array, "b n n"] | None = None
    na_ss_state: Float[Array, "b n"] | None = None
    na_ss_state_mask: Float[Array, "b n"] | None = None

    # Not a model input: the coordinate-conditioning mask the sampler uses for
    # stochastic centering (batch["coord_cond_mask"] in the torch pipeline).
    # Carried alongside the inputs so a compiled input round-trips losslessly;
    # excluded from to_dict/to_batch.
    coord_cond: Bool[Array, "b n"] | None = None

    # ------------------------------------------------------------- construct
    @classmethod
    def from_batch(cls, batch, coord_cond=None) -> "RFD4Input":
        """Extract the input fields from a (compiled) FlowBatch."""
        return cls(
            **{f: getattr(batch, f) for f in INPUT_FIELDS},
            coord_cond=coord_cond,
        )

    @classmethod
    def from_chains(cls, *chains, np_seed: int = 0,
                    pad_to: int | None = None) -> "RFD4Input":
        """Build a compiled input from chain / ligand / structure descriptions (via
        :class:`~rfd4fx.spec.RFD4Spec`, so all featurization semantics are the
        parity-verified ones).

        Args:
          *chains: each an ``int`` (unconditioned de-novo chain of that length) or
            a dict. Recognized dict keys:

            **Polymer chains** — ``num_residues`` (or inferred from the arrays):
            ``dssp`` (str/array — C_SEC), ``sequence`` (one-letter str — C_SEQ;
            ``X`` for aa, ``N``/``X`` for dna/rna mark unknown positions and
            leave C_SEQ unset), ``pos`` (float (n, 37, 3), REQUIRED with
            ``mask``) + ``mask`` (bool (n, 37) present-atoms) — motif conditioning
            (materialized slots, ground-truth coords, C_CRD group ``group``,
            default 0), ``group`` (int — coordinate-group override),
            ``chain_index`` (int (n,) — the block is split into separate chains by
            unique value, first-appearance order; each output chain gets its own
            distinct chain identity), ``residue_index`` (int (n,) — within-chain
            bond path = |Δresidue_index| and canonical linkage at index distance
            1), and the atom-level conditions ``hotspot``/``rasa``/``hbond``/
            ``pistack``/``pication``/``saltbridge``/``hydrophobic``/
            ``metal_coord`` (scalar / (n,) / (n, 37) values, or
            ``{"values", "aa", "where"}`` sub-dicts) with chain-level ``aa`` (int
            (n,)) and ``where`` (bool (n, 37)) defaults. ``hotspot`` of shape
            (n,) requires ``mask`` and applies ``mask & hotspot[:, None]``.
            String ``hotspot`` ("1,5,6-10", optional ``@ATOM``; 1-indexed) →
            C_HOT on the sidechain slots of the listed residues (requires
            ``sequence``); ``auto_condition`` (True → C_HB roles on all
            sidechain donors/acceptors, requires ``sequence``) and the explicit
            interaction keys ``h_bond``/``salt_bridge``/``pi_stack``/
            ``pi_cation`` (residue±atom selection strings like "2@NZ,OE1" →
            per-atom C_HB/C_SALTBRIDGE/C_PISTACK/C_PICATION on the eligible
            subset) are mutually exclusive with each other. ``bonds`` (list of
            ``{"atoms": ["1@NZ", "6@SG"], "order": 1|2|3}``) authors covalent
            bonds: bond_mask/bond_order on the token pair,
            incident_bond_type_counts at the endpoints, token_bond_path_length
            recomputed by BFS (requires ``sequence``; same-token bonds set the
            counts only). ``cyclic`` (True → head-to-tail C→N closure of the
            chain with backbone-linkage semantics: canonical_polymer_ordering
            ±1, suppressed from bond_mask, ring hop counts).

            **Ligands** — ``smiles`` (str, RDKit) or ``ccd`` (id, cached CCD via
            gemmi): topology-only atomized tokens (elements, charges, bond
            features; positions diffused). ``ligand_pos`` (num_atoms, 3, Å)
            additionally coordinate-conditions every atom (C_CRD, group 0).

            **Structure motifs** — ``pdb`` / ``cif`` (path): gemmi
            ``read_structure``, first model (mmCIF files are read directly);
            optional ``chains`` (list of chain names — only these chains are
            loaded); optional ``drop_aa`` (True → backbone-only mask N/CA/C/O,
            residue types not conditioned). Parsed atom37 positions/mask/
            residue_index/chain_index/residue types are added as indexed motif
            conditioning (C_CRD group 0) on fresh chains, one per selected
            PDB chain. ``neighbours`` (True → extend the selection with
            residues of the selection's chains having a sidechain heavy atom
            within ``cutoff`` of the selected residues; requires ``select``)
            and ``surface`` (True → keep only Shrake-Rupley-surface
            neighbours) amend the selection; the base selection is kept.
            ``bonds`` (list of ``{"atoms": ["A48@NZ", "A51@OE1"],
            "order": 1|2|3}`` — chain-grammar selectors naming exactly one
            motif residue+atom each) authors file-absent covalent bonds, and
            ``cyclic`` (True → head-to-tail C→N closure of each selected
            chain) adds backbone-linkage closures; both produce the same
            bond features as the polymer-dict keys.

          np_seed: seed for the pipeline's chain/molecule-iid permutations.
          pad_to: if given, pad the compiled input to exactly this TOTAL token
            count (residue tokens + ligand atom tokens + motif tokens) by
            extending every field with inert padded slots: padding_mask False
            beyond the real tokens and schema-inactive defaults elsewhere
            (coord_groups/cath_codes/idx_group/element = -1, coordinate/sequence-
            probability fields NaN, conditions 0, pair features 0 = disconnec-
            ted). Padded slots are masked out of every attention/emit path and
            never emitted into saved structures. pad_to equal to the token
            count is a no-op; a SMALLER pad_to no-ops with a warning.

        Returns:
            A compiled :class:`RFD4Input` (one design, batch dim 1).

        Raises:
            ValueError: On unknown dict keys, out-of-range chains/residues,
                invalid selections or shapes, and empty motif selections.
        """
        from .spec import RFD4Spec

        spec = RFD4Spec()
        residue_index_by_chain: dict[int, np.ndarray] = {}
        motif_dicts: list[dict] = []
        unindexed_pdb_dicts: list[dict] = []
        deferred_indexed: list[tuple] = []
        deferred_absolute: list[tuple] = []
        deferred_hotspot: list[dict] = []
        deferred_bond_edges: list[tuple] = []
        # partially-indexed motifs: a dict with a "motif" list expands into one
        # effective pdb/cif dict per sub-spec ({select, indexed, drop_backbone,
        # ordered}, inheriting the dict-level fixed/group/hotspot/... keys).
        # Indexed sub-specs compile into the spec; unindexed ones append as a
        # block; shared positive coordinate groups auto-merge at to_batch.
        expanded: list = []
        for chain in chains:
            if isinstance(chain, dict) and "motif" in chain:
                for sub in chain["motif"]:
                    eff = {k: v for k, v in chain.items() if k != "motif"}
                    eff.update(sub)
                    indexed = eff.pop("indexed", True)
                    if not indexed:
                        eff["unindexed_motif"] = True
                    if eff.get("fixed") is False and "group" not in eff:
                        eff["group"] = 1  # sub-specs share one rigid body
                    expanded.append(eff)
            else:
                expanded.append(chain)
        chains = tuple(expanded)
        for chain in chains:
            if isinstance(chain, int):
                spec.add_chain(chain)
                continue
            if not isinstance(chain, dict):
                raise TypeError(
                    f"from_chains: chains must be int or dict, got {type(chain).__name__}")
            if "pdb" in chain or "cif" in chain:
                if chain.get("unindexed_motif"):
                    unindexed_pdb_dicts.append(chain)
                else:
                    actions = _add_pdb_motifs(spec, chain, residue_index_by_chain)
                    deferred_bond_edges.extend(actions.pop("bond_edges", ()))
                    if actions.get("fixed_false") is not None:
                        deferred_indexed.append(actions["fixed_false"])
                    if actions.get("fixed_true") is not None:
                        deferred_absolute.append(actions["fixed_true"])
                    if actions.get("hotspot") is not None \
                            or actions.get("hbond") is not None:
                        deferred_hotspot.append(actions)
                continue
            if chain.get("unindexed_motif"):  # takes precedence over smiles/ccd
                motif_dicts.append(chain)
                continue
            if "smiles" in chain or "ccd" in chain:
                _add_ligand(spec, chain)
                continue
            chain_index = chain.get("chain_index")
            if chain_index is None:
                actions = _add_dict_chain(
                    spec, chain, residue_index_by_chain,
                    token_offset=sum(c.num_residues for c in spec.chains))
                if actions:
                    deferred_bond_edges.extend(actions.pop("bond_edges", ()))
                    deferred_hotspot.append(actions)
            else:
                n = _chain_length(chain)
                ci_arr = np.asarray(chain_index)
                if ci_arr.shape != (n,):
                    raise ValueError(
                        f"from_chains: chain_index must have length {n}, got {ci_arr.shape}")
                for idx in _unique_in_order(ci_arr):
                    _add_dict_chain(spec, _slice_chain_dict(chain, ci_arr == idx),
                                    residue_index_by_chain)

        batch, coord_cond = spec.to_batch(np_seed=np_seed)
        if residue_index_by_chain:
            batch = _apply_residue_indices(batch, spec, residue_index_by_chain)
        out = cls.from_batch(batch, coord_cond)
        # deferred fixed=False indexed overlays (pdb/cif motifs): applied on
        # the compiled input BEFORE any appended motifs
        for pos_sys, where_sys, restype_sys, group in deferred_indexed:
            out = out.add_indexed_motif(pos_sys, restype_sys, where_sys,
                                        group=group, fixed=False)
        # deferred fixed=True overlays: rewrite the coordinate channels
        # as-given (undoing the spec's compile-time centering) so that all
        # absolutely-conditioned motifs live in one frame
        for pos_sys, where_sys, restype_sys in deferred_absolute:
            out = out.add_indexed_motif(pos_sys, restype_sys, where_sys,
                                        group=0, fixed=True)
        for actions in deferred_hotspot:
            for name, adder in _DEFERRED_ADDERS.items():
                payload = actions.get(name)
                if payload is None:
                    continue
                vals, where, start = payload
                out = getattr(out, adder)(vals, aa=None, where=where,
                                          start=start)
        if deferred_bond_edges:
            out = _apply_bond_edges(out, deferred_bond_edges)
        # unindexed-motif dicts route through RFD4Input.add_unindexed_motif so
        # ligand routing / fixed semantics are identical on both paths
        for md in motif_dicts:
            kwargs = _unindexed_motif_kwargs(md)
            out = out.add_unindexed_motif(**kwargs)
        for ud in unindexed_pdb_dicts:
            out = _add_pdb_unindexed_motif(out, ud)
        # ONE global centering over all absolutely-conditioned atoms (native
        # CenterOnCoordinateConditionedAtoms semantics): the spec centers its
        # motifs at compile time and post-compile overlays/blocks write
        # as-given — a final shared shift restores one consistent frame for
        # mixed indexed/unindexed motifs.
        out = _center_on_conditioned_atoms(out)
        if pad_to is not None:
            out = _pad_input(out, int(pad_to))
        return out

    # ------------------------------------------------------------- consumers
    def pad_to_nearest(self, multiple: int = 32) -> "RFD4Input":
        """Pad to the next multiple of `multiple` (total tokens, rounded UP —
        padding never truncates) and return the padded RFD4Input.

        Uses the same inert padded slots as `from_chains(pad_to=...)`
        (padding_mask False beyond the real tokens, schema-inactive defaults
        elsewhere — see `_pad_batch`). Returns `self` unchanged when the token
        count is already a multiple.
        """
        if multiple < 1:
            raise ValueError(f"pad_to_nearest: multiple must be >= 1, got {multiple}")
        n = int(self.padding_mask.shape[1])
        target = ((n + multiple - 1) // multiple) * multiple
        if target == n:
            return self
        return _pad_input(self, target)

    def to_dict(self) -> dict:
        """All populated RFD4 input keys -> jnp arrays (unset keys omitted)."""
        return {f: getattr(self, f) for f in INPUT_FIELDS if getattr(self, f) is not None}

    def to_batch(self):
        """Pack the inputs into a FlowBatch for the denoiser/AE.

        fixed=False motif pieces sharing a coordinate group are merged into
        one rigid body first (single centroid + rotation over the union of
        their atoms — see `_intcrd_sources`)."""
        from .types import FlowBatch

        inp = self._finalize_intcrd_groups() if self._intcrd_sources else self
        d = inp.to_dict()
        if "padding_mask" not in d:
            raise ValueError("to_batch: padding_mask must be set")
        return FlowBatch(**d)

    def _finalize_intcrd_groups(self) -> "RFD4Input":
        """Merge multi-piece internal-coordinate groups (auto-merge rule).

        All fixed=False motif pieces recorded under the same positive
        coordinate-group label form ONE rigid body: the internal-coordinate
        payload is recomputed over the union of their atoms (single centroid +
        single random rotation). Single-piece groups keep the per-call payload
        written by the adder. Returns a copy with `_intcrd_sources` cleared
        (merging happens once)."""
        by_group: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
        for g, pos, slots in self._intcrd_sources:
            by_group.setdefault(int(g), []).append((np.asarray(pos), np.asarray(slots)))
        if not any(len(pieces) > 1 for pieces in by_group.values()):
            return dataclasses.replace(self, _intcrd_sources=())
        if self.cond_internal_coords is None:
            raise ValueError(
                "_finalize_intcrd_groups: shared coordinate groups recorded "
                "but cond_internal_coords is unset")
        intcrd = np.array(self.cond_internal_coords, copy=True)
        for g, pieces in by_group.items():
            if len(pieces) < 2:
                continue  # single piece: per-call payload already correct
            pos = np.concatenate([p for p, _ in pieces])
            slots = np.concatenate([s for _, s in pieces]).astype(np.intp)
            if not np.isfinite(pos).all():
                raise ValueError(
                    f"_finalize_intcrd_groups: group {g} has non-finite motif "
                    "positions")
            center = pos.mean(0, keepdims=True)
            payload = ((pos - center) @ _random_rotation(np.random)).astype(np.float32)
            intcrd.reshape(-1, 3)[slots] = payload
        return dataclasses.replace(
            self, cond_internal_coords=jnp.asarray(intcrd), _intcrd_sources=())

    # --------------------------------------------------------------- editing
    def add_dssp(self, dssp: str | np.ndarray | Array, start: int = 0,
                 where=True) -> "RFD4Input":
        """Replace the secondary-structure condition (C_SEC) and return a new
        RFD4Input.

        A string decodes strictly as ``H``→1, ``E``→2, ``L``→3, ``X``/``_``→0
        (RFD4Spec.add_dssp semantics); an np.ndarray is converted with
        ``jnp.array`` into the same jax path as an already-jitted array. The
        window is residues ``start:start + len(dssp)`` (must fit within the
        total residues). ``where`` selects the positions to change: ``True``
        changes the entire window, a bool (num_residues,) array changes only the
        True residues (keeping the rest as-is); a (num_residues, 37) array is
        rejected (C_SEC is residue-level).
        """
        if isinstance(dssp, str):
            bad = sorted(set(dssp) - set(_DSSP_LETTERS))
            if bad:
                raise ValueError(
                    f"add_dssp: unsupported DSSP character(s) {bad!r} — this parser "
                    "is strictly 3-state: H (helix), E (strand), L (loop), X/_ "
                    "(unspecified).")
            values = jnp.asarray([_DSSP_LETTERS[ch] for ch in dssp], jnp.int32)
        else:  # np.ndarray and jax.Array share the jnp conversion path
            values = jnp.asarray(dssp).astype(jnp.int32)
        if values.ndim != 1:
            raise ValueError(f"add_dssp: expected a 1-D string/array, got shape {values.shape}")
        b, n = self.padding_mask.shape
        if b != 1:
            raise ValueError("add_dssp: RFD4Input is a single-design container (b == 1)")
        start = int(start)
        L = values.shape[0]
        if start < 0 or start + L > n:
            raise ValueError(
                f"add_dssp: residues [{start}, {start + L}) out of range for {n} residues")
        sel = _resolve_where(where, L, atom_level=False)
        ss = self.secondary_structure
        if ss is None:  # unconditioned so far: C_SEC default is 0 (unspecified)
            ss = jnp.zeros((b, n), jnp.int32)
        old = ss[0, start:start + L]
        ss = ss.at[0, start:start + L].set(
            jnp.where(sel, values.astype(ss.dtype), old))
        return dataclasses.replace(self, secondary_structure=ss)

    def add_bp(self, bp: str, start: int = 0, where=True) -> "RFD4Input":
        """Add dot-bracket NA secondary-structure conditioning (C_NA_SS_STATE +
        C_BP) to the input, mirroring :meth:`add_dssp`.

        The dot-bracket string (native syntax: ``.``, ``()[]{}<>`` and letter
        families for nested/crossing pairs, plus ``x``/``_``/``X`` for
        unknown/unspecified) is parsed into per-position states
        (BasePairEnum: 0=unspecified, 1=paired, 2=explicitly-unpaired loop)
        and base pairs. ``L = len(bp)``; the window is tokens
        ``start:start + L`` (the compiled token axis, like add_dssp).

        - states: written where ``where`` selects (``True`` → the whole
          window; a bool (L,) array → per-token), replacing previous values;
          ``na_ss_state_mask`` = value != 0.
        - pairs: ``pair_where = where[:, None] & where[None, :]`` gates the
          ``[start:start+L, start:start+L]`` block. The block is cleared where
          ``pair_where`` (replace semantics), then each parsed pair is written
          at ``[start+i, start+j]`` AND ``[start+j, start+i]`` where selected
          (symmetric, as the native C_BP featurization).
        - LOOP negatives: tokens whose NEW state is 2 (explicitly unpaired)
          get their ENTIRE row and column of ``na_ss_pair_mask`` set True and
          any pair values cleared (native ConvertToAtom37Format semantics).
        - validation: any window token receiving a non-UNSPECIFIED state or a
          pair endpoint must be a nucleic-acid token (token_type 1 = rna /
          2 = dna), else ValueError.

        Functional update (returns a new module).
        """
        from .spec import BP_LOOP, BP_UNSPECIFIED, parse_dot_bracket

        states_np, pairs = parse_dot_bracket(bp)
        L = states_np.shape[0]
        states = jnp.asarray(states_np.astype(np.float32))
        b, n = self.padding_mask.shape
        if b != 1:
            raise ValueError("add_bp: RFD4Input is a single-design container (b == 1)")
        start = int(start)
        if start < 0 or start + L > n:
            raise ValueError(
                f"add_bp: window [{start}, {start + L}) out of range for {n} tokens")
        sel = _resolve_where(where, L, atom_level=False)
        # pair-endpoint mask over the window
        endp = np.zeros(L, bool)
        for i, j in pairs:
            endp[i] = endp[j] = True
        endp = jnp.asarray(endp)
        # nucleic-acid validation: tokens receiving a state or a pair endpoint
        tt = self.token_type
        if tt is not None:
            tt_w = np.asarray(tt)[0, start:start + L]
            required = np.asarray(sel) & ((np.asarray(states_np) != BP_UNSPECIFIED)
                                          | np.asarray(endp))
            bad = sorted(int(t) for t in np.flatnonzero(required)
                         if int(tt_w[t]) not in (1, 2))
            if bad:
                raise ValueError(
                    f"add_bp: non-nucleic-acid token(s) {bad} in window "
                    f"[{start}, {start + L}) — C_NA_SS_STATE/C_BP require "
                    "token_type 1 (rna) or 2 (dna)")
        # states: replace semantics inside the window where selected
        old_state = (self.na_ss_state if self.na_ss_state is not None
                     else jnp.zeros((b, n), jnp.float32))
        old_smask = (self.na_ss_state_mask if self.na_ss_state_mask is not None
                     else jnp.zeros((b, n), bool))
        win = slice(start, start + L)
        new_state = old_state.at[0, win].set(
            jnp.where(sel, states, old_state[0, win]))
        new_smask = old_smask.at[0, win].set(
            jnp.where(sel, states != BP_UNSPECIFIED, old_smask[0, win]))
        # pairs: clear the gated block, then write the parsed pairs (both
        # directions) where pair_where
        pair_where = sel[:, None] & sel[None, :]
        old_pair = (self.na_ss_pair if self.na_ss_pair is not None
                    else jnp.zeros((b, n, n), jnp.float32))
        old_pmask = (self.na_ss_pair_mask if self.na_ss_pair_mask is not None
                     else jnp.zeros((b, n, n), bool))
        new_pair = old_pair.at[0, win, win].set(
            jnp.where(pair_where, 0.0, old_pair[0, win, win]))
        new_pmask = old_pmask.at[0, win, win].set(
            jnp.where(pair_where, False, old_pmask[0, win, win]))
        for i, j in pairs:
            if not bool(pair_where[i, j]):
                continue
            a, c = start + i, start + j
            new_pair = new_pair.at[0, a, c].set(1.0).at[0, c, a].set(1.0)
            new_pmask = new_pmask.at[0, a, c].set(True).at[0, c, a].set(True)
        # LOOP explicit negatives: clear pair values and set the full row/col
        # mask for tokens whose NEW state is LOOP
        new_states_w = np.asarray(new_state)[0, start:start + L]
        for t in (start + int(t) for t in
                  np.flatnonzero((new_states_w == BP_LOOP) & np.asarray(sel))):
            new_pair = new_pair.at[0, t, :].set(0.0).at[0, :, t].set(0.0)
            new_pmask = new_pmask.at[0, t, :].set(True).at[0, :, t].set(True)
        return dataclasses.replace(self, na_ss_state=new_state,
                                   na_ss_state_mask=new_smask,
                                   na_ss_pair=new_pair,
                                   na_ss_pair_mask=new_pmask)

    def add_rasa(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Binned RASA (C_RAS): 0=unspecified, 1=buried, 2=intermediate, 3=exposed."""
        return self._add_atom_level("rasa", values, aa, where, start)

    def add_hbond(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Hydrogen-bond roles (C_HB): 0=none, 1=donor, 2=acceptor, 3=both."""
        return self._add_atom_level("hbond", values, aa, where, start)

    def add_hotspot(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Hotspot residues (C_HOT, boolean)."""
        return self._add_atom_level("hotspot", values, aa, where, start)

    def add_pistack(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Pi-stacking (C_PISTACK, boolean)."""
        return self._add_atom_level("pistack", values, aa, where, start)

    def add_pication(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Pi-cation (C_PICATION)."""
        return self._add_atom_level("pication", values, aa, where, start)

    def add_saltbridge(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Salt bridges (C_SALT)."""
        return self._add_atom_level("saltbridge", values, aa, where, start)

    def add_hydrophobic(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Hydrophobic contacts (C_HPHOB, boolean)."""
        return self._add_atom_level("hydrophobic", values, aa, where, start)

    def add_metal_coord(self, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Metal coordination (C_MCOORD)."""
        return self._add_atom_level("metal_coord", values, aa, where, start)

    def add_indexed_motif(self, pos, aa, mask, group=0, where=True,
                          fixed=True) -> "RFD4Input":
        """Indexed motif edit over the WHOLE system (pos/aa/mask have shape[0]
        equal to the token count).

        At every position selected by ``where`` (``True`` → everywhere, a bool
        (n,) array → whole residues, a bool (n, 37) array → per atom) with a
        finite ``pos`` value:

        - ``fixed=True`` (default): ABSOLUTE coordinate conditioning — the
          motif coordinate is written to cond_coords (Å) AND the ground-truth
          coords (nm), and the coordinate group is set to ``group`` (scalar,
          or a per-token (n,) array).
        - ``fixed=False``: INTERNAL coordinate conditioning — cond_coords and
          the ground-truth coords stay untouched, the coordinate group is set
          to ``group`` (which must be positive — group 0 is reserved for
          absolute conditioning), and cond_internal_coords is written with the
          payload centered per unique group over the selected positions and
          randomly rotated (native MaterializeInternalCoords semantics).

        ``mask`` is unioned into atom37_mask (materializing the motif's atoms)
        and the identity is set from ``aa`` at motif rows where ``aa != 0``.
        Positions are used as-given (the caller authors in the current frame —
        no re-centering). Functional update (returns a new module).
        """
        b, n = self.padding_mask.shape
        if b != 1:
            raise ValueError("add_indexed_motif: RFD4Input is a single-design container")
        pos_arr = np.asarray(pos, np.float32)
        aa_arr = np.asarray(aa).astype(np.int32).reshape(-1)
        mask_arr = np.asarray(mask).astype(bool)
        m = aa_arr.shape[0]
        if m != n:
            raise ValueError(
                f"add_indexed_motif: pos/aa/mask must span all {n} tokens, got {m}")
        if pos_arr.shape != (n, 37, 3) or mask_arr.shape != (n, 37):
            raise ValueError(
                f"add_indexed_motif: pos must be (n, 37, 3) and mask (n, 37) for "
                f"{n} tokens, got {pos_arr.shape} and {mask_arr.shape}")
        sel = np.asarray(_resolve_where(where, n, atom_level=True))
        grp = np.asarray(group)
        grp = (np.full(n, int(grp), np.int32) if grp.ndim == 0
               else grp.astype(np.int32).reshape(-1))
        if grp.shape != (n,):
            raise ValueError(
                f"add_indexed_motif: group must be a scalar or length-{n}, got {grp.shape}")
        if not fixed and (grp < 1).any():
            raise ValueError(
                "add_indexed_motif: fixed=False requires positive coordinate "
                "groups (group 0 is reserved for absolute conditioning)")
        finite = np.isfinite(pos_arr).all(-1) & sel  # (n, 37)
        src = list(self._intcrd_sources)

        cc = (np.array(self.cond_coords, copy=True) if self.cond_coords is not None
              else np.full((b, n, 37, 3), np.nan, np.float32))
        coords = (np.array(self.coords_nm, copy=True) if self.coords_nm is not None
                  else np.zeros((b, n, 37, 3), np.float32))
        if fixed:
            cc[0, finite] = pos_arr[finite]  # C_CRD stays in Å
            coords[0, finite] = pos_arr[finite] / 10.0  # ground truth in nm
        cg = (np.array(self.coord_groups, copy=True) if self.coord_groups is not None
              else np.full((b, n, 37), -1, np.int32))
        cg[0, finite] = np.broadcast_to(grp[:, None], (n, 37))[finite]
        intcrd = (np.array(self.cond_internal_coords, copy=True)
                  if self.cond_internal_coords is not None
                  else np.full((b, n, 37, 3), np.nan, np.float32))
        if not fixed:
            # per-group centered + randomly rotated payload over the selected
            # finite positions (native MaterializeInternalCoords semantics);
            # pieces sharing a group are re-merged over the union at to_batch
            for g in np.unique(grp[finite.any(-1)]):
                gsel = finite & (grp[:, None] == g)
                xy = pos_arr[gsel]
                payload = (xy - xy.mean(0, keepdims=True)) @ _random_rotation(np.random)
                intcrd[0, gsel] = payload
                slots = np.flatnonzero(gsel.reshape(-1))
                src.append((int(g), pos_arr.reshape(-1, 3)[slots], slots))
        mask_f = (np.array(self.atom37_mask, copy=True) if self.atom37_mask is not None
                  else np.zeros((b, n, 37), bool))
        mask_f[0] |= mask_arr
        seq = np.array(self.cond_seq, copy=True) if self.cond_seq is not None \
            else np.zeros((b, n), np.int32)
        rows = mask_arr.any(-1) & (aa_arr != 0)
        seq[0] = np.where(rows, aa_arr, seq[0])
        # identity must ALSO reach the trunk via seq_probabilities (the spec
        # path builds it from condition_sequence; the overlay bypasses the
        # spec, so maintain the one-hot here)
        sp = (np.array(self.seq_probabilities, copy=True)
              if self.seq_probabilities is not None
              else np.full((b, n, 28), np.nan, np.float32))
        sp = _apply_seq_identity(sp, np.flatnonzero(rows), aa_arr[rows])
        cc_cond = (np.array(self.coord_cond, copy=True) if self.coord_cond is not None
                   else np.zeros((b, n), bool))
        if fixed:  # only absolute conditioning enters the sampler's coord_cond
            cc_cond[0] |= finite.any(-1)
        return dataclasses.replace(
            self, cond_coords=jnp.asarray(cc), coords_nm=jnp.asarray(coords),
            coord_groups=jnp.asarray(cg), atom37_mask=jnp.asarray(mask_f),
            cond_internal_coords=jnp.asarray(intcrd),
            cond_seq=jnp.asarray(seq), seq_probabilities=jnp.asarray(sp),
            coord_cond=jnp.asarray(cc_cond),
            _intcrd_sources=tuple(src))

    def add_unindexed_motif(self, pos, aa, mask, residue_index=None,
                            chain_index=None, group=0, fixed=True,
                            element=None, charge=None, bonds=None,
                            chain=None, ordered: bool | None = None,
                            target_chain=None) -> "RFD4Input":
        """Append an unindexed motif: `num_motif_tokens` real tokens built from
        position / residue-type / atom-mask information of shape
        (num_motif_tokens, ...). Appending does not overwrite existing
        information (hence no ``where``).

        ``aa`` carries RFD4 encoding indices per motif token (residues: AE
        residue-vocabulary indices; small-molecule atoms: the atomized ``<A>``
        index — each atom one token). ``mask`` marks the specified atom37
        slots.

        **Ligand routing** (the trained native shape,
        `TipAtomPartialLigandTask`): maximal runs of consecutive ``<A>`` tokens
        are LIGANDS and are added as INDEXED design tokens (is_in_context=
        False, token_type=3, fresh chain/molecule ids per ligand, bond-graph
        path lengths within the ligand) with full chemistry from the optional
        ``element`` (atomic numbers), ``charge`` (formal charges) and ``bonds``
        (``(n_bonds, 4) = [i, j, order, aromatic]``, local to the ligand
        tokens, in order of appearance). All non-ligand residues stay fully
        unindexed (in-context) as before.

        **``fixed``** selects the coordinate-conditioning mode:
        ``fixed=True`` (default) prepares ABSOLUTE coordinate conditioning
        (C_CRD: cond_coords from ``pos``, coord_cond set, coordinate groups per
        the shared rule — ``group`` with ``residue_index``, otherwise fresh
        per-residue groups with each ligand run sharing one). ``fixed=False``
        prepares INTERNAL coordinate conditioning (C_INTCRD/C_CRDGRP, the
        trained native shape): cond_coords stay NaN, coord_cond is not set, and
        each rigid body — the residue motif (one group with ``residue_index``,
        per-residue groups otherwise) and each ligand (its own group) — gets a
        fresh positive coordinate group with the payload centered on the
        group's atoms and randomly rotated (mirroring the native
        MaterializeInternalCoords).

        ``residue_index`` (an int array, recorded for authoring/reference
        purposes; the native unindexed semantics carry NO index information on
        in-context tokens — each gets its own index group with no bond paths
        and no polymer ordering). ``chain_index`` (array) sets the motif
        tokens' chain index (chain_iid).

        ``ordered`` (pdb/cif motif route; None = the default per-token
        semantics): True -> one index group per CONTIGUOUS same-chain run of
        the motif residues (contiguity from ``residue_index`` +
        ``chain_index``); False -> per-token groups. Ligand runs are always
        isolated into their own index groups (native: indexed ligand tokens
        can never share an index group with unindexed tokens).

        ``chain`` selects the motif residues' CHAIN TOPOLOGY: ``None`` (the
        native unindexed-benchmark shape) joins the motif residues to the LAST
        non-motif chain (shared chain_iid / molecule ids); an ``int`` attaches
        them to the i-th existing non-motif chain (first-appearance order,
        ValueError if out of range); an int array of shape ``(m,)`` attaches
        each motif residue to the i-th existing chain; the string ``"new"``
        gives the residues their own fresh chain (the pre-native-faithful
        behavior). Ligand runs ALWAYS get their own fresh chains (a ligand is
        a separate molecule, the trained theozyme shape). Raises if the input
        already contains padded slots (append semantics assume real tokens at
        the front — pad after adding motifs)."""
        b, n = self.padding_mask.shape
        if b != 1:
            raise ValueError("add_unindexed_motif: RFD4Input is a single-design container")
        if not bool(np.asarray(self.padding_mask).all()):
            raise ValueError(
                "add_unindexed_motif: the input contains padded slots — append "
                "semantics require all-real tokens (call pad_to_nearest AFTER "
                "adding motifs)")
        pos_arr = np.asarray(pos, np.float32)
        aa_arr = np.asarray(aa).astype(np.int32).reshape(-1)
        mask_arr = np.asarray(mask).astype(bool)
        m = aa_arr.shape[0]
        if m == 0:
            return self
        if pos_arr.shape != (m, 37, 3) or mask_arr.shape != (m, 37):
            raise ValueError(
                f"add_unindexed_motif: pos must be (num_motif_tokens, 37, 3) and "
                f"mask (num_motif_tokens, 37) with num_motif_tokens={m}, got "
                f"{pos_arr.shape} and {mask_arr.shape}")
        ri = None
        if residue_index is not None:
            ri = np.asarray(residue_index, np.int32).reshape(-1)
            if ri.shape != (m,):
                raise ValueError(
                    f"add_unindexed_motif: residue_index must have length {m}, "
                    f"got {ri.shape}")
        ci = None
        if chain_index is not None:
            ci = np.asarray(chain_index, np.float32).reshape(-1)
            if ci.shape != (m,):
                raise ValueError(
                    f"add_unindexed_motif: chain_index must have length {m}, "
                    f"got {ci.shape}")
        chem = _validate_ligand_chemistry(aa_arr, element, charge, bonds)
        return _append_motif_tokens(self, pos_arr, aa_arr, mask_arr, ri, ci,
                                    group, fixed=fixed, chem=chem, chain=chain,
                                    ordered=ordered, target_chain=target_chain)

    def add_bonds(self, *bonds) -> "RFD4Input":
        """Author covalent bonds on the compiled input (functional update).

        Each entry is a tuple ``(res_1, atom_1, res_2, atom_2, order)`` with
        all indices 0-based into the ``(num_residues, 37)`` atom array
        (res = token index, atom = atom37 slot) and ``order`` ∈ {1, 2, 3}.

        Sets `bond_mask`/`bond_order` (symmetric) for the token pair,
        increments `incident_bond_type_counts` at all four endpoint slots
        (biotite ids: single=1, double=2, triple=3), and recomputes
        `token_bond_path_length` by BFS over consecutive-token adjacency plus
        the authored edges. Bonds within one token set the counts only (no
        pair-track representation). Features are identical to authoring the
        same bonds through `from_chains` `"bonds"` keys.
        """
        n = int(np.asarray(self.padding_mask).shape[1])
        edges = []
        for bond in bonds:
            r1, a1, r2, a2, order = bond
            if not (0 <= int(r1) < n and 0 <= int(r2) < n):
                raise ValueError(
                    f"add_bonds: residue indices {(r1, r2)} outside 0..{n - 1}")
            if not (0 <= int(a1) < 37 and 0 <= int(a2) < 37):
                raise ValueError(
                    f"add_bonds: atom indices {(a1, a2)} outside 0..36")
            if int(order) not in (1, 2, 3):
                raise ValueError(
                    f"add_bonds: order must be 1, 2 or 3, got {order!r}")
            edges.append((int(r1), int(a1), int(r2), int(a2), int(order), False))
        return _apply_bond_edges(self, edges)

    def cyclize(self, chain_index: int) -> "RFD4Input":
        """Head-to-tail closure of one polymer chain (functional update).

        Bonds `C` (atom37 slot 2) of the chain's LAST token to `N` (slot 0) of
        its FIRST token with backbone-linkage semantics: `bond_mask` stays
        False (suppressed), `canonical_polymer_ordering[first, last] = +1`
        (first = downstream N-entering, last = upstream C-leaving) and -1 at
        the transpose, `incident_bond_type_counts` +1 (single) on both closure
        atoms, and `token_bond_path_length` recomputed through the closure
        edge (ring hop counts).

        `chain_index` selects the polymer chain by 0-based from_chains add
        order (mapped through the first-appearance order of `chain_iid`).
        """
        ranges = _chain_token_ranges(self)
        if not 0 <= int(chain_index) < len(ranges):
            raise ValueError(
                f"cyclize: chain_index {chain_index} outside 0..{len(ranges) - 1}")
        start, end = ranges[int(chain_index)]
        if end - start < 2:
            raise ValueError(
                f"cyclize: chain {chain_index} has fewer than 2 tokens")
        return _apply_bond_edges(self, [(end - 1, 2, start, 0, 1, True)])

    def _add_atom_level(self, field: str, values, aa, where=True, start: int = 0) -> "RFD4Input":
        """Shared atom-level edit over residues ``start:start + k`` (k resolved
        from an array `where`, else a 1-D `values`, else `aa`).

        ``where`` selects the positions to change: ``True`` changes the entire
        conditioning input over the window (all 37 slots of every residue), a
        bool (k,) array changes all 37 slots of the True residues only, and a
        bool (k, 37) array changes only the True atom positions. Every written
        position materializes its atom37 slot (union into atom37_mask) and the
        channel value is written only at the written positions (values elsewhere
        are preserved). ``aa`` sets the per-residue identity (C_SEQ) at residues
        touched by `where` where ``aa != 0`` (0 = <M> = leave unspecified).
        All jnp, functional update (returns a new module).
        """
        b, n = self.padding_mask.shape
        if b != 1:
            raise ValueError(f"add_{field}: RFD4Input is a single-design container (b == 1)")
        where_arr = None if where is True else jnp.asarray(where).astype(bool)
        vals_in = jnp.asarray(values)
        aa_in = None if aa is None else jnp.asarray(aa).astype(jnp.int32).reshape(-1)
        # window length: array where > 1-D values > aa
        k = None
        if where_arr is not None:
            if where_arr.ndim not in (1, 2):
                raise ValueError(
                    f"add_{field}: where must be True, (k,) or (k, 37) bool; got "
                    f"shape {where_arr.shape}")
            k = where_arr.shape[0]
        elif vals_in.ndim >= 1:
            k = vals_in.shape[0]
        elif aa_in is not None:
            k = aa_in.shape[0]
        if k is None:
            raise ValueError(
                f"add_{field}: cannot determine the number of residues — pass a "
                "per-residue `values` array, `aa`, or an array `where`")
        if aa_in is not None and aa_in.shape[0] != k:
            raise ValueError(
                f"add_{field}: aa must have length {k}, got {aa_in.shape[0]}")
        if aa_in is not None and vals_in.ndim >= 1 and vals_in.shape[0] != k:
            raise ValueError(
                f"add_{field}: values must have length {k}, got {vals_in.shape[0]}")
        sel = _resolve_where(where, k, atom_level=True)  # (k, 37) bool
        start = int(start)
        if start < 0 or start + k > n:
            raise ValueError(
                f"add_{field}: residues [{start}, {start + k}) out of range for {n} residues")
        vals = (jnp.full((k,), vals_in) if vals_in.ndim == 0
                else (vals_in if vals_in.ndim == 2 and vals_in.shape == (k, 37)
                      else vals_in.reshape(-1)))

        # identity (C_SEQ): aa != 0 overrides at residues touched by `where`
        sp = None
        seq = self.cond_seq
        if aa_in is not None:
            if seq is None:
                seq = jnp.zeros((b, n), jnp.int32)
            touched = sel.any(axis=-1)
            aa_t = jnp.where(touched, aa_in, 0)
            tgt_seq = seq[0, start:start + k]
            seq = seq.at[0, start:start + k].set(
                jnp.where(aa_t != 0, aa_t, tgt_seq).astype(seq.dtype))
            # the trunk reads identity via seq_probabilities (see
            # add_indexed_motif) — maintain the one-hot here as well
            sp = (np.array(self.seq_probabilities, copy=True)
                  if self.seq_probabilities is not None
                  else np.full((b, n, 28), np.nan, np.float32))
            hits = np.flatnonzero(np.asarray(aa_t) != 0)
            sp = _apply_seq_identity(sp, start + hits,
                                     np.asarray(aa_t)[hits])

        # materialize the written slots (union with anything already there)
        mask = self.atom37_mask
        if mask is None:
            mask = jnp.zeros((b, n, 37), bool)
        mask = mask.at[0, start:start + k].set(mask[0, start:start + k] | sel)

        # channel values: write only inside `where`, preserve everything else
        ch = getattr(self, field)
        dtype = (ch.dtype if ch is not None
                 else (jnp.bool_ if _ATOM_LEVEL[field] else jnp.int32))
        if ch is None:
            ch = jnp.zeros((b, n, 37), dtype)
        vals_w = (vals.astype(dtype)[:, None] if vals.ndim == 1
                  else vals.astype(dtype))
        ch = ch.at[0, start:start + k].set(
            jnp.where(sel, vals_w, ch[0, start:start + k]))

        out_kwargs = {field: ch}
        if sp is not None:
            out_kwargs["seq_probabilities"] = jnp.asarray(sp)
        return dataclasses.replace(self, cond_seq=seq, atom37_mask=mask,
                                   **out_kwargs)


def _recompute_token_path_lengths(n: int, edges) -> np.ndarray:
    """BFS hop counts over consecutive-token adjacency + authored bond edges
    (native _compute_token_path_lengths semantics, unclipped; the closure
    edge of a ring IS part of the graph — verified against the native
    pipeline on the head-to-tail fixture)."""
    idx = np.arange(n, dtype=np.int32)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(np.int32)
    pairs = [(int(e[0]), int(e[2])) for e in edges if int(e[0]) != int(e[2])]
    if not pairs or n == 0:
        return dist
    portals = np.array(sorted({p for ab in pairs for p in ab}), dtype=np.int32)
    p_of = {int(p): i for i, p in enumerate(portals)}
    dp = np.abs(portals[:, None] - portals[None, :]).astype(np.int32)
    for a, b in pairs:
        pa, pb = p_of[a], p_of[b]
        dp[pa, pb] = min(int(dp[pa, pb]), 1)
        dp[pb, pa] = min(int(dp[pb, pa]), 1)
    for k in range(len(portals)):
        np.minimum(dp, dp[:, k:k + 1] + dp[k:k + 1, :], out=dp)
    d_to_p = np.abs(idx[:, None] - portals[None, :]).astype(np.int32)
    via_p = (d_to_p[:, :, None] + dp[None, :, :]).min(axis=1)
    for q_i in range(len(portals)):
        np.minimum(dist, via_p[:, q_i:q_i + 1] + d_to_p[:, q_i:q_i + 1].T, out=dist)
    return dist


def _apply_bond_edges(inp: "RFD4Input", edges) -> "RFD4Input":
    """Author token-level bond edges on a compiled input.

    Each edge: `(tok_i, slot_i, tok_j, slot_j, order, backbone_linkage)`.
    Non-backbone edges set `bond_mask`/`bond_order` (symmetric); backbone
    linkages (head-to-tail closures) are suppressed from `bond_mask` and set
    `canonical_polymer_ordering` (+1 at [downstream, upstream]) instead —
    bonds.py semantics, see info.md. Both kinds increment
    `incident_bond_type_counts` at every endpoint slot and recompute
    `token_bond_path_length` over the full edge set."""
    n = int(np.asarray(inp.padding_mask).shape[1])
    zeros_bm = np.zeros((n, n), dtype=bool)
    zeros_i = np.zeros((n, n), np.int32)
    bm = (np.asarray(inp.bond_mask)[0].copy() if inp.bond_mask is not None
          else zeros_bm)
    bo = (np.asarray(inp.bond_order)[0].copy() if inp.bond_order is not None
          else zeros_i)
    canon = (np.asarray(inp.canonical_polymer_ordering)[0].copy()
             if inp.canonical_polymer_ordering is not None else zeros_i)
    inc = (np.asarray(inp.incident_bond_type_counts)[0].copy()
           if inp.incident_bond_type_counts is not None
           else np.zeros((n, 37, 10), np.float32))
    all_edges = tuple(getattr(inp, "_bond_edges", ())) + tuple(edges)
    for ti, si, tj, sj, order, backbone in edges:
        if backbone:
            canon[tj, ti] = 1        # tj = downstream (entering N)
            canon[ti, tj] = -1       # ti = upstream (leaving C)
        elif ti != tj:
            bm[ti, tj] = bm[tj, ti] = True
            bo[ti, tj] = bo[tj, ti] = order
        inc[ti, si, order] += 1.0
        inc[tj, sj, order] += 1.0
    pl = _recompute_token_path_lengths(n, all_edges)
    return dataclasses.replace(
        inp, bond_mask=jnp.asarray(bm)[None],
        bond_order=jnp.asarray(bo)[None],
        canonical_polymer_ordering=jnp.asarray(canon)[None],
        incident_bond_type_counts=jnp.asarray(inc)[None],
        token_bond_path_length=jnp.asarray(pl)[None],
        _bond_edges=tuple(all_edges))


def _chain_token_ranges(inp: "RFD4Input") -> list[tuple[int, int]]:
    """(start, end) token ranges of the polymer chains, in first-appearance
    order of their chain_iid values (== from_chains add order)."""
    ciid = np.asarray(inp.chain_iid)[0]
    ranges: list[tuple[int, int]] = []
    seen: dict[float, int] = {}
    for t, v in enumerate(ciid):
        if v not in seen:
            seen[v] = len(ranges)
            ranges.append([t, t + 1])
        else:
            ranges[seen[v]][1] = t + 1
    return [(int(a), int(b)) for a, b in ranges]


def _chain_length(chain: dict) -> int:
    """Infer a dict chain's residue count (explicit key first, then contents)."""
    for key in ("num_residues", "length"):
        if key in chain:
            return int(chain[key])
    if "sequence" in chain:
        return len(chain["sequence"])
    if "dssp" in chain:
        return len(chain["dssp"])
    if "bp" in chain:
        return len(chain["bp"])
    if "pos" in chain:
        return int(np.asarray(chain["pos"]).shape[0])
    for key in ("residue_index", "aa"):
        if key in chain:
            return len(chain[key])
    for name in _ATOM_LEVEL:  # atom-level values may state the length alone
        if name in chain:
            item = chain[name]["values"] if isinstance(chain[name], dict) else chain[name]
            arr = np.asarray(item["values"] if isinstance(item, dict) else item)
            if arr.ndim >= 1:
                return int(arr.shape[0])
    raise ValueError(
        "from_chains: cannot infer the chain length from the dict — add "
        "'num_residues', 'dssp', 'bp', 'pos', 'residue_index', 'aa', or an atom-level "
        "condition with a 1-D value array")


def _apply_residue_indices(batch, spec, ri_by_chain: dict[int, np.ndarray]):
    """Re-write the within-chain pair features for chains with explicit
    residue indices: token_bond_path_length = |Δresidue_index| (the verified
    native semantics — path = |Δres_id| for same-group pairs) and canonical
    polymer linkage ±1 exactly at residue-index distance 1. Cross-chain pairs
    stay 0/disconnected. numpy host-side (constructor path), jnp afterwards.
    """
    from .types import FlowBatch

    fields = {f: np.array(getattr(batch, f), copy=True)
              for f in INPUT_FIELDS if getattr(batch, f) is not None}
    lo = 0
    for ci, ch in enumerate(spec.chains):
        hi = lo + ch.num_residues
        if ci in ri_by_chain:
            ri = ri_by_chain[ci]
            dist = np.abs(ri[:, None] - ri[None, :]).astype(np.int32)
            fields["token_bond_path_length"][0, lo:hi, lo:hi] = dist
            # +1 at [downstream, upstream] (larger residue index first) — the
            # same convention as the linear-chain default (sign(i - j))
            link = (np.sign(ri[:, None] - ri[None, :]) * (dist == 1)).astype(np.int32)
            fields["canonical_polymer_ordering"][0, lo:hi, lo:hi] = link
        lo = hi
    return FlowBatch(**{k: jnp.asarray(v) for k, v in fields.items()})


# ------------------------------------------------------- from_chains internals
def _unique_in_order(arr: np.ndarray) -> list:
    """Unique values in order of first appearance (chain_index grouping)."""
    seen, out = set(), []
    for v in np.asarray(arr).tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


#: per-residue dict keys sliced on chain_index grouping (others pass through)
_SLICEABLE_KEYS = ("pos", "mask", "where", "aa", "residue_index", "dssp")


def _slice_chain_dict(chain: dict, sel: np.ndarray) -> dict:
    """One chain_index group's sub-dict: per-residue arrays sliced by `sel`."""
    sub = {}
    n = len(sel)
    for key, value in chain.items():
        if key in ("chain_index", "num_residues", "length"):
            continue  # slices define the group's length
        if key in _SLICEABLE_KEYS:
            if isinstance(value, str):
                sub[key] = "".join(value[i] for i in np.flatnonzero(sel))
            else:
                arr = np.asarray(value)
                if arr.ndim >= 1 and arr.shape[0] == n:
                    sub[key] = arr[sel]
                else:
                    sub[key] = value  # scalar / slot-level array: pass through
        elif isinstance(value, dict) and key in _ATOM_LEVEL:
            sub[key] = _slice_chain_dict(value, sel)
        elif isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == n \
                and key in _ATOM_LEVEL:
            sub[key] = value[sel]  # bare per-residue atom-level values
        else:
            sub[key] = value
    return sub


_AA_BACKBONE = frozenset({"N", "CA", "C", "O", "OXT"})
_NA_BACKBONE = frozenset({"P", "OP1", "OP2", "O5'", "C5'", "C4'", "C3'", "O3'"})
_NA_SUGAR = frozenset({"C1'", "C2'", "C3'", "O2'", "O4'"})


def _sidechain_slots(res_name: str, kind: str) -> np.ndarray:
    """Atom37 slot indices of one residue's sidechain: AA = every non-backbone
    slot; NA = base slots only (sugar-phosphate backbone and sugar excluded).
    Unknown (ambiguous-letter) residues have no sidechain."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    from .spec import _AMBIGUOUS_NAMES

    if res_name in _AMBIGUOUS_NAMES:
        return np.zeros(0, np.int64)
    excluded = _AA_BACKBONE if kind == "aa" else (_NA_BACKBONE | _NA_SUGAR)
    return np.asarray(
        [s for s, name in enumerate(ENC.token_atoms[res_name])
         if str(name).strip() and str(name).strip() not in excluded], np.int64)


def _table_hbond_roles(res_name: str, atom_names) -> dict[str, int]:
    """C_HB roles (1=donor, 2=acceptor, 3=both) from the native atomworks
    (res_name, atom_name) tables only — no coordinate-based fallback."""
    from atomworks.ml.transforms import interactions as _ia

    roles: dict[str, int] = {}
    for an in atom_names:
        donor = ("*", an) in _ia._DONOR_ATOMS or (res_name, an) in _ia._DONOR_ATOMS
        acceptor = ("*", an) in _ia._ACCEPTOR_TABLE or (res_name, an) in _ia._ACCEPTOR_TABLE
        if donor and acceptor:
            roles[an] = 3
        elif donor:
            roles[an] = 1
        elif acceptor:
            roles[an] = 2
    return roles


# from_chains interaction-condition keys -> compiled channel names
_INTERACTION_KEYS = ("h_bond", "salt_bridge", "pi_stack", "pi_cation")
_INTERACTION_CHANNEL = {"h_bond": "hbond", "salt_bridge": "saltbridge",
                        "pi_stack": "pistack", "pi_cation": "pication"}

# deferred from_chains payloads: channel name -> RFD4Input adder
_DEFERRED_ADDERS = {"hotspot": "add_hotspot", "hbond": "add_hbond",
                    "pistack": "add_pistack", "pication": "add_pication",
                    "saltbridge": "add_saltbridge"}


def _resolve_seq_atom(selector: str, names3: list[str]) -> tuple[int, int]:
    """Resolve a hotspot-syntax atom selector ("2@NZ"; 1-based residue) to a
    0-based (residue, atom37 slot). The @ atom specification is required and
    must name exactly one atom of the residue."""
    import re
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    m = re.fullmatch(r"(\d+)@([\w,'\-\[\]]+)", selector.strip())
    if m is None:
        raise ValueError(
            f"from_chains: bond selector {selector!r} must be 'RES@ATOM' "
            "(1-based residue, required @ATOM)")
    res = int(m.group(1))
    if not 1 <= res <= len(names3):
        raise ValueError(
            f"from_chains: bond selector {selector!r} outside 1..{len(names3)}")
    an = m.group(2)
    slots = [s for s, nm in enumerate(ENC.token_atoms[names3[res - 1]])
             if str(nm).strip() == an]
    if len(slots) != 1:
        raise ValueError(
            f"from_chains: bond selector {selector!r} does not name exactly "
            f"one atom of {names3[res - 1]}")
    return res - 1, slots[0]


def _parse_residue_list(text: str, n: int) -> list[int]:
    """Parse a 1-indexed, comma-separated residue list ("1,5,6-10"; ranges
    inclusive) into 0-based indices, validated against 1..n."""
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        lo, sep, hi = part.partition("-")
        try:
            lo_i, hi_i = int(lo), int(hi if sep else lo)
        except ValueError:
            raise ValueError(
                f"from_chains: malformed residue list entry {part!r} "
                f"(expected an index or 'lo-hi')") from None
        if not 1 <= lo_i <= hi_i <= n:
            raise ValueError(
                f"from_chains: residue-list entry {part!r} outside 1..{n}")
        out.extend(range(lo_i - 1, hi_i))
    return out


def _add_dict_chain(spec, chain: dict, residue_index_by_chain: dict,
                    token_offset: int = 0) -> dict | None:
    """One polymer-chain dict → spec chain (+ its conditions).

    `token_offset` is the chain's first token in the compiled layout (the sum
    of previously added spec chains' lengths); string "hotspot" residue lists
    and "auto_condition" hbond conditioning are applied post-compile at that
    offset. Returns deferred actions for from_chains, or None."""
    from .spec import sequence_to_names

    # polymer kind: "aa"/"AA", "dna"/"DNA", "rna"/"RNA" (strict); "bp" implies
    # a nucleic-acid chain and conflicts with "dssp" (protein SS)
    _KINDS = {"aa": "aa", "AA": "aa", "dna": "dna", "DNA": "dna",
              "rna": "rna", "RNA": "rna"}
    if "dssp" in chain and "bp" in chain:
        raise ValueError(
            "from_chains: 'dssp' (protein C_SEC) and 'bp' (nucleic-acid "
            "C_NA_SS_STATE/C_BP) cannot condition the same chain")
    kind = "aa"
    if "kind" in chain:
        if chain["kind"] not in _KINDS:
            raise ValueError(
                f"from_chains: kind must be one of {sorted(_KINDS)}, got {chain['kind']!r}")
        kind = _KINDS[chain["kind"]]
    if "bp" in chain and kind not in ("rna", "dna"):
        raise ValueError(
            f"from_chains: 'bp' requires kind 'rna'/'RNA' or 'dna'/'DNA' "
            f"(the chain is {kind!r})")
    n = _chain_length(chain)
    ci = spec.add_chain(n, kind=kind)
    actions: dict = {}
    seq_str = chain.get("sequence")
    if seq_str is not None:
        if not isinstance(seq_str, str):
            raise ValueError(
                "from_chains: 'sequence' must be a one-letter-code string")
        if len(seq_str) != n:
            raise ValueError(
                f"from_chains: 'sequence' length {len(seq_str)} does not match "
                f"the chain length {n}")
        spec.add_sequence(seq_str, chain=ci)
    if isinstance(chain.get("hotspot"), str):
        # residue-list hotspot: C_HOT on the sidechain slots of the listed
        # (1-indexed) residues — requires a sequence for the slot layout
        if seq_str is None:
            raise ValueError(
                "from_chains: a string 'hotspot' residue list requires 'sequence'")

        names3 = sequence_to_names(seq_str, kind)
        where = np.zeros((n, 37), dtype=bool)
        for i0 in _parse_residue_list(chain["hotspot"], n):
            where[i0, _sidechain_slots(names3[i0], kind)] = True
        if not where.any():
            raise ValueError(
                "from_chains: none of the hotspot residues has sidechain slots "
                "to condition")
        actions["hotspot"] = (np.int32(1), where, token_offset)
    if chain.get("auto_condition") is True:
        # C_HB donor/acceptor roles on all sidechain donors/acceptors of the
        # added chain (table lookup only — no coordinates to probe)
        if seq_str is None:
            raise ValueError("from_chains: 'auto_condition' requires 'sequence'")
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

        hb = np.zeros((n, 37), dtype=np.int32)
        for i, rn in enumerate(sequence_to_names(seq_str, kind)):
            slots = _sidechain_slots(rn, kind)
            slot_names = [str(ENC.token_atoms[rn][s]).strip() for s in slots]
            for s, role in zip(slots, map(_table_hbond_roles(rn, slot_names).get,
                                          slot_names)):
                hb[i, s] = role or 0
        if (hb > 0).any():
            actions["hbond"] = (hb, hb > 0, token_offset)
    explicit_inter = {k: chain[k] for k in _INTERACTION_KEYS if k in chain}
    if explicit_inter and chain.get("auto_condition") is True:
        raise ValueError(
            "from_chains: 'auto_condition' cannot be combined with explicit "
            "interaction keys (h_bond/salt_bridge/pi_stack/pi_cation)")
    if explicit_inter:
        if seq_str is None:
            raise ValueError("from_chains: interaction keys require 'sequence'")
        from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

        names3 = sequence_to_names(seq_str, kind)
        for key, text in explicit_inter.items():
            if not isinstance(text, str):
                raise ValueError(
                    f"from_chains: {key!r} must be a selection string "
                    "('1,5,6-10' or '2@NZ,OE1')")
            ivals = np.zeros((n, 37),
                             dtype=bool if key == "pi_stack" else np.int32)
            iwhere = np.zeros((n, 37), dtype=bool)
            hit = False
            for i0, i_atoms in _parse_seq_selection(text, n):
                slots = _sidechain_slots(names3[i0], kind)
                slot_names = [str(ENC.token_atoms[names3[i0]][s]).strip()
                              for s in slots]
                values = _seq_interaction_values(key, names3[i0], set(slot_names))
                for s, an in zip(slots, slot_names):
                    if an in values and (i_atoms is None or an in i_atoms):
                        iwhere[i0, s] = True
                        ivals[i0, s] = values[an]
                        hit = True
            if hit:
                actions[_INTERACTION_CHANNEL[key]] = (ivals, iwhere, token_offset)
            else:
                _warn_empty_selection(key, text)
    chain_edges: list[tuple] = []
    explicit_bonds = chain.get("bonds")
    if explicit_bonds is not None:
        if seq_str is None:
            raise ValueError("from_chains: 'bonds' requires 'sequence'")
        if not isinstance(explicit_bonds, (list, tuple)):
            raise ValueError(
                "from_chains: 'bonds' must be a list of "
                "{'atoms': [selector, selector], 'order': 1|2|3} dicts")
        names3 = sequence_to_names(seq_str, kind)
        for entry in explicit_bonds:
            atoms = entry.get("atoms") if isinstance(entry, dict) else None
            order = entry.get("order", 1) if isinstance(entry, dict) else None
            if (not isinstance(atoms, (list, tuple)) or len(atoms) != 2
                    or int(order) not in (1, 2, 3)):
                raise ValueError(
                    "from_chains: each 'bonds' entry must be "
                    "{'atoms': [selector, selector], 'order': 1|2|3}")
            (t_i, s_i), (t_j, s_j) = (_resolve_seq_atom(a, names3)
                                      for a in atoms)
            chain_edges.append((token_offset + t_i, s_i, token_offset + t_j,
                                s_j, int(order), False))
    if chain.get("cyclic") is True:
        if seq_str is None:
            raise ValueError("from_chains: 'cyclic' requires 'sequence'")
        chain_edges.append((token_offset + n - 1, 2, token_offset, 0, 1, True))
    elif chain.get("cyclic") not in (False, None):
        raise ValueError("from_chains: 'cyclic' must be True or False")
    if chain_edges:
        actions["bond_edges"] = chain_edges
    if "bp" in chain:
        spec.add_bp(chain["bp"], chain=ci)
    if "dssp" in chain:
        spec.add_dssp(chain=ci, dssp=chain["dssp"])
    if "pos" in chain or "mask" in chain:
        if "pos" not in chain or "mask" not in chain:
            raise ValueError(
                "from_chains: 'pos' and 'mask' must be passed together "
                "(mask labels the present atoms of pos)")
        pos = np.asarray(chain["pos"], dtype=np.float32)
        mask = np.asarray(chain["mask"], dtype=bool)
        if pos.shape != (n, 37, 3) or mask.shape != (n, 37):
            raise ValueError(
                f"from_chains: pos must be (n, 37, 3) and mask (n, 37) for {n} "
                f"residues, got {pos.shape} and {mask.shape}")
        spec.add_pos(pos, mask, chain=ci, coord_group=int(chain.get("group", 0)))
    if "residue_index" in chain:
        ri = np.asarray(chain["residue_index"], dtype=np.int32)
        if ri.shape != (n,):
            raise ValueError(
                f"from_chains: residue_index must have length {n}, got {ri.shape}")
        residue_index_by_chain[ci] = ri
    for name in _ATOM_LEVEL:
        if name not in chain:
            continue
        if name == "hotspot" and isinstance(chain[name], str):
            continue  # residue-list hotspot: handled post-compile above
        item = chain[name]
        if isinstance(item, dict):
            values, aa, where = (item.get("values"), item.get("aa"),
                                 item.get("where"))
        else:
            values, aa, where = item, chain.get("aa"), chain.get("where")
        aa = np.zeros(n, dtype=np.int32) if aa is None else np.asarray(aa, np.int32)
        if aa.shape != (n,):
            raise ValueError(
                f"from_chains: {name} aa must have length {n}, got {aa.shape}")
        vals_arr = np.asarray(values)
        if name == "hotspot" and vals_arr.shape == (n,):
            # residue-level hotspot: per-residue values at mask-selected atoms
            if "mask" not in chain:
                raise ValueError(
                    "from_chains: residue-level 'hotspot' (shape (n,)) requires "
                    "'mask' (and 'pos')")
            where = (np.asarray(chain["mask"], dtype=bool)
                     & vals_arr.astype(bool)[:, None])
            values = vals_arr  # per-residue bool array (scalar would broadcast
            # to the whole chain in the spec adder)
        if where is None:
            if "mask" in chain:
                where = np.asarray(chain["mask"], dtype=bool)
            elif "where" in chain:
                where = chain["where"]
            else:
                raise ValueError(
                    f"from_chains: atom-level {name!r} needs 'where' (a (n, 37) "
                    "atom37 slot mask), or 'pos' + 'mask' to materialize slots")
        getattr(spec, f"add_{name}")(values, aa=aa, where=where, chain=ci)
    return actions or None


_LIGAND_DICT_KEYS = ("smiles", "ccd", "ligand_pos", "res_name",
                     "auto_condition", "buried") + _INTERACTION_KEYS


def _add_ligand(spec, chain: dict) -> None:
    """A smiles/ccd dict → spec ligand (atomized tokens, topology + bonds).

    "auto_condition": True autofeaturizes the interaction conditions
    (C_PISTACK on aromatic-ring atoms, C_HB donor/acceptor roles, C_SALT
    charge-sign roles — rfd4fx.ligand_conditions, native AnnotateInter-
    actions ligand semantics). "buried": True sets C_RAS 1 (buried) on every
    atom of the ligand chain. The explicit interaction keys ("h_bond",
    "salt_bridge", "pi_stack", "pi_cation") take 1-based atom-index selection
    strings ("1,3-5", in SMILES/CCD atom order) and condition only the
    eligible subset of those atoms; they cannot be combined with
    "auto_condition". Unknown keys raise (typo guard).
    """
    unknown = sorted(set(chain) - set(_LIGAND_DICT_KEYS))
    if unknown:
        raise ValueError(
            f"from_chains: unknown ligand key(s) {unknown} — supported: "
            f"{list(_LIGAND_DICT_KEYS)}")
    explicit_inter = {k: chain[k] for k in _INTERACTION_KEYS if k in chain}
    if explicit_inter and chain.get("auto_condition"):
        raise ValueError(
            "from_chains: 'auto_condition' cannot be combined with explicit "
            "interaction keys (h_bond/salt_bridge/pi_stack/pi_cation)")
    kwargs = {}
    if "ligand_pos" in chain:
        kwargs["ligand_pos"] = chain["ligand_pos"]
    if "res_name" in chain:
        kwargs["res_name"] = chain["res_name"]
    li = spec.add_ligand(smiles=chain.get("smiles"), ccd=chain.get("ccd"), **kwargs)
    lig = spec.ligands[li]
    if chain.get("auto_condition"):
        from .ligand_conditions import ligand_roles

        lig.interactions = ligand_roles(lig.elements, lig.charges, lig.bonds)
    elif explicit_inter:
        from .ligand_conditions import (SALT_POSITIVE, ligand_roles)

        k = int(lig.elements.shape[0])
        roles = ligand_roles(lig.elements, lig.charges, lig.bonds)
        interactions = {"hbond": np.zeros(k, np.int8),
                        "pistack": np.zeros(k, bool),
                        "salt": np.zeros(k, np.int8),
                        "pication": np.zeros(k, np.int8)}
        for key, text in explicit_inter.items():
            if not isinstance(text, str):
                raise ValueError(
                    f"from_chains: ligand {key!r} must be a 1-based atom-index "
                    "selection string (e.g. '1,3-5')")
            hit = False
            for i0 in _parse_residue_list(text, k):
                if key == "h_bond" and roles["hbond"][i0]:
                    interactions["hbond"][i0] = roles["hbond"][i0]
                elif key == "salt_bridge" and roles["salt"][i0]:
                    interactions["salt"][i0] = roles["salt"][i0]
                elif key == "pi_stack" and roles["pistack"][i0]:
                    interactions["pistack"][i0] = True
                elif key == "pi_cation":
                    if roles["pistack"][i0]:
                        interactions["pication"][i0] = 1      # AROMATIC
                    elif roles["salt"][i0] == SALT_POSITIVE:
                        interactions["pication"][i0] = 2      # CATION
                else:
                    continue
                hit = True
            if not hit:
                _warn_empty_selection(key, text)
        lig.interactions = interactions
    if chain.get("buried"):
        lig.buried = True


def _backbone_slots() -> np.ndarray:
    """atom37 slot indices of the backbone atoms N/CA/C/O (from the encoding)."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    ref = ENC.token_atoms.get("ALA")
    slots = [i for i, nm in enumerate(ref)
             if str(nm).strip().upper() in ("N", "CA", "C", "O")]
    return np.array(sorted(slots), dtype=np.int32)


def _parse_pdb_selection(entries, model):
    """Parse "select" entries into {chain: {resnum: atom-set | None}}.

    Entry grammar: CHAIN (whole chain) | CHAINRES | CHAINRES-RES (inclusive
    author-resnum range) | ...[@ATOM,ATOM,...]. Chain names are matched by
    longest prefix against the model's chain names (disambiguates numeric
    names). Atom selections are unioned per residue; a whole-residue
    selection (no @) wins over atom lists. Chain-level markers use the key
    "*".
    """
    import re

    chain_names = [c.name for c in model]
    sel: dict[str, dict] = {}
    for entry in entries:
        entry = str(entry).strip()
        if not entry:
            continue
        name = None
        rest = entry
        for cn in sorted(chain_names, key=len, reverse=True):
            if entry.startswith(cn):
                name, rest = cn, entry[len(cn):]
                break
        if name is None:
            raise ValueError(
                f"from_chains: select entry {entry!r} does not start with a "
                f"chain name of {chain_names}")
        if rest == "":
            sel.setdefault(name, {})["*"] = None          # whole chain
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?(?:@([\w,'\-]+))?", rest)
        if m is None:
            raise ValueError(
                f"from_chains: malformed select entry {entry!r} (expected "
                "CHAIN, CHAINRES, CHAINRES-RES, with optional @ATOM,ATOM)")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        atoms = (frozenset(a.strip() for a in m.group(3).split(",") if a.strip())
                 if m.group(3) else None)
        d = sel.setdefault(name, {})
        for resnum in range(lo, hi + 1):
            if resnum not in d:
                d[resnum] = atoms          # None = whole residue, or atom set
            elif d[resnum] is None or atoms is None:
                d[resnum] = None           # a whole-residue selection wins
            else:
                d[resnum] = frozenset(d[resnum] | atoms)
    return sel


def _perceive_bonds(coords: np.ndarray, elements: list[str]) -> list[tuple[int, int]]:
    """Covalent-radius distance cutoff connectivity over one residue's atoms."""
    import gemmi

    radii = [gemmi.Element(e).covalent_r for e in elements]
    bonds = []
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            d = float(np.linalg.norm(coords[i] - coords[j]))
            if d < radii[i] + radii[j] + 0.4:
                bonds.append((i, j))
    return bonds


def _ligand_chemistry_from_residue(res, ligands_map: dict | None):
    """(elements, charges, bonds) for one HETATM residue, in file atom order.

    Chemistry source: the "ligands" SMILES map (bond perception from the
    residue's coordinates + element-labeled graph isomorphism against the
    RDKit molecule, transferring bond orders/aromaticity/charges), else the
    CCD entry (name-mapped), else elements parsed from the atom names with
    zero charges and no bonds."""
    import gemmi

    names = [atom.name.strip() for atom in res]
    elements = [gemmi.Element(atom.element.name).name for atom in res]
    coords = np.array([[atom.pos.x, atom.pos.y, atom.pos.z] for atom in res],
                      np.float32)
    charges = [0] * len(names)
    bonds: list[tuple[int, int, int, int]] = []

    if ligands_map and res.name in ligands_map:
        from rdkit import Chem

        mol = Chem.MolFromSmiles(str(ligands_map[res.name]))
        if mol is not None and mol.GetNumAtoms() == len(names):
            import networkx as nx

            g_file = nx.Graph()
            for i, e in enumerate(elements):
                g_file.add_node(i, z=gemmi.Element(e).atomic_number)
            for i, j in _perceive_bonds(coords, elements):
                g_file.add_edge(i, j)
            g_smiles = nx.Graph()
            for atom in mol.GetAtoms():
                g_smiles.add_node(atom.GetIdx(), z=atom.GetAtomicNum())
            for bond in mol.GetBonds():
                g_smiles.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            gm = nx.algorithms.isomorphism.GraphMatcher(
                g_file, g_smiles, node_match=lambda a, b: a["z"] == b["z"])
            if gm.is_isomorphic():
                mapping = gm.mapping          # file atom idx -> smiles atom idx
                for i, j in g_file.edges:
                    b = mol.GetBondBetweenAtoms(mapping[i], mapping[j])
                    order = int(b.GetBondTypeAsDouble()) if b is not None else 1
                    bonds.append((i, j, min(order, 3),
                                  int(bool(b.GetIsAromatic()))
                                  if b is not None else 0))
                for fi, si in mapping.items():
                    charges[fi] = mol.GetAtomWithIdx(si).GetFormalCharge()
                return elements, charges, bonds
    try:
        from .ccd import get_ccd_entry

        rec = get_ccd_entry(res.name)
        ccd_names = [n.decode() if isinstance(n, bytes) else str(n)
                     for n in rec.atom_names]
        ccd_elems = [e.decode() if isinstance(e, bytes) else str(e)
                     for e in rec.elements]
        ccd_charges = [int(c) for c in rec.charges]
        name_idx = {n: i for i, n in enumerate(ccd_names)}
        if all(n in name_idx for n in names):
            elements = [ccd_elems[name_idx[n]] for n in names]
            charges = [ccd_charges[name_idx[n]] for n in names]
            for i, j, order, arom in zip(rec.bond_i, rec.bond_j,
                                         rec.bond_order, rec.bond_aromatic):
                ni, nj = ccd_names[int(i)], ccd_names[int(j)]
                if ni in set(names) and nj in set(names):
                    fi, fj = names.index(ni), names.index(nj)
                    bonds.append((min(fi, fj), max(fi, fj), int(order),
                                  int(bool(arom))))
            return elements, charges, bonds
    except Exception:
        pass
    return elements, charges, bonds


def _add_pdb_unindexed_motif(inp: "RFD4Input", chain: dict) -> "RFD4Input":
    """pdb/cif dict with unindexed_motif=True -> unindexed motif block.

    The selection (chains/residues/atoms) is parsed identically to the
    indexed route; vocabulary residues become residue tokens, non-vocabulary
    (HETATM) residues become atomized <A> runs with chemistry from the
    "ligands" SMILES map / CCD / element fallback. fixed selects absolute vs
    internal coordinate conditioning; ordered selects the index-group mode
    (contiguous runs vs per-token; ligand runs always isolated)."""
    import gemmi
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    path = chain.get("pdb", chain.get("cif"))
    model = gemmi.read_structure(str(path))[0]
    sel = (_parse_pdb_selection(chain["select"], model)
           if "select" in chain else None)
    if sel is not None:
        names = [c.name for c in model if c.name in sel]
    elif "chains" in chain:
        names = [str(s) for s in chain["chains"]]
    else:
        names = [c.name for c in model]
    fixed = bool(chain.get("fixed", True))
    ordered = bool(chain.get("ordered", False))
    drop_backbone = bool(chain.get("drop_backbone", False))
    sel = _apply_interface_selection(model, sel, names, chain)
    sel = _apply_neighbour_selection(model, sel, names, chain)
    hotspot_sel = _pdb_hotspot_sel(chain, model, sel, names)
    auto_condition = bool(chain.get("auto_condition"))
    inter_sel = _interaction_sel(chain, model, sel, names)
    if inter_sel and auto_condition:
        raise ValueError(
            "from_chains: 'auto_condition' cannot be combined with explicit "
            "interaction keys (h_bond/salt_bridge/pi_stack/pi_cation)")
    # "target_chain": pin the unindexed motif residues' chain_iid (int value
    # or A-Z character index) instead of minting a fresh chain; ligand runs
    # always keep their own fresh chains (native hard rule)
    target_iid = (_resolve_target_chain(chain["target_chain"])
                  if "target_chain" in chain else None)
    hotspot = chain.get("hotspot")

    pos_rows, mask_rows, aa_rows, ri_rows, ci_rows = [], [], [], [], []
    motif_res_unidx: list[tuple[str, object, np.ndarray]] = []
    hotspot_where: list[tuple[int, int]] = []
    hbond_hot: list[tuple[int, int, int]] = []
    inter_hot: dict[str, list[tuple[int, int, object]]] = {
        k: [] for k in (inter_sel or {})}
    lig_elements: list[int] = []
    lig_charges: list[int] = []
    lig_bonds: list[tuple[int, int, int, int]] = []
    next_iid = float(np.asarray(inp.chain_iid).max()) + 1.0
    iid_of: dict[str, float] = {}
    for name in names:
        pdb_chain = model[name]
        if pdb_chain is None:
            raise ValueError(f"from_chains: chain {name!r} not found in {path!r}")
        if name not in iid_of:
            iid_of[name] = next_iid if target_iid is None else target_iid
            next_iid += 1.0
        for res in pdb_chain:
            if res.name in ("HOH", "DOD", "WAT"):
                continue
            if sel is not None:
                chain_sel = sel.get(name, {})
                if "*" not in chain_sel and res.seqid.num not in chain_sel:
                    continue
                atom_sel = chain_sel.get("*", chain_sel.get(res.seqid.num, None))
            else:
                atom_sel = None
            if res.name in ENC.token_atoms:
                n37 = np.full((37, 3), np.nan, np.float32)
                mask37 = np.zeros(37, dtype=bool)
                seen = set()
                for atom in res:
                    an = atom.name.strip()
                    if an in seen:
                        continue
                    seen.add(an)
                    if atom_sel is not None and an not in atom_sel:
                        continue
                    if drop_backbone and an in ("N", "CA", "C", "O", "OXT"):
                        continue  # backbone dropped (sidechains-only motif)
                    slot = None
                    for s, slot_name in enumerate(ENC.token_atoms[res.name]):
                        if str(slot_name).strip() == an:
                            slot = s
                            break
                    if slot is None:
                        continue
                    mask37[slot] = True
                    n37[slot] = [atom.pos.x, atom.pos.y, atom.pos.z]
                if drop_backbone and not mask37.any():
                    continue  # nothing left after the backbone drop (e.g. GLY)
                if hotspot_sel is not None:
                    h_chain = hotspot_sel.get(name, {})
                    if "*" in h_chain or res.seqid.num in h_chain:
                        h_atoms = h_chain.get("*", h_chain.get(res.seqid.num))
                        for s in range(37):
                            an = str(ENC.token_atoms[res.name][s]).strip()
                            if (mask37[s] and an
                                    and (h_atoms is None or an in h_atoms)):
                                hotspot_where.append((len(pos_rows), s))
                        if auto_condition:
                            roles = _residue_hbond_roles(
                                res, {str(ENC.token_atoms[res.name][s]).strip()
                                      for s in range(37) if mask37[s]
                                      and str(ENC.token_atoms[res.name][s]).strip()
                                      and (h_atoms is None
                                           or str(ENC.token_atoms[res.name][s]).strip()
                                           in h_atoms)})
                        if auto_condition:
                            for an, role in roles.items():
                                for s, slot_name in enumerate(
                                        ENC.token_atoms[res.name]):
                                    if (str(slot_name).strip() == an
                                            and mask37[s]):
                                        hbond_hot.append((len(pos_rows), s, role))
                for key, res_map in (inter_sel or {}).items():
                    ic = res_map.get(name, {})
                    if "*" not in ic and res.seqid.num not in ic:
                        continue
                    i_atoms = ic.get("*", ic.get(res.seqid.num))
                    ivalues = _interaction_residue_values(
                        key, res, i_atoms, chain.get("ligands"))
                    for s, an in enumerate(
                            str(nm).strip()
                            for nm in ENC.token_atoms.get(res.name, ("",) * 37)):
                        if mask37[s] and an and an in ivalues and (
                                i_atoms is None or an in i_atoms):
                            inter_hot[key].append((len(pos_rows), s, ivalues[an]))
                motif_res_unidx.append((name, res, mask37))
                pos_rows.append(n37)
                mask_rows.append(mask37)
                aa_rows.append(ENC.token_to_idx.get(res.name, 0))
                ri_rows.append(res.seqid.num)
                ci_rows.append(iid_of[name])
            else:
                # HETATM ligand: one atomized <A> token per heavy atom
                elements, charges, bonds = _ligand_chemistry_from_residue(
                    res, chain.get("ligands"))
                ivalues_all = {key: _interaction_residue_values(
                    key, res, atom_sel, chain.get("ligands"))
                    for key in (inter_sel or {})}
                file_to_lig: dict[int, int] = {}
                h_atoms = None
                if hotspot_sel is not None:
                    h_chain = hotspot_sel.get(name, {})
                    if "*" in h_chain or res.seqid.num in h_chain:
                        h_atoms = h_chain.get("*", h_chain.get(res.seqid.num))
                roles = (_residue_hbond_roles(res, set(names_ := [a.name.strip()
                          for a in res])) if auto_condition and h_atoms
                         is not None else {})
                for a, atom in enumerate(res):
                    an = atom.name.strip()
                    if atom_sel is not None and an not in atom_sel:
                        continue
                    n37 = np.full((37, 3), np.nan, np.float32)
                    n37[1] = [atom.pos.x, atom.pos.y, atom.pos.z]
                    pos_rows.append(n37)
                    mask_rows.append(np.eye(37, dtype=bool)[1])
                    aa_rows.append(ENC.token_to_idx["<A>"])
                    ri_rows.append(res.seqid.num)
                    ci_rows.append(next_iid)   # ligand residue: own chain
                    file_to_lig[a] = len(lig_elements)
                    lig_elements.append(elements[a])
                    lig_charges.append(charges[a])
                    if h_atoms is not None and (h_atoms is None or an in h_atoms):
                        hotspot_where.append((len(pos_rows) - 1, 1))
                        if an in roles:
                            hbond_hot.append((len(pos_rows) - 1, 1, roles[an]))
                    for key in (inter_sel or {}):
                        v = ivalues_all[key].get(an)
                        if v is not None and v:
                            inter_hot[key].append((len(pos_rows) - 1, 1, v))
                next_iid += 1.0
                for i, j, order, arom in bonds:
                    if i in file_to_lig and j in file_to_lig:
                        lig_bonds.append((file_to_lig[i], file_to_lig[j],
                                          order, arom))
    if not pos_rows:
        raise ValueError(
            f"from_chains: the pdb/cif selection of {path!r} selected no tokens")
    pos = np.asarray(pos_rows, np.float32)
    mask = np.asarray(mask_rows, bool)
    aa = np.asarray(aa_rows, np.int32)
    ri = np.asarray(ri_rows, np.int32)
    ci = np.asarray(ci_rows, np.float32)
    bonds = (np.asarray(lig_bonds, np.int32) if lig_bonds else None)
    from .spec import _atomic_number_from_symbol

    element = (np.asarray([_atomic_number_from_symbol(e) for e in lig_elements],
                          np.int32) if lig_elements else None)
    charge = (np.asarray(lig_charges, np.int32) if lig_charges else None)
    out = inp.add_unindexed_motif(
        pos, aa, mask, residue_index=ri, chain_index=ci,
        group=(int(chain.get("group", 0)) if not fixed else 0),
        fixed=fixed, ordered=ordered, element=element, charge=charge,
        bonds=bonds, target_chain=chain.get("target_chain"))
    block_start = int(np.asarray(inp.padding_mask).shape[1])
    if hotspot_where:
        rows, slots = zip(*hotspot_where)
        where_block = np.zeros((len(pos), 37), bool)
        where_block[np.asarray(rows), np.asarray(slots)] = True
        out = out.add_hotspot(np.int32(1), aa=None, where=where_block,
                              start=block_start)
    if hbond_hot:
        hb = np.zeros((len(pos), 37), np.int32)
        for r, s_, role in hbond_hot:
            hb[r, s_] = role
        out = out.add_hbond(hb, aa=None, where=hb > 0, start=block_start)
    for key, hits in inter_hot.items():
        if not hits:
            for entry in chain[key]:
                _warn_empty_selection(key, entry)
            continue
        ivals = np.zeros((len(pos), 37),
                         dtype=bool if key == "pi_stack" else np.int32)
        iwhere = np.zeros((len(pos), 37), bool)
        for r, s_, v in hits:
            ivals[r, s_] = v
            iwhere[r, s_] = True
        out = getattr(out, _DEFERRED_ADDERS[_INTERACTION_CHANNEL[key]])(
            ivals, aa=None, where=iwhere, start=block_start)
    bond_edges = _pdb_bond_edges(chain, model, motif_res_unidx, block_start)
    if bond_edges:
        out = _apply_bond_edges(out, bond_edges)
    return out


_TARGET_CHAIN_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _resolve_target_chain(value) -> float:
    """from_chains ``"target_chain"``: an int sets chain_iid directly; a
    single character resolves as
    ``"ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(char)``. Anything else raises."""
    if isinstance(value, bool):
        raise ValueError(
            f"from_chains: target_chain must be an int or a single A-Z "
            f"character, got {value!r}")
    if isinstance(value, (int, np.integer)):
        return float(value)
    if isinstance(value, str) and len(value) == 1:
        if value not in _TARGET_CHAIN_ALPHABET:
            raise ValueError(
                f"from_chains: target_chain {value!r} is not in "
                f"{_TARGET_CHAIN_ALPHABET!r}")
        return float(_TARGET_CHAIN_ALPHABET.index(value))
    raise ValueError(
        f"from_chains: target_chain must be an int or a single A-Z "
        f"character, got {value!r}")


def _apply_interface_selection(model, sel, names, chain: dict):
    """pdb/cif dict "interface_to" support: restrict the dict's selection to
    residues with at least one heavy atom within ``cutoff`` (default 4.5 A)
    of any heavy atom of the ``interface_to`` chain (solvent skipped).

    Chains other than ``interface_to`` keep only their interface residues;
    a ``"*"`` residue entry expands to the interface residues (with the
    entry's atom subset); the ``interface_to`` chain itself is left
    unrestricted. Without a parsed selection the interface residues BECOME
    the selection. Returns the updated selection (unchanged when
    ``interface_to`` is absent)."""
    interface_to = chain.get("interface_to")
    if interface_to is None:
        return sel
    interface_to = str(interface_to)
    cutoff = float(chain.get("cutoff", 4.5))
    acc_chain = model[interface_to]
    if acc_chain is None:
        raise ValueError(
            f"from_chains: interface_to chain {interface_to!r} not found")
    acc = np.asarray(
        [[a.pos.x, a.pos.y, a.pos.z] for res in acc_chain
         if res.name not in ("HOH", "DOD", "WAT")
         for a in res if a.element.name.upper() not in ("H", "D")], np.float32)
    if not len(acc):
        raise ValueError(
            f"from_chains: interface_to chain {interface_to!r} has no heavy atoms")
    c2 = cutoff * cutoff
    iface: dict[str, set[int]] = {}
    for name in names:
        if name == interface_to:
            continue
        pdb_chain = model[name]
        if pdb_chain is None:
            continue
        atom_rows: list[list[float]] = []
        atom_seqids: list[int] = []
        for res in pdb_chain:
            if res.name in ("HOH", "DOD", "WAT"):
                continue
            seq_num = res.seqid.num
            for a in res:
                if a.element.name.upper() in ("H", "D"):
                    continue
                atom_rows.append([a.pos.x, a.pos.y, a.pos.z])
                atom_seqids.append(seq_num)
        hits: set[int] = set()
        if atom_rows:
            pts = np.asarray(atom_rows, np.float32)
            seq_arr = np.asarray(atom_seqids, np.int64)
            for lo in range(0, len(pts), 4096):
                block = pts[lo:lo + 4096]
                d2 = ((block[:, None, :] - acc[None, :, :]) ** 2).sum(-1)
                hits.update(int(x) for x in seq_arr[lo:lo + 4096][(d2 < c2).any(1)])
        iface[name] = hits

    def _restrict(nm, res_map):
        out = {}
        for rn, atoms in res_map.items():
            if rn == "*":
                for hit in sorted(iface.get(nm, ())):
                    out[hit] = atoms
            elif rn in iface.get(nm, ()):
                out[rn] = atoms
        return out

    if sel is None:
        out = {}
        for nm in names:
            if nm == interface_to:
                # the interface_to chain itself stays fully selected
                out[nm] = {res.seqid.num: None for res in model[nm]
                           if res.name not in ("HOH", "DOD", "WAT")}
            elif iface.get(nm):
                out[nm] = {rn: None for rn in sorted(iface[nm])}
        return out
    return {nm: (res_map if nm == interface_to else _restrict(nm, res_map))
            for nm, res_map in sel.items()}


def is_surface(position, occluders, *, radius=4.0, n_points=64, threshold=0.2):
    """Vectorized Shrake-Rupley surface test on CB positions (numpy only).

    Places ``n_points`` test points (golden-spiral lattice) on a sphere of
    ``radius`` around every position in ``position`` (CB coordinates); a
    point is exposed when no position in ``occluders`` (backbone heavy atoms)
    lies within ``radius`` of it. A position is surface when the exposed
    fraction of its test points is >= ``threshold``.

    Occlusion test: the point ``c + r*u`` is covered by an occluder ``o``
    iff ``(o - c)/|o - c| . u > |o - c| / (2r)``, so only dot products per
    (CB, occluder, test point) triple are needed; CBs are processed in
    chunks to bound memory.
    """
    position = np.asarray(position, np.float64).reshape(-1, 3)
    occluders = np.asarray(occluders, np.float64).reshape(-1, 3)
    if not len(position):
        return np.zeros(0, bool)
    if not len(occluders):
        return np.ones(len(position), bool)
    # golden-spiral unit-sphere test points
    k = (np.arange(n_points) + 0.5) / n_points
    unit = np.stack([
        np.sqrt(1 - k * k) * np.cos(2 * np.pi * k * (1 + 5 ** 0.5) / 2),
        np.sqrt(1 - k * k) * np.sin(2 * np.pi * k * (1 + 5 ** 0.5) / 2),
        1 - 2 * k,
    ], axis=1)                                                   # (P, 3)
    exposed = np.empty(len(position), np.float64)
    # occluders farther than 2*radius can never cover any test point
    chunk = max(1, int(4e6 / max(1, n_points)))
    for lo in range(0, len(position), chunk):
        c = position[lo:lo + chunk]                              # (C, 3)
        d = occluders[None, :, :] - c[:, None, :]                # (C, M, 3)
        s = (d * d).sum(-1)                                      # (C, M)
        keep = s.min(0) <= (2 * radius) ** 2
        if not keep.any():
            exposed[lo:lo + chunk] = 1.0
            continue
        d, s = d[:, keep], s[:, keep]                            # (C, M', 3)
        dot = np.einsum("cmd,pd->cmp", d, unit)                  # (C, M', P)
        covered = dot > s[:, :, None] / (2 * radius)
        exposed[lo:lo + chunk] = (~covered).all(1).mean(1)
    return exposed >= threshold


def _apply_neighbour_selection(model, sel, names, chain: dict):
    """pdb/cif dict "neighbours" support: extend the parsed selection with
    neighbouring residues — residues of the selection's chains with at least
    one sidechain heavy atom (not N/CA/C/O/OXT, not H/D) within ``cutoff``
    of any heavy atom of the selected residues. Neighbours join as whole
    residues; the base selection is always kept. With "surface"=True only
    neighbours whose CB passes :func:`is_surface` (Shrake-Rupley against
    backbone-atom occluders) are added. Requires a parsed selection."""
    if not chain.get("neighbours"):
        return sel
    if sel is None:
        raise ValueError(
            "from_chains: neighbours=True requires a 'select' entry")
    cutoff = float(chain.get("cutoff", 4.5))
    backbone = {"N", "CA", "C", "O", "OXT"}

    # anchor: all heavy atoms of the selected (base) residues
    base: dict[str, set[int]] = {}
    anchor: list[np.ndarray] = []
    for name, res_map in sel.items():
        pdb_chain = model[name]
        if pdb_chain is None:
            continue
        nums: set[int] = set()
        for rn in res_map:
            if rn == "*":
                nums.update(res.seqid.num for res in pdb_chain
                            if res.name not in ("HOH", "DOD", "WAT"))
            else:
                nums.add(rn)
        base[name] = nums
        anchor.extend(
            np.asarray([[a.pos.x, a.pos.y, a.pos.z] for a in res
                        if a.element.name.upper() not in ("H", "D")],
                       np.float32)
            for res in pdb_chain
            if res.seqid.num in nums
            and res.name not in ("HOH", "DOD", "WAT"))
    if not anchor:
        return sel
    anchors = np.concatenate(anchor, 0)

    # candidates: sidechain heavy atoms of the selection's chains' residues
    cand_rows: list[np.ndarray] = []
    cand_ids: list[tuple[str, int]] = []
    row_owner: list[int] = []
    for name in names:
        pdb_chain = model[name]
        if pdb_chain is None:
            continue
        for res in pdb_chain:
            if (res.name in ("HOH", "DOD", "WAT")
                    or res.seqid.num in base.get(name, ())):
                continue
            sc = np.asarray(
                [[a.pos.x, a.pos.y, a.pos.z] for a in res
                 if a.name.strip() not in backbone
                 and a.element.name.upper() not in ("H", "D")], np.float32)
            if len(sc):
                row_owner.extend([len(cand_ids)] * len(sc))
                cand_rows.append(sc)
                cand_ids.append((name, res.seqid.num))
    if not cand_rows:
        return sel
    cand = np.concatenate(cand_rows, 0)
    owner = np.asarray(row_owner, np.int64)

    # a candidate is a neighbour iff any of its sidechain heavy atoms lies
    # within cutoff of any anchor heavy atom
    near = np.zeros(len(cand_ids), bool)
    c2 = cutoff * cutoff
    for lo in range(0, len(cand), 4096):
        block = cand[lo:lo + 4096]                               # (B, 3)
        d2 = ((block[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        hits = owner[lo:lo + 4096][(d2 < c2).any(1)]
        near[hits] = True
    neighbours = [cid for cid, hit in zip(cand_ids, near) if hit]
    if chain.get("surface") and neighbours:
        cb_pos, cb_ids, occ = [], [], []
        for c in model:
            for res in c:
                if res.name in ("HOH", "DOD", "WAT"):
                    continue
                for a in res:
                    an = a.name.strip()
                    if a.element.name.upper() in ("H", "D"):
                        continue
                    p = [a.pos.x, a.pos.y, a.pos.z]
                    if an == "CB":
                        cb_pos.append(p)
                        cb_ids.append((c.name, res.seqid.num))
                    elif an in backbone:
                        occ.append(p)
        surf = is_surface(cb_pos, occ)
        keep = {cb_ids[i] for i in np.flatnonzero(surf)}
        neighbours = [cid for cid in neighbours if cid in keep]

    out = {nm: dict(res_map) for nm, res_map in sel.items()}
    for nm, rn in neighbours:
        out.setdefault(nm, {}).setdefault(rn, None)
    return out


def _residue_hbond_roles(res, atom_names) -> dict[str, int]:
    """C_HB values (1=donor, 2=acceptor, 3=both) for one residue's selected
    atoms.

    Standard residues (AA + NA): the native (res_name, atom_name) tables
    (atomworks interactions._DONOR_ATOMS / _ACCEPTOR_TABLE, NA-extended).
    Non-standard residues: element-rule fallback (acceptor = N/O/S; donor =
    N/O with an attached H within 1.3 A in the file)."""
    from atomworks.ml.transforms import interactions as _ia

    res_name = res.name.strip().upper()
    known = {rn for rn, _ in _ia._DONOR_ATOMS | _ia._ACCEPTOR_TABLE} - {"*"}
    atoms = {atom.name.strip(): atom for atom in res
             if atom.name.strip() in atom_names}
    h_pos = [np.array([a.pos.x, a.pos.y, a.pos.z]) for a in res
             if a.element.name.upper() == "H"]
    roles: dict[str, int] = {}
    for an, atom in atoms.items():
        donor = ("*", an) in _ia._DONOR_ATOMS or (res_name, an) in _ia._DONOR_ATOMS
        acceptor = ("*", an) in _ia._ACCEPTOR_TABLE or (res_name, an) in _ia._ACCEPTOR_TABLE
        if not donor and not acceptor and res_name not in known:
            z = atom.element.name.upper()
            if z in ("N", "O", "S"):
                acceptor = True
            if z in ("N", "O") and h_pos:
                pos = np.array([atom.pos.x, atom.pos.y, atom.pos.z])
                if any(float(np.linalg.norm(pos - h)) < 1.3 for h in h_pos):
                    donor = True
        if donor and acceptor:
            roles[an] = 3
        elif donor:
            roles[an] = 1
        elif acceptor:
            roles[an] = 2
    return roles


def _positive_charged(res_name: str, atom_name: str) -> bool:
    """Salt-bridge cation eligibility (atomworks charged-atom tables)."""
    from atomworks.ml.transforms import interactions as _ia

    return (res_name, atom_name) in _ia._POSITIVE_CHARGED_AA


def _negative_charged(res_name: str, atom_name: str) -> bool:
    """Salt-bridge anion eligibility (atomworks charged-atom tables)."""
    from atomworks.ml.transforms import interactions as _ia

    return (res_name, atom_name) in _ia._NEGATIVE_CHARGED


def _aromatic_ring_atoms(res_name: str) -> frozenset[str]:
    """Aromatic-ring atom names of one standard residue (empty for others)."""
    from atomworks.ml.transforms import interactions as _ia

    return frozenset(a for ring in _ia._AROMATIC_RINGS.get(res_name, ())
                     for a in ring)


def _warn_empty_selection(key: str, entry: str) -> None:
    print(f"warning: {key} selection {entry!r} names no eligible atoms — skipped",
          flush=True)


def _interaction_payloads(inter_sel, motif_res, chain, motif_start: int) -> dict:
    """Deferred payloads for the interaction-condition keys over a motif.

    `motif_res` lists the motif residues in row order as (chain, res, mask37)
    — conditioning lands on selected ∩ eligible ∩ materialized slots. Returns
    {compiled channel name: (vals (rows, 37), where (rows, 37), motif_start)};
    selections with no eligible atoms are warned and skipped."""
    if not inter_sel:
        return {}
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    payloads: dict[str, tuple] = {}
    for key, res_map in inter_sel.items():
        entries = [str(e) for e in chain[key]]
        vals = np.zeros((len(motif_res), 37),
                        dtype=bool if key == "pi_stack" else np.int32)
        where = np.zeros((len(motif_res), 37), dtype=bool)
        hit = False
        for row, (name, res, mask37) in enumerate(motif_res):
            res_map_c = res_map.get(name, {})
            if "*" not in res_map_c and res.seqid.num not in res_map_c:
                continue
            atom_sel = res_map_c.get("*", res_map_c.get(res.seqid.num))
            values = _interaction_residue_values(key, res, atom_sel,
                                                 chain.get("ligands"))
            for s, an in enumerate(
                    str(nm).strip()
                    for nm in ENC.token_atoms.get(res.name, ("",) * 37)):
                if mask37[s] and an and an in values and (
                        atom_sel is None or an in atom_sel):
                    where[row, s] = True
                    vals[row, s] = values[an]
                    hit = True
        if hit:
            payloads[_INTERACTION_CHANNEL[key]] = (vals, where, motif_start)
        else:
            for entry in entries:
                _warn_empty_selection(key, entry)
    return payloads


def _seq_interaction_values(key: str, res_name: str,
                            atom_names: set[str]) -> dict[str, object]:
    """Per-atom interaction values for one sequence-only residue's selected
    atoms, from the atomworks tables (table-only: no coordinates)."""
    if key == "h_bond":
        return dict(_table_hbond_roles(res_name, atom_names))
    if key == "salt_bridge":
        out = {an: 1 for an in atom_names if _positive_charged(res_name, an)}
        out.update({an: 2 for an in atom_names if _negative_charged(res_name, an)})
        return out
    if key == "pi_stack":
        rings = _aromatic_ring_atoms(res_name)
        return {an: True for an in atom_names if an in rings}
    out: dict[str, object] = {}
    rings = _aromatic_ring_atoms(res_name)
    for an in atom_names:                       # pi_cation
        if an in rings:
            out[an] = 1                          # PiCationRole.AROMATIC
        elif _positive_charged(res_name, an):
            out[an] = 2                          # PiCationRole.CATION
    return out


def _interaction_residue_values(key: str, res, atom_sel, ligands) -> dict[str, object]:
    """Eligible selected atoms of one structured residue -> per-atom values.

    `atom_sel` None = whole residue. Standard vocabulary residues go through
    the atomworks tables (h_bond via _residue_hbond_roles, which keeps the
    coordinate-based H-donor fallback); HETATM residues through the RDKit
    ligand heuristics (ligand_conditions.ligand_roles)."""
    res_name = res.name.strip().upper()
    names = [a.name.strip() for a in res
             if atom_sel is None or a.name.strip() in atom_sel]
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    if res.name in ENC.token_atoms:
        return _residue_hbond_roles(res, set(names)) if key == "h_bond" \
            else _seq_interaction_values(key, res_name, set(names))
    # HETATM: native ligand heuristics (RDKit), in file atom order
    from .ligand_conditions import ligand_roles

    elements, charges, bonds = _ligand_chemistry_from_residue(res, ligands)
    from .spec import _atomic_number_from_symbol

    elements = np.asarray([_atomic_number_from_symbol(e) for e in elements],
                          np.int32)
    roles = ligand_roles(elements, charges, bonds)
    names = [a.name.strip() for a in res]
    if key == "h_bond":
        return {an: int(roles["hbond"][i]) for i, an in enumerate(names)
                if roles["hbond"][i]}
    if key == "salt_bridge":
        return {an: int(roles["salt"][i]) for i, an in enumerate(names)
                if roles["salt"][i]}
    if key == "pi_stack":
        return {an: True for i, an in enumerate(names) if roles["pistack"][i]}
    out: dict[str, object] = {}
    for i, an in enumerate(names):               # pi_cation
        if roles["pistack"][i]:
            out[an] = 1
        elif roles["salt"][i] == 1:              # SALT_POSITIVE
            out[an] = 2
    return out


def _parse_seq_selection(text: str, n: int) -> list[tuple[int, frozenset[str] | None]]:
    """Parse a chain-prefix-free selection ("1,5,6-10", "2@NZ,OE1") into
    0-based (residue, atom-names-or-None) pairs — the _parse_pdb_selection
    grammar without the chain name (1-indexed, ranges inclusive; a
    whole-residue entry wins over atom lists)."""
    import re

    out: dict[int, frozenset[str] | None] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        m = re.fullmatch(r"(\d+)(?:-(\d+))?(?:@([\w,'\-]+))?", part)
        if m is None:
            raise ValueError(
                f"from_chains: malformed selection entry {part!r} (expected "
                "RES, RES-RES, with optional @ATOM,ATOM)")
        lo, hi = int(m.group(1)), int(m.group(2) or m.group(1))
        if not 1 <= lo <= hi <= n:
            raise ValueError(
                f"from_chains: selection entry {part!r} outside 1..{n}")
        atoms = (frozenset(a.strip() for a in m.group(3).split(",") if a.strip())
                 if m.group(3) else None)
        for rn in range(lo - 1, hi):
            prev = out.get(rn)
            if prev is None:
                out[rn] = atoms
            elif atoms is not None:
                out[rn] = prev | atoms
    return sorted(out.items())


def _resolve_pdb_bond_atom(entry: str, model, row_of, motif_res) -> tuple[int, int]:
    """Resolve one pdb-grammar bond selector ("A48@NZ" — exactly one residue,
    exactly one atom) to a (motif row, atom37 slot)."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    sel = _parse_pdb_selection([str(entry)], model)
    if len(sel) != 1:
        raise ValueError(
            f"from_chains: bonds selector {entry!r} must name residues of "
            "exactly one chain")
    nm, res_map = next(iter(sel.items()))
    if len(res_map) != 1 or "*" in res_map:
        raise ValueError(
            f"from_chains: bonds selector {entry!r} must name exactly one "
            "residue (RES@ATOM)")
    rn = next(iter(res_map))
    atoms = res_map[rn]
    if atoms is None or len(atoms) != 1:
        raise ValueError(
            f"from_chains: bonds selector {entry!r} must name exactly one "
            "atom (RES@ATOM)")
    key = (nm, int(rn))
    if key not in row_of:
        raise ValueError(
            f"from_chains: bonds selector {entry!r} names a residue outside "
            "the loaded motif")
    row = row_of[key]
    an = next(iter(atoms))
    res = motif_res[row][1]
    slots = [s for s, nm2 in enumerate(ENC.token_atoms.get(res.name, ("",) * 37))
             if str(nm2).strip() == an]
    if len(slots) != 1:
        raise ValueError(
            f"from_chains: bonds selector {entry!r} does not name exactly one "
            f"atom of {res.name}")
    return row, slots[0]


def _pdb_bond_edges(chain: dict, model, motif_res, token_start: int) -> list[tuple]:
    """Resolve the 'bonds'/'cyclic' keys of a pdb/cif dict into absolute-token
    bond edges. 'bonds' selectors use the chain grammar with a required
    single @ATOM (e.g. "A48@NZ"); 'cyclic': True closes each selected chain
    head-to-tail (last C -> first N of the chain's motif rows)."""
    if not (chain.get("bonds") or chain.get("cyclic")):
        return []
    row_of: dict[tuple[str, int], int] = {}
    rows_by_chain: dict[str, list[int]] = {}
    for row, (name, res, _mask) in enumerate(motif_res):
        row_of[(name, int(res.seqid.num))] = row
        rows_by_chain.setdefault(name, []).append(row)
    edges: list[tuple] = []
    entries = chain.get("bonds") or ()
    if not isinstance(entries, (list, tuple)):
        raise ValueError(
            "from_chains: 'bonds' must be a list of "
            "{'atoms': [selector, selector], 'order': 1|2|3} dicts")
    for entry in entries:
        atoms = entry.get("atoms") if isinstance(entry, dict) else None
        order = entry.get("order", 1) if isinstance(entry, dict) else None
        if (not isinstance(atoms, (list, tuple)) or len(atoms) != 2
                or int(order) not in (1, 2, 3)):
            raise ValueError(
                "from_chains: each 'bonds' entry must be "
                "{'atoms': [selector, selector], 'order': 1|2|3}")
        (row_i, s_i), (row_j, s_j) = (
            _resolve_pdb_bond_atom(a, model, row_of, motif_res) for a in atoms)
        edges.append((token_start + row_i, s_i, token_start + row_j, s_j,
                      int(order), False))
    if chain.get("cyclic") is True:
        for name, rows in rows_by_chain.items():
            if len(rows) >= 2:
                edges.append((token_start + rows[-1], 2,
                              token_start + rows[0], 0, 1, True))
    elif chain.get("cyclic") not in (False, None):
        raise ValueError("from_chains: 'cyclic' must be True or False")
    return edges


def _interaction_sel(chain: dict, model, sel, names):
    """Parse the optional interaction-condition keys ("h_bond", "salt_bridge",
    "pi_stack", "pi_cation" — lists of selection strings) of a pdb/cif dict
    into {key: {chain: {resnum: atom-set | None}}}, validating against the
    loaded motif (out-of-motif chains/residues raise). None when no key is
    present."""
    parsed: dict[str, dict] = {}
    for key in _INTERACTION_KEYS:
        entries = chain.get(key)
        if entries is None:
            continue
        if not isinstance(entries, (list, tuple)):
            raise ValueError(
                f"from_chains: {key!r} must be a list of selection strings")
        parsed[key] = _parse_pdb_selection([str(e) for e in entries], model)
    if not parsed:
        return None
    loaded = set(names)
    motif: dict[str, set[int]] = {}
    for nm, res_map in (sel or {}).items():
        nums = {rn for rn in res_map if rn != "*"}
        if "*" in res_map:
            nums.update(res.seqid.num for res in model[nm]
                        if res.name not in ("HOH", "DOD", "WAT"))
        motif[nm] = nums
    for key, res_map in parsed.items():
        for nm, nums in res_map.items():
            if nm not in loaded:
                raise ValueError(
                    f"from_chains: {key} selection names chain {nm!r}, which "
                    f"is not part of the loaded motif")
            if "*" not in nums:
                bad = set(nums) - motif.get(nm, set())
                if bad:
                    raise ValueError(
                        f"from_chains: {key} selection names residues {sorted(bad)} "
                        f"of chain {nm!r}, which are not part of the loaded motif")
    return parsed


def _pdb_hotspot_sel(chain: dict, model, sel, names):
    """Parse a str "hotspot" into a selection; validate against the loaded
    motif (out-of-motif chains/residues raise). None when "hotspot" is
    absent or not a str."""
    hotspot = chain.get("hotspot")
    if not isinstance(hotspot, str):
        return None
    hotspot_sel = _parse_pdb_selection(hotspot.split(";"), model)
    loaded_chains = set(names)
    for h_chain, h_res in hotspot_sel.items():
        if h_chain not in loaded_chains:
            raise ValueError(
                f"from_chains: hotspot chain {h_chain!r} is not part of the "
                f"loaded motif {sorted(loaded_chains)}")
        if sel is not None and "*" not in sel.get(h_chain, {}):
            for rn in h_res:
                if rn != "*" and rn not in sel.get(h_chain, {}):
                    raise ValueError(
                        f"from_chains: hotspot residue {h_chain}{rn} is not "
                        "part of the loaded motif selection")
    return hotspot_sel


def _add_pdb_motifs(spec, chain: dict, residue_index_by_chain: dict) -> None:
    """A pdb/cif dict -> indexed motif conditioning on fresh chains.

    gemmi.read_structure (first model; `pdb` or `cif` key — mmCIF is read
    directly); the loaded chains/residues/atoms come from "chains" (list of
    chain names) or "select" (CHAIN / CHAINRES / CHAINRES-RES entries with
    optional @ATOM,ATOM lists; overrides "chains"); per PDB chain: a fresh
    spec chain whose residues are conditioned through add_indexed_motif.
    `fixed` (default True) selects absolute coordinate conditioning (C_CRD
    group `group`, default 0) vs internal coordinates (one fresh positive
    coordinate group for the whole selection). Author residue ids drive the
    |Δres_id| path/linkage post-processing. `drop_aa=True` restricts the
    mask to N/CA/C/O and drops the identity conditioning (C_SEQ stays <M> —
    backbone-only motif).
    """
    import gemmi
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    path = chain.get("pdb", chain.get("cif"))
    structure = gemmi.read_structure(str(path))
    model = structure[0]  # first model
    sel = (_parse_pdb_selection(chain["select"], model)
           if "select" in chain else None)
    if sel is not None:
        names = [c.name for c in model if c.name in sel]
    elif "chains" in chain:
        names = [str(s) for s in chain["chains"]]
    else:
        names = [c.name for c in model]
    fixed = bool(chain.get("fixed", True))
    group = int(chain.get("group", 0))
    if not fixed and group == 0:
        group = 1  # 0 is reserved for absolute conditioning
    drop_aa = bool(chain.get("drop_aa", False))
    bb = _backbone_slots()
    drop_backbone = bool(chain.get("drop_backbone", False))
    sel = _apply_interface_selection(model, sel, names, chain)
    sel = _apply_neighbour_selection(model, sel, names, chain)
    inter_sel = _interaction_sel(chain, model, sel, names)
    if inter_sel and chain.get("auto_condition"):
        raise ValueError(
            "from_chains: 'auto_condition' cannot be combined with explicit "
            "interaction keys (h_bond/salt_bridge/pi_stack/pi_cation)")
    hotspot_sel = _pdb_hotspot_sel(chain, model, sel, names)
    auto_condition = bool(chain.get("auto_condition"))
    hotspot_where = np.zeros((0, 37), bool) if hotspot_sel is not None else None
    hbond_vals = (np.zeros((0, 37), np.int32)
                  if auto_condition and hotspot_sel is not None else None)

    pos_rows, where_rows, restype_rows, ridx_by_new_chain = [], [], [], []
    motif_res: list[tuple[str, object, np.ndarray]] = []
    # str-hotspot / auto_condition accumulators (indexed rows, filled below)
    n_motif_rows = 0
    for name in names:
        pdb_chain = model[name]
        if pdb_chain is None:
            raise ValueError(f"from_chains: chain {name!r} not found in {path!r}")
        chain_sel = sel.get(name, {}) if sel is not None else None
        token_atoms = ENC.token_atoms
        pos_ch, where_ch, restype_ch, ridx_ch = [], [], [], []
        for res in pdb_chain:
            if res.name in ("HOH", "DOD", "WAT"):
                continue  # solvent is never part of the design
            if chain_sel is not None and "*" not in chain_sel \
                    and res.seqid.num not in chain_sel:
                continue  # residue not selected
            atom_sel = (chain_sel.get("*", chain_sel.get(res.seqid.num))
                        if chain_sel is not None else None)
            n37 = np.full((37, 3), np.nan, np.float32)
            mask37 = np.zeros(37, dtype=bool)
            seen = set()
            for atom in res:
                an = atom.name.strip()
                if an in seen:  # first altloc wins
                    continue
                seen.add(an)
                if atom_sel is not None and an not in atom_sel:
                    continue  # atom not selected
                if drop_backbone and an in ("N", "CA", "C", "O", "OXT"):
                    continue  # backbone dropped (sidechains-only motif)
                res_atoms = token_atoms.get(res.name)
                slot = None
                if res_atoms is not None:
                    for s, slot_name in enumerate(res_atoms):
                        if str(slot_name).strip() == an:
                            slot = s
                            break
                if slot is None:
                    continue  # hydrogen / unmapped atom
                mask37[slot] = True
                n37[slot] = [atom.pos.x, atom.pos.y, atom.pos.z]
            if drop_aa:
                keep = np.zeros(37, dtype=bool)
                keep[bb] = True
                mask37 &= keep
            if drop_backbone and not mask37.any():
                continue  # nothing left after the backbone drop (e.g. GLY)
            motif_pos = n_motif_rows   # this residue's local motif index
            motif_res.append((name, res, mask37))
            if hotspot_where is not None:
                hotspot_where = np.vstack([
                    hotspot_where, np.zeros((1, 37), bool)])
            if hbond_vals is not None:
                hbond_vals = np.vstack([hbond_vals, np.zeros((1, 37), np.int32)])
            n_motif_rows += 1
            if hotspot_sel is not None:
                h_chain = hotspot_sel.get(name, {})
                if "*" in h_chain or res.seqid.num in h_chain:
                    h_atoms = h_chain.get("*", h_chain.get(res.seqid.num))
                    for s in range(37):
                        an = str(ENC.token_atoms.get(res.name, ("",) * 37)[s]).strip()
                        if (mask37[s] and an
                                and (h_atoms is None or an in h_atoms)):
                            hotspot_where[motif_pos, s] = True
                    if auto_condition:
                        roles = _residue_hbond_roles(
                            res, {str(ENC.token_atoms[res.name][s]).strip()
                                  for s in range(37) if mask37[s]
                                  and str(ENC.token_atoms[res.name][s]).strip()
                                  and (h_atoms is None
                                       or str(ENC.token_atoms[res.name][s]).strip()
                                       in h_atoms)})
                        for an, role in roles.items():
                            for s, slot_name in enumerate(
                                    ENC.token_atoms.get(res.name, ("",) * 37)):
                                if (str(slot_name).strip() == an and mask37[s]):
                                    hbond_vals[motif_pos, s] = role
            pos_ch.append(n37)
            where_ch.append(mask37)
            restype_ch.append(0 if drop_aa else ENC.token_to_idx.get(res.name, 0))
            ridx_ch.append(res.seqid.num)
        if not pos_ch:
            raise ValueError(
                f"from_chains: chain {name!r} in {path!r} has no (solvent-free, "
                "selected) residues")
        ci = spec.add_chain(len(pos_ch))
        residue_index_by_chain[ci] = np.asarray(ridx_ch, dtype=np.int32)
        pos_rows.append(np.asarray(pos_ch, dtype=np.float32))
        where_rows.append(np.asarray(where_ch, dtype=bool))
        restype_rows.append(np.asarray(restype_ch, dtype=np.int32))

    # one indexed-motif overlay over the whole system: the pdb chains were added
    # LAST (spec add order), so chains from earlier dicts precede them — the pad
    # rows (all-False, untouched by the overlay) come FIRST, pdb rows last.
    pos_sys = np.concatenate(pos_rows, axis=0)
    where_sys = np.concatenate(where_rows, axis=0)
    restype_sys = np.concatenate(restype_rows, axis=0)
    pad = sum(c.num_residues for c in spec.chains) - pos_sys.shape[0]
    if pad:  # chains added before this dict: all-False rows, then the pdb rows
        pos_sys = np.concatenate(
            [np.full((pad, 37, 3), np.nan, np.float32), pos_sys])
        where_sys = np.concatenate([np.zeros((pad, 37), dtype=bool), where_sys])
        restype_sys = np.concatenate([np.zeros(pad, dtype=np.int32), restype_sys])
    assert pos_sys.shape[0] == sum(c.num_residues for c in spec.chains)
    motif_start = pad   # the prepended all-False rows precede the pdb motif
    hotspot_payload = None
    hotspot = chain.get("hotspot")
    if hotspot_sel is not None:
        hotspot_payload = (np.int32(1), hotspot_where[:n_motif_rows], motif_start)
    elif hotspot is not None:
        # non-str hotspot: regular atom-level conditioning over the motif window
        n_motif = pos_sys.shape[0] - pad
        if isinstance(hotspot, dict):
            vals, w = hotspot.get("values", 1), hotspot.get("where", True)
        else:
            vals, w = hotspot, True
        vals_arr = np.asarray(vals)
        if vals_arr.ndim == 0:
            vals_arr = np.full((n_motif, 37), vals_arr.item(), np.int32)
        elif vals_arr.ndim == 1:
            vals_arr = np.broadcast_to(vals_arr[:, None], (n_motif, 37))
        if w is True:
            where_arr = np.ones((n_motif, 37), bool)
        else:
            where_arr = np.asarray(w, bool)
            if where_arr.ndim == 1:
                where_arr = np.broadcast_to(where_arr[:, None], (n_motif, 37))
        if vals_arr.shape != (n_motif, 37) or where_arr.shape != (n_motif, 37):
            raise ValueError(
                f"from_chains: pdb hotspot values/where must cover the motif "
                f"window ({n_motif} tokens)")
        hotspot_payload = (vals_arr, where_arr, motif_start)
    hbond_payload = None
    if auto_condition and hotspot_sel is not None and hbond_vals.any():
        hbond_payload = (hbond_vals[:n_motif_rows], hbond_vals[:n_motif_rows] > 0,
                         motif_start)
    interaction_payloads = _interaction_payloads(inter_sel, motif_res, chain,
                                                 motif_start)
    bond_edges = _pdb_bond_edges(chain, model, motif_res, motif_start)
    if fixed:
        spec.add_indexed_motif(pos_sys, restype_sys, where_sys)
        # the spec path centers coordinates at compile time; the post-compile
        # overlay rewrites cond_coords/coords_nm AS-GIVEN so that mixed
        # indexed/unindexed motifs share one frame (from_chains applies one
        # global centering over all conditioned atoms at the end)
        return {"hotspot": hotspot_payload, "hbond": hbond_payload,
                **interaction_payloads,
                "bond_edges": bond_edges,
                "fixed_true": (pos_sys, where_sys, restype_sys)}
    # fixed=False: the internal-coordinate overlay needs the compiled input
    # (RFD4Input.add_indexed_motif supports fixed/group); the chains were
    # added as plain de-novo chains — the overlay is applied post-compile.
    return {"fixed_false": (pos_sys, where_sys, restype_sys, group),
            "hotspot": hotspot_payload, "hbond": hbond_payload,
            **interaction_payloads, "bond_edges": bond_edges}



def _resolve_where(where, k: int, *, atom_level: bool) -> jnp.ndarray:
    """Normalize an adder's `where` into a selection mask.

    ``True`` → the entire window (all k residues; all 37 slots when the target
    is an atom-level channel); a bool (k,) array → per-residue selection
    (broadcast to all 37 slots for atom-level channels); a bool (k, 37) array →
    per-atom selection (atom-level channels only; rejected for residue-level
    targets such as C_SEC).
    """
    if where is True:
        if atom_level:
            return jnp.ones((k, 37), bool)
        return jnp.ones((k,), bool)
    arr = jnp.asarray(where).astype(bool)
    if atom_level:
        if arr.ndim == 1:
            if arr.shape[0] != k:
                raise ValueError(f"where must have length {k}, got {arr.shape[0]}")
            return jnp.broadcast_to(arr[:, None], (k, 37))
        if arr.ndim == 2:
            if arr.shape != (k, 37):
                raise ValueError(
                    f"where must be (k, 37) with k = {k}, got {arr.shape}")
            return arr
        raise ValueError(f"where must be True, (k,) or (k, 37); got shape {arr.shape}")
    if arr.ndim != 1 or arr.shape[0] != k:
        raise ValueError(
            f"where must be True or a bool array of length {k} (residue-level "
            f"target), got shape {arr.shape}")
    return arr


def _center_on_conditioned_atoms(inp: "RFD4Input") -> "RFD4Input":
    """Shift cond_coords (and the matching ground-truth coords_nm) so that the
    centroid of all absolutely-conditioned atoms sits at the origin. No-op
    when there is no absolute conditioning (all-NaN cond_coords)."""
    if inp.cond_coords is None:
        return inp
    cc = np.asarray(inp.cond_coords)
    finite = np.isfinite(cc).all(-1)
    if not finite.any():
        return inp
    centroid = cc[finite].mean(axis=0)
    if not np.isfinite(centroid).all() or np.allclose(centroid, 0.0, atol=1e-6):
        return inp
    cc = cc.copy()
    cc[finite] -= centroid
    kwargs = {"cond_coords": jnp.asarray(cc)}
    if inp.coords_nm is not None:
        cn = np.asarray(inp.coords_nm, copy=True)
        cn_finite = np.isfinite(cn).all(-1)
        # shift only tokens that carry ground truth (de-novo tokens stay 0)
        both = finite & cn_finite
        if both.any():
            cn[both] -= centroid / 10.0
        kwargs["coords_nm"] = jnp.asarray(cn)
    return dataclasses.replace(inp, **kwargs)


# ------------------------------------------------------------------- padding
#: per-token fields whose inert padded value is -1 (UNKNOWN / unset ids)
_PAD_NEG1 = ("coord_groups", "cath_codes", "idx_group", "element")
#: per-token fields whose inert padded value is NaN (unspecified coordinates /
#: sequence-probability channels)
_PAD_NAN = ("cond_coords", "cond_internal_coords", "chain_centroid",
            "seq_probabilities")
#: pair fields — padded rows/cols zeroed (disconnected / no feature)
_PAD_PAIR = ("token_bond_path_length", "bond_mask", "bond_order", "is_aromatic",
             "canonical_polymer_ordering", "na_ss_pair", "na_ss_pair_mask")


def _pad_batch(batch, coord_cond, pad_to: int):
    """Extend a compiled FlowBatch to `pad_to` total tokens with inert padded
    slots (padding_mask False; schema-inactive defaults elsewhere).

    The trunk/AE/flow all mask via `padding & ~in_context`, so padded slots are
    excluded as attention keys and zeroed as queries, emit nothing, and carry no
    noise. The defaults matter wherever features reduce GLOBALLY over the batch:
    e.g. CoordGroupSeqFeat's `present`-group reduction spans all tokens, so
    padded coord_groups must stay -1 to keep the real tokens' group-label
    bijection unchanged.
    """
    import warnings

    from .types import FlowBatch

    fields = {f: getattr(batch, f) for f in INPUT_FIELDS if getattr(batch, f) is not None}
    n_real = int(np.asarray(fields["padding_mask"]).shape[1])
    if pad_to <= n_real:
        if pad_to < n_real:
            warnings.warn(
                f"pad_to={pad_to} is smaller than the compiled token count "
                f"{n_real}; leaving the batch unpadded", stacklevel=2)
        return batch, coord_cond
    pad = pad_to - n_real
    out = {}
    for f, v in fields.items():
        arr = np.array(v)
        if f == "padding_mask":
            out[f] = np.concatenate([arr, np.zeros((1, pad), bool)], axis=1)
        elif f in _PAD_PAIR:
            out[f] = np.pad(arr, ((0, 0), (0, pad), (0, pad)))
        else:
            block_shape = (1, pad) + arr.shape[2:]
            if f in _PAD_NEG1:
                block = np.full(block_shape, -1, arr.dtype)
            elif f in _PAD_NAN:
                block = np.full(block_shape, np.nan, arr.dtype)
            else:
                block = np.zeros(block_shape, arr.dtype)  # 0 / False by dtype
            out[f] = np.concatenate([arr, block], axis=1)
    new_batch = FlowBatch(**{k: jnp.asarray(v) for k, v in out.items()})
    new_cc = None
    if coord_cond is not None:
        cc = np.array(coord_cond)
        new_cc = jnp.asarray(np.concatenate([cc, np.zeros((1, pad), bool)], axis=1))
    return new_batch, new_cc


def _pad_input(inp: "RFD4Input", pad_to: int) -> "RFD4Input":
    """Pad an ``RFD4Input`` directly to ``pad_to`` tokens without ``to_batch()`` roundtrip."""
    import warnings

    inp = inp._finalize_intcrd_groups() if inp._intcrd_sources else inp
    n_real = int(inp.padding_mask.shape[1])
    if pad_to <= n_real:
        if pad_to < n_real:
            warnings.warn(
                f"pad_to={pad_to} is smaller than the compiled token count "
                f"{n_real}; leaving the batch unpadded", stacklevel=2)
        return inp
    pad = pad_to - n_real
    updates: dict[str, Array] = {}
    for f in INPUT_FIELDS:
        v = getattr(inp, f)
        if v is None:
            continue
        arr = np.asarray(v)
        if f == "padding_mask":
            padded = np.concatenate([arr, np.zeros((1, pad), bool)], axis=1)
        elif f in _PAD_PAIR:
            padded = np.pad(arr, ((0, 0), (0, pad), (0, pad)))
        else:
            block_shape = (1, pad) + arr.shape[2:]
            if f in _PAD_NEG1:
                block = np.full(block_shape, -1, arr.dtype)
            elif f in _PAD_NAN:
                block = np.full(block_shape, np.nan, arr.dtype)
            else:
                block = np.zeros(block_shape, arr.dtype)
            padded = np.concatenate([arr, block], axis=1)
        updates[f] = jnp.asarray(padded)
    if inp.coord_cond is not None:
        cc = np.asarray(inp.coord_cond)
        updates["coord_cond"] = jnp.asarray(
            np.concatenate([cc, np.zeros((1, pad), bool)], axis=1))
    return dataclasses.replace(inp, **updates)


# ------------------------------------------------- unindexed motifs (BF13)
def _unindexed_motif_kwargs(chain: dict) -> dict:
    """Extract add_unindexed_motif kwargs from an
    {"unindexed_motif": True, "pos", "mask", "aa"[, "residue_index", "fixed",
    "element", "charge", "bonds", "smiles"/"ccd"]} dict.

    With "smiles"/"ccd" the ligand chemistry (element/charge/bonds) is decoded
    through the RFD4Spec.add_ligand machinery (RDKit/CCD); the decoded atom
    order must match the block's <A> token order."""
    for key in ("pos", "mask", "aa"):
        if key not in chain:
            raise ValueError(f"from_chains: 'unindexed_motif' requires {key!r}")
    kwargs = {"pos": chain["pos"], "mask": chain["mask"], "aa": chain["aa"],
              "fixed": bool(chain.get("fixed", True))}
    if "residue_index" in chain:
        kwargs["residue_index"] = chain["residue_index"]
    if "chain" in chain:
        kwargs["chain"] = chain["chain"]
    if "element" in chain:
        kwargs["element"] = chain["element"]
    if "charge" in chain:
        kwargs["charge"] = chain["charge"]
    if "bonds" in chain:
        kwargs["bonds"] = chain["bonds"]
    if "smiles" in chain or "ccd" in chain:
        from .spec import RFD4Spec

        tmp = RFD4Spec()
        tmp.add_ligand(smiles=chain.get("smiles"), ccd=chain.get("ccd"))
        lig = tmp.ligands[0]
        if "element" in kwargs or "bonds" in kwargs:
            raise ValueError(
                "from_chains: 'unindexed_motif' takes either smiles/ccd or "
                "explicit element/charge/bonds, not both")
        # LigandSpec.bonds carry biotite bond-type ids; the method wants raw
        # order + aromatic flag
        order_map = {1: 1, 2: 2, 3: 3, 5: 1, 6: 2, 7: 3, 9: 1}
        bonds = np.stack([
            lig.bonds[:, 0], lig.bonds[:, 1],
            np.array([order_map[int(bt)] for bt in lig.bonds[:, 2]], np.int32),
            lig.bonds[:, 3].astype(np.int32)], axis=1).astype(np.int32)
        kwargs["element"] = lig.elements
        kwargs["charge"] = lig.charges.astype(np.int32)
        kwargs["bonds"] = bonds
    return kwargs


def _apply_seq_identity(sp, tokens, values):
    """One-hot identity rows in seq_probabilities (the trunk's C_SEQ feature).

    Spec semantics: an identity-conditioned token carries an all-zero FINITE
    row with a 1.0 at its residue-name channel (NaN elsewhere = unspecified).
    Overwrites the touched tokens' rows. `sp` is a numpy (1, n, 28) array."""
    from .featurize import SEQ_PROB_ORDER

    idx_to_tok = _idx_to_token()
    atomize = _atomize_index()
    for t, v in zip(tokens, values):
        v = int(v)
        if v == 0 or v == atomize:
            continue
        key = f"condition_sequence-probability-{str(idx_to_tok.get(v, ''))}_1_residue"
        if key in SEQ_PROB_ORDER:
            sp[0, t, :] = 0.0
            sp[0, t, SEQ_PROB_ORDER.index(key)] = 1.0
    return sp


@lru_cache(maxsize=1)
def _idx_to_token() -> dict:
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    enc = ENC.idx_to_token
    return enc if isinstance(enc, dict) else dict(enumerate(enc))


@lru_cache(maxsize=1)
def _atomize_index() -> int:
    """The AE vocabulary index of the atomized token <A>."""
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    return int(ENC.token_to_idx["<A>"])


@lru_cache(maxsize=1)
def _vocab_token_type_table() -> np.ndarray:
    """Cached (V+1,) token-type lookup indexed by vocabulary class id."""
    from rfproteina.encoding import RESIDUES_BY_TOKEN_TYPE as GROUPS

    idx_to_tok = _idx_to_token()
    v_size = max(idx_to_tok) + 1 if idx_to_tok else 0
    tbl = np.zeros(v_size + 1, dtype=np.int32)
    group_of: dict[str, int] = {}
    for group, tt_id in (("dna", 2), ("rna", 1), ("ncaa", 6), ("glycan", 5)):
        for g in GROUPS[group]:
            group_of[g.upper()] = tt_id
    for idx, tok in idx_to_tok.items():
        tbl[idx] = group_of.get(str(tok).upper(), 0)
    return tbl


def _token_types_for(aa_arr: np.ndarray, atomized: np.ndarray) -> np.ndarray:
    """Vocabulary-group token types for motif tokens (mirrors the compiled
    direct featurizer: dna/rna/ncaa/glycan groups from the restype names,
    atomized tokens -> 3)."""
    tbl = _vocab_token_type_table()
    sentinel = len(tbl) - 1
    aa_i = np.asarray(aa_arr, dtype=np.int32)
    safe = np.where((aa_i >= 0) & (aa_i < sentinel), aa_i, sentinel)
    tt = tbl[safe].copy()
    tt[atomized] = 3
    return tt


def _fresh_coord_groups(cg_existing: np.ndarray, atomized: np.ndarray,
                        group: int | None) -> np.ndarray:
    """Per-token coordinate groups for one motif block (the shared rule):
    `group` given (residue_index route) → that value for every token; None
    (no residue_index) → fresh ids, one per residue, with each run of
    consecutive atomized (<A>) tokens — a small molecule — sharing one id."""
    m = len(atomized)
    if group is not None:
        return np.full(m, int(group), np.int32)
    next_cg = int(max(cg_existing.max(), 0)) + 1
    cg = np.empty(m, np.int32)
    i = 0
    while i < m:
        if atomized[i]:
            j = i
            while j < m and atomized[j]:
                j += 1
            cg[i:j] = next_cg
            next_cg += 1
            i = j
        else:
            cg[i] = next_cg
            next_cg += 1
            i += 1
    return cg


def _append_motif_tokens(inp: "RFD4Input", pos: np.ndarray, aa: np.ndarray,
                         mask: np.ndarray, residue_index: np.ndarray | None,
                         chain_index, group, fixed: bool = True,
                         chem: dict | None = None, chain=None,
                         ordered: bool | None = None,
                         target_chain=None) -> "RFD4Input":
    """Append `m` motif tokens to a compiled RFD4Input (see
    add_unindexed_motif): residue tokens are in-context conditioning;
    ligand runs (consecutive <A> tokens) become INDEXED design tokens with
    full chemistry when supplied.

    fixed=True -> absolute coordinate conditioning (C_CRD, cond_coords from
    pos, coord_cond set, coordinate groups per the shared rule);
    fixed=False -> internal coordinate conditioning (C_INTCRD/C_CRDGRP: the
    trained TipAtomPartialLigandTask shape): cond_coords stay NaN, coord_cond
    is not set, and each rigid body (the residue motif when residue_index is
    given, per-residue groups otherwise; each ligand run) gets a fresh
    positive coordinate group whose payload is centered on the group's atoms
    and randomly rotated (mirroring the native MaterializeInternalCoords).

    chain: None -> the motif residues join the last non-motif chain (the
    native unindexed-benchmark topology); int / int (m,) array -> attach to
    the i-th existing non-motif chain; "new" -> a fresh chain. Ligand runs
    always get their own fresh chains.
    """
    n = int(np.asarray(inp.padding_mask).shape[1])
    m = len(aa)
    atomized = aa == _atomize_index()
    # ligand runs: maximal consecutive <A> token runs
    lig_id = np.full(m, -1, np.int32)
    run = -1
    i = 0
    while i < m:
        if atomized[i]:
            run += 1
            j = i
            while j < m and atomized[j]:
                lig_id[j] = run
                j += 1
            i = j
        else:
            i += 1
    is_ligand = lig_id >= 0
    n_lig_runs = run + 1

    finite_slot = mask & np.isfinite(pos).all(-1)  # (m, 37)
    finite_tok = finite_slot.any(-1)  # (m,)
    center = (pos[finite_slot].mean(0) if finite_slot.any()
              else np.zeros(3, np.float32))

    # ---- coordinate groups
    cg_existing = (np.asarray(inp.coord_groups) if inp.coord_groups is not None
                   else np.full((1, n, 37), -1, np.int32))
    next_cg = int(max(cg_existing.max(), 0)) + 1
    cg_tok = np.empty(m, np.int32)
    if fixed:
        # absolute route: C_CRD atoms are coordinate group 0 (native semantics,
        # independent of residue_index)
        cg_tok[:] = int(group)
    else:
        # internal route: an explicit positive `group` pins the residue motif
        # to that coordinate group (shared rigid body with other pieces of
        # the same group — merged over the union at to_batch); group 0 mints
        # a fresh group (the pre-native-faithful default). Each ligand run
        # always gets its own fresh group (trained shape).
        g_res = int(group)
        if g_res < 0:
            raise ValueError(
                f"_append_motif_tokens: fixed=False requires group >= 0 "
                f"(0 = fresh group), got {g_res}")
        if g_res == 0:
            g_res = next_cg
        cg_tok[~is_ligand] = g_res
        next_cg = max(next_cg, g_res + 1)
        for r in range(n_lig_runs):
            cg_tok[lig_id == r] = next_cg
            next_cg += 1

    # ---- index groups + residue linkage
    # Native semantics for IN-CONTEXT (unindexed) tokens (verified against the
    # native 1QJG unindexed-motif featurization): C_IDX=False -> each token
    # carries its OWN index group, NO token-bond paths and NO canonical
    # polymer ordering — they are index-disconnected holes in the chain.
    # residue_index is accepted for authoring/reference purposes only.
    # `ordered` (pdb/cif motif route) overrides this: True -> one group per
    # CONTIGUOUS same-chain run of the motif residues (contiguity from
    # residue_index + chain_index); False -> per-token. Ligand runs are
    # ALWAYS isolated into their own groups (native: indexed ligand tokens
    # can never share an index group with unindexed tokens).
    ig_existing = (np.asarray(inp.idx_group) if inp.idx_group is not None
                   else np.zeros((1, n), np.int32))
    next_ig = int(ig_existing.max()) + 1
    if ordered is None:
        ig_tok = next_ig + np.arange(m, dtype=np.int32)
    else:
        ig_tok = np.empty(m, np.int32)
        next = next_ig
        i = 0
        while i < m:
            if is_ligand[i]:                      # one isolated group per run
                j = i
                while j < m and lig_id[j] == lig_id[i]:
                    j += 1
                ig_tok[i:j] = next
                next += 1
                i = j
            elif ordered:
                if residue_index is None or chain_index is None:
                    raise ValueError(
                        "_append_motif_tokens: ordered=True requires "
                        "residue_index and chain_index (author numbering per "
                        "token, for contiguity)")
                j = i + 1
                while (j < m and not is_ligand[j]
                       and chain_index[j] == chain_index[i]
                       and residue_index[j] == residue_index[j - 1] + 1):
                    j += 1
                ig_tok[i:j] = next
                next += 1
                i = j
            else:
                ig_tok[i] = next
                next += 1
                i += 1
    block_dist = block_link = None

    # ---- internal-coordinate payload (fixed=False): per group, centered and
    # randomly rotated (native MaterializeInternalCoords semantics); pieces
    # sharing a group are re-merged over the union at to_batch
    intcrd = np.full((m, 37, 3), np.nan, np.float32)
    intcrd_src = list(getattr(inp, "_intcrd_sources", ()) or ())
    abs_slots = (n + np.arange(m))[:, None] * 37 + np.arange(37)[None, :]
    if not fixed:
        for g in np.unique(cg_tok):
            sel = (cg_tok[:, None] == g) & finite_slot
            if not sel.any():
                continue
            xy = pos[sel]  # (k, 3)
            payload = xy - xy.mean(0, keepdims=True)
            payload = payload @ _random_rotation(np.random)
            intcrd[sel] = payload
            intcrd_src.append((int(g), xy, abs_slots[sel]))

    # ---- chain / molecule ids: `chain` selects the motif RESIDUES' topology
    # (None -> join the last non-motif chain, the native unindexed-benchmark
    # shape; int / int (m,) array -> attach to the i-th existing non-motif
    # chain in first-appearance order; "new" -> a fresh chain). Ligand runs
    # ALWAYS get their own fresh chains (a ligand is a separate molecule).
    ciid_existing = np.asarray(inp.chain_iid)
    miid_existing = np.asarray(inp.molecule_iid)
    ent_existing = (np.asarray(inp.molecule_entity_id)
                    if inp.molecule_entity_id is not None
                    else np.zeros((1, n), np.int32))
    motif_ranges = tuple(getattr(inp, "motif_token_ranges", ()) or ())
    non_motif = np.ones(n, bool)
    for lo, hi in motif_ranges:
        non_motif[lo:hi] = False
    nm_idx = np.flatnonzero(non_motif)
    if nm_idx.size == 0:
        raise ValueError("add_unindexed_motif: no non-motif chain to join "
                         "(chain=None requires an existing chain)")
    # existing non-motif chains in first-appearance order
    chain_order: list[float] = []
    for t in nm_idx:
        v = float(ciid_existing[0, t])
        if v not in chain_order:
            chain_order.append(v)

    def _resolve_int(c: int) -> float:
        idx = int(c)
        if not 0 <= idx < len(chain_order):
            raise ValueError(
                f"add_unindexed_motif: chain={idx} out of range "
                f"({len(chain_order)} existing non-motif chains)")
        return chain_order[idx]

    if isinstance(chain, str):
        if chain != "new":
            raise ValueError(f"add_unindexed_motif: chain={chain!r} not "
                             "recognized (None, int, int array or 'new')")
        res_chain_vals = None  # fresh chain for the motif residues
    elif chain is None:
        res_chain_vals = np.full(m, float(ciid_existing[0, nm_idx[-1]]), np.float32)
    elif isinstance(chain, (int, np.integer)):
        res_chain_vals = np.full(m, _resolve_int(int(chain)), np.float32)
    else:
        arr_c = np.asarray(chain).reshape(-1)
        if arr_c.shape[0] != m:
            raise ValueError(
                f"add_unindexed_motif: chain array must have length {m}, got {arr_c.shape[0]}")
        res_chain_vals = np.array([_resolve_int(int(c)) for c in arr_c], np.float32)

    next_chain_iid = float(ciid_existing.max()) + 1.0
    next_molecule_iid = int(miid_existing.max()) + 1
    next_entity = int(ent_existing.max()) + 1
    ciid_tok = np.empty(m, np.float32)
    miid_tok = np.empty(m, np.int32)
    ent_tok = np.empty(m, np.int32)
    for t in range(m):
        if is_ligand[t]:
            ciid_tok[t] = next_chain_iid + 1 + lig_id[t]
            miid_tok[t] = next_molecule_iid + 1 + lig_id[t]
            ent_tok[t] = next_entity + 1 + lig_id[t]
        elif res_chain_vals is None:
            # "new": the motif residues get their own fresh chain/molecule ids
            ciid_tok[t] = next_chain_iid
            miid_tok[t] = next_molecule_iid
            ent_tok[t] = next_entity
        else:
            # join an existing chain: share its chain/molecule/entity ids
            src = np.flatnonzero((ciid_existing[0] == res_chain_vals[t]) & non_motif)
            if src.size == 0:  # unknown chain value -> fresh ids
                ciid_tok[t] = next_chain_iid
                miid_tok[t] = next_molecule_iid
                ent_tok[t] = next_entity
            else:
                s0 = int(src[0])
                ciid_tok[t] = ciid_existing[0, s0]
                miid_tok[t] = miid_existing[0, s0]
                ent_tok[t] = ent_existing[0, s0]

    if target_chain is not None:
        # "target_chain": pin the residue tokens' chain_iid and make the
        # molecule/entity FOLLOW the resolved chain (native semantics: a
        # chain never straddles molecules). If an existing chain carries the
        # resolved iid, its molecule/entity ids are copied; otherwise the
        # residue tokens get their own fresh molecule/entity (as `chain="new"`).
        t_iid = _resolve_target_chain(target_chain)
        ciid_tok[~is_ligand] = t_iid
        match = np.flatnonzero(ciid_existing[0] == t_iid)
        if match.size:
            s0 = int(match[0])
            miid_tok[~is_ligand] = miid_existing[0, s0]
            ent_tok[~is_ligand] = ent_existing[0, s0]
        else:
            miid_tok[~is_ligand] = next_molecule_iid
            ent_tok[~is_ligand] = next_entity

    def block_of(f: str, arr: np.ndarray) -> np.ndarray:
        """The m new rows for one field."""
        if f == "padding_mask":
            return np.ones((1, m), bool)
        if f == "is_in_context_conditioning":
            return (~is_ligand)[None]  # residues in-context; ligand indexed
        if f == "supports_local_latent":
            return np.zeros((1, m), bool)
        if f == "token_type":
            tt = _token_types_for(aa, np.zeros(m, bool))
            tt[is_ligand] = 3  # indexed atomized design tokens (native)
            return tt[None]
        if f == "seq_probabilities":
            from .featurize import SEQ_PROB_ORDER

            block = np.full((1, m, 28), np.nan, np.float32)
            identity = aa != 0
            block[0, identity] = 0.0
            for t in np.flatnonzero(identity & ~atomized):
                name = str(_idx_to_token().get(int(aa[t]), ""))
                key = f"condition_sequence-probability-{name}_1_residue"
                if key in SEQ_PROB_ORDER:
                    block[0, t, SEQ_PROB_ORDER.index(key)] = 1.0
            return block
        if f == "atom37_mask":
            return mask[None]
        if f == "coords_nm":
            # ground truth AS-GIVEN (the caller's frame — consistent with the
            # indexed-motif overlay; from_chains applies ONE global centering
            # over all conditioned atoms at the end)
            block = np.zeros((1, m, 37, 3), np.float32)
            block[0, finite_slot] = pos[finite_slot] / 10.0  # Å -> nm
            return block
        if f == "cond_coords":
            block = np.full((1, m, 37, 3), np.nan, np.float32)
            if fixed:  # absolute conditioning only
                block[0, finite_slot] = pos[finite_slot]  # Å, as-given
            return block
        if f == "cond_internal_coords":
            block = np.full((1, m, 37, 3), np.nan, np.float32)
            if not fixed:
                block[0, finite_slot] = intcrd[finite_slot]  # Å (group-local)
            return block
        if f == "coord_groups":
            block = np.full((1, m, 37), -1, np.int32)
            wide = np.broadcast_to(cg_tok[:, None], (m, 37))
            block[0, finite_slot] = wide[finite_slot]
            return block
        if f == "cond_seq":
            return aa[None]
        if f == "chain_centroid":
            return np.full((1, m, 3), np.nan, np.float32)
        if f == "cath_codes":
            return np.full((1, m, 3), -1, np.int32)
        if f == "idx_group":
            return ig_tok[None]
        if f in ("element", "ground_truth_element"):
            block = np.full((1, m, 37), -1, np.int32)
            # residue identity via the encoding lookup (native fill)
            for t in range(m):
                if aa[t] == 0 or is_ligand[t]:
                    continue
                res_atoms = _token_atoms_for(int(aa[t]))
                if res_atoms is None:
                    continue
                for s in np.flatnonzero(mask[t]):
                    an = str(res_atoms[s]).strip()
                    if not an:
                        continue
                    letters = "".join(c for c in an if c.isalpha()).upper()
                    two = letters[:2]
                    block[0, t, s] = (34 if two == "SE" else
                                      {"C": 6, "N": 7, "O": 8, "S": 16,
                                       "P": 15}.get(letters[:1], -1))
            if chem is not None:  # ligand chemistry from the kwargs
                block[0, is_ligand, 1] = chem["element"][is_ligand]
            if f == "ground_truth_element":
                # the truth channel: 0 where unknown (native semantics)
                block = np.where(block > 0, block, 0)
            return block
        if f == "charge":
            block = np.zeros((1, m, 37), np.int32)
            if chem is not None:
                block[0, is_ligand, 1] = chem["charge"][is_ligand]
            return block
        if f == "molecule_iid":
            return miid_tok[None]
        if f == "molecule_entity_id":
            return ent_tok[None]
        if f == "chain_iid":
            return ciid_tok[None]
        if f == "incident_bond_type_counts":
            block = np.zeros((1, m) + arr.shape[2:], arr.dtype)
            # intra-residue covalent chemistry for full-atom residue tokens
            # (native carries these even though the token-level bond_mask is 0)
            for t in range(m):
                if is_ligand[t] or aa[t] == 0:
                    continue
                counts = _residue_incident_counts(int(aa[t]))
                if counts is not None:
                    block[0, t] = counts.astype(arr.dtype)
            return block
        if f == "chirality":
            return np.zeros((1, m) + arr.shape[2:], np.float32)
        if f in ("chiral_frame_atoms", "chiral_slot_owner"):
            return np.full((1, m) + arr.shape[2:], -1, np.int32)
        return np.zeros((1, m) + arr.shape[2:], arr.dtype)

    new: dict = {}
    for f in INPUT_FIELDS:
        v = getattr(inp, f)
        if v is None:
            continue  # absent keys stay absent
        arr = np.array(v)
        if f in _PAD_PAIR:
            new[f] = np.pad(arr, ((0, 0), (0, m), (0, m)))
        else:
            new[f] = np.concatenate([arr, block_of(f, arr)], axis=1)

    rows = np.arange(n, n + m)
    if residue_index is not None and block_dist is not None:
        new["token_bond_path_length"][0][np.ix_(rows, rows)] = block_dist
        new["canonical_polymer_ordering"][0][np.ix_(rows, rows)] = block_link
    if chain_index is not None:
        new["chain_iid"][0, n:] = np.asarray(chain_index, np.float32).reshape(-1)

    # ---- ligand bond features (+ bond-graph BFS paths within each ligand)
    if chem is not None and chem["bonds"] is not None and n_lig_runs:
        from .featurize_atoms import (BT_AROMATIC, BT_AROMATIC_DOUBLE,
                                      BT_AROMATIC_SINGLE)

        def _bt(order: int, arom) -> int:
            if arom:
                return {1: BT_AROMATIC_SINGLE, 2: BT_AROMATIC_DOUBLE, 3: 7}.get(
                    int(order), BT_AROMATIC)
            return int(order) if int(order) in (1, 2, 3) else 1

        order_lookup = np.zeros(10, np.int16)
        order_lookup[1], order_lookup[2] = 1, 2
        order_lookup[5], order_lookup[6] = 1, 2
        order_lookup[7], order_lookup[9] = 3, 1
        bm = new["bond_mask"]
        bo = new["bond_order"]
        ia = new["is_aromatic"]
        inc = new["incident_bond_type_counts"]
        path = new["token_bond_path_length"]
        lig_tok = np.flatnonzero(is_ligand)  # motif-local; +n for global
        for (i, j, order, arom) in chem["bonds"]:
            ta, tb = n + int(lig_tok[i]), n + int(lig_tok[j])
            bt = _bt(order, arom)
            bm[0, ta, tb] = bm[0, tb, ta] = True
            bo[0, ta, tb] = bo[0, tb, ta] = int(order_lookup[int(bt)])
            if arom:
                ia[0, ta, tb] = ia[0, tb, ta] = True
            inc[0, ta, 1, int(bt)] += 1
            inc[0, tb, 1, int(bt)] += 1
        # BFS hop counts within each ligand run (bond graph; cutoff None);
        # bond indices are ligand-local -> map through the motif-local tokens
        if len(chem["bonds"]):
            from scipy.sparse import csr_matrix
            from scipy.sparse.csgraph import shortest_path

            bonds_arr = np.asarray(chem["bonds"], dtype=np.int32)
            mi = lig_tok[bonds_arr[:, 0]]
            mj = lig_tok[bonds_arr[:, 1]]
            rows_g = np.concatenate([mi, mj])
            cols_g = np.concatenate([mj, mi])
            graph = csr_matrix(
                (np.ones(len(rows_g), dtype=np.int8), (rows_g, cols_g)),
                shape=(m, m),
            )
            sp = shortest_path(graph, directed=False, unweighted=True)
            sp_int = np.where(np.isfinite(sp), sp, 0.0).astype(np.int32)
            path[0, n + lig_tok, n:n + m] = sp_int[lig_tok]

    # coordinate conditioning per token: only the absolute route
    cc = (np.array(inp.coord_cond, copy=True) if inp.coord_cond is not None
          else np.zeros((1, n), bool))
    cc_tok = finite_tok if fixed else np.zeros(m, bool)
    new["coord_cond"] = np.concatenate([cc, cc_tok[None]], axis=1)
    out = dataclasses.replace(
        inp, motif_token_ranges=motif_ranges + ((n, n + m),),
        _intcrd_sources=tuple(intcrd_src),
        **{k: jnp.asarray(v) for k, v in new.items()})
    # keep the whole system in one coordinate frame: re-center on ALL
    # absolutely-conditioned atoms after every append (a uniform shift, so
    # relative geometry — including previously added motifs — is preserved)
    return _center_on_conditioned_atoms(out)


def _validate_ligand_chemistry(aa: np.ndarray, element, charge, bonds):
    """Validate the optional ligand chemistry kwargs against the <A> runs.

    Returns None when no chemistry is given, else a dict with per-ligand-token
    ``element``/``charge`` arrays (over ALL motif tokens, -1/0 on non-ligand
    tokens) and the ``bonds`` array local to the concatenated ligand tokens."""
    atomized = aa == _atomize_index()
    n_lig = int(atomized.sum())
    if element is None and charge is None and bonds is None:
        if n_lig:
            return None  # chemistry-free ligand tokens (documented fallback)
    n = len(aa)

    def _per_token(values, name, default):
        arr = np.asarray(values)
        if arr.ndim == 0:
            arr = np.full(n_lig, arr.item())
        arr = arr.reshape(-1)
        if arr.shape[0] != n_lig:
            raise ValueError(
                f"add_unindexed_motif: {name} must have one value per ligand "
                f"atom ({n_lig}), got {arr.shape[0]}")
        out = np.full(n, default, np.int32)
        out[atomized] = arr.astype(np.int32)
        return out

    elem_tok = _per_token(element, "element", -1) if element is not None \
        else np.full(n, -1, np.int32)
    charge_tok = _per_token(charge, "charge", 0) if charge is not None \
        else np.zeros(n, np.int32)
    bonds_arr = None
    if bonds is not None:
        bonds_arr = np.asarray(bonds)
        if bonds_arr.ndim != 2 or bonds_arr.shape[-1] != 4 or bonds_arr.shape[0] == 0:
            raise ValueError(
                "add_unindexed_motif: bonds must be (n_bonds, 4) = "
                "[i, j, order, aromatic] local to the ligand tokens")
        if n_lig and (bonds_arr[:, :2].min() < 0 or bonds_arr[:, :2].max() >= n_lig):
            raise ValueError(
                f"add_unindexed_motif: bond indices must be local to the "
                f"ligand tokens (0..{n_lig - 1})")
        bonds_arr = bonds_arr.astype(np.int32)
    return {"element": elem_tok, "charge": charge_tok, "bonds": bonds_arr,
            "n_lig": n_lig}


@lru_cache(maxsize=None)
def _token_atoms_for(aa_idx: int):
    """The atom37 slot-name array for one AE residue-vocabulary index (or None)."""
    res_name = _idx_to_token().get(aa_idx)
    if res_name is None:
        return None
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    return ENC.token_atoms.get(res_name)


@lru_cache(maxsize=None)
def _residue_incident_counts(aa_idx: int) -> np.ndarray | None:
    """Intra-residue covalent bond counts (37, 10 biotite-type channels) for one
    AE residue-vocabulary index, from the cached CCD entry (the native
    featurization carries these on full-atom residue tokens even though the
    token-level bond_mask stays 0). None when the type has no CCD entry."""
    res_name = _idx_to_token().get(aa_idx)
    if not res_name or len(res_name) != 3:
        return None
    try:
        from .ccd import get_ccd_entry

        entry = get_ccd_entry(str(res_name).upper())
    except Exception:
        return None
    from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

    slot_names = ENC.token_atoms.get(str(res_name).upper())
    if slot_names is None:
        return None
    slot_of = {str(an).strip().upper(): s for s, an in enumerate(slot_names)}
    order_to_bt = {1: 1, 2: 2, 3: 3}
    arom_to_bt = {1: 5, 2: 6, 3: 7}
    counts = np.zeros((37, 10), np.int32)

    def _name(x):
        x = x.decode() if isinstance(x, bytes) else str(x)
        return x.strip().upper()

    for i, j, order, arom in zip(entry.bond_i.tolist(), entry.bond_j.tolist(),
                                 entry.bond_order.tolist(),
                                 entry.bond_aromatic.tolist()):
        # hydrogens are dropped before the bond graph in the native pipeline,
        # so bonds involving H never reach the counts; the C-terminal OXT
        # variant bond is likewise absent from the native polymer context
        if (_name(entry.elements[i]) == "H" or _name(entry.elements[j]) == "H"
                or _name(entry.atom_names[i]) in ("OXT", "HXT")
                or _name(entry.atom_names[j]) in ("OXT", "HXT")):
            continue
        s_i, s_j = slot_of.get(_name(entry.atom_names[i])), slot_of.get(_name(entry.atom_names[j]))
        bt = (arom_to_bt if arom else order_to_bt).get(int(order), 1)
        # endpoints without an atom37 slot (e.g. H) are skipped; the partner
        # still counts (ConvertToAtom37Format semantics)
        if s_i is not None:
            counts[s_i, bt] += 1
        if s_j is not None:
            counts[s_j, bt] += 1
    return counts


def _random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniform random rotation matrix (Mezzadri's QR construction)."""
    a = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(a)
    q = q @ np.diag(np.sign(np.diag(r)))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return q
