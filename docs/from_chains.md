# `RFD4Input.from_chains` — dictionary format reference

`RFD4Input.from_chains(*chains, np_seed=0, pad_to=None)` builds a compiled,
model-ready input from a sequence of chain / ligand / structure descriptions.
Featurization goes through `rfd4fx.spec.RFD4Spec`, so all semantics are
the parity-verified ones.

```python
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(
    80,                                  # de-novo 80-residue protein chain
    {"kind": "rna", "bp": "((((...))))"},  # RNA chain with dot-bracket C_BP
    {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
     "auto_condition": True, "buried": True},  # caffeine ligand
    {"pdb": "target.pdb", "chains": ["A"], "hotspot": 1},  # target motif
    np_seed=0,
)
batch = inp.pad_to_nearest(64).to_batch()
```

## Top-level arguments

| Argument | Meaning |
|---|---|
| `*chains` | Each entry: an `int` (unconditioned de-novo protein chain of that length) or a dict (see below). Entries are processed in order; **polymer chains come first in the compiled token layout, then ligand atom tokens**. |
| `np_seed` | Seed for the pipeline's chain/molecule-iid permutations (default 0). |
| `pad_to` | Pad the compiled input to exactly this TOTAL token count with inert padded slots (`padding_mask` False beyond the real tokens, schema-inactive defaults elsewhere). Padded slots are excluded from every attention/emit path. `pad_to_nearest(multiple)` is the equivalent post-hoc helper. |

## Entry types

### 1. Polymer chain — `int` or dict

An `int` is shorthand for an unconditioned de-novo protein chain of that
length. Polymer dicts recognize:

| Key | Type / values | Meaning |
|---|---|---|
| `num_residues` | int | Chain length. Inferred from other keys when absent (`length` → `dssp` → `bp` → `pos` → `residue_index`/`aa` → atom-level 1-D values). |
| `kind` | `"aa"`/`"AA"`, `"dna"`/`"DNA"`, `"rna"`/`"RNA"` | Polymer vocabulary group → `token_type` 0/2/1. Default `"aa"`. |
| `dssp` | str or (n,) int array | Protein secondary structure (C_SEC): `H`→1, `E`→2, `L`→3, `X`/`_`→0. Implies a protein chain when `kind` is absent. |
| `bp` | str | Dot-bracket NA secondary structure (C_NA_SS_STATE + C_BP): `.` unpaired, `()[]{}<>` + letter families pair (pseudoknots supported), `x`/`_`/`X` unknown. **Requires** `kind` ∈ {dna, rna}; infers `num_residues` from its length; conflicts with `dssp`. At compile time the `bp` strings of all nucleic-acid chains are concatenated in add order (an NA chain without `bp` contributes an all-`x` dummy), so pairs may cross chain boundaries. |
| `pos` + `mask` | float (n, 37, 3) + bool (n, 37) | Motif coordinate conditioning: materialized atom37 slots, ground-truth coordinates, C_CRD group `group` (default 0). `pos` requires `mask`. |
| `group` | int | Coordinate-group override for `pos` (default 0). |
| `chain_index` | int (n,) | Splits the block into separate chains by unique value (first-appearance order), each with its own chain identity. |
| `residue_index` | int (n,) | Within-chain bond path = \|Δresidue_index\|; canonical linkage at index distance 1. |
| `hotspot`, `rasa`, `hbond`, `pistack`, `pication`, `saltbridge`, `hydrophobic`, `metal_coord` | scalar / (n,) / (n, 37) — or `{"values", "aa", "where"}` | Atom-level conditions written at the materialized slots. Chain-level `aa` (int (n,)) and `where` (bool (n, 37)) set defaults for the whole block (the per-condition sub-dict keys override). Residue-level `hotspot` (shape (n,)) requires `mask` and applies `mask & hotspot[:, None]`. Value domains: `rasa` 0–3 (1=buried, 2=intermediate, 3=exposed), `hbond` 0–3 (1=donor, 2=acceptor, 3=both), `saltbridge` 0–2 (1=positive, 2=negative), the rest boolean. |

### 2. Ligand — dict with `smiles` or `ccd`

Ligands become atomized tokens (one per heavy atom) appended after all
polymer chains; positions are diffused design targets unless anchored.

| Key | Type | Meaning |
|---|---|---|
| `smiles` | str | Ligand from SMILES (RDKit; heavy atoms, bond orders, aromatic flags). Exactly one of `smiles`/`ccd`. |
| `ccd` | str | Ligand from the cached chemical component dictionary (gemmi; atom order = CCD file order). |
| `ligand_pos` | (num_atoms, 3) float, Å | Coordinate-conditions every ligand atom (C_CRD group 0). |
| `res_name` | str | Residue code written to PDB/mmCIF output (default `"LIG"` for SMILES, the CCD id). |
| `auto_condition` | bool | Autofeaturize interaction conditions from the ligand graph (native AnnotateInteractions/PLIP semantics): C_PISTACK on RDKit-aromatic 5/6-ring atoms, C_HB donor/acceptor roles (acceptor = any N/O/S; donor = N/O with H), C_SALT charge sign (1=positive, 2=negative). Works for SMILES and CCD ligands. |
| `buried` | bool | Set C_RAS 1 (buried) on every atom of this ligand chain. |

### 3. Structure motif — dict with `pdb` or `cif`

Loads a structure as motif conditioning. Two routes:

**Indexed route (default)** — fixed motif residues INSIDE the generated
system: fresh chains, one per selected PDB chain, in file order — one token
per solvent-free residue (HOH/DOD/WAT skipped).

| Key | Type | Meaning |
|---|---|---|
| `pdb` / `cif` | path (str/Path) | Structure file; read with `gemmi.read_structure`, first model (mmCIF files are read directly). Exactly one. |
| `chains` | list of chain names | Only these chains are loaded (default: all chains, file order). Ignored when `select` is present. |
| `select` | list of selection entries | Fine-grained chain/residue/atom selection (grammar below); overrides `chains`. |
| `fixed` | bool (default True) | True: absolute coordinate conditioning (C_CRD, coordinate group `group`, default 0). False: internal coordinates (C_INTCRD/C_CRDGRP) — one fresh positive coordinate group for the whole selection; the overlay is applied on the compiled input. |
| `group` | int | Coordinate-group override (`fixed=True`: default 0; `fixed=False`: default 1 — 0 is reserved for absolute conditioning). |
| `drop_aa` | bool | Backbone-only motif: mask restricted to N/CA/C/O and residue identity not conditioned (C_SEQ stays `<M>`). |
| `drop_backbone` | bool | Sidechains-only motif: N/CA/C/O/OXT are dropped from the selection; residues left without atoms (e.g. GLY) are skipped (no token) and reported. |
| `interface_to` / `cutoff` | str chain name / float (default 4.5) | Restrict the selection to interface residues: residues of the selected chains (other than `interface_to` itself) with at least one heavy atom within `cutoff` of any heavy atom of the `interface_to` chain. Without a `"select"`, the interface residues BECOME the selection; a `"*"` entry expands to the interface residues. |
| `motif` | list of sub-specs | Partially-indexed motif: each entry `{"select", "indexed", "drop_backbone", "ordered"}` becomes its own pdb/cif dict inheriting the dict-level `fixed`/`group`/`hotspot`/`auto_condition`; `"indexed": False` routes the sub-spec through the unindexed route. Sub-specs share the dict-level coordinate `group` (default 1 when `fixed: False` and unset) => one rigid body (see the shared-group rule below). |
| `hotspot` | str (selection grammar) or atom-level values | str: C_HOT on the selected motif residues/atoms (entries reference the loaded motif; out-of-motif entries raise). Non-str: regular atom-level conditioning (`{"values", "aa", "where"}` or scalar/(n,)/(n, 37)) over the motif window. |
| `auto_condition` | bool | C_HB donor/acceptor conditioning on the hotspot residues' atoms (str `hotspot` required): standard residues via the native (res_name, atom_name) tables, non-standard via the element rule (acceptor = N/O/S; donor = N/O with a file H). Values 1=donor, 2=acceptor, 3=both; intersected with the hotspot's atom subset. |

**Shared coordinate groups (auto-merge)**: every `fixed: False` motif piece
(indexed overlay or appended unindexed block) with the same POSITIVE
coordinate-group label forms ONE rigid body — at `to_batch` time the
internal-coordinate payload is recomputed over the union of all pieces' atoms
(single centroid + single random rotation). The unindexed pdb/cif route
passes the dict's `"group"` through on the `fixed: False` path (group 0 keeps
the fresh-group default); `RFD4Input.add_unindexed_motif(..., fixed=False,
group=g)` accepts the same explicit group. Pieces with distinct groups keep
independent frames (as before).

**Unindexed route** (`"unindexed_motif": True`): the selection becomes an
appended unindexed motif block via `add_unindexed_motif` (in-context motif
residues; atomized `<A>` ligand tokens are indexed design tokens). All keys
above apply, plus:

| Key | Type | Meaning |
|---|---|---|
| `fixed` | bool (default True) | Absolute vs internal coordinate conditioning (see `add_unindexed_motif`). |
| `ordered` | bool (default False) | Index-group mode: True -> one index group per CONTIGUOUS same-chain run of the motif residues (contiguity from author residue numbers + chain membership); False -> per-token groups. Ligand runs are always isolated into their own groups (native: indexed ligand tokens can never share an index group with unindexed tokens). Coordinate groups (fixed=False) follow the adder's rigid-body rule (one group for the residue motif, one per ligand run). |
| `hotspot` / `auto_condition` | as above | C_HOT / C_HB conditioning on the appended block's tokens (str `hotspot` per the selection grammar). |
| `ligands` | dict {resname: SMILES} | Chemistry for HETATM residues in the selection: bonds/orders/aromaticity + formal charges recovered from the SMILES via coordinate-perceived graph isomorphism. Falls back to the CCD entry (name-mapped) for ligands not covered, then to elements-only (no bonds, zero charges). |

#### Selection grammar (`"select"` entries)

| Entry | Selects |
|---|---|
| `A` | the whole chain A |
| `A1` | residue A1 (author numbering), all present atoms |
| `A1-10` | residues A1 through A10 (inclusive) |
| `A1@N,CA,C` | residue A1, only the named atoms (others masked out) |
| `A1-10@CA` | atom lists combine with ranges (apply to every residue in the range) |

Chain names are matched by longest prefix against the file's chain names
(disambiguates numeric names); atom selections are unioned per residue and a
whole-residue selection wins over atom lists. `select` overrides `chains`.

Author residue ids drive the |Δres_id| bond-path/linkage post-processing.
Hotspot tokens for a loaded motif sit at `motif_start + (index of the
residue within the concatenated selected chains)` where `motif_start` is the
total token count of the preceding entries.

### 4. Unindexed motif — dict with `unindexed_motif: True`

Routed through `RFD4Input.add_unindexed_motif` (in-context motif residues +
optionally indexed ligand tokens): required keys `pos` (float (m, 37, 3)),
`mask` (bool (m, 37)), `aa` (int (m,)); optional `fixed` (True = absolute
coordinate conditioning, False = internal-coordinate conditioning with
per-group random rotation), `residue_index` (int (m,)), `chain`
(`"new"` / None = fresh ids / a chain-iid value to join an existing chain),
`element` / `charge` / `bonds` (ligand chemistry for the block's atomized
tokens), `smiles` / `ccd` (decode the chemistry from RDKit/CCD — the decoded
atom order must match the block's `<A>` token order).

## Token layout

```
[ polymer chains in add order | ligand atom tokens | ... ]
```

`supports_local_latent` is True exactly on polymer tokens; atomized
(ligand) tokens carry `token_type` 3 and are never AE-emitted (their
coordinates come from the flow state). The compiled width is the total
token count; use `pad_to` / `pad_to_nearest` to round up.

## Editing after construction

Conditions can also be added to a compiled input through the adder API —
`add_dssp`, `add_bp`, `add_hotspot`, `add_rasa`, `add_hbond`, `add_pistack`,
`add_pication`, `add_saltbridge`, `add_hydrophobic`, `add_metal_coord`,
`add_indexed_motif`, `add_unindexed_motif` — all functional updates
returning a new `RFD4Input`. See the individual docstrings for window/
`where` semantics.
