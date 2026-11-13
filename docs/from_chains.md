# `RFD4Input.from_chains` — usage guide

`RFD4Input.from_chains(*chains, np_seed=0, pad_to=None)` builds a compiled,
model-ready input from a sequence of chain / ligand / structure descriptions.
Featurization goes through `rfd4fx.spec.RFD4Spec`, so all semantics are the
parity-verified ones. This page is organized by **mode of use**; the full key
reference for each chain-dictionary type is in [Chain input
dictionaries](#chain-input-dictionaries), selection syntax in [Selection
syntax](#selection-syntax).

```python
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(
    80,                                     # de-novo 80-residue protein chain
    {"kind": "rna", "bp": "((((...))))"},   # RNA chain with dot-bracket C_BP
    {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
     "auto_condition": True, "buried": True},          # caffeine ligand
    {"pdb": "target.pdb", "chains": ["A"], "hotspot": 1},  # target motif
    np_seed=0,
)
batch = inp.pad_to_nearest(64).to_batch()
```

Top-level arguments: `np_seed` seeds the pipeline's chain/molecule-iid
permutations (default 0); `pad_to` pads to an exact total token count with
inert padded slots (`pad_to_nearest(m)` is the post-hoc equivalent).

## Modes of use

### 1. De-novo chains (unconditioned, dssp, dot-bracket, sequence)

An `int` entry is an unconditioned de-novo protein chain. Polymer dicts add
conditioning:

```python
RFD4Input.from_chains(100)                                    # unconditioned
RFD4Input.from_chains({"dssp": "HHHLLL"})                     # C_SEC
RFD4Input.from_chains({"kind": "rna", "bp": "((((...))))"})   # C_BP / C_NA_SS
RFD4Input.from_chains({"kind": "aa", "sequence": "MVK…"})     # fixed C_SEQ
```

`sequence` is a one-letter string (`"X"` for aa, `"N"`/`"X"` for dna/rna mark
unknown positions and leave C_SEQ unset there — those residues are designed).
Fixed-sequence chains compile through the direct featurizer (no native parity
route). Scripts: `scripts/generate_uncond_jax.py`, `scripts/generate_dssp_jax.py`,
`scripts/generate_rna_bp_jax.py`, `scripts/generate_rna_ligand_jax.py`.

### 2. Binder to a disordered target (sequence-only conditioning)

A polymer dict with `sequence` and **no coordinates** conditions the target by
identity only — its fold and pose are generated jointly with the binder.

```python
inp = RFD4Input.from_chains(
    72,
    {"kind": "aa", "sequence": "HESRIGLAEGLE",
     "hotspot": "1,5,6-10",        # 1-indexed residues; C_HOT on sidechains
     "auto_condition": True},      # C_HB roles on all sidechain donors/acceptors
)
```

Eligibility (which atoms each condition may touch) follows the native tables:
`h_bond` → donor/acceptor roles (1/2/3), `salt_bridge` → chargeable atoms
(1/2), `pi_stack` → aromatic-ring atoms, `pi_cation` → ring (1) / cation (2).
The explicit keys `h_bond`, `salt_bridge`, `pi_stack`, `pi_cation` take the
same residue±atom selection strings (e.g. `"2@NZ,OE1"`) and intersect the
selection with eligibility; combined with `auto_condition` they raise.
Script: `scripts/binder_disordered.py`.

### 3. Ligands

`smiles`/`ccd` dicts become atomized tokens (one per heavy atom) appended
after the polymer chains; positions are diffused unless `ligand_pos` anchors
them (C_CRD group 0).

```python
inp = RFD4Input.from_chains(
    80,
    {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O", "auto_condition": True},
    {"ccd": "ATP", "h_bond": "22,24,27", "pi_stack": "22-26"},  # adenine ring
)
```

`auto_condition` autofeaturizes all eligible atoms (C_PISTACK aromatic rings,
C_HB roles, C_SALT charge signs); the explicit interaction keys take 1-based
atom-index strings in SMILES/CCD order and condition only the eligible subset
(`auto_condition` + explicit keys raise). Script: `scripts/generate_ligand_jax.py`,
`scripts/generate_random_ligand_dssp.py`.

### 4. Structure motifs (pdb/cif)

A `pdb`/`cif` dict loads a structure (gemmi, first model) as motif
conditioning. Two routes:

```python
# indexed (default): fixed motif chains inside the generated system
inp = RFD4Input.from_chains(
    80, {"pdb": "target.pdb", "select": ["A"], "hotspot": "A45;A47@N,CA,C",
         "auto_condition": True})

# unindexed: appended in-context motif block (free pose w/ internal coords)
inp = RFD4Input.from_chains(
    80, {"pdb": "target.pdb", "select": ["A17-131"], "unindexed_motif": True,
         "fixed": False, "ordered": True})

# interface / surface motifs
inp = RFD4Input.from_chains(
    80, {"pdb": "complex.pdb",
         "motif": [{"select": ["A"], "indexed": False, "interface_to": "B",
                    "cutoff": 4.5, "drop_backbone": True, "ordered": True},
                   {"select": ["B"]}]})
```

Key semantics: `fixed` (True = absolute C_CRD in group `group`; False =
internal coordinates C_INTCRD/C_CRDGRP — one rigid body per positive group,
auto-merged across motif pieces), `drop_aa` (backbone-only), `drop_backbone`
(sidechains-only), `interface_to` + `cutoff` (interface residues),
`neighbours` (extend the selection to residues with a sidechain heavy atom
within `cutoff`) + `surface` (keep only Shrake–Rupley-surface neighbours),
`hotspot` (str selection → C_HOT; non-str → atom-level values),
`auto_condition` (C_HB on hotspot atoms), and the explicit `h_bond`,
`salt_bridge`, `pi_stack`, `pi_cation` keys (lists of selection strings,
subset to eligible atoms). Scripts: `scripts/binder_minimal.py`,
`scripts/generate_motif_scaffold_jax.py`, `scripts/generate_interface_motif_jax.py`,
`scripts/generate_surface_motif_jax.py`, `scripts/side_by_side_1qjg.py`.

### 5. Partially-indexed motifs

A `motif` list splits one structure into sub-specs with per-piece routing:

```python
inp = RFD4Input.from_chains(
    72, {"pdb": "complex.pdb", "fixed": False,
         "motif": [{"select": ["B"]},                       # indexed backbone
                   {"select": ["A"], "indexed": False,      # unindexed sidechains
                    "drop_backbone": True, "ordered": True,
                    "target_chain": "A"}]})
```

Sub-specs inherit the dict-level `fixed`/`group`/`hotspot`/`auto_condition`;
pieces with the same positive coordinate group merge into one rigid body at
`to_batch` time. Script: `scripts/generate_interface_motif_jax.py`.

## Chain input dictionaries

Four dict families are recognized (dispatched by their presence keys, in this
order): `motif` sub-spec expansion → `pdb`/`cif` structure motifs →
`unindexed_motif` raw blocks → `smiles`/`ccd` ligands → polymer dicts.

### Polymer dict

| Key | Type / values | Meaning |
|---|---|---|
| `num_residues` | int | Chain length; inferred when absent (`length` → `sequence` → `dssp` → `bp` → `pos` → `residue_index`/`aa` → atom-level values). |
| `kind` | `"aa"`, `"dna"`, `"rna"` (any case) | Vocabulary group → token_type 0/2/1. Default `"aa"`. |
| `sequence` | str | One-letter codes → C_SEQ per residue; ambiguity codes leave C_SEQ unset (see mode 2). |
| `dssp` | str / (n,) int | Protein C_SEC (`H`→1, `E`→2, `L`→3). Conflicts with `bp`. |
| `bp` | str | Dot-bracket C_NA_SS_STATE + C_BP; requires `kind` ∈ {dna, rna}; NA chains concatenate at compile time (pairs may cross chains). |
| `pos` + `mask` | (n, 37, 3) + (n, 37) | Coordinate conditioning: materialized slots, C_CRD group `group` (default 0). |
| `group` | int | Coordinate-group override for `pos`. |
| `chain_index` | int (n,) | Split the block into separate chains by unique value. |
| `residue_index` | int (n,) | Bond path = \|Δresidue_index\|; linkage at distance 1. |
| `hotspot` | str (residue list) or atom-level values | Str: 1-indexed residue list (`"1,5,6-10"`, optional `@ATOM`) → C_HOT on sidechain slots; requires `sequence`. Non-str: atom-level values (residue-level (n,) shape requires `mask`). |
| `auto_condition` | bool | C_HB roles on all sidechain donors/acceptors; requires `sequence`. |
| `h_bond`, `salt_bridge`, `pi_stack`, `pi_cation` | str | Residue±atom selections → per-atom C_HB/C_SALTBRIDGE/C_PISTACK/C_PICATION on the eligible subset. |
| `bonds` | list of `{"atoms": [sel, sel], "order": 1\|2\|3}` | Author covalent bonds: selectors are hotspot-syntax atom selections (`"1@SG"`, `@` required, exactly one atom each, different residues). Sets `bond_mask`/`bond_order` (symmetric), `incident_bond_type_counts` at the endpoints, and recomputes `token_bond_path_length` by BFS (ring distances shrink through closures). Same-token bonds set the counts only. |
| `cyclic` | bool | True → head-to-tail C→N closure of the chain with **backbone-linkage semantics**: `canonical_polymer_ordering[first, last] = ±1`, suppressed from `bond_mask`, ring hop counts, incident counts on both closure atoms. |
| `rasa`, `hbond`, `pistack`, `pication`, `saltbridge`, `hydrophobic`, `metal_coord` | scalar / (n,) / (n, 37), or `{"values", "aa", "where"}` | Atom-level conditions at materialized slots (domains: rasa 0–3, hbond 0–3, saltbridge 0–2, rest boolean). |

### Ligand dict

| Key | Type | Meaning |
|---|---|---|
| `smiles` / `ccd` | str | Exactly one; RDKit SMILES or cached-CCD chemistry. |
| `ligand_pos` | (num_atoms, 3) Å | Coordinate-condition every atom (C_CRD group 0). |
| `res_name` | str | Residue code in PDB/mmCIF output. |
| `auto_condition` | bool | Autofeaturize C_PISTACK / C_HB / C_SALT on all eligible atoms. |
| `h_bond`, `salt_bridge`, `pi_stack`, `pi_cation` | str | 1-based atom-index selections ("1,3-5") → eligible subset only. |
| `buried` | bool | C_RAS 1 (buried) on every atom. |

### Structure-motif dict (`pdb` / `cif`)

| Key | Type | Meaning |
|---|---|---|
| `pdb` / `cif` | path | Structure file (gemmi, first model). Exactly one. |
| `chains` / `select` | list of names / entries | Chain filter; `select` (grammar below) overrides `chains`. |
| `unindexed_motif` | bool | Route through the appended unindexed block instead of fresh indexed chains. |
| `fixed` / `group` | bool / int | Absolute (C_CRD, group 0) vs internal (C_INTCRD/C_CRDGRP) conditioning. |
| `drop_aa` / `drop_backbone` | bool | Backbone-only vs sidechains-only motif. |
| `motif` | list of sub-specs | Per-piece `{select, indexed, drop_backbone, ordered}`; shared groups merge (see mode 5). |
| `interface_to` / `cutoff` | str / float | Interface-residue restriction (cutoff default 4.5). |
| `neighbours` / `surface` | bool | Extend to sidechain-neighbours within `cutoff`; filter to surface neighbours. |
| `hotspot` | str or atom-level values | C_HOT on the selection (str, out-of-motif raises) / atom-level overlay. |
| `auto_condition` | bool | C_HB on hotspot residues' atoms. |
| `h_bond`, `salt_bridge`, `pi_stack`, `pi_cation` | list of str | Selection strings → per-atom interaction conditions on the eligible subset. |
| `bonds` | list of `{"atoms": [sel, sel], "order": 1\|2\|3}` | Chain-grammar selectors (`"A48@NZ"`, one motif residue+atom each) → file-absent covalent bonds (same features as the polymer-dict key). |
| `cyclic` | bool | Head-to-tail C→N closure of each selected chain (backbone-linkage semantics). |
| `ligands` | {resname: SMILES} | Chemistry for HETATM residues (bonds/charges from SMILES graph matching; CCD fallback). |
| `ordered` | bool (unindexed) | One index group per contiguous same-chain run vs per-token. |
| `target_chain` | int / "A"–"Z" | Pin the unindexed motif residues' chain identity (join an existing chain). |

### Raw unindexed-motif dict (`unindexed_motif: True`)

Direct `RFD4Input.add_unindexed_motif` route (no structure file): required
`pos` (m, 37, 3), `mask` (m, 37), `aa` (m,); optional `fixed`, `residue_index`,
`chain` (None/`"new"` = fresh ids, or a chain-iid value to join),
`element`/`charge`/`bonds` or `smiles`/`ccd` for atomized chemistry.

## Selection syntax

**Structure motifs (pdb/cif dicts)** — entries name chains by longest-prefix
match against the file's chain names:

| Entry | Selects |
|---|---|
| `A` | whole chain A |
| `A1` | residue A1 (author numbering), all present atoms |
| `A1-10` | residues A1–A10 (inclusive) |
| `A1@N,CA,C` | residue A1, only the named atoms |
| `A1-10@CA` | atom lists combine with ranges |

Atom selections union per residue; a whole-residue entry wins over atom lists.

**Sequence-only chains** — the same grammar without the chain prefix, indexed
by position in `sequence` (1-based): `"1,5,6-10"`, `"2@NZ,OE1"`. Atom names are
validated against the residue's encoding slots; unknown residues (X/N) have no
slots and never match.

**Where each key applies**:

| Key | Structure motifs | Sequence-only chains | Ligands |
|---|---|---|---|
| `hotspot` | selection strings (`A45@N,CA`) | residue list (`"1,5"`) | — |
| `h_bond` / `salt_bridge` / `pi_stack` / `pi_cation` | selection strings | residue±atom strings | atom-index strings |
| `auto_condition` | C_HB on hotspot atoms | C_HB on all sidechains | C_HB/C_PISTACK/C_SALT on all atoms |

Eligibility subsetting (identical rules everywhere): `h_bond` keeps only
donors/acceptors (value = role), `salt_bridge` only charged atoms
(1=positive, 2=negative), `pi_stack` only aromatic-ring atoms, `pi_cation`
ring (1) and cation (2) atoms. Selections naming no eligible atoms are warned
and skipped; out-of-motif selections raise.

## Token layout

```
[ polymer chains in add order | ligand atom tokens | ... ]
```

`supports_local_latent` is True exactly on polymer tokens; atomized (ligand)
tokens carry `token_type` 3 and are never AE-emitted. Use `pad_to` /
`pad_to_nearest` to round the compiled width up.

## Editing after construction

Every condition can also be added to a compiled input through functional
adder methods — each returns a **new** `RFD4Input`, leaving the original
untouched. Atom-level adders share a window convention: `start` selects the
first token of the edit window, and `where` selects positions within it —
`True` (whole window), a bool `(k,)` array (all 37 slots of the True
residues), or a bool `(k, 37)` array (specific atoms). `aa` (int (k,)) sets
per-residue identity (C_SEQ) at touched residues; `0` = leave unspecified.

```python
inp = RFD4Input.from_chains(80, {"pdb": "data/8znl.pdb", "select": ["A48-52"]})
# tokens: [0..79] designable chain, [80..84] motif (A48-A52)

# residue-level: C_SEC over a window of the designable chain
inp = inp.add_dssp("HHHLLL", start=10)

# atom-level: C_HOT on specific atoms of the motif (per-atom `where`)
import numpy as np
where = np.zeros((5, 37), bool); where[0, 3] = True    # A48's CB (atom37 slot 3)
inp = inp.add_hotspot(1, aa=None, where=where, start=80)

# atom-level with roles: C_HB donors on the motif (A48 = ARG: NE, NH1, NH2)
hb = np.zeros((5, 37), np.int32)
hb[0, [23, 29, 30]] = 1
inp = inp.add_hbond(hb, aa=None, where=hb > 0, start=80)

# scalar-per-residue value: C_RAS exposed on a window (k inferred from values)
inp = inp.add_rasa(np.full(80, 3), aa=None, where=True, start=0)

# other interaction channels: identical call shapes
inp = inp.add_pistack(True, aa=None, where=where, start=80)
inp = inp.add_saltbridge(1, aa=None, where=where, start=80)

# nucleic acids: C_BP / C_NA_SS over a token window
# inp = inp.add_bp("((((...))))", start=0)
```

Whole-motif edits use the system-wide adders. Coordinates are authored in
**Ångström** (converted internally); `aa` carries RFD4 encoding indices
(`ENC.token_to_idx`), `0` = `<M>` leaves the identity untouched.

```python
import numpy as np
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as ENC

# add_indexed_motif: overlay coordinate conditioning over the WHOLE compiled
# system (pos/aa/mask have one row per token; `where` selects per token/atom;
# positions with NaN are ignored)
n = int(np.asarray(inp.padding_mask).sum())
pos = np.full((n, 37, 3), np.nan, np.float32)   # Å
mask = np.zeros((n, 37), bool)
aa = np.zeros(n, np.int32)
pos[10, 1] = [3.7, 1.2, -0.4]                   # pin token 10's CA
mask[10, 1] = True
aa[10] = ENC.token_to_idx["ALA"]

inp = inp.add_indexed_motif(pos, aa, mask)      # absolute: C_CRD, group 0
inp = inp.add_indexed_motif(pos, aa, mask, group=1, fixed=False)
# fixed=False: internal coordinates — the payload is centered and randomly
# rotated per group (free global pose); the group must be positive

# add_unindexed_motif: APPEND an in-context motif block (no `where`; tokens
# are added after the existing ones)
m = 3
pos = np.full((m, 37, 3), np.nan, np.float32)
pos[:, 1] = [[0.0, 0.0, 0.0], [0.38, 0.0, 0.0], [0.76, 0.0, 0.0]]  # CA trace
mask = np.zeros((m, 37), bool); mask[:, 1] = True
aa = np.array([ENC.token_to_idx[r] for r in ("ALA", "GLY", "SER")])

inp = inp.add_unindexed_motif(pos, aa, mask, fixed=False, ordered=True,
                              residue_index=np.array([1, 2, 3]),
                              chain_index=np.ones(3, np.int32))
# ordered=True groups contiguous same-chain runs (requires residue_index +
# chain_index); fixed=False gives the block its own internal-coordinate frame

# anchor an appended block and fuse it into an existing chain (target_chain
# takes an int chain-iid or a letter index: "A" = chain 0)
inp = inp.add_unindexed_motif(pos, aa, mask, fixed=True, target_chain="A")
```

Atomized small-molecule blocks use the same adder with `aa` = the atomized
`<A>` index and `element`/`charge`/`bonds` (or `smiles`/`ccd`) for the
chemistry — see the `add_unindexed_motif` docstring for ligand routing and
the full parameter reference.
