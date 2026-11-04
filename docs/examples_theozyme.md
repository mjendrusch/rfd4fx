# Theozyme and motif-scaffolding recipes

Scaffolding a catalytic motif into a fresh fold: the motif's atoms are
conditioned (absolutely or via internal coordinates) and the surrounding
protein is generated. `from_chains.md` is the complete dictionary reference.

## 1. Theozyme with a ligand

The ligand is featurized automatically — from SMILES (RDKit graph) or a CCD
code — giving element/charge/bond conditioning and an atomized
`<A>`-token block:

```python
import jax
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(
    120,
    {"smiles": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",   # caffeine
     "auto_condition": True, "buried": True},  # interaction conditions from the ligand graph
)
```

Ligand conditioning is automatic ("auto_condition" semantics from the native
PLIP rules): `C_PISTACK` on aromatic-ring atoms, `C_HB` donor/acceptor roles
(acceptor = any N/O/S; donor = N/O with H), `C_SALT` charge signs. CCD codes
work the same way:

```python
inp = RFD4Input.from_chains(120, {"ccd": "ATP", "auto_condition": True})
# CCD codes without RDKit-sanitizable bond orders (e.g. HEM) work without
# auto_condition: inp = RFD4Input.from_chains(120, {"ccd": "HEM"})
```

## 2. Protein motif scaffolding (indexed)

A protein motif from a PDB, absolutely fixed; the rest is designed:

```python
inp = RFD4Input.from_chains(
    100,
    {"pdb": "scaffold.pdb", "select": ["A10-25@N,CA,C,O,CB"]},
)
```

`"drop_aa": True` reduces the motif to backbone-only conditioning (identity
stays unspecified); `"hotspot"`/`"auto_condition"` work on pdb motifs exactly
as for binders.

## 3. Internal-coordinate motifs (`fixed=False`)

With `"fixed": False` the motif's geometry is preserved but its **pose is
free** — the model decides where the motif sits in the design:

```python
inp = RFD4Input.from_chains(
    100,
    {"pdb": "scaffold.pdb", "select": ["A10-25"], "fixed": False, "group": 1},
)
```

## 4. Unindexed motifs

Unindexed motifs are in-context holes: geometry/identity conditioning without
chain indices — the trained shape for motif insertion. The pdb/cif route:

```python
inp = RFD4Input.from_chains(
    100,
    {"pdb": "scaffold.pdb", "unindexed_motif": True,
     "select": ["A10-25"], "fixed": False, "ordered": True},
)
```

- `"ordered": True` gives one index group per contiguous residue run;
  `False` (default) is per-token.
- `"target_chain"` pins the block's `chain_iid` (int or A-Z letter); absent,
  a fresh chain is minted.
- The raw-array route is `RFD4Input.add_unindexed_motif(pos, aa, mask, ...)`
  with `chain=None | int | "new"`, explicit `group=` on `fixed=False`, and
  optional ligand chemistry (`element`/`charge`/`bonds`).

## 5. Mixed indexed + unindexed motifs in one rigid body

Multiple motif pieces sharing a positive coordinate group are merged into one
rigid body by the library (single centroid + rotation over the union — native
`MaterializeInternalCoords` semantics):

```python
inp = RFD4Input.from_chains(
    100,
    {"pdb": "scaffold.pdb", "fixed": False,
     "motif": [
         {"select": ["B"]},                            # indexed part
         {"select": ["A"], "indexed": False},          # unindexed part
     ]})
```

Both pieces carry the same `C_CRDGRP`; the model sees one internally-
consistent frame regardless of how the pieces were authored.

## 6. Sampling

```python
batch = inp.pad_to_nearest(64).to_batch()
model = RFD4.from_pretrained()
sampler = RFD4Sampler(model.step(), model)
result = sampler.sample(batch, key=jax.random.key(0), recycle_schedule=THINKING_HIGH)
result.save_pdb("theozyme.pdb")
```

Note the token layout: from_chains compiles **all chains first, then all
ligands** (`token_type` runs `[0 × n_chain, 3 × n_lig]`) — scripts slicing
per-token arrays must respect these strides.
