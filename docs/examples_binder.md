# Binder design recipes

All examples use the public API (`RFD4Input` → `RFD4Sampler` → `RFD4Result`)
and assume a converted checkpoint reachable by `RFD4.from_pretrained()`. See
`from_chains.md` for the complete dictionary-format reference.

## 1. Minimal binder

A fresh 72-residue chain conditioned on an indexed target motif:

```python
import jax
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(
    72,
    {"pdb": "target.pdb", "select": ["B"]},     # whole chain B, absolutely fixed
)
batch = inp.pad_to_nearest(64).to_batch()
model = RFD4.from_pretrained()
sampler = RFD4Sampler(model.step(), model)
result = sampler.sample(batch, key=jax.random.key(0), recycle_schedule=THINKING_HIGH)
result.save_pdb("binder.pdb")
```

## 2. Hotspots and hydrogen-bond conditioning

`"hotspot"` (a selection-grammar string) sets `C_HOT` on the chosen
residues/atoms; `"auto_condition"` adds `C_HB` donor/acceptor roles from the
native tables (1 = donor, 2 = acceptor, 3 = both):

```python
inp = RFD4Input.from_chains(
    72,
    {"pdb": "target.pdb", "select": ["B"],
     "hotspot": "B38;B97@N,CA,C",     # residue 38 fully; residue 97 backbone only
     "auto_condition": True},
)
```

Out-of-motif hotspot entries raise a `ValueError` — hotspots must reference
residues that were actually loaded.

## 3. Sub-selecting the target

`"select"` entries are `CHAIN`, `CHAINRES`, `CHAINRES-RES` and
`CHAINRES@ATOM,ATOM` (longest-prefix chain matching, several entries per dict):

```python
{"pdb": "target.pdb", "select": ["B20-90", "B120@N,CA,C,O"]}
```

## 4. Interface-motif scaffolding (partially-indexed motifs)

Redesign a binder around its own interface: the target is an **indexed**
motif, the donor chain's interface sidechains an **unindexed** motif, and both
share **one coordinate group** (one rigid body). Distance selection
(`"interface_to"` + `"cutoff"`) and backbone dropping (`"drop_backbone"`) are
built in:

```python
inp = RFD4Input.from_chains(
    72,
    {"pdb": "complex.pdb", "fixed": True,
     "motif": [
         {"select": ["B"]},                                   # indexed partner
         {"select": ["A"], "indexed": False,                  # interface sidechains
          "interface_to": "B", "cutoff": 4.5,
          "drop_backbone": True, "ordered": True,
          "target_chain": "A"},                               # join chain iid 0
     ]})
```

- `"fixed": False` conditions both motifs through **internal coordinates**
  (`C_INTCRD`/`C_CRDGRP`): the geometry is preserved, the pose is free.
  Sub-specs default to shared group 1; the library auto-merges all pieces of
  a group into one rigid body (single centroid + rotation over the union).
- `"fixed": True` (per dict) pins the motifs absolutely (`C_CRD`, group 0) —
  coordinates are then compared in the input frame.
- `"target_chain"` pins the unindexed tokens' `chain_iid` (int, or `"A"`-style
  letter index); molecule/entity ids follow the resolved chain.

A ready-made CLI for this recipe lives at `scripts/generate_interface_motif_jax.py`
(`--fixed`, `--cutoff`, `--save_input`).

## 5. Batched designs

Design several sequences in one process (per-design keys, shared compile):

```python
batch = inp.pad_to_nearest(64).to_batch()
for i in range(5):
    result = sampler.sample(batch, key=jax.random.fold_in(jax.random.key(0), i),
                            recycle_schedule=THINKING_HIGH)
    result.save_pdb(f"designs/binder_{i:02d}.pdb")
```

`RFD4Result` is tuple-compatible (`x, x_1, data, dec = result`) and carries
`is_single_sample`, `to_atom24()` and `to_atoms()` for programmatic access.

## 6. Beam search over trajectories

`RFD4Search` refines designs by branching partial ODE continuations and
scoring full rollouts (or clean predictions) with a user callable:

```python
from rfd4fx.search import RFD4Search

def evaluate(result):
    """Score one decoded RFD4Result; returning stop=True retires the beam."""
    metrics = {"clashes": count_clashes(result.to_atoms())}
    return {"score": -metrics["clashes"], "pass": metrics["clashes"] < 5,
            "stop": False, "metrics": metrics}

search = RFD4Search(model.step(), model, evaluate, n_branch=4, beam_width=4)
results = search.search(batch, key=jax.random.key(0))   # -> RFD4SearchResults
best = results[0]            # score-descending; an RFD4Result
best.save_pdb("best.pdb")
```

`batch_evaluate_vmap`/`batch_evaluate_map` (passed as the
`batch_evaluate` constructor argument) vectorize scoring across beams;
`save_states_dir`/`resume_from` on `search()` support interruptible runs.
