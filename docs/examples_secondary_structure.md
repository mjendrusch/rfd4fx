# Secondary-structure-conditioned generation

`RFD4Input.add_dssp` conditions the secondary-structure channel (`C_SEC`) of
a design: `"H"` (helix, 1), `"E"` (strand, 2), `"L"` (loop, 3), `"X"/"_"` /
unspecified (0). The parser is strictly 3-state — unsupported characters
raise.

## 1. A designed helical bundle

```python
import jax
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(120)
inp = inp.add_dssp("H" * 18 + "L" * 4, start=0) \
         .add_dssp("H" * 18 + "L" * 4, start=22) \
         .add_dssp("H" * 18 + "L" * 4, start=44)
```

`add_dssp(value, start, where)` writes the window `start:start + len(value)`;
`where` optionally selects a subset of residues (bool array).

## 2. A mixed alpha/beta fold

```python
dssp = "EEEEELHHHHHHLLLEEEEEELHHHHHHL"
inp = RFD4Input.from_chains(len(dssp)).add_dssp(dssp)
```

## 3. Secondary structure + motif conditioning

`add_dssp` composes with every other conditioner — e.g. constrain a motif and
the fold around it:

```python
inp = RFD4Input.from_chains(
    120,
    {"pdb": "motif.pdb", "select": ["A5-15"], "fixed": False, "group": 1},
).add_dssp("E" * 11, start=4) \
 .add_dssp("H" * 20, start=20)
```

## 4. Array-valued conditioning

Already-compiled integer maps can be written directly (values follow the
`SS_NONE=0, SS_HELIX=1, SS_STRAND=2, SS_LOOP=3` enum):

```python
import numpy as np
sec = np.zeros(120, np.int32)
sec[10:30] = 1        # helix
sec[50:70] = 2        # strand
inp = inp.add_dssp(sec)
```

## 5. Nucleic-acid base-pair motifs

RNA/DNA chains accept `kind` (`"aa" | "rna" | "dna"`) and dot-bracket base
pairs (`C_BP`/`C_NA_SS_STATE`); cross-chain pairs are allowed:

```python
inp = RFD4Input.from_chains(
    40,
    {"kind": "rna", "sequence": "GGGAAACCCGGGAAACCCGGGAAACCCGGGAAACCCGGG",
     "bp": "((((.....))))((((.....))))((((.....))))"},
)
```

## 6. Sampling and inspection

```python
batch = inp.pad_to_nearest(64).to_batch()
model = RFD4.from_pretrained()
sampler = RFD4Sampler(model.step(), model)
result = sampler.sample(batch, key=jax.random.key(0), recycle_schedule=THINKING_HIGH)
result.save_pdb("design.pdb")
```

The generated structures can be checked against the conditioning with
`scripts/check_backbone_geometry.py` and the DSSP-conditioned generation
campaign script `scripts/generate_dssp_jax.py`.
