# rfd4fx

**A flexcraft-compatible JAX/Equinox port of RFD4-Proteina — torch-free protein binder and motif design.**

`rfd4fx` reimplements the inference path of [RFD4-Proteina](https://huggingface.co/nvidia/RFD4-Proteina)
(flow-matching protein structure generation with indexed/unindexed motifs,
internal-coordinate conditioning and beam search) in pure JAX/Equinox.
Weights are converted **once** from the native PyTorch checkpoints; sampling,
search and export afterwards run without torch. Every module is validated by a
parity test-suite against the native implementation, and the input API mirrors
the native conditioning semantics (indexed/unindexed motifs, shared coordinate
groups, hotspot and interaction conditions), which makes `rfd4fx` directly
usable as a design backend inside [flexcraft](https://github.com/mjendrusch/flexcraft).

## Installation

Requires Python 3.12 (pinned by the native dependency stack). We recommend a
fresh [uv](https://docs.astral.sh/uv/) environment:

```bash
uv venv --python 3.12 .venv
uv pip install git+https://github.com/mjendrusch/rfd4fx.git
# …or, from a checkout:
uv pip install -e .
```

Optional extras:

```bash
uv pip install -e ".[convert]"     # torch for one-time checkpoint conversion (CPU wheels only)
uv pip install -e ".[test]"        # pytest for the parity suite
uv pip install -e ".[flexcraft]"   # flexcraft co-installation
```

> **torch is CPU-only.** The project pins torch to the PyTorch CPU wheel index
> (`[tool.uv]` in `pyproject.toml`), so CUDA wheels are never installed in any
> uv environment rooted here. For non-uv installs, install torch manually
> before the extras:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

### Checkpoint

Convert the native RFD4-Proteina checkpoint once (requires `.[convert]` and a
local checkout of the native repository):

```bash
python scripts/translate_rfd4.py \
    --checkpoint ~/.cache/huggingface/hub/models--nvidia--RFD4-Proteina/snapshots/<hash> \
    --out ~/.rfd4fx/models/rfd4_release
```

Sampling loads it torch-free:

```python
from rfd4fx.model import RFD4
model = RFD4.from_pretrained()   # resolves $RFD4FX_MODEL_CACHE / ~/.rfd4fx / local names
```

## Your first binder designs

`scripts/binder_minimal.py` conditions a fresh designable chain on a target
PDB motif with hotspot and hydrogen-bond constraints:

```bash
python scripts/binder_minimal.py \
    --in_path target.pdb \
    --target_selection "B" \
    --hotspots "B38;B97" \
    --auto_condition \
    --num_aa 72 --num_designs 5 \
    --out_path designs/
```

This writes `designs/binder_00.pdb … binder_04.pdb`, each a 72-residue binder
packed against the selected target chain with the requested hotspot residues
conditioned (`C_HOT` + donor/acceptor `C_HB`).

## API overview

The public surface is four objects:

| Object | Role |
|---|---|
| `RFD4Input` | Authoring-level conditioning input. `RFD4Input.from_chains(num_aa, *chain_dicts)` compiles designable chains, indexed/unindexed motifs, ligands and interaction conditions into model-ready channels (see `docs/from_chains.md` for the full dictionary format). |
| `RFD4` | The ported denoiser + autoencoder. `RFD4.from_pretrained()` loads a converted checkpoint; `RFD4.step()` is the jitted flow-matching update. |
| `RFD4Sampler` | Flow-matching sampling: `sampler = RFD4Sampler(model.step(), model)`; `result = sampler.sample(batch, key=..., recycle_schedule=THINKING_HIGH)`. |
| `RFD4Result` | Sampling output: tuple-compatible container with `save_pdb(path)` / `save_cif(path)`, `to_atom24()` / `to_atoms()`. |

and one refinement layer:

| Object | Role |
|---|---|
| `RFD4Search` | Trajectory-level beam search over the sampler: `RFD4Search(sampler).run(batch, evaluate=...)` with per-input beams, save/resume, and `RFD4SearchResults` scoring. |

A minimal end-to-end sample:

```python
import jax
from rfd4fx.flow_matching import RFD4Sampler, THINKING_HIGH
from rfd4fx.model import RFD4
from rfd4fx.rfd4_input import RFD4Input

inp = RFD4Input.from_chains(100, {"pdb": "motif.pdb", "select": ["A1-20"]})
batch = inp.pad_to_nearest(64).to_batch()
model = RFD4.from_pretrained()
sampler = RFD4Sampler(model.step(), model)
result = sampler.sample(batch, key=jax.random.key(0), recycle_schedule=THINKING_HIGH)
result.save_pdb("design.pdb")
```

## Documentation

- [`docs/from_chains.md`](docs/from_chains.md) — complete `from_chains` dictionary-format reference.
- [`docs/examples_binder.md`](docs/examples_binder.md) — binder design recipes (hotspots, interface motifs, partially-indexed motifs).
- [`docs/examples_theozyme.md`](docs/examples_theozyme.md) — theozyme/motif scaffolding with ligand autofeaturization.
- [`docs/examples_secondary_structure.md`](docs/examples_secondary_structure.md) — secondary-structure-conditioned generation.

## Testing

```bash
uv pip install -e ".[convert,test]"
pytest tests/module_parity/ -q
```

The parity suite compares every module against the native torch implementation
(205 tests); inference itself is torch-free.

## License

rfd4fx is licensed under [Apache 2.0](LICENSE). Parts of the code are ported
from [RFD4-Proteina](https://github.com/RosettaCommons/RFD4-Proteina)
(BSD 3-Clause, Copyright (c) 2026, Institute for Protein Design, University of
Washington); see [NOTICE](NOTICE) for the full derivation map and the retained
upstream license text ([LICENSES/BSD-3-Clause.txt](LICENSES/BSD-3-Clause.txt)).
License declarations follow the [REUSE](https://reuse.software/) specification
([REUSE.toml](REUSE.toml), license texts in [LICENSES/](LICENSES/)).
