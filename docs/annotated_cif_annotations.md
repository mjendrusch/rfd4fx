# RFD4 Annotated-CIF annotations → input features

Reference for the RFD4-Proteina annotated-CIF format: every annotation is a column in
the `_atom_site` loop (per-atom granularity), named after the condition/annotation's
`full_name`. Level suffixes: `_1_atom` (per-atom), `_1_residue` (per-residue),
`_1_token` (per-token), `_2_residue` (pairwise, residue × residue). Coordinate
conditions may alternatively use a separate `_condition_coordinate_1_atom` loop
(`idx0`/`val0`–`val2` rows). All annotations are optional; absent ⇒ schema default
("unspecified"). The same `full_name` strings double as the feature-tensor keys
produced by the featurization pipeline (`data["feats"]`) and by `collate_batch`.

Legend for the last columns: **feature key** = tensor in the featurized batch;
**FlowBatch field** = field in `rfd4fx.types.FlowBatch` (the jnp batch consumed
by the JAX port). Sequence-probability channels are stacked in canonical order
(20 aa, A/C/G/U, DA/DC/DG/DT) into `seq_probabilities[..., 0..27]`.

## 1. Design-task annotations (drive segment realization / indexing)

| CIF column (`_atom_site.*`) | Meaning | Feature key(s) | FlowBatch field |
|---|---|---|---|
| `annotation_expsegmin_1_residue` / `annotation_expsegmax_1_residue` | Expandable (de-novo) segment bounds; the segment is realized to `max−min+1` copies of the residue | (consumed by `RealizeExpandableSegments`; not a model feature) | — |
| `condition_index_1_residue` (bool) | Indexed/fixed token (in-context conditioning) vs generated | `is_in_context_conditioning` | `is_in_context_conditioning` |
| `condition_sequence_1_residue` (str; `<M>` = masked/unconditioned) | Fixed sequence per residue | `condition_sequence_1_residue` | `cond_seq` |
| `condition_sequence-probability-XXX_1_residue` (float, 28 channels) | Per-residue sequence probability biases | same (28 keys) | `seq_probabilities` |
| `condition_index-group_1_token` (int) | Index group id (groups tokens for bond-path/clash semantics) | `condition_index-group_1_token` | `idx_group` |

## 2. Structure conditions

| CIF column | Alias | Level / dtype | Feature key | FlowBatch field | Default |
|---|---|---|---|---|---|
| `condition_coordinate_1_atom` | `C_CRD` | atom, float×3 | `condition_coordinate_1_atom`, `coord_cond_mask` | `cond_coords` (+ sampler's `coord_cond_mask`) | NaN / False |
| `condition_InternalCoordinate_1_atom` | `C_INTCRD` | atom, float×3 | `condition_InternalCoordinate_1_atom` | `cond_internal_coords` | NaN |
| `condition_CoordinateGroup_1_atom` | `C_CRDGRP` | atom, int (−1 = ungrouped) | `condition_CoordinateGroup_1_atom` | `coord_groups` | −1 |
| `condition_ChainCentroid_1_atom` | `C_CENTROID` | atom, float×3 | `condition_ChainCentroid_1_atom` | `chain_centroid` | NaN |

## 3. Per-residue chemical / structural conditions

| CIF column | Alias | Level / values | Feature key | FlowBatch field | Default |
|---|---|---|---|---|---|
| `condition_secondary-structure_1_residue` | `C_SEC` | residue; SSEnum (0=none,1=helix,2=strand,3=loop) | `condition_secondary-structure_1_residue` | `secondary_structure` | 0 |
| `condition_cath-label_1_residue` | `C_CATH` | residue; CATH class id | `condition_cath-label_1_residue` | `cath_codes` [n,3] | −1 |
| `condition_na-ss-state_1_residue` | `C_NA_SS_STATE` | residue (nucleic acid SS state) | `na_ss_state`, `na_ss_state_mask` | `na_ss_state`, `na_ss_state_mask` | 0 / False |
| `condition_basepair_2_residue` | `C_BP` | residue × residue (nucleic acid pairing) | `na_ss_pair`, `na_ss_pair_mask` | `na_ss_pair`, `na_ss_pair_mask` | 0 / False |

## 4. Per-atom interaction conditions

| CIF column | Alias | Values | Feature key | FlowBatch field | Default |
|---|---|---|---|---|---|
| `condition_hotspot_1_atom` | `C_HOT` | bool | `condition_hotspot_1_atom` | `hotspot` | False |
| `condition_rasa-binned_1_atom` | `C_RAS` | 0=unspecified,1=buried,2=intermediate,3=exposed | `condition_rasa-binned_1_atom` | `rasa` | 0 |
| `condition_hbond_1_atom` | `C_HB` | 0=none,1=donor,2=acceptor,3=both | `condition_hbond_1_atom` | `hbond` | 0 |
| `condition_pistack_1_atom` | `C_PISTACK` | bool | `condition_pistack_1_atom` | `pistack` | False |
| `condition_pication_1_atom` | `C_PICATION` | 0=none,1=aromatic,2=cation,3=both | `condition_pication_1_atom` | `pication` | 0 |
| `condition_saltbridge_1_atom` | `C_SALT` | 0=none,1=positive,2=negative | `condition_saltbridge_1_atom` | `saltbridge` | 0 |
| `condition_hydrophobic_1_atom` | `C_HPHOB` | bool | `condition_hydrophobic_1_atom` | `hydrophobic` | False |
| `condition_metal-coord_1_atom` | `C_MCOORD` | 0=none, otherwise metal-coordination role id | `condition_metal-coord_1_atom` | `metal_coord` | 0 |

## 5. Bookkeeping annotations (authored or derived; not consumed by the model at inference)

| CIF column / source | Feature key | Notes |
|---|---|---|
| `loss_weight` | `loss_weight` | training loss weighting (default 1.0) |
| `insertion_source_idx` | `insertion_source_idx` | unindexed-token bookkeeping (training) |
| `annotation_src-component_1_atom` | — | provenance annotation (parsed, not featurized) |
| — | `residue_type`, `tok_idx`, `is_resolved_mask`, `tetrahedral_vertices`, `centroid_of_atoms_without_coordinate_conditioning` | derived by the pipeline; training-only or bookkeeping (verified inert at inference — see `scripts/generate_uncond_nativefeat_jax.py`) |

## 6. Derived (never authored; computed from the structure by the featurization pipeline)

| Feature key | FlowBatch field | Source |
|---|---|---|
| `coords` (Å) → `coords_nm` | `coords_nm`, `atom37_mask` | `ConvertToAtom37Format` (token pooling of atom coords; Å→nm in the assembler) |
| `token_type` | `token_type` | residue-name → unified 147-class encoding |
| `chain_iid`, `molecule_iid`, `molecule_entity_id` | same names | chain/molecule instance ids (random permutation is seedable; entity id only used via equality) |
| `token_bond_path_length`, `bond_mask`, `bond_order`, `is_aromatic`, `incident_bond_type_counts`, `canonical_polymer_ordering` | same names | `ComputeTokenLevelBondFeaturesLikeAF3` (bond graph BFS; signed linkage adjacency ±1) |
| `chirality`, `chiral_frame_atoms`, `chiral_slot_owner` | same names | `AddChiralityFeature` (packed tetrahedral geometry) |
| `element`, `ground_truth_element`, `charge` | same names | CCD lookups |
| `supports_local_latent` | `supports_local_latent` | residue ∈ AE latent alphabet |
| `padding_mask` | `padding_mask` | added at batch assembly (all-True for b=1 unpadded input) |

## Parity status

`scripts/verify_annotated_cif_parity.py` compares `annotated_cif_to_batch` against the
original torch featurization (`parse` → `build_design_validation_pipeline` →
`collate_batch`). Full-repository sweep (`--all`, 2026-10-23): **389/397 example files
× 70 model-relevant fields exact (27,230 comparisons, zero mismatches)** — every
benchmark family (monomer, motif indexed/unindexed/internal-coords, PPI all-atom and
bb-only, DNA/RNA binder, small-molecule binder, MCSA, sequence bias, ligand sets,
recent-CCD, structure prediction, adherence interactions) and tests/data. The 8
rejected files fail in the featurization semantics themselves (missing atom37 slot-1
reference atoms, batch-invariant violations, or `<M>` residues with atoms — e.g.
`tests/data/lencapavir_design_with_covale.cif`), identically in the original torch
pipeline; they are not valid RFD4 inference inputs.
