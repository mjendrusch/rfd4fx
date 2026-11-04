"""Parity tests for the active sequence/pair feature subset (random weights)."""

from conftest import run_feature_parity

import rfproteina.nn.feature_factory as ff

PAD = ("bool", (8,))
COMMON = {"padding_mask": PAD, "x_t": {"bb_ca": ("float", (8, 3))}}

# torch kwargs shared by the feature constructors used below
KWARGS = dict(
    token_type_emb_dim=16,
    cath_emb_dim=8,
    latent_dim=8,
    latent_z_dim=8,
    seq_sep_dim=32,
    xt_pair_dist_dim=15,
    xt_pair_dist_min=0.1,
    xt_pair_dist_max=3.0,
    t_emb_dim=32,
    data_mode_use="bb_ca",
    recycle={"grad": "last"},
    n_residue_types=32,
    charge_emb_dim=16,
    max_atomic_num=64,
    element_emb_dim=16,
    noise_align_chirality=False,
    noise_align_chirality_translate=False,
    noise_align_internal_coords=False,
    noise_align_internal_coords_translate=False,
)

COORD_SPECS = {
    **COMMON,
    "atom37_mask": ("bool", (8, 37)),
    "coords_nm": ("float", (8, 37, 3)),
    "charge": ("enum", (8, 37)),
    "element": ("enum", (8, 37)),
    "ground_truth_element": ("enum", (8, 37)),
    "incident_bond_type_counts": ("float", (8, 37, 10)),
}


def _run(torch_cls, specs, name, **ctor):
    torch_feat = torch_cls(**{**KWARGS, **ctor})
    from rfd4fx.backend import from_torch, register_base_types
    from rfd4fx.nn import composites, features, layers

    register_base_types()
    layers.register()
    composites.register()
    features.register()
    jax_feat = from_torch(torch_feat)
    mean, std, gmax = run_feature_parity(torch_feat, jax_feat, specs)
    print(f"{name}: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_stateless_seq_features():
    _run(ff.XtBBCASeqFeat, {**COMMON, "x_t": {"bb_ca": ("float", (8, 3))}}, "xt_bb_ca")
    _run(ff.XtLocalLatentsSeqFeat, {**COMMON, "x_t": {"local_latents": ("float", (8, 8))}}, "xt_local_latents")
    _run(ff.CaCoorsNanometersSeqFeat, {**COMMON, "ca_coors_nm": ("float", (8, 3))}, "ca_coors_nm")
    _run(ff.LatentVariableSeqFeat, {**COMMON, "z_latent": ("float", (8, 8))}, "z_latent")
    _run(ff.ChainIdxSeqFeat, {**COMMON, "chain_iid": ("float", (8,))}, "chain_idx")


def test_embedding_seq_features():
    specs = {**COMMON, "token_type": ("enum", (8,))}
    _run(ff.TokenTypeSeqFeat, specs, "token_type")

    # sequence probabilities: 28 condition channels with NaN holes
    specs = {**COMMON, "condition_sequence-probability-ALA": ("const", 1)}
    _run(ff.SequenceProbabilitiesSeqFeat, specs, "sequence_probabilities")

    _run(ff.SecondaryStructureSeqFeat, {**COMMON, "condition_secondary-structure_1_residue": ("enum", (8,))}, "secondary_structure")


def test_coord_features():
    _run(ff.Atom37NanometersCoorsSeqFeat, COORD_SPECS, "a37coors_abs")
    _run(ff.Atom37NanometersCoorsSeqFeat, COORD_SPECS, "a37coors_rel", rel=True)
    # conditional coords: C_CRD (finite = conditioned, NaN = not)
    _run(
        ff.ConditionalAtom37NanometersCoorsSeqFeat,
        {**COORD_SPECS, "condition_coordinate_1_atom": ("coord", (8, 37, 3))},
        "cond_a37coors",
    )
    # internal coords without alignment (release config path)
    _run(
        ff.ConditionalAtom37NanometersInternalCoorsSeqFeat,
        {**COORD_SPECS, "condition_InternalCoordinate_1_atom": ("coord", (8, 37, 3)), "condition_CoordinateGroup_1_atom": ("int", (8, 37), dict(lo=-1, hi=4))},
        "cond_internal",
    )
    _run(ff.ChainCentroidXyzSeqFeat, {**COMMON, "condition_ChainCentroid_1_atom": ("coord", (8, 3))}, "chain_centroid")


def test_annotation_features():
    _run(ff.RasaBinnedSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_rasa-binned_1_atom": ("enum", (8, 37))}, "rasa")
    _run(ff.ModernizedHydrogenBondSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_hbond_1_atom": ("enum", (8, 37))}, "hbond")
    _run(ff.PiStackingSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_pistack_1_atom": ("bool", (8, 37))}, "pistack")
    _run(ff.PiCationSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_pication_1_atom": ("enum", (8, 37))}, "pication")
    _run(ff.SaltBridgeSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_saltbridge_1_atom": ("enum", (8, 37))}, "saltbridge")
    _run(ff.HydrophobicSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_hydrophobic_1_atom": ("bool", (8, 37))}, "hydrophobic")
    _run(ff.MetalCoordinationSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_metal-coord_1_atom": ("enum", (8, 37))}, "metal_coord")
    _run(ff.HotspotSeqFeat, {**COMMON, "atom37_mask": ("bool", (8, 37)), "condition_hotspot_1_atom": ("bool", (8, 37))}, "hotspot")
    _run(
        ff.NucleicSecondaryStructureStateSeqFeat,
        {**COMMON, "na_ss_state": ("float", (8,)), "na_ss_state_mask": ("float", (8,))},
        "na_ss_state",
    )


def test_atomized_token_features():
    _run(
        ff.AtomChargeSeqFeat,
        {**COMMON, "charge": ("enum", (8, 37)), "token_type": ("enum", (8,))},
        "atom_charge",
    )
    _run(
        ff.AtomicElementSeqFeat,
        {**COMMON, "element": ("enum", (8, 37)), "token_type": ("enum", (8,))},
        "atomic_element",
    )
    _run(
        ff.Atom37ElementChargeEmbeddingSeqFeat,
        {**COORD_SPECS, "charge": ("int", (8, 37), dict(lo=0, hi=11)), "ground_truth_element": ("int", (8, 37), dict(lo=0, hi=64))},
        "a37_element_charge",
    )


def test_time_and_recycle_features():
    _run(ff.TimeEmbeddingBatchFeat, {**COMMON, "t": {"bb_ca": ("float", ()), "local_latents": ("float", ())}}, "time_emb_bb_ca")
    _run(
        ff.TimeEmbeddingBatchFeat,
        {**COMMON, "t": {"bb_ca": ("float", ()), "local_latents": ("float", ())}},
        "time_emb_local_latents",
        data_mode_use="local_latents",
    )
    _run(ff.RecycleRemainingBatchFeat, {**COMMON, "x_t": {"bb_ca": ("float", (8, 3))}, "recycle_remaining": ("const", 3)}, "recycle_remaining=3")
    _run(ff.RecycleRemainingBatchFeat, {**COMMON, "x_t": {"bb_ca": ("float", (8, 3))}}, "recycle_remaining=None")


def test_pair_features():
    x_t = {"bb_ca": ("float", (8, 3))}
    _run(ff.XtBBCAPairwiseDistancesPairFeat, {**COMMON, "x_t": x_t}, "xt_pair_dists")
    _run(ff.TokenBondPathLengthPairFeat, {**COMMON, "token_bond_path_length": ("int", (8, 8), dict(lo=0, hi=40))}, "bond_path")
    _run(ff.MoleculeIdxPairFeat, {**COMMON, "molecule_iid": ("int", (8,), dict(lo=0, hi=4))}, "molecule_idx")
    _run(ff.MoleculeEntityPairFeat, {**COMMON, "molecule_entity_id": ("int", (8,), dict(lo=0, hi=4))}, "molecule_entity")
    _run(ff.IndexGroupPairFeat, {**COMMON, "condition_index-group_1_token": ("int", (8,), dict(lo=0, hi=4))}, "index_group")
    _run(ff.BondMaskPairFeat, {**COMMON, "bond_mask": ("bool", (8, 8))}, "bond_mask")
    _run(ff.BondOrderPairFeat, {**COMMON, "bond_order": ("int", (8, 8), dict(lo=0, hi=5))}, "bond_order")
    _run(ff.IsAromaticPairFeat, {**COMMON, "is_aromatic": ("bool", (8, 8))}, "is_aromatic")
    _run(ff.CanonicalPolymerOrderingPairFeat, {**COMMON, "canonical_polymer_ordering": ("int", (8, 8), dict(lo=-1, hi=2))}, "polymer_ordering")
    _run(ff.NucleicSecondaryStructurePairFeat, {**COMMON, "na_ss_pair": ("float", (8, 8)), "na_ss_pair_mask": ("float", (8, 8))}, "na_ss_pair")
    _run(ff.CaCoorsNanometersPairwiseDistancesPairFeat, {**COMMON, "ca_coors_nm": ("float", (8, 3))}, "ca_pair_dists")
    _run(ff.BackbonePairDistancesNanometerPairFeat, COORD_SPECS, "bb_pair_dists")


def test_conditional_residue_type():
    # <M> mask token is index 0; unconditioned tokens map to row 0 then zeroed.
    _run(ff.ConditionalResidueTypeSeqFeat, {**COMMON, "condition_sequence_1_residue": ("enum", (8,))}, "cond_residue_type")


def test_fold_embedding():
    _run(ff.FoldEmbeddingSeqFeat, {**COMMON, "condition_cath-label_1_residue": ("int", (8, 3), dict(lo=-1, hi=5))}, "cath_code")
