"""Native 1QJG run v2: full_simulation with the sequence readout saved, then a
proper PDB (designed sequence via the token-type-restricted argmax, real atom
names/elements per residue type, motif reference at input coords)."""

import pathlib
import numpy as np
import torch
from functools import partial
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rfproteina.datapipes.pipeline import build_design_validation_pipeline
from rfproteina.inference_utils.datasets import DesignProblemDataset
from rfproteina.datapipes.collate import collate_batch
from rfproteina.proteina import Proteina

OUT = pathlib.Path("/Users/mjendrusch/repos/rfd4j/rfd4-test-native/1qjg_run")
SNAP = pathlib.Path("/Users/mjendrusch/repos/rfd4j/.hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502")
model = Proteina.from_pretrained(SNAP); model.eval(); model.cpu()
model.inf_cfg = OmegaConf.create({"condition_on_recycle_remaining": True,
                                  "args": {"guidance_w": 1.0,
                                           "condition_decoder_on_sequence_prediction": False}})

p = pathlib.Path("data/benchmarks/motif_scaffolding/motif_unindexed/1QJG_AA_NATIVE.cif.zst")
pipe = build_design_validation_pipeline(metrics_tags=None,
                                        expandable_segment_rng=np.random.default_rng(115))
ds = DesignProblemDataset([str(p)], transform=pipe)
batch = next(iter(DataLoader(ds, batch_size=1, shuffle=False, collate_fn=partial(
    collate_batch, schema="rfproteina.datapipes.schema.proteina.ProteinaDataSample",
    fixed_dimensions=None))))
batch = model.preprocess_batch_for_generation(batch)
batch = model.add_clean_samples(batch)
n = batch["padding_mask"].shape[1]

recycle_depths = sum([[d] * w for w, d in [(5, 6), (10, 5), (5, 4), (10, 3), (20, 2), (10, 1)]], [])
sampling_model_args = {
    "bb_ca": {"schedule": {"mode": "log", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
              "gt": {"mode": "1/t", "p": 1.0, "clamp_val": None},
              "simulation_step_params": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0,
                                         "t_lim_ode": 0.95, "t_lim_ode_below": 0.02,
                                         "center_every_step": False, "tsr_k": 16.0, "tsr_sigma": 3.0}},
    "local_latents": {"schedule": {"mode": "power", "p": 2.0, "tail_steps": 10, "tail_step_multiplier": 2},
                      "gt": {"mode": "tan", "p": 1.0, "clamp_val": None},
                      "simulation_step_params": {"sampling_mode": "vf_tsr", "sc_scale_noise": 1.0,
                                                 "t_lim_ode": 0.95, "t_lim_ode_below": 0.02,
                                                 "center_every_step": False, "tsr_k": 4.0, "tsr_sigma": 12.0}},
}
torch.manual_seed(0)
gen, extra = model.fm.full_simulation(
    batch=batch, predict_for_sampling=model.predict_for_sampling,
    recycle_depths=recycle_depths, nsteps=60, nsamples=1, n=n,
    sampling_model_args=sampling_model_args, device=torch.device("cpu"),
    guidance_w=1.0, guidance_model=None)

seq_logits = extra.get("sequence_logits")
sample_prots = model.sample_formatting(x=gen, ret_mode="coors37_n_aatype", batch=batch)
coors = sample_prots["coors"][0].numpy()
np.savez_compressed(OUT / "native_v2.npz", coors=coors,
                    sequence_logits=seq_logits.numpy() if seq_logits is not None else np.zeros(0),
                    padding_mask=batch["padding_mask"][0].numpy(),
                    coord_cond_mask=batch["coord_cond_mask"][0].numpy(),
                    cond_seq=batch["condition_sequence_1_residue"][0].numpy(),
                    token_type=batch["token_type"][0].numpy(),
                    atom37_mask=batch["atom37_mask"][0].numpy(),
                    coords_in=batch["coords"][0].numpy())
print("native v2 saved; sequence_logits:", None if seq_logits is None else tuple(seq_logits.shape))
