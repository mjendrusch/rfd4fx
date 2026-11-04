import numpy as np, sys, torch
sys.path.insert(0, '/Users/mjendrusch/repos/rfd4j/rfd4fx/src')
sys.path.insert(0, '/Users/mjendrusch/repos/rfd4j/RFD4-Proteina/src')
import jax, jax.numpy as jnp, equinox as eqx
import rfd4fx.convert
from rfd4fx.featurize import make_unconditioned_batch
from rfd4fx.model import RFD4
from rfd4_proteina import Proteina

torch.manual_seed(0)
model = Proteina.from_pretrained('/Users/mjendrusch/repos/rfd4j/.hf_cache/hub/models--nvidia--RFD4-Proteina/snapshots/a06e2509c7033b010fc43614e9fd840303588502')
model.eval()
jax_model = RFD4.from_torch(model)

host, batch = make_unconditioned_batch(100, b=1)
x, x_1_pred, dec = jax_model.generate(batch, key=jax.random.key(0), nsteps=60)
z = np.asarray(x_1_pred['local_latents']); ca = np.asarray(x_1_pred['bb_ca'])
seq_jax = np.asarray(dec.aatype_max)[0]
mask_jax = np.asarray(dec.atom_mask)[0]

tb = {
    'padding_mask': torch.ones(1, 100, dtype=torch.bool),
    'token_type': torch.zeros(1, 100, dtype=torch.int32),
    'atom37_mask': torch.zeros(1, 100, 37, dtype=torch.bool),
    'coords': torch.zeros(1, 100, 37, 3),
    'condition_sequence_1_residue': torch.zeros(1, 100, dtype=torch.int64),
    'condition_CoordinateGroup_1_atom': torch.full((1, 100, 37), -1, dtype=torch.int16),
    'condition_coordinate_1_atom': torch.full((1, 100, 37, 3), float('nan')),
    'condition_InternalCoordinate_1_atom': torch.full((1, 100, 37, 3), float('nan')),
    'molecule_iid': torch.zeros(1, 100, dtype=torch.int16),
    'chain_iid': torch.zeros(1, 100, dtype=torch.int16),
    'is_in_context_conditioning': torch.zeros(1, 100, dtype=torch.bool),
    'supports_local_latent': torch.ones(1, 100, dtype=torch.bool),
}
from omegaconf import OmegaConf
model.inf_cfg = OmegaConf.create({'args': {'condition_decoder_on_sequence_prediction': True}})
from rfd4fx.flow_matching import _set_x_t
t_ones = jnp.ones(1)
out_final = jax_model.nn(_set_x_t(batch, jnp.asarray(ca), jnp.asarray(z), t_ones, t_ones), n_recycle=1, key=jax.random.key(7))
tb['sequence_logits'] = torch.from_numpy(np.asarray(out_final.seq_logits))
tb['x_t'] = {'bb_ca': torch.from_numpy(ca), 'local_latents': torch.from_numpy(z)}
sample = model._format_sample_local_latents(tb['x_t'], ret_mode='coors37_n_aatype', batch=tb)
seq_torch = np.asarray(sample['residue_type'])[0]
mask_torch = np.asarray(sample['atom_mask'])[0]
print('SEQUENCE match (jax decode vs torch decode of same z,ca):', bool((seq_jax == seq_torch).all()))
print('ATOM MASK match:', bool((mask_jax == mask_torch).all()))
if not (seq_jax == seq_torch).all():
    diff = np.where(seq_jax != seq_torch)[0]
    print('mismatching tokens:', diff[:10].tolist(), 'of', len(diff))
from rfproteina.encoding import UNIFIED_ATOM37_ENCODING as E
names_t = ''.join((E.idx_to_token[int(t)][0] if int(t) >= 0 else '-') for t in seq_torch)
names_j = ''.join((E.idx_to_token[int(t)][0] if int(t) >= 0 else '-') for t in seq_jax)
print('torch seq:', names_t)
print('jax   seq:', names_j)
