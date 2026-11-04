from conftest import make_pair, run_parity

import torch.nn as tnn


def test_embedding():
    torch_mod, jax_mod = make_pair(tnn.Embedding, 32, 16)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"indices": ("int", (4, 8))})
    print(f"embedding: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
