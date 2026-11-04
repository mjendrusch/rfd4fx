from conftest import make_pair, run_parity

from rfproteina.nn.modules.blocks import FourierEmbedding


def test_fourier_embedding():
    torch_mod, jax_mod = make_pair(FourierEmbedding, 32)
    # t is a per-example scalar vector in torch usage; include values outside [0, 1].
    spec = {"t": ("float", (8,))}
    mean, std, gmax = run_parity(torch_mod, jax_mod, spec)
    print(f"fourier_embedding: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")


def test_fourier_embedding_batched():
    torch_mod, jax_mod = make_pair(FourierEmbedding, 64)
    mean, std, gmax = run_parity(torch_mod, jax_mod, {"t": ("float", (2, 8))})
    print(f"fourier_embedding batched: {mean:.3e} ± {std:.3e} (max {gmax:.3e})")
