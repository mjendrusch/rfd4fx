"""Convert the release RFD4-Proteina torch checkpoint to a torch-free Equinox
model pair (``<name>.eqx`` + ``<name>.skeleton.pkl``, protenij/esmjfold2
convention) and verify the round trip.

Requires torch (conversion only); loading the result never does.

Example:
    python scripts/translate_rfd4.py \
        --checkpoint /path/to/snapshot --out ~/.rfd4fx/models/rfd4_release
"""

from __future__ import annotations

import argparse
import pathlib

# Release weights, fetched from the Hugging Face hub on first use.
SNAPSHOT = "nvidia/RFD4-Proteina"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=SNAPSHOT)
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="HF cache dir for the release checkpoint "
                             "(default: the huggingface default cache)")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("~/.rfd4fx/models/rfd4_release"))
    args = parser.parse_args()
    args.out = args.out.expanduser()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    import torch

    import rfd4fx.convert  # noqa: F401  (registers all from_torch converters)
    from rfd4fx.model import RFD4, load_model, save_model

    from rfd4_proteina import Proteina

    print("loading torch release ...", flush=True)
    model = Proteina.from_pretrained(args.checkpoint, cache_dir=args.cache)
    model.eval()
    print("converting to equinox ...", flush=True)
    eqx_model = RFD4.from_torch(model)
    save_model(eqx_model, str(args.out))
    print(f"saved {args.out}.eqx + {args.out}.skeleton.pkl")

    print("verifying round trip (torch-free load, weight-exact) ...", flush=True)
    import equinox as eqx

    reloaded = load_model(str(args.out))
    assert eqx.tree_equal(eqx_model, reloaded), "round-trip mismatch"
    print("round trip exact")
    print("done — the saved model loads torch-free via "
          "rfd4fx.model.load_model")


if __name__ == "__main__":
    main()
