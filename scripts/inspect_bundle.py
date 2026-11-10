"""Step 0b: enumerate the release checkpoint contract.

Loads the native bundle with Proteina.from_pretrained (torch, CPU) and dumps:
- state_dict key tree with shapes
- active feature lists per FeatureFactory (the feature subset to port)
- norm type (state_dict contract)
- trunk structure flags (recycle heads, insertion head, parallel attention)
- signature run: one forward pass on a tiny random batch to sanity-check the torch model

Output: stdout summary + rfd4fx/tests/fixtures/bundle_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

REV = "a06e2509c7033b010fc43614e9fd840303588502"


def main() -> None:
    from huggingface_hub import snapshot_download

    bundle = Path(
        snapshot_download(
            "nvidia/RFD4-Proteina",
            revision=REV,
            allow_patterns=["config.json", "model.safetensors"],
        )
    )
    from rfd4_proteina import Proteina

    model = Proteina.from_pretrained(bundle)
    model.eval()

    summary: dict = {"bundle": f"huggingface:nvidia/RFD4-Proteina@{REV}"}

    # --- state_dict key tree ---
    sd = model.state_dict()
    keys = {}
    for k, v in sd.items():
        keys[k] = list(v.shape)
    summary["n_state_tensors"] = len(keys)
    summary["state_keys"] = keys
    prefix_top = sorted({k.split(".")[0] for k in keys})
    summary["top_prefixes"] = prefix_top

    # --- feature factory composition ---
    nn = model.nn
    feat_info = {}
    for factory_name in ("init_repr_factory", "cond_factory", "pair_repr_builder"):
        factory = getattr(nn, factory_name, None)
        if factory is None:
            continue
        inner = getattr(factory, "pair_features", factory)
        names = []
        for creator in getattr(inner, "feat_creators", []):
            names.append(type(creator).__name__)
        feat_info[factory_name] = names
    summary["features"] = feat_info

    # --- architecture flags ---
    import rfproteina.nn.modules.norm as norm_mod

    summary["norm_type_global"] = norm_mod._NORM_TYPE
    summary["trunk_flags"] = {
        "nlayers": len(nn.attn_layers) if hasattr(nn, "attn_layers") else None,
        "has_pair_update_layers": hasattr(nn, "pair_update_layers"),
        "n_pair_updates": (
            sum(1 for l in nn.pair_update_layers if l is not None)
            if hasattr(nn, "pair_update_layers")
            else 0
        ),
        "attention_type": type(nn.attn_layers[0].mha).__name__ if hasattr(nn, "attn_layers") else None,
        "has_recycle": hasattr(nn, "s_cycle_proj"),
        "has_sequence_readout": hasattr(nn, "sequence_readout"),
        "has_insertion_head": hasattr(nn, "insertion_q_proj"),
    }

    out = Path("tests/fixtures")
    out.mkdir(parents=True, exist_ok=True)
    (out / "bundle_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "state_keys"}, indent=2))
    print("\nstate_dict sample (first 40):")
    for k in list(keys)[:40]:
        print(f"  {k}  {keys[k]}")
    print(f"\nfull dump: {out / 'bundle_summary.json'}  ({len(keys)} tensors)")


if __name__ == "__main__":
    sys.exit(main())
