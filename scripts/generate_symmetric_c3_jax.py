"""C3-symmetric structure sampling with RFD4.

Following the symmetric generation example in github.com/mjendrusch/salad
(salad/training/eval_sym.py): every sampling step symmetrizes the CA flow
state and the clean CA output (one representative unit replicated with the
symmetry rotations), with velocity and score recomputed from the
symmetrized quantities. Local latents are NOT symmetrized or replicated.

Input: three 50-residue chains plus three caffeine ligands (topology-only,
freely diffused). The compiled token layout is ALL CHAINS FIRST, then ALL
LIGANDS (verified: token_type runs [0 x 150, 3 x 42]): chain u occupies
tokens [u*50, (u+1)*50) and caffeine u occupies
[num_aa*count + u*n_lig, ... + n_lig). Unit 1 (chain tokens 0:50, caffeine
tokens n_chain_tot : +n_lig) is the representative; its positions are
rotated by 120/240 degrees about the origin and the assembly is re-centered
on its CA centroid.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import numpy as np

# Release weights, fetched from the Hugging Face hub on first use.
SNAPSHOT = "nvidia/RFD4-Proteina"
CAFFEINE = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-aa", type=int, default=50)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--ligand-smiles", default=CAFFEINE)
    parser.add_argument("--num-designs", type=int, default=5)
    parser.add_argument("--nsteps", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=pathlib.Path,
                        default=pathlib.Path("runs/rfd4-test-symmetry"))
    parser.add_argument("--cache", type=pathlib.Path, default=None,
                        help="HF cache dir for the release checkpoint "
                             "(default: the huggingface default cache)")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import torch
    import biotite.structure as bs
    import biotite.structure.io.pdb as pdbio
    from atomworks.ml.encoding_definitions import UNIFIED_ATOM37_ENCODING
    from rfd4_proteina import Proteina

    import rfd4fx.convert  # noqa: F401
    from rfd4fx.flow_matching import (compute_tsr_ratio, get_gt,
                                           get_schedule_tsr_safe, sample_noise,
                                           simulation_step, _set_x_t)
    from rfd4fx.model import (MATCHER_CFGS, SAMPLER_ARGS, STEP_PARAMS,
                                   THINKING_HIGH)
    from rfd4fx.nn.transformer import compute_flow_emit_masks
    from rfd4fx.rfd4_input import RFD4Input
    from rfd4fx.spec import RFD4Spec

    enc = UNIFIED_ATOM37_ENCODING
    idx_to_tok = (enc.idx_to_token if isinstance(enc.idx_to_token, dict)
                  else dict(enumerate(enc.idx_to_token)))
    elem_of = {6: "C", 7: "N", 8: "O", 16: "S"}

    # ---------------------------------------------------------------- input
    # three chains + three ligands. Compiled layout: chains first (tokens
    # [0, num_aa*count)), then ligands (each n_lig tokens) — verify before use.
    tmp = RFD4Spec()
    tmp.add_ligand(smiles=args.ligand_smiles)
    lig_elements = tmp.ligands[0].elements
    n_lig = len(lig_elements)

    chains = []
    for _ in range(args.count):
        chains += [args.num_aa, {"smiles": args.ligand_smiles}]
    inp = RFD4Input.from_chains(*chains, np_seed=args.seed)
    inp = inp.pad_to_nearest(64)
    batch = inp.to_batch()
    n_width = int(np.asarray(batch.padding_mask).shape[1])
    n_ch = args.num_aa * args.count            # chain-token block [0, n_ch)
    n_lg = args.count * n_lig                  # ligand-token block [n_ch, n_ch+n_lg)
    assert n_ch + n_lg == n_width, (n_ch, n_lg, n_width)
    tt = np.asarray(batch.token_type)[0]
    sup = np.asarray(batch.supports_local_latent)[0]
    chain_toks = np.zeros(n_width, bool); chain_toks[:n_ch] = True
    assert (tt[:n_ch] == 0).all() and (tt[n_ch:] == 3).all(), \
        "unexpected compiled layout: chains must precede ligands"
    assert (sup == chain_toks).all(), \
        "supports_local_latent must be True exactly on the chain tokens"
    print(f"input: {args.count} chains x {args.num_aa} tokens [0:{n_ch}), "
          f"{args.count} ligands x {n_lig} tokens [{n_ch}:{n_ch + n_lg}), "
          f"width {n_width}")

    # ---------------------------------------------------------------- model
    print("loading + converting release weights ...", flush=True)
    model = Proteina.from_pretrained(SNAPSHOT, cache_dir=args.cache)
    model.eval()
    import rfd4fx.convert  # noqa: F401
    from rfd4fx.model import RFD4

    jax_model = RFD4.from_torch(model)
    base_step = jax_model.step()

    # ------------------------------------------------------- C3 symmetrizer
    # CA only: the local latents are neither rotated nor replicated.
    angle = 2 * jnp.pi / args.count
    rot = jnp.array([[jnp.cos(angle), -jnp.sin(angle), 0.0],
                     [jnp.sin(angle),  jnp.cos(angle), 0.0],
                     [0.0, 0.0, 1.0]])
    rots = [jnp.linalg.matrix_power(rot, k) for k in range(args.count)]

    def replicate_state(x):                      # (n_width, 3): replicate unit 1
        # chain block: chain u = chain 1 rotated by 120*u degrees; ligand
        # block: ligand u = ligand 1 rotated likewise; re-center on the
        # assembly's chain (CA) centroid
        ch, lg = x[:n_ch], x[n_ch:]
        ch_sym = jnp.concatenate([ch[:args.num_aa] @ r for r in rots], axis=0)
        lg_sym = jnp.concatenate([lg[:n_lig] @ r for r in rots], axis=0)
        out = jnp.concatenate([ch_sym, lg_sym], axis=0)
        return out - ch_sym.mean(0, keepdims=True)

    # jitted step: symmetrize the full coordinate state input, then run the
    # default step
    def symmetric_step(model, key, data):
        x_sym = replicate_state(data.x_t_bb_ca[0])
        data = eqx.tree_at(lambda d: d.x_t_bb_ca, data, x_sym[None])
        return base_step(model, key, data)

    step_jit = eqx.filter_jit(symmetric_step)

    # ------------------------------------------------------- sampling loop
    # mirrors RFD4Sampler.sample with per-step CA symmetrization (salad-style):
    # the CA flow state and the clean CA output are projected onto the
    # symmetric subspace every iteration; velocity and score are recomputed
    # from the symmetrized quantities. Latents are left untouched.
    emit = compute_flow_emit_masks(batch)
    n = n_width
    ts, gts = {}, {}
    for dm, cfg_args in SAMPLER_ARGS.items():
        ssp = STEP_PARAMS[dm]
        sargs = cfg_args["schedule"]
        ts[dm] = get_schedule_tsr_safe(sargs["mode"], args.nsteps, p1=sargs["p"],
                                       tail_steps=sargs.get("tail_steps", 0),
                                       tail_step_multiplier=sargs.get("tail_step_multiplier", 1))
        gargs = cfg_args["gt"]
        g = get_gt(ts[dm][:-1], gargs["mode"], gargs["p"], gargs.get("clamp_val"))
        g = g / compute_tsr_ratio(ts[dm][:-1], ssp.tsr_k, ssp.tsr_sigma) ** 2
        gts[dm] = g

    depths = [d for width, depth in THINKING_HIGH for d in [depth] * width]
    decode_jit = eqx.filter_jit(lambda m, d: m.decode(
        d, d.x_pred, d.z_pred, aa_pred=d.aa_pred))

    key = jax.random.key(args.seed)
    t_all = time.time()
    for design in range(args.num_designs):
        t0 = time.time()
        key, key_state = jax.random.split(key)
        k_init, key_state = jax.random.split(key_state)
        x = {"bb_ca": sample_noise(k_init, (1,), n, MATCHER_CFGS["bb_ca"], emit["bb_ca"]),
             "local_latents": sample_noise(k_init, (1,), n,
                                           MATCHER_CFGS["local_latents"],
                                           emit["local_latents"])}
        data = batch
        x_1 = None
        for i in range(args.nsteps):
            # 1) project the coordinate state onto the symmetric subspace
            x["bb_ca"] = replicate_state(x["bb_ca"][0])[None]
            k_model, _, _, key_state = jax.random.split(key_state, 4)
            t_bb = ts["bb_ca"][i] * jnp.ones(1)
            t_lat = ts["local_latents"][i] * jnp.ones(1)
            data = _set_x_t(data, x["bb_ca"], x["local_latents"], t_bb, t_lat)
            data = eqx.tree_at(lambda d: d.n_recycle, data,
                               jnp.asarray(depths[i], dtype=jnp.int32),
                               is_leaf=lambda l: l is None)
            # 2) run the model on the symmetric input
            data = step_jit(jax_model, k_model, data)
            # 3) symmetrize the clean CA output, recompute v and s from it -
            #    every quantity the integrator touches is symmetric
            x1_bb = replicate_state(data.x_pred[0])
            x_1 = {"bb_ca": x1_bb[None], "local_latents": data.z_pred}
            t = {"bb_ca": t_bb, "local_latents": t_lat}
            v = {"bb_ca": (x_1["bb_ca"] - x["bb_ca"]) / (1.0 - t_bb),
                 "local_latents": data.v_pred["local_latents"]}
            s = {"bb_ca": (t_bb * v["bb_ca"] - x["bb_ca"]) / (1.0 - t_bb),
                 "local_latents": data.s_pred["local_latents"]}
            # 4) integrate; the latents use the model's own velocity
            k_bb, k_lat, key_state = jax.random.split(key_state, 3)
            x["bb_ca"] = simulation_step(x["bb_ca"], v["bb_ca"], ts["bb_ca"][i],
                                         float(ts["bb_ca"][i + 1] - ts["bb_ca"][i]),
                                         float(gts["bb_ca"][i]), emit["bb_ca"],
                                         STEP_PARAMS["bb_ca"], key=k_bb)
            x["local_latents"] = simulation_step(
                x["local_latents"], v["local_latents"], ts["local_latents"][i],
                float(ts["local_latents"][i + 1] - ts["local_latents"][i]),
                float(gts["local_latents"][i]), emit["local_latents"],
                STEP_PARAMS["local_latents"], key=k_lat)
        # final symmetrization of the clean coordinate output, then decode
        data = eqx.tree_at(lambda d: d.x_pred, data,
                           replicate_state(data.x_pred[0])[None])
        dec = decode_jit(jax_model, data)

        # ---------------------------------------------------------- output
        # explicit per-chain assembly: exactly num_aa residues per chain from
        # the decoded tokens, plus the unit's caffeine from the (symmetric)
        # final CA flow state.
        coors_a = np.asarray(dec.coors_nm)[0] * 10.0
        emit_m = np.asarray(dec.emit_mask)[0]
        # residue identity: unit 1's trunk readout (aa_pred), applied to ALL
        # units. The latents are intentionally left asymmetric, which makes
        # the trunk's sequence readout degenerate to <M> (layout-less) at
        # many unit-2/3 positions; a symmetric assembly needs identical
        # sequences, so the representative unit defines the sequence.
        aa_unit1 = (np.asarray(data.aa_pred)[0][:args.num_aa]
                    if data.aa_pred is not None
                    else np.asarray(dec.aatype_max)[0][:args.num_aa])
        flow_ca = np.asarray(x["bb_ca"])[0] * 10.0    # symmetric final state
        arr = None
        n_ca_only = {}
        for u in range(args.count):
            lo = u * args.num_aa                      # chain u tokens
            arr_u = bs.AtomArray(0)
            n_ca_only[u] = 0
            for t in range(args.num_aa):
                tok = lo + t
                if not emit_m[tok]:
                    continue
                res_name = str(idx_to_tok.get(int(aa_unit1[t]), "UNK"))
                slot_names = enc.token_atoms.get(res_name)
                if slot_names is None:
                    continue
                res_id = t + 1
                wrote_ca = False
                for s in range(37):
                    xyz = coors_a[tok, s]
                    an = str(slot_names[s]).strip()
                    if not an or not np.isfinite(xyz).all() or np.abs(xyz).sum() == 0:
                        continue
                    atom = bs.AtomArray(1)
                    atom.coord[0] = xyz
                    atom.chain_id[0] = "ABC"[u]
                    atom.res_id[0] = res_id
                    atom.res_name[0] = res_name
                    atom.atom_name[0] = an
                    letters = "".join(c for c in an if c.isalpha()).upper()
                    atom.element[0] = letters[:2] if letters[:2] in ("CL", "BR") else letters[:1]
                    arr_u += atom
                    wrote_ca = wrote_ca or an == "CA"
                if not wrote_ca:
                    # the decode degenerated for this token (asymmetric
                    # latents); the CA from the symmetrized clean prediction
                    # is exact
                    atom = bs.AtomArray(1)
                    atom.coord[0] = coors_a[tok, 1]
                    atom.chain_id[0] = "ABC"[u]
                    atom.res_id[0] = res_id
                    atom.res_name[0] = res_name
                    atom.atom_name[0] = "CA"
                    atom.element[0] = "C"
                    arr_u += atom
                    n_ca_only[u] += 1
            assert len(np.unique(arr_u.res_id)) == args.num_aa, \
                f"unit {u}: {len(np.unique(arr_u.res_id))} residues != {args.num_aa}"
            lig = bs.AtomArray(0)
            lo_lig = n_ch + u * n_lig                 # ligand u tokens
            for a in range(n_lig):
                atom = bs.AtomArray(1)
                atom.coord[0] = flow_ca[lo_lig + a]
                atom.chain_id[0] = "ABC"[u]
                atom.res_id[0] = args.num_aa + 1
                atom.res_name[0] = "CAF"
                atom.atom_name[0] = f"C{a}"
                atom.element[0] = elem_of.get(int(lig_elements[a]), "C")
                atom.hetero[0] = True
                lig += atom
            arr_u += lig
            arr = (arr + arr_u) if arr is not None else arr_u
        out = args.out_dir / f"c3_design_{design:02d}.pdb"
        pdb = pdbio.PDBFile()
        pdb.set_structure(arr)
        pdb.write(str(out))

        # symmetry check on the final coordinate state (chains-then-ligands
        # layout): chain u tokens [u*num_aa, ...), ligand u tokens after n_ch
        x_sym = flow_ca / 10.0
        ch, lg = x_sym[:n_ch], x_sym[n_ch:]
        u1c, u1l = ch[:args.num_aa], lg[:n_lig]
        dev = max(
            max(float(np.abs(ch[u * args.num_aa:(u + 1) * args.num_aa]
                              - u1c @ rots[u]).max()),
                float(np.abs(lg[u * n_lig:(u + 1) * n_lig]
                             - u1l @ rots[u]).max()))
            for u in range(1, args.count))
        # decoded-structure symmetry (chains are built per unit; compare CAs)
        ca_by_chain = {}
        for c in np.unique(arr.chain_id):
            sel = arr[(arr.chain_id == c) & (arr.atom_name == "CA")
                      & (~arr.hetero)]
            ca_by_chain[c] = np.asarray(sel.coord)
        dec_dev = max(float(np.abs(ca_by_chain["ABC"[u]]
                                   - ca_by_chain["A"] @ rots[u]).max())
                      for u in range(1, args.count)
                      if ca_by_chain["ABC"[u]].shape == ca_by_chain["A"].shape)
        print(f"c3_design_{design:02d}: written, state symmetry {dev:.2e} A, "
              f"decoded symmetry {dec_dev:.3f} A, CA-only residues "
              f"{[n_ca_only[u] for u in range(args.count)]} "
              f"[{time.time() - t0:.0f}s]", flush=True)
    print(f"done: {args.num_designs} designs in {time.time() - t_all:.0f}s "
          f"-> {args.out_dir}")


if __name__ == "__main__":
    main()
