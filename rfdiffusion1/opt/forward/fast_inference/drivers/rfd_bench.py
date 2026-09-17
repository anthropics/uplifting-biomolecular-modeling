#!/usr/bin/env python
"""
RFdiffusion v1 resident driver: scripts/run_inference.py's design loop with the model kept resident across cases.

Reproduces run_inference.py semantics exactly (the same sampler object, the same per-design seeding
`make_deterministic(i_des)` = torch/np/random seeded with the design index under inference.deterministic, the same
inference.cautious skip of a design whose <prefix>_<i>.pdb already exists, the same sample_init -> 50 x sample_step
loop, the same writepdb / .trb / trajectory output) and applies the levers named on its command line inside this process:

  --fastpath ...     C1  memoised per-design constants (rfd_fastpath)
  --prep 1           P   fast _preprocess (rfd_prep)
  --einsum-route 1   E   two opt_einsum signatures on torch.einsum, checked bitwise at start-up (rfd_einsum)
  --fullgraph 1      W1  one CUDA graph per forward phase (rfd_fullgraph)
  --triton-ln 1      K2  Triton row LayerNorm (rfd_layernorm; tolerance tier)
  --tf32 1           TF32 matmul / cuDNN (tolerance tier)

Per design it prints one line and appends the design's row (index, length, steps, output file) to the run record
<out>/<tag>_timings.json — the manifest (versions, card, flags), per case the torch numerics in force and the designs
upstream's cautious rule skipped, and each lever's final counters; each lever prints its own applied line. Nothing here
times the run: the .trb `time` field is upstream's own per-design record, written as run_inference.py writes it.
"""
import argparse, json, os, sys, time, pickle, subprocess
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--cases", required=True, help="json list of cases")
p.add_argument("--out", required=True)
p.add_argument("--rfd-root", default="/opt/rfd")
p.add_argument("--weights", default="/weights")
p.add_argument("--tag", default="run")
p.add_argument("--tf32", type=int, default=0, help="appendix only: allow TF32 matmul/cudnn (numerics-changing)")
p.add_argument("--no-traj", type=int, default=1, help="skip trajectory pdb writing (write_trajectory=False); I/O lever, timed separately when on")
p.add_argument("--fastpath", default="", help="comma list of rfd_fastpath levers (chain_breaks,full_graph,rbf)")
p.add_argument("--fullgraph", type=int, default=0, help="lever W1: ONE CUDA graph for embeddings+templ+36 blocks (+1 for heads); refinement eager (needs the pinned torch/dgl stack + fastpath incl. msa_index)")
p.add_argument("--triton-ln", type=int, default=0, help="lever K2 (tier 2, numerics-changing at last-bit level): Triton row LayerNorm replaces F.layer_norm")
p.add_argument("--einsum-route", type=int, default=0, help="lever E (exact): route two opt_einsum signatures to torch.einsum (live op-level bitwise check at start-up; applied before graph capture)")
p.add_argument("--prep", type=int, default=0, help="lever P: exact fast _preprocess (dead CPU xyz_to_t2d skipped, per-design constants memoised)")
args = p.parse_args()

os.makedirs(args.out, exist_ok=True)
sys.path.insert(0, args.rfd_root)
os.environ.setdefault("DGLBACKEND", "pytorch")

import torch
torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
torch.backends.cudnn.allow_tf32 = bool(args.tf32)

# ----------------------------------------------------------------------------- card identity (the manifest's record)
def gpu_name():
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "none"

def nvsmi():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception as e:
        return f"nvidia-smi failed: {e}"

manifest = dict(tag=args.tag, fastpath=args.fastpath, fullgraph=bool(getattr(args, "fullgraph", 0)), prep=bool(getattr(args, "prep", 0)), triton_ln=bool(getattr(args, "triton_ln", 0)), einsum_route=bool(getattr(args, "einsum_route", 0)), env_levers={k: os.environ.get(k) for k in ["RFD_FG_HEADS", "RFD_FG_MAX_GENERATIONS", "RFD_PREP", "RFD_TRITON_LN_MIN_NUMEL", "CUDA_MPS_PIPE_DIRECTORY"]}, gpu_name=gpu_name(), nvidia_smi=nvsmi(),
                torch=torch.__version__, cuda=torch.version.cuda, tf32=bool(args.tf32),
                run_label=os.environ.get("RUN_LABEL"),
                start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
try:
    import dgl; manifest["dgl"] = dgl.__version__
except Exception as e:
    manifest["dgl"] = f"import failed: {e}"
print(json.dumps(manifest, indent=1), flush=True)
assert "H100" in manifest["gpu_name"] or os.environ.get("ALLOW_ANY_GPU") == "1", f"GPU is {manifest['gpu_name']}, expected H100 (set ALLOW_ANY_GPU=1 to override)"

cases = json.load(open(args.cases))

# ----------------------------------------------------------------------------- resident mode
import logging
logging.basicConfig(level=logging.WARNING)
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir
from rfdiffusion.inference import utils as iu
from rfdiffusion.util import writepdb_multi, writepdb
from rfdiffusion.kinematics import xyz_to_t2d
import random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
FAST = []
if args.fullgraph and not args.fastpath:
    args.fastpath = 'chain_breaks,full_graph,rbf,msa_index'
if args.fullgraph and 'msa_index' not in args.fastpath:
    args.fastpath += ',msa_index'
if args.fastpath:
    import rfd_fastpath
    FAST = rfd_fastpath.apply(set(args.fastpath.split(',')))
    print('fastpath levers:', FAST, flush=True)

def make_deterministic(seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

# ---- build sampler via hydra compose (same config path as run_inference.py) -----------------
def build_conf(c, out_prefix):
    ov = [f"inference.input_pdb={c['pdb']}", f"inference.output_prefix={out_prefix}",
          f"inference.model_directory_path={args.weights}", f"inference.num_designs={c['num_designs']}",
          f"contigmap.contigs={c['contigs']}", f"ppi.hotspot_res={c['hotspots']}",
          "inference.deterministic=True", f"inference.write_trajectory={not bool(args.no_traj)}", "inference.cautious=False"]
    if c.get("ckpt"): ov.append(f"inference.ckpt_override_path={args.weights}/{c['ckpt']}")
    with initialize_config_dir(config_dir=f"{args.rfd_root}/config/inference", version_base=None):
        conf = compose(config_name="base", overrides=ov)
    return conf

results = dict(manifest=manifest, cases=[])
sampler = None
for ci, c in enumerate(cases):
    cname = c["name"]
    out_prefix = os.path.abspath(c["prefix"]) if c.get("prefix") else os.path.join(os.path.abspath(args.out), cname, "des")   # the case's own output prefix (upstream's inference.output_prefix) when it names one
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    conf = build_conf(c, out_prefix)
    torch.cuda.synchronize()
    if sampler is None:
        sampler = iu.sampler_selector(conf)      # loads checkpoint + builds model (first case)
        init_kind = "cold_model_load"
        if args.einsum_route:
            import rfd_einsum
            print("einsum lever E:", rfd_einsum.apply(), flush=True)
        if args.fullgraph:
            import rfd_fullgraph
            ok, why = rfd_fullgraph.supported()
            if not ok:
                print("fullgraph mode REFUSED:", why, "-> exiting with code 3", flush=True); sys.exit(3)
            print("fullgraph mode: applied =", rfd_fullgraph.apply(sampler.model), flush=True)
        if args.prep:
            import rfd_prep
            print("prep lever: active =", rfd_prep.apply(sampler), flush=True)
        if args.triton_ln:
            import rfd_layernorm
            print("triton LN lever (Tier 2): active =", rfd_layernorm.apply(), flush=True)
    else:
        sampler.initialize(conf)                # re-init target/contig only (model stays resident unless ckpt changes)
        init_kind = "reinit"
    torch.cuda.synchronize()
    prec = dict(param_dtype=str(next(sampler.model.parameters()).dtype),
                allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, cudnn_tf32=torch.backends.cudnn.allow_tf32,
                autocast_enabled=torch.is_autocast_enabled())
    case_res = dict(case=cname, cfg=c, init_kind=init_kind, precision=prec, ckpt=sampler.ckpt_path, designs=[], skipped_existing=[])
    print(f"[case {cname}] init {init_kind} ckpt={sampler.ckpt_path}", flush=True)
    for i_des in range(c.get("startnum", 0), c.get("startnum", 0) + c["num_designs"]):
        if sampler.inf_conf.deterministic: make_deterministic(i_des)   # upstream's rule (run_inference.py:72-73): seeded per design under inference.deterministic=True only
        out_pdb = f"{out_prefix}_{i_des}.pdb"
        if sampler.inf_conf.cautious and os.path.exists(out_pdb):    # upstream's rule (run_inference.py:78-82): under inference.cautious=True an existing design is skipped, not recomputed
            print(f"[{cname} des {i_des}] (cautious mode) Skipping this design because {out_pdb} already exists.", flush=True)
            case_res["skipped_existing"].append(i_des); continue
        torch.cuda.synchronize()
        d0 = time.time()                                                  # upstream's per-design clock: the .trb `time` field below (run_inference.py writes time.time() - start)
        x_init, seq_init = sampler.sample_init()
        torch.cuda.synchronize()
        L = seq_init.shape[0]
        denoised_xyz_stack, px0_xyz_stack, seq_stack, plddt_stack = [], [], [], []
        x_t = torch.clone(x_init); seq_t = torch.clone(seq_init)
        ts = list(range(int(sampler.t_step_input), sampler.inf_conf.final_step - 1, -1))
        for si, t in enumerate(ts):
            # ---- replicate SelfConditioning.sample_step (equal ops/order) ----
            msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d, t2d, xyz_t, alpha_t = sampler._preprocess(seq_t, x_t, t)
            B, N, Lx = xyz_t.shape[:3]
            if (t < sampler.diffuser.T) and (t != sampler.diffuser_conf.partial_T):
                zeros = torch.zeros(B, 1, Lx, 24, 3).float().to(xyz_t.device)
                xyz_t = torch.cat((sampler.prev_pred.unsqueeze(1), zeros), dim=-2)
                t2d_44 = xyz_to_t2d(xyz_t)
            else:
                xyz_t = torch.zeros_like(xyz_t); t2d_44 = torch.zeros_like(t2d[..., :44])
            t2d[..., :44] = t2d_44
            with torch.no_grad():
                msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = sampler.model(
                    msa_masked, msa_full, seq_in, xt_in, idx_pdb, t1d=t1d, t2d=t2d, xyz_t=xyz_t, alpha_t=alpha_t,
                    msa_prev=None, pair_prev=None, state_prev=None, t=torch.tensor(t), return_infer=True,
                    motif_mask=sampler.diffusion_mask.squeeze().to(sampler.device), cyclic_reses=sampler.cyclic_reses)
            sampler.prev_pred = torch.clone(px0)
            _, px0 = sampler.allatom(torch.argmax(seq_in, dim=-1), px0, alpha)
            px0 = px0.squeeze()[:, :14]
            seq_t_1 = torch.clone(seq_t)
            if t > sampler.inf_conf.final_step:
                x_t_1, px0 = sampler.denoiser.get_next_pose(xt=x_t, px0=px0, t=t, diffusion_mask=sampler.mask_str.squeeze(),
                                                           align_motif=sampler.inf_conf.align_motif,
                                                           include_motif_sidechains=sampler.preprocess_conf.motif_sidechain_input)
            else:
                x_t_1 = torch.clone(px0).to(x_t.device); px0 = px0.to(x_t.device)
            px0_c, x_t, seq_t, plddt_c = px0, x_t_1, seq_t_1, plddt
            torch.cuda.synchronize()
            px0_xyz_stack.append(px0_c); denoised_xyz_stack.append(x_t); seq_stack.append(seq_t); plddt_stack.append(plddt_c[0])
        # ---- outputs equal to run_inference.py ----
        denoised_xyz_stack = torch.flip(torch.stack(denoised_xyz_stack), [0]); px0_xyz_stack = torch.flip(torch.stack(px0_xyz_stack), [0])
        plddt_stack = torch.stack(plddt_stack)
        final_seq = torch.where(torch.argmax(seq_init, dim=-1) == 21, 7, torch.argmax(seq_init, dim=-1))
        bfacts = torch.ones_like(final_seq.squeeze()); bfacts[torch.where(torch.argmax(seq_init, dim=-1) == 21, True, False)] = 0
        writepdb(out_pdb, denoised_xyz_stack[0, :, :4], final_seq, sampler.binderlen, chain_idx=sampler.chain_idx, bfacts=bfacts, idx_pdb=sampler.idx_pdb)
        trb = dict(config=OmegaConf.to_container(sampler._conf, resolve=True), plddt=plddt_stack.cpu().numpy(), device=gpu_name(), time=time.time() - d0)
        if hasattr(sampler, "contig_map"):
            for key, value in sampler.contig_map.get_mappings().items(): trb[key] = value
        with open(f"{out_prefix}_{i_des}.trb", "wb") as f_out: pickle.dump(trb, f_out)
        if sampler.inf_conf.write_trajectory:
            traj_prefix = os.path.dirname(out_prefix) + "/traj/" + os.path.basename(out_prefix); os.makedirs(os.path.dirname(traj_prefix), exist_ok=True)
            writepdb_multi(f"{traj_prefix}_{i_des}_Xt-1_traj.pdb", denoised_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)
            writepdb_multi(f"{traj_prefix}_{i_des}_pX0_traj.pdb", px0_xyz_stack, bfacts, final_seq.squeeze(), use_hydrogens=False, backbone_only=False, chain_ids=sampler.chain_idx)
        drow = dict(i_des=i_des, L=int(L), binderlen=int(sampler.binderlen), n_steps=len(ts), pdb=os.path.relpath(out_pdb, args.out))
        if args.fullgraph:
            drow["fullgraph"] = rfd_fullgraph.stats()
        if args.prep:
            drow["prep"] = rfd_prep.stats()
        case_res["designs"].append(drow)
        print(f"[{cname} des {i_des}] L={L} steps={len(ts)} -> {drow['pdb']}", flush=True)
    results["cases"].append(case_res)
    json.dump(results, open(os.path.join(args.out, f"{args.tag}_timings.json"), "w"), indent=1)
if args.fullgraph:
    results["fullgraph_final"] = rfd_fullgraph.stats()
    print("FGSTATS_FINAL", args.tag, json.dumps(results["fullgraph_final"]), flush=True)
    assert results["fullgraph_final"]["individual_graph_pops"] == 0
if args.prep:
    results["prep_final"] = rfd_prep.stats(); print("PREPSTATS_FINAL", args.tag, json.dumps(results["prep_final"]), flush=True)
if args.triton_ln:
    results["triton_ln_final"] = rfd_layernorm.stats(); print("LNSTATS_FINAL", args.tag, json.dumps(results["triton_ln_final"]), flush=True)
if args.einsum_route:
    results["einsum_route_final"] = rfd_einsum.stats(); print("ESTATS_FINAL", args.tag, json.dumps(results["einsum_route_final"]), flush=True)
json.dump(results, open(os.path.join(args.out, f"{args.tag}_timings.json"), "w"), indent=1)
print("DONE", flush=True)
