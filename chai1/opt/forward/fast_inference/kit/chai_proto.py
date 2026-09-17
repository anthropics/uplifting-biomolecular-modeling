"""chai_proto.py — shared protocol helpers for the Chai-1 (chai_lab 0.6.1) fast-inference kit.

The stock protocol (one chai_lab.run_inference call per seed), byte-for-byte:
  FASTA = a chai FASTA as chai-lab reads it (records '>protein|name=X' / '>ligand|name=Y' ...), folded verbatim;
  MSAs = whatever chai .aligned.pqt files the run's msa_directory holds (chai-lab matches them by sequence; none =
  single-sequence mode); no templates;
  run_inference(use_esm_embeddings=True, use_msa_server=False, use_templates_server=False, msa_directory=msa_local/,
                num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=1, seed=s, device='cuda:0', low_memory=False)
  one call per seed.
Nothing in here changes the model, weights, MSA, recycles, steps, samples or seeds.
"""
import os, time, shutil, subprocess
from pathlib import Path

# ---- optional deterministic mode: CHAI_DETERMINISTIC=1|warn ----
def apply_deterministic_mode():
    mode = os.environ.get("CHAI_DETERMINISTIC", "0")
    if mode in ("0", "", None): return "off"
    import torch
    # DET-recipe item (not a speed lever): with the TorchScript profiling executor in profiling mode, the traced fp16 ESM2-3B module gives
    # different bits on its first forward than on later forwards of a process (see KNOWN_ISSUES.md §2; a dummy warm-up does NOT remove it because
    # the optimised graph is re-specialised per input shape; the legacy executor fails on these modules). Profiling mode off = every call runs
    # the same (first-call) graph -> stock becomes reproducible within a process. Must be set before any scripted module runs.
    if os.environ.get("CHAI_JIT_PROFILING_OFF", "1") != "0": torch._C._jit_set_profiling_mode(False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=(mode == "warn"))
    return mode


RUN_KW = dict(use_esm_embeddings=True, use_msa_server=False, msa_server_url="https://api.colabfold.com", constraint_path=None,
              use_templates_server=False, template_hits_path=None, recycle_msa_subsample=0,
              num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=1, num_trunk_samples=1, device="cuda:0", low_memory=False)


def log(*a):
    print(time.strftime("[%H:%M:%S]", time.gmtime()), *a, flush=True)


# ----------------------------------------------------------------------------------------------------------------
# inputs: a chai FASTA, folded verbatim
# ----------------------------------------------------------------------------------------------------------------
def fasta_spec(fasta_path, uid=None):
    """An existing chai FASTA (records '>protein|name=X' / '>ligand|name=Y' ...), folded as given; ``key`` — the output directory name —
    is the FASTA stem. MSAs: optional <msa_dir> with chai .aligned.pqt files (chai-lab matches them by sequence hash); none = single-sequence."""
    uid = uid or Path(fasta_path).stem
    return dict(uid=uid, key=uid, fasta_src=str(fasta_path))


def input_spec(uid):
    """uid forms: 'fasta:<path>' | an existing .fasta / .fa path -> fasta_spec."""
    u = str(uid)
    if u.startswith("fasta:"):
        return fasta_spec(u[len("fasta:"):])
    if u.endswith((".fasta", ".fa")):
        return fasta_spec(u)
    raise SystemExit(f"unknown input id {uid!r}: pass a FASTA as 'fasta:<path>' or a .fasta / .fa path")


def write_fasta(spec, path):
    shutil.copy(spec["fasta_src"], path); return path


def ensure_msa_pqt(spec, msa_dir):
    """The run's MSA directory as given: whatever chai .aligned.pqt files the caller placed there (none => single-sequence mode)."""
    msa_dir = Path(msa_dir).absolute(); msa_dir.mkdir(exist_ok=True, parents=True)
    return msa_dir


def env_report():
    import torch, chai_lab
    rep = dict(chai_lab=getattr(chai_lab, "__version__", "?"), torch=torch.__version__, cuda=torch.version.cuda,
               cudnn=torch.backends.cudnn.version(), gpu=torch.cuda.get_device_name(0),
               allow_tf32_matmul=torch.backends.cuda.matmul.allow_tf32, allow_tf32_cudnn=torch.backends.cudnn.allow_tf32,
               float32_matmul_precision=torch.get_float32_matmul_precision(),
               cudnn_benchmark=torch.backends.cudnn.benchmark, cudnn_deterministic=torch.backends.cudnn.deterministic,
               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
               flash_sdp=torch.backends.cuda.flash_sdp_enabled(), mem_efficient_sdp=torch.backends.cuda.mem_efficient_sdp_enabled(),
               math_sdp=torch.backends.cuda.math_sdp_enabled())
    try:
        rep["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,clocks.max.sm", "--format=csv,noheader"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        rep["nvidia_smi"] = repr(e)
    try:
        rep["cpu"] = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")][0]
        fl = [l for l in open("/proc/cpuinfo") if l.startswith("flags")][0]
        rep["cpu_flags"] = sorted({f for f in ("avx2", "avx512f", "avx512_bf16", "amx_tile", "amx_bf16") if f" {f}" in fl})
        rep["ncpu"] = os.cpu_count()
    except Exception:
        pass
    return rep


# ----------------------------------------------------------------------------------------------------------------
# per-seed output capture: upstream's own two files per seed
# ----------------------------------------------------------------------------------------------------------------
def save_seed_outputs(cand, od, sd):
    """cand = StructureCandidates from run_inference/run_folding_on_context; od = chai output dir; sd = keep dir. Keeps upstream's own two
    files of the first sample (pred.model_idx_0.cif, scores.model_idx_0.npz) where upstream wrote them — od itself, or od/trunk_0/ when
    the fold ran more than one trunk sample (run_inference's trunk_<i>/ layout) — nothing else."""
    import numpy as np
    cif = Path(cand.cif_paths[0]); npz = cif.parent / "scores.model_idx_0.npz"
    sd = Path(sd) / cif.parent.relative_to(od); sd.mkdir(parents=True, exist_ok=True)
    shutil.copy(cif, sd / "pred.model_idx_0.cif")
    if npz.exists(): shutil.copy(npz, sd / "scores.model_idx_0.npz")
    sc = dict(np.load(npz)) if npz.exists() else {}
    row = dict(iptm=float(np.asarray(sc.get("iptm", np.nan)).ravel()[0]), ptm=float(np.asarray(sc.get("ptm", np.nan)).ravel()[0]),
               aggregate_score=float(np.asarray(sc.get("aggregate_score", np.nan)).ravel()[0]),
               has_inter_chain_clashes=(bool(np.asarray(sc["has_inter_chain_clashes"]).ravel()[0]) if "has_inter_chain_clashes" in sc else None),
               pae_shape=list(cand.pae[0].shape))
    pcp = sc.get("per_chain_pair_iptm")
    if pcp is not None: row["per_chain_pair_iptm"] = np.asarray(pcp).squeeze().tolist()
    return row
