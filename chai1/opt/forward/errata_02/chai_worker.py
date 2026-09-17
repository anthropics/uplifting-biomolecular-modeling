"""chai_worker.py — Chai-1 (chai_lab 0.6.1) persistent worker. Usage-level levers only; the model code that runs per seed is
chai_lab.chai1.run_folding_on_context UNCHANGED (same exported TorchScript modules, same crop size, same feature factory/collate,
same set_seed(seed) -> same torch RNG stream for the diffusion noise, same 3 recycles / 200 steps / 1 sample).

Levers (each independently switchable):
  W1  resident modules      chai1.load_exported is memoised: the 6 TorchScript components are torch.jit.load'ed once per process
                            instead of once per seed (stock reloads them from disk on every run_inference call). The ESM-2 3B fp16
                            traced model is likewise kept on the GPU between calls instead of being shuttled GPU->CPU->GPU.
  W2  feature-context reuse  make_all_atom_feature_context(fasta, msa_dir, esm) is computed ONCE per input and the same
                            AllAtomFeatureContext object is passed to run_folding_on_context for every seed. Justification (exact):
                            in stock the context is built BEFORE set_seed(seed) is called inside run_folding_on_context, from
                            deterministic inputs (FASTA parse, RDKit reference conformer lookup from the cached conformer library,
                            MSA parquet load + deterministic pairing, ESM forward); it does not depend on the seed.
  W5  cross-input ESM memo   per-sequence ESM embeddings are memoised across the inputs of one process.
  Several uids in one invocation share the process (python import + weight load + ESM load amortised across inputs); one uid per
  invocation disables that sharing.
Not levers (unchanged): MSA content/depth, recycles, diffusion steps, samples, seeds, crop sizes, precision, kernels.

Process-global state: run_folding_on_context calls set_seed([seed]) itself at entry (numpy/torch/python RNG fully re-seeded per
seed exactly as stock); torch numerics flags are never touched by this file. torch.jit fusion strategy is set by stock load_exported on
first load (same call, same value).

usage: python chai_worker.py <uid>[,<uid>...] <seeds> <out_root> [--tag worker] [--levels W1,W2,W5]   (uid: fasta:<path> | <x>.fasta)
       --levels W1        : resident modules only; features rebuilt per seed (stock make_all_atom_feature_context per seed)
       --levels W1,W2     : + feature/ESM context reuse across seeds
       --levels W1,W2,W5  : + the cross-input ESM memo (default)
"""
import os, sys, json, time, argparse, traceback, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
# the worker lives outside kit/: the kit module chai_proto is importable via $KIT
if os.environ.get("KIT"): sys.path.insert(0, os.environ["KIT"])
import chai_proto as cp

ap = argparse.ArgumentParser()
ap.add_argument("uids"); ap.add_argument("seeds"); ap.add_argument("out_root")
ap.add_argument("--tag", default="worker")
ap.add_argument("--levels", default="W1,W2,W5", help="comma list of levers: W1 W2 W5 (default: all three)")
ap.add_argument("--msa_dir", default="msa_local")
a = ap.parse_args()
levels = set(a.levels.split(","))
if levels - {"W1", "W2", "W5"}: ap.error(f"unknown lever(s) {sorted(levels - {'W1', 'W2', 'W5'})}: the worker's levers are W1, W2, W5")
uids = a.uids.split(","); seeds = [int(s) for s in a.seeds.split(",")]

t_proc0 = time.perf_counter()
import numpy as np, torch
import chai_lab
from chai_lab import chai1
import chai_lab.data.dataset.embeddings.esm as esm_mod

device = torch.device(cp.RUN_KW.get("device") or "cuda:0")                    # run_inference's torch_device: --device, else cuda:0
_opt_path = lambda v: Path(v) if v else None                                   # RUN_KW carries paths as strings (None = not given)

# ---------------- W1: resident exported modules + resident ESM ----------------
_MODULE_CACHE = {}
_orig_load_exported = chai1.load_exported
LOAD_LOG = []
def load_exported_cached(comp_key, device):
    k = (comp_key, str(device))
    if k not in _MODULE_CACHE:
        t0 = time.perf_counter()
        _MODULE_CACHE[k] = _orig_load_exported(comp_key, device)   # stock loader (sets the same jit fusion strategy)
        _MODULE_CACHE[k]._label = comp_key.replace(".pt", "")
        LOAD_LOG.append((comp_key, round(time.perf_counter() - t0, 2)))
    return _MODULE_CACHE[k]

if "W1" in levels:
    chai1.load_exported = load_exported_cached
    # resident ESM: stock esm_model() context manager moves the traced 3B model back to CPU after every use; keep it on GPU.
    from contextlib import contextmanager
    _orig_esm_model = esm_mod.esm_model
    @contextmanager
    def esm_model_resident(device):
        local_esm_path = esm_mod.downloads_path.joinpath("esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt")
        esm_mod.download_if_not_exists(esm_mod.ESM_URL, local_esm_path)
        if len(esm_mod._esm_model) == 0:
            t0 = time.perf_counter()
            if device != torch.device("cuda:0"):
                model = torch.jit.load(local_esm_path, map_location="cpu").to(device)
            else:
                model = torch.jit.load(local_esm_path).to(device)
            esm_mod._esm_model.append(model); LOAD_LOG.append(("esm2_3B", round(time.perf_counter() - t0, 2)))
        [model] = esm_mod._esm_model
        model.to(device); model.eval()
        yield model            # stock yields the same object in the same state; here it is not moved back to CPU afterwards
    esm_mod.esm_model = esm_model_resident

# ---------------- W5: exact per-sequence ESM embedding cache across inputs (a chain that repeats across inputs is embedded once) ----------------
# stock computes each unique protein sequence with batch size 1 (token_ids[None, :]) through the traced fp16 ESM2-3B and stores the fp32 CPU
# tensor; the same (model, input tensor) pair gives the same bits, so memoising the per-sequence EmbeddingContext changes nothing.
ESM_CACHE = {}; ESM_STATS = dict(hits=0, misses=0)
if "W5" in levels:
    _orig_get_ctx = esm_mod._get_esm_contexts_for_sequences
    def _get_ctx_cached(prot_sequences, device):
        need = {s for s in prot_sequences if s not in ESM_CACHE}
        ESM_STATS["hits"] += len(prot_sequences) - len(need); ESM_STATS["misses"] += len(need)
        if need:
            ESM_CACHE.update(_orig_get_ctx(prot_sequences=need, device=device))
        return {s: ESM_CACHE[s] for s in prot_sequences}
    esm_mod._get_esm_contexts_for_sequences = _get_ctx_cached

DET_MODE = cp.apply_deterministic_mode()
env = cp.env_report(); env["deterministic_mode"] = DET_MODE
env["jit_profiling_mode"] = bool(DET_MODE == "off" or os.environ.get("CHAI_JIT_PROFILING_OFF", "1") == "0")
cp.log("ENV", json.dumps(env)); cp.log("LEVELS", sorted(levels), "uids", len(uids), "seeds", seeds)
out_root = Path(a.out_root); (out_root / a.tag).mkdir(parents=True, exist_ok=True)
summary = []
for di, uid in enumerate(uids):
    t_input0 = time.perf_counter()
    spec = cp.input_spec(uid); key = spec["key"]
    keep = out_root / a.tag / key; keep.mkdir(parents=True, exist_ok=True)
    work = Path("work_chai") / f"{a.tag}__{key}"
    if work.exists():
        import shutil; shutil.rmtree(work)
    work.mkdir(parents=True)
    fasta = Path(cp.write_fasta(spec, work / f"{key}.fasta"))
    msa_dir = cp.ensure_msa_pqt(spec, a.msa_dir)
    rows = []; t_feat = None; feature_context = None
    for s in seeds:
        row = dict(uid=uid, key=key, tag=a.tag, levels=",".join(sorted(levels)), seed=s, status="error", error="",
                   gpu=env["gpu"], chai_lab=env["chai_lab"], torch=env["torch"], input_index_in_process=di)
        t0 = time.perf_counter()
        try:
            od = work / f"seed_{s}"
            if feature_context is None or "W2" not in levels:
                tf0 = time.perf_counter()
                # identical call + args to run_inference's internal call (its keywords from RUN_KW; esm_device = torch_device)
                feature_context = chai1.make_all_atom_feature_context(
                    fasta_file=fasta, output_dir=od, use_esm_embeddings=cp.RUN_KW["use_esm_embeddings"],
                    use_msa_server=cp.RUN_KW["use_msa_server"], msa_server_url=cp.RUN_KW["msa_server_url"], msa_directory=msa_dir,
                    constraint_path=_opt_path(cp.RUN_KW["constraint_path"]), use_templates_server=cp.RUN_KW["use_templates_server"],
                    templates_path=_opt_path(cp.RUN_KW["template_hits_path"]), esm_device=device)
                torch.cuda.synchronize(); t_feat = time.perf_counter() - tf0
                row["features_s"] = round(t_feat, 3)
            else:
                row["features_s"] = 0.0; row["features_reused_from_seed"] = seeds[0]
            # identical to run_inference's body after the feature context (chai1.py:526-540): its loop over trunk samples, verbatim —
            # one run_folding_on_context call per trunk sample into trunk_<i>/ (the seed directory itself at one sample), then concat
            n_trunk = int(cp.RUN_KW["num_trunk_samples"])
            all_candidates = []
            for trunk_idx in range(n_trunk):
                cand = chai1.run_folding_on_context(
                    feature_context, output_dir=(od / f"trunk_{trunk_idx}" if n_trunk > 1 else od),
                    num_trunk_recycles=cp.RUN_KW["num_trunk_recycles"], num_diffn_timesteps=cp.RUN_KW["num_diffn_timesteps"],
                    num_diffn_samples=cp.RUN_KW["num_diffn_samples"], recycle_msa_subsample=cp.RUN_KW["recycle_msa_subsample"],
                    seed=s + trunk_idx if s is not None else None, device=device, low_memory=cp.RUN_KW["low_memory"])
                all_candidates.append(cand)
            cand = chai1.StructureCandidates.concat(all_candidates)
            torch.cuda.synchronize()
            row["wall_s"] = time.perf_counter() - t0
            row.update(cp.save_seed_outputs(cand, od, keep / f"seed_{s}"))
            row["n_tokens"] = int(cand.pae[0].shape[0]); row["n_trunk_samples"] = n_trunk; row["status"] = "ok"
        except Exception as e:
            row["wall_s"] = time.perf_counter() - t0
            row["error"] = (repr(e) + " | " + traceback.format_exc()[-1500:])[:2000]
            cp.log("SEED FAIL", uid, s, row["error"])
            feature_context = None
        row["max_mem_alloc_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        rows.append(row)
        cp.log("SEED", uid, s, row["status"], "iptm", row.get("iptm"), "wall", round(row["wall_s"], 1), "feat", row.get("features_s"))
    proc = dict(uid=uid, key=key, tag=a.tag, levels=sorted(levels), input_wall_s=time.perf_counter() - t_input0,
                n_ok=sum(r["status"] == "ok" for r in rows), n=len(rows), trunk_samples=int(cp.RUN_KW["num_trunk_samples"]), load_log=LOAD_LOG[:], env=env,
                input_index_in_process=di, esm_cache=dict(ESM_STATS), process_wall_so_far_s=time.perf_counter() - t_proc0)
    summary.append(proc)
    import shutil; shutil.rmtree(work, ignore_errors=True)
    del feature_context; gc.collect(); torch.cuda.empty_cache()
    cp.log("INPUT DONE", key, proc["n_ok"], "/", len(rows), "input_wall", round(proc["input_wall_s"], 1))
cp.log("WORKER DONE", len(uids), "inputs", "process_wall", round(time.perf_counter() - t_proc0, 1))
sys.exit(0 if all(p["n_ok"] == p["n"] for p in summary) else 2)
