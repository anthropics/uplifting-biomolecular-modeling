#!/usr/bin/env python3
"""bz_worker.py — persistent Boltz-2 worker (Layer A): ONE process, model built ONCE, loops over (design, seed) pairs and reproduces the
'single-design stock run' semantics of `boltz predict <one yaml> --seed s` EXACTLY (same RNG streams), without re-importing / re-loading
the model per seed. boltz 2.2.1, inference path only. Every numerics-relevant setting is byte-identical to the stock CLI:
  recycling_steps 3, sampling_steps 200, diffusion_samples 1, max_parallel_samples None, bf16-mixed autocast, float32 matmul 'highest',
  subsample_msa False (CLI default), no templates / constraints / potentials, same checkpoint bytes.

Stock RNG stream per `boltz predict <one yaml> --seed s` (audited in boltz/main.py, pytorch_lightning 2.5.0, torch DataLoader):
  seed_everything(s)                       -> random / numpy / torch CPU generators seeded; PL_GLOBAL_SEED set; PL_SEED_WORKERS unset (workers=False)
  Boltz2.load_from_checkpoint(...)         -> Boltz2.__init__ consumes torch-CPU RNG (nn.init, trunc_normal via scipy -> numpy global RNG);
                                              then load_state_dict overwrites all parameters (RNG already consumed)
  trainer.predict(...)                     -> DataLoader iterator: base_seed = torch.empty((),int64).random_() (CPU generator); worker 0 seeded
                                              base_seed+0 (torch), random.seed, numpy per torch's worker loop; featurization in the worker
                                              (center_random_augmentation -> torch RNG of the worker; numpy featurizer RNG = default_rng(42));
                                              model.to(cuda) (no RNG); forward: MSA subsample OFF (CLI default -> no randperm);
                                              diffusion: torch.randn on the CUDA generator (seeded by seed_everything via torch.manual_seed).
Replay per (design, seed) in this worker:
  seed_everything(s) -> replay the CPU-RNG consumption of Boltz2.__init__ (torch CPU + numpy global; recorded ONCE as the exact number of
  draws by re-running the constructor on the 'meta' device is NOT bit-faithful, so we re-run the real constructor once per seed on CPU
  [~X s] OR, faster, restore the generator states captured right after construction at the first seed and advance them identically —
  both are implemented: --mode fast replays the captured states, --mode slow rebuilds the module per seed] -> fresh 1-item
  DataLoader(num_workers=1, identical args) -> Lightning Trainer.predict with the SAME Trainer/precision/writer objects as stock.
The outputs are stock's own file set per (input, seed), moved under by_seed/<name>/s<seed>/.
"""
import os, sys, json, time, glob, shutil, hashlib, random, datetime, subprocess, socket, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--batch", required=True, help="json: {tag, items:[{name, uid, yaml, seeds:[...]}], seeds, cache, checkpoint, out_dir, write_full_pae}")
ap.add_argument("--mode", default="fast", choices=["fast", "slow"], help="fast = restore captured post-construction RNG states; slow = rebuild the module per seed")
ap.add_argument("--kernels", default="off")
ap.add_argument("--num_workers", type=int, default=1)
ap.add_argument("--pipeline", type=int, default=0, help="1 = pre-featurize item k+1 (DataLoader iterator created under the stock RNG state) while item k predicts; Lightning predict loop replicated in-process")
ap.add_argument("--keep_on_gpu", type=int, default=0, help="1 = skip Lightning Strategy.teardown() .cpu() move (model stays resident; parameter bytes unchanged)")
args = ap.parse_args()
B = json.load(open(args.batch))
T0 = time.time()
def utc(): return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
LOG = {"events": [], "per_item": [], "env": {}}
def ev(evname, **kw):
    LOG["events"].append({"t": round(time.time() - T0, 3), "utc": utc(), "event": evname, **kw}); print(f"[worker {utc()}] {evname} {kw}", flush=True)

import torch
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision("highest")          # as stock main.py
from rdkit import Chem
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
for key in ["CUEQ_DEFAULT_CONFIG", "CUEQ_DISABLE_AOT_TUNING"]:
    os.environ[key] = os.environ.get(key, "1")
from pathlib import Path
from dataclasses import asdict
from pytorch_lightning import Trainer, seed_everything
import boltz.main as BM
from boltz.main import (Boltz2DiffusionParams, PairformerArgsV2, MSAModuleArgs, BoltzSteeringParams, BoltzProcessedInput,
                        filter_inputs_structure)   # check_inputs / process_inputs run in the zygote's child (boltz2_opt.prep.parse_one)
from boltz.data.types import Manifest
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.write.writer import BoltzWriter
from boltz.model.models.boltz2 import Boltz2
import warnings; warnings.filterwarnings("ignore", ".*that has Tensor Cores. To properly utilize them.*")
ev("imports_done", torch=torch.__version__, device=torch.cuda.get_device_name(0))
# ---- BOLTZ-2 TRUNK levers (inert unless BOLTZ_LEVERS is set; stock bytes otherwise) ----
sys.path.insert(0, os.environ.get("BOLTZ_OPT_WORKDIR") or os.getcwd()); import boltz_trunk_levers as _LEV; LOG["env"]["boltz_levers"] = list(_LEV.apply()); ev("levers", levers=LOG["env"]["boltz_levers"])
import boltz_flash_triattn_patch as F2P; LOG['env']['f2_patch_active'] = bool(F2P.apply()); LOG['env']['BOLTZ_TRIATTN'] = os.environ.get('BOLTZ_TRIATTN'); LOG['env']['BOLTZ_TRIATTN_MIN_TOKENS'] = os.environ.get('BOLTZ_TRIATTN_MIN_TOKENS')   # F2 Tier-2 flash tri-attention (active iff BOLTZ_TRIATTN=flash)
import atexit as _ax; _ax.register(lambda: (LOG.__setitem__("lever_report", _LEV.report()), json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)) if "KD" in globals() else None)
# optional: disable inference-time chunking (env BOLTZ_CHUNK=off; stock path untouched otherwise) — same patch as bz_prof.py
if os.environ.get("BOLTZ_CHUNK", "stock") == "off":
    from boltz.data import const as _const
    _const.chunk_size_threshold = 10**9
    from boltz.model.layers.triangular_attention.attention import TriangleAttention as _TA
    _orig_ta_fwd = _TA.forward
    def _ta_fwd(self, x, mask=None, chunk_size=None, use_kernels=False):
        return _orig_ta_fwd(self, x, mask=mask, chunk_size=None, use_kernels=use_kernels)
    _TA.forward = _ta_fwd
    LOG["env"]["chunk_mode"] = "off (chunk_size_threshold=1e9, tri-attn chunk None)"
else:
    LOG["env"]["chunk_mode"] = "stock"
if args.keep_on_gpu:
    from pytorch_lightning.strategies.strategy import Strategy as _PLStrategy
    _orig_teardown = _PLStrategy.teardown
    def _teardown_keep(self):
        # identical to Strategy.teardown minus `self.lightning_module.cpu()`: device placement only, no numerics
        from lightning_fabric.utilities.optimizer import _optimizers_to_device
        _optimizers_to_device(self.optimizers, torch.device("cpu"))
        self.precision_plugin.teardown(); assert self.accelerator is not None; self.accelerator.teardown(); self.checkpoint_io.teardown()
    _PLStrategy.teardown = _teardown_keep
    ev("patch_keep_on_gpu")

CACHE = Path(B.get("cache", "/bw/boltz2")); CKPT = B["checkpoint"]; OUT = Path(B["out_dir"]); OUT.mkdir(parents=True, exist_ok=True); KD = Path(B["kit_dir"]); KD.mkdir(parents=True, exist_ok=True)   # OUT: the predictions (by_seed/); KD: the launch's kit directory (this worker's log, the KERNELS record, the affinity leg's hand-over)
KERN = args.kernels == "on"; PRED = B["predict"]; OPTS = B.get("options") or {}   # the call's boltz-predict settings: predict {recycling_steps, …}, step_scale, write_full_pae/pde, output_format; options = the other boltz predict options the caller GAVE (absent = upstream's default, stated below)
mol_dir = CACHE / "mols"; ccd_path = CACHE / "ccd.pkl"

def stock_model_kwargs():
    diffusion_params = Boltz2DiffusionParams(); diffusion_params.step_scale = float(B["step_scale"])   # main.py: 1.5 for Boltz-2 unless --step_scale
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=bool(OPTS.get("subsample_msa", False)), num_subsampled_msa=int(OPTS.get("num_subsampled_msa", 1024)), use_paired_feature=True)   # main.py:1240-1244; --subsample_msa / --num_subsampled_msa (CLI defaults: flag off, 1024)
    steering_args = BoltzSteeringParams(); steering_args.fk_steering = steering_args.physical_guidance_update = bool(OPTS.get("use_potentials", False))   # main.py:1309-1311: --use_potentials
    predict_args = {"recycling_steps": PRED["recycling_steps"], "sampling_steps": PRED["sampling_steps"], "diffusion_samples": PRED["diffusion_samples"],
                    "max_parallel_samples": PRED["max_parallel_samples"],                     # the call's `boltz predict` settings (batch json "predict": upstream's defaults unless the caller gave them)
                    "write_confidence_summary": True, "write_full_pae": bool(B["write_full_pae"]), "write_full_pde": bool(B["write_full_pde"])}
    return dict(predict_args=predict_args, map_location="cpu", diffusion_process_args=asdict(diffusion_params), ema=False, use_kernels=KERN,
                pairformer_args=asdict(pairformer_args), msa_args=asdict(msa_args), steering_args=asdict(steering_args))

# ---------------- process inputs ONCE per item: stock's check_inputs + process_inputs (parse yaml/a3m, ccd mols, SMILES conformers) in a FRESH child of the
# parsing zygote per input (prep.py) — the process context `boltz predict <one yaml>` parses in: RDKit's conformer RNG is process-global and unseeded upstream,
# so only a process that has embedded nothing before hands a SMILES ligand the conformer stock hands it ----------------
from boltz2_opt import prep as _prep, skipped as SK, writer as WR   # prep: the parsing zygote (worker_launch forks it before CUDA / hooks); skipped: inputs the stock parser skips, named on one SKIPPED line; writer: the output writer off the critical path (registry writer_overlap, BOLTZ_WRITER=overlap, attached by worker_launch) — when it is not installed every WR call below is the stock statement it stands beside
PREP = _prep.zygote(); LOG["env"]["prep"] = PREP.describe()
PROCESS_INPUTS_KW = dict(dict(use_msa_server=False, msa_server_url="https://api.colabfold.com", msa_pairing_strategy="greedy", preprocessing_threads=1, max_msa_seqs=8192), boltz2=True,
                         **{k: OPTS[k] for k in ("use_msa_server", "msa_server_url", "msa_pairing_strategy", "preprocessing_threads", "max_msa_seqs") if k in OPTS})   # main.py predict()'s arguments to process_inputs: the CLI defaults, or the caller's --use_msa_server / --msa_server_url / --msa_pairing_strategy / --preprocessing-threads / --max_msa_seqs
def process_item(it):
    """exactly main.py: out_dir/boltz_results_<stem>/processed/...; one yaml per dir (single-design semantics), parsed in a fresh child of the zygote
    (PREP.parse: check_inputs + process_inputs; its stdout / stderr land in this transcript verbatim). Returns (out_dir, processed, process_inputs_s,
    skip): `skip` is None, or the named reason when stock's parser skipped the input — process_inputs catches its own parse error, prints
    'Failed to process <yaml>. Skipping. Error: <e>.' and leaves the record out of the manifest; such an item is SKIPPED here (one
    '[boltz2-opt] SKIPPED item=<name> reason=...' line, a `skipped` entry in the log) and never reaches a DataLoader or predict_step."""
    data = Path(it["yaml"]); out_dir = OUT / f"boltz_results_{data.stem}"; out_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    res = PREP.parse(str(data), str(out_dir), str(ccd_path), str(mol_dir), **PROCESS_INPUTS_KW)
    sys.stdout.write(res.out); sys.stdout.flush(); sys.stderr.write(res.err); sys.stderr.flush()   # stock's words (and a parse failure's traceback), unchanged
    if res.rc != 0:
        raise RuntimeError(f"parsing {data} in the zygote's child failed (rc {res.rc}); its traceback is above")
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(manifest=manifest, targets_dir=processed_dir / "structures", msa_dir=processed_dir / "msa",
                                    constraints_dir=(processed_dir / "constraints") if (processed_dir / "constraints").exists() else None,
                                    template_dir=(processed_dir / "templates") if (processed_dir / "templates").exists() else None,
                                    extra_mols_dir=(processed_dir / "mols") if (processed_dir / "mols").exists() else None)
    skip = None
    if it["name"] not in [r.id for r in manifest.records]:      # stock wrote no record for this input: its parser skipped it
        skip = SK.reason(SK.stock_error(res.out, res.err, data.stem))
        print(SK.line(it["name"], skip), flush=True)
        LOG.setdefault("skipped", []).append({"name": it["name"], "uid": it.get("uid"), "yaml": str(data), "reason": skip, "utc": utc()})
        ev("item_skipped", item=it["name"][:40], reason=skip[:160])
        json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)
    return out_dir, processed, round(time.time() - t, 3), skip

# ---------------- RNG state capture after a stock-equivalent construction ----------------
def rng_state():
    return {"torch": torch.get_rng_state().clone(), "np": np.random.get_state(), "py": random.getstate(),
            "cuda": torch.cuda.get_rng_state().clone()}
def rng_set(st):
    torch.set_rng_state(st["torch"]); np.random.set_state(st["np"]); random.setstate(st["py"]); torch.cuda.set_rng_state(st["cuda"])
def rng_digest(st):
    h = hashlib.sha256(); h.update(st["torch"].numpy().tobytes()); h.update(repr(st["np"][1].tolist() if hasattr(st["np"][1], "tolist") else st["np"][1]).encode())
    h.update(str(st["np"][2:]).encode()); h.update(repr(st["py"]).encode()); h.update(st["cuda"].numpy().tobytes()); return h.hexdigest()[:16]

def build_model(seed):
    """stock: seed_everything(seed) THEN load_from_checkpoint (constructor consumes RNG) -> returns module on CPU."""
    seed_everything(seed)
    t = time.time()
    m = Boltz2.load_from_checkpoint(CKPT, strict=True, **stock_model_kwargs())
    m.eval()
    return m, round(time.time() - t, 3)

MODEL = None; POST_CTOR = {}   # seed -> rng state right after construction (CPU gens; cuda gen untouched by ctor but captured anyway)
SEEDS_ALL = sorted({s for it in B["items"] for s in it.get("seeds", B.get("seeds", [0]))})
ev("build_start")
# first construction: real, also becomes the persistent module
MODEL, dt = build_model(SEEDS_ALL[0]); POST_CTOR[SEEDS_ALL[0]] = rng_state(); LOG["env"]["build_first_s"] = dt
ev("build_first_done", s=dt, seed=SEEDS_ALL[0], rng=rng_digest(POST_CTOR[SEEDS_ALL[0]]))
# capture post-construction RNG states for the other seeds: the constructor's RNG consumption is the same sequence of draws for every seed
# (pure function of the architecture), so we replay it on a throw-away construction per seed ONCE (slow but exact), then discard.
# fast path: replay the constructor's draw sequence by constructing the module on the 'meta' device is NOT faithful (scipy truncnorm draws
# happen on numpy regardless, but torch draws on meta are skipped), so we keep the exact per-seed construction, done once at startup and
# amortised over all designs.
if args.mode == "fast":
    for s in SEEDS_ALL[1:]:
        seed_everything(s); t = time.time()
        tmp = Boltz2(**{**{k: v for k, v in MODEL.hparams.items() if k not in ("validators",)}})   # same hparams as the checkpoint; RNG draws identical to load_from_checkpoint's construction
        POST_CTOR[s] = rng_state(); del tmp
        ev("ctor_replay", seed=s, s=round(time.time() - t, 2), rng=rng_digest(POST_CTOR[s]))


# ---- per-item model time + peak memory around predict_step (the row dict d; merged into LOG per_item via DIGS["last"]) ----
DIGS = {}
_orig_ps = Boltz2.predict_step
def _ps(self, batch, batch_idx, dataloader_idx=0):
    d = {}; torch.cuda.synchronize(); t = time.time()
    r = _orig_ps(self, batch, batch_idx, dataloader_idx); torch.cuda.synchronize()
    d["model_s"] = round(time.time() - t, 3); d["peak_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    d["upstream_skipped"] = bool(isinstance(r, dict) and r.get("exception"))   # boltz's own catch of a CUDA out-of-memory error (boltz2.py:1123-1128): the writer gets {"exception": True} and writes nothing — the unit is FAILED by name below (SK.upstream_skipped)
    try: d["name"] = batch["record"][0].id
    except Exception: pass
    DIGS["last"] = d
    return r
Boltz2.predict_step = _ps

TRAINER = None
def make_trainer(out_dir, processed, write_embeddings=False):
    pred_writer = BoltzWriter(data_dir=processed.targets_dir, output_dir=out_dir / "predictions", output_format=B["output_format"], boltz2=True, write_embeddings=write_embeddings or bool(OPTS.get("write_embeddings", False)))   # main.py:1252 --write_embeddings
    return Trainer(default_root_dir=out_dir, strategy="auto", callbacks=[pred_writer], accelerator="gpu", devices=1, precision="bf16-mixed",
                   logger=False, enable_progress_bar=False, enable_model_summary=False)

# ---------------- upstream's affinity leg (main.py predict, after the structure pass) for an input declaring `properties: - affinity`: stock's own second model on
# stock's own code path — run VERBATIM in a clean interpreter (boltz2_opt.affinity_leg: no hook, no route, no lever of this process) handed the four RNG streams
# as they stand right after this input's structure pass, where stock's one process reaches that leg; it writes predictions/<id>/affinity_<id>.json ----------------
from boltz2_opt import affinity_leg as AL
def warm_affinity_leg(processed):
    """Ahead of an affinity input's structure pass (the loops below, right after process_item): start the pass's served clean interpreter
    (AL.warm, 0.3.24 — upstream's imports and its CUDA context come up while this process predicts the structure; every (input, seed) leg of
    the pass then runs in that one interpreter); nothing for an input without the property."""
    if any(getattr(r, "affinity", None) for r in processed.manifest.records):
        AL.warm()
def run_affinity_leg(out_dir, processed, it, s):
    """Right after predict for (it, s), before the outputs move: None when the input declares no affinity property, else the leg's seconds.
    A leg that exits non-zero or ends without affinity_<id>.json is named here (event affinity_leg_failed; its transcript is above) and the pass
    goes on with the other units; the parent accounts that unit failed by the missing file (worker.py expected_files: outputs short)."""
    rec = processed.manifest.records[0]
    if not rec.affinity:
        return None
    WR.wait_item(rec.id)                                            # writer_overlap: the leg reads this input's pre_affinity_<id>.npz — its background write lands first (0.0 s when the writer ran inline)
    t = time.time(); ev("affinity_leg_start", item=it["name"][:40], seed=s)
    mk = stock_model_kwargs()
    req = AL.request(out_dir=out_dir, targets_dir=processed.targets_dir, msa_dir=processed.msa_dir, constraints_dir=processed.constraints_dir,
                     template_dir=processed.template_dir, extra_mols_dir=processed.extra_mols_dir, cache=CACHE, mol_dir=mol_dir, num_workers=args.num_workers,
                     diffusion_process_args=mk["diffusion_process_args"], pairformer_args=mk["pairformer_args"], msa_args=mk["msa_args"], **(OPTS.get("affinity") or {}))
    path = AL.write_request(KD / "_affinity_leg" / f"{it['name']}_s{s}", req, rng_state())
    sys.stdout.flush(); sys.stderr.flush()
    rc = AL.execute(path)                                           # the clean interpreter (AL.command): this transcript, this environment (no BOLTZ2_OPT in it), this working directory; at n_gpu > 1 on rank 0 only, the other ranks receive its exit code and result (affinity_leg=rank0_only)
    out_json = out_dir / "predictions" / rec.id / f"affinity_{rec.id}.json"
    dt = round(time.time() - t, 3)
    if rc != 0 or not out_json.is_file():
        if out_json.is_file(): out_json.unlink()                    # a leg that exited non-zero leaves no half result behind: the unit is failed, named
        ev("affinity_leg_failed", item=it["name"][:40], seed=s, rc=rc, s=dt, reason=f"upstream's affinity leg exited {rc}, {out_json.name} absent (its transcript is above)")
        return dt
    ev("affinity_leg_done", item=it["name"][:40], seed=s, s=dt)
    return dt

ITEMS = B["items"]
LOG["env"]["affinity_leg"] = AL.leg_form()                          # `process` (one served clean interpreter per pass for upstream's affinity legs, 0.3.24) | `item` (one per leg): BOLTZ_AFFINITY_LEG

# ---------------- pipelined path: replicate Lightning's predict loop in-process, featurize ahead ----------------
from pytorch_lightning.strategies.single_device import SingleDeviceStrategy  # noqa (import only for parity of module init)
DEV = torch.device("cuda:0")
def make_iter(processed, s):
    """create the DataLoader iterator for (item, seed s) under the STOCK RNG state: base_seed draw == stock's; worker featurizes in the background."""
    dm = Boltz2InferenceDataModule(manifest=processed.manifest, target_dir=processed.targets_dir, msa_dir=processed.msa_dir, mol_dir=mol_dir,
                                   num_workers=args.num_workers, constraints_dir=processed.constraints_dir, template_dir=processed.template_dir,
                                   extra_mols_dir=processed.extra_mols_dir, override_method=OPTS.get("method"))
    rng_set(POST_CTOR[s])
    dl = dm.predict_dataloader(); it_ = iter(dl)                # draws base_seed from the CPU generator exactly as Lightning's fetcher does
    st = rng_state()                                             # RNG state right after iterator creation (== stock state at predict_step entry)
    return dm, dl, it_, st

def predict_pipelined(dm, it_, st, out_dir, processed):
    writer = BoltzWriter(data_dir=processed.targets_dir, output_dir=out_dir / "predictions", output_format=B["output_format"], boltz2=True, write_embeddings=bool(OPTS.get("write_embeddings", False)))   # main.py:1252 --write_embeddings
    t_handoff = time.time()
    try:
        batch = next(it_)                                        # featurized in the worker (seeded base_seed+0 by torch's worker loop, as stock)
    except StopIteration:                                        # an input the parser skipped never reaches here (process_item names it SKIPPED); anything else is a defect, named
        raise RuntimeError("the DataLoader yielded no batch for an input whose record stock's manifest carries") from None
    t_handoff = time.time() - t_handoff
    rng_set(st)                                                  # CPU/numpy/python/CUDA generators == stock at this point
    with torch.inference_mode():                                 # Trainer.predict default inference_mode=True
        t_h2d = time.time()
        batch = dm.transfer_batch_to_device(batch, DEV, 0)       # strategy.batch_to_device -> datamodule hook (stock)
        t_h2d = time.time() - t_h2d
        with torch.autocast("cuda", dtype=torch.bfloat16):       # precision="bf16-mixed" predict_step_context (stock)
            pred = MODEL.predict_step(batch, 0, 0)
        t_write = time.time()
        writer.write_on_batch_end(None, MODEL, pred, None, batch, 0, 0)   # stock's writer call; under BOLTZ_WRITER=overlap the class method is the kit's (boltz2_opt.writer: pinned D2H staging here, the same function on the host copies in the writer process)
        t_write = time.time() - t_write
    DIGS.setdefault("last", {}).update(handoff_s=round(t_handoff, 3), h2d_s=round(t_h2d, 3), write_call_s=round(t_write, 3))   # host time of the non-forward segments of this item (the writer's is the whole write inline, the staging under writer_overlap)
    try:
        next(it_)
    except StopIteration:
        pass
    return pred

if args.pipeline:
    MODEL.to(DEV); MODEL.eval()
    def writer_collect():
        """writer_overlap: finished background writes / moves since the last call -> their item rows (files, bg_write_s, bg_move_s); a write or move
        that raised -> the (input, seed) unit under `failed` by the writer's reason (the parent prints FAILED item=… by name; the exit is non-zero)."""
        rows = {(r["name"], r["seed"]): r for r in LOG["per_item"]}
        for c in WR.collect():
            row = rows.get((c["name"], c["seed"]))
            if row is not None:
                if c["kind"] == "write": row["bg_write_s"] = c.get("write_s")
                else: row["files"] = c.get("files"); row["bg_move_s"] = c.get("move_s"); row["writer"] = "done" if c["ok"] else "failed"
            if not c["ok"]:
                uid = next((i.get("uid") for i in ITEMS if i["name"] == c["name"]), None)
                LOG.setdefault("failed", []).append({"name": c["name"], "uid": uid, "seed": c["seed"], "reason": f"writer_overlap {c['kind']} failed: {c.get('error')}", "utc": utc()})
                ev("item_failed", item=str(c["name"])[:40], seed=c["seed"], reason=f"writer_overlap:{c['kind']}:{c.get('error')}"[:200])
    pairs = [(it, s) for it in ITEMS for s in it.get("seeds", B.get("seeds", [0]))]
    PROC = {}
    def get_proc(it):
        if it["name"] not in PROC:
            PROC[it["name"]] = process_item(it)
            if PROC[it["name"]][3] is None: warm_affinity_leg(PROC[it["name"]][1])   # an affinity input: the pass's served interpreter starts ahead of its structure pass
        return PROC[it["name"]]
    def ready(k):
        """the first pair at or after k whose input stock's parser accepted (process_item names a skipped one as it meets it), or len(pairs)."""
        return SK.next_ready(k, len(pairs), lambda i: get_proc(pairs[i][0])[3] is not None)
    # prime: iterator for the first ready pair
    k = ready(0)
    if k < len(pairs):
        it0, s0 = pairs[k]; out_dir0, processed0, _, _ = get_proc(it0); nxt = make_iter(processed0, s0)
    while k < len(pairs):
        it, s = pairs[k]
        t_item = time.time(); out_dir, processed, t_proc, _ = get_proc(it)
        dm, dl, it_, st = nxt
        # create the NEXT ready pair's iterator now so its featurization overlaps this prediction
        k2 = ready(k + 1)
        if k2 < len(pairs):
            it2, s2 = pairs[k2]; od2, pr2, _, _ = get_proc(it2); nxt = make_iter(pr2, s2)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t_pred = time.time(); DIGS["last"] = {}
        WR.begin_item(it["name"], s)                                 # writer_overlap: the background write of this prediction carries its (input, seed) unit
        predict_pipelined(dm, it_, st, out_dir, processed)
        torch.cuda.synchronize(); t_pred = time.time() - t_pred
        del it_, dl, dm
        why_failed = SK.upstream_skipped(DIGS.get("last"))          # boltz's predict_step skipped this batch on CUDA out of memory (its WARNING line is above): the unit is FAILED by that name, the pass goes on as stock's does
        if why_failed:
            LOG.setdefault("failed", []).append({"name": it["name"], "uid": it.get("uid"), "seed": s, "reason": why_failed, "utc": utc()}); ev("item_failed", item=it["name"][:40], seed=s, reason=why_failed)
        t_aff = run_affinity_leg(out_dir, processed, it, s)          # upstream's affinity leg when the input declares the property (None otherwise)
        pred = out_dir / "predictions" / it["name"]; dst = OUT / "by_seed" / it["name"] / f"s{s}"; dst.mkdir(parents=True, exist_ok=True)
        files = WR.finish_item(it["name"], s, pred, dst)             # stock: every file under predictions/<name>/ moved under by_seed/<name>/s<seed>/ now (the list); writer_overlap: queued behind this prediction's background write (None: the list arrives with WR.collect())
        LOG["per_item"].append({"name": it["name"], "uid": it.get("uid"), "seed": s, "predict_s": round(t_pred, 3), "item_s": round(time.time() - t_item, 3), **DIGS.get("last", {}),
                                "process_inputs_s": t_proc, "affinity_s": t_aff, "files": [os.path.basename(f) for f in files] if files is not None else None, "finished_utc": utc(), "pipeline": 1,
                                **({"writer": "pending"} if files is None else {})})
        ev("item_done", item=it["name"][:40], seed=s, predict_s=round(t_pred, 2), model_s=DIGS.get("last", {}).get("model_s"))
        writer_collect()
        json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)
        k = k2
    WCENSUS = WR.join(); writer_collect(); LOG["writer_census"] = WCENSUS   # every background write / move has landed or is failed by name before the log is final and EXIT is said
    LOG["total_wall_s"] = round(time.time() - T0, 1)
    json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)
    ev("done", total_s=LOG["total_wall_s"], writer_failed=WR.failed_count()); sys.exit(1 if WR.failed_count() else 0)   # a background write that raised fails the run (its units are under `failed` by name)

for it in ITEMS:
    out_dir, processed, t_proc, skip = process_item(it)
    if skip:                                                     # named SKIPPED by process_item; no DataLoader, no predict_step for it
        continue
    warm_affinity_leg(processed)                                 # an affinity input: the pass's served interpreter for upstream's affinity legs starts ahead of its structure pass
    for s in it.get("seeds", B.get("seeds", [0])):
        t_item = time.time()
        # --- exact stock RNG position: state right after construction under seed s ---
        rng_set(POST_CTOR[s])
        # stock: load_from_checkpoint already done (weights are the same bytes); predict_args identical -> MODEL reused as-is
        data_module = Boltz2InferenceDataModule(manifest=processed.manifest, target_dir=processed.targets_dir, msa_dir=processed.msa_dir, mol_dir=mol_dir,
                                                num_workers=args.num_workers, constraints_dir=processed.constraints_dir, template_dir=processed.template_dir,
                                                extra_mols_dir=processed.extra_mols_dir, override_method=OPTS.get("method"))
        trainer = make_trainer(out_dir, processed)
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t_pred = time.time(); DIGS["last"] = {}
        trainer.predict(MODEL, datamodule=data_module, return_predictions=False)
        torch.cuda.synchronize(); t_pred = time.time() - t_pred
        why_failed = SK.upstream_skipped(DIGS.get("last"))          # boltz's predict_step skipped this batch on CUDA out of memory (its WARNING line is above): the unit is FAILED by that name, the pass goes on as stock's does
        if why_failed:
            LOG.setdefault("failed", []).append({"name": it["name"], "uid": it.get("uid"), "seed": s, "reason": why_failed, "utc": utc()}); ev("item_failed", item=it["name"][:40], seed=s, reason=why_failed)
        t_aff = run_affinity_leg(out_dir, processed, it, s)          # upstream's affinity leg when the input declares the property (None otherwise)
        # stock names the output dir by record id; with one record per yaml the files land at predictions/<name>/...; move to a per-seed dir
        pred = out_dir / "predictions" / it["name"]; dst = OUT / "by_seed" / it["name"] / f"s{s}"; dst.mkdir(parents=True, exist_ok=True)
        files = sorted(glob.glob(str(pred / "*")))
        for f in files: shutil.move(f, dst / os.path.basename(f))
        LOG["per_item"].append({"name": it["name"], "uid": it.get("uid"), "seed": s, "predict_s": round(t_pred, 3), "item_s": round(time.time() - t_item, 3), **DIGS.get("last", {}),
                                "process_inputs_s": t_proc, "affinity_s": t_aff, "files": [os.path.basename(f) for f in files], "finished_utc": utc()})
        ev("item_done", item=it["name"][:40], seed=s, predict_s=round(t_pred, 2), model_s=DIGS.get("last", {}).get("model_s"))
        json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)
LOG["total_wall_s"] = round(time.time() - T0, 1)
json.dump(LOG, open(KD / f"{B['tag']}_worker_log.json", "w"), indent=1)
ev("done", total_s=LOG["total_wall_s"])
