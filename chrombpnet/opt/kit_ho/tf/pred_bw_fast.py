#!/usr/bin/env python
"""Drop-in for `chrombpnet pred_bw [-cm <model.h5>] [-cmb <nobias.h5>] [-bm <bias.h5>] -r <regions> -g <genome.fa> -c <chrom.sizes> -op <prefix> [-bs N] [-os <stats>]
[-bw <observed.bw>] [-t 0|1]` using chrombpnet_fastkit — OUTPUT PARITY: every file the stock route writes for the user is written here, byte-identical, per model
given, in stock's order and with stock's suffixes (<sfx> = _chrombpnet_nobias for -cmb, _chrombpnet for -cm, _bias for -bm):
  <prefix><sfx>_preds.bed         (regions actually predicted; the stock's to_csv on the same frame)
  <prefix><sfx>.bw                (the bigWig, stock block layout)
  -os <file>                      (the quantile stats on the same all_entries, the stock's format; as in stock, each model rewrites it and the last one given stands)
  -bw <observed.bw>               -> <prefix><sfx>_predictions.h5, <prefix><sfx>_metrics.json, <prefix><sfx>.counts_pearsonr.png,
                                     <prefix><sfx>.profile_jsd.png — by the kit's metrics stage (stock's arithmetic) on the kit's (bitwise) heads.
The -cm model runs the route the class tables give (K1 on H100 / H200); -cmb and -bm run stock's own graph inside the same pipeline (the TensorFlow route).
The run record (what ran per item: route, levers, units in / dropped / written, timings) goes to the file named by CHROMBPNET_FASTKIT_RECORD — the package sets it
and folds the record into opt_manifest.json; a bare run keeps it in a temporary file named on its last line — never beside the outputs. -d / --debug-chr is
REFUSED by name: upstream 1.0.1's pred_bw does not run with it (a known upstream issue, README) and no fix ships. -t (tqdm progress) is accepted and ignored
(progress only). The script takes the stock arguments (and --items); the levers are composed from the detected GPU class and the kit's tables
(chrombpnet_fastkit.fastdefault), not from flags."""
import os, sys, json, argparse
import time as _time_p; _T_PROC0 = _time_p.time()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# this kit directory (opt/kit_ho) carries tf/ (the entry script, the chrombpnet_fastkit package, det_subprocess); the torch-side package
# and the driver-cache tarballs are read from opt/kit (_BASE_KIT) — one copy of those bytes.
_BASE_KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kit")
if not os.path.isdir(os.path.join(_BASE_KIT, "torch", "chrombpnet_k1")): raise SystemExit("[pred_bw_fast] opt/kit_ho: the carried kit is not beside this directory at %s — refused" % _BASE_KIT)
from chrombpnet_fastkit import malloc_env as _malloc_env                     # the malloc env arm — mallopt + env export before numpy/TF load (stdlib only)
_MALLOC_STAMP = _malloc_env.ensure()
from chrombpnet_fastkit._oom import is_oom                                   # the kit's one out-of-memory classifier: a handler below that reroutes re-raises an OOM first
from chrombpnet_fastkit import png_workers as _png_workers                    # the -bw finish stage's PNG helpers (spawned below, after the refusals)
ap = argparse.ArgumentParser()
ap.add_argument("-cm", "--chrombpnet-model", default=None); ap.add_argument("-r", "--regions", default=None); ap.add_argument("-g", "--genome", required=True)
ap.add_argument("-c", "--chrom-sizes", required=True); ap.add_argument("-op", "--output-prefix", default=None); ap.add_argument("-bs", "--batch-size", type=int, default=64)
ap.add_argument("-os", "--output-prefix-stats", default=None); ap.add_argument("-bw", "--bigwig", default=None, help="observed bigWig -> predictions.h5 + metrics (the stock's compare_with_observed)")
ap.add_argument("-t", "--tqdm", type=int, default=1); ap.add_argument("-d", "--debug-chr", nargs="+", default=None); ap.add_argument("-bm", "--bias-model", default=None); ap.add_argument("-cmb", "--chrombpnet-model-nb", default=None)
ap.add_argument("--items", default=None, help="many items in one process (the package's `pred_bw --items` form): one item per line `regions<TAB>output_prefix[<TAB>stats]`; -r/-op/-os refused with it; the entry script's default composition only")
args = ap.parse_args()
# the models given, in stock's order and with stock's output suffixes (predict_to_bigwig.main: -cmb, then -cm, then -bm; at least one, as stock asserts)
_MODELS = [(p, sfx) for p, sfx in ((args.chrombpnet_model_nb, "_chrombpnet_nobias"), (args.chrombpnet_model, "_chrombpnet"), (args.bias_model, "_bias")) if p]
if not _MODELS: sys.exit("[pred_bw_fast] no input model: at least one of -bm, -cm, -cmb is required (as stock)")
if args.debug_chr: sys.exit("[pred_bw_fast] REFUSED: -d / --debug-chr — upstream chrombpnet 1.0.1's pred_bw does not run with it (predict_to_bigwig.py keeps no region: each region's chromosome is compared with the whole -d list, and the bigWig writer then fails with ValueError 'need at least one array to concatenate'); no fix ships for this debugging flag (README: Known upstream issues)")
_EXTRA = [m for m in _MODELS if m[1] != "_chrombpnet"]   # -cmb / -bm: stock's own graph inside the kit's pipeline (the TensorFlow route), whatever the -cm route
_JIT_TAR = os.environ.get("CHROMBPNET_JIT_CACHE_TAR")   # the driver-cache tarball the package stages for this class (chrombpnet_opt.stack); else the kit's own cache/ below
_RECORD_PATH = os.environ.get("CHROMBPNET_FASTKIT_RECORD") or None   # the package's handshake: the file this process's run record goes to (cli.run_fast folds it into opt_manifest.json); None = a bare run
_RECORD = {"kit": "chrombpnet_fastkit", "items": []}
def _write_record():
    """The run record as ONE JSON document (every finished item so far), rewritten whole per item (tmp + rename): the package reads it after this process exits and
        exit_fast re-opens and parses it before leaving. A bare run (no CHROMBPNET_FASTKIT_RECORD) keeps it in a temporary file, never beside the outputs."""
    global _RECORD_PATH
    if _RECORD_PATH is None:
        import tempfile as _tf_rec; _fd_rec, _RECORD_PATH = _tf_rec.mkstemp(prefix="pred_bw_fast_record_", suffix=".json"); os.close(_fd_rec); _RECORD["bare_run"] = True
    with open(_RECORD_PATH + ".tmp", "w") as _fh_rec: json.dump(_RECORD, _fh_rec, indent=1, default=str)
    os.replace(_RECORD_PATH + ".tmp", _RECORD_PATH); return _RECORD_PATH
def _refuse_mode(reason):
    """A lever of this class's `fast` set cannot run on this host (its stack does not import, its kernels do not start, its inputs are absent): the mode refuses
        BY NAME — exit 3, the reason on the line and in the run record — never a run under the mode's name with a subset of its levers. An untested
        environment (an unlisted card, a library off its pin, a cache miss) is not this case: those engage and are named."""
    print(f"[pred_bw_fast] REFUSED: {reason} — `fast` on this class is all of its levers or nothing; `--mode off` runs stock", flush=True)
    _RECORD["refused"] = reason; _write_record()
    raise SystemExit(3)
# --- the multi-item loop (--items): the item list, its refusals by name (the NOT ACTIVE line first on stdout, exit 3 = the package's code), the one-item form otherwise
def _multi_refuse(reason):
    sys.stdout.write("[pred_bw_fast] NOT ACTIVE multi: %s — refused\n" % reason); sys.stdout.flush(); sys.stderr.flush(); sys.exit(3)
_MULTI_REFUSED = ("-r", "--regions", "-op", "--output-prefix", "-os", "--output-prefix-stats", "-bm", "--bias-model", "-cmb", "--chrombpnet-model-nb", "-d", "--debug-chr")
def _read_items(path):
    items = []; seen = set()
    with open(path) as fh:
        for ln, line in enumerate(fh, 1):
            s = line.rstrip("\n")
            if not s.strip() or s.lstrip().startswith("#"): continue
            cols = s.split("\t")
            if len(cols) < 2 or not cols[0] or not cols[1]: _multi_refuse("%s:%d needs `regions<TAB>output_prefix[<TAB>stats]`" % (path, ln))
            if cols[1] in seen: _multi_refuse("output prefix %s named twice in %s" % (cols[1], path))
            if not os.path.isfile(cols[0]): _multi_refuse("regions file %s (item at line %d) not found" % (cols[0], ln))
            seen.add(cols[1]); items.append((cols[0], cols[1], (cols[2] if len(cols) > 2 and cols[2] else None)))
    if not items: _multi_refuse("no items in %s" % path)
    return items
_MULTI_CTX = None
if args.items:
    for _f in _MULTI_REFUSED:
        if _f in sys.argv[1:]: _multi_refuse("%s is not served by the multi-item loop (every item names its own regions / prefix / stats; the entry script's default composition only)" % _f)
    _ITEMS = _read_items(args.items); _MULTI_CTX = {"n_items": len(_ITEMS), "items_file": os.path.abspath(args.items)}
    args.regions, args.output_prefix, args.output_prefix_stats = _ITEMS[0]      # the first item stands in for the setup's per-item reads (the .fai check, the route)
else:
    if not args.regions or not args.output_prefix: sys.exit("[pred_bw_fast] REFUSED: -r/--regions and -op/--output-prefix are required (or --items for the multi-item loop)")
    _ITEMS = [(args.regions, args.output_prefix, args.output_prefix_stats)]
# the -bw finish stage's two PNG helpers are SPAWNED here — after the argument checks, before the kit's numeric
# imports and the model load (their own imports overlap it): fresh interpreters (png_workers.py), never a fork of this process — the fork-after-TF hazard;
# on the deterministic arm TensorFlow is already imported by the recipe's seed hook before the first script line, so no fork-before-TF point exists.
_PNG_WORKERS = _png_workers.start() if args.bigwig else None
# the process's ONE bigWig writer interpreter is pre-spawned here — beside the PNG helpers, before the kit's numeric imports, the TF probe and
# the torch stack (its own imports overlap the model load); every item binds to it (chrombpnet_fastkit.predict_pipelined).
import chrombpnet_fastkit as _kit_ps
_WRITER_PS = _kit_ps.prespawn_writer()
print("[pred_bw_fast] helpers + writer spawned at +%.3fs" % (_time_p.time() - _T_PROC0), flush=True)
import numpy as np, pandas as pd, glob
import time as _time_c
_t_ki = _time_c.time()
import chrombpnet_fastkit as kit
_T_KIT_IMPORT = _time_c.time() - _t_ki
_T_IMPORT_TF = None   # TensorFlow is imported ONLY on the TF routes, after the route decision (below) — dead work on the K1 route
# the route, the native-dilation default and the warm-up come from the detected class + the class tables (fastdefault.resolve); no flag or env var chooses a lever
from chrombpnet_fastkit import fastdefault as _fd
_FD = _fd.resolve(kit_root=_BASE_KIT)
if _FD.get("k1_wanted") and _FD["forward"] != "k1":   # the class tables put the K1 forward in this mode's lever set and it cannot start here (its stack does not import, the tree lacks the package)
    _refuse_mode(next((s for s in _FD.get("skipped") or [] if str(s).startswith("K1 route: OFF")), _FD.get("forward_reason")))
_TF_FORWARD = _FD["forward"] if _FD["forward"] != "k1" else "tf_function"   # this class and mode's TensorFlow route: -cmb / -bm run it, and the whole line when no -cm is given (K1 composes the -cm model only)
if _FD["forward"] == "k1" and not args.chrombpnet_model:
    _why = "K1: not used — no -cm model on the line (K1 composes the -cm model; -cmb / -bm run stock's own graph)"
    _FD["forward"] = _TF_FORWARD; _FD["forward_reason"] = _why + " | " + _FD["forward_reason"]; _FD["skipped"].append(_why)
if _FD["forward"] == "k1" and _EXTRA and not os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH"):   # K1 (torch) and stock's graph (TensorFlow) share the device in this process: TensorFlow allocates as it goes instead of reserving the whole card at its first op
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"; print("[pred_bw_fast] -cmb / -bm beside the K1 -cm model: TF_FORCE_GPU_ALLOW_GROWTH=true for this process (TensorFlow and torch share the device)", flush=True)
if _FD["forward"] != "k1" or _EXTRA:
    _t_tf = _time_c.time()
    import tensorflow as _tf_probe   # the TF import timed here — a process term the stock pays too; TF routes only
    _T_IMPORT_TF = _time_c.time() - _t_tf
# the K1 route's COMPONENT MODELS (bias_scaled.h5 + nobias.h5) are derived beside -cm (the ENCODE model dir's layout); not found -> the mode refuses by
# name (K1 is in this class's lever set and cannot run without them)
_K1_BIAS = _K1_NOBIAS = None
if _FD["forward"] == "k1":
    _mdir = os.path.dirname(os.path.abspath(args.chrombpnet_model))
    def _find(names, pat):
        for nm in names:
            if os.path.exists(os.path.join(_mdir, nm)): return os.path.join(_mdir, nm)
        g = sorted(glob.glob(os.path.join(_mdir, pat))); return g[0] if len(g) == 1 else None
    _b = _find(("bias_scaled.h5",), "*bias_scaled*.h5"); _n = _find(("nobias.h5", "chrombpnet_nobias.h5"), "*nobias*.h5")
    if _b and _n: _K1_BIAS, _K1_NOBIAS = _b, _n; _FD["k1_component_models"] = {"bias": _b, "nobias": _n, "source": "beside -cm"}
    else: _refuse_mode(f"K1 (the -cm forward of `fast` on {_FD['gpu_class']}) needs the component models bias_scaled.h5 + nobias.h5 beside -cm ({_mdir}); they are not there")
_FORWARD = _FD["forward"]; _ND = bool(_FD["native_dilation"]) if _FORWARD != "k1" else False   # the route and the native-dilation state the tables give this class and mode
# the tail form is DEVICE-KEYED (fastdefault.TAIL_CLASS_TABLE); chrombpnet_fastkit.predict_pipelined reads it from these two process variables
os.environ["CHROMBPNET_FASTKIT_TAIL"] = _FD["tail"]
os.environ["CHROMBPNET_FASTKIT_TAIL_SOURCE"] = _FD["tail_reason"]
_JIT_DEFAULT = os.path.join(_BASE_KIT, "cache", "nv_compute_cache_%s.tar" % _FD["gpu_class"])
# the TF route's batch ceiling by class (fastdefault.NATIVE_DILATION_BATCH_PIN: A100 <= 256 with native dilation) is applied below — clamped, said, never silent
_PRE = None   # the stage-2 import chain on a daemon thread under stage 1 (chrombpnet_k1/_preimport.py loaded by path; every route); K1_PREIMPORT=0 = off (a debug arm)
if _FD.get("forward") in ("k1", "tf_function", "keras_predict_fileorder") and os.environ.get("K1_PREIMPORT", "1") != "0":   # the TF routes too (the chain is torch-free; otherwise its imports land in the finish stage)
    try:
        import importlib.util as _ilu0
        _pre_py = os.path.join(_BASE_KIT, "torch", "chrombpnet_k1", "_preimport.py")
        _spec0 = _ilu0.spec_from_file_location("_kit_preimport", _pre_py); _PRE = _ilu0.module_from_spec(_spec0); _spec0.loader.exec_module(_PRE); _PRE.start()
    except Exception as _e:
        if is_oom(_e): raise                                                  # an out-of-memory error propagates; every other failure refuses the mode by name
        _refuse_mode("the stage-2 preimport thread could not start (%s)" % repr(_e)[:160])
if _FD.get("batch_pin") and args.batch_size > _FD["batch_pin"]:
    print(f"[pred_bw_fast] batch clamped {args.batch_size} -> {_FD['batch_pin']} on {_FD['gpu_class']} (the TF route with native dilation: throughput falls past B={_FD['batch_pin']} on this class; under the deterministic recipe the outputs do not depend on the batch)", flush=True); args.batch_size = _FD["batch_pin"]
if not _JIT_TAR and os.path.exists(_JIT_DEFAULT): _JIT_TAR = _JIT_DEFAULT      # a shipped per-class driver cache is applied at load when present
_CACHES = {}
def _cache_entries(d):
    return sum(len(fs) for _, _, fs in os.walk(d)) if d and os.path.isdir(d) else 0
if _FORWARD == "k1" and not _EXTRA and _JIT_TAR and os.path.exists(_JIT_TAR):
    # on the K1 route no driver PTX JIT happens (shipped cubins) — the ComputeCache install would be dead work there; it is installed ONLY on the
    # TF route (after the route decision, before TensorFlow's first CUDA context)
    _CACHES["cuda_jit"] = "not installed on the K1 route (shipped cubins, no driver PTX JIT: the triton census reports 0 compile-class calls; the tarball stays in the kit for the TF route)"
    _FD["fixed_cost_s"]["cache_install_skipped_k1_route"] = 0.0
elif _JIT_TAR and os.path.exists(_JIT_TAR):
    _s = kit.jit_cache_boot(_JIT_TAR); _js = kit.JIT_CACHE_STAMP
    _FD["fixed_cost_s"]["cache_install" + ("_marker_hit" if (_js.get("install_marker") or {}).get("installed_utc") and not (_js.get("install_marker") or {}).get("written") else "")] = round(float(_js.get("untar_s") or 0.0), 3)   # the cache stage on the fixed-cost line (one-time install vs the O(1) marker hit)
    _CACHES["cuda_jit"] = ("NOTE %s | " % _js["identity_note"] if _js.get("identity_note") else "") + "shipped %d members applied in %.1f s (identity %s; loader: staged, no overwrite, before any CUDA context; proof = the CONTENT witness after the first forward, printed at the end, + the first-forward seconds)" % (len(_js.get("members") or {}), _s, "matched" if _js.get("identity") else "unstamped")
else: _CACHES["cuda_jit"] = "none (no shipped cache for this class; MISS = driver JIT at first use)"
_t_in = _time_c.time()
if not os.path.exists(args.genome + ".fai"):
    import pyfaidx; pyfaidx.Faidx(args.genome)
_FD["process_terms_s"] = {"kit_import": round(_T_KIT_IMPORT, 3), "import_tf": (round(_T_IMPORT_TF, 3) if _T_IMPORT_TF is not None else "skipped (the K1 route needs no TensorFlow: the metrics chain is TF-free)")}   # inputs_open is added in _run_model; the model load is on the stamp (timings_s.load_model)
_TF_ND = bool(_FD["native_dilation"])   # the TensorFlow route's native-dilation state on this class and mode (off under the deterministic recipe by construction)
if _FORWARD == "k1":
    # ONE-PROCESS K1 ROUTE (torch/triton beside TensorFlow in this process): the kit's one-hot featurisation (the stock's bytes), the K1 Triton forward,
    # then the same writer / stats / preds.bed / metrics path as the TF route. apply() = pins + patch, no forward; the first forward is the warm-up below.
    import sys, time
    # HERE-FIRST: the kit's OWN torch/ (opt/kit/torch) is FIRST on sys.path unconditionally; CHROMBPNET_K1_DIR is a fallback only when the kit has no
    # torch/ (an environment variable must never shadow the kit's own package). The imported package's dir is stamped and asserted below.
    _own = os.path.join(_BASE_KIT, "torch"); _envdir = os.environ.get("CHROMBPNET_K1_DIR", "")
    _cands = [c for c in (_own, _envdir) if c and os.path.isdir(os.path.join(c, "chrombpnet_k1"))]
    for cand in reversed(_cands): sys.path.insert(0, cand)
    try:
        import torch, triton, bpnetlite   # the stack REALLY imported here too; a failure -> the mode refuses by name (K1 is in this class's lever set)
        from chrombpnet_k1 import apply as _probe_apply
    except BaseException as e:
        if is_oom(e) or isinstance(e, (SystemExit, KeyboardInterrupt)): raise    # an out-of-memory error, a refusal by name and an interrupt propagate
        _refuse_mode(f"the K1 stack failed to import in the job process ({type(e).__name__}: {str(e)[:160]})")
if _FORWARD == "k1":
    from chrombpnet_k1 import apply as k1_apply, predict as k1_predict, cache_witness as k1_cache_witness
    import chrombpnet_k1 as _k1mod
    _k1_imported_dir = os.path.dirname(os.path.dirname(os.path.abspath(_k1mod.__file__)))
    if os.path.isdir(os.path.join(_own, "chrombpnet_k1")) and os.path.realpath(_k1_imported_dir) != os.path.realpath(_own):
        raise SystemExit(f"[pred_bw_fast] HERE-FIRST violated: the kit has its own torch package at {os.path.realpath(_own)} but chrombpnet_k1 was imported from {_k1_imported_dir} (PYTHONPATH/CHROMBPNET_K1_DIR shadowing) — refused")
    _FD["k1_package_imported_from"] = _k1_imported_dir; _FD["k1_package_here_first"] = os.path.realpath(_k1_imported_dir) == os.path.realpath(_own)
    _pins_k1 = _FD.get("k1_cache_pins") or {}
    _CACHES["k1"] = "from %s (here_first=%s; shipped Triton cache pins: %s)" % (_FD.get("k1_package_imported_from"), _FD.get("k1_package_here_first"), ("MATCH — the package's loader applies %s" % _pins_k1.get("cache_dir")) if _pins_k1.get("ok") else ("MISMATCH/ABSENT — NOT applied, a cold JIT this process (never a HIT): %s" % str(_pins_k1.get("note"))[:160]))   # the shipped Triton cache pins, said on the status line
    try:   # the K1 kernels START here (apply: pins + shipped cache + patch; the warm-up: every kernel compiled or loaded and launched at the job's shape) —
           # a card or stack they cannot start on (a Triton compile / launch failure) refuses the mode BY NAME (exit 3): never a run under `fast` with K1 dropped
        _FD["k1_first_process_terms_s"] = _fd.k1_first_process_terms(_K1_BIAS, _K1_NOBIAS)   # the pre-warms timed BEFORE apply (cost moved into named terms, not removed; outputs unchanged)
        t0 = time.time(); k1, apply_stamp = k1_apply(_K1_BIAS, _K1_NOBIAS, precision=_FD["precision"], tc_tile=_FD.get("tc_tile")); apply_s = time.time() - t0   # the conv arithmetic follows the mode (fastdefault.resolve: tf32 tensor cores at shipped numerics, fp32 in stock's order under the recipe)
        _FD["k1_first_process_terms_s"]["apply_after_prewarm"] = round(apply_s, 3)
        import pandas as pd
        kb = args.batch_size; lp_stamp = None
        if kb > 1024:   # the K1 kernels' indexing limit (int32 indexing: an illegal memory access past B=1024) — clamped, said, never silent
            print(f"[pred_bw_fast] K1 batch clamped {kb} -> 1024 (the K1 kernels index in int32: at most 1024 units per forward call; a larger -bs is chunked; under the deterministic recipe the outputs do not depend on the batch)", flush=True); kb = 1024
        # the K1 forward runs INSIDE the kit's pipelined path (sorted chunks, featurise prefetch thread, the pre-spawned bigWig writer + -os stats fed per
        # chunk while the GPU works). The K1 kernels' per-unit outputs do not depend on order or batch membership, so sorted-order prediction gives the
        # same bits as file order. The first real batch is timed inside the callable (k1_forward's first_batch_s); the warm-up here is one zero-input forward.
        _wu = {}
        if _FD["warmup"]:
            _t = time.time(); k1_predict(k1, np.zeros((kb, kit.INPUTLEN if hasattr(kit, "INPUTLEN") else 2114, 4), dtype=np.int8), batch_size=kb); _wu = {"shapes": [kb], "inputlen": 2114, "s": time.time() - _t}
    except BaseException as _e:
        if is_oom(_e) or isinstance(_e, (SystemExit, KeyboardInterrupt)): raise   # an out-of-memory error, a refusal by name and an interrupt propagate
        _refuse_mode(f"the K1 kernels did not start on this card ({_FD['gpu_class']}, cc {_FD.get('compute_cap')}; {type(_e).__name__}: {str(_e)[:200]})")
if _FORWARD == "k1":
    _k1_dir = apply_stamp.get("cache_dir"); _k1_ref = {}
    if _k1_dir and os.path.isdir(_k1_dir):
        for _r, _, _fs in os.walk(_k1_dir):
            for _f in _fs: _pp = os.path.join(_r, _f); _k1_ref[os.path.relpath(_pp, _k1_dir)] = __import__("hashlib").sha256(open(_pp, "rb").read()).hexdigest()
    _CACHES["triton"] = "by-count %s (compiles not excluded); CONTENT witness after the job — printed at the end" % str(k1_cache_witness(apply_stamp.get("cache_dir"), apply_stamp.get("cache_entries_at_apply")))[:40]
    print(_fd.r8_line(_FD, _CACHES, _wu if _FD["warmup"] else None), flush=True)
def _run_model(_model, _sfx, _route, _nd, _primary):
    """ONE model of one item — the per-model section (the overlap, the forward closure, predict_pipelined, the preds BED, the metrics stage, the model's
    run-record entry, the output-parity check) for `_model` written at args.output_prefix + `_sfx` on `_route` ('k1' = the composed -cm model on the Triton
    forward; else stock's own graph on the TensorFlow route); returns (the expected outputs, the record path). `_primary`: the model whose entry the item's
    prefix keys in the run record (the -cm model, or the first model given)."""
    global _PRE
    _ov_fits = None
    prefix = args.output_prefix + _sfx
    _OV = None
    if args.bigwig:                                                                       # the metrics stage runs DURING the forward (chrombpnet_fastkit.metrics_overlap)
        from chrombpnet_fastkit import metrics_overlap as _mo
        _ov_df = pd.read_csv(args.regions, sep="\t", names=kit.NARROWPEAK_SCHEMA); _T_ov = {}
        # the stock's own drop rule (bigwig_helper.get_seq: the inputlen window must lie inside the chromosome) applied UP FRONT from the chrom sizes,
        # so the metrics frame = exactly the regions the forward predicts; a region dropped by name never reaches the observed-bigWig reader
        _sizes = {l.split()[0]: int(l.split()[1]) for l in open(args.chrom_sizes) if l.strip()}
        _FD["process_terms_s"]["inputs_open"] = round(_time_c.time() - _t_in, 3)   # the .fai check + the regions/chrom-sizes reads
        _c = (_ov_df["start"].astype(int) + _ov_df["summit"].astype(int)); _half = (kit.INPUTLEN if hasattr(kit, "INPUTLEN") else 2114) // 2
        _ov_fits = np.asarray((_c - _half >= 0) & (_c + _half <= _ov_df["chr"].map(_sizes).fillna(-1).astype(int)), dtype=bool)
        _ov_dropped = int((~_ov_fits).sum())
        if _ov_dropped: print(f"[pred_bw_fast] metrics frame: {_ov_dropped} region(s) dropped by name (the inputlen window falls outside the chromosome — the stock's own rule; the stock CLI fails on this input with -bw): metrics on the {int(_ov_fits.sum())} predicted regions", flush=True)
        _ov_df = _ov_df[_ov_fits].reset_index(drop=True)
        _OV = _mo.MetricsOverlap(args.bigwig, _ov_df, 1000, prefix, h5_workers=8, timings=_T_ov)      # outputlen 1000 as the prefetch worker; compared with the heads' outputlen at finish (refused on a mismatch)
    if _route == "k1":
        _st = {"first_batch_s": None, "fstamp": None, "t_fwd": 0.0}
        def k1_forward(Xc):
            t0 = time.time()
            if _st["first_batch_s"] is None:
                lg0, lc0, f0 = k1_predict(k1, Xc[:kb], batch_size=kb); _st["first_batch_s"] = time.time() - t0
                if len(Xc) > kb:
                    lg1, lc1, f1 = k1_predict(k1, Xc[kb:], batch_size=kb); lg, lc, fst = np.concatenate([lg0, lg1]), np.concatenate([lc0, lc1]), f1
                else: lg, lc, fst = lg0, lc0, f0
            else:
                lg, lc, fst = k1_predict(k1, Xc, batch_size=kb)
            _st["fstamp"] = fst; _st["t_fwd"] += time.time() - t0; return lg, lc
        chunk = 4096 if 4096 % kb == 0 else kb * max(1, 4096 // kb)
        stamp, logits, logcts, used = kit.predict_pipelined(_model, args.regions, args.genome, args.chrom_sizes, prefix + ".bw", batch=kb, chunk=chunk, max_zooms=10,
                                                             writer="process", save_heads=None, writer_backend="numpy",
                                                             bigwig_layout="stock", stats_file=args.output_prefix_stats, forward_fn=k1_forward, prefetch=True, overlap=_OV)
        fstamp = _st["fstamp"] or {}; forward_s = _st["t_fwd"]; first_batch_s = _st["first_batch_s"] or 0.0
        import hashlib as _hl
        stamp.update({"forward": "k1 (Triton exact forward, torch " + str(fstamp.get("torch")) + ", triton " + str(fstamp.get("triton")) + ") inside predict_pipelined (sorted chunks, overlapped writer)",
                      "apply_s": round(apply_s, 3), "first_batch_s": round(first_batch_s, 3), "forward_s": round(forward_s, 3), "precision": _FD.get("precision"), "k1_apply": apply_stamp,
                      "k1_forward_stamp": {k: fstamp.get(k) for k in ("ms_per_unit", "batch_size", "padded_rows", "device", "cudnn")},
                      "md5_logits": _hl.md5(np.ascontiguousarray(np.asarray(logits, dtype=np.float32)).tobytes()).hexdigest(), "md5_logcts": _hl.md5(np.ascontiguousarray(np.asarray(logcts, dtype=np.float32)).tobytes()).hexdigest(),
                      "k1_cache_witness_by_count": "dropped in v0.12.47 (it was composed from the pre-install snapshot and misread a HIT as MISS/PARTIAL; the content witness above replaces it)", "k1_cache_witness_content": kit.cache_content_witness(_k1_dir, _k1_ref), "k1_triton_cache_form": apply_stamp.get("install_form") or "k1r4: form (2) effective-dir add-only (the loader rule)", "k1_after_job_witness": (k1_cache_witness(apply_stamp.get("cache_dir"), apply_stamp.get("cache_snapshot_at_apply"), apply_stamp.get("compiles_at_apply"), apply_stamp.get("jit_meter_n0", 0)) if callable(k1_cache_witness) else None), "k1_cache_relocation": "k1r4: the installed entries __grp__ child paths rewritten to the effective dir at apply (form (2)); T2C v0.4 form", "k1_batch": kb, "k1_pinned": False, "lp_model": lp_stamp, "fastdefault": _FD, "warmup": _wu, "caches": _CACHES})
    else:
        stamp, logits, logcts, used = kit.predict_pipelined(_model, args.regions, args.genome, args.chrom_sizes, prefix + ".bw", batch=args.batch_size, chunk=4096,
                                                        max_zooms=10, writer="process", save_heads=None,
                                                        pad_to_batch=True, native_dilation=_nd, prefetch=True, writer_backend="numpy", forward=_route,
                                                        bigwig_layout="stock", stats_file=args.output_prefix_stats, overlap=_OV, warmup=_FD["warmup"], fastdefault=_FD, caches=_CACHES)
    # the stock's regions file of the predicted regions (predict_to_bigwig.main: regions_df[regions_used].to_csv(..., sep="\t", header=False, index=False))
    regions_df = pd.read_csv(args.regions, sep="\t", names=kit.NARROWPEAK_SCHEMA)
    regions_df[np.asarray(used, dtype=bool)].to_csv(prefix + "_preds.bed", sep="\t", header=False, index=False)
    if _PRE is not None:
        try:
            _PRE.join(timeout=60); stamp["preimport"] = {k: v for k, v in _PRE.STATE.items() if k != "thread"}; print("[pred_bw_fast] " + str(_PRE.STATE.get("line")), flush=True)
        except Exception as _e: _refuse_mode("the stage-2 preimport thread failed (%s)" % repr(_e)[:160])
    if args.bigwig:                                                                       # the stock's metrics stage, its own code on the kit's heads
        import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper          # bigwig_helper only (predict_to_bigwig would import TensorFlow; the K1 route stays TF-free)
        outputlen = int(np.asarray(logits).shape[1]); regions = bigwig_helper.get_regions(args.regions, outputlen, np.asarray(used, dtype=bool))
        if int(outputlen) != int(_OV.L): raise SystemExit(f"[pred_bw_fast] the metrics stage was built for outputlen {_OV.L} but the heads have {outputlen} — refused")
        _um = np.asarray(used, dtype=bool)
        if _ov_fits is not None and len(_um) == len(_ov_fits): _um = _um[_ov_fits]   # the overlap frame is the window-fits subset; the forward's used-mask on it must be all-true (else refused)
        T_m = _OV.finish(np.asarray(logcts, dtype=np.float32), [[r[0], r[-1]] for r in regions], used_mask=_um)
        stamp["metrics_stage"] = "kit metrics_overlap (the fast metrics stage during the forward: obs/RNG ahead in file order; softmax/jsd/h5 deflate per arriving chunk; assembly after)"; stamp["metrics_stage_timings_s"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in T_m.items()}
    stamp["outputs"] = sorted(os.path.basename(f) for f in ([prefix + ".bw", prefix + "_preds.bed"] + ([args.output_prefix_stats] if args.output_prefix_stats else []) +
                              ([prefix + "_predictions.h5", prefix + "_metrics.json", prefix + ".counts_pearsonr.png", prefix + ".profile_jsd.png"] if args.bigwig else [])) if f and os.path.exists(f))
    stamp["cuda_cache_witness"] = {"verdict": "n/a (the K1 route: the driver cache is not installed; shipped cubins, no driver PTX JIT)"} if _route == "k1" else kit.cache_content_witness(os.environ.get("CUDA_CACHE_PATH"), (kit.JIT_CACHE_STAMP.get("members") if _JIT_TAR else None)) if (_JIT_TAR and os.path.exists(_JIT_TAR) and not kit.JIT_CACHE_STAMP.get("fallback")) else {"witness": "content", "verdict": "n/a (no shipped cache applied)" + (" — " + kit.JIT_CACHE_STAMP["fallback"] if kit.JIT_CACHE_STAMP.get("fallback") else "")}
    print("[pred_bw_fast] cache witness (content) cuda_jit:", stamp["cuda_cache_witness"]["verdict"], flush=True)
    try:
        _w = stamp.get("k1_after_job_witness")
        if isinstance(_w, dict):
            _sc = _w.get("stage_counter") or {}; _jm = _w.get("jit_meter") or {}
            print("[chrombpnet_fastkit v%s] triton census (this process): stage counters make_ttir=%s make_cubin=%s (bound_on=%s; admissible=%s) | jit meter: compile-class calls (>= %s s) %s, loads 5-150 ms %s | content witness: %s new, %s changed of %s files at apply (put = %s; new hash dirs = %s) | install form: %s | verdict: %s" % (
                stamp.get("version", "?"), _sc.get("make_ttir"), _sc.get("make_cubin"), str(_sc.get("bound_on"))[:60], _sc.get("admissible"), _jm.get("threshold_s"), _jm.get("compile_class_calls_over_150ms"), [(n, s) for n, s in (_jm.get("loads_5_150ms") or [])][:6],
                _w.get("n_added"), _w.get("n_changed"), _w.get("files_at_apply"), _w.get("n_added"), sum(1 for p in (_w.get("added") or []) if p.count("/") == 1 and p.endswith(".json") and "__grp__" not in p), str(stamp.get("k1_triton_cache_form"))[:80], str(_w.get("verdict"))[:120]), flush=True)
    except Exception as _e:
        print("[chrombpnet_fastkit] triton census: unavailable (%s)" % type(_e).__name__, flush=True)

    # the REGIME line — what the kit chose from the inputs, why, and the counters per call (one line; zero knobs)
    try:
        _tail = stamp.get("tail") or {}; _M = stamp.get("metrics_stage_timings_s") or {}
        print("[pred_bw_fast] regime: N=%s units (%s dropped by name) -> %s forward call(s) at batch %s (padded rows %s; tail %s); route=%s; h5=%s; writer=%s; metrics=%s; outputs=%s — chosen from the inputs (job size, the model's shapes, the requested outputs); counters: forward_calls=%s units_forwarded=%s writer_regions=%s" % (
            stamp.get("units_in"), stamp.get("units_dropped_named"), stamp.get("forward_calls"), stamp.get("batch"), stamp.get("padded_rows"), _tail.get("tail_form"), stamp.get("forward"),
            (str(_M.get("h5_form")) if isinstance(_M, dict) else "n/a")[:60], stamp.get("writer_start_method"), (str(_M.get("form")) if isinstance(_M, dict) else "n/a")[:40],
            "bw" + ("+stats" if getattr(args, "output_stats", None) or getattr(args, "os", None) else "") + ("+metrics" if getattr(args, "bigwig", None) or getattr(args, "bw", None) else ""),
            stamp.get("forward_calls"), stamp.get("units_forwarded"), stamp.get("writer_regions")) + (" | stock path unavailable at N=1 (stock defect: AxisError in the stock's own post-processing - the batch dim squeezed); kit route" if int(stamp.get("units_in") or 0) == 1 else ""), flush=True)
    except Exception as _e:
        print("[pred_bw_fast] regime: (line unavailable: %s)" % type(_e).__name__, flush=True)
    stamp["host_side"] = {"kit_dir": os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "base_kit": _BASE_KIT, "png_workers": _png_workers.stamp(), "writer": "prespawned (one writer interpreter per process, started before the runtimes; items bound in sequence)", "finish_legs": "counts PNG (spawned helper) | jsd PNG (spawned helper) | h5 (the kit's fast writer in its checked regime N >= %d; the stock's own writer below)" % __import__("chrombpnet_fastkit.h5_fast", fromlist=["H5_FAST_MIN_N"]).H5_FAST_MIN_N}
    stamp["import_witness"] = {"tensorflow_loaded": "tensorflow" in sys.modules, "torch_loaded": "torch" in sys.modules, "modules_with_tensorflow_prefix": sum(1 for m in sys.modules if m.startswith("tensorflow")), "forward": _route}   # which runtimes this process loaded (the K1 route imports no TensorFlow)
    print("[pred_bw_fast] import witness", json.dumps(stamp["import_witness"]))
    stamp["jit_cache"] = kit.JIT_CACHE_STAMP   # jit_cache_boot()'s install stamp folded into the run record
    stamp["featurise_prefetch"] = stamp.get("prefetch")   # the top-level 'prefetch' IS the featurise prefetch thread — the same value under a clearer key
    stamp["malloc_env"] = _MALLOC_STAMP                                              # the three malloc tunables + their source, in every record entry
    if _MULTI_CTX is not None: stamp["multi"] = dict(_MULTI_CTX, resident="the K1 model (weights), the Triton kernels, the genome map; every other kit object per item")
    _RECORD["items"].append(dict(stamp, item=len(_RECORD["items"]) + 1, prefix=(args.output_prefix if _primary else prefix), model_suffix=_sfx, regions=args.regions)); _record_path = _write_record()   # this model's entry of the run record (the item's prefix keys its primary model)
    # OUTPUT PARITY first — every expected output on disk (the PNGs where the metrics stage drew them); a missing/empty output RAISES here (rc != 0,
    # never through _exit). The caller's LAST statement is exit_fast: it re-opens each output + the run record (non-empty, fsync'd) and os._exit(0)s past
    # the interpreter's teardown (object GC, the CUDA context destruction, shared-lib unload). Every failure path leaves with rc != 0 through the
    # excepthook / the writer sentinel. K1_EXIT_TEARDOWN=1 keeps the teardown (a debug arm, not a knob).
    _EXPECTED = ([prefix + ".bw", prefix + "_preds.bed"] + ([args.output_prefix_stats] if args.output_prefix_stats else []) + ([prefix + "_predictions.h5", prefix + "_metrics.json"] if args.bigwig else [])
                 + [f for f in ([prefix + ".counts_pearsonr.png", prefix + ".profile_jsd.png"] if args.bigwig else []) if os.path.exists(f)])
    _missing = [f for f in _EXPECTED if not (os.path.exists(f) and os.path.getsize(f) > 0)]
    if _missing: raise SystemExit("[pred_bw_fast] OUTPUT PARITY FAILED: expected output(s) missing or empty: %s" % _missing)
    return _EXPECTED, _record_path
def _run_item(_regions, _output_prefix, _stats_file):
    """ONE item: every model given, in stock's order (-cmb, -cm, -bm) on args.{regions, output_prefix, output_prefix_stats} set to the item's — the -cm model on
    the process's route, -cmb / -bm on the TensorFlow route; returns (the expected outputs of all its models, the record path)."""
    args.regions, args.output_prefix, args.output_prefix_stats = _regions, _output_prefix, _stats_file
    _primary_sfx = "_chrombpnet" if args.chrombpnet_model else _MODELS[0][1]
    _exp_all = []; _rp = None
    for _model, _sfx in _MODELS:
        _is_cm = (_sfx == "_chrombpnet")
        _exp, _rp = _run_model(_model, _sfx, (_FORWARD if _is_cm else _TF_FORWARD), (_ND if _is_cm else _TF_ND), _sfx == _primary_sfx)
        _exp_all += _exp
    return _exp_all, _rp
import importlib.util as _ilu
_exit_py = os.path.join(_BASE_KIT, "torch", "chrombpnet_k1", "_exit.py")
_spec = _ilu.spec_from_file_location("_kit_exit", _exit_py); _kx = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_kx)
if _MULTI_CTX is None:
    _EXPECTED, _STAMP = _run_item(*_ITEMS[0])
    _t_stop = _time_p.time(); _png_workers.stop(); _kit_ps.stop_prespawned_writer()       # the helpers and the writer leave before the main process (a None job, waited)
    print("[pred_bw_fast] item done at +%.3fs; helpers + writer stopped in %.3fs" % (_t_stop - _T_PROC0, _time_p.time() - _t_stop), flush=True)
    if _RECORD.get("bare_run"): print("[pred_bw_fast] run record: %s (a bare run: the package's CHROMBPNET_FASTKIT_RECORD was not set)" % _STAMP, flush=True)
    _kx.exit_fast(_STAMP, outputs=_EXPECTED)
# --- the multi-item loop: the resident objects are the ones the setup above built (the K1 model, the kernels, the genome map); each item's objects are built and dropped in _run_item
import time as _time_m, hashlib as _hl_m
_T_LOAD = _time_m.time() - _T_PROC0; _ALL = []; _STAMP = None
print("[pred_bw_fast] multi: ready load=%.3fs items=%d route=%s" % (_T_LOAD, len(_ITEMS), _FORWARD), flush=True)
for _i, (_rg, _op, _os_) in enumerate(_ITEMS, 1):
    _t_item = _time_m.time(); _MULTI_CTX["item_index"] = _i; _MULTI_CTX["load_s"] = round(_T_LOAD, 3)
    _exp, _STAMP = _run_item(_rg, _op, _os_)
    for _p in _exp + [_STAMP]:                                                   # the item's outputs made durable before its wall is clocked (what exit_fast does once at the end, here per item)
        _fd_ = os.open(_p, os.O_RDONLY)
        try: os.fsync(_fd_)
        finally: os.close(_fd_)
    # a finished item's outputs are final: taken off the kit's fail-fast release list (the kit registers a NEW final bigWig name when the writer is bound and
    # unlinks it on an uncaught exception — in one process serving several items that would delete an EARLIER item's finished bigWig)
    kit._FINAL_OUTPUTS_NEW[:] = [_p for _p in kit._FINAL_OUTPUTS_NEW if _p not in set(_exp)]
    _ALL += _exp; _wall = _time_m.time() - _t_item
    _sha = {os.path.basename(_p): _hl_m.sha256(open(_p, "rb").read()).hexdigest() for _p in _exp}
    _RECORD["items"][-1].update(wall_s=round(_wall, 3), cold=_i == 1, sha256=_sha); _STAMP = _write_record()   # the item's wall and digests join its record entry
    print("[pred_bw_fast] multi: item %d/%d %s wall=%.3fs rc=0" % (_i, len(_ITEMS), _op, _wall), flush=True)     # the per-item wall line of the resident process (a caller may grep it)
print("[pred_bw_fast] multi: DONE items=%d load=%.3fs wall=%.3fs" % (len(_ITEMS), _T_LOAD, _time_m.time() - _T_PROC0), flush=True)
_t_stop = _time_p.time(); _png_workers.stop(); _kit_ps.stop_prespawned_writer()
print("[pred_bw_fast] items done at +%.3fs; helpers + writer stopped in %.3fs" % (_t_stop - _T_PROC0, _time_p.time() - _t_stop), flush=True)
if _RECORD.get("bare_run"): print("[pred_bw_fast] run record: %s (a bare run: the package's CHROMBPNET_FASTKIT_RECORD was not set)" % _STAMP, flush=True)
_kx.exit_fast(_STAMP, outputs=_ALL)
