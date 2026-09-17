"""Activation: the gates, the kit on the path, the levers applied to the runner the stock CLI builds, the report.

`activate(mode)` (= `protenix_v1_opt.enable`) runs once per process:
  1. gates (opt_core.gates records, every one run, every refusal named) — the core is the pinned one (opt/pyproject.toml
     [tool.opt_core] vs the importable opt_core: _core_gate.gate at every entry, before this module imports), the pinned `protenix` is installed at the stock version
     (`stock/PINS.json`), the weights root carries the pinned weights (kit.frozen_weights_check: the upstream's boot-time download
     fallback is never reached), every required kit file is present (kit.check_files()), torch sees a CUDA GPU (a class outside
     PINS.json "supported_gpus" is admitted untested with a NOTE), no Protenix model instance exists yet, and (with det) the recipe can still precede the
     model package;
  2. the deterministic recipe when asked (det.install); the flash triangle-attention floor of the mode exported as the kit's gate
     variable (export_triattn_floor: modes.TRIATTN_FLOOR -> PTX_TRIATTN_MIN_TOKENS unless the caller set it);
  3. the kernels the levers reach (opt_core.kernels: `flash_triattn`, `fpf_trimul` imported by name;
     `fpf_triatt_pro`, `fpf_triatt_epi`, `fpf_triatt_k2b`, `fpf_transition`, `lnl_fused` served through opt_core.attn.pair_fused — carried once, in the
     core; the kit directory holds no copy) are routed BY NAME to the core copies (kernels.route, one meta-path finder serving exactly those names), the kit's
     the core's own cell tables and data files for the
     others (pair_fused.carried_exports), and every route
     checked before the first import (kernels.route_check: the resolved bytes are the sums file's, the export present) —
     `[protenix-v1-opt] KERNELS ...` names the resolved paths; then the
     kit's three directories go to the front of sys.path (kit.KIT_SYS_PATHS, the kit's own PYTHONPATH order) — the rest of the kit's modules
     (the graph package, the hoist, the fast-LN build) keeps importing from there — and `levers_ptx1` is imported
     from there (its file is asserted to be the carried one);
  4. the template guard is installed (templates.install: the stock template search's dropped hits become named TEMPLATE lines) and
     `runner.inference.InferenceRunner.__init__` is wrapped: after the stock constructor built and loaded the model, the kit's own
     sequence runs on it — `levers_ptx1.bind_model(runner.model)`, `levers_ptx1.apply(<arm of the mode>)`; the returned CFG is
     checked against the mode's arm and the ACTIVE line is printed. A second
     runner in the same process is refused (the kit's sampler graphs and hoist bind one model per process). `InferenceRunner.predict` is
     wrapped too: each item's sizes are recorded and its template slots censused (templates.check_item).
The report (`status()`) is "armed" after step 4 and "active" once the runner has been built and the levers applied; `pred` prints it
(the ACTIVE / LEVER / EXIT lines). Every refusal prints `[protenix-v1-opt] NOT ACTIVE: <reason>` (report.not_active_line) and, under strict=True,
raises ActivationError.

PARTIAL activation (the exit rule, report.py): a lever of the mode the kit's CFG does not carry after `apply(arm)` is a partial
activation at step 4 — refused there (the family line, exit 3 on the environment route) unless the allowance is on (`--allow-partial`
on every verb; `PROTENIX_V1_OPT_ALLOW_PARTIAL=1` on the environment route, its only surface for it), in which case the runner proceeds with `partial` /
`partial_reason` / `allow_partial` in the report and on the ACTIVE line. A lever that fell back at run time (the kit's own counters,
report.kit_evidence) is judged by the verb at exit from `refresh_levers()`; on the environment route the package owns no exit — the
hook's lines at process exit (`_hook_exit_record`: the exit rule's lines) are the evidence and `pred` is the gated form.
"""
from __future__ import annotations

import atexit
import gc
import importlib
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

from opt_core import gates as G
from opt_core import stock_proof as SP

from . import ActivationError, __version__
from . import kit as K
from . import big as BIG
from . import modes as M
from . import ngpu as NGPU
from . import phase as PHASE
from . import report as R
from . import rowpair as ROWPAIR
from . import templates as TEMPLATES

PREFIX = R.PREFIX
ENV_MODE = "PROTENIX_V1_OPT"
ENV_DET = "PROTENIX_V1_OPT_DET"
ENV_ALLOW_PARTIAL = "PROTENIX_V1_OPT_ALLOW_PARTIAL"                # =1: the environment route's `--allow-partial` (read by the .pth hook at interpreter start, _autoload; `pred` reads its flag only)
PACKAGE_ENV = (ENV_MODE, ENV_DET, ENV_ALLOW_PARTIAL, K.ENV_KIT, NGPU.ENV_N_GPU, K.WEIGHTS_MEMO_ENV)     # the package's own switches (stripped from the stock arm, never exported): every name read under the PROTENIX_V1_OPT prefix (== _autoload.DECLARED)
PINS_RELPATH = os.path.join("stock", "PINS.json")
RUNNER_MODULE = "runner.inference"
RUNNER_CLASS = "InferenceRunner"
MODEL_MODULE = "protenix.model.protenix"
MODEL_CLASS = "Protenix"

_REPORT: Optional[dict] = None
_LEVERS = None
_RUNNERS = {"built": 0, "wrapped": False, "hook_registered": False}
_ITEMS: List[dict] = []                                             # one record per `InferenceRunner.predict` call: the item's sizes (the budget rule reads N_token)


def pins(reload: bool = False) -> dict:
    global _PINS
    if reload or "_PINS" not in globals():
        with open(os.path.join(K.tree_home(), PINS_RELPATH), "r", encoding="utf-8") as fh:
            _PINS = json.load(fh)
    return _PINS


dist_version = G.dist_version                                     # the installed distribution's version from its metadata, None when absent


def upstream_versions() -> Dict[str, Optional[str]]:
    return {k: dist_version(k) for k in ("protenix", "torch", "triton", "cuequivariance-torch", "cuequivariance-ops-torch-cu12", "deepspeed")}


def torch_gpu_info() -> Optional[dict]:
    try:
        import torch
    except Exception as e:
        return {"error": f"torch: {e!r}"}
    if not torch.cuda.is_available():
        return None
    cc = torch.cuda.get_device_capability(0)
    return {"name": torch.cuda.get_device_name(0), "sm": f"{cc[0]}.{cc[1]}", "count": torch.cuda.device_count(), "torch": torch.__version__, "cuda": torch.version.cuda}


def supported_class(gpu: Optional[dict]) -> Optional[str]:
    """The PINS.json supported_gpus key whose compute capability matches the running GPU, else None."""
    if not gpu or "sm" not in gpu:
        return None
    table = pins()["supported_gpus"]
    for name, spec in table.items():                              # by name first (H100 and H200 share compute capability 9.0)
        if name in (gpu.get("name") or "") and spec["compute_capability"] == gpu["sm"]:
            return name
    for name, spec in table.items():
        if spec["compute_capability"] == gpu["sm"]:
            return name
    return None


def model_instances() -> int:
    mod = sys.modules.get(MODEL_MODULE)
    if mod is None:
        return 0
    cls = getattr(mod, MODEL_CLASS, None)
    if cls is None:
        return 0
    return sum(1 for o in gc.get_objects() if isinstance(o, cls))


def stock_env_violations(environ=None) -> List[str]:
    """Names in `environ` the stock arm must not carry (PINS.json stock_environment.must_be_absent_prefixes)."""
    environ = os.environ if environ is None else environ
    return SP.forbidden(environ, pins()["stock_environment"]["must_be_absent_prefixes"])


def status() -> dict:
    return dict(_REPORT) if _REPORT is not None else {"active": False, "armed": False, "reason": "enable() has not run"}


def _base(mode: str, trigger: Optional[str], det: bool, allow_partial: bool) -> dict:
    return {"active": False, "armed": False, "mode": mode, "package_version": __version__, "trigger": trigger, "det": bool(det),
            "allow_partial": bool(allow_partial), "partial": [], "partial_reason": None, "n_gpu": 1, "sharding": NGPU.NG.SHARDING_NONE,
            "protenix_version": dist_version("protenix"), "versions": upstream_versions(), "kit": K.kit_home()}


def _refuse(rep: dict, reason: str, strict: bool, **fields) -> dict:
    global _REPORT
    rep = dict(rep, active=False, armed=False, reason=reason, **fields)
    R.log(R.not_active_line(reason))
    _REPORT = rep
    if strict:
        raise ActivationError(reason)
    return rep


def gate_records(res: M.Resolution, det: bool, refresh_weights: bool = False) -> List[G.Gate]:
    """Every gate of an activation, run in order and recorded (opt_core.gates.run_gates): the kit's own checks with their own reasons (the
    core pin was gated at entry, before this module imported: _core_gate.gate). A gate that raises is a refused gate naming the exception."""
    def stock_version() -> G.Gate:
        want, pkg = pins()["stock"]["version"], pins()["stock"]["package"]
        have = dist_version(pkg)
        ok = have == want
        return G.Gate("stock_version", ok, None if ok else f"protenix {have!r} installed, the stock pin is {want} (stock/PINS.json)", {"package": pkg, "pinned": want, "found": have})

    def kit_files() -> G.Gate:
        try:
            chk = K.check_files()
        except Exception as e:
            return G.Gate("kit_files", False, f"kit files: {e!r}")
        return G.Gate("kit_files", chk["ok"], None if chk["ok"] else f"kit files: missing={chk['missing']} (required {chk['required']}, kit dir {K.kit_home()})",
                      {k: chk[k] for k in ("required", "missing", "files")})

    def gpu_class() -> G.Gate:
        gpu = torch_gpu_info()
        if gpu is None or "error" in gpu:
            return G.Gate("gpu", False, f"no CUDA GPU visible to torch ({gpu['error'] if gpu else 'torch.cuda.is_available() is False'})", {"gpu": gpu})
        cls = supported_class(gpu)
        if cls is None:                                             # a compute capability no class of stock/PINS.json supported_gpus has: admitted untested and NAMED — a lever whose kernel has no cell for this card answers by name (gated / fallback in its LEVER line), never silently
            ccs = sorted({spec["compute_capability"] for spec in pins()["supported_gpus"].values()})
            R.log(R.note_line(f"GPU {gpu['name']} (sm {gpu['sm']}) is not a tested class {sorted(pins()['supported_gpus'])} (compute capabilities {ccs}): admitted untested — a lever without a kernel cell for this card takes the stock statement by name"))
            return G.Gate("gpu", True, None, {"gpu": gpu, "class": None, "untested": True})
        if cls not in (gpu.get("name") or ""):                       # admitted by capability under another name (an A800 as A100, an H800 as H100): named, never refused
            R.log(R.note_line(f"GPU {gpu['name']} (sm {gpu['sm']}) admitted by compute capability as class {cls}"))
        return G.Gate("gpu", True, None, {"gpu": gpu, "class": cls})

    def no_model_yet() -> G.Gate:
        n = model_instances()
        return G.Gate("no_model_yet", not n, None if not n else f"{n} Protenix model instance(s) already exist: enable() must precede the model", {"instances": n})

    def no_runner_yet() -> G.Gate:
        return G.Gate("no_runner_yet", not _RUNNERS["built"], None if not _RUNNERS["built"] else "an InferenceRunner was already built in this process")

    def frozen_weights() -> G.Gate:
        try:
            w = K.frozen_weights_check(refresh=refresh_weights)       # the `check` verb hashes afresh (refresh_weights); an activation reads the digest memo
        except K.FrozenWeightsError as e:
            return G.Gate("frozen_weights", False, f"frozen weights: {e}")
        return G.Gate("frozen_weights", True, None, {"root": w["root"], "weights": w["weights"], "sha256": w["sha256"], "cached_utc": w["cached_utc"], "pinned": w["pinned"],
                                                     "checkpoint_bytes": w["checkpoint_bytes"], "caches": sorted(w["caches"])})

    def det_precedes_model() -> G.Gate:
        from . import det as D
        late = D.STOCK_MODULE in sys.modules and D.installed() is None
        return G.Gate("det", not late, None if not late else f"det: {D.STOCK_MODULE} is already imported; the recipe must precede the model package")

    return G.run_gates([stock_version, frozen_weights, kit_files, gpu_class, no_model_yet, no_runner_yet] + ([det_precedes_model] if det else []))


def gates(res: M.Resolution, det: bool, refresh_weights: bool = False) -> List[str]:
    """Every reason this process cannot run `res` now (empty = go): the refused gates' reasons, in gate order."""
    return [g.reason or f"{g.name} refused" for g in gate_records(res, det, refresh_weights) if not g.ok]


ROUTED_KERNELS = ("flash_triattn", "fpf_trimul",                                                           # imported by name (the flash cell; the exact TriMul kernels whose per-N buffers release_trimul_buffers drops)
                  "fpf_triatt_pro", "fpf_triatt_epi", "fpf_triatt_k2b", "fpf_transition", "lnl_fused")   # served through opt_core.attn.pair_fused by the gblock | gflash | xtr | ttr levers;
# all carried once, in opt_core/kernels (SUMS-held); the kit directory holds no copy of them


def route_kernels(kit: Optional[str] = None) -> dict:
    """Route the kernels to the core copies, export the cell tables they read at import, check every route before the first import.
    Returns {"routed": {name: {"resolved", "core_copy", "runtime_imports"}}, "exports": {var: value}}; a failed check raises
    ActivationError with the gate's reason (a neighbouring version or a missing export is never served silently)."""
    from opt_core import kernels as KR
    from opt_core.attn import pair_fused as PF
    kit = kit or K.kit_home()
    for name in ROUTED_KERNELS:
        KR.route(name)
    exp = dict(PF.carried_exports(list(ROUTED_KERNELS)))                                            # the core's own data files of the fused-surround cells (the lnl tile table); the TriMul provider's rows
                                                                                                    # read the core's own cell tables (no kit table: the levers bind it by tier word)
    os.environ.update(exp)
    routed = {}
    for name in ROUTED_KERNELS:
        g = KR.route_check(name)
        if not g.ok:
            raise ActivationError(g.reason)
        routed[name] = {"resolved": g.details["resolved"], "core_copy": g.details["core_copy"], "runtime_imports": g.details["runtime_imports"]}
    return {"routed": routed, "exports": exp}


def _put_kit_on_path() -> List[str]:
    added = []
    for p in reversed(K.kit_sys_paths()):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)
    return list(reversed(added))


def _import_levers():
    global _LEVERS
    mod = importlib.import_module(K.LEVERS_MODULE)
    want = os.path.realpath(K.levers_file())
    have = os.path.realpath(getattr(mod, "__file__", "") or "")
    if have != want:
        raise ActivationError(f"{K.LEVERS_MODULE} imported from {have}, not the carried {want}")
    _LEVERS = mod
    return mod


def _export_gate(value: Optional[int], env_name: str, what: str, environ=None) -> Optional[dict]:
    """One size gate of the mode exported as the kit's own variable before the kit applies the arm; a value already in the environment is
    the caller's and wins. Returns {mode: the mode's value, exported: the value the kit will read, source: "mode" | "environment"}, None
    where the arm does not carry the lever (nothing exported)."""
    environ = os.environ if environ is None else environ
    if value is None:
        return None
    given = environ.get(env_name)
    if given is None or not str(given).strip():
        environ[env_name] = str(int(value))
        return {"mode": int(value), "exported": int(value), "source": "mode"}
    try:
        return {"mode": int(value), "exported": int(str(given).strip()), "source": "environment"}
    except ValueError:
        raise ActivationError(f"{env_name}={given!r} is not an integer (the kit's {what}; the mode's value is {value})")


def export_triattn_floor(res: M.Resolution, environ=None) -> Optional[dict]:
    """The flash triangle-attention token floor of the mode (modes.TRIATTN_FLOOR) as the kit's gate variable (kit.TRIATTN_FLOOR_ENV); None
    for an arm without `gflash` or a mode without a floor (every shipped mode: modes.TRIATTN_FLOOR is None, nothing is exported); the exported
    value is the one the report's floor rule reads (report.floor_state)."""
    return _export_gate(res.triattn_floor, K.TRIATTN_FLOOR_ENV, "flash triangle-attention token floor", environ)


def item_record(data) -> dict:
    """The sizes of one `InferenceRunner.predict` item as the stock runner logs them (runner/inference.py:470-474: N_asym, N_token,
    N_atom, N_msa; 0-d tensors or ints in the feature dict); a size the dict lacks is None."""
    out = {"item": (data.get("sample_name") if isinstance(data, dict) else None)}
    for k in ("N_token", "N_atom", "N_msa", "N_asym"):
        v = data.get(k) if isinstance(data, dict) else None
        try:
            out[k] = int(v.item()) if hasattr(v, "item") else (int(v) if v is not None else None)
        except Exception:
            out[k] = None
    return out


def items() -> List[dict]:
    return list(_ITEMS)


def item_featurisation_failed(data, message: str) -> dict:
    """An item whose featurisation failed before `predict` (n_gpu>1: rowpair.item_of_rank0 reports it on every rank — rank 0 featurised, the
    stock loop skips the item on every rank in step, runner/inference.py:460-467): recorded failed by name (the ITEM line, the ITEMS census,
    `pred` exits 1) exactly as an item that raised inside `predict`. Returns the record."""
    rec = dict(item_record(data if isinstance(data, dict) else {}), failed=f"featurisation: {' '.join(str(message or '').split())[:200]}")
    _ITEMS.append(rec)
    R.log(R.item_failed_line(rec))
    return rec


def items_census() -> dict:
    """Every item in exactly one bucket: ok (returned a prediction), failed (raised inside `predict`: the stock's per-item handler logs the
    traceback to <out>/ERR/<item>.txt and continues with exit 0 — the kit's ITEM line names it and `pred` exits 1; or its featurisation
    failed, item_featurisation_failed), open (entered, neither yet: a process that died inside the item)."""
    its = items()
    failed = [i for i in its if i.get("failed")]
    ok = [i for i in its if i.get("ok")]
    return {"total": len(its), "ok": len(ok), "failed": len(failed), "open": len(its) - len(ok) - len(failed),
            "failed_items": [f"{i.get('item') or '?'}:{(i.get('failed') or '').split(':')[0]}" for i in failed]}


_BUFREL = {"state": None, "released": 0, "calls": 0}                   # release_trimul_buffers' record (report: the trimul lever's buf_release)
TRIMUL_KERNELS_MODULE = "fpf_trimul.kernels"                            # the routed exact-TriMul kernel module; `_BUF` = its persistent per-(key, C, Np, dtype) buffers


def release_trimul_buffers(res: M.Resolution) -> Optional[int]:
    """At an item's entry, drop the exact TriMul kernel's persistent scratch buffers (opt_core/kernels/fpf_trimul/kernels.py `_BUF`: ab, x,
    the LN row statistics and planes — one set per distinct padded N the process has seen, growing with N²) so the
    item about to run allocates its own set fresh and the sets of earlier items are not resident: the kernel's `_buffers` hands a new
    (key, N) exactly what a fresh process gets (zeroed `ab`, empty scratch fully written before read), so the arithmetic is the first
    item's of any process — identical numerics, lower peak on a multi-item process. Only for the `exact` TriMul (the fast cell keeps no
    such table); `na` before the kernel module is imported (the first item). Returns the number of sets released, None when off."""
    if res.trimul != "exact":
        _BUFREL["state"] = "na:trimul_" + str(res.trimul)
        return None
    mod = sys.modules.get(TRIMUL_KERNELS_MODULE)
    if mod is None:
        _BUFREL["state"] = _BUFREL["state"] or "na:not_imported"
        return 0
    buf = getattr(mod, "_BUF", None)
    if not isinstance(buf, dict):
        _BUFREL["state"] = "off:no_BUF_table"
        return None
    n = len(buf)
    buf.clear()
    _BUFREL["state"] = "on"; _BUFREL["released"] += n; _BUFREL["calls"] += 1
    return n


def _apply_to_runner(runner, res: M.Resolution) -> Tuple[dict, List[str], Optional[str]]:
    """The kit's own sequence on a freshly built stock runner (bind_model, then apply(arm)); returns the kit's CFG and the
    partial state: the levers of the mode the CFG does not carry as requested (empty = the whole arm applied) with the kit's reason."""
    LV = _LEVERS
    model = runner.model
    model.eval()
    LV.bind_model(model)
    cfg = LV.apply(res.arm)
    _, all_levers = K.lever_grammar()
    extra = [lv for lv in all_levers if lv not in res.levers and bool(cfg.get(lv))]
    if cfg.get("trimul") not in (res.trimul, "stock") or extra:  # another arm's numerics on the model: refused outright, never admitted as partial
        raise ActivationError(f"the kit applied {cfg} for arm {res.arm!r}: trimul={cfg.get('trimul')!r} expected {res.trimul!r}, levers on that the mode excludes {extra}")
    missing = ([res.trimul] if cfg.get("trimul") != res.trimul else []) + [lv for lv in res.levers if not bool(cfg.get(lv))]
    reason = None if not missing else f"the kit applied {cfg} for arm {res.arm!r}: " + ", ".join(f"{k}={cfg.get('trimul' if k == res.trimul else k)!r} expected {res.trimul if k == res.trimul else True!r}" for k in missing)
    return cfg, missing, reason


def kit_lever_summary() -> dict:
    """The kit's own account (levers_ptx1.describe()), JSON-safe: cfg; `counts` (every wrapper's per-call counters: served, gated,
    fallback — the names report.kit_evidence reads); `sampler` (graphs / hoist
    installed, fast-LN, the loop's and the hoist's counters); fpf_trimul_v4; `card_rows` (the exact-class pair levers' served row ranges on
    this card: levers_ptx1.row_range_facts(), {} where the card has none); `gates` (the kit's two size gates: trimul_tokens, transition_rows); flash_triattn_launch (the routed flash_triattn's
    fast_launch_stats(), when a lever imported it: the direct-launcher state report.kit_evidence reads for the DEGRADED line)."""
    if _LEVERS is None:
        return {}
    try:
        d = _LEVERS.describe()
        out = {"cfg": d.get("cfg"), "counts": d.get("counts"), "sampler": d.get("sampler"), "fpf_trimul_v4": d.get("fpf_trimul_v4"), "card_rows": d.get("card_rows") or {}, "gates": d.get("gates") or {}, "keep_pool": d.get("keep_pool"), "templ": d.get("templ"), "summary_hostidx": d.get("summary_hostidx"), "apb": d.get("apb"), "dit_attn_exact": d.get("dit_attn_exact"), "ditfast": d.get("ditfast"), "lazy_init": d.get("lazy_init"), "trunk2": d.get("trunk2")}   # keep_pool: lib/ptx1_keep_pool.report() (served = the stock in-forward releases skipped); summary_hostidx: lib/ptx1_summary_host.report() (served = samples on the host-index path)
        out["trimul_buffers"] = dict(_BUFREL)                                    # release_trimul_buffers' record (report: the trimul lever's buf_release)
        ft = sys.modules.get("flash_triattn")                                        # the routed kernel, when a lever imported it: its launch-path state (report: DEGRADED gflash fast_launch)
        if ft is not None and hasattr(ft, "fast_launch_stats"):
            out["flash_triattn_launch"] = {k: v for k, v in ft.fast_launch_stats().items() if k not in ("kernel_info", "last_kernel_info")}
        return json.loads(json.dumps(out, default=str))
    except Exception as e:
        return {"error": repr(e)}


def gpu_peak() -> dict:
    """The process's peak device memory so far (torch's allocator: max_memory_allocated / max_memory_reserved, MiB, device 0); {} where
    torch is not loaded or no CUDA device is present. Recorded at exit under report.gpu."""
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return {}
    pk = PHASE.process_peak()                           # the process-wide peak: phase.py resets torch's counters per item for the PEAK line and carries the max
    alloc = max(int(pk.get("alloc") or 0), int(torch.cuda.max_memory_allocated(0))); res = max(int(pk.get("reserved") or 0), int(torch.cuda.max_memory_reserved(0)))
    return {"peak_allocated_mib": alloc // 2**20, "peak_reserved_mib": res // 2**20}


def refresh_levers() -> dict:
    """Re-read the kit's account into the report (after the run: the counters are the evidence the exit rule reads) and the process's
    peak device memory (gpu_peak); returns the report."""
    global _REPORT
    if _REPORT is not None and _REPORT.get("active"):
        _REPORT = dict(_REPORT, levers=kit_lever_summary(), items=items(), items_census=items_census(), gpu=dict(_REPORT.get("gpu") or {}, **gpu_peak()),
                       templates=TEMPLATES.tally(), rowpair=ROWPAIR.record(), big=BIG.fields())
    return status()


def _hook_exit_record() -> Optional[dict]:
    """The environment route's record at process exit (registered once the runner is active under a trigger): the kit's account after
    the run and the exit rule's lines (report.log_exit_lines, the EXIT line). The exit code stays the stock CLI's: the package owns no exit
    on this route."""
    rep = refresh_levers()
    if not rep.get("active"):
        return None
    res = M.resolve(rep["mode"])
    v = R.verdict(rep, res.trimul, res.levers, bool(rep.get("allow_partial")), run_ok=False)
    R.log_exit_lines(v, rep)
    if v["partial"] and not v["allow_partial"] and not v.get("items_failed"):
        R.log(f"{PREFIX} EXIT the exit code is the stock CLI's on the environment route (`pred` is the gated form)")
    R.log(R.exit_line(rep["mode"], "stock", rep.get("n_gpu", 1)))
    return v


STOCK_ATTENTION_MODULES = ("cuequivariance_torch", "cuequivariance_ops_torch")   # the upstream's attention packages; the second builds large autotune key lists AT IMPORT


def _preload_stock_attention() -> dict:
    """Import the stock attention packages NOW, at start-up and at this shallow call depth, in every kit mode. Left to the
    upstream they are imported lazily by the first stock triangle-attention / TriMul call — inside the first item (the confidence head's
    stock triangle attention reaches it first under `fast`), where the same import takes far longer, with the GPU idle. The import
    is the upstream's own dependency and happens in every run anyway: only its moment moves; no statement of the forward changes. A CPU
    process or an absent package is named (preloaded=False reason=...) and the upstream imports as before — never a refusal. The big line's
    own preload (big.preload_stock_attention) finds the package loaded."""
    import time
    t0 = time.time()
    facts = dict(BIG.preload_stock_attention())                 # cuequivariance_torch (CUDA process only; reason= when not)
    extra = []
    if facts.get("preloaded"):
        for name in STOCK_ATTENTION_MODULES[1:]:
            try:
                importlib.import_module(name); extra.append(name)
            except Exception as e:                               # noqa: BLE001 — named; the upstream imports it itself at its first call
                facts[f"{name}"] = f"absent:{type(e).__name__}"
    t1 = time.time()
    facts.update(modules=",".join([facts.get("package") or STOCK_ATTENTION_MODULES[0]] + extra) if facts.get("preloaded") else "-",
                 total_s=round(t1 - t0, 1), t0=f"{t0:.1f}", t1=f"{t1:.1f}")
    R.log(f"{PREFIX} NOTE stock_attention_preload " + " ".join(f"{k}={v}" for k, v in facts.items()))
    try:                                                         # the shared core's own start-up import of the attention libraries (opt_core.warm_imports:
        import opt_core                                          # free when the modules are already loaded, as they are here) — one
        words = opt_core.warm_imports(origin="protenix_v1")      # statement for every kit; named when the importable core lacks it (the pin floor carries it)
        R.log(f"{PREFIX} NOTE core_warm_imports " + (" ".join(f"{k}={v}" for k, v in words.items()) if isinstance(words, dict) else str(words)))
    except AttributeError:
        R.log(f"{PREFIX} NOTE core_warm_imports absent (opt_core < 0.5.66.0)")
    except Exception as e:                                       # noqa: BLE001 — never a refusal: the kit's own preload above already ran
        R.log(f"{PREFIX} NOTE core_warm_imports error={type(e).__name__}")
    return facts


def stock_knobs_of(configs) -> dict:
    """Upstream's stock kernel / dtype knobs as its own configs carry them after its CLI parsed the run's flags (--triatt_kernel ->
    configs.triangle_attention, --trimul_kernel -> configs.triangle_multiplicative, --dtype -> configs.dtype): the words the exit rule reads
    (report.STOCK_KNOB_DEFAULTS / knob_states). A missing attribute reads None (an upstream without the knob)."""
    def get(name):
        try:
            v = configs.get(name) if hasattr(configs, "get") else getattr(configs, name, None)
        except Exception:                                        # noqa: BLE001 — a config object that raises on a missing key
            v = getattr(configs, name, None)
        return None if v is None else str(v)
    if configs is None:
        return {}
    return {"triatt_kernel": get("triangle_attention"), "trimul_kernel": get("triangle_multiplicative"), "dtype": get("dtype")}


def _wrap_runner(res: M.Resolution) -> None:
    if _RUNNERS["wrapped"]:
        return
    mod = importlib.import_module(RUNNER_MODULE)
    cls = getattr(mod, RUNNER_CLASS)
    if "lazy_init" in res.levers:                                # the one lever that acts AHEAD of the model: the stock constructor's dead random init skipped
        try:                                                     # (lib/ptx1_lazy_init.py; the strict checkpoint load overwrites every parameter). PTX_LAZY_INIT=recheck:
            _put_kit_on_path()                                   # both constructions in this process + the N/N state-tensor comparison. A failed install is
            import ptx1_lazy_init as LZ                          # named by the activation (levers_ptx1.set_lazy_init leaves the word unset -> partial by name)
            LZ.install(cls, mode="recheck" if os.environ.get("PTX_LAZY_INIT", "") == "recheck" else "1", log=R.log)
        except Exception as e:                                   # noqa: BLE001
            R.log(f"{PREFIX} NOTE lazy_init not installed: {e!r}"[:300])
    _preload_stock_attention()                                   # every kit mode, before the model: the stock attention packages imported at start-up (below)
    orig = cls.__init__

    def __init__(self, *a, _orig=orig, **kw):
        global _REPORT
        if _RUNNERS["built"]:
            raise ActivationError("a second InferenceRunner in one process: the kit binds one model per process (start a new process)")
        _orig(self, *a, **kw)
        _RUNNERS["built"] += 1
        rep = dict(_REPORT or {}, stock_knobs=stock_knobs_of(getattr(self, "configs", None)))   # upstream's kernel / dtype knobs of this run (the exit rule reads them: report.knob_states)
        cfg, partial, why = None, [], None
        try:
            cfg, partial, why = _apply_to_runner(self, res)
            if partial and not rep.get("allow_partial"):
                raise ActivationError(R.partial_exit_reason(partial, why))
        except Exception as e:                       # the kit refused the arm (or applied another): never run the model half-patched
            reason = f"the kit refused arm {res.arm!r}: {e}" if not isinstance(e, ActivationError) else str(e)
            fields = {"partial": partial, "partial_reason": why} if isinstance(e, ActivationError) and partial else {}
            try:
                _refuse(rep, reason, strict=True, **fields)
            except ActivationError:
                if rep.get("trigger"):               # the environment route (.pth): stop the process here, exit 3, like every other refusal
                    sys.exit(R.EXIT_NOT_ACTIVE)
                raise
        try:
            memory = BIG.apply(self, res) if res.memory_preset else None   # the memory mode's install: the engine adapter (big.py: the settings on the configs, the patched stock statements, the census hooks); a refusal by name, never a half-patched model
        except ActivationError as e:
            try:
                _refuse(rep, str(e), strict=True)
            except ActivationError:
                if rep.get("trigger"):
                    sys.exit(R.EXIT_NOT_ACTIVE)
                raise
        rep = dict(rep, active=True, armed=True, cfg=cfg, partial=partial, partial_reason=why, levers=kit_lever_summary(), gpu=torch_gpu_info(), memory=memory, big=BIG.fields())
        _REPORT = rep
        R.log_activation(rep)
        if memory:
            R.log_memory(rep)
        if partial:
            R.log(R.partial_allowed_line(partial, why))
        if rep.get("trigger") and not _RUNNERS["hook_registered"]:
            atexit.register(_hook_exit_record)
            _RUNNERS["hook_registered"] = True

    __init__.__wrapped__ = orig
    cls.__init__ = __init__
    PHASE.install(PREFIX, log=R.log)                            # the per-item PHASE line (phase.py): inside this wrap, around the stock predict body
    orig_predict = cls.predict

    def predict(self, data, *a, _orig=orig_predict, **kw):     # the per-item entry (runner/inference.py:204): the item's sizes recorded, the exact TriMul's buffers of earlier items released, the template slots censused (templates.check_item: an all-dummy templated item is named and counted, then runs untemplated exactly as the stock runs it), the call unchanged
        _ITEMS.append(item_record(data))                       # n_gpu>1: rank 0's item on every rank already (rowpair.item_of_rank0, inside the dataset's __getitem__)
        try:
            release_trimul_buffers(res)
            TEMPLATES.check_item(getattr(self, "configs", None), data)
            data = ROWPAIR.host_side_inputs(data, getattr(self, "configs", None))   # n_gpu>1: pair-shaped input features stay on the host (no-op at n_gpu=1)
            out = _orig(self, data, *a, **kw)
        except SystemExit:
            raise
        except BaseException as e:                              # the stock's per-item handler (runner/inference.py:498-509) logs and continues with exit 0; the kit names the item failed (ITEM line, the ITEMS census, `pred` exits 1)
            _ITEMS[-1]["failed"] = f"{type(e).__name__}: {str(e)[:200]}"
            R.log(R.item_failed_line(_ITEMS[-1]))
            raise
        _ITEMS[-1]["ok"] = True
        return out

    predict.__wrapped__ = orig_predict
    cls.predict = predict
    _RUNNERS["wrapped"] = True


def activate(mode: str, strict: bool = False, trigger: Optional[str] = None, det: bool = False, allow_partial: Optional[bool] = None,
             n_gpu=None) -> dict:
    global _REPORT
    if _REPORT is not None:
        return dict(_REPORT)
    allow = bool(allow_partial)
    try:
        m = M.check_mode(mode)
    except ValueError as e:
        return _refuse(_base(str(mode), trigger, det, allow), str(e), strict)
    rep = _base(m, trigger, det, allow)
    try:                                                         # the --n_gpu axis (ngpu.py over opt_core.mem.ngpu): P recorded, every rule a named refusal
        p = NGPU.effective(n_gpu)
        rep.update(NGPU.record(p))
        NGPU.admit(p, m)
    except ValueError as e:
        return _refuse(rep, f"n_gpu: {e}", strict)
    except NGPU.NG.NGpuRefused as e:
        return _refuse(rep, str(e.reason), strict)
    if p > 1 and not ROWPAIR.is_rank_process():                 # P>1 activates inside the rank processes `pred --n_gpu P` launches (rowpair.py), not in the launching process
        return _refuse(rep, f"n_gpu={p}: the mode is activated inside the rank processes `pred --n_gpu {p}` launches, not in the launching process", strict)
    if m == "off":
        from . import ablation as A                          # MODEL_OPT_LEVERS_OFF under the stock mode: refused by name (the stock route applies no lever)
        if A.requested():
            return _refuse(rep, A.off_refusal(A.requested()), strict)
        _REPORT = dict(rep, active=False, armed=False, reason="mode off applies nothing in this process (the stock route is `pred --mode off`)")
        return dict(_REPORT)
    try:
        if p > 1 and m == BIG.MODE:                             # n_gpu>1: the row-sharded pair stack's host-parking levers exported unless the caller set them (the launching process exports the same to its ranks: cli._run_ranks); the P=1-only memory levers are off by the line's in-process selection (big.rowpair_switches)
            rep["ngpu_regime"] = {k: {"value": v, "source": src} for k, (v, src) in NGPU.large_input_regime(os.environ).items()}
        res = M.resolve(m)
    except Exception as e:
        return _refuse(rep, f"mode {m}: {e}", strict)
    rep.update(arm=res.arm, line=res.line, tier=res.tier, levers_requested=list(res.levers),
               levers_ablated=list(res.ablated), mode_arm=res.mode_arm,
               levers_capped=list(res.capped), graph_cap=res.graph_cap)     # + the sampler-graph levers above the graph cap for this run's input (modes.graph_cap); MODEL_OPT_LEVERS_OFF (ablation.py, applied in modes.resolve): the withheld levers ride on the report for the ARMED / ACTIVE token and the LEVER lines
    reasons = gates(res, det)
    if reasons:
        return _refuse(rep, "; ".join(reasons), strict)
    if det:
        from . import det as D
        try:
            rep["det_report"] = D.install()
        except Exception as e:
            return _refuse(rep, f"det: {e}", strict)
    if res.mode == BIG.MODE:                                   # the memory levers' activation-time part (big.arm: opt_core.mem.apply with the line's explicit selection and settings, the census opt-out of this route, the allocator lever exported, recycle_carry's host park opened) before the stock runner exists
        try:
            rep["big"] = BIG.arm(res, allow_partial=allow, opt_out=BIG.route_opt_out(rep))
        except BIG.BigError as e:
            return _refuse(rep, f"big: {e}", strict)
    try:
        rep["kernels"] = route_kernels()
    except ActivationError as e:
        return _refuse(rep, str(e), strict)
    except Exception as e:
        return _refuse(rep, f"routing the kernels: {e!r}", strict)
    rep["triattn_floor"] = export_triattn_floor(res)
    rep["sys_path_added"] = _put_kit_on_path()
    try:
        _import_levers()
    except ActivationError as e:
        return _refuse(rep, str(e), strict)
    except Exception as e:
        return _refuse(rep, f"importing {K.LEVERS_MODULE}: {e!r}", strict)
    try:
        TEMPLATES.install()                                      # the template guard (templates.py): dropped hits named, all-dummy templated items named and counted
        rep["template_guard"] = "installed"
    except Exception as e:
        return _refuse(rep, f"template guard: {e!r}", strict)
    if p > 1:                                                    # a rank process of a P>1 launch: join the group and install the sharded pair stack (after the recipe and the levers: the model package imports here)
        try:
            st = ROWPAIR.init_rank()
            if int(st.get("P") or 0) != p:                       # fail-closed: the group this rank joined must be the P the command line asked for
                return _refuse(rep, NGPU.mismatch_reason(p, st.get("P")), strict)
            ROWPAIR.install(det=bool(det), on_item_error=item_featurisation_failed)   # the replicated-tensor sync policy follows the determinism level (rowpair.sync_policy); a featurisation failure rank 0 met is every rank's failed item
            rep["rowpair"] = ROWPAIR.record()
            rep["gpu_rank"] = st.get("rank")
        except Exception as e:
            return _refuse(rep, f"rowpair: {e!r}", strict)
    _REPORT = dict(rep, armed=True, active=False)
    try:
        _wrap_runner(res)
    except Exception as e:
        return _refuse(rep, f"wrapping {RUNNER_MODULE}.{RUNNER_CLASS}: {e!r}", strict)
    rep = dict(_REPORT, gpu=torch_gpu_info())
    _REPORT = rep
    R.log_armed(rep)
    R.log_kernels(rep)
    return dict(rep)
