"""``design`` — the upstream tool's own action, one input contract for every mode: upstream's experiment file (the YAML `genie3 generate
-c` reads: ``experiment``, ``paths`` {rootdir, dataset}, ``generation`` {dataset {source, selections, n_sample, batch_size}, sampler {sampler
{direction_scale}}, io {outdir}, inference}, ``runtime``; src/genie3/config/loader.py) plus upstream's own `generate` flags, passed to the stock
child verbatim: ``--verbose``, ``--log-dir <dir>``, ``--num-devices N``, ``--shard-id K``, ``--num-shards M`` (src/genie3/cli.py:57-110).

One request = one process on every arm. The package writes its OWN copy of the request file (``<out>/request.yaml``; the input is never
edited) with only these keys composed, each only when the pass asks for it: ``paths.rootdir`` = ``--out_dir`` when given (upstream's
``generation.io.outdir`` follows it, loader.py:246-259; without ``--out_dir`` the request's own output directory stands), ``experiment.seed`` =
``--seed`` when given, else the file's own, else — under ``--det 1`` only — the recipe's 0 (det.py; ``--det 0``, the default, leaves an unseeded
request unseeded: upstream's shipped numerics), ``generation.base.checkpoint`` / ``generation.base.config`` = the two weight files under
GENIE3_WEIGHTS as absolute paths ONLY when the request does not name its own (upstream's defaults are the checkout-relative ``pretrained/v1/…``,
loader.py:201-202; with GENIE3_WEIGHTS unset and no key in the file, upstream's default stands), and, when given, ``--n`` →
``generation.dataset.n_sample``, ``--selections`` → ``generation.dataset.selections``, ``--batch_size`` → ``generation.dataset.batch_size``.
Everything else passes verbatim (``paths.dataset`` stays checkout-relative: every line runs with cwd = $GENIE3_ROOT, the layout upstream's
problem files assume). The batch size is the request's ``generation.dataset.batch_size`` on every arm (``--batch_size`` when given, else the
file's own key; absent: upstream's default 1 on every line — modes.with_batch; the batched throughput configuration is ``--batch_size 8``).

* ``--mode off``: the stock caller (stock_cli.py) in a clean subprocess: ``genie3 generate -c <out>/request.yaml --log-dir <out>/logs
  [the caller's generate flags]`` (src/genie3/cli.py:61,71) with cwd = the checkout, plus ``<out>/stock_env_proof.json`` carrying the pins
  report. Before the child starts the KERNELS census of the child's environment is printed (``kernels_line``: torch's TF32 switches as the
  stripped environment leaves them — no library override reaches the child, stack.DROP_ENV_NAMES / stock/PINS.json must_be_absent_prefixes).
  Every request and every generate flag upstream accepts runs here.
* a kit mode: the mode's line from modes.resolve — ``<genie3_opt>/g3batch.py --config <out>/request.yaml --outdir <out> <flags> --timings
  <out>/timings.json [--shard-id K --num-shards M]`` (stack.driver_command) in one process, cwd = the checkout. Upstream's dataset shards
  (``--num-shards M --shard-id K``) run on every mode: the driver slices the request as upstream's loader does (``sample_index_offset``: designs
  ``<problem>_<i>`` for the shard's share of ``n_sample``) and the pass writes upstream's shard marker (``shard_marker``); a shard with no share
  runs nothing and exits 0 with its marker, as upstream does. The sequence stage (``sampler.predict_sequence``) runs on every mode: the driver
  runs upstream's own decode after its structure loop and the PDBs carry the sequence. A request that names a computation the kit line does not
  run is REFUSED BY NAME before anything runs (``kit_refusals``: upstream's beam search ``generation.inference.search``, the sampler's
  ``predict_sidechain`` stage, ``--num-devices`` / ``runtime.num_devices`` > 1): one line ``[genie3-opt] NOT ACTIVE: mode=<m> cannot serve
  <features> — …; run --mode off for the stock path``, exit EXIT_NOT_ACTIVE (3); ``--mode off`` runs every request upstream accepts. After a kit
  pass the driver's log and timings JSON are read for every lever's own evidence (registry.Lever.evidence) and the log for the forbidden lines. A
  mode is ALL of its levers, never a subset: a planned lever without its evidence could not run on this box (a kernel that would not compile or
  launch, an unsupported shape) and the MODE REFUSES BY NAME — ``[genie3-opt] NOT ACTIVE: mode=<m> refused — lever(s) <ids> could not run …``,
  ``levers_missing: [<lever>, ...]`` in the manifest, status ``refused``, exit EXIT_NOT_ACTIVE (3); the files the pass wrote are not the mode's.
  Uncertainty about the environment alone (another GPU, another torch / triton patch level, a card without a cell row) is named on the
  activation lines and every lever engages — it is never a reason to refuse. Two records fail a pass instead — a forbidden line (a traceback, an
  assertion, the RNG-bookkeeping or graph-replay divergence markers: status ``failed``, exit EXIT_FAIL (1), ``forbidden_lines``) and a lever
  that RAN but whose own record contradicts its contract (registry evidence kind ``judged``: the hoist's anomaly count, the batch table, the
  TF32 readback; ``levers_broken``, status ``failed``, exit 1). Never silent. A batch the driver's memory model refuses before running (``CAPACITY … verdict=refused``) is exit 3 with the batch size that fits named.
* both: a PDB count that differs from the request (problems x n_sample, counted over the design names the request produces and written during
  this pass) is the other named state, ``incomplete: <found>/<expected>``, status ``incomplete``, exit EXIT_FAIL (1); a driver or stock
  process that fails keeps EXIT_FAIL, and a refused mode keeps its refusal, with ``incomplete`` recorded beside it. ``run`` returns (exit code, manifest): the
  in-process route (``genie3_opt.enable(M)`` + ``run``) carries the code in that return and never sets the host process's exit; ``genie3-opt
  design`` is the gated form.
* both: outputs as upstream's writer writes them — ``<out>/<problem>/pdbs/<problem>_<i>.pdb`` (src/genie3/generation/runner/
  postprocess.py:150-163; the driver calls the same writer) — plus ``opt_manifest.json`` beside them (manifest.py) and the exit tally
  (report.py). The stock line also writes ``<out>/generation_stats/main_rank_0.json`` (runner.py:619-641) and its run log under ``--log-dir``;
  the driver line writes ``<out>/timings.json``. Neither is a design output. An output directory that already holds designs is written into,
  as upstream writes into it.
"""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import yaml

from . import det, manifest as _manifest, outputs as _outputs, report as _report, stack
from .codes import EXIT_FAIL, EXIT_NOT_ACTIVE, EXIT_OK
from .modes import ModeError, batch_size, effective, precision_of, trimul_of, request_batch, resolve
from .registry import FORBIDDEN_LINES, KIT_DIRS, LEVERS

REQUEST_NAME, TIMINGS_NAME, STOCK_LOG, DRIVER_LOG, PROOF_NAME, PINS_NAME = "request.yaml", "timings.json", "stock.log", "design.log", "stock_env_proof.json", "pins_check.json"
STOCK_VERB = "generate"                                              # src/genie3/cli.py: the generation-only verb
SELECTION_RE = re.compile(r"^[A-Za-z0-9._-]+$")
TARGET_SOURCES = ("target", "unconditional", "motif")                # src/genie3/generation/config/data/sample_dataset.py
TF32_ENV = ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "TORCH_ALLOW_TF32_CUDNN_OVERRIDE")   # the library overrides that would move a process off strict fp32 (read at cuBLAS / torch initialisation); the KERNELS census names any present in the stock child's environment


class DesignError(ValueError):
    """A request the package cannot read, with the named fact."""


# ------------------------------------------------------------------------------------------------------------- stock generate flags
def stock_flag_words(verbose: bool = False, log_dir: Optional[str] = None, num_devices: Optional[int] = None, shard_id: Optional[int] = None,
                     num_shards: Optional[int] = None) -> List[str]:
    """Upstream's own `generate` flags as words for the stock child, in upstream's spelling (src/genie3/cli.py:57-110); None / False = not given."""
    words: List[str] = []
    if verbose:
        words.append("--verbose")
    if log_dir is not None:
        words += [det.LOG_DIR_FLAG, os.path.abspath(log_dir)]
    if num_devices is not None:
        words += ["--num-devices", str(int(num_devices))]
    if shard_id is not None:
        words += ["--shard-id", str(int(shard_id))]
    if num_shards is not None:
        words += ["--num-shards", str(int(num_shards))]
    return words


def kit_refusals(raw: dict, num_devices: Optional[int] = None) -> List[str]:
    """The named computations a kit line does not run, each ``<feature>: <mechanism>`` — a request naming one is refused by name before anything
    runs (exit 3; ``--mode off`` runs it): upstream's beam search (generation.inference.search, loader.py:221-228: BeamSearch.run replaces the
    sampler's step loop), the sidechain stage (generation.sampler.sampler.predict_sidechain: a second generation stage over the first stage's
    PDBs, workflow.py:194-216), more than one device (--num-devices / runtime.num_devices: Lightning DDP ranks; the kit line is one process on
    one GPU — dataset shards, --num-shards / --shard-id, are the kit line's multi-GPU form and run on every mode). The sequence stage
    (sampler.predict_sequence: upstream's Sampler._sample_sequence after the structure loop) runs on every mode, in the driver."""
    out: List[str] = []
    gen = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    inf = gen.get("inference") if isinstance(gen.get("inference"), dict) else {}
    if inf.get("search"):
        out.append("inference.search: upstream's beam search replaces the sampler's step loop with BeamSearch.run (beam_width² look-ahead branches denoised to t=0 and scored "
                   "by a reward model every score_interval steps); the kit's driver runs the single-trajectory DDIM loop")
    smp = gen.get("sampler") if isinstance(gen.get("sampler"), dict) else {}
    inner = smp.get("sampler") if isinstance(smp.get("sampler"), dict) else {}
    if inner.get("predict_sidechain") or smp.get("predict_sidechain"):
        out.append("sampler.predict_sidechain: upstream runs a second generation stage over the first stage's PDBs (dataset source `sidechain`, workflow.py); the kit's "
                   "driver runs the backbone stage only")
    rt = raw.get("runtime") if isinstance(raw.get("runtime"), dict) else {}
    asked = num_devices if num_devices is not None else rt.get("num_devices")   # --num-devices, else runtime.num_devices; YAML "2" / 2.0 name two devices like 2
    nd: Optional[int] = None
    if asked is not None:
        try:
            nd = int(asked) if float(asked) == int(float(asked)) else None
        except (TypeError, ValueError):
            nd = None
        if nd is None:
            out.append(f"num-devices={asked!r}: not an integer device count (upstream's --num-devices / runtime.num_devices)")
    if nd is not None and nd > 1:
        out.append(f"num-devices={nd}: upstream's multi-device run is Lightning DDP (N ranks over an interleaved sampler split); the kit line is one process on one GPU — "
                   f"run one pass per GPU with --shard-id K --num-shards {nd}")
    return out


def shard_slice(n_sample: int, shard_id: Optional[int], num_shards: Optional[int]) -> Tuple[int, int]:
    """(first design index, one past the last) of this pass's share of every problem's ``n_sample`` designs — upstream's slicing
    (config/loader.py _apply_shard_to_config: ceil(n / M) per shard, shard K starts at K·ceil(n / M), clipped to n; ``num_shards`` 1 / None = the
    whole request, ``shard_id`` None = 0)."""
    n = int(n_sample)
    m = int(num_shards) if num_shards is not None else 1
    if m <= 1:
        return 0, n
    per = -(-n // m)
    start = min(int(shard_id or 0) * per, n)
    return start, min(start + per, n)


def shard_marker(out_dir: str, shard_id: int, num_shards: int, problems: List[str]) -> List[str]:
    """Upstream's shard-completion marker, written where upstream writes it (runtime/shards.py write_generation_shard_marker: ``<dir>/.shard_markers/
    generate_shard_<K>_of_<M>.done`` in every problem directory of the pass that holds ``pdbs/``, else in the output directory itself) so that
    upstream's ``evaluate`` / ``status`` read a sharded kit pass as they read a stock one. Returns the marker paths."""
    written = []
    for d in _marker_targets(out_dir, problems):
        md = os.path.join(d, ".shard_markers")
        os.makedirs(md, exist_ok=True)
        mp = os.path.join(md, f"generate_shard_{int(shard_id)}_of_{int(num_shards)}.done")
        open(mp, "a").close()
        written.append(mp)
    return written


# ---------------------------------------------------------------------------------------------------------------------- request
def _marker_targets(out_dir: str, problems: List[str]) -> List[str]:
    """Where upstream keeps a pass's shard markers (runtime/shards.py write_generation_shard_marker / is_generation_shard_done, one rule for
    both): every directory of the output directory that holds ``pdbs/`` and is one of the pass's selected problems (all of them when the
    request selects none), else the output directory itself."""
    if not os.path.isdir(out_dir):
        return [out_dir]
    selected = [p for p in problems if p != _outputs.UNCONDITIONAL]
    dirs = sorted(os.path.join(out_dir, d) for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d, "pdbs")))
    if selected:
        dirs = [d for d in dirs if os.path.basename(d) in selected]
    return dirs or [out_dir]


def shard_done(out_dir: str, shard_id: int, num_shards: int, problems: List[str]) -> Optional[List[str]]:
    """Upstream's resume rule (workflow.py: a shard whose ``generate_shard_<K>_of_<M>.done`` marker exists under every marker target is
    'already complete' — nothing is generated, exit 0): the marker paths when this shard is done, else None."""
    paths = [os.path.join(d, ".shard_markers", f"generate_shard_{int(shard_id)}_of_{int(num_shards)}.done") for d in _marker_targets(out_dir, problems)]
    return paths if all(os.path.isfile(p) for p in paths) else None


def load_request(path: str) -> dict:
    try:
        raw = yaml.safe_load(open(path, encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise DesignError(f"request file {path}: {e}") from e
    if not isinstance(raw, dict) or not isinstance(raw.get("generation"), dict) or not isinstance(raw["generation"].get("dataset"), dict):
        raise DesignError(f"request file {path}: expected upstream's experiment YAML with a `generation.dataset` section (src/genie3/config/loader.py)")
    return raw


def request_out_dir(raw: dict, root: Optional[str]) -> Optional[str]:
    """The request's own output directory as upstream resolves it (loader.py:246-259: generation.io.outdir, else paths.rootdir; a relative path is
    relative to the process cwd = the checkout), absolute; None when the file names neither."""
    gen = raw.get("generation") if isinstance(raw.get("generation"), dict) else {}
    io = gen.get("io") if isinstance(gen.get("io"), dict) else {}
    paths = raw.get("paths") if isinstance(raw.get("paths"), dict) else {}
    d = io.get("outdir") or paths.get("rootdir")
    if not d:
        return None
    d = str(d)
    return os.path.abspath(d if os.path.isabs(d) else os.path.join(root or os.getcwd(), d))


def weights_paths(raw: dict, root: Optional[str]) -> Optional[Tuple[str, str]]:
    """The two weight files the request names itself (generation.base.checkpoint / config, checkout-relative when relative), or None when it
    names neither — then GENIE3_WEIGHTS (or upstream's default under the checkout) supplies them."""
    base = (raw.get("generation") or {}).get("base") if isinstance((raw.get("generation") or {}).get("base"), dict) else {}
    if not base.get("checkpoint") and not base.get("config"):
        return None
    out = []
    for key, rel in zip(("checkpoint", "config"), stack.WEIGHT_FILES):
        v = base.get(key)
        if v:
            out.append(os.path.abspath(v if os.path.isabs(str(v)) else os.path.join(root or os.getcwd(), str(v))))
        else:                                                            # one key named, the other upstream's default under the checkout
            out.append(os.path.join(root or os.getcwd(), "pretrained", "v1", *rel.split("/")))
    return out[0], out[1]


def compose_request(raw: dict, out_dir: Optional[str], weights: Optional[str], seed: Optional[int] = None, n_sample: Optional[int] = None,
                    selections: Optional[str] = None, det_level: int = det.DEFAULT_LEVEL, batch: Optional[int] = None) -> Tuple[dict, dict]:
    """The package's copy of a request (a NEW dict) and the record of what was composed into it — only the keys this pass asks for."""
    req = copy.deepcopy(raw)
    composed: Dict[str, object] = {}
    req.setdefault("paths", {})
    if not isinstance(req["paths"], dict):
        raise DesignError("request: `paths` must be a mapping")
    gen = req["generation"]
    if out_dir is not None:                                             # --out_dir given: the one output directory of this pass
        req["paths"]["rootdir"] = os.path.abspath(out_dir)
        composed["paths.rootdir"] = req["paths"]["rootdir"]
        io = gen.get("io")
        if isinstance(io, dict) and "outdir" in io:                    # loader.py:246-250 refuses an outdir that disagrees with paths.rootdir
            del io["outdir"]
            composed["generation.io.outdir"] = "removed (--out_dir is paths.rootdir, the one output directory)"
    req.setdefault("experiment", {})
    if not isinstance(req["experiment"], dict):
        raise DesignError("request: `experiment` must be a mapping")
    if out_dir is not None:
        req["experiment"].setdefault("name", os.path.basename(os.path.abspath(out_dir)))
    s = det.request_seed(raw, seed, det_level=det_level)
    if s is None:
        composed["experiment.seed"] = "absent (upstream's default, unseeded: config/models.py:18)"
    else:
        if req["experiment"].get("seed") != s:
            composed["experiment.seed"] = s
        req["experiment"]["seed"] = s
    base = gen.get("base") if isinstance(gen.get("base"), dict) else None
    if weights and not (base and (base.get("checkpoint") or base.get("config"))):   # the request names no weights of its own: GENIE3_WEIGHTS supplies them; a request's own keys stand
        base = gen.setdefault("base", {})
        base["checkpoint"] = os.path.join(weights, *stack.WEIGHT_FILES[0].split("/"))
        base["config"] = os.path.join(weights, *stack.WEIGHT_FILES[1].split("/"))
        composed["generation.base.checkpoint"] = base["checkpoint"]
        composed["generation.base.config"] = base["config"]
    ds = gen["dataset"]
    if n_sample is not None:
        ds["n_sample"] = int(n_sample)
        composed["generation.dataset.n_sample"] = int(n_sample)
    if selections is not None:
        ds["selections"] = selections
        composed["generation.dataset.selections"] = selections
    if batch is not None:                                               # --batch_size: upstream's own key, read by every route (loader.py; a kit line's --batch-size follows it, modes.with_batch)
        ds["batch_size"] = int(batch)
        composed["generation.dataset.batch_size"] = int(batch)
    return req, composed


def request_problems(req: dict, genie3_root: Optional[str]) -> List[str]:
    """The problem keys a request runs (its selections, else every problem JSON of its dataset under the checkout), or [] when unknown;
    an unconditional request runs the one pseudo-problem UNCONDITIONAL (its designs `<length>_<i>` land in `<out>/pdbs/`, postprocess.py:69-82)."""
    ds = req["generation"]["dataset"]
    if ds.get("source") == "unconditional":
        return [_outputs.UNCONDITIONAL]
    sel = ds.get("selections")
    if sel:
        return [x.strip() for x in str(sel).split(",") if x.strip() and SELECTION_RE.match(x.strip())]
    if ds.get("source") in ("target", "motif") and genie3_root and req["paths"].get("dataset"):
        d = os.path.join(genie3_root, req["paths"]["dataset"], "problems")
        if os.path.isdir(d):
            return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".json"))
    return []


def request_record(req: dict, composed: dict, problems: List[str], request_path: str, shard_id: Optional[int] = None, num_shards: Optional[int] = None) -> dict:
    """What the pass is asked to write: the problems, and per problem the design names of this pass's share of ``n_sample`` — the whole request, or
    under ``--num-shards M --shard-id K`` the shard's slice (shard_slice; names ``<problem>_<i>`` / ``<length>_<i>`` over the slice's indices, as
    upstream's datasets name them from ``sample_index_offset``: sample_dataset/target.py:98-102, unconditional.py:61-65)."""
    ds = req["generation"]["dataset"]
    n = ds.get("n_sample")
    lo, hi = shard_slice(n, shard_id, num_shards) if isinstance(n, int) else (0, None)
    k = (hi - lo) if hi is not None else None
    rec = {"path": os.path.abspath(request_path), "source": ds.get("source"), "problems": problems, "n_sample": n, "seed": req["experiment"].get("seed"),
           "batch_size": ds.get("batch_size"), "selections": ds.get("selections"), "dataset": req["paths"].get("dataset"),
           "sampler": (req["generation"].get("sampler") or req["generation"].get("inference") or {}),
           "composed": composed, "designs_expected": (len(problems) * k) if (problems and k is not None) else None, "names_expected": None,
           "shard": f"{int(shard_id or 0)}/{int(num_shards)}" if (num_shards is not None and int(num_shards) > 1) else None, "design_indices": [lo, hi] if hi is not None else None}
    if problems and k is not None and ds.get("source") in ("target", "motif"):
        rec["names_expected"] = {p: [f"{p}_{i}" for i in range(lo, hi)] for p in problems}   # postprocess.py: <problem>_<i>.pdb under <out>/<problem>/pdbs/
    if ds.get("source") == "unconditional" and all(isinstance(ds.get(x), int) for x in ("min_length", "max_length", "length_step")) and k is not None:
        lengths = list(range(int(ds["min_length"]), int(ds["max_length"]) + 1, int(ds["length_step"])))   # one design per (length, sample): sample_dataset/unconditional.py:55-65
        rec.update(lengths=lengths, designs_expected=len(lengths) * k, names_expected={_outputs.UNCONDITIONAL: [f"{L}_{i}" for L in lengths for i in range(lo, hi)]})
    return rec


def write_request(req: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, REQUEST_NAME)
    with open(p, "w", encoding="utf-8") as fh:
        yaml.safe_dump(req, fh, sort_keys=False)
    return p


# --------------------------------------------------------------------------------------------------------------------- stock arm
def stock_environment(environ=None) -> Tuple[Dict[str, str], List[str], List[str]]:
    """The stock process's environment: every name under stock/PINS.json must_be_absent_prefixes stripped except the allowed exceptions,
    PYTHONPATH entries inside a kit directory removed. Returns (env, stripped names, the names to prove absent)."""
    environ = os.environ if environ is None else environ
    se = stack.pins()["stock_environment"]
    absent, allowed = list(se["must_be_absent_prefixes"]), list(se.get("allowed_exceptions", []))
    env, stripped = {}, []
    for k, v in environ.items():
        hit = any((n.endswith("_") and k.startswith(n)) or k == n for n in absent) and k not in allowed
        if hit:
            stripped.append(k)
            continue
        env[k] = v
    kits = [stack.kit_dir(k) for k in KIT_DIRS]
    if env.get("PYTHONPATH"):
        keep = [p for p in env["PYTHONPATH"].split(os.pathsep) if p and not any(os.path.realpath(p).startswith(os.path.realpath(d)) for d in kits)]
        env["PYTHONPATH"] = os.pathsep.join(keep)
        if not env["PYTHONPATH"]:
            del env["PYTHONPATH"]
    return env, sorted(stripped), absent


def kernels_census(env: Dict[str, str]) -> dict:
    """The stock child's matmul numerics as its environment leaves them, read WITHOUT starting a process: torch initialises TF32 matmuls off and
    cuDNN TF32 on unless a library override is present, and the stripped environment carries none (TF32_ENV) — any present is named."""
    present = {k: env.get(k) for k in TF32_ENV if k in env}
    tf32 = any((present.get(k) or "").strip() not in ("", "0") for k in ("NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"))
    return {"tf32_matmul": bool(tf32), "cudnn_tf32": "default", "float32_matmul_precision": "default", "torch": stack.dist_version("torch"), "overrides": present, "source": "environment"}


def stock_command(request_path: str, out_dir: str, root: str, python: Optional[str] = None, stock_words: Optional[List[str]] = None) -> List[str]:
    """``python -s -m genie3_opt.stock_cli --proof-json … --env-absent … --allowed … --kit-dirs … --genie3-root … --pins-check … -- generate -c <request>
    --log-dir <out>/logs [upstream's generate flags as the caller gave them]`` (the caller's own ``--log-dir`` replaces ``<out>/logs``)."""
    se = stack.pins()["stock_environment"]
    kits = os.pathsep.join(stack.kit_dir(k) for k in KIT_DIRS)
    words = list(stock_words or [])
    cmd = [python or sys.executable, "-s", "-m", "genie3_opt.stock_cli", "--proof-json", os.path.join(out_dir, PROOF_NAME), "--env-absent", ",".join(se["must_be_absent_prefixes"]),
           "--allowed", ",".join(se.get("allowed_exceptions", [])), "--kit-dirs", kits, "--genie3-root", root, "--pins-check", os.path.join(out_dir, PINS_NAME)]
    tail = [STOCK_VERB, "-c", request_path]
    if det.LOG_DIR_FLAG not in words:
        tail += [det.LOG_DIR_FLAG, os.path.join(out_dir, "logs")]
    return cmd + ["--"] + tail + words


def write_pins_check(rep: dict, out_dir: str) -> str:
    """The activation's pins report (stock/check_pins.py) written beside the outputs for the stock caller to copy into its proof."""
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, PINS_NAME)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump({"bad": rep.get("pins", {}).get("bad"), "detail": rep.get("pins", {}).get("detail")}, fh, indent=1, default=str)
    return p


# ------------------------------------------------------------------------------------------------------------------- evidence
def read_timings(path: str) -> dict:
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def numerics_live(T: dict) -> Optional[dict]:
    """The driver's numerics readback as {"matmul_tf32": bool, "matmul": word, "cudnn_tf32": bool} from its timings record (``numerics_readback``:
    opt_core.precision.policy.live at set-up; else the process-global ``global_numerics_state`` snapshot the loop ran under), or None when
    the record carries neither (a pass that returned 0 without one is failed by the caller: the readback is the fact, never the flag)."""
    rb = T.get("numerics_readback")
    if isinstance(rb, dict) and "matmul_tf32" in rb:
        return {"matmul_tf32": bool(rb.get("matmul_tf32")), "matmul": rb.get("matmul"), "cudnn_tf32": rb.get("cudnn_tf32")}
    g = T.get("global_numerics_state")
    if isinstance(g, dict) and "cuda.matmul.allow_tf32" in g:
        return {"matmul_tf32": bool(g.get("cuda.matmul.allow_tf32")), "matmul": g.get("float32_matmul_precision"), "cudnn_tf32": g.get("cudnn.allow_tf32")}
    return None


def numerics_verdict(T: dict, levers: List[str]) -> Tuple[str, Optional[str]]:
    """(the line's precision word, None | the named reason the readback contradicts it): a line WITH L13 must read TF32 matmuls live
    (``matmul_tf32`` True, matmul ``high``); a line WITHOUT it — every exact line — must read them off (False, ``highest``). No readback in the
    record is a contradiction on either line."""
    want = "L13" in levers
    word = "tf32" if want else "fp32"
    live = numerics_live(T)
    if live is None:
        return word, "no numerics readback in the driver's record"
    if live["matmul_tf32"] is not want or (live["matmul"] not in (None, "high" if want else "highest")):
        return word, f"numerics readback matmul_tf32={live['matmul_tf32']} matmul={live['matmul']} on a {word} line"
    return word, None


def _judge_hoist(T: dict) -> bool:
    L = T.get("lever") or {}
    return bool(T.get("hoist")) and int(L.get("hoist_fills") or 0) >= 1 and int(L.get("hoist_anomalies") or 0) == 0


def _judge_batches(T: dict) -> bool:
    L = T.get("lever") or {}
    return bool(L.get("finished")) and L.get("batches") is not None and L.get("batches_done") == L.get("batches") and int(L.get("batches") or 0) >= 1 and T.get("stock_batch") is not None


def _judge_tf32(T: dict) -> bool:
    return numerics_verdict(T, ["L13"])[1] is None


JUDGES = {"hoist": _judge_hoist, "batches": _judge_batches, "tf32": _judge_tf32}   # registry evidence kind ("judged", name): the lever's fact read back from the driver's record — hoist: filled >= 1 batch with no anomaly; batches: every batch of the request sampled and written (finished); tf32: TF32 matmuls live by readback


def _at(T: dict, path) -> object:
    """T[k0][k1]… or None."""
    for k in path:
        if not isinstance(T, dict):
            return None
        T = T.get(k)
    return T


def read_evidence(log_path: str, timings_path: Optional[str], levers: List[str], log_offset: int = 0) -> dict:
    """Which levers left their own evidence (a driver log line, a timings JSON key, a judged predicate), which did not, which declined the
    request by name (registry.Lever.declined), and the forbidden lines found in the log."""
    try:
        with open(log_path, "rb") as fh:                                # THIS pass's lines only: the log is appended to across passes of one output directory (log_offset = its size before this pass)
            fh.seek(max(0, int(log_offset or 0)))
            text = fh.read().decode("utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        text = ""
    lines = text.splitlines()
    T = read_timings(timings_path) if timings_path else {}
    applied, missing, declined = [], [], {}
    for lid in levers:
        ev = LEVERS[lid].evidence
        if not ev:
            continue
        kind, arg = ev
        if kind == "log":
            ok = any(re.search(arg, ln) for ln in lines)
        elif kind == "timings":                                   # the key present with a value: a number (0 included), a non-empty list, or True — never False / None / empty
            v = T.get(arg)
            ok = v is not None and v is not False and (bool(v) or (isinstance(v, (int, float)) and v == 0))
        elif kind == "judged":
            ok = bool(T) and JUDGES[arg](T)
        else:
            ok = False
        dc = LEVERS[lid].declined                                    # a planned lever that could not serve THIS request said so by name twice — its line in this pass's log AND the reason in this pass's timings record: declined (by that reason), not missing; one without the other stays missing
        if not ok and dc and dc[0] == "log+timings":
            why = _at(T, dc[2])
            if why and any(re.search(dc[1], ln) for ln in lines):
                declined[lid] = str(why)
                continue
        (applied if ok else missing).append(lid)
    forbidden = [ln for ln in lines if any(re.search(f, ln) for f in FORBIDDEN_LINES)]
    return {"applied": applied, "missing": missing, "declined": declined, "forbidden": forbidden[:20], "n_lines": len(lines), "timings_present": bool(T)}


def timings_figures(T: dict) -> dict:
    """The driver's own figures of a pass, as recorded in the manifest (the per-batch table stays in the timings JSON)."""
    out = {k: T.get(k) for k in ("argv", "gpu", "torch", "seed", "patches", "matmul_precision", "model_setup_s", "featurize_s", "n_designs", "shard", "n_designs_this_shard",
                                 "sampling_loop_s", "s_per_design_sampling", "write_s", "total_s", "s_per_design_total_inprocess", "D59_global_state_checks_passed",
                                 "rng_final_state_matches_stock_sampler", "sampler") if k in T}
    if isinstance(T.get("graph_capture_s"), list):
        out["graph_capture_s"] = {"n": len(T["graph_capture_s"]), "sum": round(sum(T["graph_capture_s"]), 3), "first": T["graph_capture_s"][0] if T["graph_capture_s"] else None}
    if isinstance(T.get("per_batch"), list):
        out["batches"] = {"n": len(T["per_batch"]), "sizes": [len(b.get("names") or []) for b in T["per_batch"]], "wall_s": [b.get("wall_s") for b in T["per_batch"]]}
    return out


# ------------------------------------------------------------------------------------------------------------------------ runs
def _run(cmd: List[str], env: Dict[str, str], cwd: str, log_path: str) -> Tuple[int, float, float]:
    """Run one child process with its output appended to log_path. Returns (rc, wall seconds, the start time)."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    t0 = time.time()
    with open(log_path, "ab") as log:
        log.write(("$ " + " ".join(cmd) + "\n").encode())
        log.flush()
        p = subprocess.run(cmd, env=env, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0, t0


def first_design(out_dir: str, problems: List[str], t0: float) -> Optional[Tuple[float, str]]:
    """The first PDB this process wrote: (seconds from the process start t0 to the earliest <problem>/pdbs/*.pdb mtime at or after t0, that file
    relative to out_dir) — the ready line's value. None when the pass wrote no design (files older than t0 are an earlier pass's)."""
    best = None
    dirs = [_outputs.pdb_dir(out_dir, p) for p in (problems or _outputs.problem_dirs(out_dir))]
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith(".pdb"):
                m = os.path.getmtime(os.path.join(d, f))
                if m + 1.0 < t0:
                    continue
                if best is None or m < best[0]:
                    best = (m, os.path.relpath(os.path.join(d, f), out_dir))
    if best is None:
        return None
    return max(0.0, best[0] - t0), best[1]


def run(request_path: str, out_dir: Optional[str], mode: Optional[str], *, tag: str = "run", seed: Optional[int] = None, n_sample: Optional[int] = None,
        selections: Optional[str] = None, batch: Optional[int] = None, python: Optional[str] = None,
        det_level: int = det.DEFAULT_LEVEL, verbose: bool = False, log_dir: Optional[str] = None, num_devices: Optional[int] = None,
        shard_id: Optional[int] = None, num_shards: Optional[int] = None) -> Tuple[int, dict]:
    """The design pass. Returns (exit code, the manifest dict). Refused by name before anything runs (exit 3): a deployment fact, or on a kit mode a
    request naming a computation the kit line does not run (kit_refusals); a planned lever that could not run makes the mode refuse by name after
    the pass (exit 3: a mode is all of its levers); a short output set exits 1 (``incomplete``). ``--num-shards M --shard-id K`` runs the shard's
    share on every route. ``out_dir`` None = the request's own output directory (generation.io.outdir / paths.rootdir). ``batch`` = `design
    --batch_size B`: generation.dataset.batch_size for this pass on every route (None: the request's own key, else upstream's default 1 on every
    route — modes.with_batch); a batch below 1 is refused by name before anything runs."""
    try:                                                                          # the mode word first (--mode, else GENIE3_OPT, else the default): an unknown mode is refused on every route, before anything runs
        res0 = resolve(stack.mode_from_env(mode))
    except ModeError as e:
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {e}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": str(e)}
    if batch is not None and int(batch) < 1:                                     # --batch_size: upstream's loader and the kit line both take an integer >= 1
        why = f"--batch_size {batch}: the batch size is an integer >= 1 (generation.dataset.batch_size)"
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {why}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": why}
    root_env = stack.genie3_root() or stack.genie3_root_env()
    try:
        raw = load_request(request_path)
    except DesignError as e:
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {e}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": str(e)}
    m = res0.mode
    if res0.attach == "driver":
        refusals = kit_refusals(raw, num_devices=num_devices)
        if refusals:                                                              # the request names a computation the kit line does not run: refused by name before anything runs (--mode off runs it)
            _report.emit(_report.refused_line(m, refusals))
            return EXIT_NOT_ACTIVE, {"status": "refused", "mode": m, "reason": f"mode={m} cannot serve " + "; ".join(refusals) + " — run --mode off for the stock path",
                                     "refused_features": [r.split(":", 1)[0] for r in refusals]}
    sharded = num_shards is not None and int(num_shards) > 1
    if sharded and not 0 <= int(shard_id or 0) < int(num_shards):               # upstream: 0 <= shard_id < num_shards (loader.py to_generation_config)
        why = f"--shard-id {shard_id} --num-shards {num_shards}: the shard index is 0 <= K < M"
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {why}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": why}
    stock_words = stock_flag_words(verbose, log_dir, num_devices, shard_id, num_shards)
    b_pass = int(batch) if batch is not None else request_batch(raw)              # this pass's batch: --batch_size, else the request's own key (None = the route's default)
    rep = stack.activate(m, python=python, batch=b_pass if m != "off" else None, weights_paths=weights_paths(raw, root_env))   # the report (ACTIVE line) spells the effective line: this pass's batch size and TriangleMultiplication provider
    if rep.get("reason") or rep.get("would_refuse") or not rep.get("active"):
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": rep.get("reason") or "; ".join(rep.get("would_refuse") or []), "activation": rep}
    root, weights = rep.get("genie3_root"), rep.get("weights")
    res = resolve(m) if m == "off" else effective(m, b_pass)
    try:
        req, composed = compose_request(raw, out_dir, weights, seed, n_sample, selections, det_level=det_level, batch=batch)
        problems = request_problems(req, root)
    except DesignError as e:
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {e}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": str(e), "activation": rep}
    out = os.path.abspath(out_dir) if out_dir is not None else request_out_dir(req, root)
    if out is None:
        why = "the request names no output directory (paths.rootdir / generation.io.outdir) and no --out_dir was given"
        _report.emit(f"{_report.PREFIX} NOT ACTIVE: {why}")
        return EXIT_NOT_ACTIVE, {"status": "refused", "reason": why, "activation": rep}
    rec = request_record(req, composed, problems, request_path, shard_id=shard_id if sharded else None, num_shards=num_shards if sharded else None)
    seed_for_record = req["experiment"].get("seed")
    man = _manifest.start(rep, res, rec, out, tag, seed=int(seed_for_record) if seed_for_record is not None else None, det_level=det_level)
    man["batch_size"] = batch_size(res) if res.attach == "driver" else rec.get("batch_size")   # the kit line's --batch-size | the stock request's own key (None = upstream's default, 1)
    man["precision"] = precision_of(res)                                        # fp32 | tf32: the line as run
    man["trimul"] = trimul_of(res)                                              # fpf (lever L7) | stock: the TriangleMultiplication provider the line runs
    man["stock_flags"] = stock_words
    man["shard"] = rec.get("shard")                                             # "K/M" under --num-shards M --shard-id K (the pass writes the shard's share of every problem's designs), else None
    n_designs = rec["designs_expected"]
    share = (rec["design_indices"][1] - rec["design_indices"][0]) if rec.get("design_indices") else None   # designs per problem this pass writes (the shard's slice of n_sample, or all of it); None = n_sample is not an integer in the request
    t_pass = time.time()
    if sharded and share == 0:                                                    # a shard with no share of the designs: nothing to sample — upstream writes the shard's marker and returns 0 (workflow.py), on every route, whatever the problems
        _report.emit(_report.note_line(f"shard {man['shard']} has no share of the request's designs (upstream's ceil slicing of n_sample {rec['n_sample']} over {int(num_shards)} shards leaves shard {int(shard_id or 0)} none): nothing runs; the shard marker is written"))
        man["outputs"] = _outputs.list_outputs(out, problems, since=t_pass, names=rec.get("names_expected"))
        man["shard_markers"] = shard_marker(out, int(shard_id or 0), int(num_shards), problems)
        man["status"] = "ok"
        _manifest.finish(man, out)
        return EXIT_OK, man
    done = shard_done(out, int(shard_id or 0), int(num_shards), problems) if sharded else None
    if done:                                                                      # upstream's resume rule on every route: a shard whose marker exists is already complete — nothing is generated, exit 0 (workflow.py is_generation_shard_done)
        _report.emit(_report.note_line(f"shard {man['shard']} already complete (marker {done[0]}{' +%d more' % (len(done) - 1) if len(done) > 1 else ''}): nothing runs, as upstream's generate skips a done shard; remove the marker to re-run it"))
        man["outputs"] = _outputs.list_outputs(out, problems, names=rec.get("names_expected"))   # the done shard's designs as they stand in the directory (written by the pass that completed it)
        man["shard_markers"], man["skipped"] = done, "shard already complete (marker)"
        man["status"] = "skipped"
        _manifest.finish(man, out)
        return EXIT_OK, man
    if n_designs is None:                                                         # the pass's design count cannot be stated up front: named, never silent — the census below then reports what was written without a completeness verdict
        why = (f"n_sample {rec['n_sample']!r} is not an integer" if share is None else
               f"the request selects no problems and its dataset's problem list is not readable under the checkout (paths.dataset={req['paths'].get('dataset')!r})" if not problems else
               "an unconditional request without integer min_length / max_length / length_step")
        man["designs_expected_unknown"] = why
        _report.emit(_report.note_line(f"designs_expected=unknown ({why}): the outputs are counted after the pass without a completeness verdict"))
    request_copy = write_request(req, out)
    timings = os.path.join(out, TIMINGS_NAME)
    _report.emit(_report.run_line(res.mode, len(problems), n_designs if n_designs is not None else "unknown", seed_for_record if seed_for_record is not None else "unseeded", out,
                                  batch=man["batch_size"] if man["batch_size"] is not None else "request", precision=man["precision"], shard=man["shard"]))
    rc_all, status = EXIT_OK, "ok"
    if res.attach == "stock-cli":
        env, stripped, absent = stock_environment()
        write_pins_check(rep, out)
        cmd = stock_command(request_copy, out, root, python, stock_words)
        log_path = os.path.join(out, STOCK_LOG)
        census = kernels_census(env)                                             # the child's matmul numerics as its environment leaves them, printed before the child runs (no probe process)
        _report.emit(_report.kernels_line(census))
        man["stock"] = {"env_stripped": stripped, "must_be_absent": absent, "kernels": census, "cwd": root, "log": log_path, "flags": stock_words}
        rc, wall, t0 = _run(cmd, env, root, log_path)
        proof_p = os.path.join(out, PROOF_NAME)
        proof = json.load(open(proof_p)) if os.path.isfile(proof_p) else {"ok": False, "error": "no proof written"}
        _report.emit(_report.stock_line(proof))
        ready = first_design(out, problems, t0)
        if ready is not None:
            _report.emit(_report.ready_line(res.mode, ready[0], tag, ready[1]))
        man["stock"].update({"rc": rc, "wall_s": round(wall, 2), "cmd": cmd, "proof": proof,
                             "ready_s": round(ready[0], 2) if ready else None, "first_design": ready[1] if ready else None})
        if not proof.get("ok"):
            status, rc_all = "failed", EXIT_NOT_ACTIVE
        elif rc != 0:
            status, rc_all = "failed", EXIT_FAIL
    else:
        if log_dir is not None:                                                  # upstream's run-log tree is the stock CLI's; the kit line's record is opt_manifest.json + timings.json
            _report.emit(_report.note_line(f"--log-dir {log_dir}: the kit line writes no upstream run-log tree; its record is {out}/{_manifest.NAME} and {TIMINGS_NAME}"))
        env, dropped = stack.driver_environment()
        man["env_dropped"] = dropped
        cmd = stack.driver_command(res, request_copy, out, timings, python, verbose=verbose, shard=(int(shard_id or 0), int(num_shards)) if sharded else None)
        log_path = os.path.join(out, DRIVER_LOG)
        if os.path.isfile(timings):                                              # an earlier pass's record in a reused directory is not this pass's evidence
            os.remove(timings)
        log_offset = os.path.getsize(log_path) if os.path.isfile(log_path) else 0   # nor are its log lines: the evidence read-back starts at this pass's first byte of the appended log
        rc, wall, t0 = _run(cmd, env, root, log_path)
        ev = read_evidence(log_path, timings, list(res.levers), log_offset=log_offset)
        _report.emit(_report.evidence_line(ev["applied"], ev["missing"], ev["forbidden"]))
        T = read_timings(timings)
        for lid, why in ev["declined"].items():                                  # a planned lever that stood down for this request by its declared gate, by name and reason (L7 under the kernel's token floor): neither partial nor refused
            _report.emit(_report.lever_skipped_line(res.mode, lid, why))
        stack.record_pass(ev)                                                    # the report's levers_applied / levers_unavailable / levers_declined / partial
        word, why_num = numerics_verdict(T, list(res.levers)) if rc == 0 else (precision_of(res), None)
        _report.emit(_report.numerics_line(res.mode, word, numerics_live(T), why_num, batch=T.get("stock_batch"), planned=man["batch_size"]))
        ready = first_design(out, problems, t0)
        if ready is not None:
            _report.emit(_report.ready_line(res.mode, ready[0], tag, ready[1]))
        _report.add_tally_source(tag, timings)
        man["driver_pass"] = {"tag": tag, "rc": rc, "wall_s": round(wall, 2), "cmd": cmd, "cwd": root, "log": log_path, "timings_json": timings if T else None,
                              "timings": timings_figures(T), "evidence": ev, "ready_s": round(ready[0], 2) if ready else None, "first_design": ready[1] if ready else None}
        man["levers_missing"] = sorted(ev["missing"])                            # planned levers that left no evidence this pass (as manifest.finish records them): they could not run here, and a mode is all of its levers —
        man["levers_broken"] = [lid for lid in ev["missing"] if (LEVERS[lid].evidence or ("",))[0] == "judged"]   #  the mode refuses by name below; except a lever that RAN and whose record contradicts its contract
        man["forbidden_lines"] = len(ev["forbidden"])                            #  (registry evidence kind `judged`: the hoist's anomaly count, the batch table): that FAILS the pass, like a registry.FORBIDDEN_LINES match (the count)
        man["levers_declined"] = list(ev["declined"])                            # ids, registry order (the reasons: driver_pass.evidence.declined)
        man["driver_pass"]["numerics"] = {"line": word, "live": numerics_live(T), "verdict": "ok" if why_num is None else why_num}
        cap = (T.get("lever") or {}).get("capacity_refused")
        if cap:                                                                  # the driver refused a batch by name before running it (its memory model vs the device): the pass cannot activate at this batch size on this card
            advice = (f"set generation.dataset.batch_size <= {cap.get('suggest_batch')} in the request (or run --mode off)" if cap.get("suggest_batch")
                      else "no batch size fits this device for this request on the kit line (run --mode off)")
            man["capacity_note"] = f"batch {cap.get('batch')} at {cap.get('n_token')} tokens needs ≈{cap.get('need_gb')} GB: refused before running — {advice}"
            _report.emit(f"{_report.PREFIX} CAPACITY mode={res.mode} batch={cap.get('batch')} n_token={cap.get('n_token')} need_gb={cap.get('need_gb')} verdict=refused suggest_batch={cap.get('suggest_batch')}")
            _report.emit(f"{_report.PREFIX} NOT ACTIVE: capacity — {man['capacity_note']}")
        if cap:
            status, rc_all = "refused", EXIT_NOT_ACTIVE
        elif rc != 0:
            status, rc_all = "failed", EXIT_FAIL
        elif why_num is not None:                                                # the readback contradicts the line (TF32 live on an fp32 line, or off on the tf32 line, or absent): the pass's claim is void
            status, rc_all = "failed", EXIT_FAIL
            man["numerics_note"] = why_num
        elif T.get("stock_batch") is not None and man["batch_size"] is not None and int(T["stock_batch"]) != int(man["batch_size"]):
            status, rc_all = "failed", EXIT_FAIL
            man["numerics_note"] = f"the driver ran batch {T['stock_batch']}, the line asked {man['batch_size']}"
        elif man["forbidden_lines"]:                                             # a traceback / assertion / divergence marker in the driver's log: the outputs are not trusted, whatever else was printed
            status, rc_all = "failed", EXIT_FAIL
            _report.emit(_report.forbidden_line(len(ev["forbidden"])))
        elif man["levers_broken"]:                                               # a lever ran and its own record contradicts its contract (a judged predicate: registry.Lever.evidence): the outputs are not trusted
            status, rc_all = "failed", EXIT_FAIL
            _report.emit(_report.broken_line(",".join(man["levers_broken"])))
        elif man["levers_missing"]:                                              # a planned lever could not run on this box: the MODE refuses by name — a mode is all of its levers, never a subset under its name
            status, rc_all = "refused", EXIT_NOT_ACTIVE
            man["refused"] = "levers could not run: " + ",".join(ev["missing"])
            _report.emit(_report.levers_refused_line(res.mode, ",".join(ev["missing"])))   # registry order, as the EVIDENCE line lists them
    man["outputs"] = _outputs.list_outputs(out, problems, since=t_pass, names=rec.get("names_expected"))
    if n_designs is None:
        _report.emit(_report.note_line(f"designs written={man['outputs']['n_pdb']} (designs_expected=unknown: {man['designs_expected_unknown']})"))
    if n_designs is not None and man["outputs"]["n_pdb"] != n_designs:
        share = f"n_sample {rec['n_sample']}" if not sharded else f"designs {rec['design_indices'][0]}..{rec['design_indices'][1] - 1} of n_sample {rec['n_sample']}, shard {man['shard']}"
        man["incomplete"] = f"{man['outputs']['n_pdb']}/{n_designs}"
        man["outputs_note"] = f"expected {n_designs} PDB files ({len(problems)} problems x {share}) written by this pass, found {man['outputs']['n_pdb']}"
        if status == "ok":
            status, rc_all = "incomplete", EXIT_FAIL
            _report.emit(f"{_report.PREFIX} incomplete: {man['incomplete']} PDB files for the request ({man['outputs_note']})")
    if sharded and status == "ok" and res.attach == "driver":                   # the shard's share is complete: upstream's shard marker, where its evaluate / status read it (the stock child writes its own)
        man["shard_markers"] = shard_marker(out, int(shard_id or 0), int(num_shards), problems)
    man["status"] = status
    _manifest.finish(man, out)
    return rc_all, man
