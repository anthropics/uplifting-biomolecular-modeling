"""Where the kits are, whether this box can run a mode, and how the mode's worker is launched — the package's one place for all three.

Kit locations (opt/-relative, ``modes.py``): the package finds ``boltz2/opt/`` from its own location or from ``$MODEL_OPT`` (the ``boltz2/``
directory, exported by ``configs/*.env`` and ``run.sh``). Every carried kit file's presence is checked live before a route stages it (a
missing file refuses the launch by name — the version is the git commit, never a stored manifest).

The worker route (``modes.PINNED_ROUTE``): the mode's kit files are staged flat into a working directory, the worker variant is derived by the sampler add-on's own
``make_worker_variant.py`` from the mode's base worker (never stored: derived per run, sha printed by the kit script), the batch json the
kit worker takes (bz_worker_lev.py:29) is written from the caller's YAMLs, and ``python bz_worker.py --batch ... --mode fast --kernels off
--num_workers 1 --pipeline 1 --keep_on_gpu 1`` (the kits' probe arm's arguments, modes.WORKER_ARGS, with ``--kernels`` from the mode
row: ``on`` for exact — modes.worker_args) runs under the mode's activation row.
The child environment carries the row and nothing else of the kits' switch families (``stock/PINS.json`` stock_environment prefixes are
stripped first, so a stray switch in the caller's shell cannot change the line), and never BOLTZ2_OPT (the kit worker applies the levers
itself). After the worker exits its own log (``<kit_dir>/<tag>_worker_log.json``) is the evidence: the trunk levers it reports applied, the graph
sampler mode and replay count and the hoist level per item; under ``fast`` also the flash patch's own per-call counters, printed as one
JSON line at the worker's exit (``modes.EVIDENCE_ENV``). Missing evidence is a refusal, never a silent stock run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from opt_core import gates as core_gates, home as core_home, instances as core_instances

from . import digest_memo, modes, registry, report as rep, tp

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
PYPROJECT = os.path.join(os.path.dirname(PKG_DIR), "pyproject.toml")   # the package's own pyproject: the opt_core pin ([tool.opt_core])


# ---------------------------------------------------------------- locations (opt_core.home) ----------------------------------------------------------------
def opt_dir() -> str:
    """boltz2/opt/: $MODEL_OPT/opt when MODEL_OPT is set (the boltz2/ directory), else the directory this package lives in."""
    return core_home.opt_home(__file__)


def tree_dir() -> str:
    return core_home.tree_home(__file__)


def kit_path(rel: str) -> str:
    return core_home.kit_path(opt_dir(), rel)


def pins_path() -> str:
    return os.path.join(tree_dir(), "stock", "PINS.json")


def load_pins() -> dict:
    return json.load(open(pins_path()))


# ---------------------------------------------------------------- kit files (a live directory walk, the sha rule) ----------------------------------------------------------------
sha256_file = core_gates.sha256_file

# The core no longer carries a tree-walking utility (its own sha-manifest-generation toolkit -- tree_files,
# tree_lines, tree_sha, the IGNORED_DIRS/IGNORED_FILES rule -- was removed as a unit once nothing in the core
# itself needed to generate a manifest); this kit's own live-hash use (never compared to a stored value) has
# no remaining core call to make, so it carries the walk itself, same ignore rule as the core's former one.
_IGNORED_DIRS = ("__pycache__", "*.egg-info", ".venv", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", ".idea", ".vscode", "do_not_commit", ".git")
_IGNORED_FILES = ("*.py[cod]", ".python-version", ".env", ".env.*", "*.pem", "*.key", ".DS_Store")


def _tree_files(root: str) -> List[str]:
    """The relative paths of the tree's files under the ignore rule above, sorted."""
    import fnmatch
    out = []
    for d, dns, fns in os.walk(root):
        dns[:] = sorted(x for x in dns if not any(fnmatch.fnmatchcase(x, p) for p in _IGNORED_DIRS + _IGNORED_FILES))
        out += [os.path.relpath(os.path.join(d, f), root) for f in fns if not any(fnmatch.fnmatchcase(f, p) for p in _IGNORED_FILES)]
    return sorted(out)


def read_sums() -> Dict[str, str]:
    """Every carried file under opt/forward/, hashed live: {opt-relative path: sha256}. The version is the git commit, not a stored
    manifest -- this walks the tree fresh (the ignore rule above) and hashes each file; nothing here is compared to a recorded value."""
    root = os.path.join(opt_dir(), "forward")
    return {f"forward/{rel}": sha256_file(os.path.join(root, rel)) for rel in _tree_files(root)}


def check_kit_files(rels: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, str]]:
    """Check the named opt-relative kit files are present (default: every carried file, read_sums()). Returns ``(problems, sha_by_rel)``:
    a missing file is a problem, by name; the sha of a present file is this run's own evidence (the run record's staged_kit_files),
    never a byte-equality check against a stored value."""
    rels = list(rels) if rels is not None else sorted(read_sums())
    problems, got = [], {}
    for rel in rels:
        p = os.path.join(opt_dir(), rel)
        if os.path.isfile(p):
            got[rel] = sha256_file(p)
        else:
            problems.append(f"{rel}: missing")
    return problems, got


# ---------------------------------------------------------------- the box ----------------------------------------------------------------
def gpu_probe() -> Optional[dict]:
    """The first NVIDIA GPU by nvidia-smi (no torch import): {name, memory_mib, driver, compute_capability}; None when there is none.
    ``compute_capability`` is 'M.m' (nvidia-smi's compute_cap field; None on a driver too old to report it)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20)
        if out.returncode != 0:                                             # a driver without the compute_cap field: the three classic fields
            out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    parts = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
    d = {"name": parts[0]}
    try:
        d["memory_mib"] = int(float(parts[1]))
    except (IndexError, ValueError):
        d["memory_mib"] = None
    d["driver"] = parts[2] if len(parts) > 2 else None
    d["compute_capability"] = parts[3] if len(parts) > 3 and re.fullmatch(r"\d+\.\d+", parts[3] or "") else None
    return d


def visible_gpus() -> Optional[int]:
    """How many NVIDIA GPUs this process may use (no torch import): nvidia-smi's device list narrowed by CUDA_VISIBLE_DEVICES when it is set
    (a comma list of indices or UUIDs; an empty value hides all). None when nvidia-smi is absent or fails — the n_gpu gate then refuses P > 1
    by name (tp.check: 'the visible GPU count was not taken')."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    devs = [l for l in out.stdout.splitlines() if l.strip()]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return len(devs)
    want = [x.strip() for x in cvd.split(",") if x.strip()]
    return min(len(want), len(devs))


# The cards a configs/<card>.env names in MODEL_OPT_TARGET_GPU: compute capability (what the kit's cells are keyed and measured by) and the
# memory totals nvidia-smi reports for the class's members (MiB; the number that tells an H100 80GB HBM3, 81559, from an H100 NVL, 95830 —
# and an A100 80GB, 81920, from an A100 40GB, 40960). A name or memory mismatch with the target is a NOTE on `check` and a run-record fact
# (gpu_supported), never a gate; the one gate on the card is compute capability (MIN_COMPUTE_CAPABILITY, capability_reasons).
GPU_CLASSES = {"H100": {"compute_capability": "9.0", "memory_mib": 81559, "memory_mib_members": (81559,)},
               "A100": {"compute_capability": "8.0", "memory_mib": 81920, "memory_mib_members": (81920, 40960)}}
GPU_MEMORY_TOLERANCE_MIB = 1024
MIN_COMPUTE_CAPABILITY = (8, 0)      # Boltz-2's own kernels-on route needs compute capability >= 8 (boltz2.py: use_kernels) and bf16 tensor cores; every Triton cell
                                     # of the kit is compiled for sm_80 or newer — below it the kit modes are refused by name (the stock route says so itself)


def parse_cc(text) -> Optional[Tuple[int, int]]:
    """'8.0' -> (8, 0); None when absent or malformed."""
    m = re.fullmatch(r"\s*(\d+)\.(\d+)\s*", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def card_class_of(gpu: Optional[dict]) -> Optional[str]:
    """The GPU_CLASSES name whose compute capability is the probed card's (the name is a label; capability decides), else None."""
    cc = (gpu or {}).get("compute_capability")
    for name, cls in GPU_CLASSES.items():
        if cc and cls["compute_capability"] == cc and name.lower() in ((gpu or {}).get("name") or "").lower():
            return name
    for name, cls in GPU_CLASSES.items():
        if cc and cls["compute_capability"] == cc:
            return name
    return None


def capability_reasons(gpu: Optional[dict]) -> List[str]:
    """The card gate, by capability only: a reason BY NAME when the probed compute capability is below MIN_COMPUTE_CAPABILITY; nothing for an
    unknown capability (a driver that does not report it) or any card at or above it — listed in GPU_CLASSES or not (an unlisted card is a NOTE)."""
    cc = parse_cc((gpu or {}).get("compute_capability"))
    if cc is not None and cc < MIN_COMPUTE_CAPABILITY:
        return [f"GPU {gpu.get('name')} has compute capability {cc[0]}.{cc[1]}, below {MIN_COMPUTE_CAPABILITY[0]}.{MIN_COMPUTE_CAPABILITY[1]}: "
                "the kit's Triton cells and Boltz-2's kernels-on route need sm_80 or newer (Ampere+)"]
    return []


def gpu_class_check(gpu: Optional[dict], target: Optional[str]) -> dict:
    """Compare the probed GPU with the class `target` names (MODEL_OPT_TARGET_GPU): the compute capability must be the class's; the name
    should contain the class and the memory total be one of the class's members within GPU_MEMORY_TOLERANCE_MIB. Returns ``{target,
    class_memory_mib, class_compute_capability, name, memory_mib, compute_capability, card_class, name_match, memory_match,
    capability_match, supported}`` — ``supported`` is None when there is no target, no GPU or no class row; it is the capability match when
    the card reports its capability (name / memory differences are notes), else name and memory as before."""
    gpu = gpu or {}
    out = {"target": target or None, "class_memory_mib": None, "class_compute_capability": None, "name": gpu.get("name"), "memory_mib": gpu.get("memory_mib"),
           "compute_capability": gpu.get("compute_capability"), "card_class": card_class_of(gpu) if gpu else None,
           "name_match": None, "memory_match": None, "capability_match": None, "supported": None}
    if not target or not gpu:
        return out
    cls = GPU_CLASSES.get(target.upper())
    out["name_match"] = target.lower() in (gpu.get("name") or "").lower()
    if cls is None:
        return out
    out["class_memory_mib"] = cls["memory_mib"]; out["class_compute_capability"] = cls["compute_capability"]
    mem = gpu.get("memory_mib")
    out["memory_match"] = mem is not None and any(abs(int(mem) - m) <= GPU_MEMORY_TOLERANCE_MIB for m in cls["memory_mib_members"])
    if gpu.get("compute_capability"):
        out["capability_match"] = gpu["compute_capability"] == cls["compute_capability"]
        out["supported"] = bool(out["capability_match"])
    else:
        out["supported"] = bool(out["name_match"] and out["memory_match"])
    return out


def gpu_class_note(chk: dict) -> Optional[str]:
    """One line naming what differs from the target class (None when nothing differs or nothing can be compared): a capability other than
    the class's, a name without the class label, a memory total that is not a member of the class — notes, never gates."""
    if not chk.get("target") or chk.get("name") is None:
        return None
    parts = []
    if chk.get("capability_match") is False:
        parts.append(f"the GPU {chk.get('name')} has compute capability {chk.get('compute_capability')}, not the class's {chk.get('class_compute_capability')}")
    elif not chk.get("name_match"):
        parts.append(f"the GPU is {chk.get('name')}")
    if chk.get("memory_match") is False:
        parts.append(f"memory {chk.get('memory_mib')} MiB is not the class's {chk.get('class_memory_mib')} MiB")
    if not parts:
        return None
    return f"MODEL_OPT_TARGET_GPU={chk.get('target')} but " + " and ".join(parts) + ": not the GPU this configuration targets"


def pins_check(ckpt: bool = False) -> Tuple[List[str], dict]:
    """stock/check_pins.py's check() in this interpreter (the one pin check; standard library only)."""
    p = os.path.join(tree_dir(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("boltz2_stock_check_pins", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.check(mod.pins(), ckpt=ckpt)


def cache_dir() -> Optional[str]:
    c = os.environ.get("BOLTZ_CACHE")
    return os.path.abspath(os.path.expanduser(c)) if c else None


# The frozen cache: every file the kits' worker reads (boltz2_conf.ckpt, ccd.pkl, mols/; bz_worker_lev.py:90) and every file upstream's
# CLI DOWNLOADS when absent (stock/src/boltz/main.py download_boltz2 :198-256 — mols.tar :213-216, mols/ :220-227, boltz2_conf.ckpt :229-235,
# boltz2_aff.ckpt :246-252 — called from predict :1141; the cache is $BOLTZ_CACHE or --cache, :262-278; upstream has no disable switch).
# A frozen-weights run never downloads: an absent file is a refusal by name, before any launch on any route.
CACHE_FILES = ("boltz2_conf.ckpt", "boltz2_aff.ckpt", "ccd.pkl", "mols", "mols.tar")


WEIGHTS_FILE = "boltz2_conf.ckpt"            # the structure checkpoint: the one weights file with a pinned digest (stock/PINS.json "weights")


WEIGHTS_DIGEST_DIR_ENV = "MODEL_OPT_WEIGHTS_DIGEST_DIR"   # the deployment's override of the weights digest memo directory (read first; outside the policed BOLTZ2_OPT prefix)


def weights_memo_dir(cache: Optional[str] = None) -> Optional[str]:
    """The directory of the weights digest memo (``digest_memo.MEMO_NAME`` = weights_digests.json): ``MODEL_OPT_WEIGHTS_DIGEST_DIR`` when set
    (the deployment's writable directory of choice), else the kit's writable cache root — the parent of ``TRITON_CACHE_DIR`` when set
    (configs/h100.env keys it as <MODEL_OPT_JIT_ROOT>/<MODEL_OPT_STACK_KEY>/triton when MODEL_OPT_JIT_ROOT is set, so the memo lands in the
    stack-key directory; a Boltz-2 cache mounted read-only stays untouched), else the Boltz-2 cache directory (``BOLTZ_CACHE``)."""
    o = os.environ.get(WEIGHTS_DIGEST_DIR_ENV)
    if o:
        return os.path.abspath(os.path.expanduser(o))
    t = os.environ.get("TRITON_CACHE_DIR")
    if t:
        return os.path.dirname(os.path.abspath(os.path.expanduser(t)))
    return cache if cache is not None else cache_dir()


def weights_status(cache: Optional[str] = None, pinned: Optional[str] = None, refresh: bool = False, memo_dir: Optional[str] = None) -> dict:
    """The checkpoint's digest status against the pin: sha256 of ``WEIGHTS_FILE`` under the cache vs stock/PINS.json ``weights.files`` ->
    ``status`` pinned | unknown | absent. The status is decided by the digest only. The digest comes through the on-disk memo
    (``digest_memo.digest``: an entry keyed by the file's realpath / size / mtime / inode — those select the entry, they never decide the status —
    written only after a full sha256); ``refresh=True`` (the ``check`` verb) hashes afresh and rewrites the entry; a memo directory that cannot
    be written (read-only mount) is named on the line and the digest computed afresh stands (``memo_note``). A status note printed on
    the WEIGHTS line and recorded in the plan — never a refusal: a checkpoint that is not the pinned one is named and the run proceeds."""
    cache = cache if cache is not None else cache_dir()
    if pinned is None:
        pinned = ((load_pins().get("weights") or {}).get("files") or {}).get(WEIGHTS_FILE, {}).get("sha256")
    path = os.path.join(cache or "", WEIGHTS_FILE)
    if not cache or not os.path.isfile(path):
        return {"file": path, "status": "absent", "sha256": None, "pinned_sha256": pinned, "cached_utc": None, "memo": None}
    md = memo_dir if memo_dir is not None else weights_memo_dir(cache)
    computed = {}

    def _hasher(p):                                                          # the reference hasher, its result kept so an unwritable memo never costs a second pass
        computed["sha256"] = digest_memo.sha256_file(p); return computed["sha256"]
    memo_note = None
    try:
        digest, cached_utc = digest_memo.digest(path, md, refresh=refresh, hasher=_hasher)
    except OSError as e:                                                     # the memo could not be written (a read-only or absent memo directory): the digest computed afresh
        if "sha256" not in computed:                                         # stands, named on the WEIGHTS line — never a refusal; an error before the hash (the checkpoint itself) propagates
            raise
        digest, cached_utc = computed["sha256"], None
        memo_note = f"digest memo not written ({os.path.join(md or '', digest_memo.MEMO_NAME)}: {e.strerror or type(e).__name__}); hashed afresh"
    return {"file": path, "status": "pinned" if pinned and digest == pinned else "unknown", "sha256": digest, "pinned_sha256": pinned,
            "cached_utc": cached_utc, "memo": os.path.join(md or "", digest_memo.MEMO_NAME), "refreshed": bool(refresh), "memo_note": memo_note}


def weights_line(w: dict) -> str:
    """``[boltz2-opt] WEIGHTS pinned[ (cached digest <utc>)] sha256=… file=…`` | ``… WEIGHTS unknown[ (cached digest <utc>)] sha256=… (not the
    pinned checkpoint …); WARNING: proceeding`` | ``… absent``."""
    word = digest_memo.word(w["status"], w.get("cached_utc"))
    tail = f"; {w['memo_note']}" if w.get("memo_note") else ""
    if w["status"] == "pinned":
        return f"[boltz2-opt] WEIGHTS {word} sha256={w['sha256'][:16]} file={w['file']} (the pinned checkpoint){tail}"
    if w["status"] == "unknown":
        return f"[boltz2-opt] WEIGHTS {word} sha256={w['sha256'][:16]} file={w['file']} (not the pinned checkpoint {str(w['pinned_sha256'])[:16]}); WARNING: proceeding — results are this checkpoint's, not the pin's{tail}"
    return f"[boltz2-opt] WEIGHTS absent file={w['file']}"


def cache_check() -> List[str]:
    """The frozen Boltz-2 cache: BOLTZ_CACHE set and holding every file of CACHE_FILES — the kits' inputs and upstream's download set."""
    c = cache_dir()
    if not c:
        return ["BOLTZ_CACHE is not set — see STOCK.md Variables (the Boltz-2 cache directory with " + ", ".join(CACHE_FILES) + "; upstream's CLI would download into ~/.boltz)"]
    missing = [f for f in CACHE_FILES if not os.path.exists(os.path.join(c, f))]
    return [f"BOLTZ_CACHE={c} lacks {', '.join(missing)} (upstream's CLI downloads what its cache lacks, main.py:198-256: refused)"] if missing else []


# ---------------------------------------------------------------- the gate ----------------------------------------------------------------
def attachment_problems(mode: str) -> List[str]:
    """The mode's attachments (modes.attachments → worker_launch.ATTACH): each adapter module must be importable by name and carry installed
    levers (its ``LEVERS``) before a worker is launched; an adapter without levers is a named reason (the memory modes refuse until the levers
    are in — the worker refuses the same way, `[boltz2-opt attach] REFUSED`)."""
    from .worker_launch import ATTACH
    out = []
    for name in modes.attachments(mode):
        if name not in ATTACH:
            out.append(f"attachment {name}: not an attachment of this tree"); continue
        module = ATTACH[name]["module"]; contract = "GUARDS" if ATTACH[name].get("kind") == "guard" else "LEVERS"
        try:
            spec = importlib.util.find_spec(module)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            out.append(f"attachment {name}: {module} is not importable (the mode refuses by name)"); continue
        mod = importlib.import_module(module)
        if not getattr(mod, contract, ()):
            out.append(f"attachment {name}: {module} has no {contract.lower()} installed yet (the mode refuses by name)")
        probe = getattr(mod, "problems", None)   # an adapter that states its own refusal words for this environment row before launch (boltz2_opt.msa_kernels.problems)
        env = dict(os.environ); env.update(modes.env_row(mode) if hasattr(modes, "env_row") else modes.resolve(mode)["env"])
        if callable(probe):
            out += [f"attachment {name}: {p}" for p in probe(env)]
        for sw, w in ROW_WORDS_REFUSED.get(name, {}).items():   # adapter words that are not levers of this kit's registry (development instruments, parked words kept out of every row): refused by name before launch
            got = [t.strip().lower() for t in (env.get(sw) or "").split(",") if t.strip()]
            bad = [t for t in got if t in w]
            if bad:
                out += [f"attachment {name}: {sw}={env.get(sw)!r}: {t} — {w[t]} (refused by name)" for t in bad]
    return out


# Words an adapter would accept that no row of this kit may carry (the registry names the levers; these are not levers here): refused by name at the
# launch gate. msa2's `hoist` overlaps the WASTE hoists (WASTE owns the exact MSA-module hoists) and `probe` is a development
# instrument; pairfuse's `fp32` residency is not released (refused by the adapter too); the DiT fused step's / graph_trunk's words are checked by their adapters.
DITEXACT_GLUE_PER_CALL = 120        # the fused elementwise glue replicas: 5 sites per layer x 24 layers per denoiser call (ditexact: glue_fused counts one per site)
DITEXACT_LAYERS = 24                # the diffusion token transformer's depth (boltz2 2.2.1 token_transformer: 24 DiffusionTransformerLayers): the DITEXACT fused passes count one per layer per denoiser call
MSA2_DECLARED_FALLBACKS = frozenset({"training", "autocast_off", "capture_unverified", "dims", "dtype", "device", "ln_affine", "batch", "shape", "tiny"})
MSA2_DECLARED_PREFIXES = ("bitcmp_failed:", "pin:", "small_class:", "undetermined:")   # a class below the unit's size floor (the stock statements serve, by name) / a selecting compare where several candidates matched (the stock output was returned, selection re-attempted)   # a shape class whose run-time bit comparison failed / a pin that does not hold: the stock forward serves it, named on the LEVER line (exact either way)   # boltz2_opt.msa2's by-name stock words that leave the run valid (the stock statements serve the call, counted): eval-only / autocast / a shape class first met under capture / an instance outside the kernel's form. Every other fallback word refuses the run (stack.evidence)
ROW_WORDS_REFUSED: Dict[str, Dict[str, Dict[str, str]]] = {
    "msa2": {"BOLTZ_MSA2": {"hoist": "not a lever of this kit (the WASTE words own the exact MSA-module hoists)", "probe": "a development instrument, never in a run of this kit"}},
    "pairfuse": {"BOLTZ_PAIRFUSE": {"fp32": "the fp32-resident driver is not a released variant (bf16 is the fast row's word)"}},
}


def gate(mode: str, need_gpu: bool = True, need_cache: bool = True, check_pins: bool = True, n_gpu=1, refresh_weights: bool = False) -> dict:
    """Resolve `mode` on this box and apply nothing. Returns the plan: ``{mode, tier, route, levers, env, worker_base, stage,
    kit_ok, gpu, n_gpu, visible, pins, cache, reasons}`` — ``reasons`` empty means the route can launch here. ``n_gpu``: the requested P
    (tp.check decides: P > 1 outside --mode big, a P outside tp.SUPPORTED_P, fewer than P visible GPUs are reasons BY NAME with the
    core's words; ``plan["n_gpu"]`` is the accepted P or None). ``refresh_weights``: the checkpoint digest is hashed afresh and its memo
    entry rewritten (the ``check`` verb); otherwise a memo hit serves it (pred / warm)."""
    gpu = gpu_probe()
    modes.set_card((gpu or {}).get("compute_capability") if need_gpu else None, (gpu or {}).get("memory_mib") if need_gpu else None)   # the card (+ its memory: the memory row's sampler ceiling), once: a lever the row does not run on this compute capability leaves it BY NAME (modes.CARD_DROPS)
    row = modes.resolve(mode)
    plan = {"mode": row["mode"], "tier": row["tier"], "route": row["route"], "levers": list(row["levers"]), "env": dict(row["env"]), "card_off": dict(row.get("card_off") or {}), "off": dict(row.get("off") or {}),
            "worker_base": row.get("worker_base"), "stage": modes.stage_files(row["mode"]),
            "cli_route": row["cli_route"], "reasons": [], "gpu": gpu, "n_gpu": None, "visible": None, "pins": None, "cache": None, "kit_ok": None}
    try:
        p_req = tp.parse(n_gpu)
    except ValueError as e:
        plan["reasons"].append(f"refused: {e}"); p_req = None
    except tp.Refused as e:                                        # the imported core does not carry the n_gpu words (reason=core_missing:…): NOT ACTIVE by name
        plan["reasons"].append(str(e)); p_req = None
    if row["mode"] == "off":
        if p_req is not None and p_req > 1:
            try:
                tp.check(p_req, "off")
            except tp.Refused as e:
                plan["reasons"].append(str(e))
        else:
            plan["n_gpu"] = p_req
        return plan
    # the core pin is gated at entry (_core_gate.gate: pinned == the importable opt_core, before any core import) — no second pin call here
    if p_req is not None:
        try:
            plan["visible"] = visible_gpus() if (need_gpu and p_req > 1) else None
            plan["n_gpu"] = tp.check(p_req, row["mode"], visible=plan["visible"])
        except tp.Refused as e:
            plan["reasons"].append(str(e))
    problems, _ = check_kit_files(plan["stage"])
    plan["kit_ok"] = not problems
    plan["reasons"] += [f"kit file {p}" for p in problems]
    plan["reasons"] += attachment_problems(mode)
    if check_pins:
        bad, detail = pins_check(); plan["pins"] = detail
        plan["reasons"] += [f"pin: {b}" for b in bad]
    if need_cache:
        c = cache_check(); plan["cache"] = cache_dir(); plan["reasons"] += c
        if not c:                                                            # the checkpoint's digest status: named on the WEIGHTS line (pinned | unknown), never a reason — an unknown checkpoint proceeds
            plan["weights"] = weights_status(plan["cache"], refresh=refresh_weights); print(weights_line(plan["weights"]), flush=True)
    if need_gpu and plan["gpu"] is None:
        plan["reasons"].append("no NVIDIA GPU (nvidia-smi found none): the kits' worker needs one CUDA GPU")
    if need_gpu:
        plan["reasons"] += capability_reasons(plan["gpu"])                  # the card gate: compute capability only (a name / memory-class mismatch is check's NOTE)
    return plan


# ---------------------------------------------------------------- late activation (opt_core.instances) ----------------------------------------------------------------
MODEL_CLASS = ("boltz.model.models.boltz2", "Boltz2")   # the upstream model class whose instances the late-activation rule counts


def arm_instance_counter() -> None:
    """Count Boltz2 instances constructed from now on (the core's constructor wrap on the model class, installed at its import)."""
    core_instances.register_instance_counter(*MODEL_CLASS)


def kit_lever_applied() -> List[str]:
    """Names of kit lever modules loaded in this process that report a lever applied."""
    out = []
    m = sys.modules.get("boltz_trunk_levers")
    if m is not None and getattr(m, "_STATE", {}).get("applied"):
        out.append("boltz_trunk_levers")
    m = sys.modules.get("boltz_graph_patch")
    if m is not None and getattr(m, "_APPLIED", {}).get("done"):
        out.append("boltz_graph_patch")
    m = sys.modules.get("boltz_dit_hoist")
    if m is not None and (getattr(m, "STATS", {}).get("level") or 0):
        out.append("boltz_dit_hoist")
    m = sys.modules.get("boltz_flash_triattn_patch")
    if m is not None and getattr(m, "_STATE", {}).get("applied"):
        out.append("boltz_flash_triattn_patch")
    return out


def late_activation_block() -> Optional[str]:
    """The named reason activation is refused now: a model instance exists, or a kit lever is already applied in this process."""
    built = core_instances.instance_check(*MODEL_CLASS)["built"]
    if built:
        return f"a Boltz2 model instance already exists in this process ({built} constructed since activation was armed)"
    applied = kit_lever_applied()
    if applied:
        return f"a kit lever is already applied in this process ({', '.join(applied)})"
    return None



# ---------------------------------------------------------------- the worker route ----------------------------------------------------------------
def stripped_env(base: Optional[dict] = None) -> Dict[str, str]:
    """The caller's environment (`base`, default os.environ) with every kit switch family stripped: the names under stock/PINS.json
    stock_environment must_be_absent_prefixes — the one strip the worker route, the stock route and the kits' probes start from."""
    prefixes = tuple(load_pins()["stock_environment"]["must_be_absent_prefixes"])
    return {k: v for k, v in (os.environ if base is None else base).items() if not k.startswith(prefixes)}


def child_env(mode: str, base: Optional[dict] = None, n_gpu: int = 1) -> Dict[str, str]:
    """The worker process environment: stripped_env(base) with BOLTZ2_OPT removed, then the mode's activation row set, then the mode's
    evidence switches (modes.EVIDENCE_ENV: report switches of the kits, never levers); at ``n_gpu > 1`` the row-sharding switch
    (``BOLTZ_TP=<P>``, registry rowpair_tp — the rank variables are the core launcher's, layered per rank at launch), the launcher's line tag
    (``ROWPAIR_TAG=boltz2-opt``: its ``[boltz2-opt] RANKENV hashseed=<v> source=default|inherited ranks=<P>`` line — one ``PYTHONHASHSEED`` for
    every rank, decided by the core launcher from this mapping), the sync policy word and the ×P line's placement words (``modes.TP_EXPORTS``).
    Every one of these is the row's alone: a caller's copy is stripped with the other lever words (stock/PINS.json must_be_absent_prefixes)
    and the row's value assigned."""
    env = stripped_env(base)
    env.pop("BOLTZ2_OPT", None)
    env.update(modes.env_row(mode))
    env.update(modes.EVIDENCE_ENV.get(modes.resolve(mode)["mode"], {}))
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if int(n_gpu or 1) > 1:
        env[registry.LEVERS["rowpair_tp"]["switch"]] = str(int(n_gpu))
        env[RANK_TAG_ENV] = rep.TAG                                          # the core launcher's and the ranks' line tag: `[boltz2-opt] RANKENV …`, `[boltz2-opt rK] …`
        env[registry.LEVERS["rowpair_tp"]["sync_switch"]] = registry.LEVERS["rowpair_tp"]["sync_default"]   # the replicated-tensor sync policy of this (det 0) route, explicit
        for k, v in modes.TP_EXPORTS.items():                                # the ×P line's placement words (the core's host-parking levers): the row's values
            env[k] = v
    return env


def tp_exports(env: dict, n_gpu: int) -> Dict[str, str]:
    """The ×P line's placement words as the worker environment `env` carries them at ``n_gpu > 1`` (``modes.TP_EXPORTS`` names, the row's
    values); ``{}`` at ``n_gpu = 1``. The run report's ``env`` and the run record carry them beside the mode row."""
    if int(n_gpu or 1) <= 1:
        return {}
    return {k: str(env[k]) for k in modes.TP_EXPORTS if k in env}


def stage(mode: str, workdir: str) -> dict:
    """Copy the mode's kit files flat into `workdir` (checked present first) and derive the worker: the `bz2dit` variant of the
    mode's base by the kit's own make_worker_variant.py (row ``variant`` = "bz2dit"), or the base as shipped (``variant`` None:
    whose base carries its own sampler and hoist hook). Returns ``{files: {basename: sha256}, worker: path, variant_line: str, variant_sha256: str, routed}``;
    raises ``RuntimeError`` naming the problem."""
    row = modes.resolve(mode)
    rels = modes.stage_files(mode)
    problems, shas = check_kit_files(rels)
    if problems:
        raise RuntimeError("kit files are missing: " + "; ".join(problems))
    os.makedirs(workdir, exist_ok=True)
    files = {}
    for rel in rels:
        dst = os.path.join(workdir, os.path.basename(rel))
        shutil.copyfile(kit_path(rel), dst); files[os.path.basename(rel)] = shas[rel]
    base = os.path.join(workdir, os.path.basename(row["worker_base"]))
    worker = os.path.join(workdir, "bz_worker.py")
    if row.get("variant") == "bz2dit":
        r = subprocess.run([sys.executable, os.path.join(workdir, "make_worker_variant.py"), base, worker], capture_output=True, text=True, cwd=workdir)
        if r.returncode != 0 or not os.path.isfile(worker):
            raise RuntimeError(f"make_worker_variant.py rc={r.returncode}: {(r.stdout + r.stderr).strip()[-400:]}")
        variant_line = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
    elif row.get("variant") is None:
        shutil.copyfile(base, worker); variant_line = f"{os.path.basename(base)} as shipped (no variant)"
    else:
        raise RuntimeError(f"unknown worker variant {row.get('variant')!r} for mode {row['mode']}")
    return {"files": files, "worker": worker, "variant_line": variant_line, "variant_sha256": sha256_file(worker),
            "routed": route_kernels(modes.routed_kernels(mode))}


def route_kernels(names: List[str]) -> dict:
    """The core's by-name route for each kernel of `names`, held in THIS process before the launch (route -> exports -> route_check,
    opt_core.kernels): the resolved file is the core copy, its bytes held byte-for-byte against the core's own copy live (route_check
    itself; nothing stored to restate here). Returns ``{name: {resolved, core_copy, kind, version, files, runtime_imports}}``; raises
    ``RuntimeError`` on a refused route. The worker process installs the same route itself (worker_launch) — this gate refuses before
    a GPU is touched."""
    from opt_core import kernels as core_kernels
    from . import report as rep
    routed = {}
    for name in names:
        core_kernels.route(name)
        exported = core_kernels.exports(name, **{k: kit_path(v) for k, v in modes.kernel_exports(name).items()})   # the worker exports the same in its own process (worker_launch); this process only checks
        g = core_kernels.route_check(name, environ={**os.environ, **exported})
        if not g.ok:
            raise RuntimeError(f"kernel route refused: {g.reason}")
        d = g.details; doc = core_kernels.sums(name)
        routed[name] = {"resolved": os.path.abspath(d["resolved"]), "core_copy": os.path.abspath(d["core_copy"]), "kind": doc["kind"], "version": doc.get("version"),
                        "files": len(doc["files"]), "runtime_imports": d["runtime_imports"]}
        rep.say(rep.route_line(name, routed[name]))
    return routed


def worker_name(mode: str) -> str:
    """The worker the route runs for `mode`, as the ACTIVE line names it: ``<base>+bz2dit`` (the derived variant) or ``<base>`` (as shipped)."""
    row = modes.resolve(mode)
    base = os.path.basename(row["worker_base"])
    return f"{base}+bz2dit" if row.get("variant") == "bz2dit" else base


def write_batch(workdir: str, tag: str, items: List[dict], seeds: List[int], out_dir: str, settings: Optional[dict] = None) -> str:
    """The batch json the kit worker takes (bz_worker_lev.py:29): items = [{name, uid, yaml, seeds}], ``out_dir`` (the predictions, by_seed/),
    ``kit_dir`` (this launch's kit directory ``workdir``: the worker's log ``<tag>_worker_log.json``, the KERNELS record and the affinity leg's
    hand-over are written there and removed by worker.run once folded — nothing but the predictions and the worker's transcript is written under
    ``out_dir``) and the call's `boltz predict` settings
    (settings.worker_settings: upstream's defaults with the caller's options over them): ``predict`` = Boltz2's predict_args dims
    (recycling_steps, sampling_steps, diffusion_samples, max_parallel_samples), ``step_scale``, ``write_full_pae``, ``write_full_pde``,
    ``output_format``, ``checkpoint`` (the caller's ``--checkpoint``, else the cache's boltz2_conf.ckpt) and ``options``: the other `boltz
    predict` options the caller GAVE (settings.OPTION_KNOBS), each read by the worker at the one site it states upstream's default."""
    from . import settings as _settings
    eff = settings if settings is not None else _settings.worker_settings("exact", None)
    cache = cache_dir(); opts = dict(eff.get("options") or {})
    bj = {"tag": tag, "items": [{"name": it["name"], "uid": it.get("uid", it["name"]), "yaml": os.path.abspath(it["yaml"]), "seeds": list(seeds)} for it in items],
          "seeds": list(seeds), "cache": cache, "checkpoint": opts.pop("checkpoint", None) or os.path.join(cache, "boltz2_conf.ckpt"), "out_dir": os.path.abspath(out_dir), "kit_dir": os.path.abspath(workdir),
          "write_full_pae": bool(eff["write_full_pae"]), "write_full_pde": bool(eff["write_full_pde"]), "output_format": str(eff["output_format"]), "step_scale": float(eff["step_scale"]),
          "predict": {k: int(eff[k]) for k in _settings.PREDICT_FLAGS},
          "options": {**{k: v for k, v in opts.items() if k not in _settings.AFFINITY_KNOBS}, "affinity": {k: v for k, v in opts.items() if k in _settings.AFFINITY_KNOBS}}}
    p = os.path.join(workdir, f"batch_{tag}.json"); json.dump(bj, open(p, "w"), indent=1)
    return p


TP_ATTACHMENT = "tp"   # worker_launch.ATTACH: the row-sharded pair stack, attached at n_gpu > 1 only (nothing is installed at n_gpu = 1)
RANK_TAG_ENV = "ROWPAIR_TAG"                                                # the core launcher's tag variable (opt_core.mem.rowpair.launch.ENV_TAG): the word inside the brackets of its RANKENV line and the ranks' lines




def worker_command(batch_json: str, mode: str, n_gpu: int = 1, settings_word: str = "defaults", num_workers: Optional[int] = None) -> List[str]:
    """The kit's worker line (modes.worker_args: the probe arm's arguments with the row's kernel state); a mode with routed kernels or an
    attach-time lever module runs it under boltz2_opt.worker_launch (the route / the attach hook installed in the worker process before its
    first import; the script and its arguments unchanged). At ``n_gpu > 1`` the attach list gains ``tp`` (the same line for every rank)."""
    a = modes.worker_args(mode)
    cmd = ["bz_worker.py", "--batch", batch_json, "--mode", a["mode"], "--kernels", a["kernels"], "--num_workers", str(num_workers) if num_workers is not None else a["num_workers"],   # the caller's boltz predict --num_workers (its DataLoader's), else the line's own
           "--pipeline", a["pipeline"], "--keep_on_gpu", a["keep_on_gpu"]]
    opts = launch_opts(mode, n_gpu, settings_word)
    if opts:
        return [sys.executable, "-m", "boltz2_opt.worker_launch"] + opts + ["--"] + cmd
    return [sys.executable] + cmd


def launch_opts(mode: str, n_gpu: int = 1, settings_word: str = "defaults") -> List[str]:
    """boltz2_opt.worker_launch's options for the mode's worker process — ``--route <the mode's core-served kernels>`` ``--attach <its
    attach-time modules>`` (modes.routed_kernels / modes.attachments; ``tp`` added at ``n_gpu > 1``); ``[]`` when the mode takes neither
    (the interpreter runs the script directly). ``settings_word``: the KERNELS census settings token (settings.settings_word: defaults | flags)."""
    routed, attached = modes.routed_kernels(mode), list(modes.attachments(mode))
    if int(n_gpu or 1) > 1:
        attached = attached + [TP_ATTACHMENT]
    return (["--route", ",".join(routed)] if routed else []) + (["--attach", ",".join(attached)] if attached else []) + kernels_opts(mode, n_gpu, settings_word)


def kernels_opts(mode: str, n_gpu: int = 1, settings_word: str = "defaults") -> List[str]:
    """boltz2_opt.worker_launch's KERNELS-census options for the mode's worker process (kernels.parse_cli_opts reads them; the same spelling the
    stock caller takes): the route word, the expected word kind per accelerator (modes.kernels_expected — the one statement), the settings
    word (defaults | flags) and n_gpu; the census record lands beside the worker log (worker_launch derives its path from the --batch json)."""
    from . import kernels
    P = int(n_gpu or 1)
    return ["--kernels-route", kernels.route_word(modes.resolve(mode)["mode"], P), "--kernels-expect", kernels.format_expect(modes.kernels_expected(mode, n_gpu=P)),
            "--kernels-settings", settings_word, "--kernels-ngpu", str(P), "--kernels-mode", modes.resolve(mode)["mode"]]


FORBIDDEN_LOG_PATTERNS = ("not applied", "autoapply failed", "PATCHED TREE MISMATCH", "[boltz2-opt attach] REFUSED")
XL_ACTIVITY = {"xl_trans": "trans_rowchunked_calls", "xl_cond": "cond_rowchunked_calls", "xl_free": "free_events", "relpos_lazy": "relpos_lazy_released"}   # memory lever -> the xl_report.stats counter that says it acted (boltz2_opt.big.STATS)


def _templ_all_elided(worker_log) -> bool:
    """True when the dummy-template elision (templskip_report) served every template-module call of the run: no templated pass, no pass without a
    mask, no error — the template Pairformer (the row-chunked transition's only caller outside the layer driver's C=128 stacks) never ran."""
    r = (worker_log or {}).get("templskip_report") or {}
    c = r.get("census") or {}
    return ("templ_skip" in (r.get("applied") or []) and int(c.get("skipped") or 0) > 0 and not int(c.get("live") or 0)
            and not int(c.get("no_mask") or 0) and not int(c.get("errors") or 0))
F2_REPORT_PREFIX = "[boltz_flash_triattn_patch] "


def read_worker_log(path: str) -> Optional[dict]:
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def f2_report(stdout_text: str) -> Optional[dict]:
    """The flash patch's own exit report — the last ``[boltz_flash_triattn_patch] {json}`` line of the worker's stdout (printed at
    interpreter exit under BOLTZ_TRIATTN_REPORT=1, boltz_flash_triattn_patch.py:267-268, report() :337-346); None when absent."""
    for line in reversed(stdout_text.splitlines()):
        s = line.strip()
        if s.startswith(F2_REPORT_PREFIX) and s[len(F2_REPORT_PREFIX):].lstrip().startswith("{"):
            try:
                return json.loads(s[len(F2_REPORT_PREFIX):])
            except ValueError:
                return None
    return None


def f2_summary(rep: dict) -> dict:
    """The patch's per-call counters (its STATS keys, boltz_flash_triattn_patch.py:170-186,202-209), summarized: flash and stock call
    counts, the exception fallbacks (``stock|exception:<E>`` / ``kfn_stock|exception:<E>`` — the patch returned the stock result for
    that call after the flash path raised), and the stock calls at or above the size gate (a ``stock|<why>|<site>|N=<n>|…`` key whose
    reason is not the gate itself)."""
    stats = rep.get("stats") or {}
    min_tokens = rep.get("min_tokens")
    out = {"flash_calls": int(rep.get("n_flash") or stats.get("flash_calls") or 0), "stock_calls": int(rep.get("n_stock") or stats.get("stock_calls") or 0),
           "min_tokens": min_tokens, "exceptions": {}, "stock_above_gate": {}, "errors": list(rep.get("errors") or [])}
    for k, v in stats.items():
        if "exception:" in k:
            out["exceptions"][k] = int(v)
            continue
        if k.startswith("stock|"):
            parts = k.split("|")
            why = parts[1] if len(parts) > 1 else ""
            n = None
            for p in parts:
                if p.startswith("N="):
                    try:
                        n = int(p[2:])
                    except ValueError:
                        n = None
            if why.startswith("n<"):
                continue
            if n is None or min_tokens is None or n >= int(min_tokens):
                out["stock_above_gate"][k] = int(v)
    return out


def partial_activation(mode: str, ev: dict) -> Tuple[List[str], Dict[str, str]]:
    """The kit's own fallbacks and documented gates for the mode's levers, from the worker's lever_report (boltz_trunk_levers.py report()):
    ``(fallbacks, gates)``. A FALLBACK is a lever of the row that ran without its optimization for a reason other than a documented rule —
    mask2: a call whose mask the scope could not classify (stats mask_scope_error) ran the stock mask arithmetic. A GATE is a documented rule
    of a lever: graph_sampler / dit_hoist's memory-headroom gate (a prediction whose projected sampler working set does not fit samples on
    the stock path, counted ``headroom_gated``). Fallbacks are a partial
    activation: the verb exits EXIT_NOT_ACTIVE unless --allow-partial is passed; gates are recorded and the verb exits by its outputs.
    rowpair_tp (``n_gpu > 1``): a ×P placement word in force in a rank that acted in a degraded form the core NAMED (a pageable
    park, a shard left resident: ``modes.placement_findings`` read off the rank's schedule census by ``evidence``) is a FALLBACK. Only the
    row's levers are judged."""
    levers = modes.levers(mode); fallbacks: List[str] = []; gates: Dict[str, str] = {}
    masks = [l for l in ("mask2",) if l in levers]
    if masks and ev.get("mask_scope_errors"):                # the mask-elision levers' scope failed to classify a mask: those calls took the stock mask path — a partial activation, never a silent counter
        fallbacks.append(f"{','.join(masks)}: {ev['mask_scope_errors']} calls could not classify the pair mask (mask_scope_error) and ran the stock mask arithmetic")
    for lever, cell in sorted((ev.get("safe_nets") or {}).items()):   # a core lever served an unpinned cell with its SAFE settings: process-wide, every pinned cell of it slower than released — named, exit 3 unless --allow-partial
        fallbacks.append(f"{lever}:safe({cell})")
    if ev.get("tp_placement_fallbacks"):                     # n_gpu > 1: a ×P placement word in force that acted in a degraded form the core named (modes.placement_findings, read by evidence())
        fallbacks.append("rowpair_tp: " + "; ".join(ev["tp_placement_fallbacks"]))
    sampler = [l for l in ("rollout", "graph_sampler", "dit_hoist") if l in levers]
    if sampler and ev.get("headroom_gated"):                  # the sampler levers' memory-headroom gate (boltz_dit_hoist.headroom_gate): a documented gate — those predictions sampled on the stock path, same bytes
        last = ev.get("headroom_last") or {}
        gates[",".join(sampler)] = (f"memory_headroom: {ev['headroom_gated']}/{ev.get('n_items')} predictions sampled on the stock path (the levers' projected working set "
                                    f"{last.get('projected_gib')} GiB did not fit the {last.get('free_gib')} GiB free with the step transients and margin); outputs identical")
    return fallbacks, gates


def rowpair_tpx_words(tpx) -> List[str]:
    """boltz2_opt.rowpair.tpx_fallback_words on a rank's ``tp_report.tpx`` dict (restated torch-free: this module never imports the adapter):
    one ``<switch>=<word>:unavailable:<why>`` per fused-kernel switch naming a kernel the installed core could not provide."""
    out = []
    for kind, ent in sorted((tpx or {}).items()):
        if isinstance(ent, dict) and ent.get("state") == "unavailable":
            out.append(f"{ent.get('env')}={ent.get('word')}:unavailable:{ent.get('reason')}")
    return out


# A core lever's SAFE settings net (opt_core.kernels.safe_settings): an UNPINNED cell key served by the lever's safe settings switches that
# lever's launches process-wide to the safe settings — every pinned cell of the lever then runs slower than released. The core names it on the
# lever's census line (evidence tail ``settings=safe:<reason>[;…]``, a served key ``…+safe`` / ``:safe``); the kit rows pin every cell they
# launch, so a safe net in a kit row is a partial activation: named on the ACTIVE line (``fallbacks=<lever>:safe(<cell>)``), exit 3 unless
# --allow-partial (a demoted run says so).
SAFE_NET_RE = re.compile(r"settings=safe:([^\s]+)")
SAFE_KEY_RE = re.compile(r"keys=([^\s]*(?:\+safe|:safe)[^\s]*)")
CORE_LINE_LEVER = {"F5.transition": "fused_transition", "F1.pairblock": "pairblock", "F2.fpf_trimul": "fpf_trimul", "F2.trimul": "fpf_trimul", "F2.tmk3": "fpf_trimul_exact"}
REPORT_LEVER = {"transition_report": "fused_transition", "pairblock_report": "pairblock", "trimul_report": "fpf_trimul", "pairfuse_report": "pairfuse"}


def safe_net_findings(mode: str, worker_log: Optional[dict], stdout_text: str = "") -> Dict[str, str]:
    """{lever: cell word} for every core census line of the pass that says a SAFE settings net served: the block adapters' report ``line``
    (the core's own lever line, kept in the worker log) and any ``[boltz2-opt] LEVER`` line on the pass' stdout."""
    found: Dict[str, str] = {}
    row = set(modes.levers(mode)) if mode in modes.MODE_NAMES else set()
    def lever_of(line: str, default: str) -> str:
        m = re.search(r"name=([A-Za-z0-9_.]+)", line or "")
        lv = CORE_LINE_LEVER.get(m.group(1), default) if m else default
        if lv == "fpf_trimul" and "fpf_trimul_exact" in row and "fpf_trimul" not in row:
            lv = "fpf_trimul_exact"
        return lv
    def scan(line: str, default: str) -> None:
        if not isinstance(line, str):
            return
        m = SAFE_NET_RE.search(line); k = SAFE_KEY_RE.search(line)
        if m or k:
            found.setdefault(lever_of(line, default), (m.group(1) if m else k.group(1)).rstrip(";,"))
    for key, lever in REPORT_LEVER.items():
        r = (worker_log or {}).get(key) or {}
        lines = r.get("line")
        for ln in (lines if isinstance(lines, list) else [lines]):
            scan(ln, lever)
    for ln in (stdout_text or "").splitlines():
        if "settings=safe:" in ln or "+safe" in ln:
            scan(ln, "core")
    return found


def _served_total(served) -> int:
    """An adapter census' served count: a per-kind dict (pairblock / transition) or a plain count."""
    if isinstance(served, dict):
        return int(sum(served.values()))
    try:
        return int(served or 0)
    except (TypeError, ValueError):
        return 0


GRAPH_PAIRFUSE_BODIES = {"pf": "pairfuse.pfm_forward", "pfnoseq": "pairfuse.pfnm_forward"}   # the PAIRFUSE driver's stack forwards the trunk CUDA-graph units capture under BOLTZ_PAIRFUSE (graph BODIES table, digests pinned there)


def graph_bodies_expected(env: dict) -> Dict[str, str]:
    """Which body each pair-stack unit of BOLTZ_GRAPH_TRUNK must report bound: the PAIRFUSE driver's under BOLTZ_PAIRFUSE (fast row), the
    stock layer stack otherwise (exact row). The template unit hoists a boundary and captures nothing: no body."""
    units = [u.strip() for u in (env.get("BOLTZ_GRAPH_TRUNK") or "").split(",") if u.strip()]
    pf = str(env.get("BOLTZ_PAIRFUSE") or "").strip().lower() not in ("", "0", "off", "none")
    return {u: (GRAPH_PAIRFUSE_BODIES[u] if pf else "stock") for u in units if u in ("pf", "pfnoseq")}


R5A2_STATES = ("comparing", "proven", "refused", "floor")        # the per-class state a bit-compared replica's census prints
R5A2_BUDGET = 10 ** 6                                            # output elements a class's per-call compares must cover before it is `proven` (GEMM / softmax / LayerNorm replicas)


def r5a2_findings(worker_log: Optional[dict]) -> List[str]:
    """For every adapter report that carries a per-class census in the agreed shape — ``<report>["classes"]`` (or
    ``<report>[<census>]["classes"]``) mapping class -> {state, compared_calls, compared_elems, served_uncompared, [budget], [candidates_tested],
    [undetermined]}: a class in an unknown state, a class that served replica output uncompared while not `proven`, or a `proven` class whose
    compares covered fewer than its budget of output elements (and not every call) is named — the run is refused. Reports without such a census
    are not this rule's (their own gates stand); the MSA exact cells and any replica adopting the shape are."""
    out: List[str] = []
    for key, rep in sorted((worker_log or {}).items()):
        if not (key.endswith("_report") and isinstance(rep, dict)):
            continue
        tables = []
        if isinstance(rep.get("classes"), dict):
            tables.append((key, rep["classes"]))
        for ck, cv in rep.items():
            if ck.endswith("_census") and isinstance(cv, dict):
                for tk in ("classes", "selftest"):                   # msa2: pwa2_census.classes / trans2_census.selftest
                    if isinstance(cv.get(tk), dict):
                        tables.append((f"{key}.{ck}.{tk}", cv[tk]))
        for where, classes in tables:
            for cls, c in sorted(classes.items(), key=lambda kv: str(kv[0])):
                if not (isinstance(c, dict) and "state" in c):
                    continue                                      # an older census shape (e.g. class -> candidate): not judged here
                st = str(c.get("state")); budget = int(c.get("budget") or R5A2_BUDGET)
                if st not in R5A2_STATES:
                    out.append(f"{where} class {cls}: state {st!r} is none of {R5A2_STATES}")
                elif int(c.get("served_uncompared") or 0) > 0 and st != "proven":
                    out.append(f"{where} class {cls}: served {c.get('served_uncompared')} call(s) of replica output uncompared while {st} (compared_elems={c.get('compared_elems')} < budget {budget})")
                elif st == "proven" and int(c.get("compared_elems") or 0) < budget and not c.get("every_call_compared"):
                    out.append(f"{where} class {cls}: proven on {c.get('compared_elems')} compared output elements < budget {budget} (compared_calls={c.get('compared_calls')})")
    return out


def evidence(mode: str, worker_log: Optional[dict], stdout_text: str = "", attached: bool = True, n_gpu: int = 1,
             rank_worker_logs: Optional[Dict[int, Optional[dict]]] = None, sampling_steps: Optional[int] = None) -> Tuple[dict, List[str]]:
    """What the worker's own log says was applied, against what the mode requires. Returns ``(evidence, problems)``; problems empty = the
    row was in force for every item. ``attached`` = the worker ran under this package's worker_launch (pred): the guard
    attachments' reports (templ_report) are then required evidence (attached=False: a worker launched bare, without
    worker_launch — no kit mode does). ``n_gpu > 1``: every rank's ``tp_report`` (rank 0's in ``worker_log``, the
    others' in ``rank_worker_logs``) must say the row-sharded pair stack stood and ran (installed, sharded-module calls > 0, the rows census
    tiling 0:N once, no errors) — a rank without it is a problem by name; ``tp_peaks`` = ``r<k>:<GiB>`` per rank. ``sampling_steps``: the
    call's (upstream's default when None); the graph sampler replays steps 1..S-1 of every prediction, so each item's ``n_replay`` must be S-1."""
    row = modes.resolve(mode); env = row["env"]; problems = []
    from . import templates as templ_guard
    ev = {"levers": [], "graph_mode": None, "hoist_level": None, "f2": None, "f2_report": None, "n_items": 0, "n_replay": None, "captured_with_cache": None,
          "xl_report": None, "trimul_report": None, "opm_report": None, "transition_report": None, "pairblock_report": None, "triattn_exact_report": None, "exactln_report": None,
          "sampler_report": None, "ditexact_report": None, "graph_report": None, "pairfuse_report": None, "atom_report": None,
          "waste_report": None, "msa2_report": None, "conf_report": None, "precision_report": None,
          "templ_report": None, "prefetch_report": None, "writer_report": None, "has_worker_log": worker_log is not None,
          "forbidden_lines": [l.strip() for l in stdout_text.splitlines() if any(p in l for p in FORBIDDEN_LOG_PATTERNS)],
          "template_all_dummy": [l.strip() for l in stdout_text.splitlines() if l.strip().startswith(templ_guard.ALL_DUMMY_MARK)],   # named events of the run (the input ran untemplated, as stock runs it), not problems
          "template_dropped": [l.strip() for l in stdout_text.splitlines() if l.strip().startswith(templ_guard.DROPPED_MARK)]}
    if ev["forbidden_lines"]:
        problems.append("the kit reported a lever not applied: " + " | ".join(ev["forbidden_lines"][:3]))
    from . import kernels as _kernels                                  # the KERNELS census: one pass line per worker process in its transcript (kernels.emit at exit; rank 0's is the run's) — parsed here, judged by report.kernels_exit (the one gate: no record -> NOT ACTIVE, verdict != PASS -> exit 5)
    ev["kernels_lines"] = [l.strip() for l in stdout_text.splitlines() if _kernels.LINE_MARK in l and _kernels.REFUSED_MARK not in l]
    ev["kernels_refused_lines"] = [l.strip() for l in stdout_text.splitlines() if _kernels.REFUSED_MARK in l]
    ev["kernels"] = (_kernels.find_lines(stdout_text) or [None])[0]
    if worker_log is None:
        problems.append("no worker log (<tag>_worker_log.json): the kit worker did not run to its log")
        return ev, problems

    if attached and "templ" in modes.attachments(row["mode"]):   # the guard's own census is the evidence that it stood in the worker (templ_report.installed)
        tr = worker_log.get("templ_report"); ev["templ_report"] = tr
        if not tr or not tr.get("installed"):
            problems.append("no templ_report in the worker log (or installed=false): the template guard did not stand in the worker process")
    ev["tp_report"] = worker_log.get("tp_report"); ev["tp_peaks"] = None
    if int(n_gpu or 1) > 1:                                        # the row-sharded pair stack (registry rowpair_tp): every rank's own report is the evidence
        logs = dict(rank_worker_logs or {}); logs[0] = worker_log
        peaks = []
        placement: List[str] = []
        for rk in range(int(n_gpu)):
            tr = (logs.get(rk) or {}).get("tp_report") if logs.get(rk) is not None else None
            if not tr or not tr.get("installed"):                 # fail-closed: the requested P must be the P every rank's model process reports — never a pass on fewer
                problems.append(f"reason=n_gpu_mismatch requested={n_gpu} active={sum(1 for x in range(int(n_gpu)) if ((logs.get(x) or {}).get('tp_report') or {}).get('installed'))}: "
                                f"rank {rk} has no tp_report (or installed=false) in its worker log: the row-sharded pair stack did not stand in that rank"); continue
            calls = tr.get("calls") or {}
            if int(calls.get("trunk_rows", 0)) == 0:                    # the born-sharded trunk ran on this rank (one per item)
                problems.append(f"n_gpu={n_gpu}: rank {rk} ran no row-sharded trunk (tp_report.calls.trunk_rows=0)")
            for seam, key in (("diffusion conditioning rows", "zcond_rows"), ("DiffusionTransformer local rows", "dit_blocks_sharded"), ("confidence rows", "conf_rows")):
                if int(calls.get("trunk_rows", 0)) and int(calls.get(key, 0)) == 0:
                    problems.append(f"n_gpu={n_gpu}: rank {rk}: the {seam} seam did not run (tp_report.calls.{key}=0)")
            idle = [k for k in (tr.get("required_calls") or ()) if int(calls.get(k, 0)) < 1]   # every bound seam ran at least once on this rank (the counter is the evidence, not the ACTIVE line)
            if int(calls.get("trunk_rows", 0)) and idle:
                problems.append(f"n_gpu={n_gpu}: rank {rk}: bound seams that never ran: {','.join(idle)} (tp_report.calls)")
            for key in ("gathers_in_trunk", "gathers_named", "whole_named"):  # nothing N x N whole on a rank: a gathered pair tensor is a named defect, never a pass
                if int(calls.get(key, 0)):
                    problems.append(f"n_gpu={n_gpu}: rank {rk}: {key}={calls.get(key)} (a whole pair tensor stood on the rank)")
            if int(tr.get("n_gpu") or 0) != int(n_gpu):
                problems.append(f"reason=n_gpu_mismatch requested={n_gpu} active={tr.get('n_gpu')}: rank {rk}'s model process reports n_gpu={tr.get('n_gpu')}")
            rc_ = tr.get("rows_census") or {}
            if rc_ and not rc_.get("rows_tiled", False):
                problems.append(f"n_gpu={n_gpu}: rank {rk} rows census says the ranks' rows do not tile 0:N ({rc_})")
            if tr.get("errors"):
                problems.append(f"n_gpu={n_gpu}: rank {rk} tp_report errors: {'; '.join(map(str, tr['errors']))[:300]}")
            fb, pb = modes.placement_findings(tr.get("tp_exports") or {}, tr.get("schedule") or {})   # the ×P placement words READ off this rank's census against the words in force there
            fb = list(fb) + rowpair_tpx_words(tr.get("tpx"))                # a fused-kernel switch naming a kernel the core could not provide: the kit's own statement served, BY NAME
            placement.extend(f"r{rk}: {f}" for f in fb)
            problems.extend(f"n_gpu={n_gpu}: rank {rk}: {p}" for p in pb)
            peaks.append(f"r{rk}:{tr.get('peak_alloc_gib')}")
        ev["tp_peaks"] = ",".join(peaks) if peaks else None
        ev["tp_placement_fallbacks"] = placement
    elif (worker_log.get("tp_report") or {}).get("installed"):
        problems.append("n_gpu=1 but the worker log carries an installed tp_report: the row-sharded pair stack must install nothing at n_gpu=1")
    if env.get("BOLTZ_TRIATTN") == "flash":
        f2r = f2_report(stdout_text)
        if f2r is None:
            problems.append("no flash tri-attention report line ([boltz_flash_triattn_patch] {…}, printed at the worker's exit under BOLTZ_TRIATTN_REPORT=1): "
                            "the per-call flash/stock counters are the evidence the lever was in force")
        else:
            s = f2_summary(f2r); ev["f2_report"] = s
            if s["exceptions"] or s["errors"]:
                problems.append(f"flash tri-attention fell back to stock at run time: {s['exceptions'] or s['errors'][:2]} (flash_calls={s['flash_calls']}, stock_calls={s['stock_calls']})")
            if s["flash_calls"] == 0 and s["stock_above_gate"]:
                problems.append(f"flash tri-attention never engaged although calls at or above the size gate ran on the stock path: {s['stock_above_gate']}")
    wenv = worker_log.get("env") or {}; rep = worker_log.get("lever_report") or {}
    ev["levers"] = list(wenv.get("boltz_levers") or rep.get("levers") or [])
    st = rep.get("stats") or {}
    ev["mask_scope_errors"] = int(st.get("mask_scope_error") or 0)                       # boltz_trunk_levers.py:143 — a mask the scope could not classify: that call ran the stock mask arithmetic
    ev["graph_mode"] = wenv.get("graph_diffusion"); ev["hoist_level"] = wenv.get("dit_hoist"); ev["f2"] = wenv.get("f2_patch_active")
    items = worker_log.get("per_item") or []; ev["n_items"] = len(items)
    if env.get("BOLTZ_LEVERS"):
        want = _expand_levers(env["BOLTZ_LEVERS"])
        missing = [l for l in want if l not in ev["levers"]]
        if missing:
            problems.append(f"trunk levers not reported applied by the worker: {','.join(missing)} (worker env.boltz_levers={ev['levers']})")
    if env.get("BOLTZ_GRAPH_DIFFUSION") and ev["graph_mode"] != env["BOLTZ_GRAPH_DIFFUSION"]:
        problems.append(f"graph sampler not applied (worker env.graph_diffusion={ev['graph_mode']!r}, want {env['BOLTZ_GRAPH_DIFFUSION']!r})")
    if env.get("BOLTZ_DIT_HOIST") and str(ev["hoist_level"]) != env["BOLTZ_DIT_HOIST"]:
        problems.append(f"hoist not applied at level {env['BOLTZ_DIT_HOIST']} (worker env.dit_hoist={ev['hoist_level']!r})")
    if env.get("BOLTZ_TRIATTN") == "flash" and ev["f2"] is not True:
        problems.append(f"flash tri-attention patch not active (worker env.f2_patch_active={ev['f2']!r})")
    if env.get("BOLTZ_FPF_TRIMUL") in ("1", "exact"):   # the fused TriMul's own report (boltz2_opt.trimul.report(): applied / line / census / gate), written by the attach hook
        tr = worker_log.get("trimul_report") or {}; ev["trimul_report"] = tr
        want_tm = "fpf_trimul" if env["BOLTZ_FPF_TRIMUL"] == "1" else "fpf_trimul_exact"
        if not tr:
            problems.append("no trimul_report in the worker log: the fused TriMul's census (served / fallback_by / errors) and its fail-closed gate are the evidence")
        else:
            if want_tm not in (tr.get("applied") or []):
                problems.append(f"fused TriMul {want_tm} not reported applied (trimul_report.applied={tr.get('applied')})")
            g = tr.get("gate") or {}
            ev["trimul_idle"] = bool(g.get("idle"))            # installed, every call below the size gate / outside the kernel's cells (boltz2_opt.trimul.verdict)
            if not g.get("ok"):
                problems.append(f"fused TriMul gate refused: {g.get('reason') or 'no gate in the report'} (census={tr.get('census')})")
    for sw, key, lever, what in (("BOLTZ_TRANSITION", "transition_report", "fused_transition", "fused transition"), ("BOLTZ_PAIRBLOCK", "pairblock_report", "pairblock", "fused triangle-attention block"),
                                 ("BOLTZ_TRIATTN_EXACT", "triattn_exact_report", "triattn_exact", "exact triangle-attention word binding"),   # same shape (applied / variant / census / gate; idle = no call reached the block's library core)
                                 ("BOLTZ_EXACTLN", "exactln_report", "exactln", "ATen layer_norm replica"),
                                 ("BOLTZ_TEMPL_SKIP", "templskip_report", "templ_skip", "dummy-template elision")):   # same report shape (applied / variant / census / gate; idle = every pass templated or captured)   # the layer_norm replica's report has the block adapters' shape (applied / variant / census.served / gate) [EXACTLN]
        if env.get(sw):   # the adapter's own report (boltz2_opt.transition / boltz2_opt.pairblock / boltz2_opt.exactln report(): applied / line / census / gate), written by the attach hook
            r = worker_log.get(key) or {}; ev[key] = r
            if not r:
                problems.append(f"no {key} in the worker log: the {what}'s census (served / fallback by reason / errors) and its fail-closed gate are the evidence")
            else:
                if lever not in (r.get("applied") or []):
                    problems.append(f"{what} not reported applied ({key}.applied={r.get('applied')})")
                if r.get("variant") != env[sw]:
                    problems.append(f"{what} variant {r.get('variant')!r} != the row's {sw}={env[sw]!r}")
                g = r.get("gate") or {}
                ev[key[:-7] + "_idle"] = bool(g.get("idle"))    # installed, every call on a DECLARED stock path (the adapter's verdict: e.g. every pair transition under the card's row floor) — the same standing as the TriMul idling below its size gate
                if not g.get("ok"):
                    problems.append(f"{what} gate refused: {g.get('reason') or 'no gate in the report'} (census={r.get('census')})")
                elif items and not g.get("idle") and _served_total((r.get("census") or {}).get("served")) == 0 and not (env.get("BOLTZ_PAIRFUSE") and lever != "exactln"):
                    problems.append(f"{what} installed but served no call over {len(items)} item(s) ({key}.census={r.get('census')})")   # under BOLTZ_PAIRFUSE the layer driver serves the C=128 stacks itself: the block adapters stand for the stacks it hands back and may serve none — their LEVER lines say superseded_by=pairfuse@c128
    if env.get("BOLTZ_PAIRFUSE"):   # the PAIRFUSE layer driver's own report (boltz2_opt.pairfuse.report(): applied / line / census {served by stack kind, fallback by reason, sites, errors} / gate / providers), written by the attach hook [PAIRFUSE]
        r = worker_log.get("pairfuse_report") or {}; ev["pairfuse_report"] = r
        if not r:
            problems.append("no pairfuse_report in the worker log: the layer driver's census (stacks served / fallback by reason / sites / errors) and its fail-closed gate are the evidence")
        else:
            if "pairfuse" not in (r.get("applied") or []):
                problems.append(f"PAIRFUSE layer driver not reported applied (pairfuse_report.applied={r.get('applied')})")
            if r.get("variant") != env["BOLTZ_PAIRFUSE"]:
                problems.append(f"PAIRFUSE word {r.get('variant')!r} != the row's BOLTZ_PAIRFUSE={env['BOLTZ_PAIRFUSE']!r}")
            g = r.get("gate") or {}
            ev["pairfuse_idle"] = bool(g.get("idle"))
            if not g.get("ok"):
                problems.append(f"PAIRFUSE gate refused: {g.get('reason') or 'no gate in the report'} (census={r.get('census')})")
            elif items and not g.get("idle") and _served_total((r.get("census") or {}).get("served")) == 0:
                problems.append(f"PAIRFUSE installed but served no pair stack over {len(items)} item(s) (pairfuse_report.census={r.get('census')})")
    if env.get("BOLTZ_SAMPLER_ROLLOUT") or env.get("BOLTZ_SAMPLER_ALIGN") or env.get("BOLTZ_SAMPLER_DIT"):   # the sampler adapter's own report (boltz2_opt.sampler.report(): applied / levers / line / gate / module.stats), written by the attach hook [SAMPLER]
        from . import sampler as _SM
        r = worker_log.get("sampler_report") or {}; ev["sampler_report"] = r
        if not r:
            problems.append("no sampler_report in the worker log: the sampler levers' census (calls / samples / replays / scope words) and their fail-closed gate are the evidence")
        else:
            want_sm = [lv for lv, sw_ in (("rollout", "BOLTZ_SAMPLER_ROLLOUT"), ({"jacobi64": "align_jacobi64", "aligncap": "align_aligncap"}.get(env.get("BOLTZ_SAMPLER_ALIGN", ""), "align_" + env.get("BOLTZ_SAMPLER_ALIGN", "")), "BOLTZ_SAMPLER_ALIGN"), ("dit_fused", "BOLTZ_SAMPLER_DIT")) if env.get(sw_)]
            missing = [n for n in want_sm if n not in (r.get("applied") or [])]
            if missing:
                problems.append(f"sampler levers not reported applied: {','.join(missing)} (sampler_report.applied={r.get('applied')} levers={r.get('levers')})")
            g = r.get("gate") or {}
            if not g.get("ok"):
                problems.append(f"sampler levers gate refused: {g.get('reason') or 'no gate in the report'} (stats={((r.get('module') or {}).get('stats') or {}).get('scope')})")
    if env.get("BOLTZ_DIT_EXACT"):   # the DITEXACT adapter's own report (boltz2_opt.ditexact.report(): applied / words / census / gate / tune), written by the attach hook [DITEXACT]
        from . import ditexact as _DX
        dr = worker_log.get("ditexact_report") or {}; ev["ditexact_report"] = dr
        want_dx = [_DX.LEVER_OF[w] for w in _DX.words(env)]
        if not dr:
            problems.append("no ditexact_report in the worker log: the token-transformer schedule's census (calls / calls_par / mask_skipped / sba_fused / class-forward reasons) and its fail-closed gate are the evidence")
        else:
            missing = [l for l in want_dx if l not in (dr.get("applied") or [])]
            if missing:
                problems.append(f"DITEXACT levers not reported applied: {missing} (ditexact_report.applied={dr.get('applied')})")
            g = dr.get("gate") or {}
            ev["ditexact_idle"] = bool(g.get("idle"))
            if not g.get("ok"):
                problems.append(f"DITEXACT gate refused: {g.get('reason') or 'no gate in the report'} (census={dr.get('census')})")
            c = dr.get("census") or {}; ws_ = set(_DX.words(env)); calls = int(c.get("calls") or 0); L = DITEXACT_LAYERS
            if c.get("sba_torch_by"):                                            # the fused scale+bias(+mask) pass handed a class back to torch: named, refused
                problems.append(f"DITEXACT sba served by torch on {c.get('sba_torch_by')} (sba_fused={c.get('sba_fused')}, calls={calls})")
            if "smx" in ws_:                                                     # the fused softmax replica subsumes the sba pass: every layer of every call fused or declined BY A DECLARED WORD (ditexact.SMX_DECLARED: the replica's size range — those rows take torch's statements, exact either way), every class bit-compared and served, nothing else declined; sba_fused may be 0
                bc = c.get("smx_bitcmp") or {}; dec = dict(c.get("smx_declined_by") or {})
                declared = tuple(getattr(_DX, "SMX_DECLARED", ()) or ())
                undeclared = {k: v for k, v in dec.items() if k not in declared}
                n_dec = sum(int(v or 0) for k, v in dec.items() if k in declared)
                if calls and int(c.get("smx_fused") or 0) + n_dec != L * calls:
                    problems.append(f"DITEXACT smx_fused={c.get('smx_fused')} + declared declines {n_dec} ({ {k: v for k, v in dec.items() if k in declared} }) != {L}*calls={L * calls}")
                if any(v != "served" for v in bc.values()) or (calls and int(c.get("smx_fused") or 0) > 0 and not bc):
                    problems.append(f"DITEXACT smx bit comparison not served for every class: smx_bitcmp={bc}")
                if undeclared:
                    problems.append(f"DITEXACT smx declined by an undeclared word: {undeclared} (declared: {declared})")
            elif "sba" in ws_ and calls and int(c.get("sba_fused") or 0) != L * calls:   # sba without smx: every layer of every call fused
                problems.append(f"DITEXACT sba_fused={c.get('sba_fused')} != {L}*calls={L * calls}")
            if "glue" in ws_:                                                    # the fused elementwise glue replicas: every site of every layer of every call fused (5 x 24 per call), every class bit-compared and served, NOTHING declined (no declared decline word: a decline refuses)
                gb = c.get("glue_bitcmp") or {}
                if calls and int(c.get("glue_fused") or 0) != DITEXACT_GLUE_PER_CALL * calls:
                    problems.append(f"DITEXACT glue_fused={c.get('glue_fused')} != {DITEXACT_GLUE_PER_CALL}*calls={DITEXACT_GLUE_PER_CALL * calls}")
                if any(v != "served" for v in gb.values()) or (calls and int(c.get("glue_fused") or 0) > 0 and not gb) or c.get("glue_declined_by"):
                    problems.append(f"DITEXACT glue bit comparison not served for every class / declined: glue_bitcmp={gb} glue_declined_by={c.get('glue_declined_by')}")
            if c.get("cc_refused"):                                              # a word of the row refused at install on this card (unproven_cc): named — the row asked for it (modes.CARD_DROPS takes it off the row on the cards this release supports)
                problems.append(f"DITEXACT words refused at install on this card: {c.get('cc_refused')}")
    if env.get("BOLTZ_GRAPH_TRUNK", "").strip().lower() not in ("", "0", "off", "none"):   # the trunk CUDA-graph lever's own report (boltz2_opt.graph.report(): applied / units / line / census / gate), written by the attach hook [GRAPH]
        r = worker_log.get("graph_report") or {}; ev["graph_report"] = r
        if not r:
            problems.append("no graph_report in the worker log: the trunk CUDA-graph lever's census (replayed / captured / eager by word / errors) and its fail-closed gate are the evidence")
        else:
            if "graph_trunk" not in (r.get("applied") or []):
                problems.append(f"trunk CUDA-graph lever not reported applied (graph_report.applied={r.get('applied')}, units={r.get('units')})")
            g = r.get("gate") or {}
            ev["graph_idle"] = bool(g.get("idle"))               # installed, every call on a declared eager word (token range) — installed behind its gate
            if not g.get("ok"):
                problems.append(f"trunk CUDA-graph lever gate refused: {g.get('reason') or 'no gate in the report'} (census={r.get('census')})")
            if r.get("errors"):                                   # a capture / replay / body error of any unit is named and refuses the run (GRAPH LEVER_SPEC, int3)
                problems.append(f"trunk CUDA-graph lever errors: {r.get('errors')}")
            want_bodies = graph_bodies_expected(env)               # the BODY each captured pair-stack unit must have bound: the PAIRFUSE driver's forwards under BOLTZ_PAIRFUSE, the stock layer stack otherwise
            got_bodies = {u: str(b) for u, b in (r.get("bodies") or {}).items() if u in want_bodies}
            ev["graph_bodies"] = dict(r.get("bodies") or {})
            if "bodies" not in r:
                problems.append("graph_report carries no `bodies` table: which body each captured pair-stack unit bound (stock | pairfuse.pfm_forward / pfnm_forward) is the evidence of what the graphs replay")
            elif got_bodies != {u: b for u, b in want_bodies.items() if u in (r.get("units") or want_bodies)}:
                problems.append(f"trunk CUDA-graph bodies {got_bodies} != the row's {want_bodies} (BOLTZ_PAIRFUSE={env.get('BOLTZ_PAIRFUSE') or '-'})")
            if items and not g.get("idle"):                      # an item inside the token gate ran: the pair-stack units replayed or captured (else every call was eager by a declared word and the gate says idle)
                cz = r.get("census") or {}
                served_by_graph = sum(int((cz.get(u) or {}).get(w) or 0) for u in ("pf", "pfnoseq") for w in ("replayed", "captured"))
                ev["graph_served_pf"] = served_by_graph
                pf_units = [u for u in ("pf", "pfnoseq") if u in (r.get("units") or [])]
                if pf_units and served_by_graph == 0 and not env.get("BOLTZ_PAIRFUSE"):
                    problems.append(f"trunk CUDA-graph units {pf_units} neither replayed nor captured over {len(items)} item(s) though the gate is not idle (census={cz})")
                if pf_units and env.get("BOLTZ_PAIRFUSE"):            # composed with the PAIRFUSE driver: its gate ok AND (it served > 0 OR the graphs replayed/captured its bodies) — GRAPH LEVER_SPEC §11.5
                    pr_ = worker_log.get("pairfuse_report") or {}; pserved = sum(((pr_.get("census") or {}).get("served") or {}).values()) if isinstance((pr_.get("census") or {}).get("served"), dict) else 0
                    pfb = (pr_.get("census") or {}).get("fallback") or {}
                    handed_off = bool(pfb) and all(str(k).startswith("no_cell:") for k in pfb) and pserved == 0   # every stack stepped aside BY NAME to the per-layer path (a site's provider carries no cell on this card): the row's per-layer levers served, named on the driver's line
                    ev["pairfuse_handed_off"] = handed_off
                    if not ((pr_.get("gate") or {}).get("ok") and (pserved > 0 or served_by_graph > 0 or handed_off)):
                        problems.append(f"trunk CUDA-graph units {pf_units} under the PAIRFUSE driver: pairfuse gate={(pr_.get('gate') or {}).get('ok')} served={pserved} and graphs replayed+captured={served_by_graph} — neither the driver nor its captured bodies served the C=128 stacks")
    if env.get("BOLTZ_ATOM"):      # the atom-attention levers' own report (boltz2_opt.atom.report(): applied / variant / gemm / units.<unit>.census / gate), written by the attach hook [ATOM]
        from . import atom as _AT
        ar_ = worker_log.get("atom_report") or {}; ev["atom_report"] = ar_
        want_units = _AT.units(env); want_levers = [_AT.UNITS[u]["lever"] for u in want_units] + ([_AT.GEMM_LEVER["lever"]] if _AT.gemm_word(env) != "ieee" else [])
        if not ar_:
            problems.append("no atom_report in the worker log: the atom-attention levers' census per unit (served / fallback_by / errors) and their fail-closed gate are the evidence")
        else:
            missing = [n for n in want_levers if n not in (ar_.get("applied") or [])]
            if missing:
                problems.append(f"atom-attention levers not reported applied: {','.join(missing)} (atom_report.applied={ar_.get('applied')})")
            if (ar_.get("variant") or "") != ",".join(want_units):
                problems.append(f"atom-attention levers variant {ar_.get('variant')!r} != the row's BOLTZ_ATOM units {','.join(want_units)!r}")
            if (ar_.get("gemm") or "ieee") != _AT.gemm_word(env):
                problems.append(f"atom-attention levers gemm word {ar_.get('gemm')!r} != the row's BOLTZ_ATOM_GEMM {_AT.gemm_word(env)!r}")
            g = ar_.get("gate") or {}
            if not g.get("ok"):
                problems.append(f"atom-attention levers gate refused: {g.get('reason') or 'no gate in the report'} (units={ {u: (r or {}).get('census') for u, r in (ar_.get('units') or {}).items()} })")
            elif items:
                for u in want_units:
                    c = ((ar_.get("units") or {}).get(u) or {}).get("census") or {}
                    if int(c.get("served") or 0) == 0:
                        problems.append(f"atom-attention lever unit {u} installed but served no call over {len(items)} item(s) (atom_report.units.{u}.census={c})")
    if env.get("BOLTZ_WASTE"):     # the WASTE adapter's own report (boltz2_opt.waste.report(): applied / requested / module {state per unit, stats, over} / gate), written by the attach hook [WASTE]
        from . import waste as _WS
        wr_ = worker_log.get("waste_report") or {}; ev["waste_report"] = wr_
        want_units = _WS.units(env)
        if not wr_:
            problems.append("no waste_report in the worker log: the WASTE hoists' state per unit (on | skipped by name), their counters and their fail-closed verdict are the evidence")
        else:
            st_ = ((wr_.get("module") or {}).get("state") or {})
            for u in want_units:
                us = st_.get(u) or {}
                if us.get("state") not in ("on", "skipped"):
                    problems.append(f"WASTE unit {u} neither on nor skipped by name (waste_report.module.state.{u}={us or None}; applied={wr_.get('applied')})")
                elif us.get("state") == "skipped" and not (us.get("reason") or us.get("skipped")):
                    problems.append(f"WASTE unit {u} skipped without a named reason (waste_report.module.state.{u}={us})")
            g = wr_.get("gate") or {}
            ev["waste_idle"] = bool(g.get("idle"))                # installed, no chunked call ran (inputs at or below the 384-token chunking threshold): installed behind its gate
            if not g.get("ok"):
                problems.append(f"WASTE gate refused: {g.get('reason') or 'no gate in the report'} (stats={(wr_.get('module') or {}).get('stats')})")
    if env.get("BOLTZ_MSA2"):      # the MSA-module levers' adapter report (boltz2_opt.msa2.report(): applied units / requested / units.<unit> {state, mode, reason} / trans2_census incl. the exact unit's runtime self-check), written by the attach hook [MSA]
        from . import msa2 as _M2
        mr2 = worker_log.get("msa2_report") or {}; ev["msa2_report"] = mr2
        want_units = _M2.requested(env)
        if not mr2:
            problems.append("no msa2_report in the worker log: the fused dim-64 transition's census (served / calls / fallback_by / self-check classes) is the evidence")
        else:
            for u in want_units:
                us = (mr2.get("units") or {}).get(u) or {}
                if u not in (mr2.get("applied") or []) or us.get("state", "on") != "on":
                    problems.append(f"MSA2 unit {u} not applied (msa2_report.applied={mr2.get('applied')} units.{u}={us or None})")
            c = mr2.get("trans2_census") or {}
            fb = dict(c.get("fallback_by") or {})
            ev["msa2_fallback_by"] = fb
            bad = {w: n for w, n in fb.items() if w.split(":")[0] not in MSA2_DECLARED_FALLBACKS and not w.startswith(MSA2_DECLARED_PREFIXES)}     # fail-closed: the exact unit's run-time bit comparison refused a shape class (the stock statements served it, counted), the kernel raised, or any word this table does not declare — the row's lever did not act as released: refused by name
            if bad:
                problems.append(f"MSA2 fused transition fell back by an undeclared word: {bad} (declared: {sorted(MSA2_DECLARED_FALLBACKS)}; trans2_census={ {k: v for k, v in c.items() if k != 'shapes'} })")
            if items and "trans2" in want_units and int(c.get("served") or 0) == 0 and int(c.get("calls") or 0) > 0:      # the fast unit serves every supported call; the exact unit (trans2x) may serve nothing on small inputs (floor / comparing calls return the stock output)
                problems.append(f"MSA2 fused transition installed but served none of {c.get('calls')} dim-64 transition call(s) (fallback_by={fb})")
            if "trans2x" in want_units and int(c.get("calls") or 0) != int(c.get("served") or 0) + int(c.get("compared") or 0) + int(c.get("fallback") or 0):
                problems.append(f"MSA2 trans2x accounting: calls={c.get('calls')} != served {c.get('served')} + compared {c.get('compared')} + fallback {c.get('fallback')} (every call is served proven, compared-and-stock, or a declared fallback)")
            if "trans2x" in want_units and int(c.get("served") or 0) > 0 and not any((v or {}).get("state") == "proven" for v in (c.get("selftest") or {}).values() if isinstance(v, dict)):
                problems.append(f"MSA2 trans2x served {c.get('served')} call(s) with no class proven (selftest={c.get('selftest')}): replica output is returned only by a proven class")
            if "pwa2x" in want_units:                                            # the exact fused pair-weighted averaging: its own census, the same declared words, fail-closed alike
                c2 = mr2.get("pwa2_census") or {}; fb2 = dict(c2.get("fallback_by") or {}); ev["msa2_pwa2_fallback_by"] = fb2
                if "pwa2_census" not in mr2:                                     # no census, no evidence: the cell's served / fallback / bit-comparison record is the evidence it ran as released
                    problems.append("MSA2 fused pair-weighted averaging (pwa2x) rides the row but msa2_report carries no pwa2_census: its served / fallback_by / classes record is the evidence")
                bad2 = {w: n for w, n in fb2.items() if w.split(":")[0] not in MSA2_DECLARED_FALLBACKS and not w.startswith(MSA2_DECLARED_PREFIXES)}
                if bad2:
                    problems.append(f"MSA2 fused pair-weighted averaging fell back by an undeclared word: {bad2} (pwa2_census={ {k: v for k, v in c2.items() if k != 'classes'} })")
                if int(c2.get("calls") or 0) != int(c2.get("served") or 0) + int(c2.get("compared") or 0) + int(c2.get("fallback") or 0):      # served (proven class) + compared (stock output returned) + declared fallbacks account for every call; serving nothing on small inputs is the floor, by name
                    problems.append(f"MSA2 pwa2x accounting: calls={c2.get('calls')} != served {c2.get('served')} + compared {c2.get('compared')} + fallback {c2.get('fallback')}")
                if int(c2.get("served") or 0) > 0 and not any((v or {}).get("state") == "proven" for v in (c2.get("classes") or {}).values() if isinstance(v, dict)):
                    problems.append(f"MSA2 pwa2x served {c2.get('served')} call(s) with no class proven (classes={c2.get('classes')}): replica output is returned only by a proven class")
    if env.get("BOLTZ_PRECISION"):   # the PRECISION units' own report (boltz2_opt.precision.report(): applied / disabled / units / gate / bundle), written by the attach hook [PRECISION]
        from . import precision as _PR
        pr_ = worker_log.get("precision_report") or {}; ev["precision_report"] = pr_
        want_on, want_off = _PR.parse(env)                          # ValueError names a bad token (a unit that is not one, a unit both on and off)
        if want_on and not pr_:
            problems.append("no precision_report in the worker log: the precision units' census (calls per unit, pins, flag_restored) and their fail-closed gate are the evidence")
        elif pr_:
            missing = [u for u in want_on if u not in (pr_.get("applied") or [])]
            if missing:
                problems.append(f"precision units not reported applied: {','.join(missing)} (precision_report.applied={pr_.get('applied')})")
            stray = [u for u in want_off if ((pr_.get("units") or {}).get(u) or {}).get("state") != "off"]
            if stray:
                problems.append(f"precision ablation entries not reported off: {','.join(stray)} (the row says off:<unit>; precision_report.units={ {u: (r or {}).get('state') for u, r in (pr_.get('units') or {}).items()} })")
            g = pr_.get("gate") or {}
            if want_on and not g.get("ok"):
                problems.append(f"precision units gate refused: {g.get('reason') or 'no gate in the report'} (units={ {u: (r or {}).get('census') for u, r in (pr_.get('units') or {}).items()} })")
            if "dit_tf32" in want_on:                                # the TF32 unit's own contract, re-read here: the process flag restored after EVERY call, stock's policy at exit
                c = ((pr_.get("units") or {}).get("dit_tf32") or {}).get("census") or {}
                if not c.get("calls") or c.get("flag_restored") != c.get("calls"):
                    problems.append(f"dit_tf32: flag_restored={c.get('flag_restored')} != calls={c.get('calls')} (torch.backends.cuda.matmul.allow_tf32 must be restored after every score-model call)")
                b = pr_.get("bundle") or {}
                if b and (b.get("tf32_flag_now") is not False or b.get("matmul_precision") != "highest"):
                    problems.append(f"dit_tf32: the process matmul policy at exit is not stock's (allow_tf32={b.get('tf32_flag_now')}, float32_matmul_precision={b.get('matmul_precision')!r}; stock: False / 'highest')")
            if want_on and env.get("BOLTZ_TRANSITION"):              # composition: a precision unit must not hand the fused-transition adapter an unpinned cell (CR5: an unknown cell engages opt_core's SAFE
                tline = (worker_log.get("transition_report") or {}).get("line") or ""   # transition settings process-wide — trunk +7.5 %%); the F5.transition line names the settings word it ran
                m_ = re.search(r"\bsettings=(\S+)", tline)
                if m_ and m_.group(1) != "defaults":
                    problems.append(f"precision units composed with non-default fused-transition settings ({m_.group(1)}): an unpinned Transition cell reached the fused adapter while a precision unit was on")
    if env.get("BOLTZ_CONF"):      # the CONF levers' own report (boltz2_opt.conf.report(): applied / words / line / census / gate), written by the attach hook [CONF]
        from . import conf as _CF
        cr_ = worker_log.get("conf_report") or {}; ev["conf_report"] = cr_
        want_cf = _CF.parse_words(env["BOLTZ_CONF"])
        if not cr_:
            problems.append("no conf_report in the worker log: the CONF levers' census (calls / reused / elided per lever) and their gate are the evidence")
        else:
            missing = [w for w in want_cf if w not in (cr_.get("applied") or [])]
            if missing:
                problems.append(f"CONF levers not reported applied: {','.join(missing)} (conf_report.applied={cr_.get('applied')})")
            g = cr_.get("gate") or {}
            if not g.get("ok"):
                problems.append(f"CONF levers gate refused: {g.get('reason') or 'no gate in the report'} (census={cr_.get('census')})")
    if env.get("BOLTZ_FPF_MSA"):   # the fused MSA-module kernels' own report (boltz2_opt.msa_kernels.report(): applied / units.<unit>.{census, gate, cfg, tma, triton} / gate / variant), written by the attach hook
        from . import msa_kernels as _MK
        mr_ = worker_log.get("msa_report") or {}; ev["msa_report"] = mr_
        want_units = _MK.units(env); want_levers = [_MK.UNITS[u]["lever"] for u in want_units]
        if not mr_:
            problems.append("no msa_report in the worker log: the fused MSA-module kernels' census per unit (served / fallback_by / errors) and their fail-closed gate are the evidence")
        elif int(n_gpu or 1) > 1:                                     # the row statements own both computations on every rank: the report must NAME that disposition per unit, nothing installed
            ex_ = mr_.get("execution") or {}
            wrong = [u for u in want_units if ex_.get(u) != _MK.REPLACED]
            ev["msa_execution"] = dict(ex_)
            if wrong:
                problems.append(f"fused MSA-module kernels at n_gpu={n_gpu}: units {','.join(wrong)} not reported {_MK.REPLACED} (msa_report.execution={ex_}, applied={mr_.get('applied')}) — the row-sharded trunk owns those computations; an installed kernel there is a defect")
            if mr_.get("applied"):
                problems.append(f"fused MSA-module kernels installed at n_gpu={n_gpu} (msa_report.applied={mr_.get('applied')}): rowpair.REPLACED_LEVERS says they serve no call there")
        else:
            ev["msa_execution"] = dict(mr_.get("execution") or {})
            missing = [n for n in want_levers if n not in (mr_.get("applied") or [])]
            if missing:
                problems.append(f"fused MSA-module kernels not reported applied: {','.join(missing)} (msa_report.applied={mr_.get('applied')})")
            if (mr_.get("variant") or "") != ",".join(want_units):
                problems.append(f"fused MSA-module kernels variant {mr_.get('variant')!r} != the row's BOLTZ_FPF_MSA units {','.join(want_units)!r}")
            g = mr_.get("gate") or {}
            if not g.get("ok"):
                problems.append(f"fused MSA-module kernels gate refused: {g.get('reason') or 'no gate in the report'} (units={ {u: (r or {}).get('census') for u, r in (mr_.get('units') or {}).items()} })")
            elif items:
                for u in want_units:
                    ur = (mr_.get("units") or {}).get(u) or {}; c = ur.get("census") or {}
                    sk = sum(int(v) for v in (ur.get("skipped_by_name") or {}).values())     # calls disposed of by name before launch (the int32 limit): accounted, not idle
                    ev.setdefault("msa_skipped_by_name", {})[u] = dict(ur.get("skipped_by_name") or {})
                    if int(c.get("served") or 0) == 0 and sk == 0:
                        problems.append(f"fused MSA-module kernel unit {u} installed but served no call over {len(items)} item(s) (msa_report.units.{u}.census={c})")
    if env.get("BOLTZ_WRITER"):     # the background writer's own report (boltz2_opt.writer.report(): installed / served / failed / fallback_by / gate), written by the attach hook — joined, every write / move landed, none failed, no fallback, the writer process of the zygote's lineage
        wr = worker_log.get("writer_report") or {}; ev["writer_report"] = wr
        if not wr:
            problems.append("no writer_report in the worker log: the background writer's census (served / failed / fallback_by / bytes_written) and its fail-closed gate are the evidence")
        elif wr.get("installed"):
            g = wr.get("gate") or {}
            if not g.get("ok"):
                problems.append("writer_overlap: " + ("; ".join(g.get("why") or []) or "gate refused") + f" (served={wr.get('served')}/{wr.get('submitted')} moves={wr.get('moves_done')}/{wr.get('moves')} failed={wr.get('failed')} fallback={wr.get('fallback')}"
                                + (f" fallback_by={wr.get('fallback_by')}" if wr.get("fallback_by") else "") + (f" errors={list(wr.get('errors') or [])[:2]}" if wr.get("errors") else "") + ")")
        elif not wr.get("disposition"):
            problems.append("writer_overlap: BOLTZ_WRITER names the lever but it neither installed nor named a disposition")
    if env.get("BOLTZ_PREFETCH"):   # the persistent featurizer's own report (boltz2_opt.prefetch.report(): installed / served / fallback_by / bit_compare / gate), written by the attach hook
        pr = worker_log.get("prefetch_report") or {}; ev["prefetch_report"] = pr
        if not pr:
            problems.append("no prefetch_report in the worker log: the persistent featurizer's census (served / fallback_by / forks_avoided / bit_compare) and its fail-closed gate are the evidence")
        elif pr.get("installed"):
            g = pr.get("gate") or {}
            if not g.get("ok"):
                problems.append(f"persistent featurizer gate refused: {','.join(g.get('why') or []) or 'no gate in the report'} (served={pr.get('served')} fallback_by={pr.get('fallback_by')} bit_compare={pr.get('bit_compare')})")
        elif not pr.get("disposition"):
            problems.append(f"persistent featurizer neither installed nor disposed of by name (prefetch_report={ {k: pr.get(k) for k in ('installed', 'disposition', 'switch')} })")
    if env.get("BOLTZ_XL"):   # the adapter's own report (boltz2_opt.big.report(): applied / stats / env / settings + the library's record, per-unit census and exit gate), written by the attach hook
        from . import registry
        xr = worker_log.get("xl_report") or {}; ev["xl_report"] = xr
        want = [n for n in row["levers"] if n in registry.MEMORY_LEVERS]                  # the memory levers of the row (registry names; the adapter reports registry names)
        if not xr:
            problems.append("no xl_report in the worker log: the memory adapter's own accounting (applied levers, per-lever call counters, the per-unit census) is the evidence")
        else:
            applied = list(xr.get("applied") or [])
            missing = [n for n in want if n not in applied]
            if missing:
                problems.append(f"memory levers not reported applied: {','.join(missing)} (xl_report.applied={applied}{'; ' + str(xr['error']) if xr.get('error') else ''})")
            stray = [n for n in applied if n not in want]
            if stray:
                problems.append(f"memory levers applied outside the row: {','.join(map(str, stray))} (row levers {want})")
            st = xr.get("stats") or {}
            if items:
                replaced = set(((worker_log.get("tp_report") or {}).get("replaced_levers") or [])) if int(n_gpu or 1) > 1 else set()
                elided = bool(env.get("BOLTZ_PAIRFUSE")) and _templ_all_elided(worker_log)   # xl_trans's callers: the C=128 stacks are the layer driver's (superseded_by=pairfuse@c128) and the
                if elided and not st.get(XL_ACTIVITY["xl_trans"]):                            # template stack's transitions were elided with the module by templ_skip on every pass of the run —
                    ev["xl_trans_idle_by"] = "pairfuse@c128+templ_skip"                        # nothing was left to chunk, by name (a templated pass runs the stack and the lever chunks it as before)
                idle = [n for n in want if n in XL_ACTIVITY and not st.get(XL_ACTIVITY[n]) and n not in replaced and not (n == "xl_trans" and elided)]
                ev["tp_replaced_levers"] = sorted(n for n in want if n in replaced)     # named: at n_gpu > 1 the row-sharded statements stand where these unit levers act
                if idle:
                    problems.append(f"memory levers never acted although items ran (size gate BOLTZ_XL_MIN_TOKENS={env.get('BOLTZ_XL_MIN_TOKENS')}): {','.join(idle)} (stats={st})")
            rec = xr.get("record") or {}
            if rec.get("refused"):
                problems.append("memory levers refused at apply: " + "; ".join(f"{r.get('lever')}: {r.get('precondition')}: {r.get('reason')}" for r in rec["refused"]))
            ex = xr.get("exit") or {}
            if not ex:
                problems.append("no exit gate in xl_report: the adapter's census verdict (every unit lever ran on every unit) is the evidence")
            elif ex.get("partial"):
                replaced = set(((worker_log.get("tp_report") or {}).get("replaced_levers") or [])) if int(n_gpu or 1) > 1 else set()
                partial = [n for n in ex["partial"] if n not in replaced]          # a lever the row statements replace at n_gpu > 1 marks no unit by construction (named in tp_replaced_levers)
                if partial:
                    why = "; ".join(f"{n}: {(ex.get('reasons') or {}).get(n) or 'no reason recorded'}" for n in partial)
                    problems.append(f"memory levers partial per the adapter's census (exit gate refused): {why}")
            if env.get("PYTORCH_CUDA_ALLOC_CONF") and (xr.get("env") or {}).get("PYTORCH_CUDA_ALLOC_CONF") != env["PYTORCH_CUDA_ALLOC_CONF"]:
                problems.append(f"allocator setting not in the worker process (xl_report.env.PYTORCH_CUDA_ALLOC_CONF={(xr.get('env') or {}).get('PYTORCH_CUDA_ALLOC_CONF')!r})")
            elif env.get("PYTORCH_CUDA_ALLOC_CONF") and str((xr.get("env") or {}).get("alloc_effective")) == "false":
                problems.append("allocator setting carried in the environment but not effective in the worker process (xl_report.env.alloc_effective=false: the allocator read-back says expandable_segments is not in force)")
    if items:
        modes_seen = {it.get("graph_sampler_mode") for it in items}; replays = {it.get("graph_n_replay") for it in items}
        cwc = [it.get("hoist_captured_with_cache") for it in items]
        ev["n_replay"] = sorted(r for r in replays if r is not None)[0] if any(r is not None for r in replays) else None
        ev["captured_with_cache"] = min((c for c in cwc if c is not None), default=None)
        ev["released"] = sum(1 for it in items if it.get("graph_released"))                     # items whose sample() released the step graph + hoist cache at its return (boltz_graph_patch.RELEASE_MIN_TOKENS)
        ev["cache_released"] = sum(1 for it in items if it.get("hoist_cache_released"))
        rmt = {it.get("graph_release_min_tokens") for it in items if it.get("graph_release_min_tokens") is not None}
        ev["release_min_tokens"] = sorted(rmt)[0] if rmt else None
        tgated = [it for it in items if it.get("hoist_token_gated")]                                        # predictions above the memory row's token ceiling (modes.BIG_SAMPLER_CEILINGS): the
        ev["token_gated"] = len(tgated)                                                                     # sampler group stepped aside by name for them (stock eager loop + the fused step) — theirs by rule
        mts = {it.get("hoist_max_tokens") for it in items if it.get("hoist_max_tokens")}
        ev["sampler_max_tokens"] = sorted(mts)[0] if mts else ((worker_log.get("sampler_report") or {}).get("levers") or {}).get("rollout", {}).get("max_tokens")
        hgated = [it for it in items if it.get("graph_headroom_gated") or it.get("hoist_headroom_gated")]    # predictions the sampler levers' memory-headroom gate sent to the stock
        ev["headroom_gated"] = len(hgated)                                                                  # sampler (boltz_dit_hoist.headroom_gate): theirs by rule, exempt below
        gated = hgated + [it for it in tgated if it not in hgated]                                          # both gates' predictions are exempt from the group's per-item census below
        last = (gated or items)[-1]
        ev["headroom_last"] = {"projected_gib": last.get("graph_projected_gib", last.get("hoist_projected_cache_gib")), "free_gib": last.get("graph_free_gib", last.get("hoist_free_gib"))}
        if env.get("BOLTZ_GRAPH_DIFFUSION") == "graph":
            from . import settings as _settings
            S = int(sampling_steps if sampling_steps is not None else _settings.stock_defaults()["sampling_steps"])
            bad = [it for it in items if it not in gated and (it.get("graph_sampler_mode") != "graph" or it.get("graph_n_replay") != S - 1)]   # step 0 eager, steps 1..S-1 replayed (boltz_graph_patch)
            if bad:
                problems.append(f"{len(bad)}/{len(items)} items without sampler_mode=graph and n_replay={S - 1} (sampling_steps {S}; modes {sorted(map(str, modes_seen))}, replays {sorted(map(str, replays))})")
        if env.get("BOLTZ_DIT_HOIST") and not env.get("BOLTZ_SAMPLER_ROLLOUT"):   # hoist over the per-step graph patch: its capture counters are the evidence
            bad = [it for it in items if it not in gated and ((it.get("hoist_captured_with_cache") or 0) < 1 or (it.get("hoist_capture_stock_fallbacks") or 0) != 0)]
            if bad:
                problems.append(f"{len(bad)}/{len(items)} items without captured_with_cache>=1 and capture_stock_fallbacks=0 (hoist)")
        elif env.get("BOLTZ_DIT_HOIST"):        # hoist over the roll-out [SAMPLER]: the hoist's caches are read by the roll-out's boundary graph (captured per sample()); its
            # capture counters are the graph patch's and stay 0 — the roll-out's own report carries captures / replays per sample (sampler_report; its gate checked above), the hoist's
            # per-item counters that still apply are its stock fallbacks
            bad = [it for it in items if it not in gated and (it.get("hoist_capture_stock_fallbacks") or 0) != 0]
            if bad:
                problems.append(f"{len(bad)}/{len(items)} items with capture_stock_fallbacks!=0 (hoist over the roll-out)")
            sr = worker_log.get("sampler_report") or {}; st = ((sr.get("module") or {}).get("stats") or {})
            from . import settings as _settings
            S = int(sampling_steps if sampling_steps is not None else _settings.stock_defaults()["sampling_steps"])
            n_scoped = sum(int(n) for k, n in (st.get("scope") or {}).items() if k != "above_max_tokens")   # calls the stock sampler served BY NAME (above_max_tokens: counted as gated items above) (steering / force=true constrained inputs): no capture for those
            n_in = len([it for it in items if it not in gated]) - n_scoped
            want_k = {"aligncap": "aligncap", "jacobi64": "device"}.get(env.get("BOLTZ_SAMPLER_ALIGN") or "", "torch")   # the row's rigid-alignment seam as the roll-out reports it (sampler stats.kabsch): the bitwise gesvd seam / the in-graph Kabsch / stock torch
            if env.get("BOLTZ_SAMPLER_ROLLOUT") == "graph" and n_in > 0 and st.get("kabsch") != want_k:
                problems.append(f"roll-out alignment: kabsch={st.get('kabsch')} but the row's BOLTZ_SAMPLER_ALIGN={env.get('BOLTZ_SAMPLER_ALIGN') or '-'} wants {want_k} (the alignment lever is not serving)")
            if env.get("BOLTZ_SAMPLER_ROLLOUT") == "graph" and n_in > 0 and (st.get("captures", 0) < n_in or st.get("replays", 0) < n_in * (S - 1)):
                problems.append(f"roll-out: captures={st.get('captures', 0)} replays={st.get('replays', 0)} for {n_in} item(s) in scope at {S} steps (want >= {n_in} captures and {n_in * (S - 1)} replays; scope={st.get('scope')})")
    ev["safe_nets"] = safe_net_findings(mode, worker_log, stdout_text)   # a core lever's SAFE settings net engaged in this pass (partial activation, judged by partial_activation)
    r5 = r5a2_findings(worker_log)                               # any bit-compared replica census in the agreed per-class shape
    if r5:
        ev["r5a2"] = r5; problems.extend(f"msa2 lock: {x}" for x in r5)
    return ev, problems


def _expand_levers(spec: str) -> List[str]:
    """The trunk lever names of a BOLTZ_LEVERS value (the rows name levers literally: `resid,mask2`)."""
    return [t.strip().lower() for t in spec.replace(";", ",").split(",") if t.strip()]


def rank_dir(kit_dir: str, rank: int) -> str:
    """Rank ``rank`` > 0's own directory of an n_gpu run, under the launch's kit directory: ``<kit_dir>/ranks/rank<r>``."""
    return os.path.join(kit_dir, "ranks", f"rank{int(rank)}")


def worker_log_path(kit_dir: str, tag: str) -> str:
    """The kit worker's own log of a launch (``<kit_dir>/<tag>_worker_log.json``; bz_worker_lev.py writes it, evidence() reads it)."""
    return os.path.join(kit_dir, f"{tag}_worker_log.json")


def rank_batch(batch_json: str, rank: int) -> str:
    """The batch json of rank ``rank`` > 0 of an n_gpu run: the same items and settings, ``out_dir`` = ``kit_dir`` = rank_dir(kit_dir, r) (every
    rank runs the whole worker and writes its own outputs and worker log there; rank 0's are the run's). Written beside the rank-0 batch json."""
    b = json.load(open(batch_json))
    b["out_dir"] = b["kit_dir"] = rank_dir(b["kit_dir"], rank)
    os.makedirs(b["out_dir"], exist_ok=True)
    path = batch_json[:-5] + f".rank{int(rank)}.json" if batch_json.endswith(".json") else batch_json + f".rank{int(rank)}"
    with open(path, "w") as fh:
        json.dump(b, fh, indent=1)
    return path


def rank_logs(batch_json: str, n_gpu: int) -> Dict[int, Optional[dict]]:
    """Every rank's worker log of an n_gpu run (rank 0: the run's own; rank r: under rank_dir(kit_dir, r)), None where unreadable."""
    b = json.load(open(batch_json))
    out = {0: read_worker_log(worker_log_path(b["kit_dir"], b["tag"]))}
    for r in range(1, int(n_gpu)):
        out[r] = read_worker_log(worker_log_path(rank_dir(b["kit_dir"], r), b["tag"]))
    return out


def run_worker(mode: str, workdir: str, batch_json: str, log_path: str, env: Optional[dict] = None, n_gpu: int = 1,
               settings_word: str = "defaults", num_workers: Optional[int] = None) -> int:
    """Launch the staged worker under the mode's row; stdout+stderr to `log_path`. Returns the worker's exit code. At ``n_gpu = 1``: one
    process, this function's own launch (the single-GPU line). At ``n_gpu = P > 1``: P rank processes by the core's launcher
    (``opt_core.mem.rowpair.launch.run_rank_processes``: rank r on visible device r, the rendezvous / rank variables per rank, fail-fast — a
    rank that exits non-zero takes the others down and the run's code is non-zero); rank 0's transcript is the log, every rank's lands in
    ``<workdir>/ranks/rank<r>.log``; rank r > 0 runs the batch of :func:`rank_batch`."""
    P = int(n_gpu or 1)
    cmd = worker_command(batch_json, mode, P, settings_word, num_workers)
    env = dict(env or child_env(mode, n_gpu=P))
    from .worker_launch import CALLER_CWD_ENV
    env[CALLER_CWD_ENV] = os.getcwd()                           # the invoking directory: the worker (every rank) chdirs there before it parses anything, so relative paths inside
    with open(log_path, "w") as lf:                             # the input YAMLs resolve as under the stock CLI (worker_launch.enter_invoking_dir; the kit directory stays on sys.path)
        lf.write("$ " + " ".join(cmd) + (f"   # x{P} ranks (opt_core.mem.rowpair.launch)" if P > 1 else "") + "\n"); lf.flush()
        if P == 1:
            return subprocess.run(cmd, cwd=workdir, env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
        from opt_core.mem.rowpair import launch as tp_launch
        argv = {0: cmd, **{r: worker_command(rank_batch(batch_json, r), mode, P, settings_word, num_workers) for r in range(1, P)}}
        cwd = os.getcwd()
        try:
            os.chdir(workdir)
            recs = tp_launch.run_rank_processes(P, argv_of=lambda r: argv[r], mode=tp.family().MEMORY_MODE, env=env, log_dir=os.path.join(workdir, "ranks"),
                                                isolate_devices=True, nccl_timeout_s=tp.COLLECTIVE_TIMEOUT_S, run_timeout_s=None, on_line=lambda ln: (lf.write(ln if ln.endswith("\n") else ln + "\n"), lf.flush()))
        except tp_launch.RankFailed as e:
            lf.write(f"\n[boltz2-opt] RANK FAILED: {e}\n")
            rcs = [int(r.get("rc") or 0) for r in getattr(e, "records", []) or []]
            return next((rc for rc in rcs if rc not in (0, -9, -15)), 1)
        finally:
            os.chdir(cwd)
        if len(recs) != P:                                                 # fail-closed: the launcher must have stood exactly P ranks
            lf.write(f"\n[boltz2-opt] NOT ACTIVE: reason=n_gpu_mismatch requested={P} active={len(recs)}: the launcher returned {len(recs)} rank records\n"); return 3
        return max((int(r.get("rc") or 0) for r in recs), default=0)


def output_files(out_dir: str, name: str, seed: int) -> List[str]:
    d = os.path.join(out_dir, "by_seed", name, f"s{seed}")
    return sorted(os.listdir(d)) if os.path.isdir(d) else []
