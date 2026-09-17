"""Activation: pins, GPU equality, the late-activation rule, and the hook route.

`activate(mode)` resolves the mode's line (modes.resolve), gates it — openfold3 at its pin (stock/PINS.json), no conflicting switch
preset by the caller, no hook target imported yet, no model instance, no kit hook already installed through the kit's own PYTHONPATH route — then exports the line's switches BEFORE
anything initialises CUDA, puts the hook directories on sys.path in the add-ons' documented order and executes the first hook file (hooks.run), which chains the
others exactly as on the PYTHONPATH route. The add-ons' finders then apply the levers when OpenFold3's model modules are first imported;
nothing here re-implements a lever. `activate(mode, dry_run=True)` resolves, gates and reports without applying anything (`check`).
The instance counter (a constructor wrap on `OpenFold3` installed from activation on) is what refuses a later `enable()` by name.

The activation report names the line's levers as `levers_requested` and nothing as applied: on the hook route the add-ons apply them
when the target modules are imported, later. `levers_record()` reads what they installed from the add-ons' own records (flags, state
dicts, installed lists) — `status()` and the exit tally carry `levers_applied` / `levers_unavailable` /
`levers_pending`, `partial` and `arm_complete` from it. `weights_gate()` compares the checkpoint with the pinned weights
(stock/PINS.json) once; the CLI refuses or labels the run on its verdict.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

from . import ActivationError, __version__
from . import env as _env
from . import hooks as _hooks
from . import manifest as _manifest
from . import modes
from . import report as _report
from .registry import LEVERS

ENV_MODE, ENV_HOME, ENV_TARGET_GPU = "OPENFOLD3_OB0_OPT", "OPENFOLD3_OB0_OPT_HOME", "MODEL_OPT_TARGET_GPU"
MODEL_MODULE, MODEL_CLASS = _report.MODEL_MODULE, _report.MODEL_CLASS   # stock's OpenFold3 model class (openfold3/projects/of3_all_atom/model.py:56): one definition, report.py's
MIN_CC = 8.0                                                    # the add-ons' stated requirement (sm80+: trunk_kernels/README.md:52); reported, not a gate

_REPORT: Optional[dict] = None
_INSTANCES = {"wrapped": False}                     # the ready-line wrap state; the instance COUNT is opt_core.instances' (instances())
_COUNTER_FINDER = None
DS4SCI_OP = "evoformer_attn"                                    # deepspeed's DS4Sci evoformer-attention op (openfold3/core/model/primitives/attention.py:36-60)
DS4SCI_OPS_RECORD = "git_version_info_installed.py"             # deepspeed's build record beside its __init__: `installed_ops={...}` (written by its setup.py)


def evoformer_attn_op(spec=None) -> Tuple[Optional[bool], str]:
    """Whether deepspeed's DS4Sci evoformer_attn op is INSTALLED (pre-built into the package: the tree's route for upstream's shipped
    configuration) — read from deepspeed's own build record `git_version_info_installed.py`
    (`installed_ops["evoformer_attn"]`) WITHOUT importing deepspeed (no torch import, no CUDA context; the stock caller's proof needs
    both absent). Returns (True | False | None, detail): None when deepspeed is absent or its record is unreadable — the caller treats
    anything but True as `not installed`. A JIT build at first use is not a route of this tree (upstream's hacks.py sets a placeholder
    CUTLASS_PATH and deepspeed then refuses the build; the stock caller strips CUTLASS_PATH)."""
    import ast
    import importlib.util
    try:
        spec = spec or importlib.util.find_spec("deepspeed")
    except (ImportError, ValueError):
        spec = None
    if spec is None or not spec.origin:
        return None, "deepspeed is not installed"
    rec = os.path.join(os.path.dirname(spec.origin), DS4SCI_OPS_RECORD)
    if not os.path.isfile(rec):
        return None, f"deepspeed at {os.path.dirname(spec.origin)} carries no {DS4SCI_OPS_RECORD}"
    ops = version = None
    try:
        with open(rec, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), rec)
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if node.targets[0].id == "installed_ops":
                    ops = ast.literal_eval(node.value)
                elif node.targets[0].id == "version":
                    version = ast.literal_eval(node.value)
    except Exception as e:                                                   # noqa: BLE001 — the detail names the failure; the verdict is `unknown`
        return None, f"{rec} unreadable: {type(e).__name__}: {e}"
    if not isinstance(ops, dict):
        return None, f"{rec} has no installed_ops table"
    got = bool(ops.get(DS4SCI_OP))
    return got, f"deepspeed {version or '?'}: installed_ops[{DS4SCI_OP!r}] = {ops.get(DS4SCI_OP)!r} ({rec})"



DS4SCI_NOT_LOADED = "DS4Sci evoformer attention did not load"     # the words of the load assertion (ds4sci_load; cli / stock_pred exit NOT STOCK / NOT ACTIVE by name on it)


def ds4sci_load(builder=None) -> Tuple[bool, str]:
    """LOAD deepspeed's DS4Sci evoformer_attn op the way upstream's first triangle-attention call does (deepspeed/ops/deepspeed4science/evoformer_attn.py:
    `kernel_ = EvoformerAttnBuilder().load()`), once, before the run: (True, the loaded module's file) or (False, the reason). A configuration that turns
    the kernel on runs only where this returns True — upstream itself would catch the per-item load error, log it and write no structure (or, in other
    versions, fall back to its eager attention); neither is ever timed as stock here. Imports deepspeed and torch (the callers run it after the stock
    proof / after activation, in the process that runs the model)."""
    try:
        if builder is None:
            from deepspeed.ops.op_builder import EvoformerAttnBuilder as builder   # noqa: N813
        mod = builder().load(verbose=False)
    except BaseException as e:  # noqa: BLE001  (a failed JIT build raises RuntimeError / CalledProcessError / SystemExit by deepspeed version)
        return False, f"{type(e).__name__}: {str(e).strip()[:300]}"
    return True, str(getattr(mod, "__file__", None) or getattr(mod, "__name__", "loaded"))

def status() -> dict:
    """The last activation report; when a line is active, its lever fields are the add-ons' own records at the time of the call (`levers_record`)."""
    if not _REPORT:
        return {"active": False, "reason": "openfold3_ob0_opt.enable() has not run in this process"}
    rep = dict(_REPORT)
    if rep.get("active"):
        rep.update(levers_record(rep))
    return rep


def instances() -> int:
    """OpenFold3 instances built in this process since the counter was armed (opt_core.instances' count; 0 before the class exists)."""
    from opt_core import instances as _inst
    return int(_inst.instance_check(MODEL_MODULE, MODEL_CLASS).get("built", 0))


# ------------------------------------------------------------------------------------------------------------------ probes ----
def gpu_probe(environ: Optional[dict] = None) -> dict:
    """opt_core.gates.nvidia_smi_probe (name, cc, sm, memory_mib, probe) plus `supported`: the add-ons' stated requirement, cc >= MIN_CC."""
    from opt_core import gates as _gates                                   # lazily: the stock child imports this module after its proof
    g = dict(_gates.nvidia_smi_probe())
    try:
        g["supported"] = float(g.get("cc")) >= MIN_CC
    except (TypeError, ValueError):
        g["supported"] = None
    return g


# ------------------------------------------------------------------------------------------------------------------- gates ----
def version_gate(home: str) -> Optional[str]:
    pinned = _env.pins(home)["openfold3_version"]
    have = modes.openfold3_version()
    if have is None:
        return f"openfold3 is not installed (pinned {pinned})"
    if have != pinned:
        return f"openfold3 {have} is installed, the pin is {pinned} (stock/PINS.json)"
    return None


def pinned_weights(home: Optional[str] = None) -> Tuple[str, str]:
    """(file name, sha256) of the pinned weights: stock/PINS.json "weights" names the file and carries its digest — the one digest
    this tree keeps (the checkpoint is downloaded at install time, OPENFOLD3_OB0_CKPT, never tracked here)."""
    home = home or _env.tree_home()
    w = _env.pins(home)["weights"]
    return w["file"], w["sha256"]


def cache_root(environ=None) -> str:
    """The package's cache root: `$XDG_CACHE_HOME/openfold3_ob0_opt` (default `~/.cache/openfold3_ob0_opt`). The weights digest memo lives there
    (`weights_digests.json`, digest_memo.MEMO_NAME): a checkpoint's sha256 keyed by (realpath, size, mtime_ns, inode) — those fields select the
    memo entry, they never decide identity."""
    environ = os.environ if environ is None else environ
    return os.path.join(environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache"), "openfold3_ob0_opt")


def weights_gate(ckpt: Optional[str], home: Optional[str] = None, hash_it: bool = True, refresh: bool = False) -> dict:
    """The checkpoint against the pin: the file's sha256 (through the on-disk digest memo at
    cache_root(), digest_memo.digest — `refresh=True` hashes afresh and rewrites the entry, which is what `check` does) compared with the weights of
    record. `is_pinned` is True/False when hashed, None when the file is missing (or `hash_it=False`: `hashed: false`). The CLI prints one WEIGHTS line from this record — `pinned`, or `WARNING: WEIGHTS unknown …; proceeding` on False — and every route runs (cli.weights_check)."""
    home = home or _env.tree_home()
    name, pinned = pinned_weights(home)
    out = {"path": ckpt, "exists": bool(ckpt) and os.path.isfile(ckpt), "bytes": None, "sha256": None, "hashed": False,
           "pinned_file": name, "pinned_sha256": pinned, "is_pinned": None, "digest_cached_utc": None, "digest_memo": None}
    if out["exists"]:
        out["bytes"] = os.path.getsize(ckpt)                                   # recorded, never compared: identity is by digest only
        if hash_it:
            from . import digest_memo
            memo_dir = cache_root()
            out["digest_memo"] = os.path.join(memo_dir, digest_memo.MEMO_NAME)
            hasher = lambda p: _manifest.sha256_file(p, strict=True)   # noqa: E731
            unwritable = memo_dir_unwritable(memo_dir)
            if unwritable is None:
                try:
                    out["sha256"], out["digest_cached_utc"] = digest_memo.digest(ckpt, memo_dir, refresh=refresh, hasher=hasher)
                except OSError as e:                                       # the memo file could not be written (a read-only or vanished cache root): named, and the digest is taken afresh below — never a refusal
                    if not str(getattr(e, "filename", "") or "").startswith(memo_dir):
                        raise                                              # an error reading the checkpoint itself is the caller's
                    unwritable = f"{type(e).__name__}: {e}"
            if unwritable is not None:
                out["digest_memo_unwritable"] = unwritable
                sys.stderr.write(f"{_report.PREFIX} WEIGHTS digest memo not written: {out['digest_memo']} ({unwritable}); hashing afresh\n"); sys.stderr.flush()
                out["sha256"], out["digest_cached_utc"] = hasher(os.path.realpath(ckpt)), None
            out["hashed"] = True
            out["is_pinned"] = out["sha256"] == pinned
    return out


def memo_dir_unwritable(memo_dir: str) -> Optional[str]:
    """None when the digest memo's directory exists (created here) and is writable; else the reason, named (the WEIGHTS line says the memo
    was not written and the checkpoint is hashed afresh — an unwritable cache root is never a refusal)."""
    try:
        os.makedirs(memo_dir, exist_ok=True)
    except OSError as e:
        return f"{type(e).__name__}: {e}"
    if not os.access(memo_dir, os.W_OK | os.X_OK):
        return "read-only directory"
    return None


def late_activation_gate() -> Optional[str]:
    t = _hooks.targets_imported()
    if t:
        return f"late activation: {', '.join(t)} already imported — the kit hooks apply at that import (activate before it)"
    if instances() > 0:
        return f"late activation: {instances()} {MODEL_CLASS} instance(s) already built in this process"
    pre = _env.kit_hooks_installed()                                   # the tree's own hook directories (env.kit_dirs), judged by file
    if pre:
        return f"a kit hook is already installed in this process ({', '.join(pre)} on sys.meta_path: the kit's own PYTHONPATH route); one route per process"
    return None


# ---------------------------------------------------------------------------------------------------------- lever evidence ----
# What the add-ons installed, read from their own records (never from the mode table): the fast-inference add-on's flags and state dict, the
# trunk-kernels add-on's patched methods and flash-dispatch flag, the cells' and the ports' own records. A lever is `pending` until its target module (registry.Lever.target) has been imported, `applied` when the
# kit's record says so afterwards, `unavailable` otherwise (the kit fell back to the stock path and said so on its own stderr).
def _mod(name: str):
    return sys.modules.get(name)


def _code_file(fn) -> Optional[str]:
    code = getattr(fn, "__code__", None)
    return getattr(code, "co_filename", None) if code is not None else None


def _under(path: Optional[str], d: Optional[str]) -> bool:
    """`path` is `d` or inside it — the core's ONE path-containment predicate (opt_core.stock_proof._under: realpath both sides); None-safe."""
    if not path or not d:
        return False
    from opt_core.stock_proof import _under as _core_under
    return _core_under(path, [os.path.realpath(d)])


def _patched_from(hook_dir: str, module_name: str, attr_path: str) -> Optional[bool]:
    """True when `module.attr` is now a function whose code lives under `hook_dir` (the kit's patch in place on the stock class)."""
    m = _mod(module_name)
    if m is None:
        return None
    obj = m
    for a in attr_path.split("."):
        obj = getattr(obj, a, None)
        if obj is None:
            return False
    return _under(_code_file(obj), hook_dir)


def _p_fast_init(home: str) -> Optional[bool]:
    m = _mod("of3_fastinit")
    return None if m is None else bool(getattr(m, "_ENABLED", False))                       # of3_fastinit.py:11,33


def _p_cuda_graphs(home: str) -> Optional[bool]:
    m = _mod("of3_graphs")
    st = getattr(m, "_STATE", None) if m is not None else None
    return None if not isinstance(st, dict) else bool(st.get("enabled"))                      # of3_graphs.py:26


def _p_graphs_strict(home: str) -> Optional[bool]:
    g = _p_cuda_graphs(home)
    return None if g is None else bool(g and os.environ.get("OF3_GRAPHS_STRICT") == "1")     # of3_graphs.py:276 reads it live


def _line_hook_dir(home: str, kit: str) -> str:
    """The hook directory `kit` runs from on the ACTIVE line (the report's hook_dirs: an errata copy where the line names one), else the kit's own."""
    rep = _REPORT or {}
    for k, d in zip(rep.get("hooks") or [], rep.get("hook_dirs") or []):
        if k == kit:
            return d
    return modes.hook_dir(home, kit)


def _p_trunk(home: str, module_name: str, attr_path: str) -> Optional[bool]:
    if _mod("of3t_levers") is None:
        return None
    return _patched_from(_line_hook_dir(home, "trunk_kernels"), module_name, attr_path)


def _p_templ_distinct(home: str) -> Optional[bool]:
    wrapped = _p_trunk(home, "openfold3.core.model.latent.template_module", "TemplatePairStack.forward")
    if wrapped is None:
        return None
    return bool(wrapped and os.environ.get("OF3T_TEMPL_DISTINCT") == "1")                     # of3t_levers.py:150,169: the wrap is unconditional, the switch is read at install


def _p_paircache(home: str) -> Optional[bool]:
    if _mod("of3t_paircache") is None:
        return None
    return _patched_from(_line_hook_dir(home, "trunk_kernels"), "openfold3.core.model.layers.diffusion_conditioning", "DiffusionConditioning.forward")


def tp_state() -> Optional[dict]:
    """The multi-GPU line's in-process record: the rank's process group (world, rank, backend, ready) — the core adapter's
    (openfold3_ob0_opt.tp_rowpair.model.GROUP, line `tp`); None in a process that is not a rank (the launcher's printed census is the
    run's record)."""
    m = sys.modules.get("openfold3_ob0_opt.tp_rowpair.model")
    rec = getattr(m, "GROUP", None) if m is not None else None
    return dict(rec) if rec else None


def _p_tp_shard(home: str) -> Optional[bool]:
    """The tensor-parallel line's own record: the adapter installed its seams and this rank joined a P > 1 group (the sharded pair stack ran);
    None before the adapter's model module is imported (before the runner imports)."""
    m = sys.modules.get("openfold3_ob0_opt.tp_rowpair.model")
    if m is None:
        return None
    rec = getattr(m, "GROUP", None) or {}
    return bool(int(rec.get("world", 0) or 0) > 1 and rec.get("group_ready") and m.PATCHES.names())


def _p_sample_loop(home: str) -> Optional[bool]:
    """The sample_loop lever's record (``openfold3_ob0_opt.sample_loop.LAST``: the census fields of the last roll-out it drove); None before a roll-out ran."""
    m = _mod("openfold3_ob0_opt.sample_loop")
    if m is None or not getattr(m, "LAST", None):
        return None
    return True


def _p_tp_triatt(home: str) -> Optional[bool]:
    """The tp_triatt lever's record (``openfold3_ob0_opt.tp_rowpair.pairstack.TRIATT``): a kernel word other than torch bound to the row-sharded pair stack's
    TriangleAttention modules (the core dispatch constructed: flash_triattn / cueq by name, or the tier door's word on its capabilities); None before a pair block was bound (before the trunk ran)."""
    m = sys.modules.get("openfold3_ob0_opt.tp_rowpair.pairstack")
    rec = getattr(m, "TRIATT", None) if m is not None else None
    if not rec or not rec.get("bound"):
        return None
    from .tp_rowpair.env import TRIATT_DOOR_WORD               # on 9.0 / 8.0 the lever's word is the core's tier door (env.triatt_word); this proof runs for the tp line only
    return rec.get("kernel") in ("cueq", "flash_triattn", TRIATT_DOOR_WORD)


def _p_tp_trimul(home: str) -> Optional[bool]:
    """The tp_trimul lever's record (``openfold3_ob0_opt.tp_rowpair.pairstack.TRIMUL``): the core's fused provider bound to the row-sharded pair stack's
    TriangleMultiplication modules; None before a pair block was bound."""
    m = sys.modules.get("openfold3_ob0_opt.tp_rowpair.pairstack")
    rec = getattr(m, "TRIMUL", None) if m is not None else None
    if not rec or not rec.get("bound"):
        return None
    return rec.get("kernels") == "fpf_v4"


def _p_off(check):
    """A probe on the offload port's record (of3_offload: the module the `big` lines' entry hook applies after the model module executes)."""
    def probe(home: str) -> Optional[bool]:
        m = _mod("of3_offload")
        if m is None or not getattr(m, "_APPLIED", False):
            return None if m is None else False
        try:
            return bool(check(m))
        except Exception:
            return False
    return probe


def _reachable(current, target, _seen=None) -> bool:
    """`target` is `current` itself, or something `current` calls through: `current`'s `__wrapped__` chain (functools.wraps) or, absent
    that, whatever plain-closure cells `current` captured (the tree's own timer/census wrappers close over the function they time —
    `orig = getattr(cls, meth); setattr(cls, meth, _timed(name, orig, ...))` — without functools.wraps; a closure cell is the only
    trace). An external caller's phase-timing wrapper on `run_trunk`/`.forward` etc. (installed after a kit's own hook, calling through to what
    it wraps) must not read as \"my patch is gone\" here — it is still live, just one call frame further in. Walks any depth of nested
    wrapping; a function unrelated to `target` (wraps something else entirely) correctly returns False, not a false positive."""
    seen = _seen if _seen is not None else set()
    if current is None or id(current) in seen:
        return False
    seen.add(id(current))
    if current is target:
        return True
    w = getattr(current, "__wrapped__", None)
    if w is not None and _reachable(w, target, seen):
        return True
    for cell in getattr(getattr(current, "__func__", current), "__closure__", None) or ():
        try:
            val = cell.cell_contents
        except ValueError:
            continue                                                                            # an empty cell (the closure var was never assigned): nothing to follow
        if callable(val) and _reachable(val, target, seen):
            return True
    return False


def _is_fn(cls_path: str, attr: str, fn) -> bool:
    """True when `cls_path.attr` IS `fn`, or wraps it transparently (`_reachable`) -- the kit's own patch is live either way."""
    modname, cls = cls_path.rsplit(".", 1)
    c = getattr(_mod(modname), cls, None)
    if c is None:
        return False
    bound = getattr(c, attr, None)
    return bound is fn or _reachable(bound, fn)


def _fn_bound_desc(cls_path: str, attr: str) -> str:
    """A one-token, log-line-safe `<module>.<qualname>` of whatever `cls_path.attr` currently is, for the `unavailable:<cause>` reason
    when `_is_fn` says no -- names whatever DOES hold the slot (upstream's own, unpatched -- its own module.qualname reveals that
    plainly; a different kit; or the caller, if it replaced rather than transparently wrapped, no `__wrapped__` / no closure cell
    reaching this kit's function) -- or `absent` when the class/attribute itself could not be resolved."""
    modname, cls = cls_path.rsplit(".", 1)
    c = getattr(_mod(modname), cls, None)
    if c is None:
        return "absent"
    bound = getattr(c, attr, None)
    if bound is None:
        return "absent"
    owner = getattr(bound, "__module__", "?")
    qual = getattr(bound, "__qualname__", getattr(bound, "__name__", "?"))
    return f"{owner}.{qual}"


def _p_conf_chunked(m) -> Optional[bool]:
    conf = (m.STATE or {}).get("conf")
    if not conf:
        return None                                                                              # the confidence head has not run yet
    return conf.get("path") == "chunked" and not (getattr(_mod("of3o_confidence"), "CENSUS", {}) or {}).get("native_fallback_elements")


def _p_alloc(m) -> Optional[bool]:
    """The expandable-segments allocator as EFFECTIVE — the tree's shared probe (opt_core.mem.torch_alloc.effective: the caching allocator's
    own record torch.cuda.memory._snapshot()["allocator_settings"]["expandable_segments"]). True only when that record says so; False when the
    setting is absent from the environment or the record is unreadable (an unreadable record is a refusal, never a pass); None before torch
    is imported / CUDA is initialised (pending)."""
    want = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments:True" not in want:
        return False
    if "torch" not in sys.modules:
        return None
    from opt_core.mem.torch_alloc import effective
    return effective().get("expandable")


def _p_activation_lever(module: str):
    """A probe on an activation-installed package lever's record (openfold3_ob0_opt.<module>.STATE, TARGET): True once the record says on
    (installed and serving), False when the lever's target module was imported and it is not, None before."""
    def probe(home: str) -> Optional[bool]:
        m = _mod("openfold3_ob0_opt." + module)
        if m is None:
            return None
        st = getattr(m, "STATE", None) or {}
        if st.get("installed") and st.get("state") == "on":
            return True
        return False if m.TARGET in sys.modules else None
    return probe


def _p_fastjson(home: str) -> Optional[bool]:
    """A probe on the fastjson lever's record (openfold3_ob0_opt.fastjson.STATE): True once the writer's encoder class is the lever's (state on),
    False when the writer module was imported and the record says refused / armed, None before."""
    m = _mod("openfold3_ob0_opt.fastjson")
    if m is None:
        return None
    st = getattr(m, "STATE", None) or {}
    if st.get("installed") and st.get("state") == "on":
        return True
    return False if m.TARGET in sys.modules else None


def _p_conf_dtype(home: str) -> Optional[bool]:
    """A probe on the confidence-dtype lever's record (openfold3_ob0_opt.conf_dtype.STATE / its AttrPatch): True once the auxiliary heads'
    forward is the lever's wrapper (or, in a tp rank, once the word is bf16 — the rank's own statement reads it), False when the heads module is
    imported unwrapped, None before."""
    m = _mod("openfold3_ob0_opt.conf_dtype")
    if m is None:
        return None
    hm = sys.modules.get(m.TARGET)
    if hm is not None:                                                           # the record of truth: the heads' forward IS the lever's wrapper (or is not)
        return bool(getattr(getattr(getattr(hm, "AuxiliaryHeadsAllAtom", None), "forward", None), "_of3ob0_conf_dtype", False))
    st = getattr(m, "STATE", None) or {}
    if st.get("installed") and (os.environ.get("OF3TP_RANK") or "").strip():
        return True                                                              # a tp rank never imports the stock heads' forward path for the phase: the word is the record
    return None


def _p_z_dtype(home: str) -> Optional[bool]:
    """A probe on the hand-off-dtype lever's record (openfold3_ob0_opt.z_dtype.STATE / its AttrPatch): True once OpenFold3.run_trunk is the
    lever's wrapper (or, in a tp rank / under the resident port, once the word is bf16 — those routes' own trunk statements read it), False
    when the model module is imported unwrapped under bf16, None before."""
    m = _mod("openfold3_ob0_opt.z_dtype")
    st = getattr(m, "STATE", None) if m is not None else None
    if not st or not st.get("installed"):
        return False if (m is not None and m.TARGET in sys.modules and (st or {}).get("word") == "bf16") else None
    mm = sys.modules.get(m.TARGET)
    if mm is None or (os.environ.get("OF3TP_RANK") or "").strip() or (os.environ.get("OF3O_LAYER") or "").strip():
        return True                                                              # the row-sharded / resident trunks hand off through z_dtype.handoff: the word is the record
    fn = getattr(getattr(mm, "OpenFold3", None), "run_trunk", None)
    return None if fn is None else bool(getattr(fn, "_of3ob0_z_dtype", False) or getattr(getattr(fn, "__wrapped__", None), "_of3ob0_z_dtype", False))


def _p_rollout(home: str) -> Optional[bool]:
    """A probe on the bf16-rollout lever's record (openfold3_ob0_opt.cells.rollout.STATE)."""
    m = _mod("openfold3_ob0_opt.cells.rollout")
    st = getattr(m, "STATE", None) if m is not None else None
    if st and st.get("installed"):
        return True
    return False if MODEL_MODULE in sys.modules else None


def _p_dit_attn(home: str) -> Optional[bool]:
    """A probe on the DiT attention lever's record (openfold3_ob0_opt.cells.dit_attn.STATE)."""
    m = _mod("openfold3_ob0_opt.cells.dit_attn")
    st = getattr(m, "STATE", None) if m is not None else None
    if st and st.get("installed"):
        return True
    return False if MODEL_MODULE in sys.modules else None


def _p_cell(module: str, want_on: bool = False):
    """A probe on a cell lever's own record (openfold3_ob0_opt.cells.<module>.STATE): True once installed (and, want_on, still serving — a lever that
    refused itself by name after install is unavailable), False when the model module is imported and it is not, None before."""
    def probe(home: str) -> Optional[bool]:
        m = _mod("openfold3_ob0_opt.cells." + module)
        st = getattr(m, "STATE", None) if m is not None else None
        if st and st.get("installed") and (not want_on or st.get("state") == "on"):
            return True
        return False if MODEL_MODULE in sys.modules else None
    return probe


def _p_pairfused(lever: str):
    """A probe on the pair-track cells' record (openfold3_ob0_opt.cells.pairfused.STATE): True once the lever is installed on PairBlock (the hook applied it
    after the model module executed), False when the model module is imported and the lever is not among the installed ones, None before."""
    def probe(home: str) -> Optional[bool]:
        m = _mod("openfold3_ob0_opt.cells.pairfused")
        st = getattr(m, "STATE", None) if m is not None else None
        if st and st.get("installed"):
            return lever in (st.get("levers") or [])
        return False if MODEL_MODULE in sys.modules else None
    return probe


def _p_trimulprovider(home: str) -> Optional[bool]:
    """The TriMul cell's row per class is the core's provider's (openfold3_ob0_opt.cells.pairfused.STATE["trimul_provider"] set at install): True once
    trimul_v4 is installed with the router, False when the model module is imported and it is not, None before."""
    m = _mod("openfold3_ob0_opt.cells.pairfused")
    st = getattr(m, "STATE", None) if m is not None else None
    if st and st.get("installed"):
        return "trimul_v4" in (st.get("levers") or []) and st.get("trimul_provider") is not None
    return False if MODEL_MODULE in sys.modules else None


def _p_pairprovider(home: str) -> Optional[bool]:
    """The fused triangle-attention block's core is the core's provider (openfold3_ob0_opt.cells.pairfused.STATE["provider"] set at install): True once
    triatt_block is installed with the provider core, False when the model module is imported and it is not, None before."""
    m = _mod("openfold3_ob0_opt.cells.pairfused")
    st = getattr(m, "STATE", None) if m is not None else None
    if st and st.get("installed"):
        return "triatt_block" in (st.get("levers") or []) and st.get("provider") is not None
    return False if MODEL_MODULE in sys.modules else None


def _p_confhead(home: str) -> Optional[bool]:
    """The row-block confidence head: the heads' forward in this process is confhead's (openfold3_ob0_opt.confhead.installed(): None before
    `openfold3.core.model.heads.prediction_heads` is imported, False when it is and the forward is stock's) AND, once the confidence phase has
    run (the port's STATE["conf"]), at least one row block went through it (confhead.census()["blocks"]) — pending until then."""
    from . import confhead as _confhead                                                          # torch at import: inside the probe, never at module level (the CPU tests import stack without torch)
    inst = _confhead.installed()
    if not inst:
        return inst
    m = _mod("of3_offload")
    if m is None or not (getattr(m, "STATE", None) or {}).get("conf"):
        return None
    return int(_confhead.census().get("blocks", 0)) > 0


# One row per lever whose "applied" check is "does of3_offload's own function occupy this class attribute" (_is_fn) -- (cls_path, attr,
# the offload-module attribute name the port's function lives under). The single source for both OFFLOAD_PROBES' boolean checks below
# and _offload_fn_detail's `unavailable:<cause>` diagnostic -- one row edited once, both stay in sync.
_OFFLOAD_FN_TARGETS = {
    "trimul_hostsnap": ("openfold3.core.model.layers.triangular_multiplicative_update.TriangleMultiplicativeUpdate", "_inference_forward", "trimul_inference_forward"),
    "triatt_lean": ("openfold3.core.model.latent.base_blocks.PairBlock", "tri_att_start_end", "pairblock_tri_att_start_end"),
    "trans_inplace": ("openfold3.core.model.latent.base_blocks.PairBlock", "forward", "pairblock_forward"),
    "cond_once": ("openfold3.core.model.layers.diffusion_conditioning.DiffusionConditioning", "forward", "diffusion_conditioning_forward"),
    "input_rows": ("openfold3.core.model.feature_embedders.input_embedders.InputEmbedderAllAtom", "forward", "input_embedder_forward"),
    "recycle_rows": ("openfold3.projects.of3_all_atom.model.OpenFold3", "run_trunk", "run_trunk"),
    "templ_host": ("openfold3.core.model.latent.template_module.TemplateEmbedderAllAtom", "forward", "template_embedder_forward"),
}


def _offload_fn_detail(name: str) -> Optional[str]:
    """`<cause>` for `reason=unavailable:<cause>` on an offload lever in `_OFFLOAD_FN_TARGETS`: what actually occupies the class
    attribute right now, so a false-vs-true "unavailable" is legible from the log line alone -- `openfold3....run_trunk` (upstream's
    own, never patched: a genuine miss, worth investigating why this install didn't run) reads very differently from
    `some_caller.timed_run_trunk` (a non-transparent external wrapper won the slot: still worth fixing, but not this kit's bug) or
    `tp.model_s._run_trunk_patched` (a different kit's line, e.g. a mode/line mismatch). None for a lever _is_fn doesn't cover
    -- callers fall back to the plain reason in that case."""
    t = _OFFLOAD_FN_TARGETS.get(name)
    if t is None:
        return None
    cls_path, attr, _ = t
    return _fn_bound_desc(cls_path, attr)


OFFLOAD_PROBES = {
    **{name: _p_off((lambda cls_path, attr, fn_attr: lambda m: _is_fn(cls_path, attr, getattr(m, fn_attr)))(*t))
       for name, t in _OFFLOAD_FN_TARGETS.items()},
    "conf_chunked": _p_off(_p_conf_chunked),
    "bigln_guard": _p_off(lambda m: bool(getattr(getattr(_mod("openfold3.core.model.primitives.normalization"), "LayerNorm", None), "_of3o_lnsafe", False))),
    "host_pool": _p_off(lambda m: m._PIN_BYTES.get("allocs", 0) > 0),   # the record: the pool's high-water mark (the buffers are freed before the exit tally)
    "alloc_expandable": lambda home: _p_alloc(None),                  # the allocator's own record, on any line that exports the setting (fast, big/resident)
}


def offload_census() -> Optional[dict]:
    """The offload port's record (of3_offload.census()) when its module is loaded: applied flags, every fallback and pageable event, the confidence path."""
    m = _mod("of3_offload")
    if m is None:
        return None
    try:
        return m.census()
    except Exception as e:
        return {"error": repr(e)[:200]}


def offload_partial(c: Optional[dict]) -> List[str]:
    """The named reasons a big process is PARTIAL: any stock-body fallback, any native TM element in the chunked head. (A pageable host buffer — over
    OF3O_PIN_BUDGET_GB, decided up front — is a NOTE line and a count, `offload_pageable=<n>` in the exit tally, never PARTIAL: same numbers, slower copies.)"""
    if not c or "error" in c:
        return [] if not c else [f"offload census unreadable: {c['error']}"]
    out = [f"offload fallback {k} x{v}" for k, v in sorted(c.get("fallbacks", {}).items())]
    n = (c.get("conf_census") or {}).get("native_fallback_elements", 0)
    if n:
        out.append(f"offload conf native TM backend on {n} element(s)")
    return out


_PROBES = {
    **OFFLOAD_PROBES,
    "confhead": _p_confhead,
    "fast_init": _p_fast_init, "cuda_graphs": _p_cuda_graphs, "graphs_strict": _p_graphs_strict,
    "templ_distinct": _p_templ_distinct,
    "paircache": _p_paircache,
    "trimul_v4": _p_pairfused("trimul_v4"), "triatt_block": _p_pairfused("triatt_block"), "pair_transition": _p_pairfused("pair_transition"),
    "rollout_bf16": _p_rollout,
    "dit_attn": _p_dit_attn,
    "dit_glue": _p_cell("dit_glue"), "token_agg": _p_cell("token_agg", want_on=True), "atom_window": _p_cell("atom_window", want_on=True), "atom_hoist": _p_cell("atom_hoist", want_on=True),
    "apb_trunk": _p_cell("apb_trunk"), "templ_embed": _p_cell("templ_embed", want_on=True),
    "castcache": _p_cell("castcache", want_on=True), "exactln": _p_cell("exactln", want_on=True), "ln_provider": _p_cell("ln_provider", want_on=True), "apb_hoist": _p_cell("apb_hoist", want_on=True), "trunk_graph": _p_cell("trunk_graph", want_on=True),
    "triatt_exact": _p_cell("triatt_exact", want_on=True), "trimul_exact": _p_cell("trimul_exact", want_on=True), "trimul_form": _p_cell("trimul_form", want_on=True), "transition_exact": _p_cell("transition_exact", want_on=True), "triatt_provider": _p_pairprovider, "trimul_provider": _p_trimulprovider,
    "post_release": _p_cell("post_release", want_on=True), "tuner_guard": _p_cell("tuner_guard", want_on=True),
    "fastjson": _p_fastjson,
    "writer_overlap": _p_activation_lever("writer_overlap"),
    "hostfeat": _p_activation_lever("hostfeat"),
    "ckpt_mmap": _p_activation_lever("ckpt_mmap"),
    "postfwd_mem": _p_cell("postfwd_mem", want_on=True), "loader_workers": _p_cell("loader_workers", want_on=True),
    "conf_dtype": _p_conf_dtype,
    "z_dtype": _p_z_dtype,
    "tp_shard_s": _p_tp_shard, "sample_loop": _p_sample_loop, "tp_triatt": _p_tp_triatt, "tp_trimul": _p_tp_trimul,
}
CARRIER_MODULES = ("of3_fastinit", "of3_graphs")   # the lever modules a line's carrier (its `of3_levers`) must be the source of


def carrier_dir(res, home: Optional[str] = None) -> Optional[str]:
    """The directory a line's carrier lever modules (CARRIER_MODULES — the fast-inference add-on's `of3_levers`: of3_fastinit, of3_graphs) must be
    loaded from: the line's own `fast_inference` hook directory (`res.hook_dirs` at `res.hooks.index("fast_inference")` — every line that requests
    fast_init / cuda_graphs lists that hook, always last in its chain), None for a line without it. NOT a chain variable: OF3T_KIT_LEVERS /
    OF3O_KIT_LEVERS / the *_CHAIN exports each name a chaining hook's NEXT directory, which is `of3_levers` only where the trunk-kernels hook is the
    one chaining into it — on the `big --n_gpu P` (tp) line (hooks cells > tp_rowpair > fast_inference) the only chain variable,
    OPENFOLD3_OB0_OPT_PAIR_CHAIN, names tp_rowpair/hook, so a carrier read from the chain variables would judge of3_fastinit "off the carrier", the exit tally
    would say PARTIAL / arm_complete=false and a complete tp run would exit rc 3."""
    hooks = list(getattr(res, "hooks", None) or [])
    if "fast_inference" not in hooks:
        return None
    dirs = list(getattr(res, "hook_dirs", None) or [])
    if len(dirs) == len(hooks):
        return dirs[hooks.index("fast_inference")]
    return modes.hook_dir(home or _env.tree_home(), "fast_inference")
ACTIVATION_MODULES = ("templ_guard", "conf_dtype", "z_dtype", "fastjson", "writer_overlap", "hostfeat", "ckpt_mmap")  # package modules the activation itself installs in every active process (stack.enable), in this order: the template guard
                                                     # (templ_guard.install: census wrappers, no switch) and the confidence phase's dtype (conf_dtype.install: reads OPENFOLD3_OB0_OPT_CONF_DTYPE —
                                                     # a registry lever whose `kit` is PACKAGE: no hook directory carries it)
PACKAGE = "package"                                  # registry.Lever.kit of a lever ACTIVATION_MODULES installs


def lever_applied(name: str, home: Optional[str] = None) -> Optional[bool]:
    """The kit's own record for one lever: True applied, False not applied although its target module was imported, None no record yet."""
    probe = _PROBES.get(name)
    if probe is None:
        return None
    try:
        return probe(home or _env.tree_home())
    except Exception:
        return None


def levers_record(rep: Optional[dict] = None, home: Optional[str] = None) -> dict:
    """The lever fields of the activation report from the add-ons' current records: `levers_requested` (the line's list), `levers_applied`, `levers_unavailable`,
    `levers_pending` (target not imported yet), `partial` (a requested lever is unavailable, or a carrier module was not loaded from the
    line's carrier), `arm_complete` (True: every requested lever applied from the carrier; False: partial; None: still pending) and
    `lever_evidence` (the module and record each verdict was read from)."""
    rep = rep if rep is not None else (_REPORT or {})
    home = home or rep.get("home") or _env.tree_home()
    requested = list(rep.get("levers_requested") or rep.get("levers") or [])
    applied, unavailable, pending, evidence = [], [], [], {}
    for name in requested:
        lv = LEVERS.get(name)
        target = lv.target if lv is not None else ""
        if not target:
            pending.append(name)
            evidence[name] = "no import target: not applied by a hook"
            continue
        if target not in sys.modules:
            pending.append(name)
            evidence[name] = f"pending: {target} not imported"
            continue
        got = lever_applied(name, home)
        if got:
            applied.append(name)
            evidence[name] = "applied: the kit's record"
        else:
            unavailable.append(name)
            if got is False:
                detail = _offload_fn_detail(name)
                evidence[name] = f"unavailable: the kit's record ({detail} occupies the slot)" if detail else "unavailable: the kit's record"
            else:
                evidence[name] = f"unavailable: {target} imported, no kit record"
    carrier = rep.get("carrier")
    off_carrier = []
    if carrier:
        for mn in CARRIER_MODULES:
            m = sys.modules.get(mn)
            f = getattr(m, "__file__", None) if m is not None else None
            if f and not _under(f, carrier):
                off_carrier.append(f"{mn} loaded from {f}, not the line's carrier {carrier}")
    census = offload_census()
    off_events = offload_partial(census)
    partial = bool(unavailable) or bool(off_carrier) or bool(off_events)
    out = {"levers_requested": requested, "levers_applied": applied, "levers_unavailable": unavailable, "levers_pending": pending,
           "partial": partial, "arm_complete": False if partial else (True if not pending else None), "lever_evidence": evidence,
           "carrier": carrier, "off_carrier": off_carrier}
    if census is not None:
        out["offload"] = census                                                                  # the port's census (the exit tally)
        out["offload_events"] = off_events
    return out


# ------------------------------------------------------------------------------------------------------- instance counter ----
def _wrap_model_class(module) -> bool:
    """Counting is the core's (opt_core.instances.register_instance_counter on the class, now that its module exists); the kit adds the
    READY line at the first forward (weights on the device) and the per-item forward-time line (report.wrap_forward_timer, every add-on
    route). Idempotent."""
    cls = getattr(module, MODEL_CLASS, None)
    if cls is None or _INSTANCES["wrapped"]:
        return False
    from opt_core import instances as _inst
    _inst.register_instance_counter(MODEL_MODULE, MODEL_CLASS)
    _report.wrap_forward_timer(cls, route="kit")                       # the per-item `Model forward time: …` line (inside the ready wrap below)
    fwd = getattr(cls, "forward", None)
    if fwd is not None:                                                 # the ready line: once, at the first forward (weights on the device)
        def forward(self, *a, _orig=fwd, **k):
            if not _INSTANCES.get("ready"):
                _INSTANCES["ready"] = True
                _report.log_ready()
            return _orig(self, *a, **k)
        cls.forward = forward
    _INSTANCES["wrapped"] = True
    return True


class _CounterFinder:
    """Arms the instance counter (the core's) and the ready line on `OpenFold3` right after its module is executed (after the add-ons' own exec_module wrappers, which run inside ours).

    Consulted once: the add-ons' finders iterate `sys.meta_path` themselves (skipping only their own instance: of3t_hook/sitecustomize.py:66-68),
    so while this finder walks the path a kit finder calls back into it; `self.busy` makes that
    re-entrant call return None, and the module's exec_module is wrapped exactly once, outermost — the add-ons' installs run inside it, the
    class wrap after them."""
    def __init__(self):
        self.done = False
        self.busy = False
        self.wraps = 0

    def find_spec(self, fullname, path=None, target=None):
        if self.done or self.busy or fullname != MODEL_MODULE:
            return None
        self.busy = True
        try:
            spec = None
            for finder in sys.meta_path:
                if finder is self:
                    continue
                try:
                    spec = finder.find_spec(fullname, path, target)
                except Exception:
                    spec = None
                if spec is not None:
                    break
            if spec is None or spec.loader is None:
                return None
            self.done = True
            self.wraps += 1
            orig = spec.loader.exec_module

            def exec_module(module, _orig=orig):
                _orig(module)
                _wrap_model_class(module)
            spec.loader.exec_module = exec_module
            return spec
        finally:
            self.busy = False


def register_instance_counter() -> str:
    global _COUNTER_FINDER
    m = sys.modules.get(MODEL_MODULE)
    if m is not None:
        return "wrapped" if _wrap_model_class(m) else "already"
    if _COUNTER_FINDER is None:
        _COUNTER_FINDER = _CounterFinder()
        sys.meta_path.insert(0, _COUNTER_FINDER)                 # ahead of the path finder (a finder appended after it is never consulted); the add-ons'
                                                                 # finders inserted later at 0 return None for this module, so ours still wraps it
    return "armed"


# --------------------------------------------------------------------------------------------------------------- activate ----
# Levers whose kernel module is the tree's shared copy imported under an add-on's own name (common/opt_core/opt_core/kernels/<name>, routed
# BY NAME — opt_core.kernels): lever -> (core kernel name, the add-on's import name). Empty: the triangle attention is the core's
# provider's through the pair cells (of3_triattn), imported under the core's own names (CORE_CELL_LEVERS).
CORE_KERNEL_ROUTES: Dict[str, Tuple[str, str]] = {}
# Levers whose cells are the core's carried kernels imported under the core's own names (openfold3_ob0_opt/pairfused.py -> opt_core.kernels /
# opt_core.attn.pair_fused): lever -> the carried kernel names gated at activation (opt_core.kernels.route + route_check: bytes checked
# against the core's SUMS; a failed gate refuses the activation by name). No alias finder: nothing in the add-ons imports them.
CORE_CELL_LEVERS: Dict[str, Tuple[str, ...]] = {"trimul_v4": ("fpf_trimul_v4",), "triatt_block": ("flash_triattn", "fpf_triatt_pro", "fpf_triatt_epi", "lnl_fused"),
                                                "pair_transition": ("fpf_transition", "lnl_fused"), "dit_attn": ("dtk_kernels", "apb_attn"),
                                                "dit_glue": ("dtk_kernels", "apb_attn"), "apb_trunk": ("apb_attn",), "token_agg": ("dtk_kernels",), "atom_window": ("atom_window",), "templ_embed": ("templ_embed",),
                                                "tp_triatt": ("flash_triattn",), "tp_trimul": ("fpf_trimul_v4",),
                                                "triatt_provider": ("fpf_triatt_k2b", "flash_triattn"), "trimul_provider": ("fpf_trimul_v4", "fpf_trimul"), "exactln": ("ln",), "ln_provider": ("ln",), "transition_exact": ("fpf_transition",)}   # the providers (opt_core.kernels.triattn / .trimul) are package modules, never routed top-level names: a routed `triattn` would shadow the sealed package's own `triattn` tree


def core_kernel_exports() -> Dict[str, Dict[str, str]]:
    """`{core kernel: {VAR: path relative to the tree}}`: exports a carried cell reads at import — none: the v4 TriMul kernel's launch cells are the
    core's own table (opt_core/kernels/fpf_trimul_v4/table.json), read by the core."""
    return {}


def core_routes(levers) -> Dict[str, Tuple[str, str]]:
    """The core kernel routes of a lever set (CORE_KERNEL_ROUTES restricted to the levers requested)."""
    return {l: CORE_KERNEL_ROUTES[l] for l in levers if l in CORE_KERNEL_ROUTES}


def core_cell_levers(levers) -> Dict[str, Tuple[str, ...]]:
    """The core-cell levers of a lever set (CORE_CELL_LEVERS restricted to the levers requested)."""
    return {l: CORE_CELL_LEVERS[l] for l in levers if l in CORE_CELL_LEVERS}


def core_carried_exports() -> Dict[str, str]:
    """`{VAR: path}`: the exports opt_core.attn.pair_fused sets itself, before the first import, for the carried cells that read a data file at import
    (`pair_fused.carried_exports()`: the lnl tile table, the K2B cell table) — the values the activation gate checks those cells' export spec against."""
    from opt_core.attn import pair_fused as PF
    return dict(PF.carried_exports())


def core_route_gate(name: str, environ: Optional[dict] = None):
    """Gate (opt_core.gates.Gate): `import <name>` resolves to the core copy with the carried bytes and the exports the kernel reads are set
    (opt_core.kernels.route + route_check over `environ` — a neighbouring version or a missing export is a named refusal, never served)."""
    from opt_core import kernels
    kernels.route(name)
    return kernels.route_check(name, environ=environ)


class CoreAliasFinder:
    """Serves the add-on's import name `alias` from the core's carried copy of kernel `name` (opt_core.kernels.carried_path), lazily: nothing is
    imported until the hook chain imports the alias, and the module keeps the add-on's name (`__name__ == alias`) with the core's file."""

    def __init__(self, name: str, alias: str, path: str):
        self.name, self.alias, self.path = name, alias, path

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.alias:
            return None
        import importlib.util
        return importlib.util.spec_from_file_location(self.alias, self.path)


def apply_core_route(name: str, alias: str) -> str:
    """Route kernel `name` from the core (opt_core.kernels.route) and serve the add-on's import name `alias` from the same file (CoreAliasFinder,
    first on sys.meta_path; an alias already imported through another path is dropped from sys.modules so the next import takes the route). Returns
    the core file path."""
    from opt_core import kernels
    kernels.route(name)
    path = kernels.carried_path(name)
    sys.meta_path[:] = [f for f in sys.meta_path if not (isinstance(f, CoreAliasFinder) and f.alias == alias)]
    sys.meta_path.insert(0, CoreAliasFinder(name, alias, path))
    stale = sys.modules.get(alias)
    if stale is not None and os.path.abspath(getattr(stale, "__file__", "") or "") != os.path.abspath(path):
        del sys.modules[alias]
    return path


def activate(mode: str, *, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False, home: Optional[str] = None,
             environ: Optional[dict] = None, log: bool = True, n_tokens: Optional[int] = None, n_gpu=None) -> dict:
    global _REPORT
    _log = _report.log_activation if log else (lambda rep, stream=None: _report.activation_line(rep))
    environ = os.environ if environ is None else environ
    home = home or _env.tree_home(environ)
    try:
        mode = modes.check_mode(mode)
    except ValueError as e:
        rep = {"active": False, "mode": mode, "reason": str(e), "package_version": __version__}
        _log(rep)
        if strict:
            raise ActivationError(str(e))
        return rep
    if _REPORT is not None and not dry_run:
        if _REPORT.get("mode") == mode and (mode not in modes.MODE_LINES or _REPORT.get("line") == modes.line_for(mode, n_gpu=n_gpu, environ=environ)):
            return dict(_REPORT)                                  # idempotent
        reason = f"mode {mode} requested after {_REPORT.get('mode')}/{_REPORT.get('line') or '-'} was activated in this process; one mode per process"
        if strict:
            raise ActivationError(reason)
        return {"active": False, "mode": mode, "reason": reason, "package_version": __version__}
    refusal, pin = None, None
    try:                                                           # the core pin gate (kit_template/_core_gate.py: pinned vs installed, read from disk) then the producer probe —
        from ._core_gate import gate                              # an absent / mismatched / older core is refused by name on every route before anything is resolved
        from ._core_gate import CoreGateRefused
        try:
            pin = gate(__file__, tag=_report.TAG)
        except CoreGateRefused as e:
            refusal = e.line.split("NOT ACTIVE: reason=", 1)[1].strip() if "NOT ACTIVE: reason=" in e.line else e.reason
        refusal = refusal or modes.core_refusal()
    except ImportError as e:                                       # pragma: no cover — the gate file ships inside the package
        refusal = f"core_pin_unreadable: {e}"
    if refusal:
        rep = {"active": False, "mode": mode, "reason": refusal, "package_version": __version__}
        _log(rep)
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    base = {"mode": mode, "package_version": __version__, "openfold3_version": modes.openfold3_version(), "torch": modes.torch_version(),
            "triton": modes.triton_version(), "gpu": gpu_probe(environ), "trigger": trigger, "dry_run": dry_run, "home": home, "compile": modes.COMPILE_STATE,
            "target_gpu": environ.get(ENV_TARGET_GPU), "det": environ.get("OF3_DETERMINISTIC")}
    notes: List[str] = []
    if mode == "off":
        if modes.levers_off_ignored(environ):                     # MODEL_OPT_LEVERS_OFF under mode off: stock has no lever to leave off — named, never refused
            notes.append(modes.levers_off_ignored(environ))
        rep = dict(base, active=False, line=None, line_spelling="off", levers=[], levers_requested=[], levers_applied=[], levers_unavailable=[],
                   levers_pending=[], partial=False, arm_complete=None, carrier=None, hooks=[], env={}, reason="mode off: stock (no switch, no hook)", notes=notes)
        if not dry_run:
            _REPORT = rep
        _log(rep)
        return dict(rep)
    try:
        res = modes.resolve(mode, home, environ=environ, n_tokens=n_tokens, n_gpu=n_gpu)
    except ValueError as e:
        rep = dict(base, active=False, reason=str(e))
        _log(rep)
        if strict:
            raise ActivationError(str(e))
        return rep
    base["inherited"] = res.inherited                              # a process the run started (modes.ENV_RESOLVED): the primary process's resolution verbatim; its gates are not re-evaluated and nothing is printed here
    base["n_gpu"] = modes.rank_world(environ) if (mode == "big" and res.line in modes.TP_LINES) else 1     # P of this process: a big/tp rank holds 1/P of the pair rows (OF3TP_WORLD); every other route is one GPU
    base.update(line=res.line, line_spelling=modes.describe_line(res), tier=res.tier, hooks=list(res.hooks),
                hooks_spelling=">".join(modes.hook_spellings(res)), hook_dirs=list(res.hook_dirs), entry_hook=res.entry_hook,
                env=dict(res.exports), unset=list(res.unsets), levers=list(res.levers), levers_requested=list(res.levers), levers_applied=[],
                levers_unavailable=[], levers_pending=list(res.levers), levers_off=list(res.levers_off), partial=False, arm_complete=None, applied="at import",
                carrier=carrier_dir(res, home),
                lever_tiers={n: LEVERS[n].tier for n in res.levers if n in LEVERS}, source=res.source, size_gate=res.size_gate, gate_reason=res.gate_reason, of3o_gate=res.of3o_gate, of3o_reason=res.of3o_reason, of3o_min_tokens=res.of3o_min_tokens, reach_gate=res.reach_gate, reach_reason=res.reach_reason, reach_min_tokens=res.reach_min_tokens, reach_off=list(res.reach_off), conf_gate=res.conf_gate, conf_reason=res.conf_reason, conf_dtype=res.conf_dtype, z_dtype=res.z_dtype, graphs_cap=modes.graphs_cap(environ) if not res.conflicts else None, n_tokens=res.n_tokens, precision=res.precision)
    notes += res.notes
    tg = environ.get(ENV_TARGET_GPU)
    g = base["gpu"]
    if tg and g.get("name"):
        from opt_core import gates as _gates
        if not _gates.gpu_name_check(g, tg, fold_case=True).ok:                # the kit's check: a case-folded substring of the card name
            notes.append(f"{ENV_TARGET_GPU}={tg} but this box is {g['name']}")
    problems: List[str] = []
    if g.get("supported") is False:                                           # below the levers' floor (registry.MIN_SM: the bf16 / TF32 tensor-core kernels every line carries cannot run):
        problems.append(f"GPU compute capability {g.get('cc')} is below sm80: the {mode} line's levers cannot run on this card "   #  the mode refuses by name — a mode is all of its levers, never a subset under its name (`--mode off` runs anywhere upstream does)
                        f"({', '.join(res.levers)}); --mode off runs upstream as it is")
    base["core"] = pin                                            # the core pin gate's facts ({"pinned": {...}, "installed": {...}}: kit_template/_core_gate.py, the ONE pin producer; no second pin call here)
    v = version_gate(home)
    if v:
        problems.append(v)
    problems += res.conflicts
    late = late_activation_gate()
    if late and not dry_run:
        problems.append(late)
    elif late:
        notes.append(late)
    routes = core_routes(res.levers)                              # the levers whose kernel module is served from the shared core (opt_core.kernels), by name
    cells = core_cell_levers(res.levers)                          # the levers whose cells ARE core kernels under the core's own names (pairfused.py)
    exports_by_kernel = core_kernel_exports()
    cell_exports = {k: os.path.join(home, rel) for names in cells.values() for name in names for k, rel in exports_by_kernel.get(name, {}).items()}
    if cells:                                                     # what those kernels read at import: the kit's fpf_trimul_v4 cell table and the data files the core carries for
        cell_exports.update(core_carried_exports())               #  its cells (lnl tile table, K2B cells: opt_core.attn.pair_fused.carried_exports); a caller's own value stands
    cell_exports = {k: v for k, v in cell_exports.items() if k not in environ}   #  and is gated as given
    res.exports.update(cell_exports)
    gate_env = dict(environ); gate_env.update(cell_exports)
    gated = {}
    for lever, names in list((l, (n,)) for l, (n, a) in routes.items()) + list(cells.items()):
        for name in names:
            g = gated.get(name) or core_route_gate(name, gate_env)
            gated[name] = g
            base.setdefault("gates", {})[g.name] = {"ok": g.ok, "reason": g.reason, **{k: g.details.get(k) for k in ("resolved", "core_copy", "routed")}}
            if not g.ok:
                problems.append(f"lever {lever}: {g.reason}")
    if cells:
        try:
            import importlib
            importlib.import_module("opt_core.attn.pair_fused")     # the serve layer the cell levers call (opt_core >= 0.5; the pin gate above names an older core first)
        except Exception as e:  # noqa: BLE001
            problems.append(f"levers {', '.join(sorted(cells))}: opt_core.attn.pair_fused not importable ({e!r})")
    base["core_routes"] = sorted(set(routes) | set(cells))
    if res.inherited:                                             # the primary process passed these gates before this process existed (it would not have started otherwise)
        if problems:
            notes.append("inherited; not re-evaluated here: " + "; ".join(problems))
        problems = []
    if problems:
        rep = dict(base, active=False, reason="; ".join(problems), notes=notes)
        _log(rep)
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    if dry_run:
        rep = dict(base, active=False, reason=_report.DRY_RUN_OK, notes=notes)
        _log(rep)
        return rep
    # ---- apply: exports first (before CUDA initialises), then sys.path, the exit tally, the core kernel routes, the hook route, the instance counter ----
    for k in res.unsets:
        os.environ.pop(k, None)
    os.environ.update(res.exports)
    os.environ[ENV_MODE] = mode
    if not res.inherited:
        os.environ[modes.ENV_RESOLVED] = modes.freeze(res)                  # the processes this run starts inherit the resolution verbatim (modes.resolve thaws it: no re-resolution, no ACTIVE line, no gate)
    _hooks.put_on_sys_path(res.hook_dirs)
    _report.register_exit_tally(status, instances, inherited=res.inherited)   # before the add-ons register their own exit lines (atexit is LIFO); a process the run started registers none
    for lever, (name, alias) in routes.items():                  # the add-on's `import <alias>` resolves to the core copy of <name> (same bytes, held above; lazily — nothing imported here; the carried file stays on disk, shadowed)
        apply_core_route(name, alias)
    try:
        run = _hooks.run(res.entry_hook, res.hooks[0]) if res.hooks else {"ran": None, "chain": []}   # a line without hooks: nothing to run
    except Exception as e:
        rep = dict(base, active=False, apply_failed=True, reason=f"hook {res.entry_hook} failed: {e!r}", notes=notes)
        _REPORT = rep
        _log(rep)
        raise ActivationError(rep["reason"]) from e
    inst = _hooks.installed(home)
    miss = _hooks.missing(res, home)
    if miss:
        rep = dict(base, active=False, apply_failed=True, hooks_installed=inst,
                   reason=f"hook(s) of {', '.join(miss)} not installed after {os.path.basename(os.path.dirname(res.entry_hook))}/sitecustomize.py ran", notes=notes)
        _REPORT = rep
        _log(rep)
        raise ActivationError(rep["reason"])
    counter = register_instance_counter()
    if mode == "fast":                                            # the fast mode's precision on every route: pl_trainer_args.precision = bf16-mixed where upstream's Trainer reads it
        from . import precision as _precision                    #  (silent under the CLI's composed bf16 yaml; named when a config said otherwise)
        _precision.install()
    import importlib
    for _name in ACTIVATION_MODULES:                             # the package's own installs in every active process (no hook directory reads their words): the template guard's census, the confidence phase's dtype
        importlib.import_module(f"{__package__}.{_name}").install()
    from .registry import declare_arch                            # every lever's GPU-class support row in opt_core.arch (the one registry; CHANGES.md 'Cards' reads card_table)
    declare_arch()
    rep = dict(base, active=True, hooks_installed=inst, hook_run=run, instance_counter=counter, templ_guard=True, notes=notes, sys_path_head=sys.path[:len(res.hook_dirs)])
    _REPORT = rep
    _log(rep)
    _REPORT["logged"] = True
    return dict(_REPORT)
