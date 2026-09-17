"""Modes, and how a package mode resolves to the kit's own composition.

The kit is a Python-API kit: its package `engines.flashzoi.kits.v1_25` (under the one PYTHONPATH root
`opt/forward/kits_v1_25`, KIT_RELPATH) carries its tables as module-level literals — `LEVERS` (the component names, every one
ON under the kit's apply line `kit.KitRunner(model, ...)`), `LEVER_CLASS` (the class of each component), `PINS` (the pins, among them
`device_names`), `ARM` — and its wrapper `_wrap.py` carries the apply line's signature
(`KitRunner.__init__`: the knob `numerics`, its default and its accepted values). This module never transcribes a
component: `kit_table()` reads the literals out of the kit files themselves (`ast.literal_eval`, so no torch is needed), and the activation
core (stack.py) asserts the live module's tables equal the read ones before arming.

Modes are the user-facing contract; a mode is the kit's LEVERS plus ONE constructor
argument of the apply line (MODE_ARGS — the only table this module owns):
  off    stock — no environment set, no component applied (the tree's stock caller in a clean subprocess, torch's own numerics)
  exact  `KitRunner(model)` at the kit's defaults (numerics='tf32': the stock's TF32 class — torch's defaults, cuDNN TF32 on, plus matmul
         TF32 for the head GEMM — set inside each call and restored); the kit's claim: byte-identical to stock (torch's default numerics)
         — THE PACKAGE DEFAULT (DEFAULT_MODE); selected with --mode exact / FLASHZOI_OPT=exact / enable('exact') or no mode at all
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

ENV_MODE = "FLASHZOI_OPT"
MODES = ("off", "exact")
DEFAULT_MODE = "exact"                                     # the package default: a run that names no mode (no --mode, no FLASHZOI_OPT) runs exact
KIT_MODES = ("exact",)                                   # the modes the kit serves
MODE_ARGS = {"off": None, "exact": {}}                    # the ONE constructor argument per kit mode (KitRunner(model, **args)); exact = the kit's defaults (numerics='tf32')
KIT_RELPATH = os.path.join("opt", "forward", "kits_v1_25")             # the kit's one PYTHONPATH root, under the tree (flashzoi/)
KIT_MODULE = "engines.flashzoi.kits.v1_25"
KIT_INIT_RELPATH = os.path.join("engines", "flashzoi", "kits", "v1_25", "__init__.py")
KIT_WRAP_RELPATH = os.path.join("engines", "flashzoi", "kits", "v1_25", "_wrap.py")
KIT_TABLE_NAMES = ("LEVERS", "LEVER_CLASS", "PINS", "ARM")
KIT_KNOBS = ("numerics",)                                 # KitRunner.__init__'s knobs (the mode's one constructor argument)


@dataclass
class Resolution:
    mode: str
    components: Tuple[str, ...] = ()                          # kit modes: the kit's LEVERS verbatim; off: none
    knobs: Dict[str, str] = field(default_factory=dict)  # the effective knobs: KitRunner's own defaults updated by the mode's argument; off: {}
    constructor_kwargs: Dict[str, str] = field(default_factory=dict)   # MODE_ARGS[mode]: what the apply line receives beyond the model
    kit_arm: Optional[str] = None
    notes: list = field(default_factory=list)

    @property
    def apply_line(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.constructor_kwargs.items())
        return f"kit.KitRunner(model{args})" if self.mode != "off" else "none"

    @property
    def component_line(self) -> str:
        return ",".join(self.components) if self.components else "none"

    @property
    def knob_line(self) -> str:
        return ",".join(f"{k}={v}" for k, v in self.knobs.items()) if self.knobs else "none"


def _literal_assignments(path: str, names) -> dict:
    """Module-level `NAME = <literal>` and `NAME[<literal>] = <literal>` assignments of `path` (ast.literal_eval; a non-literal is
    skipped, never guessed)."""
    tree = ast.parse(open(path, "rb").read(), filename=path)
    out: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        t = node.targets[0]
        try:
            if isinstance(t, ast.Name) and t.id in names:
                out[t.id] = ast.literal_eval(node.value)
            elif isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in names and t.value.id in out:
                out[t.value.id] = dict(out[t.value.id]); out[t.value.id][ast.literal_eval(t.slice)] = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
    return out


def kit_init_path(kit_root: str) -> str:
    return os.path.join(kit_root, KIT_INIT_RELPATH)


def kit_class_rules(kit_root: str):
    """The kit dir's class rule module (kits/v1_25/_class_records.py: class_serves / pinned_serves / record_serves) loaded by path — pure, no torch."""
    import importlib.util
    rules = os.path.join(os.path.dirname(kit_init_path(kit_root)), "_class_records.py")
    if not os.path.isfile(rules): return None
    spec = importlib.util.spec_from_file_location("_fz_kit_class_records", rules); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def kit_class_record(kit_root: str, device_name: str, cc=None, memory_mib=None):
    """The kit dir's class record naming `device_name` for THIS fz_exact.py (class_records/*.json — the files the kit's own
    assert_device_class reads), or None: a record counts when it serves the device by the kit dir's own rule
    (_class_records.record_serves: the class BY CAPABILITY — `sm` == cc, the record's memory_mib documentation not a key — or by name) and its `kit` is the
    digest of the kit's fz_exact.py bytes. Pure file reads (no torch, no import of the kit package: the rule module is loaded by path)."""
    import glob, hashlib, json
    kdir = os.path.dirname(kit_init_path(kit_root)); fz = os.path.join(kdir, "fz_exact.py"); mod = kit_class_rules(kit_root)
    if not os.path.isfile(fz) or mod is None: return None
    digest = hashlib.sha256(open(fz, "rb").read()).hexdigest()
    for path in sorted(glob.glob(os.path.join(kdir, "class_records", "*.json"))):
        if not os.path.isfile(path): continue
        try: rec = json.load(open(path))
        except (OSError, ValueError): continue
        if mod.record_serves(rec, digest, device_name, cc, memory_mib): return {"path": path, **{k: rec.get(k) for k in ("class", "sm", "memory_mib")}}
    return None


def kit_table(kit_root: str) -> dict:
    """The kit's tables read from its own `__init__.py` (KIT_TABLE_NAMES) plus the apply line's knob defaults read from
    `_wrap.py` (`knob_defaults`: {numerics}). Raises FileNotFoundError / KeyError naming what is missing."""
    init = kit_init_path(kit_root)
    if not os.path.isfile(init):
        raise FileNotFoundError(f"kit module not found: {init} (the kit root carries {KIT_INIT_RELPATH})")
    t = _literal_assignments(init, KIT_TABLE_NAMES)
    missing = [n for n in ("LEVERS", "LEVER_CLASS", "PINS", "ARM") if n not in t]
    if missing:
        raise KeyError(f"kit tables {missing} not found as literals in {init}")
    t["LEVERS"] = tuple(t["LEVERS"])
    t["knob_defaults"], t["knob_choices"] = knob_signature(os.path.join(kit_root, KIT_WRAP_RELPATH))
    t["kit_root"] = kit_root
    return t


def knob_signature(wrap_path: str) -> tuple:
    """(defaults, choices) of `KitRunner.__init__`'s tier knobs (KIT_KNOBS), read from the wrapper file (the def inside `build()`):
    the defaults are what `kit.KitRunner(model)` runs with; the choices are the tuple each knob is checked against in the body
    (`if numerics not in ("tf32",)`), so a mode's argument is asserted against the kit's own accepted values."""
    tree = ast.parse(open(wrap_path, "rb").read(), filename=wrap_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            args = node.args
            names = [a.arg for a in args.args]
            if not all(k in names for k in KIT_KNOBS):
                continue
            defaults = dict(zip(names[len(names) - len(args.defaults):], args.defaults))
            out, choices = {}, {}
            for k in KIT_KNOBS:
                if k not in defaults:
                    raise KeyError(f"{wrap_path}: KitRunner.__init__ has no default for {k}")
                out[k] = ast.literal_eval(defaults[k])
            for sub in ast.walk(node):
                if (isinstance(sub, ast.Compare) and isinstance(sub.left, ast.Name) and sub.left.id in KIT_KNOBS and len(sub.ops) == 1
                        and isinstance(sub.ops[0], ast.NotIn) and isinstance(sub.comparators[0], ast.Tuple)):
                    choices[sub.left.id] = tuple(ast.literal_eval(sub.comparators[0]))
            for k in KIT_KNOBS:
                if k not in choices:
                    raise KeyError(f"{wrap_path}: KitRunner.__init__ has no `{k} not in (...)` check to read the accepted values from")
            return out, choices
    raise KeyError(f"{wrap_path}: no KitRunner.__init__ with the knobs {KIT_KNOBS}")


def knob_defaults(wrap_path: str) -> Dict[str, str]:
    return knob_signature(wrap_path)[0]


def check_mode(mode: Optional[str]) -> str:
    """The mode name: the argument, else $FLASHZOI_OPT, else DEFAULT_MODE; an unknown name is a ValueError that says why."""
    m = (mode if mode is not None else os.environ.get(ENV_MODE, "")).strip().lower() or DEFAULT_MODE
    if m not in MODES:
        raise ValueError(f"unknown mode {m!r}; expected one of {MODES}")
    return m


def resolve(mode: str, kit_root: str) -> Resolution:
    """A package mode -> the kit's composition, from the kit's own tables (kit_table) and the mode's one constructor argument
    (MODE_ARGS, asserted against the kit's own accepted knob values)."""
    mode = check_mode(mode)
    if mode == "off":
        return Resolution(mode="off", notes=["stock route: no component, no environment"])
    t = kit_table(kit_root)
    args = dict(MODE_ARGS[mode])
    for k, v in args.items():
        if v not in t["knob_choices"][k]:
            raise ValueError(f"mode {mode}: {k}={v!r} is not among the kit's accepted values {t['knob_choices'][k]}")
    knobs = dict(t["knob_defaults"]); knobs.update(args)
    return Resolution(mode=mode, components=tuple(t["LEVERS"]), knobs=knobs, constructor_kwargs=args, kit_arm=t.get("ARM"))


def stack_key(cc: Optional[str], triton_version: Optional[str]) -> str:
    """The stack key `<compute capability>|<triton major.minor>` (e.g. `9.0|3.1`), from the running machine. A missing component is
    named (`unknown:no_cc` / `unknown:no_triton`), never silently formatted as a plausible-looking key: stack.py's `_gates()`
    refuses by name when it sees either marker (a wrong-but-plausible stack key would silently mis-key the JIT/Triton cache)."""
    cc_part = cc if cc else "unknown:no_cc"
    if triton_version:
        tv = ".".join(str(triton_version).split("+")[0].split(".")[:2])
    else:
        tv = "unknown:no_triton"
    return f"{cc_part}|{tv}"


def jit_cache_key(version: Optional[str] = None, cuda: Optional[str] = None, cc: Optional[str] = None) -> str:
    """The JIT cache directory key `torch<version>-cu<cuda>-sm<cc>`: the torch version without its local tag, the CUDA version without
    the dot, the device's compute capability digits (e.g. `torch2.5.1-cu124-sm90`). Defaults: the installed torch's metadata version
    (no torch import), its `+cu<digits>` local tag (else torch.version.cuda), the device's compute capability read through torch.
    `configs/h100.env` exports it as MODEL_OPT_STACK_KEY (a pre-set value is kept; a caller may key a persistent TRITON_CACHE_DIR by it). A component that
    cannot be read is a refusal (RuntimeError, named), never a silent 'unknown' baked into the key (a wrong-but-plausible key would
    silently mis-key the JIT/Triton cache)."""
    if version is None:
        try:
            import importlib.metadata as _md
            version = _md.version("torch")
        except Exception:  # noqa: BLE001
            version = None
        if version is None:
            try:
                import torch as _t
                version = getattr(_t, "__version__", None)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"jit_cache_key: torch is not importable ({e!r}); cannot read its version") from e
        if version is None:
            raise RuntimeError("jit_cache_key: torch has no __version__ and importlib.metadata has no 'torch' distribution")
    base, _, local = version.partition("+")
    if cuda is None:
        if local.startswith("cu") and local[2:].isdigit():
            cuda = local[2:]
        else:
            try:
                import torch as _t
                cuda = getattr(getattr(_t, "version", None), "cuda", None)
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"jit_cache_key: torch is not importable ({e!r}); cannot read torch.version.cuda") from e
            if cuda is None:
                raise RuntimeError("jit_cache_key: torch.version.cuda is unset (a CPU-only torch build?); cannot key the CUDA component")
            cuda = cuda.replace(".", "")
    if cc is None:
        try:
            import torch as _t
            available = _t.cuda.is_available()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"jit_cache_key: torch is not importable ({e!r}); cannot read the device compute capability") from e
        if not available:
            raise RuntimeError("jit_cache_key: no CUDA device visible to torch; cannot read the device compute capability")
        maj, mnr = _t.cuda.get_device_capability(0)
        cc = f"{maj}{mnr}"
    cuda = str(cuda).replace(".", "")                                   # '12.4' and '124' name the same CUDA
    cc = str(cc).replace(".", "")                                       # '9.0' and '90' the same capability
    return f"torch{base}-cu{cuda}-sm{cc}"
