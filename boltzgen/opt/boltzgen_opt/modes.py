"""Modes — the one table of what a package mode means for BoltzGen, and how it resolves to the lever modules' switches.

The lever modules activate by process environment, not by a mode table of their own:
`PYTHONPATH=<opt>/forward/xattempt_addon/src:<opt>/forward/fast_inference/src` (`xattempt_addon` first, then `fast_inference`,
whose `sitecustomize.py` imports `bg_hook` in every python process) and the runner
`python forward/xattempt_addon/src/xa_run.py <run_dir> <seed>`, which sets the lever switches to their defaults itself
(`os.environ.setdefault(...)`: `BG_GRAPH=graph`, `XA_FAST_INIT=1`, `XA_HOIST=1`) and imports the `xa_*` lever modules for every
GPU step. This module never transcribes those values: ``kit_defaults()`` reads the ``setdefault`` calls out of ``xa_run.py`` by
AST, ``runner_imports()`` reads its lever imports and ``hook_imports()`` reads ``fast_inference``'s ``sitecustomize.py``; the
PYTHONPATH order is the mode table's (``KitMode.kits``). ``resolve(mode)`` turns a package mode into a ``Resolution`` (lever
directories in PYTHONPATH order, the environment to export, the modules the in-process form imports, the runner path); application is always the lever modules' own code (stack.py).

Package modes (``MODES``; ``DEFAULT_MODE`` is the one default every command reads):
  off     upstream alone: `configure`, then one clean subprocess per pipeline step of ``<run_dir>/steps.yaml`` — the stock
          caller (stock_design.py; design.run_stock), its environment checked free of every kit variable, module and directory
          before anything of upstream is imported; step i seeded ``seed + i`` when the caller passes ``--seed``, upstream's own
          unseeded form when not (upstream itself has no seed argument).
  exact   the bit-identical lever set: the in-process pipeline and the CUDA-graph design sampler (`BG_GRAPH=graph`), fast-init
          (`XA_FAST_INIT=1`), the attention-mask hoist (`XA_HOIST=1`) and the background writer (`HL_ASYNC_WRITER=1`) —
          outputs byte-identical to seeded upstream at the same seed. Every kit mode imports `sz_levers` for its out-of-memory
          trace (an evidence line where a CUDA out-of-memory was raised inside Boltz.forward, re-raised unchanged; `SZ_TD_CHUNK`
          unset: no size lever installs) and `hl_levers` (the background writer) — ``own_imports``: neither ``xa_run.py`` nor
          ``sitecustomize.py`` names them.
  big   exact's runner with the CUDA-graph sampler and the mask hoist switched off and the size lever on (``env_overrides``:
          `BG_GRAPH=off`, `XA_HOIST=0`, `SZ_TD_CHUNK=64`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`): the same bytes
          as exact wherever both complete (except the `design_iiptm` confidence field: summation order), at lower peak memory,
          and designs beyond exact's token ceiling on the same card (the mode's overrides are applied after ``EXCLUDED_VALUES``
          checks the raw runner defaults).
  fast    the tolerance class (``FAST_MODE``; the package default): exact's levers plus the fast levers (`fl_levers`, imported by
          the package itself like `sz_levers`): `cond_dedup` (`FL_COND_DEDUP=1` — the sampler's multiplicity-invariant conditioning
          computed once per step instead of once per sample: same values, summation order may differ), `attn_bf16`
          (`FL_ATTN_BF16=1` — the token transformer's pair-biased attention and the atom transformers' windowed attention through
          torch SDPA in bf16 with fp32 accumulate) with its backend pinned to cuDNN's fused kernel (`attn_cudnn`,
          `FL_ATTN_BACKEND=cudnn`), and `dit_fused` (`FL_DIT_FUSED=1` — the token transformer layers' AdaLN / output-gate / SwiGLU
          elementwise statements as the shared core's fused Triton row kernels). Not bitwise: within stock's own seed-to-seed
          variation (`CHANGES.md`).
  every kit mode also carries `async_writer` (`HL_ASYNC_WRITER=1`, `hl_levers`): upstream's design writer body runs unchanged on host copies of each unit's tensors in background worker processes of the shared core's AsyncWriter — the same files and bytes, written while the next unit computes.

Caller-set values of any lever switch (``KIT_SWITCH_PREFIXES``) are dropped from a mode's environment and reported, never obeyed — the
mode words are the only lever selectors; the one such name kept when set is the hook's timing-file path (``OBSERVABILITY_SWITCHES``).
"""
from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

ENV = "BOLTZGEN_OPT"                                   # the env route: BOLTZGEN_OPT=<mode>
MODES: Tuple[str, ...] = ("off", "exact", "fast", "big")
DEFAULT_MODE = "fast"                                  # the package default: fast (the tolerance class); `exact` (bitwise), `big` and `off` are asked for by name
FAST_MODE: Optional[str] = "fast"                      # the tolerance class's mode name (cond_dedup + attn_bf16 + attn_cudnn + dit_fused on top of exact's levers)

# lever directories under opt/ (stack.opt_home())
KIT_XATTEMPT = os.path.join("forward", "xattempt_addon")      # the runner (xa_run.py), fastinit (xa_fastinit.py), hoist (xa_hoist.py)
KIT_PARTNER = os.path.join("forward", "fast_inference")       # the seed hook (sitecustomize.py -> bg_hook.py), inproc (bg_inproc.py), graph_sampler (bg_graph_patch.py)
KIT_SIZE_LEVERS = os.path.join("forward", "size_levers")      # sz_levers.py: td_chunk (big only) and the out-of-memory trace (every kit mode)
KIT_FAST_LEVERS = os.path.join("forward", "fast_levers")      # fl_levers.py: cond_dedup, attn_bf16, attn_cudnn, dit_fused (fast only)
KIT_HOST_LEVERS = os.path.join("host", "writer_levers")        # hl_levers.py: async_writer (every kit mode)
KITS: Tuple[str, ...] = (KIT_XATTEMPT, KIT_PARTNER, KIT_SIZE_LEVERS, KIT_FAST_LEVERS, KIT_HOST_LEVERS)

RUNNER = os.path.join("src", "xa_run.py")              # under KIT_XATTEMPT: one configured job dir, one seed, GPU steps in-process
SITECUSTOMIZE = os.path.join("src", "sitecustomize.py")

# the lever modules' switch names: everything under these prefixes is theirs; a mode exports kit_defaults() plus its own env_overrides, nothing else
KIT_SWITCH_PREFIXES: Tuple[str, ...] = ("BG_", "XA_", "SZ_", "FL_", "HL_")   # BG_ (fast_inference), XA_ (xattempt_addon), SZ_ (size_levers), FL_ (fast_levers), HL_ (writer_levers): a mode exports its own values; a caller's are dropped and reported (stack.mode_env)
OBSERVABILITY_SWITCHES: Tuple[str, ...] = ("BG_TIMING_FILE",)   # the hook's timing-file path (design.kit_env sets it per run), kept when set
EXCLUDED_VALUES: Dict[str, Tuple[str, ...]] = {"BG_GRAPH": ("predraw", "off")}   # bg_graph_patch's other BG_GRAPH values: never accepted as a runner default (resolve() checks them before a mode's env_overrides apply)


@dataclass(frozen=True)
class KitMode:
    """What a package mode is made of: lever directories in PYTHONPATH order (the first is the runner's home), and whether the
    mode exports the runner's own switch defaults.

    ``env_overrides``: exports the mode declares on top of (or instead of) the runner's own defaults — applied AFTER
    ``EXCLUDED_VALUES`` is checked against the raw runner defaults, so a mode's own deliberate choice (``big``'s
    ``BG_GRAPH=off``) never bypasses the check for a value it did not itself declare.
    ``house_kits``: lever directories the package adds on PYTHONPATH after ``kits``; neither the runner nor ``sitecustomize.py``
    imports their modules, so the package imports them itself (``own_imports``).
    ``own_imports``: modules the package itself imports for this mode (``hook_imports()`` / ``runner_imports()`` only ever see
    what ``sitecustomize.py`` and ``xa_run.py`` import)."""
    kits: Tuple[str, ...]
    exports_defaults: bool
    levers: Tuple[str, ...]
    env_overrides: Dict[str, str] = field(default_factory=dict)
    house_kits: Tuple[str, ...] = ()
    own_imports: Tuple[str, ...] = ()
    kernels: Tuple[str, ...] = ()                                   # the upstream accelerators this mode routes through unchanged (census.ACCELERATORS names): expected `engaged` wherever upstream switches them on
    kernels_replaced: Dict[str, str] = field(default_factory=dict)  # accelerator -> the lever that replaces it in this mode (expected `off-by-route:<lever>`); none here


UPSTREAM_ACCELERATORS: Tuple[str, ...] = ("cueq_triatt", "cueq_trimul")   # what upstream engages (census.ACCELERATORS is held equal by a test); the stock arm routes through all of them by definition

# the one place a mode's composition lives (lever names: registry.LEVERS)
KIT_MODES: Dict[str, KitMode] = {
    "exact": KitMode(kits=(KIT_XATTEMPT, KIT_PARTNER), exports_defaults=True,
                     levers=("inproc", "graph_sampler", "fastinit", "hoist", "async_writer"),
                     env_overrides={"HL_ASYNC_WRITER": "1"},
                     house_kits=(KIT_SIZE_LEVERS, KIT_HOST_LEVERS), own_imports=("sz_levers", "hl_levers"),
                     kernels=UPSTREAM_ACCELERATORS),
    "big": KitMode(kits=(KIT_XATTEMPT, KIT_PARTNER), exports_defaults=True,
                     levers=("inproc", "fastinit", "td_chunk", "async_writer"),
                     env_overrides={"BG_GRAPH": "off", "XA_HOIST": "0", "SZ_TD_CHUNK": "64",
                                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HL_ASYNC_WRITER": "1"},
                     house_kits=(KIT_SIZE_LEVERS, KIT_HOST_LEVERS), own_imports=("sz_levers", "hl_levers"),
                     kernels=UPSTREAM_ACCELERATORS),
    "fast": KitMode(kits=(KIT_XATTEMPT, KIT_PARTNER), exports_defaults=True,
                    levers=("inproc", "graph_sampler", "fastinit", "hoist", "async_writer", "cond_dedup", "attn_bf16", "attn_cudnn", "dit_fused"),
                    env_overrides={"HL_ASYNC_WRITER": "1", "FL_COND_DEDUP": "1", "FL_ATTN_BF16": "1", "FL_ATTN_BACKEND": "cudnn", "FL_DIT_FUSED": "1"},
                    house_kits=(KIT_FAST_LEVERS, KIT_SIZE_LEVERS, KIT_HOST_LEVERS), own_imports=("fl_levers", "sz_levers", "hl_levers"),
                    kernels=UPSTREAM_ACCELERATORS),
}


def kernels_of(mode: str) -> Tuple[Tuple[str, ...], Dict[str, str]]:
    """``(routed, replaced)``: the upstream accelerators a mode routes through unchanged and those a lever of the mode replaces.
    ``off`` is the stock arm — upstream alone — so it routes through every accelerator upstream engages and replaces none."""
    mode = check_mode(mode)
    if mode == "off":
        return UPSTREAM_ACCELERATORS, {}
    km = KIT_MODES[mode]
    return tuple(km.kernels), dict(km.kernels_replaced)


@dataclass
class Resolution:
    mode: str
    kits: Tuple[str, ...]                     # PYTHONPATH order (relative to opt/)
    env: Dict[str, str]                       # exported at activation (the runner's own defaults, then the mode's own overrides) — {} for off
    imports: Tuple[str, ...]                  # the in-process form: the hook, then the runner's lever imports, then the mode's own
    levers: Tuple[str, ...]
    runner: Optional[str]                     # path relative to opt/
    notes: list = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.mode != "off"


def check_mode(mode: Optional[str]) -> str:
    """The one validator: None -> DEFAULT_MODE; anything not in MODES is an error that names the table."""
    if mode is None or mode == "":
        return DEFAULT_MODE
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}: the modes are {'|'.join(MODES)}")
    return mode


def mode_from_env(environ=None) -> Optional[str]:
    """``BOLTZGEN_OPT`` unset or empty -> None (off: nothing is armed); otherwise its value, validated by the caller."""
    environ = os.environ if environ is None else environ
    v = environ.get(ENV)
    return v if v else None


# --------------------------------------------------------------------------------------------- the lever switch table
def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def kit_defaults(runner_path: str) -> Dict[str, str]:
    """The runner's own switch defaults: every ``os.environ.setdefault(<name>, <value>)`` in ``xa_run.py``, in file order."""
    tree = ast.parse(_read(runner_path))
    out: Dict[str, str] = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "setdefault"
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "environ" and len(node.args) == 2
                and all(isinstance(a, ast.Constant) and isinstance(a.value, str) for a in node.args)):
            out[node.args[0].value] = node.args[1].value
    if not out:
        raise ValueError(f"no os.environ.setdefault(<name>, <value>) call in {runner_path}: not the pinned runner")
    return out


def _top_level_imports(path: str, prefixes: Tuple[str, ...]) -> Tuple[str, ...]:
    tree = ast.parse(_read(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(prefixes) and alias.name not in names:
                    names.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module and node.module.startswith(prefixes) and node.module not in names:
            names.append(node.module)
    return tuple(names)


def runner_imports(runner_path: str) -> Tuple[str, ...]:
    """The lever modules the runner imports for a GPU step (``import xa_fastinit`` / ``import xa_hoist``), in its order."""
    names = _top_level_imports(runner_path, ("xa_",))
    if not names:
        raise ValueError(f"no `import xa_*` in {runner_path}: not the pinned runner")
    return names


def hook_imports(sitecustomize_path: str) -> Tuple[str, ...]:
    """What ``fast_inference``'s ``sitecustomize.py`` imports in every python process (``bg_hook``)."""
    names = _top_level_imports(sitecustomize_path, ("bg_",))
    if not names:
        raise ValueError(f"no `import bg_*` in {sitecustomize_path}: not the partner's sitecustomize")
    return names


def resolve(mode: Optional[str], opt_home: str) -> Resolution:
    """Package mode -> Resolution, read from the lever directories' own files under ``opt_home`` (nothing transcribed except a
    mode's own declared ``env_overrides``/``house_kits``/``own_imports`` — the ``KIT_MODES`` table)."""
    mode = check_mode(mode)
    if mode == "off":
        return Resolution(mode="off", kits=(), env={}, imports=(), levers=(), runner=None,
                          notes=["upstream alone: no kit variable, no kit module, no kit directory on sys.path"])
    km = KIT_MODES[mode]
    runner = os.path.join(KIT_XATTEMPT, RUNNER)
    env = kit_defaults(os.path.join(opt_home, runner)) if km.exports_defaults else {}
    for k, v in env.items():
        if v in EXCLUDED_VALUES.get(k, ()):
            raise ValueError(f"the runner's default {k}={v} is a value outside the kit line (modes.EXCLUDED_VALUES): not the pinned runner")
    env.update(km.env_overrides)                          # the mode's own declared row — after the drift check, never before it
    imports = (hook_imports(os.path.join(opt_home, KIT_PARTNER, SITECUSTOMIZE))
               + runner_imports(os.path.join(opt_home, runner)) + km.own_imports)
    return Resolution(mode=mode, kits=km.kits + km.house_kits, env=env, imports=imports, levers=km.levers, runner=runner)


def describe_line(res: Resolution) -> str:
    """The mode's kit spelling for the activation line: the exported switches, ``stock`` for off."""
    if not res.active:
        return "stock"
    return ",".join(f"{k}={v}" for k, v in res.env.items()) or "no-switch"
