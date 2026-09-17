"""Activation: version gate, kit paths, GPU probe, cell tables, and the lever application.

`activate(mode)` resolves the mode's environment by sourcing the kit's env.sh (modes.resolve), exports it — before anything initialises
CUDA, so the allocator policy in PYTORCH_CUDA_ALLOC_CONF takes effect — puts env.sh's PYTHONPATH entries on sys.path (kit_sys_path), then executes the FlashPairformer kit's own
`opt/forward/flashpairformer/src/sitecustomize.py` (runpy) with PTX_LAZY_INIT masked for the duration, so the kit takes its
meta-path-hook route, and finally installs lazy init explicitly. The kit's sitecustomize
registers the lever-report atexit hook; `report.register_exit_tally()` runs after it. No lever logic and no orchestration copy live
here: the report is read from the kit modules' own `_STATS` records. `activate(mode, dry_run=True)` resolves, gates and reports
without applying anything (`check`).
"""
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
import runpy
import sys
from typing import Dict, List, Optional, Tuple

from . import ActivationError, __version__
from . import _core  # noqa: F401  (opt_core importable: installed, else the pinned path)
from . import runner_hooks as _runner_hooks
from . import templates as _templates
from . import sampler_fuse as _sampler_fuse
from . import tp
from .modes import ARM, MODES, BIG_BASE, big_levers, resolve, nvidia_smi_compute_cap
from .registry import LEVERS, BLK2_LEVERS, NO_BLK2_CELLS, NOT_IN_ROW, STOCK_TRIATT, TRIATT_STATEMENT_LEVERS, in_row
from .kits import CORE_SERVED_KERNELS, CORE_RELEASED_KERNELS, CORE_KERNELS_DIR
from . import report as _report
from . import ablation as _ablation
from . import apb_levers as _apb
from . import sampler_levers as _sl
from . import sampler_poison_aside as _poison_aside
from . import route_preload as _route_preload                      # the small-N gate package's TriMul callees, imported at activation from outside that package
from opt_core import gates as _core_gates, home as _home, instances as _instances, kernels as _kernels
from opt_core.gates import sha256_file  # noqa: F401  (the package's file hasher: the core's; tests read it here)

ENV_MODE, ENV_FORCE = "PROTENIX_OPT", "PROTENIX_OPT_FORCE"
ENV_HOME = "PROTENIX_OPT_HOME"
ALLOC_CONF = "PYTORCH_CUDA_ALLOC_CONF"                                 # the allocator policy env.sh exports (xl_policy): read by torch at CUDA initialisation
SUPPORTED_CC = {"9.0", "10.0", "8.0", "10.3"}                     # the README's supported GPUs (H100/H200 = 9.0, B200 = 10.0, A100 = 8.0, B300 = 10.3) as data: manifest gpu_supported only; never a gate
LOG = _report.PREFIX
DIRECT_APPLY_SIGNATURE = "trunk-II levers applied (direct)"      # the kit's sitecustomize: the branch that never calls fpf.enable_from_env()
HOOK_APPLY_SIGNATURE = "trunk-II levers applied:"                 # the kit's sitecustomize: the meta-path-hook route
SITECUSTOMIZE_RELPATH = os.path.join("src", "sitecustomize.py")
GRAPHED_PY = os.path.join("infopt_graphs", "protenix", "graphed.py")   # the copy the CLI route imports resolves through env.sh L6 (src/, first on the path)
MODEL_MODULE = "protenix.model.protenix"                          # where stock's Protenix model class lives
MODEL_CLASS = "Protenix"
PAIRFORMER_MODULE = "protenix.model.modules.pairformer"
KIT_CLASS_DIRNAMES = ("forward",)                                 # opt/<class>/: the kit directories (kit_class_dirs)
DEFAULT_STOCK_ENV_ABSENT = ("PTX_", "FPF_", "INFOPT_", "PF_", "PROTENIX_OPT", "CUEQ_TRITON_CACHE_DIR")   # stock/PINS.json carries the same list

_REPORT: Optional[dict] = None
_ACTIVATING = False
KERNEL_ROUTES = tuple(CORE_RELEASED_KERNELS) + CORE_SERVED_KERNELS   # carried kernels served from the core's copy (opt_core.kernels): byte-identical carries (the pair-stack TriMul is opt_core.kernels.trimul by tier word — no routed TriMul package)
KERNEL_EXPORTS = {"fpf_triatt_k2b": lambda fpf_home: {"cells": os.path.join(CORE_KERNELS_DIR, "fpf_triatt_k2b", "K2B_CELLS.json")}}      # K2B reads its launch-cell table PF_TRIATTN_TABLE = the core copy's K2B_CELLS.json (the kit lever keeps a pre-set caller value)
CALLER_EXPORTS = ("PF_TRIATTN_TABLE",)   # exports a caller may pre-set (the kit documents PF_TRIATTN_TABLE=<cells json> as an opt-in): filled only when unset, the caller's value is kept and recorded
GPU_KEYS = ("name", "sm", "cc", "probe")                          # the GPU probe dict (report gpu=, manifest gpu): the core's probes projected to it


# ---------------------------------------------------------------------------------------------------------------- paths ----
def opt_home() -> str:
    """protenix_v2/opt — the directory this package lives in (editable install). A wheel installed into site-packages has no kit
    beside it: then PROTENIX_OPT_HOME=<protenix_v2/opt> or MODEL_OPT=<protenix_v2> (run.sh and configs export it) names the tree."""
    for var, cand in ((ENV_HOME, os.environ.get(ENV_HOME)), ("MODEL_OPT", os.path.join(os.environ["MODEL_OPT"], "opt") if os.environ.get("MODEL_OPT") else None)):
        if not cand:
            continue
        if os.path.isfile(os.path.join(cand, "forward", "flashpairformer", "env.sh")):
            return os.path.abspath(cand)
        raise FileNotFoundError(f"{var}={os.environ[var]} does not contain the kit (expected {os.path.join(cand, 'forward', 'flashpairformer', 'env.sh')})")
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def tree_home() -> str:
    """protenix_v2/ — the tree root (configs/, stock/, opt/, tests/)."""
    return os.path.dirname(opt_home())


def kit_home() -> str:
    """The FlashPairformer kit directory (env.sh's FPF_HOME)."""
    return os.path.join(opt_home(), "forward", "flashpairformer")


def pins_path() -> str:
    return os.path.join(tree_home(), "stock", "PINS.json")


def pins() -> dict:
    """stock/PINS.json — the machine-readable stock pins (protenix version, wheel and checkpoint sha256, upstream commit)."""
    with open(pins_path()) as fh:
        return json.load(fh)


def protenix_pin() -> str:
    return pins()["protenix_version"]


def stock_env_absent(p: Optional[dict] = None) -> Tuple[str, ...]:
    """The environment-name prefixes a stock process must not carry (stock/PINS.json ``stock_environment.must_be_absent_prefixes``:
    the kit's switch families, the package's own switches, the cuEquivariance Triton cache override), plus the package's own
    switches should the file omit them. Every entry is a prefix (``PROTENIX_OPT`` covers ``PROTENIX_OPT_FORCE`` ...)."""
    p = p if p is not None else pins()
    spec = list((p.get("stock_environment") or {}).get("must_be_absent_prefixes") or DEFAULT_STOCK_ENV_ABSENT)
    for own in (ENV_MODE, "PTX_"):
        if not any(own.startswith(s) for s in spec):
            spec.append(own)
    return tuple(spec)


def kit_class_dirs() -> List[str]:
    """The kit directories of the tree (``opt/forward``): a sys.path or PYTHONPATH entry inside one of them puts kit code on the path."""
    return [os.path.join(opt_home(), d) for d in KIT_CLASS_DIRNAMES]


def kit_sys_path(pythonpath: Optional[List[str]] = None, fpf_home: Optional[str] = None, mode: str = "exact") -> List[str]:
    """sys.path entries for a mode ("off": none). First the PYTHONPATH env.sh L6 leaves, in its order — the FlashPairformer kit's entries
    as sourced (`src` with the kit's `ptx_lazy_init`, `third_party` with its vendored `fastln_prebuilt*` packages, `levers_addon/PTXV2_LEVERS_ADDON_v1`)
    and the caller's own PYTHONPATH entries where env.sh places them (after the add-on: kit README 'Order of env operations'
    "Order of env operations"; the CLI route imports the kit's HAZARD44-guarded `infopt_graphs` from `src`, first on the path). One `sitecustomize` is executed per process (the FlashPairformer kit's, which carries the levers
    add-on's env-gated import: sitecustomize.py "composition with the levers add-on"); `ptx_addon_levers` is the add-on's (env.sh puts it first)."""
    if mode == "off":
        return []
    h = fpf_home or kit_home()
    if pythonpath is None:
        pythonpath = env_sh_pythonpath(h)
    out = []
    for p in list(pythonpath):
        if p not in out:
            out.append(p)
    return out


def graphed_py(entries: List[str]) -> Optional[str]:
    """The `infopt_graphs/protenix/graphed.py` a process with these sys.path entries imports: the first entry carrying the
    `infopt_graphs` package decides (regular-package resolution), so this is the file `import infopt_graphs.protenix.graphed` loads."""
    for e in entries:
        if os.path.isfile(os.path.join(e, "infopt_graphs", "__init__.py")):
            p = os.path.join(e, GRAPHED_PY)
            return p if os.path.isfile(p) else None
    return None


def graphed_py_sha256(entries: List[str]) -> Optional[str]:
    p = graphed_py(entries)
    return sha256_file(p) if p else None


def env_sh_pythonpath(fpf_home: Optional[str] = None) -> List[str]:
    """The entries env.sh L6 exports, in order (without the caller's own PYTHONPATH), read from the file by sourcing it."""
    h = fpf_home or kit_home()
    return resolve("exact", {"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "")}, h, probe_gpu=False).pythonpath


def _install_sys_path(entries: List[str]) -> None:
    """Put `entries` at the front of sys.path in that order, after the script directory (PYTHONPATH semantics); an entry already on
    sys.path (the caller's own PYTHONPATH) moves to its place in the list (opt_core.home.place_on_sys_path)."""
    script_dir = sys.path[0] if sys.path and sys.path[0] in ("", os.getcwd(), os.path.dirname(os.path.abspath(sys.argv[0] or ""))) else None
    _home.place_on_sys_path(entries, after=script_dir)


# ---------------------------------------------------------------------------------------------------------------- gates ----
def protenix_version() -> Optional[str]:
    return _core_gates.dist_version("protenix")


def version_gate(force: bool = False) -> Tuple[bool, Optional[str], Optional[str]]:
    """(ok, installed version, reason). Refuses unless protenix == stock/PINS.json protenix_version; PROTENIX_OPT_FORCE=1 overrides with a warning."""
    pin = protenix_pin()
    v = protenix_version()
    if v == pin:
        return True, v, None
    reason = (f"protenix not installed (this kit targets {pin} exactly)" if v is None else f"protenix=={v}; this kit targets {pin} exactly")
    if force:
        sys.stderr.write(f"{LOG} WARNING: {reason} — {ENV_FORCE}=1 overrides the version gate; levers are untested on this install\n")
        return True, v, reason
    return False, v, reason


def gpu_probe_smi() -> dict:
    """The GPU without initialising CUDA: nvidia-smi name + compute capability (the probe env.sh itself makes; opt_core.gates.nvidia_smi_probe).
    {"name","sm","cc","probe"} — Nones with the probe naming why when there is no device."""
    return _core_gates.nvidia_smi_probe(keys=GPU_KEYS)


def gpu_info() -> dict:
    """The GPU via torch (initialises CUDA: call it only after the environment is exported; opt_core.gates.torch_gpu_probe)."""
    return _core_gates.torch_gpu_probe(keys=GPU_KEYS)


def kit_version() -> Optional[str]:
    """The FlashPairformer unit's version label (kits.KITS; the `kit=` token of env.sh's KIT_SPEC line)."""
    from . import kits
    return kits.labels()["flashpairformer"]


# ---------------------------------------------------------------------------------------------------------------- cells ----
def cells(sm: Optional[str], fpf_home: Optional[str] = None) -> dict:
    """Cell view for one arch class from the kit's CELLS.json (+ K2B_CELLS.json): which levers have a cell here."""
    h = fpf_home or kit_home()
    out: dict = {"sm": sm, "cells_json": os.path.join(h, "CELLS.json")}
    try:
        c = json.load(open(out["cells_json"]))
    except Exception as e:
        out["error"] = repr(e)
        return out
    tag = f"'{sm}'"
    out["blk2_arch"] = list(c.get("blk2_arch", []))
    out["blk2"] = sm in out["blk2_arch"]
    out["triatt_stock"] = sm in c.get("blk2_triatt_stock_arch", [])       # the block path engages here with its in-block transition; its tri-attention statement is stock (no prologue / epilogue cells for the arch)
    out["small_m_stock_rows"] = int((c.get("small_m_stock_rows") or {}).get(sm, 0) or 0) if sm else 0   # GEMM-row floor below which the fused transition kernels run the stock body on this arch (0 = no row; cc 8.0 carries one: GUARD F1)
    for sec in ("prologue", "epilogue", "t1_fused"):
        out[sec] = sorted(k for k in c.get(sec, {}) if sm and tag in k)
    out["smalln_trimul_exact_below"] = c.get("smalln_trimul_exact_below", {})
    out["by_triton"] = sorted(c.get("by_triton", {}))
    k2b = os.path.join(CORE_KERNELS_DIR, "fpf_triatt_k2b", "K2B_CELLS.json")   # the core copy (kits.CORE_SERVED_KERNELS)
    try:
        k = json.load(open(k2b))
        cc = f"{sm[2:-1]}.{sm[-1]}" if sm and sm.startswith("sm") and len(sm) >= 4 else None
        out["k2b"] = {"by_cc": sorted(k.get("by_cc", {})), "this_cc": (k.get("by_cc", {}) or {}).get(cc)}
    except Exception as e:
        out["k2b"] = {"error": repr(e)}
    return out


# ------------------------------------------------------------------------------------------------------------- activation ----
MARKERS: Dict[str, Tuple[str, Tuple[str, ...]]] = {    # registry lever -> (family prefix of the kit's `applied` entries, substrings that mean "on")
    "t1_fused_transition": ("T1:", ("T1:",)),
    "nomask": ("NM:", ("NM:1",)),
    "blk2_block_path": ("BLK", ("BLK:2(",)),
    "pwa_zcache": ("PWAZ:", ("PWAZ:",)),
    "ln_core": ("LNCORE:", ("LNCORE:on(",)),                                            # src/protenix_ptx_ln_core.py via ptx_trunk2_levers.apply_from_env; LNCORE:unavailable(...) carries BAD -> refused by name
    "blk2_chunked_exact": ("BLK", ("chunked=1)",)),
    "blk2_chunked_k2b": ("BLK", ("chunked=k2b)",)),
    "k2b_flash_triattention": ("BLK", ("BLK:2(pro+K2B", "BLK:2(pro+PROVIDER(k2b")),
    "triattn_native": ("BLK", ("BLK:2(pro+PROVIDER(ptx_native_core:attn",)),                   # ptx_trunk2_levers' provider slot bound to opt/forward/flashpairformer/src/ptx_native_core.py (env.sh PTX_T_ATT=fast|big: opt_core.kernels.triattn by the tier word); an install refusal is 'BLK_ATT:provider ptx_native_core:attn unavailable(' (BAD) and the BLK line names cueq -> NOT ACTIVE by name
    "deadskip": ("DEADSKIP:", ("DEADSKIP:hooked",)),
    "stackgraph": ("v02:GRAPH:", ("v02:GRAPH:",)),
    "trimul_core_exact": ("FPF:", ("FPF:enabled(",)),               # + MODULE_PROOF: the FPF_OPS provider src/ptx_trimul_routes.py must have loaded
    "trimul_core": ("FPF:", ("FPF:enabled(",)),
    "smalln_size_gate": ("FPF:", ("FPF:enabled(",)),
    "template_dedupe": ("TEMPL_DEDUPE:", ("TEMPL_DEDUPE:",)),
    "sampler_graph": ("SAMPLER:", ("SAMPLER:hook",)),
    "lazy_init": ("LAZY_INIT:", ("LAZY_INIT:1",)),
    "guard_lift": ("GUARD_LIFT:", ("GUARD_LIFT:1",)),
    "sampler_admit": ("SAMPLER:", ("SAMPLER:hook",)),                        # armed with the sampler hook; its words + verdict census ride the clisampler LATE record (policy)
    "pred_release": ("PRED_RELEASE:", ("PRED_RELEASE:armed", "PRED_RELEASE:patched")),   # stack.activate: pred_release.install() (the core's patch_attr_at_import on InferenceRunner.predict)
    "sampler_fuse": (_sampler_fuse.MARK, (_sampler_fuse.MARK,)),
    "keep_pool": ("KEEP_POOL:", ("KEEP_POOL:on(",)),
    "summary_hostidx": ("SUMHOST:", ("SUMHOST:on(",)),                                  # SUMHOST:unavailable(...) (pinned stock surface drifted) carries BAD -> the mode refuses by name
    "pf_attn": ("PF_ATTN:", ("PF_ATTN:",)), "opm_fused": ("OPM_FUSED:", ("OPM_FUSED:",)), "pwa_fused": ("PWA_FUSED:", ("PWA_FUSED:",)),
    "cond_dedupe": ("CONDDEDUPE:", ("CONDDEDUPE:",)), "dit_fused": ("DITFAST:", ("DITFAST:",)), "dit_lowp": ("DITLOWP:", ("DITLOWP:",)),   # sampler_levers: '<MARK>armed|patched' at activation; ':unavailable(' carries BAD; reconcile proves the install from sampler_levers.state()
    "atom_fused": ("ATOMFAST:", ("ATOMFAST:",)), "atom_attn_exact": ("ATOMATTNEXACT:", ("ATOMATTNEXACT:",)),   # as dit_attn: '<MARK>:patched' at activation (the seam armed), ':on(' after the runner is built; reconcile proves installed_on > 0 from apb_levers.state()
    "triatt_prologue_cuda": ("PROCUDA:", ("PROCUDA:on(",)),                                 # unavailable / REFUSED carry BAD -> the mode refuses by name
    "transition_core_exact": ("TRCORE:", ("TRCORE:on(word=exact",)),                       # ptx_transition_core.apply: TRCORE:on(word=<w> bind=tier:<w> ...); :unavailable( / :refused( carry BAD -> the mode refuses by name
    "triatt_exact": ("TRIATT_EXACT:", ("TRIATT_EXACT:on(",)),                          # fpf.triatt_exact.apply via ptx_trunk2_levers.apply_from_env: TRIATT_EXACT:on(opt_core.kernels.triattn <v> word=exact ...); :unavailable( carries BAD -> the mode refuses by name
    "transition_core": ("TRCORE:", ("TRCORE:on(word=fast", "TRCORE:on(word=big")),
    "dit_attn_exact": ("DITATTN:", ("DITATTN:on(",)),
    "dit_attn": (_apb.MARKS["dit_attn"], (_apb.MARKS["dit_attn"],)),            # apb_levers: <FAMILY>armed|patched at activation; the install names itself per runner (<FAMILY>on(...))
    "dit_attn_fp16": (_apb.MARKS["dit_attn_fp16"], (_apb.MARKS["dit_attn_fp16"],)),   # the precision lever: DIT_ATTN_FP16:requested at activation, DIT_ATTN_FP16:on at dit_attn's install
    "atom_attn": (_apb.MARKS["atom_attn"], (_apb.MARKS["atom_attn"],)),
    "drop_bond_mask": ("MEM:drop_bond_mask", ("MEM:drop_bond_mask(",)),        # big.py: MEM:<lever>(applied|armed) | MEM:<lever>:refused(<precondition>)
    "cond_chunk": ("MEM:cond_chunk", ("MEM:cond_chunk(",)),
    "apb_bias_chunk": ("MEM:apb_bias_chunk", ("MEM:apb_bias_chunk(",)),
    "cache_release": ("MEM:cache_release", ("MEM:cache_release(",)),
    "relp_lazy": ("MEM:relp_lazy", ("MEM:relp_lazy(",)),
    "msa_zfree": ("MEM:msa_zfree", ("MEM:msa_zfree(",)),
    "diffcache_free": ("MEM:diffcache_free", ("MEM:diffcache_free(",)),
}
MODULE_PROOF = {"trimul_core_exact": "ptx_trimul_routes", "trimul_core": "ptx_trimul_routes", "smalln_size_gate": "fpf_smalln"}   # the FPF_OPS provider module that must have loaded
BAD = ("unavailable(", "refused(", "inactive", "FAILED", "skipped(")


def _classify(mode: str, applied: List[str], environ, notes: Optional[List[str]] = None, row=None, row_key: str = "other",
              sm: Optional[str] = None) -> Tuple[List[str], List[str], List[str], dict]:
    """Split the mode's levers into applied / fallback / not-in-this-arm using the kit's own markers. Every lever lands in exactly one
    list. `row` is the kit README row for the box's kernel key (modes.readme_row): a row-dependent lever (pad8, glue v2, MK-PF,
    the tri-attention tier word) that the row does not list is not part of this arm on this key — neither applied nor unavailable."""
    on, fb, skipped, why = [], [], [], {}
    cuda_early = any("CUDA was initialised before" in n for n in (notes or []))
    row_levers = _row_levers(mode, environ)
    blk2_here = cells(sm).get("blk2") if sm else None                  # this arch has BLK2 cells (CELLS.json blk2_arch); None: no device probed
    triatt_stock = bool(sm) and bool(cells(sm).get("triatt_stock"))    # this arch runs the block core's tri-attention as the stock statement (CELLS.json blk2_triatt_stock_arch)
    for name in row_levers:
        lv = LEVERS[name]
        if name in BLK2_LEVERS and blk2_here is False:               # the block path cannot engage at all on this arch: unavailable (the mode refuses by name)
            fb.append(name); why[name] = f"{NO_BLK2_CELLS} for gpu_arch={sm} (CELLS.json blk2_arch): the block path cannot engage on this card (no launch cells: the mode refuses by name rather than run without it)"
            continue
        repl = [r for r in lv.replaced_by if r in row_levers and (not LEVERS[r].row_dependent or in_row(LEVERS[r], row))]
        if repl:                                                          # another lever of this row takes this lever's slot on this card (k2b -> triattn_cuda on cc 9.0): not part of the arm here, never a fallback
            skipped.append(name); why[name] = f"{NOT_IN_ROW}{row_key} (its slot is served by {','.join(repl)} on this row)"
            continue
        if lv.row_dependent and not in_row(lv, row):
            skipped.append(name); why[name] = f"{NOT_IN_ROW}{row_key} (the compositions table: its kernels have no cells for this key)"
            continue
        if triatt_stock and name in TRIATT_STATEMENT_LEVERS:              # the lever lives in the fused tri-attention statement, which this arch runs as the stock statement: not part of the arm on this card
            skipped.append(name); why[name] = f"{STOCK_TRIATT}{sm} (CELLS.json blk2_triatt_stock_arch: no tri-attention prologue / epilogue cells for this arch; the block path keeps its in-block transition)"
            continue
        if name == "sampler_fuse" and environ.get(lv.env_keys[0]) == "0":   # a lever the caller removed by its own switch (PTX_SAMPLER_FUSE=0)
            skipped.append(name); why[name] = f"off by flag {lv.env_keys[0]}=0 (a recorded opt-out: the record's off_by_flag; not a fallback)"
            continue
        if lv.probe == "env":
            if name == "fastln_prebuilt":
                if environ.get("INFOPT_FASTLN_PREBUILT"):
                    on.append(name)
                else:
                    fb.append(name); why[name] = (f"no $FPF_HOME/third_party/fastln_prebuilt*/manifest.json matches torch {_core_gates.dist_version('torch')} "
                                                  "(env.sh L75-86); the stack graph source-rebuilds the stream LN per process (outputs identical)")
            elif name == "xl_policy" and cuda_early:
                fb.append(name); why[name] = "PYTORCH_CUDA_ALLOC_CONF exported after CUDA was initialised in this process: allocator policy not in effect"
            elif all(environ.get(k) for k in lv.env_keys):
                on.append(name)
            else:
                fb.append(name); why[name] = "switch not in the environment: " + ",".join(k for k in lv.env_keys if not environ.get(k))
            continue
        family, good_marks = MARKERS[name]
        family_hits = [a for a in applied if a.startswith(family)]
        good = [a for a in family_hits if any(m.casefold() in a.casefold() for m in good_marks) and not any(b in a for b in BAD)]
        proof = MODULE_PROOF.get(name)
        if good and proof and proof not in sys.modules:
            fb.append(name); why[name] = f"{proof} not loaded ({good[0]})"
        elif good:
            on.append(name)
        else:
            fb.append(name); why[name] = family_hits[0] if family_hits else "no marker in the kit's applied list"
    return on, fb, skipped, why


def _row_levers_full(mode: str, environ=None) -> List[str]:
    """The registry levers of `mode`'s row: big's is its base's composition (modes.big_levers)."""
    return big_levers() if mode == "big" else MODES[mode]


def _row_levers(mode: str, environ=None) -> List[str]:
    """The registry levers the process expects of `mode`'s row: the full row less the levers ``MODEL_OPT_LEVERS_OFF`` ablates
    (ablation.requested; validated by the activation before anything is applied)."""
    environ = os.environ if environ is None else environ
    off = _ablation.requested(environ)
    return [n for n in _row_levers_full(mode, environ) if n not in off]


# ------------------------------------------------------------------------------------------------- late records ----
LATE_RECORDS: Dict[str, Tuple[str, str]] = {          # registry lever -> (key of the kit's end-of-run record in $PTX_LEVER_REPORT, the kit module whose report() writes it)
    "sampler_graph": ("clisampler", "fpf_clisampler.clisampler"),               # installs after InferenceRunner.init_model: after the activation report
    "sampler_graph_cache_policy": ("clisampler", "fpf_clisampler.clisampler"),  # the graph cache policy governs the sampler graphs: it follows their record
    "sampler_admit": ("clisampler", "fpf_clisampler.clisampler"),               # the sampler admission policy installs with the sampler hook: it follows its record (+ record["policy"]: absent / policy_err = fallback by name)
    "sampler_reach": ("clisampler", "fpf_clisampler.clisampler"),               # the graphed sampler's reach: the same record's reach= word (on(...) or the lever is a fallback by name)
    "sampler_prep": ("clisampler", "fpf_clisampler.clisampler"),                # the sampler's host path: the same record carries prep= (the parts word) / prep_poison / prep_stats
    "stackgraph": ("stackgraph", "fpf_stackgraph.stackgraph"),                  # installs at the pairformer import; its record says whether it did
}


def private_report_dir(environ=None) -> Optional[str]:
    """The directory of ``$PTX_LEVER_REPORT`` (env.sh's default: under the user's own cache directory), made with mode 0700 when absent.
    The kit appends its end-of-run records to that file and reads them back, so a directory another account could have planted is never
    used: an existing one that is a symbolic link, belongs to another uid or carries a group/other write bit is refused — one stderr line
    names it, the reason and the fix — and PTX_LEVER_REPORT is removed from the environment (nothing is appended or read; the run itself is
    unaffected). Returns the directory in use, else None."""
    import stat
    env = os.environ if environ is None else environ
    path = env.get("PTX_LEVER_REPORT")
    if not path:
        return None
    d = os.path.dirname(os.path.abspath(path))
    try:
        if not os.path.lexists(d):
            os.makedirs(os.path.dirname(d), exist_ok=True)
            os.mkdir(d, 0o700)
        st = os.lstat(d)
        why = None
        if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
            why = "is a symbolic link or not a directory"
        elif st.st_uid != os.geteuid():
            why = f"belongs to uid {st.st_uid}, not to this process (uid {os.geteuid()})"
        elif st.st_mode & 0o022:
            why = f"is writable by group or other (mode {stat.S_IMODE(st.st_mode):04o})"
    except OSError as e:
        why = f"cannot be made ({e.strerror})"
    if why is None:
        return d
    print(f"[protenix-opt] lever report: REFUSED {d}: it {why}; nothing is appended to or read from {path} in this run; "
          f"fix: remove it, or set PTX_LEVER_REPORT to a file of your own", file=sys.stderr, flush=True)
    env.pop("PTX_LEVER_REPORT", None)
    return None


def late_records(lever_report: Optional[str] = None, pid: Optional[int] = None) -> Dict[str, dict]:
    """The kit's own end-of-run records for the levers in LATE_RECORDS, keyed as the kit writes them (``clisampler``, ``stackgraph``):
    the record the kit module's ``report()`` returns in this process (the same dict its atexit dump writes to $PTX_LEVER_REPORT — read
    here because the manifest is written before that dump), else the record already in ``lever_report`` for ``pid`` (a subprocess, an
    earlier exit). A ``report()`` that raises is recorded as ``{"report_error": ...}``; a lever whose module never loaded has no record."""
    out: Dict[str, dict] = {}
    merged = _report.read_lever_report(lever_report, pid) if lever_report else None
    for key, modname in sorted(set(LATE_RECORDS.values())):
        if merged and isinstance(merged.get(key), dict):
            out[key] = dict(merged[key])
        m = sys.modules.get(modname)
        if m is not None and callable(getattr(m, "report", None)):
            try:
                out[key] = dict(m.report())
            except Exception as e:  # noqa: BLE001 — the kit's own report() failed: named in the record, never a silent gap
                out[key] = {"report_error": repr(e)}
    return out


SERVED: Dict[str, str] = {                               # registry lever -> how its served-call census is read at the end of the run (served_census)
    "k2b_flash_triattention": "fpf_smalln COUNTS att_k2b_calls vs att_cueq_calls at N >= its gate; ptx_trunk2_levers _XL_MEM lean_k2b_chunks / lean_cueq_chunks",
    "pf_attn": "apb_levers dtype-gate tally (dtype_gate): calls / served / aside by word — every call served or a declared aside (dtype_fp32|dtype_fp16) = by design",
    "opm_fused": "apb_levers dtype-gate tally (dtype_gate): calls / served / aside by word — every call served or a declared aside (dtype_fp32|dtype_fp16) = by design",
    "pwa_fused": "apb_levers dtype-gate tally (dtype_gate): calls / served / aside by word — every call served or a declared aside (dtype_fp32|dtype_fp16) = by design",
    "atom_attn_exact": "sampler_levers gate tally (dtype_gate): calls / served / aside by the package's Refused word (cublas_route_*) — every call served or a declared aside = by design",
}
GATED_SERVED: Tuple[str, ...] = ("pf_attn", "opm_fused", "pwa_fused", "atom_attn_exact")   # the SERVED levers read from the kit's per-call gate tallies (apb_levers / sampler_levers gate_census)


def served_census() -> Dict[str, Tuple[Optional[bool], str]]:
    """``{lever: (ok, detail)}`` for the levers of SERVED: ok False = the lever's kernel served NO call in this process although tri-attention statements ran at
    sizes inside its gate (example: `big` listed k2b_flash_triattention `on` while the XL lean statement ran cuEq for every call);
    ok None = nothing to judge (no statement ran / counters absent). Read from the kit modules' own counters; never raises."""
    out: Dict[str, Tuple[Optional[bool], str]] = {}
    try:
        L = sys.modules.get("ptx_trunk2_levers"); S = sys.modules.get("fpf_smalln")
        if L is not None:
            st = getattr(L, "_STATS", {}) or {}; xl = getattr(L, "_XL_MEM", {}) or {}
            cnt = (getattr(S, "COUNTS", None) or {}) if S is not None else {}
            k2b = int(cnt.get("att_k2b_calls", 0) or 0) + int(st.get("blk_att_k2b_routed", 0) or 0)
            cueq = int(cnt.get("att_cueq_calls", 0) or 0) + int(xl.get("lean_cueq_chunks", 0) or 0) + int(st.get("blk_att_chunked_cueq_instead_of_k2b", 0) or 0)
            stmts = int(st.get("blk2_tri_calls", 0) or 0) + int(st.get("blk2_tri_chunked_calls", 0) or 0)
            gate = int(getattr(S, "K2B_MIN_TOKENS", 0) or 0) if S is not None else 0
            nmax = int(xl.get("max_ntok", 0) or 0)
            detail = f"att_k2b={k2b} att_cueq={cueq} tri_stmts={stmts} max_ntok={nmax} gate={gate} lean_k2b_chunks={xl.get('lean_k2b_chunks', '-')} lean_cueq_chunks={xl.get('lean_cueq_chunks', '-')}"
            if stmts == 0 or (nmax and nmax < gate):
                out["k2b_flash_triattention"] = (None, "no tri-attention statement inside the K2B gate ran: " + detail)
            else:
                out["k2b_flash_triattention"] = (k2b > 0, detail)
    except Exception as e:  # noqa: BLE001 — a census that cannot be read is said, not hidden
        out["k2b_flash_triattention"] = (None, f"census unreadable: {e!r}"[:160])
    for unit in (_apb, _sl):                              # the per-call gates (dtype_gate): a lever whose every call stepped aside BY NAME (stock `--dtype fp32|fp16`, an atom_attn_exact
        try:                                              # Refused) is working as designed — (True, …) —; calls at the site with none served and some unaccounted = (False, …): a fallback
            out.update(unit.gate_census())
        except Exception as e:  # noqa: BLE001
            for name in GATED_SERVED:
                out.setdefault(name, (None, f"census unreadable: {e!r}"[:160]))
    return out


def kernel_census(rep: dict) -> dict:
    """``rep["kernels"][<name>]["imported_from"]`` = the file ``import <name>`` resolved to in THIS process at the time of the call (None =
    not imported yet / never imported by the row). The activation fills it once at activation end; a kernel the levers import lazily
    (fpf_trimul_v4 at the first served TriMul call) is re-read by :func:`reconcile` (end of run, the exit tally) and at manifest write."""
    for name, k in (rep.get("kernels") or {}).items():
        k["imported_from"] = getattr(sys.modules.get(name), "__file__", None)
    return rep


def reconcile(rep: dict, records: Dict[str, dict]) -> dict:
    """The activation report brought up to date with the kit's end-of-run records: a lever of LATE_RECORDS whose record says
    ``installed`` False leaves ``levers_applied`` for ``levers_fallback`` with the kit's ``why``; one whose record says installed True
    leaves ``levers_fallback`` for ``levers_applied``. ``partial`` follows the fallbacks. The report's ``reconciled`` field names the
    records seen and every move (total accounting)."""
    r = dict(rep)
    kernel_census(r)                                                   # the lazily imported kernels, as of the end of the run
    if "guard_lift" in (r.get("levers_applied") or []):                        # the package's own lever: its record (lift, patched, items, failures) rides the report
        r["runner_hooks"] = _runner_hooks.state()
        if not r["runner_hooks"]["patched"]:
            r.setdefault("levers_fallback", []).append("guard_lift"); r["levers_applied"] = [x for x in r["levers_applied"] if x != "guard_lift"]
            r.setdefault("fallback_reasons", {})["guard_lift"] = "runner.inference was never imported in this process: the hook was not applied"
    r["templates"] = _templates.state()                                        # the template census guard's record (chains, hits, real slots, drops, refusals) rides the report
    if "sampler_fuse" in (r.get("levers_applied") or []):                      # the diffusion transformer's fused kernels: their record (patched, models, sites, calls) rides the report
        r["sampler_fuse"] = _sampler_fuse.state()
        if not r["sampler_fuse"]["patched"]:
            r.setdefault("levers_fallback", []).append("sampler_fuse"); r["levers_applied"] = [x for x in r["levers_applied"] if x != "sampler_fuse"]
            r.setdefault("fallback_reasons", {})["sampler_fuse"] = f"{_sampler_fuse.TARGET} was never imported in this process: the kernels were not installed"
    if any(l in (r.get("levers_applied") or []) for l in _apb.LEVERS):        # the sampler attention levers: their record (seam patched, modules, calls, cell, named) rides the report
        r["apb_levers"] = _apb.state()
        for lever in _apb.KERNEL_LEVERS:                                          # the kernel levers; the precision lever follows its kernel lever below
            if lever in r["levers_applied"] and not r["apb_levers"]["patched"]:
                r.setdefault("levers_fallback", []).append(lever); r["levers_applied"] = [x for x in r["levers_applied"] if x != lever]
                r.setdefault("fallback_reasons", {})[lever] = "runner.inference was never imported in this process: the kernels were not installed"
            elif lever in r["levers_applied"] and lever in _apb.KERNEL_LEVERS and r["apb_levers"]["models"] and not r["apb_levers"][lever]["installed_on"]:
                r.setdefault("levers_fallback", []).append(lever); r["levers_applied"] = [x for x in r["levers_applied"] if x != lever]
                r.setdefault("fallback_reasons", {})[lever] = f"a runner was built and {lever} installed on no module" + (f": {r['apb_levers'][lever]['error']}" if r["apb_levers"][lever].get("error") else "")
        for prec, kernel in _apb.PRECISION.items():                            # a precision lever rides its kernel lever: off the applied list with it, by name
            if prec in r["levers_applied"] and kernel in (r.get("levers_fallback") or []):
                r["levers_fallback"].append(prec); r["levers_applied"] = [x for x in r["levers_applied"] if x != prec]
                r.setdefault("fallback_reasons", {})[prec] = f"rides {kernel}: {r['fallback_reasons'].get(kernel, kernel + ' fell back')}"
            elif prec in r["levers_applied"] and r["apb_levers"]["models"] and not r["apb_levers"][prec]["engaged"]:
                r.setdefault("levers_fallback", []).append(prec); r["levers_applied"] = [x for x in r["levers_applied"] if x != prec]
                r.setdefault("fallback_reasons", {})[prec] = f"a runner was built and {kernel}'s install did not engage fp16 operands"
    if any(l in (r.get("levers_applied") or []) for l in _sl.LEVERS):          # the sampler's fused-stack levers: seam patched, sites installed, census ride the report
        r["sampler_levers"] = _sl.state()
        for lever in _sl.KERNEL_LEVERS:
            if lever in r["levers_applied"] and not r["sampler_levers"]["patched"]:
                r.setdefault("levers_fallback", []).append(lever); r["levers_applied"] = [x for x in r["levers_applied"] if x != lever]
                r.setdefault("fallback_reasons", {})[lever] = "runner.inference was never imported in this process: the lever was not installed"
            elif lever in r["levers_applied"] and r["sampler_levers"]["models"] and not r["sampler_levers"][lever]["installed_on"]:
                r.setdefault("levers_fallback", []).append(lever); r["levers_applied"] = [x for x in r["levers_applied"] if x != lever]
                r.setdefault("fallback_reasons", {})[lever] = f"a runner was built and {lever} installed on no site" + (f": {r['sampler_levers'][lever]['error']}" if r["sampler_levers"][lever].get("error") else "")
        for prec, kernel in _sl.PRECISION.items():
            if prec in r["levers_applied"] and kernel in (r.get("levers_fallback") or []):
                r["levers_fallback"].append(prec); r["levers_applied"] = [x for x in r["levers_applied"] if x != prec]
                r.setdefault("fallback_reasons", {})[prec] = f"rides {kernel}: {r['fallback_reasons'].get(kernel, kernel + ' fell back')}"
            elif prec in r["levers_applied"] and r["sampler_levers"]["models"] and not r["sampler_levers"][prec]["engaged"]:
                r.setdefault("levers_fallback", []).append(prec); r["levers_applied"] = [x for x in r["levers_applied"] if x != prec]
                r.setdefault("fallback_reasons", {})[prec] = f"a runner was built and {kernel}'s install did not engage the {r['sampler_levers'][prec]['word']} word"
    if "pred_release" in (r.get("levers_applied") or []):                      # the package's prediction-release lever: its record (patch state, items released, GiB) rides the report
        from . import pred_release as _pred_release
        r["pred_release"] = _pred_release.state()
    on, fb = list(r.get("levers_applied") or []), list(r.get("levers_fallback") or [])
    why = dict(r.get("fallback_reasons") or {})
    moves: Dict[str, str] = {}
    served = served_census()                                                    # big path: a lever that is `on` but SERVED no call where calls were expected is a fallback, by name
    for name, (ok, detail) in served.items():
        if name in on and ok is False:
            on.remove(name); fb.append(name); why[name] = f"on but served 0 calls: {detail}"; moves[name] = "applied -> fallback (census)"
    asides = {name: dict(t["aside"]) for unit_state in (r.get("apb_levers") or {}, r.get("sampler_levers") or {})      # the per-call asides BY NAME of the gated levers (dtype_gate): declared,
              for name, t in ((n, (v or {}).get("gate")) for n, v in unit_state.items() if isinstance(v, dict))         # counted, by design — recorded, never a fallback
              if isinstance(t, dict) and t.get("aside")}
    sg = (records.get("clisampler") or {}).get("sampler")                        # the graphed sampler's poison self-test step-asides BY NAME (infopt_graphs graphed.py: a signature whose probe is
    sg = sg.get("aside") if isinstance(sg, dict) else None                       # non-finite — `--dtype fp16` on an input whose trunk overflowed — or whose graph failed the test runs the eager step body):
    if isinstance(sg, dict) and any(sg.values()):                                # `nonfinite_probe` is by design (recorded, never a fallback); `poison_mismatch` also sets prep_poison=failed, judged below
        asides["sampler_graph"] = {str(k): int(v) for k, v in sg.items() if v}
    if asides:
        r["asides"] = asides
    for name, (key, _) in LATE_RECORDS.items():
        rec = records.get(key)
        if not rec or "installed" not in rec:
            continue
        reason = f"{key}: installed={rec.get('installed')}" + (f" — {rec['why']}" if rec.get("why") else "")
        if name in on and rec["installed"] is False:
            on.remove(name); fb.append(name); why[name] = reason; moves[name] = "applied -> fallback"
        elif name in fb and rec["installed"] is True:
            fb.remove(name); why.pop(name, None); on.append(name); moves[name] = "fallback -> applied"
        if name == "sampler_admit" and name in on and (not isinstance(rec.get("policy"), dict) or rec.get("policy_err")):   # the policy did not install with the hook (policy:FAILED(...) on the SAMPLER line)
            on.remove(name); fb.append(name); moves[name] = "applied -> fallback"; why[name] = f"{key}: policy={rec.get('policy_err') or 'absent'}"
        if name == "sampler_reach" and name in on and not str(rec.get("reach") or "off").startswith("on("):           # reach=on(floor=1536; routes by fpf_clisampler.policy) or a fallback by name
            on.remove(name); fb.append(name); moves[name] = "applied -> fallback"; why[name] = f"{key}: reach={rec.get('reach') or 'absent'}"
        if name == "sampler_prep" and name in on:                                # all six parts or a fallback by name: the record's prep= word must be the full PARTS string
            word = str(rec.get("prep") or "off").split(";")[0].strip()
            if word != SAMPLER_PREP_PARTS or str(rec.get("prep_poison") or "").split(";")[0].strip() == "failed":     # prep_poison=ok|not-run|failed[;subsumed:..][;squash:..][;unlisted:..]
                on.remove(name); fb.append(name); moves[name] = "applied -> fallback"
                why[name] = f"{key}: prep={word} (expected {SAMPLER_PREP_PARTS})" + (f" prep_poison={rec.get('prep_poison')}" if rec.get("prep_poison") else "")
    gates = {key: {f: rec.get(f) for f in fields if f in rec} for key, fields in GATE_FIELDS.items()
             for rec in [records.get(key)] if rec and rec.get("installed") is True}
    r.update(levers_applied=on, levers_fallback=fb, levers_unavailable=list(fb), partial=bool(fb),
             fallback_reasons=why, gates=gates,
             reconciled={"records": sorted(records), "moves": moves, "record_errors": {k: v["report_error"] for k, v in records.items() if "report_error" in v}})
    if r.get("mode") == "big":
        from . import big as _big
        r = _big.reconcile(r)                                                # the census per unit: a partial memory lever joins levers_fallback by name; the block rides as `big`
    return r


SAMPLER_PREP_PARTS = "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec"   # infopt_graphs.protenix.sampler_prep PARTS joined by '+': the only accepted prep= word

GATE_FIELDS: Dict[str, Tuple[str, ...]] = {          # a documented gate is not a fallback: the kit's own counters of an INSTALLED lever, recorded under `gates`
    "stackgraph": ("captures", "replays", "eager", "eager_ineligible", "refused", "oom_skips", "evicted", "max_tok", "max_sig"),   # the memory guard / size gate
    "clisampler": ("max_tokens", "graph", "hoist", "sampler", "prep", "prep_poison", "prep_stats", "reach", "policy"),                               # PTX_SAMPLER_GRAPH_MAXTOK: larger predictions run the stock sampler; the sampler_prep words + counters
}


def _kit_sitecustomize_present() -> Optional[str]:
    sc = sys.modules.get("sitecustomize")
    f = getattr(sc, "__file__", None) if sc is not None else None
    if f and os.path.realpath(f) == os.path.realpath(os.path.join(kit_home(), SITECUSTOMIZE_RELPATH)):
        return f
    return None


class _Tee(io.TextIOBase):
    """Pass stderr through and keep a copy (to read the kit's own activation lines)."""
    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.buf.write(s)
        return self.real.write(s)

    def flush(self):
        self.real.flush()


def _run_kit_sitecustomize(fpf_home: str) -> str:
    """Execute the kit's src/sitecustomize.py as the kit would at interpreter start, with PTX_LAZY_INIT masked so its lazy-init block
    cannot import the model family before its import hook exists. Returns the stderr the kit printed."""
    path = os.path.join(fpf_home, SITECUSTOMIZE_RELPATH)
    masked = os.environ.pop("PTX_LAZY_INIT", None)
    tee = _Tee(sys.stderr)
    real = sys.stderr
    sys.stderr = tee
    try:
        runpy.run_path(path, run_name="protenix_opt_kit_sitecustomize")
        importlib.import_module("protenix.model.modules.pairformer")        # the kit's sitecustomize meta-path hook fires on this import
    finally:
        sys.stderr = real
        if masked is not None:
            os.environ["PTX_LAZY_INIT"] = masked
    return tee.buf.getvalue()


def _applied_markers() -> List[str]:
    out: List[str] = []
    lev = sys.modules.get("ptx_trunk2_levers")
    st = getattr(lev, "_STATS", {}) if lev is not None else {}
    out += [str(a) for a in st.get("applied", [])]
    v02 = sys.modules.get("ptx_fpf_v02")
    for a in getattr(v02, "STATS", {}).get("applied", []) if v02 is not None else []:       # ptx_fpf_v02.STATS["applied"] ("GRAPH:...")
        a = str(a)
        if ("v02:" + a) not in out and a not in out:
            out.append("v02:" + a)
    fpf = st.get("fpf")
    if isinstance(fpf, dict):
        out.append("FPF:enabled(" + ",".join(str(x) for x in fpf.get("enabled", [])) + ")")
    return out


def _own_levers(mode: str, environ=None) -> bool:
    """The package's own hook switch as read for `mode`: the guard lift. A bad value raises by name; the switch set under a mode whose
    row does not list the lever is refused by name — never applied silently."""
    lift = _runner_hooks.guard_lift_from_env(environ)
    if lift and "guard_lift" not in _row_levers(mode, environ):
        raise ActivationError(f"{','.join(LEVERS['guard_lift'].env_keys)} set under mode {mode}: the guard_lift lever is not in this mode's row")
    return lift


def _sampler_fuse_on(mode: str, environ=None) -> bool:
    """Whether PTX_SAMPLER_FUSE switches the sampler_fuse lever on for `mode` (sampler_fuse.from_env: a bad value raises by name). In the
    Set under a mode whose row does not list the lever: refused by name — never applied silently."""
    env = os.environ if environ is None else environ
    if not _sampler_fuse.from_env(env):
        return False
    if "sampler_fuse" in _row_levers(mode, env):
        return True
    if "sampler_fuse" in _row_levers_full(mode, env):
        return False
    raise ActivationError(f"{_sampler_fuse.ENV} set under mode {mode}: the sampler_fuse lever is not in this mode's row")


def _apb_levers_on(mode: str, environ=None) -> List[str]:
    """The sampler attention levers (apb_levers: dit_attn, atom_attn) whose switch turns them on for `mode` (a bad value raises by
    name); a switch set under a mode whose row does not list the lever is refused by name; an ablated one is left off."""
    env = os.environ if environ is None else environ
    on: List[str] = []
    for lever in _apb.LEVERS:
        if not _apb.from_env(lever, env):
            continue
        if lever in _row_levers(mode, env):
            on.append(lever)
        elif lever not in _row_levers_full(mode, env):
            raise ActivationError(f"{_apb.ENVS[lever]} set under mode {mode}: the {lever} lever is not in this mode's row")
    try:
        _apb.check_requires(on)                                        # a precision lever without its kernel lever: refused by name, never a silent no-op
    except RuntimeError as e:
        raise ActivationError(str(e)) from e
    return on


def _sampler_levers_on(mode: str, environ=None) -> List[str]:
    """The sampler's fused-stack levers (sampler_levers: cond_dedupe, dit_fused, dit_lowp, atom_fused, atom_attn_exact) whose switch turns them
    on for `mode` (a bad value raises by name); a switch set under a mode whose row does not list the lever is refused by name; an ablated one
    is left off; a precision lever without its kernel lever is refused by name."""
    env = os.environ if environ is None else environ
    on: List[str] = []
    for lever in _sl.LEVERS:
        try:
            want = _sl.from_env(lever, env)
        except ValueError as e:
            raise ActivationError(str(e)) from e
        if not want:
            continue
        if lever in _row_levers(mode, env):
            on.append(lever)
        elif lever not in _row_levers_full(mode, env):
            raise ActivationError(f"{_sl.ENVS[lever]} set under mode {mode}: the {lever} lever is not in this mode's row")
    try:
        _sl.check_requires(on)
    except RuntimeError as e:
        raise ActivationError(str(e)) from e
    return on


def _apply(fpf_home: str, mode: str, base: Optional[str] = None) -> Tuple[List[str], dict]:
    """The kit's activation (sitecustomize, hook route), then lazy init, the package's own runner hooks (the levers of `mode`'s row
    only: their switch set under another mode is refused by name) and, for big, the memory levers of big.py on `base` (the arm the
    mode composed on). Returns (applied markers, details)."""
    detail: dict = {}
    lev = sys.modules.get("ptx_trunk2_levers")
    if lev is not None and getattr(lev, "_STATS", {}).get("applied"):
        raise ActivationError("ptx_trunk2_levers.apply_from_env() already ran in this process: " + str(lev._STATS["applied"]))
    if "protenix.model.modules.pairformer" in sys.modules:
        raise ActivationError("protenix.model.modules.pairformer was imported before activation: the kit's sitecustomize would take its "
                              "direct-apply branch (no FPF_OPS provider); enable the mode before importing the protenix model family")
    detail["warm_imports"] = warm_imports()                            # the stack's heavy libraries imported early and shallow (env.sh's exports are in place), before the sitecustomize, the levers and the model import them from depths of their own
    err = _run_kit_sitecustomize(fpf_home)
    detail["sitecustomize_stderr"] = err
    if DIRECT_APPLY_SIGNATURE in err:
        raise ActivationError(f"the kit's sitecustomize applied the levers on its direct branch ('{DIRECT_APPLY_SIGNATURE}'): "
                              "fpf.enable_from_env() never ran")
    applied = _applied_markers()
    if not applied:
        raise ActivationError("the kit's sitecustomize applied nothing (no lever markers; stderr: " + err[-600:].strip() + ")")
    if HOOK_APPLY_SIGNATURE not in err:
        raise ActivationError("the kit's hook-route line '" + HOOK_APPLY_SIGNATURE + "' is missing from the sitecustomize output")
    detail["route_preload"] = _route_preload.preload(os.environ)      # the small-N gate package's TriMul callees (FPF_SMALLN_TRIMUL_EXACT_FN / _FAST_FN =
                                                                       # src/ptx_trimul_routes, named by env.sh) imported now, from the outside, before the activation proof (MODULE_PROOF)
                                                                       # reads sys.modules; the sealed package itself resolves them at its first call (its published bytes, unchanged)
    if os.environ.get("PTX_LAZY_INIT", "0") == "1":                                 # after the levers (the unit's own order: lazy init wraps the constructed levers)
        with contextlib.redirect_stdout(sys.stderr):
            try:
                import ptx_lazy_init
                applied.append(f"LAZY_INIT:{ptx_lazy_init.install()}")
            except Exception as e:
                applied.append(f"LAZY_INIT:unavailable({e!r})")
    lift = _own_levers(mode)
    applied.append(_templates.install())                            # every kit mode: the template census guard (dropped hits named; an all-dummy slot set counted)
    if lift:
        applied.extend(_runner_hooks.install(lift))
    if os.environ.get("PTX_PRED_RELEASE", "0") == "1":                                 # lever pred_release (every mode; modes.PACKAGE_POST): InferenceRunner.predict wrapped (the core's patch_attr_at_import), marker PRED_RELEASE:
        from . import pred_release as _pred_release
        applied.append(_pred_release.install())
    if _sampler_fuse_on(mode):                                       # every kit mode: the diffusion transformer's fused elementwise kernels (a switch set under a mode whose row lacks the lever is refused by name)
        applied.append(_sampler_fuse.install())
    apb = _apb_levers_on(mode)                                        # the T* rows of cc 9.0: the sampler attention levers, on the runner seam after sampler_fuse
    if apb:
        applied.extend(_apb.install(apb))
    sl = _sampler_levers_on(mode)                                     # the cc-9.0 rows: the sampler's fused-stack levers (+ the exact rows' atom_attn_exact), on the seam after the attention levers
    if sl:
        applied.extend(_sl.install(sl))
    if mode == "big":                                            # the memory levers on the shared core's registry (opt_core.mem), after the kit's own
        from . import big as _big
        applied.extend(_big.activate(base or BIG_BASE, os.environ))
    if os.environ.get("PTX_TEMPL_DEDUPE", "0") == "1":                                # env.sh L7 exports 1 on every arm
        applied.append("TEMPL_DEDUPE:" + os.environ["PTX_TEMPL_DEDUPE"] if "ptx_addon_levers" in sys.modules
                       else "TEMPL_DEDUPE:unavailable(ptx_addon_levers not imported by sitecustomize)")
    if os.environ.get("PTX_SAMPLER_GRAPH", "0") not in ("", "0") or os.environ.get("PTX_SAMPLER_HOIST", "0") not in ("", "0"):
        if "fpf_clisampler" in sys.modules:
            applied.append("SAMPLER:hook")
            detail["sampler_poison_aside"] = _poison_aside.install()     # the loop class the unit's install() constructs at init_model = the kit's step-aside subclass (sampler_poison_aside:
        else:                                                             # the poison self-test never raises to the run; no marker of its own — the loop names an aside when one happens)
            applied.append("SAMPLER:unavailable(fpf_clisampler not imported by sitecustomize)")
    lev = sys.modules.get("ptx_trunk2_levers")
    if lev is not None:
        detail["trunk2"] = {k: v for k, v in getattr(lev, "_STATS", {}).items() if k not in ("applied", "fpf_calls")}
    return applied, detail


WARM_MIN_CORE = "0.5.66.0"                                          # the first shared core that carries opt_core.warm_imports()
WARM_LINE = "WARM imports:"                                        # the activation route's one line for it (stderr, before the levers' own lines)
WARM_LIBRARIES = ("torch", "cuequivariance_ops_torch", "cuequivariance_torch")   # torch FIRST, by name: cuequivariance_ops_torch's libcue_ops.so links libnvrtc, which
                                                                    # CUDA-13 wheels resolve only once torch has loaded its own copy — imported ahead of torch it fails to load and
                                                                    # the library records its ops unavailable for the process; then the core's two (its default list, restated)


def warm_imports(stream=None) -> dict:
    """The stack's heavy model libraries (``WARM_LIBRARIES``: torch first, then the shared core's two, cuequivariance_ops_torch and cuequivariance_torch) imported ONCE, early and
    through the core's large-frame trampoline (``opt_core.warm_imports()``), before the levers and the model import them lazily from call
    depths of their own: an import-time table one of them builds in a module body costs about a second from most depths and about a minute
    from an unlucky one (CPython's stack-chunk boundary; the first item's pair stack or confidence head paid it). Exact-class: an import moved
    earlier — no kernel, no switch, no bytes change; a dictionary lookup when the libraries are imported already, the word ``absent`` for one
    that is not installed (the lever that needs it refuses by name at its own call, as before). One line on ``stream`` (stderr):
    ``[protenix-opt] WARM imports: <library>=<seconds>s|present|absent …`` (torch is named first: the activation route reaches this point before
    anything imported torch, and the cuEquivariance ops must never load ahead of it); the words are also the activation report's ``warm_imports``.
    With a shared core that has no ``warm_imports`` (or one without the ``libraries=`` word) the line says so BY NAME and the run
    continues — the first call that needs a library imports it then, as it always did."""
    stream = stream or sys.stderr
    import opt_core
    fn = getattr(opt_core, "warm_imports", None)
    if fn is None:
        have = getattr(opt_core, "__version__", "?")
        rep = {"unavailable": f"opt_core {have} < {WARM_MIN_CORE}"}
        stream.write(f"{LOG} {WARM_LINE} unavailable(opt_core {have} < {WARM_MIN_CORE}: no warm_imports) — the first call that needs the libraries imports them\n")
        stream.flush()
        return rep
    try:
        rep = dict(fn(libraries=WARM_LIBRARIES, origin="protenix_opt"))
    except TypeError:                                                # a warm_imports without the libraries= / origin= words: named, never called with an order it cannot take
        have = getattr(opt_core, "__version__", "?")
        rep = {"unavailable": f"opt_core {have}: warm_imports without libraries="}
        stream.write(f"{LOG} {WARM_LINE} unavailable(opt_core {have}: warm_imports without libraries=) — the first call that needs the libraries imports them\n")
        stream.flush()
        return rep
    words = " ".join(f"{k}={v:.3f}s" if isinstance(v, float) else f"{k}={v}" for k, v in rep.items())
    stream.write(f"{LOG} {WARM_LINE} {words or 'nothing to import'}\n"); stream.flush()
    return rep


def status() -> dict:
    if _REPORT is None:
        return {"active": False, "reason": "protenix_opt.enable() has not been called in this process"}
    return dict(_REPORT)


def activate(mode: str, strict: bool = False, trigger: Optional[str] = None, dry_run: bool = False, det: Optional[bool] = None) -> dict:
    """Apply a mode once per process (idempotent). Returns the activation report; `strict` raises ActivationError when not active;
    `dry_run` resolves, gates and reports (report["dry_run"] = True) without exporting or applying anything. The GPU never gates
    activation here: the kit's own per-key behaviour stands (env.sh probes, each lever's own refusals) and the report names what was
    left off (`partial`, `levers_unavailable`). `det` (None: PTX_DET=1 in the environment) is recorded as the report's `det`."""
    global _REPORT
    mode = (mode or "").strip().lower()
    if os.environ.get(tp.ENV_NGPU) not in (None, ""):                   # a GPU count in the environment: 1 = this process's single-GPU line;
        try:                                                              # > 1 selects the multi-GPU line, whose ranks activate the base mode —
            n = tp.n_gpu_from_env()                                       # in this process it is refused by name, never a 1-GPU run under its name
        except tp.TpError as e:
            return _refuse_line(mode, strict, reason=str(e))
        why = tp.refusal(mode, n)
        if why:
            return _refuse_line(mode, strict, reason=why)
        if tp.line_selected(mode, n):
            return _refuse_line(mode, strict)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
    if dry_run:
        return _dry_run(mode, strict, det)
    if _REPORT is not None and (_REPORT.get("active") or _REPORT.get("apply_failed")):   # levers are applied once per process
        if _REPORT.get("mode") == mode:
            return dict(_REPORT)
        r = dict(_REPORT)
        r["reason"] = f"already active as mode={_REPORT.get('mode')}; levers are applied once per process (restart to change mode)"
        if strict:
            raise ActivationError(r["reason"])
        return r
    if _ACTIVATING:                                                    # the lever imports themselves import `runner`/`protenix.model`
        return {"active": False, "mode": mode, "reason": "activation in progress (re-entrant call ignored)"}
    _disarm_autoload()
    return _activate_locked(mode, strict, trigger, det)


def _ablation_refusal(mode: str, names: List[str]) -> str:
    """The NOT ACTIVE reason for a MODEL_OPT_LEVERS_OFF request the kit refuses by name (ablation.validate's wording)."""
    try:
        _ablation.validate(mode, names)
    except _ablation.AblationError as e:
        return str(e)
    return f"{_ablation.ENV}={','.join(names)} refused (mode={mode})"


def _refuse_line(mode: str, strict: bool, reason: Optional[str] = None) -> dict:
    """The NOT ACTIVE report for the multi-GPU line selected in-process (PROTENIX_OPT=big with PROTENIX_OPT_N_GPU=P, P > 1): the line's
    ranks come from ``run.sh pred --mode big --n_gpu P``; an in-process activation would run one GPU under the line's name."""
    global _REPORT
    reason = reason or (f"{tp.ENV_NGPU}={os.environ.get(tp.ENV_NGPU)} selects the multi-GPU line ({tp.LINE}) of mode {mode}: it runs through "
                        f"`run.sh pred --mode {mode} {tp.FLAG_NGPU} P` (P > 1; base mode {tp.BASE_MODE} on every rank), never in this process")
    _REPORT = {"mode": mode, "active": False, "refused": True, "reason": reason, "levers_applied": [], "levers_fallback": [],
               "protenix_version": protenix_version(), "package_version": __version__}
    _report.log_activation(_REPORT); _REPORT["logged"] = True
    if strict:
        raise ActivationError(reason)
    return dict(_REPORT)


def _disarm_autoload() -> None:
    """An explicit enable() owns the process: an armed autoload finder must not fire a second activation from the lever imports."""
    from . import _autoload
    _autoload.disarm()


def _activate_locked(mode: str, strict: bool, trigger: Optional[str], det: Optional[bool] = None) -> dict:
    global _ACTIVATING
    _ACTIVATING = True
    try:
        return _activate_body(mode, strict, trigger, det)
    finally:
        _ACTIVATING = False


def _base(mode: str, trigger: Optional[str] = None) -> dict:
    from . import kits
    try:
        lab = kits.labels()
    except FileNotFoundError:                                          # no kit beside the package and no PROTENIX_OPT_HOME/MODEL_OPT: _gates names it
        lab = {"flashpairformer": None}
    return {"active": False, "mode": mode, "levers_applied": [], "levers_fallback": [], "gpu": {"name": None, "sm": None},
            "protenix_version": protenix_version(), "package_version": __version__, "kit_version": lab["flashpairformer"],
            "trigger": trigger, "target_gpu": os.environ.get("MODEL_OPT_TARGET_GPU") or None,
            "gate_overrides": {ENV_FORCE: os.environ[ENV_FORCE]} if os.environ.get(ENV_FORCE) == "1" else {}}   # the gate override (version / no CUDA)


# ------------------------------------------------------------------------------------------------------ model instances ----
def register_instance_counter() -> dict:
    """Count stock's ``Protenix`` model instances from now on by the constructor wrap (opt_core.instances: on the class now when its module
    is imported — the instances that already exist are found once by a gc scan — otherwise at the module's import through a meta-path
    finder; the package imports nothing). Idempotent. Returns :func:`instance_check`."""
    return _instances.register_instance_counter(MODEL_MODULE, MODEL_CLASS)


def instance_check() -> dict:
    """How many ``Protenix`` model objects exist in this process and how that is known: ``{"n", "method": "counted" | "gc" | "none",
    "built"}`` — ``counted`` = the constructor wrap (deterministic), ``gc`` = a scan of the garbage collector's objects (the class
    could not be wrapped, or its instances take no weak references), ``none`` = the model module is not imported (0 instances)."""
    return _instances.instance_check(MODEL_MODULE, MODEL_CLASS)


def late_activation_gate() -> Optional[str]:
    """The late-activation rule of the explicit API. enable(mode) is allowed after `import protenix.model` (the package is empty in
    stock 2.0.0) and REFUSED, with the reason named, once (1) the kit reports a lever already applied in this process, (2) a Protenix
    model instance exists (the levers patch the classes an instance was built from; :func:`instance_check`), or (3)
    `protenix.model.modules.pairformer` is already imported — the kit's meta-path hook fires on that import, so afterwards its
    sitecustomize could only take the direct branch (no fpf.enable_from_env): a kit fact, not a package choice. Returns the reason,
    or None when activation may proceed."""
    lev = sys.modules.get("ptx_trunk2_levers")
    applied = getattr(lev, "_STATS", {}).get("applied") if lev is not None else None
    if applied:
        return f"the kit reports levers already applied in this process ({', '.join(map(str, applied))[:200]}): enable() must run before the kit's own activation"
    chk = instance_check()
    if chk["n"]:
        return (f"a Protenix model instance already exists in this process ({chk['n']} live, {chk['method']}): the levers patch the classes it "
                "was built from; enable the mode before constructing the model")
    if PAIRFORMER_MODULE in sys.modules:
        return (f"{PAIRFORMER_MODULE} was imported before activation: the kit's hook route cannot fire afterwards (its sitecustomize "
                "would take the direct branch, fpf.enable_from_env never runs); enable the mode after `import protenix.model` at the latest, "
                f"before {MODEL_MODULE} / runner.inference")
    return None


def _route_kernels(fpf_home: str) -> dict:
    """Route every KERNEL_ROUTES name to the core's copy (opt_core.kernels.route: a by-name meta-path finder), export what the kernel reads
    at import (kernels.exports with KERNEL_EXPORTS: fpf_trimul_v4 reads the kit's cells table, fpf_triatt_k2b its core-copy K2B_CELLS.json
    via PF_TRIATTN_TABLE unless the caller pre-set one, CALLER_EXPORTS) and gate the resolution BEFORE its first
    import (kernels.route_check: the name resolves to the core copy, its bytes equal the core's sums, the exports are present). A name a
    routed kernel imports at run time (the core's META `runtime_imports`) is NOT routed: it resolves to the kit's
    own copy on the kit's sys.path entries, recorded in runtime_imports. {name: {ok, resolved, core_copy, routed, already_imported, exports,
    runtime_imports, reason}} — the activation report and the manifest carry it; a refused route is a named NOT ACTIVE reason."""
    out = {}
    for name in KERNEL_ROUTES:
        _kernels.route(name)
        exp = _kernels.exports(name, **(KERNEL_EXPORTS[name](fpf_home) if name in KERNEL_EXPORTS else {}))
        exp = {k: (os.environ[k] if (k in CALLER_EXPORTS and os.environ.get(k)) else v) for k, v in exp.items()}   # a caller's pre-set table wins (recorded as the export in force)
        os.environ.update(exp)
        g = _kernels.route_check(name)
        d = g.details
        out[name] = {"ok": g.ok, "resolved": d.get("resolved"), "core_copy": d.get("core_copy"), "routed": d.get("routed"), "already_imported": d.get("already_imported"),
                     "exports": exp, "runtime_imports": d.get("runtime_imports"), "reason": g.reason}
    return out


def _gates(base: dict) -> Optional[str]:
    """Common gates (kits present and on disk matching their MANIFEST.json's carry rule, protenix pin, GPU visible). Returns a
    refusal reason or None; fills base["gpu"] and base["kits"]."""
    try:
        fpf_home = kit_home()
    except FileNotFoundError as e:
        return str(e)
    if not os.path.isfile(os.path.join(fpf_home, "env.sh")):
        return f"kit directory not found at {fpf_home} (install the package editable from protenix_v2/opt)"
    if not os.path.isfile(pins_path()):
        return f"stock pins not found at {pins_path()}"
    force = os.environ.get(ENV_FORCE, "") == "1"
    ok, ver, why = version_gate(force)
    base["protenix_version"] = ver
    if not ok:
        return why
    facts = _core.gate()                                                              # the kit's ONE pin gate (the entry ran it; idempotent): the facts for the report
    base["core"] = {"ok": True, "pinned": facts["pinned"], "imported": {k: facts["installed"].get(k) for k in ("version", "root")}}
    g = gpu_probe_smi()
    base["gpu"] = {"name": g["name"], "sm": g["sm"], "cc": g["cc"], "probe": g.get("probe")}
    base["_cc"] = g["cc"]
    if g["sm"] is None and not force:
        return f"no CUDA device visible ({g.get('probe')}; the levers need an NVIDIA GPU); {ENV_FORCE}=1 applies them anyway"
    if g["sm"] is None:
        sys.stderr.write(f"{LOG} WARNING: no CUDA device visible; {ENV_FORCE}=1 applies the levers anyway\n")
    base["gpu_supported"] = g["cc"] in SUPPORTED_CC                    # data for the report/manifest only: the kit's own per-key gates decide what runs
    if base["target_gpu"] and g["name"] and base["target_gpu"].lower() not in g["name"].lower():
        base.setdefault("notes", []).append(f"MODEL_OPT_TARGET_GPU={base['target_gpu']} but the GPU is {g['name']}: not the GPU this configuration targets")
    return None


def _dry_run(mode: str, strict: bool, det: Optional[bool] = None) -> dict:
    from . import det as _det
    base = _base(mode)
    base["dry_run"] = True
    base["det"] = _det.is_det() if det is None else bool(det)
    abl = _ablation.requested(os.environ)
    if mode == "off":
        try:
            sc = _kit_sitecustomize_present()
        except FileNotFoundError:
            sc = None
        if abl:                                                        # the stock route applies no lever: an ablation request under it is refused by name
            rep = dict(base, refused=True, reason=_ablation_refusal(mode, abl))
            _report.log_activation(rep); rep["logged"] = True
            if strict:
                raise ActivationError(rep["reason"])
            return rep
        if sc:
            rep = dict(base, activated_by="sitecustomize", refused=True,
                       reason=f"mode off refused: the kit's sitecustomize is active in this process ({sc}, env.sh activation route); stock cannot run here")
        else:
            rep = dict(base, reason="mode off: stock protenix (no environment set, no lever applied)")
        _report.log_activation(rep); rep["logged"] = True
        if sc and strict:
            raise ActivationError(rep["reason"])
        return rep
    why = _gates(base)
    cc = base.pop("_cc", None)
    if why:
        rep = dict(base, reason=why)
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(why)
        return rep
    try:
        res = resolve(mode, os.environ, kit_home(), compute_cap=cc)
    except ValueError as e:
        rep = dict(base, reason=str(e))
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(str(e))
        return rep
    except Exception as e:
        rep = dict(base, reason=f"env.sh could not be sourced: {e!r}")
        _report.log_activation(rep); rep["logged"] = True
        if strict:
            raise ActivationError(rep["reason"])
        return rep
    if abl:                                                            # MODEL_OPT_LEVERS_OFF: validated by name (mode membership, then this card's row), then the levers' switches leave the resolution
        try:
            _ablation.validate(mode, abl, row=res.row, row_key=res.row_key)
        except _ablation.AblationError as e:
            rep = dict(base, refused=True, reason=str(e))
            _report.log_activation(rep); rep["logged"] = True
            if strict:
                raise ActivationError(str(e))
            return rep
        pre_off = _ablation.pre_switches(abl, res.row); restore = _ablation.restored_words(abl, res.row)
        if pre_off:                                                    # a switch env.sh READS (PTX_T_ATT): env.sh sourced again without the ablated word — the displaced lever's own word in its place when it has one, else env.sh's default (= the displaced lever: k2b_flash_triattention)
            res = resolve(mode, os.environ, kit_home(), compute_cap=cc, skip_pre=pre_off, pre_override=restore)
        base["levers_ablated"] = list(abl); base["ablated_switches"] = _ablation.apply(res, abl, resourced=pre_off, restored=restore); base["ablation_restored"] = dict(restore)
    final = res.final(os.environ)
    # switch-level accounting before anything is applied: a planned lever whose switch env.sh (or the package's kernel-key selection)
    # left unset on this box is unavailable here; marker-probed levers are only provable at activation and stay planned.
    _on, _fb, skipped, why = _classify(mode, [], final, notes=res.notes, row=res.row, row_key=res.row_key, sm=base["gpu"].get("sm"))
    unavailable = [n for n in _fb if LEVERS[n].probe == "env" or why.get(n, "").startswith(NO_BLK2_CELLS)]   # switch-level + card-level (no BLK2 cells on this arch) accounting
    planned = [n for n in _row_levers(mode, final) if n not in unavailable and n not in skipped]
    entries = kit_sys_path(res.pythonpath, mode=mode)
    rep = dict(base, levers_applied=planned, levers_fallback=unavailable, levers_not_in_arm=skipped,
               partial=bool(unavailable), levers_unavailable=unavailable, fallback_reasons={n: why[n] for n in unavailable + skipped},
               row_key=res.row_key, base=res.base, pre_exports=res.pre_exports, env=res.exports, unset=res.unsets,
               extras=res.extras, prebuilt=final.get("INFOPT_FASTLN_PREBUILT"), kernel_key=res.kernel_key, kit_spec=res.kit_spec,
               notes=base.get("notes", []) + res.notes, sys_path=entries, graphed_py=graphed_py(entries), graphed_py_sha256=graphed_py_sha256(entries),
               cells=cells(base["gpu"]["sm"]),
               reason="dry run: resolved and gated, nothing applied", env_final_size=len(final))
    _report.log_activation(rep); rep["logged"] = True
    return rep


def _activate_body(mode: str, strict: bool, trigger: Optional[str], det: Optional[bool] = None) -> dict:
    from . import det as _det
    global _REPORT
    base = _base(mode, trigger)
    base["det"] = _det.is_det() if det is None else bool(det)
    try:
        sc = _kit_sitecustomize_present()
    except FileNotFoundError:
        sc = None
    abl = _ablation.requested(os.environ)
    if mode == "off":
        if sc:                                                         # the env.sh route applied the levers at interpreter start: not stock
            reason = f"mode off refused: the kit's sitecustomize is active in this process ({sc}, env.sh activation route); stock cannot run here"
            _REPORT = dict(base, activated_by="sitecustomize", refused=True, reason=reason)
            _report.log_activation(_REPORT); _REPORT["logged"] = True
            if strict:
                raise ActivationError(reason)
            return dict(_REPORT)
        if abl:                                                        # the stock route applies no lever: an ablation request under it is refused by name
            reason = _ablation_refusal(mode, abl)
            _REPORT = dict(base, refused=True, reason=reason)
            _report.log_activation(_REPORT); _REPORT["logged"] = True
            if strict:
                raise ActivationError(reason)
            return dict(_REPORT)
        _REPORT = dict(base, reason="mode off: stock protenix (no environment set, no lever applied)")
        return dict(_REPORT)

    def refuse(reason: str) -> dict:
        global _REPORT
        base.pop("_cc", None)
        _REPORT = dict(base, reason=reason)
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        if strict:
            raise ActivationError(reason)
        return dict(_REPORT)

    if sc:                                                             # env.sh activation route owns this process; never double-apply
        if abl:                                                        # the levers were applied at interpreter start: an ablation cannot take effect here
            return refuse(f"{_ablation.ENV}={','.join(abl)} refused: the kit's sitecustomize applied the levers at interpreter start ({sc}, env.sh "
                          "activation route); an ablation takes effect only on the package's own routes (run.sh, the console entry point, the drop-in environment route)")
        _REPORT = dict(base, activated_by="sitecustomize",
                       reason=f"the kit's sitecustomize is active in this process ({sc}, env.sh activation route); protenix_opt applied nothing")
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        return dict(_REPORT)
    why = _gates(base)
    if why:
        return refuse(why)
    chk = register_instance_counter()                                  # the constructor wrap from here on
    late = late_activation_gate()
    if late:
        return refuse("late activation refused: " + late)
    cc = base.pop("_cc", None)
    fpf_home = kit_home()
    try:
        res = resolve(mode, os.environ, fpf_home, compute_cap=cc)
    except ValueError as e:                                            # a switch value the table refuses (PTX_LAZY_INIT)
        return refuse(str(e))
    except Exception as e:
        return refuse(f"env.sh could not be sourced: {e!r}")
    if abl:                                                            # MODEL_OPT_LEVERS_OFF: validated by name (mode membership, then this card's row), then the levers' switches leave the resolution (before any export)
        try:
            _ablation.validate(mode, abl, row=res.row, row_key=res.row_key)
        except _ablation.AblationError as e:
            return refuse(str(e))
        pre_off = _ablation.pre_switches(abl, res.row); restore = _ablation.restored_words(abl, res.row)
        if pre_off:                                                    # a switch env.sh READS: env.sh sourced again without it (its default word for the slot stands)
            res = resolve(mode, os.environ, fpf_home, compute_cap=cc, skip_pre=pre_off, pre_override=restore)
        base["levers_ablated"] = list(abl); base["ablated_switches"] = _ablation.apply(res, abl, resourced=pre_off, restored=restore); base["ablation_restored"] = dict(restore)
    notes = list(base.get("notes", [])) + res.notes
    if chk["method"] == "gc":
        notes.append(f"{MODEL_CLASS} instances are counted by a gc scan: the class could not be wrapped")
    for m in ("ptx_trunk2_levers", "ptx_fpf_v02", "fpf_smalln"):      # these read switches at import: a pre-activation import keeps the old values
        if m in sys.modules:
            notes.append(f"{m} was imported before activation; its import-time switches reflect the earlier environment")
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None and getattr(getattr(torch_mod, "cuda", None), "is_initialized", lambda: False)():
        want = res.exports.get(ALLOC_CONF); have = os.environ.get(ALLOC_CONF)
        if have and (want is None or have == want):                     # a launching process exported the policy before CUDA (tp.rank_env); env.sh keeps a
            notes.append(f"{ALLOC_CONF}={have} was set before CUDA initialised (exported by the launching process; env.sh keeps a caller's value): "
                         "the allocator policy is in effect")             # caller's value ("unless already set"), so it is not in res.exports
        else:
            notes.append(f"CUDA was initialised before activation: {ALLOC_CONF} cannot take effect in this process")
    for k in res.unsets:                                               # export BEFORE any CUDA initialisation (allocator policy)
        os.environ.pop(k, None)
    os.environ.update(res.exports)
    private_report_dir(os.environ)                                     # the lever report's directory: made 0700 when absent, refused by name when not the user's own
    os.environ[ENV_MODE] = mode                                        # child processes (stock's MSA/featurizer subprocesses) autoload the same mode
    os.environ["FPF_HOME"] = fpf_home
    entries = kit_sys_path(res.pythonpath, fpf_home, mode=mode)
    if os.environ.get("CUEQ_TRITON_TUNING"):
        notes.append("CUEQ_TRITON_TUNING is set: the L1 tuned-tile cache must run as a pure lookup; unset it")
    _install_sys_path(entries)
    kern = _route_kernels(fpf_home)                                    # before the first import of a routed kernel (the levers import it while they apply)
    base["kernels"] = kern
    if [n for n, k in kern.items() if not k["ok"]]:
        return refuse("; ".join(k["reason"] for k in kern.values() if not k["ok"]))
    _report.register_exit_tally()                                      # before the kit registers its lever-report hook (atexit is LIFO)
    try:
        applied, detail = _apply(fpf_home, mode, res.base)
    except Exception as e:
        _REPORT = dict(base, apply_failed=True, reason=f"lever application failed: {e!r}", env=res.exports, notes=notes)
        _report.log_activation(_REPORT); _REPORT["logged"] = True
        raise
    try:
        g = gpu_info()                                                 # torch's view of the device (CUDA init is fine now)
        cur = base.get("gpu") or {}                                    # the pre-activation probe (nvidia-smi, or injected): never overwritten
        base["gpu"] = {"name": cur.get("name") or g["name"], "sm": cur.get("sm") or g["sm"], "cc": cur.get("cc") or g["cc"],
                       "probe": cur.get("probe") if cur.get("name") else g["probe"], "torch": {"name": g["name"], "sm": g["sm"], "cc": g["cc"]}}
        cc = cc or g["cc"]
        if cur.get("cc") and g["cc"] and cur["cc"] != g["cc"]:
            notes.append(f"GPU probes disagree: {cur.get('probe')} says {cur.get('name')} (cc {cur['cc']}), torch says {g['name']} (cc {g['cc']})")
    except Exception as e:
        notes.append(f"torch GPU probe failed: {e!r}")
    kernel_census(base)                                                # what the levers imported so far (a lazily imported kernel is re-read at reconcile / manifest-write)
    if abl:                                                            # the kit's own applied record read back: an ablated lever that engaged anyway refuses the run by name
        leaked = _ablation.leaks(abl, applied, os.environ)
        if leaked:
            return refuse(f"{_ablation.ENV}={','.join(abl)}: the ablation did not take effect — "
                          + "; ".join(f"{n}: {ev}" for n, ev in leaked.items()) + " (the kit's own applied record says the lever engaged)")
    on, fb, skipped, why = _classify(mode, applied, os.environ, notes, row=res.row, row_key=res.row_key, sm=base["gpu"].get("sm"))
    _REPORT = dict(base, active=True, levers_applied=on, levers_fallback=fb, levers_not_in_arm=skipped, fallback_reasons=why,
                   graphed_py=graphed_py(entries), graphed_py_sha256=graphed_py_sha256(entries),   # the infopt_graphs copy this process imports (env.sh L6 order)
                   partial=bool(fb), levers_unavailable=list(fb),              # what the kit's own applied/refused records left off, relative to the README row
                   row_key=res.row_key, base=res.base, pre_exports=res.pre_exports,
                   applied_markers=applied, env=res.exports, unset=res.unsets, extras=res.extras, prebuilt=os.environ.get("INFOPT_FASTLN_PREBUILT"),
                   kernel_key=res.kernel_key, kit_spec=res.kit_spec, notes=notes, sys_path=entries, cells=cells(base["gpu"]["sm"], fpf_home),
                   detail=detail)
    _report.log_activation(_REPORT); _REPORT["logged"] = True
    return dict(_REPORT)
