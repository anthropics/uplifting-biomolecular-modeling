"""The shared core this package imports (``opt_core`` >= ``MIN_CORE``): every entry route (``python -m opendde_opt``, the console scripts,
``run.sh`` verbs, the ``.pth`` hook under a kit selection) asks :func:`refusal` FIRST and exits 3 with its sentence when a producer module is
absent — a wholly absent core reads ``core_missing:opt_core``, an older core ``producer_missing:<modules>``. Stdlib only (this module must
import when the core does not)."""
from __future__ import annotations

import importlib.util
import os

MIN_CORE = "0.4.3"
REQUIRED_PRODUCERS = (                      # every opt_core module this package imports (its own import statements; tests/test_core_gate.py derives the set from the sources)
    "opt_core", "opt_core.autoload", "opt_core.det", "opt_core.gates", "opt_core.home", "opt_core.instances",
    "opt_core.jit_cache", "opt_core.kernels", "opt_core.manifest", "opt_core.mem", "opt_core.mem.allocator", "opt_core.mem.budget", "opt_core.mem.torch_alloc",
    "opt_core.precision", "opt_core.precision.policy", "opt_core.precision.recipe", "opt_core.report", "opt_core.stock_proof", "opt_core.mem.ngpu",
    "opt_core.mem.rowpair", "opt_core.mem.rowpair.census", "opt_core.mem.rowpair.confidence", "opt_core.mem.rowpair.diffusion",
    "opt_core.mem.rowpair.dist", "opt_core.mem.rowpair.evidence", "opt_core.mem.rowpair.launch", "opt_core.mem.rowpair.msa",
    "opt_core.mem.rowpair.pairstack", "opt_core.mem.rowpair.rankdata", "opt_core.mem.rowpair.shard", "opt_core.mem.rowpair.transition", "opt_core.mem.rowpair.triatt",
    "opt_core.mem.rowpair.trimul", "opt_core.mem.rowpair.trunk", "opt_core.mem.rowpair.template", "opt_core.mem.registry", "opt_core.oom",
    "opt_core.capture", "opt_core.capture.graphs",
    "opt_core.kernels.ln",                  # ln_core: the LayerNorm provider face (select / layer_norm by tier word)
    "opt_core.kernels.apb_attn",            # the row-sharded roll-out's DiT local-row attention kernel (tp_diffusion, --n_gpu P>1 only)
    "opt_core.mem.rowpair.trimul_fused",    # the row-sharded stacks' fused row-block TriMul provider (tp_kernels.trimul_provider, --n_gpu P>1 only)
    "opt_core.host.outputs",                 # writer_overlap: the core's background writer + one-pass host copy (F6.output_overlap)
)
PREFIX = "[opendde-opt]"
EXIT_NOT_ACTIVE = 3


def _core_dir() -> str | None:
    """Where the top-level ``opt_core`` package resolves (no import: find_spec of a top-level name imports nothing), or None."""
    try:
        spec = importlib.util.find_spec("opt_core")
    except (ImportError, ValueError):       # a blocked / broken finder
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    return list(spec.submodule_search_locations)[0]


def _present(core_dir: str, name: str) -> bool:
    """``opt_core.a.b`` is the file ``a/b.py`` or the package ``a/b/__init__.py`` under the core's directory (checked on disk, so the
    ``.pth`` route imports nothing of the core to ask)."""
    rel = name.split(".")[1:]
    if not rel:
        return True
    base = os.path.join(core_dir, *rel)
    return os.path.isfile(base + ".py") or os.path.isfile(os.path.join(base, "__init__.py"))


def missing_producers() -> list[str]:
    """The REQUIRED_PRODUCERS absent here, in declaration order (``["opt_core"]`` when the core itself is absent)."""
    d = _core_dir()
    if d is None:
        return ["opt_core"]
    return [m for m in REQUIRED_PRODUCERS if not _present(d, m)]


def refusal() -> str | None:
    """The NOT ACTIVE sentence of an absent / older core, or None when every producer is importable."""
    miss = missing_producers()
    if not miss:
        return None
    if miss == ["opt_core"]:
        return f"{PREFIX} NOT ACTIVE reason=core_missing:opt_core (the shared core is not importable; pip install -e common/opt_core -e opendde/opt)"
    return (f"{PREFIX} NOT ACTIVE reason=producer_missing:{','.join(miss)} — this package imports opt_core >= {MIN_CORE} "
            f"(the installed core lacks these modules; pip install -e common/opt_core -e opendde/opt at this tree)")
