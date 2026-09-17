"""The shared-core producers this package imports, checked BEFORE anything of the package resolves (stdlib only). Every entry route
(`run.sh <verb>` → `python -m protenix_v1_opt` / the `protenix-v1-opt` console script → __main__.main; the `.pth` hook → _autoload) calls
`refuse_if_missing` first: with the shared core absent the process prints
`[protenix-v1-opt] NOT ACTIVE: reason=core_missing:opt_core — this package imports opt_core >= MIN_CORE (...); exit 3`, with an OLDER core
that lacks a producer `... reason=producer_missing:<module,...> — ...; exit 3`, and ends with exit 3 — never a traceback, never a run that
resolves a mode without its producers."""
import importlib
import importlib.util
import sys

from ._core_gate import PIN_TABLE, find_pyproject, read_table      # stdlib-only, like this module: the core gate's own pyproject reader (one reader of the pin)

PREFIX = "[protenix-v1-opt]"      # == report.PREFIX (test_producers: the documented pair; this module imports nothing of the package but the stdlib-only _core_gate)
EXIT_NOT_ACTIVE = 3               # == report.EXIT_NOT_ACTIVE
def pinned_core_line(anchor=__file__):
    """The opt_core line this package is pinned to: `[tool.opt_core] version` of the kit's own opt/pyproject.toml, read from the tree with the
    core gate's reader (facts from the tree — a re-pin moves it; nothing here restates it). None when no pyproject.toml with the table is at
    or above the package (the core gate refuses that case by name: core_pin_unreadable)."""
    pp = find_pyproject(anchor)
    version = (read_table(pp, "tool." + PIN_TABLE) or {}).get("version") if pp else None
    return str(version) if version else None


MIN_CORE = pinned_core_line()     # the opt_core line whose producers are listed below == opt/pyproject.toml [tool.opt_core] version (read, never typed)
PIN_UNREADABLE = "(pin unreadable: no opt/pyproject.toml [tool.opt_core] at or above the package)"   # the refusal's word for MIN_CORE None; the core gate refuses that tree first (core_pin_unreadable)
CORE = "opt_core"
REQUIRED_PRODUCERS = (            # every opt_core module a module of this package imports (test_producers locks the list against the sources)
    "opt_core", "opt_core.attn", "opt_core.attn.pair_fused", "opt_core.cli", "opt_core.gates", "opt_core.home", "opt_core.kernels", "opt_core.oom", "opt_core.report", "opt_core.trimul",
    "opt_core.stock_proof", "opt_core.mem", "opt_core.mem.ngpu",
    "opt_core.mem.rowpair", "opt_core.mem.rowpair.bcast", "opt_core.mem.rowpair.census", "opt_core.mem.rowpair.confidence",
    "opt_core.mem.rowpair.diffusion", "opt_core.mem.rowpair.dist", "opt_core.mem.rowpair.evidence", "opt_core.mem.rowpair.heads",
    "opt_core.mem.rowpair.launch", "opt_core.mem.rowpair.msa", "opt_core.mem.rowpair.msa_host", "opt_core.mem.rowpair.pairstack", "opt_core.mem.rowpair.rankdata",
    "opt_core.mem.rowpair.shard",
    "opt_core.mem.rowpair.template", "opt_core.mem.rowpair.transition", "opt_core.mem.rowpair.triatt", "opt_core.mem.rowpair.trimul", "opt_core.mem.rowpair.trimul_fused",
    "opt_core.mem.rowpair.trunk",
    "opt_core.modes", "opt_core.mem.registry", "opt_core.mem.record", "opt_core.mem.compose", "opt_core.mem.chunk", "opt_core.mem.ckpt",
    "opt_core.mem.allocator",       # the big line (big.py): the memory-lever library
    "opt_core.jit_cache",           # _stackkey: the persistent JIT caches' stack key (configs/<card>.env under MODEL_OPT_JIT_ROOT)
)


def _importable(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        return False


def _core_present() -> bool:
    """The core package itself imports (its __init__ is import-light by contract); a spec that exists but cannot execute counts as absent."""
    try:
        importlib.import_module(CORE)
        return True
    except Exception:
        return False


def missing_producers(names=REQUIRED_PRODUCERS) -> list:
    """The names among `names` this interpreter cannot resolve (an absent core lists as the one entry 'opt_core')."""
    if not _importable(CORE) or not _core_present():
        return [CORE]
    return [n for n in names if n != CORE and not _importable(n)]


def refusal_line(missing) -> str:
    kind = "core_missing" if list(missing) == [CORE] else "producer_missing"
    return (f"{PREFIX} NOT ACTIVE: reason={kind}:{','.join(missing)} — this package imports {CORE} >= {MIN_CORE or PIN_UNREADABLE} "
            f"(opt/pyproject.toml [tool.{CORE}]; install the release tree's common/{CORE}: pip install -e common/{CORE} -e opt); exit {EXIT_NOT_ACTIVE}")


def refuse_if_missing(stream=None, exit=sys.exit):
    """Print the refusal line and end the process with EXIT_NOT_ACTIVE when a producer is missing; return None otherwise."""
    missing = missing_producers()
    if missing:
        print(refusal_line(missing), file=stream or sys.stderr, flush=True)
        exit(EXIT_NOT_ACTIVE)
    return None


if __name__ == "__main__":                                        # `python -m protenix_v1_opt._producers`: the two gates alone (configs/h100.env, run.sh) — exit 3 with the line, else 0
    from ._core_gate import gate
    gate(__file__, tag=PREFIX.strip("[]"))
    refuse_if_missing()
    sys.exit(0)
