"""protenix_fpf_ditfast._plumbing — shared bits: refusal type, stderr prefix, DiffusionModule lookup, per-process EXIT census."""
import os, sys, json, atexit

PREFIX = "[protenix_fpf_ditfast]"


class LeverRefused(RuntimeError):
    """Raised BY NAME when a lever cannot engage on this model / card / environment. The kit hook turns it into exit 3; nothing falls back."""


def say(*a):
    print(PREFIX, *a, file=sys.stderr, flush=True)


def diffusion_module(model):
    m = getattr(model, "diffusion_module", None)
    if m is None and hasattr(model, "module"):
        m = getattr(model.module, "diffusion_module", None)
    if m is None:
        raise LeverRefused("protenix_fpf_ditfast: model has no diffusion_module")
    return m


_EXITS = {}


def register_exit(name, fn):
    """fn() -> dict printed as `[protenix_fpf_ditfast] EXIT <name> {...}` at interpreter exit (and appended to $PTX_LEVER_REPORT as one JSON line).
    Counts are Python-level calls: inside the sampler graph a stack forward is entered at the eager record step and at capture; replays do not
    re-enter Python (the graphed sampler's own stats count those)."""
    if not _EXITS:
        atexit.register(_print_exits)
    _EXITS[name] = fn


def exits() -> dict:
    return {n: f() for n, f in _EXITS.items()}


def _print_exits():
    for name, fn in _EXITS.items():
        try:
            say(f"EXIT {name} " + json.dumps(fn(), default=str))
        except Exception as e:  # pragma: no cover
            say(f"EXIT {name} census failed: {e!r}")
    p = os.environ.get("PTX_LEVER_REPORT", "")
    if p:
        try:
            with open(p, "a") as fh:
                fh.write(json.dumps({"protenix_fpf_ditfast": exits(), "pid": os.getpid()}, default=str) + "\n")
        except Exception:
            pass
