"""opendde_fpf_ditfast._plumbing — shared bits: refusal types, stderr prefix, DiffusionModule lookup, per-process EXIT census, the hoist accessor."""
import os, sys, json, atexit

PREFIX = "[opendde_fpf_ditfast]"                                   # odde: this copy's stderr prefix


class LeverRefused(RuntimeError):
    """Raised BY NAME when a lever cannot engage on this model / card / environment. The kit hook turns it into exit 3; nothing falls back."""


class StepAside(Exception):
    """Raised inside a fused forward for a CALL FORM the stack does not serve (found past the entry checks); the forward catches it,
    counts the named aside and serves that call with the module's stock forward. Never leaves the unit."""
    def __init__(self, word):
        super().__init__(word); self.word = word


def count_aside(counts: dict, word: str):
    """Flat per-word aside census on a unit's COUNTS (asides, aside_<word>) — the kit's LEVER line prints them."""
    counts["asides"] = counts.get("asides", 0) + 1
    counts["aside_" + word] = counts.get("aside_" + word, 0) + 1


def say(*a):
    print(PREFIX, *a, file=sys.stderr, flush=True)


def diffusion_module(model):
    m = getattr(model, "diffusion_module", None)
    if m is None and hasattr(model, "module"):
        m = getattr(model.module, "diffusion_module", None)
    if m is None:
        raise LeverRefused("opendde_fpf_ditfast: model has no diffusion_module")
    return m


_EXITS = {}


def register_exit(name, fn):
    """fn() -> dict printed as `[opendde_fpf_ditfast] EXIT <name> {...}` at interpreter exit (and appended to $ODDE_SERVED_LEVERS_REPORT as one JSON line).
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
    p = os.environ.get("ODDE_SERVED_LEVERS_REPORT", "")             # odde: the kit's report file (registry KNOBS)
    if p:
        try:
            with open(p, "a") as fh:
                fh.write(json.dumps({"opendde_fpf_ditfast": exits(), "pid": os.getpid()}, default=str) + "\n")
        except Exception:
            pass


def odde_hoist():                                                                    # odde: OpenDDE's DiT hoist (levers/DITFAST/tools/odde_addon.py, lever dit_hoist)
    """The installed OpenDDE DiT hoist object (levers/DITFAST odde_addon.DitHoist: the `cached(name, producer)` slot protocol) or None."""
    try:
        import odde_addon
        h = odde_addon._ACTIVE.get("dit_hoist")
        return h if (h is not None and getattr(h, "installed", False)) else None
    except Exception:
        return None
