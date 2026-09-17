"""The step observer of both arms — one vocabulary of phases, one row per unit, no numerics touched.

`install()` wraps, once per process, the ColabDesign methods BindCraft's design step drives (colabdesign/af/design.py `_af_design`):
the stage calls `design_logits` / `design_soft` / `design_hard` / `design_pssm_semigreedy` (+ `design_semigreedy` inside the last), the
graded `step`, the forward-only `predict`, and `_save_results` (a greedy round's end). Each wrapper calls the original with the caller's
own arguments and records; nothing is altered, so the stock arm and the kit arm carry the identical observer. PHASES (the words
the run record and the tests share): `soft` = a design_logits call, `temp` = design_soft, `hard` = design_hard,
`greedy` = design_pssm_semigreedy / design_semigreedy (forward-only sampling rounds). Rows, in call order:

    {"i", "k", "kind": "grad", "phase", "call": "<method>:<ordinal>", "wall_s", "first_call", "loss", <the step's log terms>, "soft", "temp",
     "hard", "dropout", "recycles", "models"}                      one graded design step (soft / temp / hard)
    {"i", "k", "kind": "forward", "phase": "greedy", ..., "loss"}   one predict() of a greedy mutant (or the round-0 baseline)
    {"i", "k", "kind": "round", "phase": "greedy", "tries", ...}    one semigreedy iteration closed (_save_results)

`end_row(...)` is built by the driver: {"kind": "end", "phase": "complete" | "terminated:<reason>", "terminate", "n_steps", "phases_run"}.
The rows live in memory (the driver's timing summary reads them); nothing here writes a file. One optional callback for in-process
callers, `set_step_callback(fn)`: `fn(step, state)` is called once per row as it is recorded — `step` = the row's index `i`, `state` = a
copy of the row — and is None by default (no call).
"""
from __future__ import annotations

import functools
import time
from typing import Callable, Optional

PHASE_OF = {"design_logits": "soft", "design_soft": "temp", "design_hard": "hard", "design_pssm_semigreedy": "greedy", "design_semigreedy": "greedy"}
PHASES = ("soft", "temp", "hard", "greedy")
STAGE_METHODS = ("design_logits", "design_soft", "design_hard", "design_pssm_semigreedy", "design_semigreedy")
MARKER = "_colabdesign_opt_units"
LOG_SCALARS = ("loss", "plddt", "pae", "i_pae", "con", "i_con", "ptm", "i_ptm", "rg", "helix", "termini", "exp_res", "dgram_cce", "fape", "rmsd", "seq_ent", "hard", "soft", "temp")

_S = {"rows": [], "stack": [], "ordinal": {}, "seen_calls": set(), "forwards_in_round": 0, "callback": None}


def reset() -> None:
    """Empty the rows and the call bookkeeping (the callback stays as set)."""
    _S.update({"rows": [], "stack": [], "ordinal": {}, "seen_calls": set(), "forwards_in_round": 0})


def set_step_callback(fn: Optional[Callable[[int, dict], None]]) -> None:
    """`fn(step, state)` once per recorded row (step = the row's index, state = a copy of the row); None = no call."""
    _S["callback"] = fn


def step_callback() -> Optional[Callable[[int, dict], None]]:
    return _S["callback"]


def rows() -> list:
    return list(_S["rows"])


def current() -> Optional[dict]:
    """{"phase", "call"} of the innermost stage call on the stack, or None outside any."""
    return dict(_S["stack"][-1]) if _S["stack"] else None


def _f(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v


def _models(x):
    x = x.tolist() if hasattr(x, "tolist") else x
    if x is None:
        return None
    return [int(v) for v in (x if isinstance(x, (list, tuple)) else [x])]


def _log_fields(log: Optional[dict]) -> dict:
    out = {}
    if not isinstance(log, dict):
        return out
    for k, v in log.items():
        if k == "models":
            out["models"] = _models(v)
        elif k in ("recycles", "num_recycles"):
            out["recycles"] = None if v is None else int(v)
        else:
            fv = _f(v)
            if fv is not None:
                out[k] = fv
    return out


def _opt_fields(model) -> dict:
    opt = getattr(model, "opt", None) or {}
    return {"soft": _f(opt.get("soft")), "temp": _f(opt.get("temp")), "hard": _f(opt.get("hard")), "dropout": opt.get("dropout") if isinstance(opt.get("dropout"), bool) else opt.get("dropout"),
            "num_recycles": opt.get("num_recycles"), "num_models": opt.get("num_models"), "sample_models": opt.get("sample_models")}


def _emit(row: dict) -> None:
    row["i"] = len(_S["rows"])
    _S["rows"].append(row)
    cb = _S["callback"]
    if cb is not None:
        cb(row["i"], dict(row))


def _wrap_stage(name, fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        inner = name == "design_semigreedy" and _S["stack"] and _S["stack"][-1]["call"].startswith("design_pssm_semigreedy:")
        if inner:                                                     # design_semigreedy inside design_pssm_semigreedy: the same greedy call, not a new one
            return fn(self, *args, **kwargs)
        n = _S["ordinal"].get(name, 0) + 1; _S["ordinal"][name] = n
        _S["stack"].append({"phase": PHASE_OF[name], "call": f"{name}:{n}"})
        _S["forwards_in_round"] = 0
        try:
            return fn(self, *args, **kwargs)
        finally:
            _S["stack"].pop()
    setattr(wrapper, MARKER, True)
    return wrapper


def _wrap_step(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        cur = current()
        t0 = time.perf_counter()
        out = fn(self, *args, **kwargs)
        wall = time.perf_counter() - t0
        if cur is not None:
            first = cur["call"] not in _S["seen_calls"]; _S["seen_calls"].add(cur["call"])
            aux = getattr(self, "aux", None) or {}
            _emit({"k": int(getattr(self, "_k", 0)) - 1, "kind": "grad", "phase": cur["phase"], "call": cur["call"], "wall_s": wall, "first_call": first,
                   **_log_fields(aux.get("log")), **{k: v for k, v in _opt_fields(self).items() if k in ("soft", "temp", "hard", "dropout")},
                   "recycles": aux.get("num_recycles") if aux.get("num_recycles") is not None else _log_fields(aux.get("log")).get("recycles")})
        return out
    setattr(wrapper, MARKER, True)
    return wrapper


def _wrap_predict(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        cur = current()
        t0 = time.perf_counter()
        out = fn(self, *args, **kwargs)
        wall = time.perf_counter() - t0
        if cur is not None and cur["phase"] == "greedy":
            first = cur["call"] not in _S["seen_calls"]; _S["seen_calls"].add(cur["call"])
            log = out.get("log") if isinstance(out, dict) else None
            _S["forwards_in_round"] += 1
            _emit({"k": int(getattr(self, "_k", 0)), "kind": "forward", "phase": "greedy", "call": cur["call"], "wall_s": wall, "first_call": first,
                   **_log_fields(log), "dropout": kwargs.get("dropout"), "recycles": (out or {}).get("num_recycles") if isinstance(out, dict) else None})
        return out
    setattr(wrapper, MARKER, True)
    return wrapper


def _wrap_save_results(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        out = fn(self, *args, **kwargs)
        cur = current()
        if cur is not None and cur["phase"] == "greedy":
            aux = getattr(self, "aux", None) or {}
            _emit({"k": int(getattr(self, "_k", 0)), "kind": "round", "phase": "greedy", "call": cur["call"], "tries": _S["forwards_in_round"], **_log_fields(aux.get("log"))})
            _S["forwards_in_round"] = 0
        return out
    setattr(wrapper, MARKER, True)
    return wrapper


def install() -> bool:
    """Wrap the ColabDesign design class once (idempotent); returns True when this call installed the wrappers."""
    from colabdesign.af import design as D
    cls = D._af_design
    if getattr(getattr(cls, "step", None), MARKER, False):
        return False
    for name in STAGE_METHODS:
        if hasattr(cls, name):
            setattr(cls, name, _wrap_stage(name, getattr(cls, name)))
    cls.step = _wrap_step(cls.step)
    cls.predict = _wrap_predict(cls.predict)
    cls._save_results = _wrap_save_results(cls._save_results)
    return True


def installed() -> bool:
    import sys
    D = sys.modules.get("colabdesign.af.design")
    return bool(D and getattr(getattr(D._af_design, "step", None), MARKER, False))


def n_steps(rs: Optional[list] = None) -> dict:
    rs = _S["rows"] if rs is None else rs
    return {"soft": sum(1 for r in rs if r.get("kind") == "grad" and r.get("phase") == "soft"),
            "temp": sum(1 for r in rs if r.get("kind") == "grad" and r.get("phase") == "temp"),
            "hard": sum(1 for r in rs if r.get("kind") == "grad" and r.get("phase") == "hard"),
            "greedy_rounds": sum(1 for r in rs if r.get("kind") == "round"), "greedy_forward": sum(1 for r in rs if r.get("kind") == "forward")}


def phases_run(rs: Optional[list] = None) -> list:
    rs = _S["rows"] if rs is None else rs
    out = []
    for r in rs:
        c = r.get("call")
        if c and (not out or out[-1] != c):
            out.append(c)
    return out


def end_row(terminate: dict, rs: Optional[list] = None) -> dict:
    """The closing row: phase `complete` when BindCraft's verdict is empty and no gate fired, else `terminated:<reason>`."""
    reason = terminate.get("gate") or terminate.get("verdict") or ""
    return {"i": len(_S["rows"] if rs is None else rs), "kind": "end", "phase": f"terminated:{reason}" if reason else "complete", "terminate": dict(terminate),
            "n_steps": n_steps(rs), "phases_run": phases_run(rs)}
