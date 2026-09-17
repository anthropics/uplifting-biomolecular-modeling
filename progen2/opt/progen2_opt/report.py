"""Observability for progen2_opt: every line the package prints on stderr, prefixed ``[progen2-opt]``. The f-string on each ``*_FMT =`` line
below is the package's line contract: a reader derives its patterns from these templates, so a line changes HERE and in every reader of it
together (never silently).

* activation — the ``stack`` line (torch / transformers / tokenizers / gpu / card / stack_key / applied / notes) once per exact process
  at activation, before any work: what the box is, and every uncertainty about it NAMED as a note (a stack off the pinned versions, a card
  outside the known classes, stock settings outside the tested defaults) — never a reason to disengage;
  ``ACTIVE mode=<m> variant=<v> route=<r> kit=<kit dir of the route> on=<a,b,...>[ notes=<note>; ...]`` once per exact process as soon as
  the route's composition has its evidence (sample: the levers the load installed in this process; score: the kit's apply record):
  ``on=`` the mode's levers — all of them, a mode is all of its levers — and ``notes=`` (only when non-empty) what the kit found untested
  at apply and engaged on (kernels JIT-compiled from PTX on a card without a prebuilt image, a library version or a replaced function's
  bytes off the tested ones);
  ``NOT ACTIVE: <reason> (mode=off variant=<v>)`` once per stock process; ``NOT ACTIVE: <reason> (mode=… variant=… route=…); exit 3 (…)``
  for a refusal BY NAME, nothing run — the stock files at the stock dir are not the pinned commit's bytes (that changes what stock
  means; every mode refuses it), the kit tree is not whole, no CUDA device or ``--device cpu`` (the levers run on CUDA), a lever that
  CANNOT run here (``<lever> cannot run: <reason>`` — the fused-kernel library missing, not loading or without an image for the card, a
  lever the load did not install, a dtype a lever does not take), or a scoring run whose kit did not compute what it names (partial
  activation: the rotary tables or the counted path contradict; the outputs stay); ``DRY-RUN ...`` from `check`;
* application — ``APPLIED model#<i> ...`` once per model instance the scoring kit's apply() ran on (the score route);
* load — ``load variant=<v> t=<s>s slots=<batch sizes|-> max_length=<L>`` once per loaded generation model (the sample route): the load's
  wall (weights read once, tokenizer, the decode levers installed), the batch sizes (`num_samples` values) whose static K/V slots are held for
  the work in front of the model (`-` = none yet: a batch size is allocated when a unit first names it), the length those slots are sized for;
* ready — ``ready variant=<v> t=<s>s`` when the route can run its first item (every route prints it: the sample route once its slots are held
  — t = the wall from the route's start; the score route after its load;
  the stock caller from the stock's own `loading parameters took` line);
* item — ``item <name> <wall>s`` after an item's block is written (both arms; ``... FAILED <reason>`` names an item that wrote none);
* peak — ``PEAK item=<name> alloc_gib=<f.2> reserved_gib=<f.2> pid=<pid>`` after a sample item's ``item`` line (exact): the CUDA caching
  allocator's high-water marks of this process (torch.cuda.max_memory_allocated / max_memory_reserved after the item's unit) — the memory of
  the process that holds the model, its weights and slots;
* score path ok — after the scoring kit's counted-path counters assert (the route's own proof, exact + score);
* exit — ``EXIT mode=<m> route=<r> items=<n> kit_modules=<csv|none>`` at interpreter exit: the kit modules imported in this process
  (`none` in a stock process — the proof no kit was on the path), never silent.
"""
from __future__ import annotations

import atexit
import os
import sys

PREFIX = "[progen2-opt]"
KIT_MODULES = ("engines.progen2.kits.v0_score_r3_1", "engines.progen2.kits.v0_ew", "progen2_decode", "sampler_exact", "oneread_loader", "kit_t1")


def _join(items) -> str:
    if not items:
        return "none"
    if isinstance(items, dict):
        return ",".join(f"{k}:{'+'.join(v) if v else 'none'}" for k, v in items.items())
    return ",".join(str(x) for x in items)


def gpu_label(gpu) -> str:
    if isinstance(gpu, dict):
        name, sm = gpu.get("name"), gpu.get("sm")
        if not name:
            return "none"
        return f"{name}({sm})" if sm else str(name)
    return str(gpu) if gpu else "none"


def route_kit(rep: dict, route: str | None) -> str:
    """The kit DIR of the route (tree-relative): `opt/serving/pipeline_v0_4` (sample) or the scoring kit dir (score); `none` for stock."""
    dirs = rep.get("kit_dirs") or {}
    home = rep.get("package_home") or ""
    key = {"sample": "serving", "score": "scoring"}.get(route or "", None)
    d = dirs.get(key) if key else None
    if not d:
        return "none"
    rel = os.path.relpath(d, os.path.dirname(home)) if home else d
    return rel.replace(os.sep, "/")


def route_on(rep: dict, route: str | None) -> list:
    """The names on for the route."""
    on = rep.get("optimizations") or {}
    if isinstance(on, dict):
        return list(on.get(route or "", []) or [])
    return list(on)


def cannot_run_words(cannot) -> str:
    """The reason of a mode's refusal over levers that CANNOT run here: `<lever> cannot run: <reason>; <lever> cannot run: <reason>` —
    every lever named with the kit's own words (a mode is all of its levers: it never runs under its name with a subset)."""
    return "; ".join(f"{lever} cannot run: {reason}" for lever, reason in (cannot or []))


def active_line(mode, variant, route, kit, on, notes=None) -> str:
    on = _join(on)
    ACTIVE_FMT = f"[progen2-opt] ACTIVE mode={mode} variant={variant} route={route} kit={kit} on={on}"
    return ACTIVE_FMT + (f" notes={'; '.join(str(n) for n in notes)}" if notes else "")


def not_active_line(reason, variant) -> str:
    NOT_ACTIVE_FMT = f"[progen2-opt] NOT ACTIVE: {reason} (mode=off variant={variant})"
    return NOT_ACTIVE_FMT


REFUSED_ESCAPE = "exit 3 (`--mode off` runs the stock route without the kit)"   # a refusal names its one-flag escape on the same line: a kit mode fails closed, running without the kit is the user's explicit choice


def refused_line(reason, mode=None, variant=None, route=None, escape: str = REFUSED_ESCAPE) -> str:
    """The NOT ACTIVE line of a refusal by name (the stock files off the pinned commit, the kit tree not whole, no CUDA device / `--device
    cpu`, a lever that cannot run here — cannot_run_words — or a kit's KitRefused escaping a verb): the reason, then the
    context this process has (mode / variant / route, whichever is known), then the exit code and the escape (REFUSED_ESCAPE: the one flag
    that runs without the kit; the stock-files refusal names its own — stack.STOCK_FILES_ESCAPE — since every mode refuses a stock off its
    pin; the cpu refusal stack.CPU_ESCAPE)."""
    REFUSED_FMT = f"[progen2-opt] NOT ACTIVE: {reason}"
    if not (mode is None and variant is None and route is None):
        REFUSED_FMT += f" (mode={mode} variant={variant}" + (f" route={route}" if route else "") + ")"
    return REFUSED_FMT + f"; {escape}"


def partial_detail(partial, what, mode, variant, route) -> str:
    """The engine's part of the partial-exit lines: the levers without evidence first (each with its reason), then the verb's own reason."""
    return f"{'; '.join(partial)}; {what} (mode={mode} variant={variant} route={route})"


def partial_line(partial, what, mode, variant, route) -> str:
    """The partial-exit line of `score` (after the items): the scoring kit's own evidence contradicts the composition it applied — its
    rotary tables differ from the stock function's, or its counters are off the counted path — so the kit did not compute what the mode
    names. Not an uncertainty about the box: terminal, exit 3, the outputs stay for inspection. The family grammar —
    `NOT ACTIVE: partial activation — <detail>; exit 3`."""
    detail = partial_detail(partial, what, mode, variant, route)
    PARTIAL_FMT = f"[progen2-opt] NOT ACTIVE: partial activation — {detail}; exit 3"
    return PARTIAL_FMT


def load_line(variant, t_load_s: float, slots, max_length: int) -> str:
    """The sample route's load line: the load wall, the batch sizes whose static K/V slots are held (`-` = none), the length they are sized for."""
    t = f"{float(t_load_s):.2f}"
    held = ",".join(str(int(b)) for b in slots) if slots else "-"
    LOAD_FMT = f"[progen2-opt] load variant={variant} t={t}s slots={held} max_length={int(max_length)}"
    return LOAD_FMT


def ready_line(variant, t_load_s: float) -> str:
    t = f"{float(t_load_s):.2f}"
    READY_FMT = f"[progen2-opt] ready variant={variant} t={t}s"
    return READY_FMT


def item_line(name: str, wall_s: float) -> str:
    wall = f"{float(wall_s):.3f}"
    ITEM_FMT = f"[progen2-opt] item {name} {wall}s"
    return ITEM_FMT


def peak_line(name: str, alloc_gib: float, reserved_gib: float, pid: int) -> str:
    """The sample item's allocator line: this process `pid`'s torch.cuda.max_memory_allocated / max_memory_reserved in GiB after the item."""
    alloc = f"{float(alloc_gib):.2f}"; reserved = f"{float(reserved_gib):.2f}"
    PEAK_FMT = f"[progen2-opt] PEAK item={name} alloc_gib={alloc} reserved_gib={reserved} pid={int(pid)}"
    return PEAK_FMT


def exit_line(mode, route, n, kit_modules) -> str:
    kit_modules = _join(kit_modules)
    EXIT_FMT = f"[progen2-opt] EXIT mode={mode} route={route} items={n} kit_modules={kit_modules}"
    return EXIT_FMT


def score_ok_line(n, forwards, counters) -> str:
    counters = counters if isinstance(counters, str) else ",".join(f"{k}={v}" for k, v in sorted((counters or {}).items()))
    SCORE_OK_FMT = f"[progen2-opt] score path ok units={n} forwards={forwards} counters={counters}"
    return SCORE_OK_FMT


def stack_line(rep: dict) -> str:
    st = rep.get("stack") or {}
    return (f"{PREFIX} stack torch={st.get('torch')} transformers={st.get('transformers')} tokenizers={st.get('tokenizers')} python={st.get('python')} "
            f"gpu={gpu_label(rep.get('gpu'))} card={rep.get('card_class')} stack_key={rep.get('stack_key')} applied={rep.get('applied')}"
            + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))


def activation_line(rep: dict | None) -> str:
    """The one line of an activation report: ACTIVE (exact: ``on`` / ``apply_notes`` as the verb recorded them — report.log_active — else
    the composition's names), NOT ACTIVE (off / refused by name), DRY-RUN (`check`)."""
    rep = rep or {}
    mode, variant, route = rep.get("mode"), rep.get("variant"), rep.get("route")
    if rep.get("active"):
        on = rep["on"] if rep.get("on") is not None else route_on(rep, route)
        return active_line(mode, variant, route, route_kit(rep, route), on, rep.get("apply_notes"))
    reason = rep.get("reason") or "no reason given by progen2_opt.enable()"
    if rep.get("dry_run") and rep.get("kit_line"):
        head = f"mode={mode} variant={variant}" + (f" route={route}" if route else "")
        return (f"{PREFIX} DRY-RUN {head} kit={rep.get('kit_line')} on={_join(rep.get('optimizations'))} gpu={gpu_label(rep.get('gpu'))} card={rep.get('card_class')} stack_key={rep.get('stack_key')}"
                + (f" would_refuse={rep['would_refuse']!r}" if rep.get("would_refuse") else "")
                + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))
    if not mode or (mode == "off" and not rep.get("refused")):
        return not_active_line(reason, variant)
    return refused_line(reason, mode, variant, route, escape=rep.get("escape") or REFUSED_ESCAPE)


def applied_line(rep: dict, rec: dict) -> str:
    return (f"{PREFIX} APPLIED model#{rec.get('model_index')} mode={rep.get('mode')} variant={rep.get('variant')} kit={rec.get('kit')} "
            f"ew={rec.get('ew')} resident_fp16={rec.get('resident_fp16')} on={_join(rec.get('optimizations'))}")


def log_activation(rep: dict | None, stream=None) -> str:
    """The activation-time line, once: the ``stack`` line when the levers engage (the ACTIVE line waits for the composition's evidence:
    log_active), else the report's one line (NOT ACTIVE / DRY-RUN)."""
    rep = rep if rep is not None else {}
    line = stack_line(rep) if rep.get("active") else activation_line(rep)
    if not rep.get("logged"):
        print(line, file=stream or sys.stderr, flush=True)
    return line


def log_active(rep: dict, on, notes=None, stream=None) -> str:
    """The ACTIVE line, once per process, as soon as the route's composition has its evidence: ``on`` = the levers that engaged (the mode's
    whole set), ``notes`` = what the kit found untested at apply and engaged on (recorded on the report: `on`, `apply_notes`)."""
    rep["on"], rep["apply_notes"] = list(on), [str(n) for n in (notes or [])]
    line = activation_line(rep)
    if not rep.get("active_logged"):
        print(line, file=stream or sys.stderr, flush=True)
        rep["active_logged"] = True
    return line


# ------------------------------------------------------------------------------------------------------------------ exit tally
def kit_modules_loaded() -> list:
    """The kit modules imported in THIS process (by name), for the EXIT line."""
    return [m for m in KIT_MODULES if m in sys.modules]


def _counters(mod: str):
    m = sys.modules.get(mod)
    if m is None:
        return None
    fn = getattr(m, "counters_snapshot", None)
    try:
        d = fn() if callable(fn) else getattr(m, "COUNTERS", None)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return dict(d) if isinstance(d, dict) else None


def memory_stats() -> dict | None:
    out = {}
    for mod in KIT_MODULES:
        d = _counters(mod)
        if d is not None:
            out[mod.rsplit(".", 1)[-1]] = d
    return out or None


_EXIT = {"registered": False, "mode": None, "route": None, "items": 0}


def note_items(n: int, mode=None, route=None) -> None:
    _EXIT["items"] += int(n)
    if mode:
        _EXIT["mode"] = mode
    if route:
        _EXIT["route"] = route


def exit_tally_line() -> str:
    line = exit_line(_EXIT["mode"], _EXIT["route"], _EXIT["items"], kit_modules_loaded())
    stats = memory_stats()
    if stats:
        fields = []
        for mod, d in stats.items():
            items = sorted((str(k), v) for k, v in d.items() if not isinstance(v, (dict, list)))
            fields.append(f"{mod}={{" + ",".join(f"{k}={v}" for k, v in items[:16]) + "}")
        line += f"\n{PREFIX} counters pid={os.getpid()} " + " ".join(fields)
    return line


def _print_exit_tally() -> None:
    try:
        line = exit_tally_line()
    except Exception as e:  # noqa: BLE001
        line = f"{PREFIX} EXIT tally failed: {e!r}"
    try:
        sys.stderr.write(line + "\n"); sys.stderr.flush()
    except Exception:
        pass


def register_exit_tally(mode=None, route=None) -> None:
    if mode:
        _EXIT["mode"] = mode
    if route:
        _EXIT["route"] = route
    if _EXIT["registered"]:
        return
    _EXIT["registered"] = True
    atexit.register(_print_exit_tally)
