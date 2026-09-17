"""The row-sharded route's row-chunking levers: one installer, called from ``rowpair.install_rank`` inside every rank AFTER
``rowpair.install`` (install re-owns the pair frames; a lever bound earlier would be shadowed).

``install(model, P, levers)`` binds the members of ``rowpair.FOR_ROUTE`` the resolved set names (``levers``), in that order, and
returns the record ``stack.settle_route`` judges: ``{"levers": {name: bool}, "why": {name: reason}, "knobs": {...}}``. A member that
cannot bind (an anchor of the statement it re-issues is missing, a dependency was left out of the set) raises
``opt_core.mem.rowpair.RowpairRefused`` naming it — the route's by-name refusal (NOT ACTIVE, exit 3); nothing here prints-and-continues.

Knobs (read ONCE here, never by the lever modules; STOCK.md "Variables"): ``EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS`` (default 1024: below it
every member keeps the whole-shard statement — a declaration on the fold line, not a refusal), ``EF2_ROWPAIR_INJECT_MB`` (512: the recycle
inject's row block), ``EF2_ROWPAIR_CONF_MB`` (512: the confidence statement's row block), ``EF2_ROWPAIR_ZBF16_MB`` (1024: the bf16 pair
copy's row band). A malformed value is a RowpairRefused naming the variable.
"""
from __future__ import annotations

import os
from typing import Dict, Sequence

MEMBERS = ("injrows", "confrows", "confbf16", "pdeskip", "confmem", "biasfree", "zbf16")   # == rowpair.FOR_ROUTE (install order)
RIDES_ON_CONFROWS = ("confbf16", "pdeskip", "confmem", "zbf16")                            # == ablation.DEPENDS: flags of the row-blocked confidence statement
RIDES_ON = {"confbf16": ("confrows", "confmem"), "pdeskip": ("confrows",), "confmem": ("confrows",), "zbf16": ("confrows",)}   # == ablation.DEPENDS for these
#   members: confbf16 needs confmem too — its bf16 pair activations reach the row-attention pooling, and only confmem's per-block pooling statement
#   upcasts what it reads (the whole-shard one would hand a bf16 operand to an fp32 Linear)
KNOBS = {"min_tokens": ("EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS", 1024, int), "inject_mb": ("EF2_ROWPAIR_INJECT_MB", 512, float),
         "conf_mb": ("EF2_ROWPAIR_CONF_MB", 512, float), "zbf16_mb": ("EF2_ROWPAIR_ZBF16_MB", 1024, float)}
STATE: Dict[str, object] = {"installed": False, "levers": {}, "why": {}, "knobs": {}, "P": 1}


def _refused(msg: str):
    from opt_core.mem.rowpair import RowpairRefused
    return RowpairRefused(msg)


def read_knobs(env=None) -> Dict[str, object]:
    """The four knobs from the environment (defaults when unset); a value that does not parse or is not positive raises RowpairRefused
    naming the variable."""
    env = os.environ if env is None else env
    out: Dict[str, object] = {}
    for key, (var, default, conv) in KNOBS.items():
        raw = (env.get(var) or "").strip()
        if not raw:
            out[key] = conv(default); continue
        try:
            val = conv(raw)
        except (TypeError, ValueError):
            raise _refused(f"rowchunk: {var}={raw!r} does not parse as {conv.__name__}") from None
        if val <= 0:
            raise _refused(f"rowchunk: {var}={raw!r} must be positive")
        out[key] = val
    return out


def install(model, P: int, levers: Sequence[str] = MEMBERS) -> dict:
    """Bind the named members on this rank (``P`` > 1; at ``P == 1`` nothing is bound and the record says so). Returns the record;
    raises RowpairRefused by name on a member that cannot bind."""
    wanted = [n for n in MEMBERS if n in set(levers)]
    unknown = sorted(set(levers) - set(MEMBERS))
    if unknown:
        raise _refused(f"rowchunk: unknown lever name(s) {unknown} (members: {list(MEMBERS)})")
    rec = {"levers": {n: False for n in MEMBERS}, "why": {}, "knobs": {}, "P": int(P)}
    if int(P) <= 1:
        rec["why"] = {n: "n_gpu=1: the row-sharded route is not installed" for n in wanted}
        STATE.update(installed=False, **{k: rec[k] for k in ("levers", "why", "knobs", "P")})
        return rec
    orphan = [n for n in wanted if n in RIDES_ON_CONFROWS and "confrows" not in wanted]
    if orphan:
        raise _refused(f"rowchunk: {orphan} ride on confrows, which is not in the set (subtract them together: the ablation variable names dependents)")
    for n in wanted:                                                             # a member named without a member it rides on is refused by name, never run
        missing = [b for b in RIDES_ON.get(n, ()) if b not in wanted]
        if missing:
            raise _refused(f"rowchunk: {n} rides on {', '.join(missing)}, which is not in the set (subtract them together: "
                           f"the ablation variable names dependents)")
    knobs = read_knobs()
    rec["knobs"] = dict(knobs)
    for n in MEMBERS:
        if n not in wanted:
            rec["why"][n] = "not in the set"
    from . import confmem, confrows, zbf16rows
    from .. import rowpair as RP
    if "injrows" in wanted:                                                       # the recycle inject per row block: rowpair._inject_rows reads INJ
        RP.INJ.update(on=True, mb=float(knobs["inject_mb"]), min_tokens=int(knobs["min_tokens"]))
        rec["levers"]["injrows"] = True
    else:
        RP.INJ.update(on=False)
    if "confrows" in wanted:                                                      # the sharded confidence statement per row block (+ its three flags)
        try:
            r = confrows.apply(mb=float(knobs["conf_mb"]), min_N=int(knobs["min_tokens"]), bf16_pair=("confbf16" in wanted),
                                  pde_skip=("pdeskip" in wanted), confmem=("confmem" in wanted))
        except confrows.ConfRowsRefused as e:
            raise _refused(f"rowchunk: confrows cannot bind: {e}") from e
        feats = set(r.get("features") or ()) if not r.get("already") else set(r.get("features") or ())
        rec["levers"]["confrows"] = True
        for flag, feat in (("confbf16", "confbf16"), ("pdeskip", "pdeskip"), ("confmem", "confmem")):
            if flag in wanted:
                rec["levers"][flag] = feat in feats
                if feat not in feats:
                    raise _refused(f"rowchunk: {flag} was named but the row-blocked confidence statement did not take it (features={sorted(feats)})")
    if "biasfree" in wanted:                                                      # the sampler's pair-bias buffers released once sample() returns
        try:
            r = confmem.apply(model)
        except RuntimeError as e:                                                # no structure_head on this model
            raise _refused(f"rowchunk: biasfree cannot bind: {e}") from e
        if not (bool((r or {}).get("biasfree")) or "sample" in confmem._ORIG):   # True = wrapped now; False with the wrapper recorded = wrapped by an earlier call
            raise _refused("rowchunk: biasfree cannot bind: structure_head.sample was not wrapped")
        rec["levers"]["biasfree"] = True
    if "zbf16" in wanted:                                                         # the confidence head reads a bf16 copy of the pair rows
        try:
            r = zbf16rows.apply(model=model, min_N=int(knobs["min_tokens"]), mb=float(knobs["zbf16_mb"]))
        except zbf16rows.ZBf16Refused as e:
            raise _refused(f"rowchunk: zbf16 cannot bind: {e}") from e
        rec["levers"]["zbf16"] = True
    STATE.update(installed=True, **{k: rec[k] for k in ("levers", "why", "knobs", "P")})
    return rec


def levers_on() -> Dict[str, bool]:
    """``{member: bound on this rank}`` — the registry probe ``("rc", <name>)`` reads it (all False before install / at n_gpu 1)."""
    return {n: bool((STATE.get("levers") or {}).get(n)) for n in MEMBERS}


def evidence(name: str) -> Dict[str, object]:
    """The LEVER line's evidence fields for a bound member: the knobs it runs under."""
    k = dict(STATE.get("knobs") or {})
    if not k or not levers_on().get(name):
        return {}
    ev: Dict[str, object] = {"min_tokens": int(k["min_tokens"])}
    if name == "injrows":
        ev["block_mb"] = k["inject_mb"]
    elif name in ("confrows", "confbf16", "confmem"):
        ev["block_mb"] = k["conf_mb"]
    elif name == "zbf16":
        ev["block_mb"] = k["zbf16_mb"]
    return {key: (int(v) if isinstance(v, float) and float(v).is_integer() else v) for key, v in ev.items()}


# The counters that show each member ENGAGED (its re-issued statement ran) — read from the modules' own STATS. calls=0 on a fold at or above
# the token floor = the member was bound but never reached: stack.guards_after_run reports it "unreached" (partial, refused by name).
ENGAGED_BY = {"injrows": "chunked", "confrows": "head_chunked", "confbf16": "prologue_calls", "pdeskip": "pde_skipped", "confmem": "wein_calls",
              "biasfree": "sample_calls", "zbf16": "conf_calls"}
COUNTERS = {"injrows": ("calls", "chunked", "blocks", "rows_per_block", "floor_skips"),
            "confrows": ("calls", "head_calls", "head_chunked", "tm_chunked", "add_chunked", "row_blocks", "rows", "floor_skips", "dispatched_blocked"),
            "confbf16": ("prologue_calls", "prologue_blocks", "pair_dtype"), "pdeskip": ("pde_skipped",),
            "confmem": ("wein_calls", "wein_blocks", "rap_calls", "rap_blocks"),
            "biasfree": ("sample_calls", "bias_frees", "bias_freed_GiB", "bias_free_errors"), "zbf16": ("calls", "conf_calls", "conf_blocks", "stock", "gib_avoided", "floor_skips")}
# Reachability: the phase of the fold each member's statement lives in, as the route records it per fold on this rank (rowpair's fold
# record: reached_inject = recycle injects issued, reached_sample = the sampler returned, reached_confidence = the confidence head
# returned). A member is judged only against folds at or above the token floor in which ITS phase ran: expected calls = the sum of its
# phase's marks over those folds. Conditions under which a bound member's counter is legitimately 0 and is therefore NOT judged:
#   every member   no fold at or above EF2_ROWPAIR_ROWCHUNK_MIN_TOKENS ran on this rank (short inputs; warm's public input), or the
#                  run died before the member's phase (the run keeps its own failure code);
#   injrows        never otherwise: the inject is issued max(1, num_loops + 1) times per sharded fold, whatever --num_loops says;
#   biasfree       the fold did not reach structure_head.sample on this rank;
#   confrows, confbf16, pdeskip, confmem, zbf16   the model has no confidence head or the fold did not reach it; the number of
#                  diffusion samples does not matter (S > 1 runs the same statement once per sample).
PHASE = {"injrows": "reached_inject", "biasfree": "reached_sample", "confrows": "reached_confidence", "confbf16": "reached_confidence",
         "pdeskip": "reached_confidence", "confmem": "reached_confidence", "zbf16": "reached_confidence"}


def _module_stats() -> Dict[str, dict]:
    """{member: its module's STATS dict} for the bound members (the three confidence flags read confrows' STATS)."""
    out: Dict[str, dict] = {}
    on = levers_on()
    try:
        from .. import rowpair as RP
        from . import confmem, confrows, zbf16rows
        src = {"injrows": RP.INJ_STATS, "confrows": confrows.STATS, "confbf16": confrows.STATS, "pdeskip": confrows.STATS,
               "confmem": confrows.STATS, "biasfree": confmem.STATS, "zbf16": zbf16rows.STATS}
    except Exception:  # noqa: BLE001
        return out
    for n in MEMBERS:
        if on.get(n):
            out[n] = dict(src[n])
    return out


def counters():
    """One flat dict of scalars for the EXIT tally (report.memory_stats -> ``rowchunk={...}``): ``levers`` (the bound members joined by
    '+', or none), the knobs, and ``<member>_<counter>`` for every bound member; None in a process where the route did not install these
    levers (n_gpu 1, or the module merely imported): the EXIT line then carries no rowchunk entry."""
    if not STATE.get("installed"):
        return None
    on = levers_on()
    out: dict = {"levers": "+".join(n for n in MEMBERS if on.get(n)) or "none"}
    out.update({k: v for k, v in (STATE.get("knobs") or {}).items()})
    for n, st in _module_stats().items():
        for c in COUNTERS[n]:
            v = st.get(c)
            if v is not None and not isinstance(v, (dict, list)):
                out[f"{n}_{c}"] = str(v).replace("torch.", "") if not isinstance(v, (int, float)) else v
    return out


def fold_fields(n_tokens: int) -> list:
    """The rowpair fold line's words for these levers: ``rowchunk=<members|none> rowchunk_floor=<tokens> rowchunk_rows=blocked|whole:below_floor``
    and ``rc_<member>_<counter>=<value so far>`` for every bound member (kit line grammar: blank-free k=v)."""
    if not STATE.get("installed"):
        return [("rowchunk", "none")]
    on = levers_on(); floor = int((STATE.get("knobs") or {}).get("min_tokens") or 0)
    pairs = [("rowchunk", ",".join(n for n in MEMBERS if on.get(n)) or "none"), ("rowchunk_floor", floor),
             ("rowchunk_rows", "blocked" if int(n_tokens) >= floor else "whole:below_floor")]
    for n, st in _module_stats().items():
        for c in COUNTERS[n]:
            v = st.get(c)
            if v is not None and not isinstance(v, (dict, list)):
                pairs.append((f"rc_{n}_{c}", str(v).replace("torch.", "") if not isinstance(v, (int, float)) else v))
    return pairs


def expected_calls(folds, floor: int) -> Dict[str, int]:
    """``{member: how many times its phase ran on this rank in folds at or above the floor}`` from the route's fold records (PHASE)."""
    above = [f for f in (folds or []) if int((f or {}).get("N") or 0) >= int(floor)]
    return {n: sum(int((f or {}).get(PHASE[n]) or 0) for f in above) for n in MEMBERS}


def engagement(folds=None) -> Dict[str, tuple]:
    """After a run: ``{member: ("unreached", note)}`` for every BOUND member whose engagement counter (ENGAGED_BY) is still 0 although its
    phase ran (``expected_calls`` > 0) in a fold at or above the token floor on this rank — bound, reported, never run is a partial
    application (stack.guards_after_run -> settle_guards: refused by name). A member whose phase never ran at or above the floor is not
    judged (see PHASE for the conditions): folds below the floor keep the whole-shard statements by declaration and prove nothing."""
    if not STATE.get("installed"):
        return {}
    floor = int((STATE.get("knobs") or {}).get("min_tokens") or 0)
    if folds is None:
        try:
            from .. import rowpair as RP
            folds = list(RP.STATE.get("folds") or [])
        except Exception:  # noqa: BLE001
            folds = []
    exp = expected_calls(folds, floor)
    n_above = sum(1 for f in folds if int((f or {}).get("N") or 0) >= floor)
    out: Dict[str, tuple] = {}
    for n, st in _module_stats().items():
        c = ENGAGED_BY[n]
        if exp.get(n, 0) > 0 and int(st.get(c) or 0) == 0:
            out[n] = ("unreached", f"{c}=0 with {PHASE[n]}={exp[n]} over {n_above} fold(s) at N>={floor} on this rank: bound, never engaged")
    return out
