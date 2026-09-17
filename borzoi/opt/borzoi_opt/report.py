"""Observability for borzoi_opt: the activation line, the swap line and the exit tally — every one on stderr, prefixed ``[borzoi-opt]``.

* ``ACTIVE mode=<m> route=<cli|env|python> kit=<name> writer=<w> entry=<path> gpu=<name> tensorflow=<v> kit_env=<KIT_FWD=1> det=<on|off>``
  — a kit mode resolved and about to run (or running: the hook in the kit entry's own process);
* ``OFF route=<r> entry=<stock path> stock=<sha8> pinned=<true|false> gpu=<name>`` — mode off, the stock script about to run;
* ``DRY-RUN mode=<m> ...`` — ``check``: the same fields, nothing runs;
* ``NOT ACTIVE: <reason> (mode=<m>)`` — the gate refused;
* ``SWAP mode=<m> stock=<path> -> kit=<path>`` — the BORZOI_OPT hook re-execs the kit entry in place of the stock script;
* ``FALLBACK <lever>=<detail>`` — a tuned heuristic took its broad default (the cores probe -> the OS core count); outputs unaffected, the exit the job's own
* ``EXIT mode=<m> rc=<n> post_applies=<bool> writer=<w> pool=<threads> partial=<levers|none> allow_partial=<on|off>
  gated=<n> routed=<n>`` — after the run, from the kit's own stamp (``kit_stamp.json``) and its judgement (modes.applied); ``NOT ACTIVE: partial
  activation — <detail>; exit 3 (--allow-partial records and proceeds)`` (or, under --allow-partial, ``PARTIAL allowed: <detail>
  (--allow-partial, recorded)``), ``GATED <reason>`` and ``ROUTED post=stock:<reason> — …`` (a per-call stock route: options outside the
  chunked post's set, variants outside its shape class — named, exit unaffected) lines precede it when they apply; ``EXIT mode=off rc=<n>
  stock_proof=<ok|FAIL>`` for stock.
"""
from __future__ import annotations

import sys

PREFIX = "[borzoi-opt]"


def _sha8(s) -> str:
    return str(s)[:8] if s else "none"


def _gpu(g) -> str:
    if not g:
        return "none"
    name = str(g.get("name") or "none").replace(" ", "_")
    return f"{name}({g.get('memory_mib')}MiB,cc{g.get('cc')})" if g.get("memory_mib") else name


def _kit_env(d) -> str:
    """The kit switches the job's environment gains (the composition), KIT_* names only — the det recipe is reported by ``det=``."""
    d = {k: v for k, v in (d or {}).items() if k.startswith("KIT_")}
    return ",".join(f"{k}={v}" for k, v in sorted(d.items())) or "none"


def activation_line(rep: dict) -> str:
    rep = rep or {}
    mode = rep.get("mode")
    tf = (rep.get("versions") or {}).get("tensorflow")
    if not rep.get("ok", True) or (not rep.get("active") and not rep.get("dry_run") and mode != "off"):
        return f"{PREFIX} NOT ACTIVE: {rep.get('reason') or 'no reason given'} (mode={mode})"
    if mode == "off":
        st = rep.get("stock") or {}
        head = "DRY-RUN mode=off" if rep.get("dry_run") else "OFF"
        line = (f"{PREFIX} {head} route={rep.get('route')} entry={st.get('entry')} stock={_sha8(st.get('sha256'))} pinned={str(bool(st.get('pinned'))).lower()} "
                f"gpu={_gpu(rep.get('gpu'))} tensorflow={tf} stripped={','.join(rep.get('env_stripped') or []) or 'none'}")
    else:
        k = rep.get("kit") or {}
        head = f"DRY-RUN mode={mode}" if rep.get("dry_run") else f"ACTIVE mode={mode}"
        line = (f"{PREFIX} {head} route={rep.get('route')} kit={k.get('name')} writer={rep.get('writer')} entry={rep.get('entry')} "
                f"gpu={_gpu(rep.get('gpu'))} tensorflow={tf} kit_env={_kit_env(rep.get('env'))} det={'on' if rep.get('det') else 'off'}")
    if rep.get("notes"):                                            # environment differences (GPU class, stack versions, the stock entry's state under a kit mode): named, never a gate
        line += " notes=" + "; ".join(rep["notes"])
    return line


def hook_active_line(rep: dict, program: str, levers) -> str:
    """``ACTIVE mode=<m> route=hook program=<path|-c|-m> levers=<a,b,c> …`` — the library route: the kit's forward call and one-hot installed
    on the stock library in a program that is not the documented command (printed once, when the library is imported)."""
    line = activation_line(rep)
    if " ACTIVE " not in line:
        return line
    return line.replace(f" route={rep.get('route')} ", f" route=hook program={program or '-'} levers={','.join(levers)} ", 1)


def hook_exit_line(mode: str, program: str, forward: dict | None, onehot: dict | None) -> str:
    """``EXIT mode=<m> route=hook program=<p> forward: n_calls=<n> n_eager_calls=<n> n_graph_calls=<n> n_traces=<n> onehot=<LUT|none>`` at
    the program's exit (atexit) — what the library levers did in this process."""
    f = forward or {}
    fw = " ".join(f"{k}={f.get(k, 0)}" for k in ("n_calls", "n_eager_calls", "n_graph_calls", "n_traces", "n_copy_free", "n_stock_calls")) if forward else "not installed (baskerville.seqnn never imported)"
    return f"{PREFIX} EXIT mode={mode} route=hook program={program or '-'} forward: {fw} onehot={(onehot or {}).get('kit_onehot', 'none')}"


def swap_line(mode: str, stock_entry: str, kit_entry: str) -> str:
    return f"{PREFIX} SWAP mode={mode} stock={stock_entry} -> kit={kit_entry}"


PARTIAL_REFUSED = "{prefix} NOT ACTIVE: partial activation — {detail}; exit {code} (--allow-partial records and proceeds)"
PARTIAL_ALLOWED = "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)"
ROUTED = "{prefix} ROUTED {detail} (per call, by name; the exit is the job's own)"   # a per-call stock route — `post=stock:<reason> — …` — one census-able line per reason, never a refusal


def routed_line(detail: str) -> str:
    """``ROUTED post=stock:<reason> — <what ran the stock path> (per call, by name; the exit is the job's own)`` — modes.applied's ``routed``."""
    return ROUTED.format(prefix=PREFIX, detail=detail)


def partial_line(partial, reasons, allow_partial: bool, exit_code: int) -> str:
    """The one partial-activation line after a kit run (the family grammar; the fixed parts are PARTIAL_REFUSED / PARTIAL_ALLOWED):
    ``NOT ACTIVE: partial activation — <detail>; exit 3 (--allow-partial records and proceeds)`` — the run refused by name — or, under
    --allow-partial, ``PARTIAL allowed: <detail> (--allow-partial, recorded)`` with the exit the job's own. ``<detail>`` is the lever
    names first, then each lever's reason (modes.applied's ``partial_reasons``, ``<lever>: <why>``), joined by `` · ``."""
    detail = ", ".join(partial) + " — " + " · ".join(reasons)
    if allow_partial:
        return PARTIAL_ALLOWED.format(prefix=PREFIX, detail=detail)
    return PARTIAL_REFUSED.format(prefix=PREFIX, detail=detail, code=exit_code)


def fallback_line(lever: str, detail: str) -> str:
    """A tuned heuristic took its broad default, by name; outputs unaffected, the exit the job's own."""
    return f"{PREFIX} FALLBACK {lever}={detail}"


def exit_line(mode: str, rc: int, stamp: dict | None = None, proof: dict | None = None, partial=None, allow_partial: bool = False, gated=None, routed=None) -> str:
    if mode == "off":
        ok = "ok" if (proof or {}).get("ok") else ("FAIL" if proof is not None else "none")
        after = (proof or {}).get("kit_modules_loaded_after") or []
        return f"{PREFIX} EXIT mode=off rc={rc} stock_proof={ok} kit_modules_after={','.join(after) or 'none'}"
    st = stamp or {}
    post = st.get("post") or {}
    writer = (post.get("writer") or {}).get("writer") if isinstance(post.get("writer"), dict) else post.get("writer")
    pool = st.get("pool")
    threads = pool.get("threads") if isinstance(pool, dict) else pool
    return (f"{PREFIX} EXIT mode={mode} rc={rc} kit={st.get('kit') or 'no stamp'} post_applies={str(st.get('post_applies')).lower()} "
            f"writer={writer if writer is not None else 'none'} pool={threads if threads is not None else 'none'} "
            f"partial={','.join(partial) if partial else 'none'} allow_partial={'on' if allow_partial else 'off'} gated={len(gated or [])} routed={len(routed or [])}")


def log(line: str, stream=None) -> str:
    (stream or sys.stderr).write(line + "\n")
    (stream or sys.stderr).flush()
    return line
