"""Observability primitives: the exit table, the exit verdict of a run, the line grammar the kit composes its lines from, the exit tally.

Contract. Exit codes are one per condition — ``EXIT_OK`` · ``EXIT_FAIL`` (failed or incomplete) · ``EXIT_USAGE`` · ``EXIT_NOT_ACTIVE``
(not active or partial). :func:`verdict` is the exit rule of a completed run: ``incomplete`` (outputs short of the request) turns a 0
into EXIT_FAIL first; a ``partial`` activation (the report's list of the mode's levers the kit's records show not applied, its declared
guards excluded) turns a 0 into EXIT_NOT_ACTIVE unless ``allow_partial`` (recorded); a run that failed on its own keeps its own code
with the partial recorded in the manifest. The exit vocabulary has three outcomes (``opt_core.gates`` states the same contract):

  * ``words`` — named UNCERTAINTY a lever engaged under: the environment is not the measured one (``stack=drift(…)``, an uncertified
    card ``card=uncertified(…)`` / ``card_support=uncertified:<sm>``, an unreadable pin ``core_pin=unreadable``, a cache miss
    ``cache=miss(n)`` / ``autotune=cold``, a safe cell ``settings=safe:…``, an unknown part inside a cache key). The lever is ACTIVE;
    the words print on the activation line and are recorded (``verdict()["words"]``); the exit code is untouched (0). :func:`word` is
    their ONE grammar (a blank-free ``key=value[(part,…)]`` token), :func:`words_of` collects them, :func:`with_words` appends them.
  * a CANNOT-RUN event — a lever of the mode whose mechanism fails where it was asked to run (compile / launch failure, unsupported
    shape or dtype at run time, a failed capture, a card below the mechanism's floor): the lever raises an exception carrying
    ``cannot_run = True`` (``opt_core.gates.is_cannot_run``), the kit refuses the MODE by name — :func:`not_active_line`, EXIT_NOT_ACTIVE
    — and nothing continues on the engine's stock code under the mode's label.
  * ``partial`` / accounting — the report's list of the mode's levers the records show not applied, a census that does not close, an
    undeclared fallback reason, an ``error:*`` event: EXIT_NOT_ACTIVE through :func:`verdict` unless ``--allow-partial`` (the recorded
    opt-out), fail-closed.

Lines are ``[<tag>] <VERB> key=value ...`` on stderr; the kit composes its ACTIVE / DRY-RUN /
APPLIED / ready lines from :func:`prefix`, :func:`kv`, :func:`join`, :func:`gpu_label`, :func:`line` and :func:`with_words` (their bytes are the kit's — the
launcher matches them) and prints them with :func:`emit`; the per-lever activation-evidence line has ONE grammar,
:func:`lever_line`; the partial lines and the EXIT tally follow the kit's grammar (the kit keeps its own formatter — see the
constants, :func:`exit_tally_line` and :func:`tally_fields` — the per-module counter grammar that never drops a key silently: equality keys
first, the rest alphabetically, a named ``…+N`` marker under a cap). :func:`register_exit_tally` prints the kit's own EXIT line once at interpreter exit; a failing
line function prints a tally-failed line, never nothing. :func:`dump` is the one JSON text form of a report.
"""
from __future__ import annotations

import atexit
import json
import os
import re
import sys
from typing import Callable, Mapping, Optional, Sequence

EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3        # one code per condition: ok · failed or incomplete · usage · not active or partial

# The partial-activation lines are not one grammar: there are four forms (a refusal naming when it was
# found, a "PARTIAL allowed by --allow-partial:" form, and two others) and the launcher holds the kit to its own bytes. The kit keeps its
# formatter; these two templates are the default form, for a kit that has none of its own.
PARTIAL_REFUSED = "{prefix} NOT ACTIVE: partial activation — {detail}; exit " + str(EXIT_NOT_ACTIVE) + " (--allow-partial records and proceeds)"
PARTIAL_ALLOWED = "{prefix} PARTIAL allowed: {detail} (--allow-partial, recorded)"


def prefix(tag: str) -> str:
    return f"[{tag}]"


def join(items) -> str:
    """``a,b,c`` — or ``none`` for an empty or missing list."""
    if not items:
        return "none"
    if isinstance(items, (str, bytes)):
        return str(items)
    return ",".join(str(x) for x in items)


def kv(*pairs, **fields) -> str:
    """``k=v k2=v2`` in order: positional ``(key, value)`` pairs first (an order the keywords cannot express, e.g. a repeated key), then
    the keywords. Lists and tuples go through :func:`join`; ``None`` prints ``none``; a dict prints ``k:v,k2:v2``."""
    out = []
    for k, v in list(pairs) + list(fields.items()):
        if v is None:
            s = "none"
        elif isinstance(v, (list, tuple, set, frozenset)):
            s = join(list(v))
        elif isinstance(v, dict):
            s = ",".join(f"{a}:{b}" for a, b in v.items()) or "none"
        else:
            s = str(v)
        out.append(f"{k}={s}")
    return " ".join(out)


def line(prefix_: str, verb: str, *pairs, **fields) -> str:
    """``<prefix> <VERB> k=v ...`` — one evidence line: the kit's prefix (:func:`prefix`), its verb (``ACTIVE``, ``COMMAND``, ``STEP``, ``DONE``
    ...) and the fields through :func:`kv` (positional pairs first). A kit binds its prefix once: ``line = functools.partial(report.line, PREFIX)``."""
    body = kv(*pairs, **fields)
    return f"{prefix_} {verb}" + (" " + body if body else "")


_BLANKS = re.compile(r"\s+")


def _token(x) -> str:
    s = "none" if x is None else str(x).strip()
    return _BLANKS.sub("_", s) or "none"


def word(key: str, value=None, *parts) -> str:
    """ONE grammar of a named evidence word (module contract, ``words``): ``<key>=<value>`` or ``<key>=<value>(<part>,<part>…)`` —
    ``word("card", "uncertified", "NVIDIA L4", "22731MiB")`` -> ``card=uncertified(NVIDIA_L4,22731MiB)``, ``word("core_pin", "unreadable")``
    -> ``core_pin=unreadable``. Blank-free (runs of blanks inside a part become ``_``; a line is split on blanks) and ASCII by the caller's
    choice of parts (``!=``, not a non-ASCII sign: the word prints on a stderr whose encoding is the box's). A word IS a ``k=v`` field."""
    body = _token(value)
    if parts:
        body += "(" + ",".join(_token(p) for p in parts) + ")"
    return f"{_token(key)}={body}"


def words_of(*items) -> list:
    """Every word of ``items`` in order, each once: an item is a word (str), an object carrying ``.words`` (a :class:`opt_core.gates.Gate`,
    a tuple attribute) or ``.words()`` (a capture / cache record), a mapping with a ``"words"`` entry (a report), or an iterable of items.
    The kit collects its activation's words with ONE call: ``words_of(gates, graph_pool, rep)``."""
    out: list = []

    def add(x):
        if x is None:
            return
        if isinstance(x, str):
            if x and x not in out:
                out.append(x)
            return
        w = getattr(x, "words", None)
        if w is not None and not isinstance(x, Mapping):
            add(list(w() if callable(w) else w))
            return
        if isinstance(x, Mapping):
            add(list(x.get("words") or []))
            return
        for y in x:
            add(y)

    for it in items:
        add(it)
    return out


def with_words(text: str, words=None) -> str:
    """``text`` with the words appended as fields (``<text> <word> <word>``); ``text`` unchanged without words. The kit's ACTIVE line:
    ``emit(with_words(line(PREFIX, "ACTIVE", mode=…, levers=…), words_of(gates, rep)))``."""
    ws = words_of(words)
    return text + (" " + " ".join(ws) if ws else "")


LEVER = "LEVER"                                                    # the verb of the per-lever activation-evidence line
LEVER_STATES = ("on", "off", "skipped")
LEVER_ORIGINS = ("core", "kit")                                    # where a lever's implementation lives


def lever_line(tag: str, name: str, state: str, *pairs, reason=None, impl=None, origin=None, strategy=None, **fields) -> str:
    """The ONE per-lever activation-evidence line a kit-arm process prints once per lever (stderr, via :func:`emit`)::

        [<tag>] LEVER name=<F<k>.strategy id | local id> state=<on|off|skipped> [reason=<why>] impl=<module or kernel@version> origin=<core|kit>
                [strategy=<canonical id>] [served=<n> fallback=<n> fallback_by=<reason:n,…> min_tokens=<n> shapes=<…> — kernel / gated levers] [evidence k=v …]

    ``state``: ``on`` = applied and live in this process; ``off`` = not in this mode's lever set; ``skipped`` = selected but not applied —
    ``reason`` is then mandatory (it feeds the kit's partial census). The grammar is enforced here, once: ``name``, ``state``,
    ``reason`` (when given), ``impl``, ``origin`` lead in this order; ``impl`` is required; ``origin`` is ``core`` (the implementation lives in
    opt_core) or ``kit``; ``strategy``, when given, has the FORM of a strategy id (:func:`strategy_form`: a family id ``F<k>.<name>`` or a
    local ``LOCAL.<name>`` / ``LOCAL.<kit>.<name>``); no value may contain a blank (a line is split on blanks). The keys
    after the pinned ones are the caller's evidence in the caller's order (positional ``pairs`` first, then ``fields``; a positional
    ``impl`` / ``origin`` / ``strategy`` pair fills the pinned slot, the keyword wins when both are given). :meth:`opt_core.counters.Ledger.line`
    renders the kernel-lever form."""
    if state not in LEVER_STATES:
        raise ValueError(f"lever_line: state must be one of {'|'.join(LEVER_STATES)}, not {state!r}")
    if state == "skipped" and not reason:
        raise ValueError(f"lever_line: lever {name!r} state=skipped needs a reason")
    rest = []
    for k, v in pairs:
        if k == "impl":
            impl = v if impl is None else impl
        elif k == "origin":
            origin = v if origin is None else origin
        elif k == "strategy":
            strategy = v if strategy is None else strategy
        elif k in ("name", "state", "reason"):
            raise ValueError(f"lever_line: {k!r} is a pinned key, pass it by argument")
        else:
            rest.append((k, v))
    if impl is None or str(impl) == "":
        raise ValueError(f"lever_line: lever {name!r} needs impl=<module or kernel@version>")
    if origin not in LEVER_ORIGINS:
        raise ValueError(f"lever_line: lever {name!r} origin must be one of {'|'.join(LEVER_ORIGINS)}, not {origin!r}")
    if strategy is not None:
        strategy_form(str(strategy))
    head = [("name", name), ("state", state)] + ([("reason", reason)] if reason is not None else []) + [("impl", impl), ("origin", origin)]
    if strategy is not None:
        head.append(("strategy", strategy))
    all_pairs = head + rest + list(fields.items())
    for k, v in all_pairs:
        text = kv((k, v))
        if any(c.isspace() for c in text):
            raise ValueError(f"lever_line: lever {name!r}: the value of {k!r} contains a blank ({text!r}); values are single tokens")
    return line(prefix(tag), LEVER, *all_pairs)


STRATEGY_FORMS = ("F<k>.<name>", "LOCAL.<name>")                # the two shapes of a strategy id: a family id (F1 … F7 …), a local id (LOCAL.<kit>.<name> for a kit's own)
_STRATEGY_RX = re.compile(r"(?:F[0-9]+|LOCAL)\.\S+")


def strategy_form(sid: str) -> str:
    """``sid`` itself when it has the form of a strategy id — a family id ``F<k>.<name>`` (``F7.tensor_parallel``) or a local id
    ``LOCAL.<name>`` (a kit's own: ``LOCAL.<kit>.<name>``), one blank-free token; a ValueError naming the two forms otherwise."""
    if _STRATEGY_RX.fullmatch(str(sid)):
        return str(sid)
    raise ValueError(f"lever_line: strategy {sid!r} is not a strategy id ({' | '.join(STRATEGY_FORMS)})")


def emit(text: str, stream=None) -> str:
    """Print one line on stderr (or ``stream``), flushed. Returns the text."""
    s = stream or sys.stderr
    s.write(text + "\n")
    s.flush()
    return text


def dump(rep) -> str:
    """The JSON text of a report: indent 1, sorted keys, ``str()`` for values JSON cannot hold."""
    return json.dumps(rep, indent=1, sort_keys=True, default=str)


def gpu_label(gpu) -> str:
    """``name(smNN)`` · ``name`` · ``none`` from a probe dict (``{"name", "sm"|"cc"}``) or a plain string."""
    if isinstance(gpu, Mapping):
        name = gpu.get("name")
        sm = gpu.get("sm")
        if sm in (None, "") and gpu.get("cc"):
            sm = "sm" + str(gpu["cc"]).replace(".", "")
        if not name:
            return "none"
        if sm in (None, ""):
            return str(name)
        sm = str(sm)
        return f"{name}({sm})" if sm.startswith("sm") else f"{name}(sm{sm})"
    return str(gpu) if gpu else "none"


def not_active_line(tag: str, reason: str, head: Optional[str] = None) -> str:
    """``[tag] NOT ACTIVE: <reason>`` plus `` (<head>)`` when the kit gives one (its ``mode=… variant=…``)."""
    return f"{prefix(tag)} NOT ACTIVE: {reason}" + (f" ({head})" if head else "")


def allow_partial(flag, env_name: str, environ: Optional[Mapping[str, str]] = None) -> bool:
    """The opt-out of every verb: the parsed ``--allow-partial`` flag or ``<env_name>=1`` — the one reader."""
    environ = os.environ if environ is None else environ
    return bool(flag) or (environ.get(env_name) or "").strip() == "1"


def verdict(rc: int, rep: Optional[Mapping], allow_partial: bool, incomplete: Optional[str] = None) -> dict:
    """The exit rule (module contract). Returns ``{exit_code, partial, gated, allow_partial, incomplete, words}`` — the manifest's fields,
    written whichever code wins. ``partial`` (the report's levers not applied) prices the exit; ``words`` (the report's ``words``: the named
    uncertainty the levers engaged under, :func:`words_of`) are recorded and never price it."""
    rep = rep or {}
    partial = list(rep.get("partial") or [])
    code = int(rc)
    if code == EXIT_OK and incomplete:
        code = EXIT_FAIL
    elif code == EXIT_OK and partial and not allow_partial:
        code = EXIT_NOT_ACTIVE
    return {"exit_code": code, "partial": partial, "gated": list(rep.get("gated") or []), "allow_partial": bool(allow_partial), "incomplete": incomplete,
            "words": words_of(rep.get("words"))}


def partial_detail(v: Mapping, reasons: Optional[Mapping] = None) -> str:
    """``<detail>`` of the partial lines: the lever names first, then the kit's reason per lever."""
    why = "; ".join(f"{n}: {(reasons or {}).get(n) or 'no reason recorded'}" for n in v["partial"])
    return f"levers={join(v['partial'])} ({why})"


def partial_line(tag: str, v: Mapping, reasons: Optional[Mapping] = None) -> Optional[str]:
    """The one line the exit rule prints on a partial run: refused (the verdict is EXIT_NOT_ACTIVE) — ``PARTIAL_REFUSED``; allowed —
    ``PARTIAL_ALLOWED``. A run that failed on its own prints its failure's own line (None here)."""
    if not v["partial"]:
        return None
    detail = partial_detail(v, reasons)
    if v["allow_partial"]:
        return PARTIAL_ALLOWED.format(prefix=prefix(tag), detail=detail)
    if v["exit_code"] == EXIT_NOT_ACTIVE:
        return PARTIAL_REFUSED.format(prefix=prefix(tag), detail=detail)
    return None


def log_once(rep: Optional[dict], line: str, stream=None) -> str:
    """Print ``line`` unless the report says it was logged; marks it logged. Returns the line."""
    if not (rep or {}).get("logged"):
        print(line, file=stream or sys.stderr, flush=True)
        if rep is not None:
            rep["logged"] = True
    return line


# ----------------------------------------------------------------------------------------------------------------- exit tally
_TALLIES: dict = {}
_TALLIES_PRINTED: set = set()             # tags whose EXIT line has been printed (atexit or a forced exit) — a forced exit prints the rest

TALLY_IDENTITY_KEYS = ("installed", "armed", "active", "mode", "state", "n_gpu", "rank", "group")   # printed FIRST in every module's tally group
TALLY_VALUE_CHARS = 40                                                      # a scalar's text longer than this is shortened WITH a marker


def _tally_scalar(v, limit: int = TALLY_VALUE_CHARS) -> str:
    s = "none" if v is None else str(v)
    return s if len(s) <= limit else s[: max(1, limit - 1)] + "…"


def tally_fields(stats: dict, *, identity_keys: Sequence[str] = TALLY_IDENTITY_KEYS, max_fields: Optional[int] = None) -> list:
    """The EXIT-tally grammar for a kit's per-module counters: ``stats = {module: {key: scalar | nested}}`` -> ``['<module>={k=v,k2=v2,…}',
    …]`` (modules in the dict's order). Inside a group the IDENTITY keys (``identity_keys`` ∩ present, in that order — whether the lever was
    installed / armed, the rank, the GPU count) print FIRST, then the remaining scalar keys alphabetically; nested dicts / lists print as
    ``key=<n>items``. NEVER silent: with ``max_fields=None`` every key prints; with a cap the first ``max_fields`` keys print and the rest is
    NAMED as ``…+N`` inside the group, so a launcher expecting ``installed=True,n_gpu=P,rank=0`` always finds the equality keys and a reader
    always sees that keys were left out."""
    out = []
    for mod, d in (stats or {}).items():
        if not isinstance(d, dict):
            out.append(f"{mod}={_tally_scalar(d)}")
            continue
        ident = [k for k in identity_keys if k in d]
        rest = sorted(k for k in d if k not in ident)
        keys = ident + rest
        shown = keys if max_fields is None else keys[: max(0, int(max_fields))]
        parts = []
        for k in shown:
            v = d[k]
            if isinstance(v, dict):
                parts.append(f"{k}=<{len(v)}keys>")
            elif isinstance(v, (list, tuple, set, frozenset)):
                parts.append(f"{k}=<{len(v)}items>")
            else:
                parts.append(f"{k}={_tally_scalar(v)}")
        if len(shown) < len(keys):
            parts.append(f"…+{len(keys) - len(shown)}")
        out.append(f"{mod}={{" + ",".join(parts) + "}")
    return out



def exit_tally_line(tag: str, fields: Optional[str], pid: Optional[int] = None, source: str = "memory") -> str:
    """``[tag] EXIT pid=<pid> source=<source> <fields>`` — or, with no fields, the never-silent form naming why. ONE kit's grammar, offered
    to a kit that has none; a kit whose EXIT line the launcher holds composes the whole line itself (``register_exit_tally`` takes the line)."""
    pid = os.getpid() if pid is None else pid
    if fields:
        return f"{prefix(tag)} EXIT pid={pid} source={source} {fields}"
    return f"{prefix(tag)} EXIT pid={pid} no lever counters: the kit's lever modules were never loaded in this process"


def _print_exit_tally(tag: str, line_of: Callable[[], str]) -> None:
    _TALLIES_PRINTED.add(tag)
    try:
        line = line_of()
    except Exception as e:  # noqa: BLE001
        line = f"{prefix(tag)} EXIT tally failed: {e!r}"
    try:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def forced_exit(code: int) -> None:
    """End the process with ``os._exit(code)`` — the exit a guard takes when interpreter exit cannot be left to run (an undrained writer with a
    lost write, a collective library whose destructors would block). ``os._exit`` skips every ``atexit`` hook not yet run and the interpreter's
    own flush, so first: every EXIT tally registered with :func:`register_exit_tally` and not yet printed is printed (latest-registered first,
    the order atexit would have used), then stdout/stderr are flushed. Other exit hooks that have not run by then do not run."""
    for tag, line_of in reversed(list(_TALLIES.items())):
        if tag not in _TALLIES_PRINTED:
            _print_exit_tally(tag, line_of)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:  # noqa: BLE001
            pass
    os._exit(int(code))


def register_exit_tally(tag: str, line_of: Callable[[], str]) -> bool:
    """Print the kit's EXIT line at interpreter exit, once per process per tag. ``line_of`` returns the WHOLE line (the kit's own grammar —
    an evidence line the launcher holds byte-for-byte; ``exit_tally_line`` is one grammar a kit may choose). Registered with atexit, so it
    prints in LIFO order with the kit's other atexit hooks: register it FIRST to print LAST. Returns True when registered now."""
    if tag in _TALLIES:
        return False
    _TALLIES[tag] = line_of
    atexit.register(_print_exit_tally, tag, line_of)
    return True

