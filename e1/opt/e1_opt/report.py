"""Observability for e1_opt: every line the package prints, in one place (all on stdout, flushed, prefixed `[e1-opt]`; the stock
subprocess's lines carry `[e1-opt stock]`). The line grammar is a contract (the RE_* patterns below hold every line to it); nothing else is printed
by the package (the kit's own lines are relayed verbatim by kit_score.py; usage errors go to stderr as `[e1-opt] ERROR: ...`).

  [e1-opt] ACTIVE mode=exact variant=<v> kit=<kit> kit_mode=eager gpu=<torch device name> stack=<key> package=<version> ... [notes="…"]
                                                       `notes="…"` (last, only when there is something to name): what of this host is off
                                                       the kit's pins — a dependency off its pin, an accelerator module that does not
                                                       import, the hub kernel snapshot off its pin, a GPU outside the tested classes, a stock
                                                       install without a commit record. NAMED, never a refusal: the levers engage wherever
                                                       their mechanism applies
  [e1-opt] NOT ACTIVE: mode off: stock e1 (the upstream CLI in a clean subprocess; no environment set, no component applied) (mode=off variant=<v>)
  [e1-opt] NOT ACTIVE: <reason>                       a refusal by name, exit 3 — mode fast / big (not shipped), the installed E1 is not the
                                                       pinned stock (commit / version), the kit's files are missing or disagree with the mode
                                                       table, the weights file is absent, no GPU is visible, det=1 without the recipe's
                                                       environment, or — in the model's process — a lever of the set cannot run / an accelerator
                                                       the levers run on is absent or fell back. Environment drift is never one (it is the
                                                       ACTIVE line's `notes=`). A --mode / --variant disagreeing with E1_OPT / E1_VARIANT prints
                                                       the same line and exits 2 (usage)
  [e1-opt] DRY-RUN mode=<m> variant=<v> ... [notes="…"] would_refuse=<reason|none>      `e1-opt check`: resolved and gated, nothing applied
  [e1-opt] KIT <kit> mode=eager size=<v> card=<class> gpu="<name>" pin=W<n> levers=<k>/<n>     the kit's own line at its apply (kits/eager), then
  [e1-opt] LEVER name=<lever> [note="…"] state=on|off [reason="…"]                             one line per lever of the set
  [e1-opt stock] ENV-CLEAN ok: absent=<vars> kit_modules=none kit_dirs=none          the stock subprocess proves its environment
  STOCK_ENV_CHECK:                                    then the `env | grep -E '^(E1_OPT|E1_KIT|E1_VARIANT|MODEL_OPT)'` output (empty on stock)
  [e1-opt] ready variant=<v> t=<s>                    after the gates, before the tool starts
  [e1-opt] stripped from the item environment: <names>   when the wrapper removed variables from the tool's environment
  [e1-opt] score <file> <s>s  /  [e1-opt stock] score <file> <s>s          the process wall of the tool's command
  [e1-opt] EXIT mode=<m> variant=<v> complete=<0|1> kernels_fallback=<none|names> rc=<code>
                                                       the last line of every `score` run: whether the scores.csv is there, which accelerators
                                                       ran fallen back (from the KERNELS line, mode off), the exit code
  [e1-opt] WARM PASS|FAIL mode=<m> variant=<v> ...    the warm verdict
  [e1-opt stock|exact] KERNELS route=<stock|default|exact> flash_attn=<word> hub_layernorm=<word> flex_attention=<word> site=<stock|kit_attn|…> upstream_says=<True|False|none>
                                                       the KERNELS proof (accel.py): printed ONCE per model process, at the first scorer
                                                       construction (the model on the device, the kit — on its route — applied), from the
                                                       objects the process BOUND at upstream's dispatch sites; word = engaged:<impl>@<version> |
                                                       fallback:<what>(<why>) | absent(<why>); the prefix names the runner (`stock` = the stock
                                                       runner, mode off; `exact` = the kit runner and the env / API arm), `route=` `stock` |
                                                       `default` on the stock route, else `exact`. Under mode off a fallback is NAMED by its word
                                                       (the stock runs the path upstream itself takes there); under exact it is the mode's
                                                       refusal by name (the NOT ACTIVE line follows, exit 3)
"""
from __future__ import annotations

import re
import sys


PREFIX = "[e1-opt]"
STOCK_PREFIX = "[e1-opt stock]"
ENV_CHECK_HEADER = "STOCK_ENV_CHECK:"
ENV_CHECK_PATTERN = r"^(E1_OPT|E1_KIT|E1_VARIANT|MODEL_OPT)"           # the grep of the contract; stock_score.py prints the matches

# the regexes the tests (and any consumer) hold the lines to
RE_ACTIVE = re.compile(r"^\[e1-opt\] ACTIVE mode=(?P<mode>\S+) variant=(?P<variant>\S+) kit=(?P<kit>\S+) kit_mode=(?P<kit_mode>\S+) "
                       r"gpu=(?P<gpu>.+?) stack=(?P<stack>\S+) package=(?P<package>\S+)(?P<rest>( \S+=\S*)*?)(?: notes=\"(?P<notes>[^\"]*)\")?$")
RE_NOT_ACTIVE = re.compile(r"^\[e1-opt\] NOT ACTIVE: (?P<reason>.+)$")
RE_OFF = re.compile(r"^\[e1-opt\] NOT ACTIVE: mode off: stock e1 \(the upstream CLI in a clean subprocess; no environment set, no component applied\) "
                    r"\(mode=off variant=(?P<variant>\S+)\)(?: notes=\"(?P<notes>[^\"]*)\")?$")
RE_DRY_RUN = re.compile(r"^\[e1-opt\] DRY-RUN mode=(?P<mode>\S+) variant=(?P<variant>\S+) .*would_refuse=(?P<would_refuse>.+)$")
RE_STRIPPED = re.compile(r"^\[e1-opt\] stripped from the item environment: (?P<names>\S+)$")
RE_ENV_CLEAN = re.compile(r"^\[e1-opt stock\] ENV-CLEAN ok: absent=(?P<absent>\S+) kit_modules=none kit_dirs=none$")
RE_READY = re.compile(r"^\[e1-opt\] ready variant=(?P<variant>\S+) t=(?P<t>[0-9.]+)$")
RE_SCORE = re.compile(r"^\[e1-opt( stock)?\] score (?P<item>\S+) (?P<s>[0-9.]+)s$")
RE_EXIT = re.compile(r"^\[e1-opt\] EXIT mode=(?P<mode>\S+) variant=(?P<variant>\S+) complete=(?P<complete>[01]) kernels_fallback=(?P<kernels_fallback>\S+) rc=(?P<rc>-?\d+)$")
RE_WARM = re.compile(r"^\[e1-opt\] WARM (?P<status>PASS|FAIL) mode=(?P<mode>\S+) variant=(?P<variant>\S+)( \S+=\S*)*$")
KIT_PREFIX = "[e1-opt exact]"                                                  # the kit runners' and the env / API arm's own lines under mode exact, printed in the model process (kernels_prefix builds it from the mode word)
KERNELS_ACCELERATORS = ("flash_attn", "hub_layernorm", "flex_attention")       # the KERNELS line's accelerator fields, in order (== accel.ACCELERATORS == modes.KERNELS_ALL)
KERNELS_RUNNERS = ("stock", "exact")                                  # the runner word in the model-process prefix: the stock runner | the kit runner under its mode (== ("stock",) + modes.KIT_MODES)
KERNELS_ROUTES = ("stock", "default", "exact")                        # the route words (== accel.ROUTES, asserted there): mode off's two lines, then one per kit mode
_RUNNER, _ROUTE = "|".join(KERNELS_RUNNERS), "|".join(KERNELS_ROUTES)
RE_KERNELS = re.compile(r"^\[e1-opt (?P<runner>" + _RUNNER + r")\] KERNELS route=(?P<route>" + _ROUTE + r") flash_attn=(?P<flash_attn>\S+) "
                        r"hub_layernorm=(?P<hub_layernorm>\S+) flex_attention=(?P<flex_attention>\S+) site=(?P<site>\S+) upstream_says=(?P<upstream_says>\S+)$")

OFF_REASON = "mode off: stock e1 (the upstream CLI in a clean subprocess; no environment set, no component applied)"


def weights_line(rec: dict, pins) -> str:
    """The weights line, once per activation on stderr: the digest worded against the pin, in the kit's own words (``pins.weights_words``)."""
    return f"{PREFIX} {pins.weights_words(rec)}"


def emit(line: str, stream=None) -> None:
    stream = sys.stdout if stream is None else stream
    stream.write(line + "\n")
    stream.flush()


def _kv(k, v) -> str:
    s = "none" if v is None else str(v)
    return f"{k}={s.replace(' ', '_') if k != 'gpu' else s}"


def _fields(rep: dict, extra_keys=()) -> str:
    """The trailing `k=v` fields after package=: fixed order, spaces inside values replaced (gpu is the one free-text field).
    `extra_keys` entries are report keys or (report key, printed name) pairs."""
    out = []
    for k in extra_keys:
        key, name = (k if isinstance(k, tuple) else (k, k))
        if key in rep and rep[key] is not None:
            out.append(_kv(name, rep[key]))
    return (" " + " ".join(out)) if out else ""


EXTRA_KEYS = ("card", "mib", "sm", "det", ("pins_summary", "pins"), "route")


def _notes(rep: dict) -> str:
    """The trailing `notes="…; …"` field: what of this host is off the kit's pins — a dependency off its pin, an accelerator
    module that does not import, a hub kernel snapshot off its pin, a GPU outside the tested cards, a stock install without a commit
    record. NAMED here, never a refusal: the levers engage wherever their mechanism applies. Absent when there is nothing to name."""
    notes = [str(n).replace('"', "'") for n in (rep.get("notes") or []) if n]
    return f' notes="{"; ".join(notes)}"' if notes else ""


def active_line(rep: dict) -> str:
    gpu = (rep.get("gpu") or {}).get("name") if isinstance(rep.get("gpu"), dict) else rep.get("gpu")
    return (f"{PREFIX} ACTIVE mode={rep.get('mode')} variant={rep.get('variant')} kit={rep.get('kit')} kit_mode={rep.get('kit_mode')} "
            f"gpu={gpu} stack={rep.get('stack_key')} package={rep.get('package_version')}" + _fields(rep, EXTRA_KEYS) + _notes(rep))


def off_line(variant, rep: dict = None) -> str:
    return f"{PREFIX} NOT ACTIVE: {OFF_REASON} (mode=off variant={variant})" + _notes(rep or {})


def stripped_line(names) -> str:
    """The variables the wrapper stripped from the item processes' environment (stock_score.clean_env): printed once per run when any."""
    return f"{PREFIX} stripped from the item environment: {','.join(names)}"


def not_active_line(reason: str) -> str:
    return f"{PREFIX} NOT ACTIVE: {reason}"


def dry_run_line(rep: dict) -> str:
    gpu = (rep.get("gpu") or {}).get("name") if isinstance(rep.get("gpu"), dict) else rep.get("gpu")
    wr = rep.get("would_refuse") or "none"
    return (f"{PREFIX} DRY-RUN mode={rep.get('mode')} variant={rep.get('variant')} kit={rep.get('kit')} kit_mode={rep.get('kit_mode')} "
            f"gpu={gpu} stack={rep.get('stack_key')} package={rep.get('package_version')}" + _fields(rep, EXTRA_KEYS)
            + _notes(rep) + f" would_refuse={wr}")


def activation_line(rep: dict) -> str:
    """The ONE line an activation prints, from its report: DRY-RUN for a dry run, ACTIVE, the off line, or NOT ACTIVE: <reason>."""
    if rep.get("dry_run"):
        return dry_run_line(rep)
    if rep.get("active"):
        return active_line(rep)
    if rep.get("mode") == "off" and rep.get("reason") == OFF_REASON:
        return off_line(rep.get("variant"), rep)
    return not_active_line(rep.get("reason") or "unknown reason")


def env_clean_line(absent) -> str:
    return f"{STOCK_PREFIX} ENV-CLEAN ok: absent={','.join(absent) if absent else 'none'} kit_modules=none kit_dirs=none"


def ready_line(variant, t_s: float) -> str:
    return f"{PREFIX} ready variant={variant} t={t_s:.2f}"


def score_line(item: str, wall_s: float, stock: bool = False) -> str:
    return f"{STOCK_PREFIX if stock else PREFIX} score {item} {wall_s:.2f}s"


def exit_line(mode, variant, *, complete: bool, kernels_fallback=(), rc: int) -> str:
    return f"{PREFIX} EXIT mode={mode} variant={variant} complete={int(bool(complete))} kernels_fallback={','.join(kernels_fallback) or 'none'} rc={int(rc)}"


def warm_line(status: str, mode, variant, **fields) -> str:
    return f"{PREFIX} WARM {status} mode={mode} variant={variant}" + _fields(fields, tuple(fields))


def error_line(msg: str) -> str:
    return f"{PREFIX} ERROR: {msg}"


# ------------------------------------------------------------------------------------------------------------- the KERNELS proof lines


def kernels_prefix(mode: str) -> str:
    """The runner's prefix on the KERNELS line: the stock runner's on mode off, `[e1-opt exact]` on the kit route."""
    return STOCK_PREFIX if mode == "off" else f"[e1-opt {mode}]"


def kernels_line(mode: str, route: str, words: dict, site, upstream_says) -> str:
    """The KERNELS proof line (one per model process): every accelerator field in KERNELS_ACCELERATORS order — `engaged:<impl>@<version>`,
    or the NAMED fallback / absence (`fallback:<what>(<why>)` | `absent(<why>)`) the model then runs on — then the site the model calls
    through and upstream's own availability word. The line itself never refuses; a kit mode's proof refuses by name after it (accel.proof)."""
    acc = " ".join(_kv(a, words.get(a, "absent(not_read)")) for a in KERNELS_ACCELERATORS)
    return f"{kernels_prefix(mode)} KERNELS route={route} {acc} {_kv('site', site)} upstream_says={'none' if upstream_says is None else upstream_says}"


