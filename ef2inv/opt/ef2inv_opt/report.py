"""Observability for ef2inv_opt: every line the package prints, on stderr, prefixed ``[ef2inv-opt]`` — one fixed grammar per line kind,
so a log reader can parse a run's stderr.

* ``ACTIVE mode=<off|exact|fast|big> route=subprocess tier=<stock|exact|2> esm=3.4.0@d0207ea3 torch=<v> gpu=<card> kit_switch=<EF2_FAST_KIT=...|none>
  cache=<form> compile=<stock|stock:not_kit_added> [ablated=<lever,…>] line=<the mode row>`` — from the mode table and the check facts
  (``mode=`` is the resolved mode; an ``alias=<word>`` token would follow it only for a run invoked by an alias of its mode — modes.ALIASES,
  empty as shipped; ``compile=``: stock's own torch.compile inherited,
  no kit-added compile lever exists — ``stock:not_kit_added`` when ``--no-compile`` / ``MODEL_OPT_LEVERS_OFF=compile`` asked the opt-out,
  which is accepted and switches nothing; ``ablated=`` only under the ablation word ``MODEL_OPT_LEVERS_OFF``: absent on a plain run of a
  mode); ``NOT ACTIVE reason=<...>`` (exit 3) when the mode
  cannot be activated (the pins, the kit bytes, the stock cookbook file, no CUDA device);
* ``NOTE <what> — <consequence>; proceeding`` — a condition read before any pass that changes no route and no exit code, named once (the card
  off the pinned card: hardware.note, logged by cli.hardware_facts; ``--kernel-backend <v>``: what the value reaches under grad,
  settings.kernel_backend_note, logged by cli.cmd_design before the launch);
* ``ENV-CLEAN ok|FAIL: ... arm=<stock|exact|fast>`` — printed by the design script in the arm before torch is imported (envproof.py);
* ``RUN mode=<m> target=<file> chains=<c|none> binder_len=<L> tokens=<T> seed=<s> steps=<n>`` — the case, before the arm starts;
* ``ready t=<s> first_call=<s> params_load=<s>`` and ``[run] target=<name> seed=<s>: tokens <T> steps <n> steady <s> s/step first_calls <s>,<s> total <s> s
  loss[-1]=<x> iptm=<x> design.pdb sha256=<16>`` — printed by the design script in the arm;
* ``LEVER name=<lever> state=<on|off|ablated> served=<n|static> fallback=<none|reason:n,…> [<word>=<value> …]`` — one per lever of the kit's composition, after
  the loop, from the lever's own counters (evidence.collect_after ``levers``): ``served`` = the lever's engagement count (``static`` = an install with no
  per-call notion), ``fallback`` = its named quiet paths taken (non-zero counters only; the by-design ones are the lever's own declared words,
  the others are refused on the EVIDENCE line);
* ``EVIDENCE levers=<applied|stock> fallback=<none|F..> missing=<none|...> refusals=<n> events=<n> [kernels=<agk> trimul=<cueq_tiles|bmm2|fused|stock> kernel_calls=<k:n,…> policy=<ckpt> trunk_graphs=<s/n|none> lm_graphs=<esmc,pppl|none> chunk=<c|None>] [ablated=<lever,…>]`` — the arm's lever state, derived only
  from the kit's own line and counters (evidence.py);
* ``EXIT rc=<n> mode=<m> kit_switch=<...> evidence=<applied|stock|partial> out=<dir>`` — once per ``design``.
"""
from __future__ import annotations

import sys

PREFIX = "[ef2inv-opt]"


def log(line: str, *, stream=None) -> None:
    (stream or sys.stderr).write(f"{PREFIX} {line}\n")
    (stream or sys.stderr).flush()


def _f(x, nd=3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "nan"


def _join(xs) -> str:
    xs = list(xs or [])
    return ",".join(str(x).split(" ")[0] for x in xs) if xs else "none"


def alias_token(alias) -> str:
    """`` alias=<word>`` when the run was invoked by an alias of its mode (modes.ALIASES, empty as shipped), else no token."""
    return f" alias={alias}" if alias else ""


def active_line(mode, facts: dict) -> str:
    sw = f"{facts.get('kit_switch_var', 'EF2_FAST_KIT')}={mode.kit_switch}" if mode.kit_switch else "none"
    ab = f" ablated={','.join(facts['ablated'])}" if facts.get("ablated") else ""      # the ablation word (MODEL_OPT_LEVERS_OFF): named levers switched off — absent on a plain run of a mode
    return (f"ACTIVE mode={mode.name}{alias_token(facts.get('alias'))} route=subprocess tier={mode.tier} esm={facts.get('esm', '?')} torch={facts.get('torch', '?')} "
            f"gpu={facts.get('device') or 'none'} kit_switch={sw} cache={facts.get('cache', 'unset')} compile={facts.get('compile', 'stock')}{ab} line={mode.what}")


def cuda_peaks(torch) -> tuple:
    """(max_memory_allocated, max_memory_reserved) in bytes, counted from the last ``reset_peak_memory_stats``; (None, None) without a CUDA device."""
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not cuda.is_available():
        return None, None
    return int(cuda.max_memory_allocated()), int(cuda.max_memory_reserved())


def peak_lines(item: str, seed, alloc_bytes, reserved_bytes) -> list:
    """The per-item memory lines, every arm (the stock arm included), peaks reset at the item's start and read after its last step:
    ``PEAK item=<the plan item's bare name> alloc_gib=<max allocated / 2^30, %.2f> reserved_gib=<max reserved / 2^30, %.2f>`` — exactly these
    three tokens in this order (a fixed grammar; a log reader takes the max per item over passes) — then ``PEAK-NOTE item=<id> seed=<s>``.
    Without a CUDA device (bytes None): no PEAK line, one ``PEAK-NOTE item=<id> seed=<s> cuda=absent``."""
    if alloc_bytes is None or reserved_bytes is None:
        return [f"PEAK-NOTE item={item} seed={seed} cuda=absent"]
    return [f"PEAK item={item} alloc_gib={alloc_bytes / 2 ** 30:.2f} reserved_gib={reserved_bytes / 2 ** 30:.2f}", f"PEAK-NOTE item={item} seed={seed}"]


def attn_line(words: str, rec: dict, require: bool) -> str:
    """``ATTN <attention.words> flash_attn=<version|none> transformer_engine=<version|none> xformers=<version|none> require_fast_env=<0|1>
    [exempt=<…>]``: every arm, after det / patches on the loaded models, before the loop."""
    x = (rec.get("xformers") or {}).get("version")
    ex = rec.get("require_exempt") or []
    return (f"ATTN {words} flash_attn={rec.get('flash_attn_version') or 'none'} transformer_engine={rec.get('transformer_engine_version') or 'none'} "
            f"xformers={x or 'none'} require_fast_env={int(bool(require))}" + (f" exempt={','.join(e.split(' ')[0] for e in ex)}" if ex else ""))


def not_active_line(reason: str) -> str:
    return f"NOT ACTIVE reason={reason}"


def fast_env_line(mode_name: str, problems) -> str:
    """``<mode>: fast environment NOT present (<the guard's sentences>) — proceeding, levers unchanged``: the ONE info line a mode prints on a box
    whose attention / MLP paths are not all the pinned stack's (attention.guard_ruling); the run proceeds with the mode's own exit code."""
    return f"{mode_name}: fast environment NOT present ({'; '.join(str(p) for p in problems)}) — proceeding, levers unchanged"


def note_line(what: str, consequence: str) -> str:
    """``NOTE <what> — <consequence>; proceeding``: named once before any pass; the verb runs unchanged and its exit code is unchanged."""
    return f"NOTE {what} — {consequence}; proceeding"


def run_line(mode_name: str, target_name: str, target_len, binder_len, tokens, seed: int, steps: int) -> str:
    """``target`` = the cookbook's target_name; ``target_len`` = the explicit sequence's length, or ``built-in`` when the name is one of the cookbook's own targets."""
    return f"RUN mode={mode_name} target={target_name} target_len={target_len} binder_len={binder_len} tokens={tokens} seed={seed} steps={steps}"


def ready_line(ready_s, first_call_s, load_s) -> str:
    return f"ready t={_f(ready_s, 1)} first_call={_f(first_call_s, 1)} params_load={_f(load_s, 1)}"


def run_summary_line(target_name: str, seed, tokens, steps, timing: dict, loss_last, iptm, pdb_sha: str) -> str:
    fc = timing.get("first_calls_s") or []
    return (f"[run] target={target_name} seed={seed}: tokens {tokens} steps {steps} steady {_f(timing.get('step_time_steady_median_s'))} s/step "
            f"first_calls {','.join(_f(x, 1) for x in fc) or 'none'} total {_f(timing.get('design_s'), 1)} s "
            f"loss[-1]={_f(loss_last, 4)} iptm={_f(iptm, 4)} design.pdb sha256={(pdb_sha or '?')[:16]}")


def evidence_line(after: dict, mode_name: str) -> str:
    fb = [r for r in (after.get("fallback") or [])]
    return (f"EVIDENCE levers={'stock' if mode_name == 'off' else ('applied' if after.get('applied') else 'partial')} fallback={_join(fb)} "
            f"missing={_join([r for r in (after.get('refusals') or []) if r.startswith('missing')])} refusals={len(after.get('refusals') or [])} "
            f"events={len(after.get('events') or [])}" + composition_words(after) + (f" ablated={','.join(after['ablated'])}" if after.get("ablated") else ""))


def composition_words(after: dict) -> str:
    """`` kernels=<agk describe> trimul=<variant|stock> kernel_calls=<k:n,…> policy=<ckpt> trunk_graphs=<slots|none> lm_graphs=<…> chunk=<size|none>`` from the first
    inversion model's census (kit arms; trimul = cueq_tiles | bmm2 | fused | stock), else ''."""
    ms = after.get("models") or []
    if not ms:
        return ""
    m = ms[0]; pool = m.get("trunk_graph_pool"); plan = m.get("bwd_ckpt") or {}
    esmc, pppl = after.get("esmc_graph") or m.get("esmc_graph"), after.get("pppl_graph")
    lm = ",".join(w for w, g in (("esmc", esmc), ("pppl", pppl)) if g) or "none"
    ks = (after.get("kernels") or {}).get("kernel_calls") or {}
    return (f" kernels={str(m.get('agk')).replace(' ', ',')} trimul={m.get('trimul_kernel', 'stock')} kernel_calls={','.join(f'{k}:{v}' for k, v in sorted(ks.items())) or 'none'} "
            f"policy={plan.get('policy', 'none')} trunk_graphs={(str(pool.get('slots')) + '/' + str(pool.get('n_slots'))) if pool else 'none'} lm_graphs={lm} chunk={m.get('chunk_size', '?')}")


def _word(v) -> str:
    return str(v).replace(" ", ",")


def lever_line(rec: dict) -> str:
    """``LEVER name=… state=… served=… fallback=… [k=v …]`` from one evidence.lever_record."""
    fb = ",".join(f"{k}:{v}" for k, v in sorted((rec.get("fallback") or {}).items())) or "none"
    served = rec.get("served")
    extra = "".join(f" {k}={_word(v)}" for k, v in (rec.get("extra") or {}).items())
    return f"LEVER name={rec['name']} state={rec['state']} served={'static' if served is None else served} fallback={fb}{extra}"


def exit_line(rc: int, mode, evidence: str, out_dir: str, cache: str = "unset") -> str:
    return f"EXIT rc={rc} mode={mode.name} kit_switch={mode.kit_switch or 'none'} evidence={evidence} cache={cache} out={out_dir}"


def cache_word(form: str, explicit: bool = False) -> str:
    """``cache=`` value of the ACTIVE / EXIT / check lines: shared | per-box (| frozen:<sha>, not shipped), ``+explicit`` when the operator set the switch."""
    return f"{form}{'+explicit' if explicit else ''}"
