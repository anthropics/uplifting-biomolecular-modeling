"""Observability for openfold3_opt: the activation line, the ready line and the exit tally.

All on stderr, all prefixed ``[openfold3-opt]``:

* activation — ``ACTIVE mode=<m> line=<spelling> openfold3=<v> gpu=<name(ccX.Y)> levers_requested=<a,b,...> hooks=<h1>h2>`` (+ ``card_support=
  uncertified:<sm>(<levers>)`` after the n_gpu fields on a GPU class a requested lever has no test record on: named, never a gate; or
  ``NOT ACTIVE: <reason>``), formatted from the activation report returned by ``openfold3_opt.enable()``: the line's levers are
  requested at activation and applied by the add-ons at the target modules' import, so the line claims none applied; ``check`` prints
  ``DRY-RUN ...`` with the same fields; every ACTIVE / DRY-RUN / exit line carries `n_gpu=P sharding=rowpair|none` (opt_core.mem.ngpu)
  followed by ``compile=none`` (report.compile_field: the compile-policy word — no mode of this kit compiles, stock does not either)
  and, when ``MODEL_OPT_LEVERS_OFF`` left levers of the line off (modes.apply_levers_off: an ablation), ``levers_off=<a,b,...>``;
* ready — ``ready t=<s>`` once per process when the weights are on the device (the CLI route: printed by ``pred`` from a wrap on the
  checkpoint load);
* forward — ``Model forward time: <s>s item=<query_id> seed=<seed> tokens=<N> route=<kit|stock>`` once per predicted item (query × seed;
  the diffusion samples ride in the batch): the CUDA-synchronised wall time of ``OpenFold3.forward`` alone — trunk, diffusion rollout,
  confidence heads; not featurisation, not the confidence scores' reduction, not the writers — from a wrap installed on every add-on route
  with the instance counter (``stack._wrap_model_class``) and on the stock route by ``stock_pred`` (recorded in its proof
  ``forward_timer``); numerically inert (two device synchronisations per item); ``FORWARD_TIME_RE`` is the line's grammar for readers;
* phase — ``PHASE item=<query_id> lm_s=- trunk_s=<s|NA> sampler_s=<s|NA> conf_s=<s|NA> total_s=<s> seed=<seed> route=<kit|stock>`` right
  after each forward line, from the same wrap on both arms (``PHASE_RE``): the item's forward split at the model's own phase callables —
  trunk = ``OpenFold3.run_trunk`` (every recycle), sampler = the ``SampleDiffusion`` module call (every step × every sample), conf =
  ``OpenFold3._rollout`` minus the sampler call inside it (the confidence / distogram heads); ``lm_s=-``: OpenFold3 has no protein language
  model; ``total_s`` is the forward line's own measurement. A phase whose callable a line does not enter through the model instance prints
  ``NA`` and one ``PHASE-NOTE <phase> not separable: <reason>`` per process — never a guessed split; device-synchronised, numerically inert;
* exit — ``exit mode=<m> line=<l> instances=<n> levers_applied=<..> levers_unavailable=<..> levers_pending=<..> arm_complete=<true|false|
  pending> [PARTIAL: <what>] captures=<..> replays=<..> fallbacks=<..> ...``: the lever verdicts from the add-ons' own records
  (``stack.levers_record``), then the add-ons' counters read from the lever modules this process loaded (``of3_graphs._STATE["stats"]``,
  ``of3t_paircache`` stats when they expose one) plus the model-instance count; ``register_exit_tally()`` is called
  BEFORE the kit hooks are executed so that (atexit being LIFO) the tally runs after the add-ons' own exit lines.
"""
from __future__ import annotations

import os
import sys
import time

TAG = "openfold3-opt"
TEMPL_POLICY = ("ignore", "upstream_0.4.1")            # the engine's one template behaviour at inference and its source (templ_guard.POLICY, the tally's templ_policy=ignore(upstream_0.4.1)): stock 0.4.1 parses a query's templates and does not consume them; `--upstream-fix OF3-001` consumes them (templ_guard.consumed_by)
PREFIX = f"[{TAG}]"                                 # == opt_core.report.prefix(TAG); spelled here so the stock process imports no core module (tests lock the pair)
SHARDING_SCHEME = "rowpair"                                             # the big/tp line row-shards the pair representation (opt_core.mem.ngpu.SCHEMES)
DRY_RUN_OK = "dry run: nothing applied"
_T0 = time.monotonic()


def _join(items) -> str:
    from opt_core.report import join
    return join(items)


def gpu_label(gpu) -> str:
    if isinstance(gpu, dict):
        name = gpu.get("name")
        cc = gpu.get("cc")
        if not name:
            return "none"
        return f"{name}(cc{cc})" if cc else str(name)
    return str(gpu) if gpu else "none"


def ngpu_fields(rep: dict | None) -> str:
    """`n_gpu=P sharding=rowpair` (P > 1: a rank of the big/tp line's row-sharded pair stack) or `n_gpu=1 sharding=none` — the exact token
    text of opt_core.mem.ngpu.active_fields, the one producer every cofold kit's ACTIVE / DRY-RUN / exit line carries."""
    from opt_core.mem import ngpu
    return ngpu.active_fields(int((rep or {}).get("n_gpu") or 1), SHARDING_SCHEME)


def card_support_field(rep: dict | None) -> str:
    """` card_support=<word>:<sm>(<levers>)[;…]` for the requested levers WITHOUT a test record on the running GPU class (opt_core.arch.supports
    over registry.declare_arch: `uncertified` = at or above the floor, no record on this class — the lever engages, the record is owed;
    `undeclared`; `unsupported:<word>`), grouped by verdict word; empty on a class every requested lever is tested on (sm90; the A100-tested
    levers on sm80) and when no GPU is readable. The environment's uncertainty is named on the activation line — it never switches a lever off."""
    rep = rep or {}
    gpu = rep.get("gpu")
    cc = gpu.get("cc") if isinstance(gpu, dict) else None
    levers = list(rep.get("levers_requested") or [])
    if not cc or not levers:
        return ""
    try:
        from opt_core import arch
        from .registry import ARCH_PREFIX, declare_arch
        declare_arch()                                                             # idempotent (the same content twice is a no-op)
        sm = arch.sm_of(cc)
        if sm is None:
            return ""
        groups: dict = {}
        for name in levers:
            word = str(arch.supports(ARCH_PREFIX + name, sm).word)
            if word != "supported":
                groups.setdefault(word if ":" in word or word == "undeclared" else f"{word}:{sm}", []).append(name)
    except Exception as e:  # noqa: BLE001 — the support registry is a statement about test records, never a gate: unreadable -> say so
        return f" card_support=unreadable({type(e).__name__})"
    if not groups:
        return ""
    return " card_support=" + ";".join(f"{w}({','.join(names)})" for w, names in groups.items())


def levers_off_field(rep: dict | None) -> str:
    """` levers_off=<a,b,…>` — the line's levers MODEL_OPT_LEVERS_OFF left off for this call (modes.apply_levers_off: an ablation, named on
    the ACTIVE / DRY-RUN / exit lines; each one's LEVER line reads state=off reason=levers_off); empty when none."""
    off = list((rep or {}).get("levers_off") or [])
    ro = list((rep or {}).get("reach_off") or [])                     # + ` reach_off=<a,b,...>`: the levers the resident line's reach gate set aside (modes.apply_reach_gate; each one's LEVER line reads state=off reason=reach_gate)
    return (f" levers_off={_join(off)}" if off else "") + (f" reach_off={_join(ro)}" if ro else "")


def compile_field(rep: dict | None) -> str:
    """` compile=<word>` right after the n_gpu fields of the ACTIVE / DRY-RUN lines — the compile-policy word every model-opt kit prints
    (modes.COMPILE_STATE: `none` on every mode of this kit — no torch.compile in stock's inference path, none in any lever, so `--no-compile` /
    MODEL_OPT_LEVERS_OFF=compile is a named no-op; a kit with a compile lever prints on | off:user | stepped_aside:<reason> here)."""
    word = (rep or {}).get("compile")
    if not word:
        from .modes import COMPILE_STATE as word
    return f" compile={word}"


def activation_line(rep: dict | None) -> str:
    rep = rep or {}
    mode = rep.get("mode")
    if rep.get("active"):
        return (f"{PREFIX} ACTIVE mode={mode} line={rep.get('line_spelling') or rep.get('line') or '-'} openfold3={rep.get('openfold3_version')} "
                f"gpu={gpu_label(rep.get('gpu'))} levers_requested={_join(rep.get('levers_requested'))} hooks={rep.get('hooks_spelling') or _join(rep.get('hooks'))}"
                + (f" precision={rep['precision']}" if rep.get("precision") else "")
                + (f" {rep['size_gate']} gate={rep.get('gate_reason')} n_tokens={rep.get('n_tokens') if rep.get('n_tokens') is not None else 'unknown'}" if rep.get("size_gate") else "")
                + (f" {rep['conf_gate']} n_tokens={rep.get('n_tokens') if rep.get('n_tokens') is not None else 'unknown'}" if rep.get("conf_gate") else "")
                + (f" {rep['of3o_gate']} n_tokens={rep.get('n_tokens') if rep.get('n_tokens') is not None else 'unknown'}" if rep.get("of3o_gate") else "")
                + (f" {rep['reach_gate']} n_tokens={rep.get('n_tokens') if rep.get('n_tokens') is not None else 'unknown'}" if rep.get("reach_gate") else "")
                + " " + ngpu_fields(rep) + compile_field(rep)
                + card_support_field(rep)
                + levers_off_field(rep)
                + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))
    if rep.get("dry_run") and rep.get("env") is not None:
        tg = rep.get("target_gpu")
        return (f"{PREFIX} DRY-RUN mode={mode} line={rep.get('line_spelling') or '-'} openfold3={rep.get('openfold3_version')} gpu={gpu_label(rep.get('gpu'))}"
                + (f" target_gpu={tg}" if tg else "")
                + f" levers_requested={_join(rep.get('levers_requested'))} hooks={rep.get('hooks_spelling') or _join(rep.get('hooks'))} env_vars={len(rep.get('env') or {})}"
                + (f" precision={rep['precision']}" if rep.get("precision") else "")
                + (f" {rep['size_gate']} gate={rep.get('gate_reason')}" if rep.get("size_gate") else "")
                + (f" {rep['conf_gate']}" if rep.get("conf_gate") else "")
                + (f" {rep['of3o_gate']}" if rep.get("of3o_gate") else "")
                + (f" {rep['reach_gate']}" if rep.get("reach_gate") else "")
                + " " + ngpu_fields(rep) + compile_field(rep)
                + card_support_field(rep)
                + levers_off_field(rep)
                + (f" refused={rep.get('reason')}" if rep.get("reason") and rep.get("reason") != DRY_RUN_OK else "")
                + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))
    reason = rep.get("reason") or "no reason given by openfold3_opt.enable()"
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" (mode={mode})" if mode else "")


def log_activation(rep: dict | None, stream=None) -> str:
    line = activation_line(rep)
    if not (rep or {}).get("logged"):
        print(line, file=stream or sys.stderr, flush=True)
    return line


MODEL_MODULE = "openfold3.projects.of3_all_atom.model"          # where stock's OpenFold3 model class lives (openfold3/projects/of3_all_atom/model.py:56)
MODEL_CLASS = "OpenFold3"
FORWARD_TIME_WORDS = "Model forward time:"                        # the per-item line's fixed words
FORWARD_TIME_RE = r"Model forward time: (?P<s>[0-9]+\.[0-9]+)s item=(?P<name>\S+) seed=(?P<seed>\d+) tokens=(?P<tokens>-?[0-9]+) route=(?P<route>\w+)"   # its grammar — an external timer may parse it verbatim (groups s / name / seed)
PHASE_WORDS = "PHASE"                                            # the per-item phase line's fixed word (phase_line)
PHASE_RE = (r"PHASE item=(?P<name>\S+) lm_s=(?P<lm>\S+) trunk_s=(?P<trunk>\S+) sampler_s=(?P<sampler>\S+) conf_s=(?P<conf>\S+) "
            r"total_s=(?P<total>[0-9]+\.[0-9]+) seed=(?P<seed>\d+) route=(?P<route>\w+)")   # its grammar; a phase value is seconds, `NA` (not separable on the route) or `-` (the engine has none)
TRUNK_METHOD = "run_trunk"                                       # OpenFold3.run_trunk (model.py:172): the trunk, every recycle — Algorithm 1 lines 1-14
ROLLOUT_METHOD = "_rollout"                                      # OpenFold3._rollout (model.py:318): the diffusion sampler call then the confidence / distogram heads
SAMPLER_ATTR = "sample_diffusion"                                # OpenFold3.sample_diffusion (model.py:105): the SampleDiffusion module (diffusion_module.py:287), every step × every sample


def forward_time_line(seconds: float, item: str, seed, tokens: int, route: str) -> str:
    """`[openfold3-opt] Model forward time: <s>s item=<query_id> seed=<seed> tokens=<N> route=<kit|stock>` — one predicted item's model time."""
    return f"{PREFIX} {FORWARD_TIME_WORDS} {seconds:.2f}s item={item} seed={seed} tokens={tokens} route={route}"


def pass_seed() -> int:
    """The model seed in force in this process: torch's initial seed (upstream seeds torch per predicted item, `seed_everything(seed)` →
    `torch.manual_seed`), 0 when torch is not loaded. Always an int."""
    torch = sys.modules.get("torch")
    try:
        return int(torch.initial_seed()) if torch is not None else 0
    except Exception:  # noqa: BLE001
        return 0


def batch_item(batch) -> tuple:
    """(item, seed, tokens) of a predict batch as upstream builds it: `query_id` (a list of one per item → joined with `_`, spaces replaced;
    `?` when absent), `seed` (a tensor / list / scalar → its first value as an int; when the batch carries none that reads as an integer, the
    seed in force in the process, pass_seed() — ALWAYS digits) and the token count (`token_mask`'s last dimension; -1 when absent)."""
    item, seed, tokens = "?", None, -1
    try:
        q = batch.get("query_id") if hasattr(batch, "get") else None
        if q is not None:
            item = ("_".join(str(x) for x in q) if isinstance(q, (list, tuple)) else str(q)).replace(" ", "_") or "?"
    except Exception:  # noqa: BLE001
        pass
    try:
        sd = batch.get("seed") if hasattr(batch, "get") else None
        if sd is not None:
            if hasattr(sd, "flatten") and hasattr(sd, "shape"):
                sd = sd.flatten()[0]
                sd = sd.item() if hasattr(sd, "item") else sd
            elif isinstance(sd, (list, tuple)):
                sd = sd[0] if sd else "?"
            seed = int(sd) if float(sd).is_integer() else None
    except Exception:  # noqa: BLE001
        seed = None
    if seed is None or seed < 0:
        seed = pass_seed()
    try:
        tm = batch.get("token_mask") if hasattr(batch, "get") else None
        if tm is not None and hasattr(tm, "shape"):
            tokens = int(tm.shape[-1])
    except Exception:  # noqa: BLE001
        pass
    return item, seed, tokens


def _device_sync():
    """torch.cuda.synchronize when torch is loaded and a device is present, else a no-op (CPU tests, the dry paths)."""
    torch = sys.modules.get("torch")
    try:
        if torch is not None and torch.cuda.is_available():
            return torch.cuda.synchronize
    except Exception:  # noqa: BLE001
        pass
    return lambda: None


def phase_line(item: str, seed, route: str, total: float, trunk=None, sampler=None, conf=None, lm=None) -> str:
    """`[openfold3-opt] PHASE item=<query_id> lm_s=<s|-> trunk_s=<s|NA> sampler_s=<s|NA> conf_s=<s|NA> total_s=<s> seed=<seed> route=<kit|stock>` —
    one predicted item's forward split by phase (seconds, 3 decimals); None prints `NA` (not separable on this route), lm None prints `-`."""
    def f(v, absent):
        return absent if v is None else f"{v:.3f}"
    return (f"{PREFIX} {PHASE_WORDS} item={item} lm_s={f(lm, '-')} trunk_s={f(trunk, 'NA')} sampler_s={f(sampler, 'NA')} "
            f"conf_s={f(conf, 'NA')} total_s={total:.3f} seed={seed} route={route}")


_PHASE = {"acc": None, "notes": set()}          # the phase accumulator of the item in flight (one forward at a time per process) and the PHASE-NOTE keys printed


def _phase_note(phase: str, reason: str, route: str) -> None:
    """`PHASE-NOTE <phase> not separable: <reason> route=<route>` once per process per (phase, reason)."""
    key = (phase, reason)
    if key in _PHASE["notes"]:
        return
    _PHASE["notes"].add(key)
    print(f"{PREFIX} PHASE-NOTE {phase} not separable: {reason} route={route}", file=sys.stderr, flush=True)


def _phase_timed(key: str, fn, *a, **k):
    """Run fn synchronised and add its wall time to the in-flight item's accumulator under key; a plain call when no item is in flight."""
    acc = _PHASE["acc"]
    if acc is None:
        return fn(*a, **k)
    sync = acc["sync"]
    sync()
    t0 = time.perf_counter()
    try:
        return fn(*a, **k)
    finally:
        sync()
        acc[key][0] += time.perf_counter() - t0
        acc[key][1] += 1


def _sampler_pre(module, args):
    acc = _PHASE["acc"]
    if acc is not None:
        acc["sync"]()
        acc["sampler_t0"] = time.perf_counter()


def _sampler_post(module, args, output):
    acc = _PHASE["acc"]
    if acc is not None and acc.get("sampler_t0") is not None:
        acc["sync"]()
        acc["sampler"][0] += time.perf_counter() - acc.pop("sampler_t0")
        acc["sampler"][1] += 1


def _phase_attach(model) -> dict:
    """Once per model instance, install the phase timers and return their record {phase: boundary}: `run_trunk` and `_rollout` as INSTANCE
    attributes that time a call to whatever `type(model).<name>` is at call time (the class attributes stay untouched: the lines patch and
    compare them by `is` — stack.levers_record — and a line's replacement installed on the class is what gets timed), and a forward pre-hook +
    forward hook on the `sample_diffusion` module (fires on `Module.__call__`; the graphed step, the bf16 rollout hooks and the pair cache live
    inside it). A name the class lacks, or a sampler without hook registration, installs nothing for that phase (its value prints `NA`)."""
    rec = getattr(model, "_of3_phase_timers", None)
    if rec is not None:
        return rec
    rec = {}
    cls = type(model)
    for key, name in (("trunk", TRUNK_METHOD), ("rollout", ROLLOUT_METHOD)):
        if not callable(getattr(cls, name, None)):
            continue

        def shim(*a, _name=name, _key=key, _model=model, **k):
            return _phase_timed(_key, getattr(type(_model), _name), _model, *a, **k)

        try:
            setattr(model, name, shim)
            rec[key] = f"{cls.__module__}.{cls.__qualname__}.{name}"
        except Exception:  # noqa: BLE001
            pass
    sd = getattr(model, SAMPLER_ATTR, None)
    if sd is not None and hasattr(sd, "register_forward_pre_hook") and hasattr(sd, "register_forward_hook"):
        try:
            sd.register_forward_pre_hook(_sampler_pre)
            sd.register_forward_hook(_sampler_post)
            rec["sampler"] = f"{type(sd).__module__}.{type(sd).__qualname__}.__call__"
        except Exception:  # noqa: BLE001
            pass
    try:
        setattr(model, "_of3_phase_timers", rec)
    except Exception:  # noqa: BLE001
        pass
    return rec


def _phase_values(acc: dict, rec: dict, total: float, route: str) -> dict:
    """{trunk, sampler, conf} seconds or None from one item's accumulator; a None carries its PHASE-NOTE (once per process); the sum is checked
    against the forward total (nested synchronised boundaries: trunk + rollout <= forward)."""
    trunk = acc["trunk"][0] if acc["trunk"][1] else None
    sampler = acc["sampler"][0] if acc["sampler"][1] else None
    rollout = acc["rollout"][0] if acc["rollout"][1] else None
    conf = None
    if trunk is None:
        _phase_note("trunk", f"{TRUNK_METHOD} not entered through the model instance ({'no such method' if 'trunk' not in rec else 'the line calls its trunk as a function'})", route)
    if sampler is None:
        _phase_note("sampler", f"{SAMPLER_ATTR} module call not observed ({'no hookable module' if 'sampler' not in rec else 'the line drives the sampler below Module.__call__'})", route)
    if rollout is None:
        _phase_note("conf", f"{ROLLOUT_METHOD} not entered through the model instance ({'no such method' if 'rollout' not in rec else 'the line calls its rollout as a function'}); conf = {ROLLOUT_METHOD} - sampler undefined", route)
    elif sampler is None:
        _phase_note("conf", f"sampler unobserved inside {ROLLOUT_METHOD}; conf = {ROLLOUT_METHOD} - sampler undefined", route)
    else:
        conf = rollout - sampler
        if conf < 0:
            _phase_note("conf", f"sampler time exceeds {ROLLOUT_METHOD} time (sampler called outside the rollout)", route)
            conf = None
    s = sum(v for v in (trunk, sampler, conf) if v is not None)
    if s > total * 1.001 + 1e-3:
        print(f"{PREFIX} PHASE-NOTE sum>total: trunk+sampler+conf={s:.3f} total={total:.3f} route={route}", file=sys.stderr, flush=True)
    return {"trunk": trunk, "sampler": sampler, "conf": conf}


def wrap_forward_timer(cls, route: str) -> bool:
    """Wrap `cls.forward` once so that every call — one per predicted item — prints the forward-time line on stderr (forward_time_line: the
    synchronised wall time of the model call alone) and, right after it, the phase line (phase_line: the same call split at the model's
    trunk / sampler / rollout callables, _phase_attach; total_s is the forward line's measurement). A forward that raises prints nothing (the
    run's own error names the item). Returns False when the class has no forward or it is already wrapped (idempotent: one line per item
    whatever installs it twice)."""
    fwd = getattr(cls, "forward", None)
    if fwd is None or getattr(fwd, "_of3_forward_timer", False):
        return False

    def forward(self, batch, *a, _orig=fwd, _route=route, **k):
        sync = _device_sync()
        rec = _phase_attach(self)
        acc = {"sync": sync, "trunk": [0.0, 0], "rollout": [0.0, 0], "sampler": [0.0, 0]}
        _PHASE["acc"] = acc
        try:
            sync()
            t0 = time.perf_counter()
            out = _orig(self, batch, *a, **k)
            sync()
            total = time.perf_counter() - t0
        finally:
            _PHASE["acc"] = None
        item, seed, tokens = batch_item(batch)
        try:                                                                 # the template guard's observation (templ_census.observe): featurised slots real vs allocated; observe-only
            from . import templ_census as _templ_census
            _templ_census.observe(item, batch)
        except Exception:                                                    # noqa: BLE001 — the guard then judges this item `unobserved` by name; the forward line is never lost to it
            pass
        print(forward_time_line(total, item, seed, tokens, _route), file=sys.stderr, flush=True)
        print(phase_line(item, seed, _route, total, **_phase_values(acc, rec, total, _route)), file=sys.stderr, flush=True)
        return out

    forward._of3_forward_timer = True
    forward.__wrapped__ = fwd
    cls.forward = forward
    return True


def install_forward_timer(route: str) -> dict:
    """Import stock's model module and wrap its model class's forward with the forward-time line (the stock route's form: `stock_pred`, after
    its environment proof, before the stock call; the kit routes wrap at the module's import instead, `stack._wrap_model_class`). The record
    says what was wrapped or why not; a process whose timer is not installed prints `forward timer NOT installed (<route>): <why>` once —
    its per-item lines are then absent by name, never silently."""
    rec = {"target": f"{MODEL_MODULE}.{MODEL_CLASS}.forward", "words": FORWARD_TIME_WORDS, "route": route}
    try:
        import importlib
        cls = getattr(importlib.import_module(MODEL_MODULE), MODEL_CLASS)
        wrap_forward_timer(cls, route)
        rec["installed"] = bool(getattr(cls.forward, "_of3_forward_timer", False))
    except Exception as e:  # noqa: BLE001
        rec.update(installed=False, error=f"{type(e).__name__}: {e}")
        print(f"{PREFIX} forward timer NOT installed ({route}): {rec['error']}", file=sys.stderr, flush=True)
    return rec


def ready_line(t: float | None = None) -> str:
    return f"{PREFIX} ready t={(time.monotonic() - _T0) if t is None else t:.1f}"


def log_ready(stream=None) -> str:
    line = ready_line()
    print(line, file=stream or sys.stderr, flush=True)
    return line


def _numbers(d: dict, prefix: str) -> dict:
    """The int/float entries of `d` under `prefix`; one level of nested dicts is flattened (`<prefix><key>_<subkey>`)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[f"{prefix}{k}"] = v
        elif isinstance(v, dict):
            out.update({f"{prefix}{k}_{kk}": vv for kk, vv in v.items() if isinstance(vv, (int, float)) and not isinstance(vv, bool)})
    return out


def kit_counters() -> dict:
    """The counters the loaded kit modules expose, read from the names the add-ons use: of3_graphs._STATE["stats"] (captures / replays /
    eager_steps / fallbacks; of3_graphs.py:26), of3t_paircache.STATS (of3t_paircache.py:33), of3t_levers.STATS (of3t_levers.py:25),
    templ_guard.counters() (templ_guard=on|off, templ_policy=ignore(upstream_0.4.1), templ_parsed/ignored/sampled/real/dropped) and templ_census.counters() (templ_declared / templ_untemplated / templ_files / templ_files_resolved: what the call's query declared). Read-only; absent modules contribute nothing. The CUDA allocator's peaks of this process
    (max_memory_reserved / max_memory_allocated, MiB) come first: the peak of a memory row is max(nvidia-smi high-water, this) — a
    2 Hz sampler misses transients, on graph-capture arms by gigabytes."""
    out = {}
    torch = sys.modules.get("torch")
    if torch is not None and getattr(torch, "cuda", None) is not None and torch.cuda.is_available() and torch.cuda.is_initialized():
        try:
            out["cuda_max_reserved_mib"] = int(torch.cuda.max_memory_reserved() // 2**20)
            out["cuda_max_allocated_mib"] = int(torch.cuda.max_memory_allocated() // 2**20)
        except Exception:
            out["cuda_max_reserved_mib"] = "unreadable"
    m = sys.modules.get("of3_graphs")
    st = getattr(m, "_STATE", None)
    if isinstance(st, dict) and isinstance(st.get("stats"), dict):
        out.update(_numbers(st["stats"], ""))
    for mod, prefix, attrs in (("of3t_paircache", "paircache_", ("STATS",)), ("of3t_levers", "of3t_", ("STATS",))):
        mm = sys.modules.get(mod)
        for attr in attrs:
            s = getattr(mm, attr, None)
            if isinstance(s, dict):
                out.update(_numbers(s, prefix))
                break
    om = sys.modules.get("of3_offload")                                                          # the offload port (big): fallbacks / pageable buffers / the confidence path
    if om is not None and callable(getattr(om, "census", None)):
        try:
            oc = om.census()
            out["offload_fallbacks"] = sum((oc.get("fallbacks") or {}).values())
            out["offload_pageable"] = sum((oc.get("pageable") or {}).values())
            out["offload_conf_native"] = int((oc.get("conf_census") or {}).get("native_fallback_elements", 0))
            out["offload_conf"] = (oc.get("conf") or {}).get("path") or "-"
        except Exception:
            out["offload_census"] = "unreadable"
    tg = sys.modules.get("openfold3_opt.templ_guard")                                       # the template featurizer's guard, when installed in this process: what it saw (templ_guard.counters)
    if tg is not None and tg.CENSUS.get("installed"):
        out.update(tg.counters())
    tc = sys.modules.get("openfold3_opt.templ_census")                                      # the call's template census, when one was taken in this process (templ_census.counters)
    if tc is not None:
        out.update(tc.counters())
    word = fallback_word()                                                                   # the per-call fallback census over every loaded source (FALLBACK_SOURCES: the tp launcher / a tp rank's
    if word != "none":                                                                       # named fallbacks, declared templates dropped): the census word, last, so it is the line's `fallbacks=`
        out["fallbacks"] = word                                                              # even on a graphed line (whose eager-fallback counter it then replaces)
    return out


FALLBACK_SOURCES = (                                                   # (loaded module, label on the EXIT line, reader -> {reason: count}): the per-call fallback records read at exit
    ("openfold3_opt.tp", "tp", lambda m: m.fallbacks()),                 # the tp launcher: the ranks' FALLBACK lines summed, plus its own `small_n_unsharded`
    ("openfold3_opt.tp_rowpair.core", "tp", lambda m: m.fallbacks()),    # a tp rank process: its own named per-call fallbacks (tp_rowpair.core.log_fallback)
    ("openfold3_opt.templ_census", "templates_dropped", lambda m: m.fallbacks()),   # the template guard: declared templates that reached the model as an empty stack (templ_census.judge)
)


def fallback_census() -> dict:
    """`{<label>:<reason>: count}` over every loaded module of FALLBACK_SOURCES with a non-zero count."""
    out = {}
    for modname, label, reader in FALLBACK_SOURCES:
        m = sys.modules.get(modname)
        if m is None:
            continue
        try:
            for reason, n in (reader(m) or {}).items():
                if n:
                    out[f"{label}:{reason}"] = int(n)
        except Exception:                                              # noqa: BLE001 — an unreadable record is absent from the word, never a crash at exit
            continue
    return out


def fallback_word() -> str:
    """`none`, else `<label>:<reason>=<n>,…` sorted (fallback_census): the census word an equality or timing row fails closed on."""
    c = fallback_census()
    return "none" if not c else ",".join(f"{k}={v}" for k, v in sorted(c.items()))


def process_role() -> str:
    """`main` for a primary process — the interpreter the user (or a launcher, per rank) started — and `worker` for a helper interpreter the run
    itself forks or spawns: a multiprocessing child (a DataLoader worker: multiprocessing.parent_process() is set in it) or one of
    multiprocessing's own service interpreters (forkserver / resource tracker, started with `-c 'from multiprocessing. …'`). The transcript
    grammar is ONE census line per primary process, so a worker counts silently."""
    try:
        import multiprocessing
        if multiprocessing.parent_process() is not None:
            return "worker"
    except Exception:                                            # noqa: BLE001 — an interpreter without multiprocessing state is a main process
        pass
    orig = " ".join(getattr(sys, "orig_argv", None) or sys.argv or [])
    if "from multiprocessing." in orig:
        return "worker"
    return "main"


def partial_label(rep: dict | None) -> str:
    """`PARTIAL: <unavailable levers>[; <module off the carrier>]` when the record says the arm is not the line as tested, else ""."""
    rep = rep or {}
    if not rep.get("partial"):
        return ""
    what = []
    if rep.get("levers_unavailable"):
        what.append("unavailable " + _join(rep["levers_unavailable"]))
    what += list(rep.get("off_carrier") or [])
    what += list(rep.get("offload_events") or [])                                              # the offload port's named events (stack.offload_partial)
    return "PARTIAL: " + "; ".join(what)


def exit_tally_line(rep: dict | None, instances: int | None = None) -> str:
    """The exit line from the activation report as `status()` returns it at exit (its lever fields are the add-ons' records) and the counters."""
    rep = rep or {}
    c = kit_counters()
    body = " ".join(f"{k}={v}" for k, v in c.items()) or "counters=none"
    aor = rep.get("arm_complete")
    levers = ""
    if rep.get("active"):
        levers = (f" levers_applied={_join(rep.get('levers_applied'))} levers_unavailable={_join(rep.get('levers_unavailable'))}"
                  f" levers_pending={_join(rep.get('levers_pending'))} arm_complete={'pending' if aor is None else str(bool(aor)).lower()}")
        pl = partial_label(rep)
        if pl:
            levers += " " + pl
    if rep.get("active"):
        levers += " " + ngpu_fields(rep) + levers_off_field(rep)                                # `n_gpu=P sharding=rowpair|none` on the exit line too (the caller reads either line)
    line = f"{PREFIX} exit mode={rep.get('mode')} line={rep.get('line') or '-'} instances={instances if instances is not None else '-'}{levers} {body}"
    return "\n".join([line] + lever_lines(rep))                        # the EXIT line's bytes first (the caller holds them), then one LEVER line per lever (S6)


def lever_lines(rep: dict | None) -> list:
    """One activation-evidence line per lever of the kit for this kit-arm process — the tree's shared grammar (opt_core.report.lever_line:
    `[openfold3-opt] LEVER name=<id> state=<on|off|skipped> [reason=<why>] impl=<carried kit> origin=<kit|core> mode=… line=… tier=…`):
    `on` = the add-ons' own records show it applied and live in this process; `skipped` = requested by the line but not applied (reason
    mandatory: `unavailable` — the kit's record says no — or `pending` — no record was written, e.g. no model ran; for an offload lever
    whose check is a class-attribute equality check (stack._OFFLOAD_FN_TARGETS), `unavailable` carries the concrete occupant as
    `unavailable:<module>.<qualname>` — stack._offload_fn_detail, e.g. `unavailable:some_caller.timed_run_trunk` names a wrapper that
    won the class attribute rather than shadowing it at the instance level the way this kit's own phase timer does, report._phase_attach)
    or `off` = not in this
    call's lever set (reason `levers_off` for a lever MODEL_OPT_LEVERS_OFF left off, `graph_gate` for the graph levers the size gate dropped,
    `conf_gate` for the confidence levers their size gate dropped, `of3o_gate` for the offload port's units its item gate set aside, else `not_in_line`). The `cuda_graphs` line carries the graph gate's decision once (`gate=<reason> n_tokens=… cap=…`,
    opt_core.mem.graph_gate) and, when on, the keep-across-items knob and census (`keep=on|off captures=… kept=… gen_drops=…`, modes.ENV_GRAPHS_KEEP); each confidence lever the line carries (modes.CONF_LEVERS) carries its gate's decision (`gate=<served|gated:lt_min|
    n_tok_unknown> n_tokens=… min_tokens=…`, modes.conf_gate). Under the fast mode one more line names the precision parameter. Empty for a
    process where nothing is active (the stock arm, a refused activation)."""
    rep = rep or {}
    if not rep.get("active"):
        return []
    from opt_core.report import lever_line

    from . import modes
    from .registry import LEVERS, STRATEGY
    applied, unavailable = set(rep.get("levers_applied") or ()), set(rep.get("levers_unavailable") or ())
    requested = set(rep.get("levers_requested") or ())
    gate_reason = rep.get("gate_reason")
    gated_off = bool(rep.get("size_gate")) and not str(rep.get("size_gate")).endswith("=capture")
    conf_reason = rep.get("conf_reason")                                # the confidence levers' gate word (modes.conf_gate), None when the line carries none
    line_key = (rep.get("mode"), rep.get("line"))
    conf_line = set(modes.LINES[line_key].levers) & set(modes.CONF_LEVERS) if conf_reason and line_key in modes.LINES else set()
    of3o_reason = rep.get("of3o_reason")                                # the offload port's item-gate word (modes.of3o_gate), None when the line has no port
    of3o_line = set(modes.LINES[line_key].levers) & set(modes.OF3O_UNIT_LEVERS) if of3o_reason and line_key in modes.LINES else set()
    reach_off = set(rep.get("reach_off") or ())                         # the levers the resident line's reach gate set aside (modes.apply_reach_gate): off by the gate, named
    left_off = set(rep.get("levers_off") or ())                          # the line's levers MODEL_OPT_LEVERS_OFF left off (modes.apply_levers_off): off by request, named
    out = []
    for name, lever in LEVERS.items():
        if name in applied:
            state, reason = "on", None
        elif name in requested:
            if name in unavailable:
                from . import stack as _stack                                          # local: stack imports this module, so this stays a call-time import
                cause = _stack._offload_fn_detail(name)                                 # what actually occupies the class attribute -- None for a lever outside that table
                state, reason = "skipped", (f"unavailable:{cause}" if cause else "unavailable")
            else:
                state, reason = "skipped", "pending"
        else:
            state, reason = "off", ("levers_off" if name in left_off else modes.REACH_GATE_REASON if name in reach_off else "graph_gate" if gated_off and name in modes.GRAPHS_LEVERS else
                                    ("of3o_gate" if name in of3o_line and str(of3o_reason).startswith("aside") else ("conf_gate" if name in conf_line else "not_in_line")))
        origin = "core" if name in (rep.get("core_routes") or ()) else "kit"
        fields = dict(impl=lever.kit, origin=origin, mode=rep.get("mode"), line=rep.get("line") or "-", tier=lever.tier)
        if name == "cuda_graphs" and gate_reason:
            fields.update(gate=gate_reason, n_tokens=rep.get("n_tokens") if rep.get("n_tokens") is not None else "unknown",
                          cap=("none" if rep.get("graphs_cap") is None else rep.get("graphs_cap")))
        if name == "cuda_graphs" and state == "on":                        # the keep-across-items knob (modes.ENV_GRAPHS_KEEP) as read, and the graphed step's per-process census
            gm = sys.modules.get("of3_graphs")
            keep_on = gm.keep_enabled() if gm is not None and callable(getattr(gm, "keep_enabled", None)) else (os.environ.get(modes.ENV_GRAPHS_KEEP, "") or "1") not in ("0", "off", "false", "no")
            gst = (getattr(gm, "_STATE", None) or {}).get("stats") or {}
            fields.update(keep="on" if keep_on else "off", captures=int(gst.get("captures", 0)), kept=int(gst.get("kept", 0)), gen_drops=int(gst.get("gen_drops", 0)))
        strategy = STRATEGY.get(name)                                   # the canonical strategy id of the tree's own levers that name one (registry.STRATEGY: the confidence levers)
        if name == "tp_shard_s":                                         # the row-sharded pair stack: the tree's tensor-parallel strategy id + P
            from opt_core.mem import ngpu
            strategy = ngpu.TP_LEVER
            fields.update(ngpu.active_pairs(int(rep.get("n_gpu") or 1), SHARDING_SCHEME))
        if name in ("tp_triatt", "tp_trimul") and name in requested:         # the kernel word and this process's served / fallback counts (tp_rowpair/pairstack + the core's counters)
            m = sys.modules.get("openfold3_opt.tp_rowpair.pairstack")
            if name == "tp_triatt":
                tc = m.triatt_census() if m is not None else {}
                fields.update(kernel=tc.get("kernel") or "-", calls=tc.get("calls", 0))
            else:
                tc = m.trimul_census() if m is not None else {}
                fields.update(kernels=tc.get("kernels") or "-", cells=tc.get("cells") or "-", k1_impl=tc.get("k1_impl") or "-")
            fields.update(served=tc.get("served", 0), fallback=tc.get("fallback", 0), fallback_by=",".join(f"{k}:{v}" for k, v in sorted((tc.get("fallback_by") or {}).items())) or "none")
        if name in conf_line:
            fields.update(gate=conf_reason, n_tokens=rep.get("n_tokens") if rep.get("n_tokens") is not None else "unknown", min_tokens=modes.CONF_MIN_TOKENS)
        if name in reach_off:                                                # the reach gate's decision on the levers it set aside (modes.reach_gate)
            fields.update(gate=rep.get("reach_reason"), n_tokens=rep.get("n_tokens") if rep.get("n_tokens") is not None else "unknown",
                          min_tokens=("none" if rep.get("reach_min_tokens") is None else rep.get("reach_min_tokens")))
        if name in of3o_line:                                                # the port's units carry their item gate's decision (modes.of3o_gate)
            fields.update(gate=of3o_reason, n_tokens=rep.get("n_tokens") if rep.get("n_tokens") is not None else "unknown",
                          min_tokens=("none" if rep.get("of3o_min_tokens") is None else rep.get("of3o_min_tokens")))
        out.append(lever_line(TAG, name, state, reason=reason, strategy=strategy, **fields))
    if rep.get("mode") == "fast":                                       # the fast line's precision parameter (upstream's pl_trainer_args.precision via the runner yaml)
        prec = rep.get("precision") or "fp32"
        out.append(lever_line(TAG, "precision_bf16", "on" if prec == "bf16" else "off", reason=(None if prec == "bf16" else "not_selected"),
                              impl="pl_trainer_args.precision:bf16-mixed", origin="kit", mode="fast", line=rep.get("line") or "-", tier="tolerance"))
    return out


def exit_code(e: SystemExit) -> int:
    """The exit code a SystemExit carries, as the interpreter would exit: None -> 0, an int as it is, anything else (a message) -> 1."""
    return 0 if e.code is None else (e.code if isinstance(e.code, int) else 1)


def register_exit_tally(status_fn, instances_fn) -> None:
    """Register the exit tally once (opt_core.report.register_exit_tally: atexit, once per process per tag; a failing collector prints the
    `EXIT tally failed` line, never silence); `status_fn()` returns the activation report at exit, `instances_fn()` the model-instance count."""
    from opt_core.report import register_exit_tally as _register
    _register(TAG, lambda: exit_tally_line(status_fn(), instances_fn()))
