"""big — the opendde adapter of the ``big`` memory mode over ``opt_core.mem``: the levers that let this engine fold bigger inputs.

The mode composes on ``fast`` (line ``LSTAR2A``, ``modes.BIG_BASE``) without fast's ``alloc_auto``, ``zprep_hoist`` and the fused
token stack (``modes.BIG_DROP``: a big line's allocator is its fixed field; the others cost resident memory) and adds MEMORY LEVERS, each registered in the core's registry (``opt_core.mem.register``) with its
exactness label and canonical strategy id, each recorded through the core's record and
census (``opt_core.mem.AppliedRecord``: one unit per (item, seed) prediction). One mode, one line (``BIG_TP`` at ``--n_gpu P>1``):

  BIG_F   fast − modes.BIG_DROP + [pair_offload*, no_dit_hoist*, sample_chunk*]   the mode's line (``--mode big``): the trunk /
            structural / confidence pair tensors live in pinned host RAM and stream through the GPU (the offload unit,
            ``levers/OFFLOAD``, at its shipped defaults); host RAM sized per input
  (* SIZE-GATED, :func:`plan` / :func:`size_gate_policy`: pair_offload on at >= ``MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`` residue tokens and
     sample_chunk with it on the line that carries both — below that gate the offload unit is not installed and the line IS fast's
     resident lever set on the fixed allocator; no_dit_hoist on at >= ``MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS``; each off below its gate;
     each gate = its variable when set, else the package's default for the running card, :func:`gate_default`: the 40 GB card's value under
     64 GiB of device memory, the 80 GB card's otherwise)

Every big line runs on the expandable-segments allocator (``modes.EXPANDABLE``), written by the kit's ONE allocator writer
(``alloc.apply`` over ``opt_core.mem.torch_alloc``) and cross-checked present at every prediction.

The levers (``opt_core.mem`` registry names; the kit registry rows each turns on are ``modes.BIG_KIT_ROWS``):
  pair_offload   offload · band     the offload unit: ``ODDE_OFFLOAD=all`` + its knobs exported, its shim on the line's path (rows
                                    pair_offload_{struct,trunk,conf} + diffz + bigln_guard + free_templ); settings rows / cc / memfrac / pin;
                                    size-gated (``MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS``): below the gate the unit's switches are not exported and
                                    its directory is off the path — the pair tensors stay resident as on ``fast``
  no_dit_hoist   setting · bitwise  the DITFAST hoist (kit rows dit_hoist + dit_align) left out: ``ODDE_ADDON_LEVERS`` not exported — the 24
                                    resident hoisted [1,16,n_s,n_s] fp32 pair biases of the diffusion transformer are recomputed per block per
                                    step as stock does (device memory for wall time); size-gated (``MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS``)
  sample_chunk   chunk · band       the diffusion samples one chunk at a time: ``configs.infer_setting.sample_diffusion_chunk_size`` set to
                                    ``samples`` (default 1) for each ``OpenDDE.run_sample_diffusion_stage`` call; N_sample <= samples is a named
                                    skip; follows pair_offload's size gate on the line that carries both (``GATE_FOLLOWERS``: it bounds the
                                    sampler's batch for the inputs the unit exists for; below the gate fast's one-batch sampler runs)
``drop_bond_mask`` is the tree's own lever on every house line (``bondmask.py``), not a lever of this adapter.

Contract with the package (P=1 semantics; ``--n_gpu P>1`` composes on these lines in ``tp.py``):
  compose(name, environ) -> Line     called by ``modes.resolve`` for every big line: the static row, or — once :func:`plan` decided a size
                                     gate leaves a lever out — the line without that lever (``modes.big_line`` over the core's ``selection``
                                     at the package's own switches); a caller's ``OPENDDE_BIG_*`` variable is a :class:`BigRefusal` naming it
  plan(res, query_path, environ)     the ONE pre-activation hook (``cli.pred``): the size-gate decisions of the line's gated levers from the
      -> plan                        query's residue-token count, kept in this process for ``compose`` / ``apply``; recorded
                                     (``kit.big.no_dit_hoist_policy`` / ``kit.big.pair_offload_policy``)
  apply(res, jobs) -> facts          called by ``stack._apply`` before the line's exports: the allocator through ``alloc.apply`` (its facts are
                                     the return value, unchanged), then — for a big line — ``opt_core.mem.apply`` (strict: a refusal raises
                                     :class:`BigRefusal`), the census unit armed on the stock runner (``opt_core.autoload.patch_attr_at_import``)
  refresh(rep) -> rep                called by ``stack.refresh``: the record folded into the activation report — the block (``rep["big"]``
                                     = the manifest's ``kit.big``), the census (a lever that did not run on a prediction is a fallback by
                                     name: PARTIAL), the policy, the allocator checks
  report(planned) / kit_stats()      the offload unit's state for the refresh (planned levers it did not install, its runtime
                                     fallback counters, the manifest's ``kit_stats.offload`` block) — every line with the unit on its path
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Callable, Optional

from . import modes

TAG = "opendde-opt"
PREFIX = "OPENDDE"                                   # the record's stem (opt_core.mem Ctx.prefix); the mode reads no OPENDDE_BIG_* variable
WORDS = PREFIX + "_BIG_"                           # a caller's variable under this stem is refused by name (refuse_words)
NO_HOIST_GATE = "MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS"
NO_HOIST_GATE_DEFAULT = "2565"                       # residue tokens: at 2956 WITH the hoist the offload line exhausts an 80 GB card
OFFLOAD_GATE = "MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS"
OFFLOAD_GATE_DEFAULT = "1400"                        # residue tokens: below it fast's resident path holds the input on an 80 GB card (its resident envelope: peak memory
                                                     # under stock's) — the offload unit is not installed; at / above it the pair tensors go host-resident
SIZE_GATES = {"pair_offload": (OFFLOAD_GATE, OFFLOAD_GATE_DEFAULT),               # memory lever -> (its size gate: the variable — a value set in the environment (configs/<gpu>.env, the
              "no_dit_hoist": (NO_HOIST_GATE, NO_HOIST_GATE_DEFAULT)}            # caller) wins; the default of a card of 64 GiB and over when unset — under 64 GiB see gate_default):
                                                                                  # the lever ON when the largest item counts >= the gate, OFF below
SMALL_CARD_MIB = 64 * 1024                           # the card-class boundary on device 0's total memory in MiB (nvidia-smi memory.total, as stack.gpu_info reads it): under it
                                                     # the 40 GB A100 (40960 MiB); the 80 GB cards (H100, A100 80 GB) read about 81000 MiB
SMALL_CARD_GATES = {"pair_offload": "1160",          # the gates' defaults under SMALL_CARD_MIB (gate_default), the 40 GB card's:
                    "no_dit_hoist": "1856"}          # 1160 — fast's resident lever set holds a 1,152-token input on the 40 GB card and runs out of memory at 1,216,
                                                     # so big offloads from 1,160; 1856 — the largest size at which the offload line's allocator peak with the hoist's
                                                     # 24 resident fp32 pair biases (1536·N² bytes) stays inside the 40 GB card (extrapolated from the peak's growth, not measured)
_CARD = {"probed": False, "memory_mib": None}        # device 0's total memory in MiB, probed once per process (card_memory_mib); not part of _ST: the card does not change
GATE_FOLLOWERS = {"pair_offload": ("sample_chunk",)}  # lever -> the levers that follow ITS gate on a line carrying both: sample_chunk bounds the sampler's batch memory for the
                                                     # inputs the offload unit exists for; below the offload gate the resident path runs fast's one-batch sampler
BELOW_GATE = "below_gate"                            # the LEVER-row reason word of a lever its size gate left out: `reason=below_gate:<largest item tokens>/<gate>` (the
                                                     # core's offload vocabulary, opt_core.mem.offload) — a named, non-partial `state=off`
UNIT_MODULE = "opendde.model.opendde"                # one (item, seed) prediction = one call of OpenDDE.run_sample_diffusion_stage
UNIT_SITE = "OpenDDE.run_sample_diffusion_stage"
ALLOCATOR_WRITER = "opendde_opt.alloc.apply (opt_core.mem.torch_alloc): the line's fixed PYTORCH_CUDA_ALLOC_CONF, bound before the first CUDA call"
STRATEGY = {"pair_offload": "F7.pair_offload", "no_dit_hoist": "F7.hoist_off", "sample_chunk": "F7.chunked_eval"}   # canonical ids (opt_core/STRATEGIES.json)

LINES = tuple(n for n, ln in modes.LINES.items() if modes.OFFLOAD in ln.path_order)   # the lines with the offload unit on their path (report / kit_stats read it)

_LOCK = threading.Lock()
_ST_EMPTY = {"record": None, "line": None, "levers": (), "settings": {}, "plan": None, "patch": None,
             "predictions": 0, "allocator_absent": 0, "allocator_conf": None}
_ST = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _ST_EMPTY.items()}


def _reset() -> None:
    """Forget this process's record (the CPU tests activate several lines in one interpreter; a kit process activates once)."""
    with _LOCK:
        patch = _ST.get("patch")
        _ST.clear()
        _ST.update({k: (dict(v) if isinstance(v, dict) else v) for k, v in _ST_EMPTY.items()})
    if patch is not None:                        # an armed hook withdrawn, an installed wrapper restored (the next apply re-installs it)
        patch.disarm()
        patch.restore()


class BigRefusal(modes.OpenModeError):
    """The memory mode refused by name: an unknown / malformed OPENDDE_BIG_* flag, a malformed size gate, a lever's precondition."""


# ----------------------------------------------------------------------------------------------------------- the levers (opt_core.mem)
def _registry():
    """``opt_core.mem`` with this engine's memory levers registered (once per process). The core is imported here, not at the package's
    import: ``exact`` / ``fast`` never load the memory mode's registry."""
    try:
        import opt_core
        from opt_core import mem
    except ImportError as e:                                                     # the core absent (or its mem package): refused by name, never a traceback
        raise BigRefusal(f"core_missing:opt_core.mem — the big lines import opt_core >= 0.4.0 ({e}) — nothing applied") from None
    if not all(hasattr(mem, n) for n in ("register", "selection", "apply", "Ctx", "LEVERS")):
        raise BigRefusal(f"core_missing: opt_core {getattr(opt_core, '__version__', '?')} at {os.path.dirname(opt_core.__file__)} carries no memory-lever registry "
                           f"(opt_core.mem.register / selection / apply): the big lines need opt_core >= 0.4.0 — nothing applied")
    if "sample_chunk" not in mem.LEVERS:
        _register(mem)
    return mem


def _exports(ctx, lever: str) -> dict:
    return ctx.require(lever, "exports")["exports"]


def _register(mem) -> None:
    R = mem.register

    def _unit_present(unit_key: str, lever: str):
        def applies(ctx):
            tree = ctx.require(lever, "tree")["tree"]
            shim = os.path.join(modes.kit_dir(tree, unit_key), modes.SHIM)
            if not os.path.isfile(shim):
                return mem.refuse(lever, f"the {unit_key} unit in the tree", f"{shim} is absent")
            return None
        return applies

    def _no_precondition(ctx):
        return None

    @R("pair_offload", family="offload", exact="band", strategy=STRATEGY["pair_offload"],
       exact_reason="the offload unit's own parity testing (CA-RMSD 0.005 A vs stock DET at 1,038 tokens; contact probs <= 2.2e-3 at TF32, ERRATA_03): tier 2",
       applies=_unit_present(modes.OFFLOAD, "pair_offload"),
       description="the trunk / structural / confidence pair tensors host-resident in pinned RAM, streamed through the GPU in row / column blocks (levers/OFFLOAD at its shipped defaults)",
       preconditions=(f"the {modes.OFFLOAD} unit in the tree", "the line's exports carry the unit's switches"), settings=("rows", "cc", "memfrac", "pin"))
    def pair_offload(ctx):
        ex = _exports(ctx, "pair_offload")
        want = {**modes._OFFLOAD_EXPORTS, "ODDE_OFFLOAD": ex.get("ODDE_OFFLOAD") if ex.get("ODDE_OFFLOAD") in modes.BIG_OFFLOAD_STAGES.values() else modes._OFFLOAD_EXPORTS["ODDE_OFFLOAD"],
                **{k: ex.get(k) for k in ("ODDE_OFFLOAD_ROWS", "ODDE_OFFLOAD_CC", "ODDE_OFFLOAD_MEMFRAC", "ODDE_OFFLOAD_PIN")}}   # the stages a big line declares (modes.BIG_OFFLOAD_STAGES: all | struct)
        wrong = {k: ex.get(k) for k, v in want.items() if ex.get(k) != v or v is None}
        if wrong:
            raise mem.RefusalError(mem.refuse("pair_offload", "the line's exports carry the unit's switches", f"the resolution does not export the unit's switches: {wrong}"))
        if ex.get("ODDE_OFFLOAD_PIN") not in ("0", "1"):
            raise mem.RefusalError(mem.refuse("pair_offload", "the line's exports carry the unit's switches", f"ODDE_OFFLOAD_PIN={ex.get('ODDE_OFFLOAD_PIN')!r}; one of 0, 1 is required (1: pinned host buffers; 0: pageable)"))
        return mem.Applied(lever="pair_offload", settings={k: ex[k] for k in ("ODDE_OFFLOAD", "ODDE_OFFLOAD_ROWS", "ODDE_OFFLOAD_CC", "ODDE_OFFLOAD_MEMFRAC", "ODDE_OFFLOAD_PIN")},
                           sites=("odde_offload.install (the unit's shim, at the runner's construction)",))

    @R("no_dit_hoist", family="setting", exact="bitwise", strategy=STRATEGY["no_dit_hoist"],
       exact_reason="the hoist is the kit's exact class (dit_hoist + dit_align replay the stock op's own results); leaving it out runs the stock op per step",
       applies=_no_precondition,
       description="the DITFAST hoist left out (ODDE_ADDON_LEVERS not exported): the 24 resident [1,16,n_s,n_s] fp32 hoisted pair biases recomputed per block per step as stock does",
       preconditions=("the line's exports leave ODDE_ADDON_LEVERS out",), settings=())
    def no_dit_hoist(ctx):
        ex = _exports(ctx, "no_dit_hoist")
        if "ODDE_ADDON_LEVERS" in ex:
            raise mem.RefusalError(mem.refuse("no_dit_hoist", "the line's exports leave ODDE_ADDON_LEVERS out", f"the resolution exports ODDE_ADDON_LEVERS={ex['ODDE_ADDON_LEVERS']!r}"))
        return mem.Applied(lever="no_dit_hoist", settings={}, sites=("ODDE_ADDON_LEVERS (absent: the DITFAST addon installs no hoist)",))

    @R("sample_chunk", family="chunk", exact="band", strategy=STRATEGY["sample_chunk"],
       exact_reason="the per-chunk noise draws of the stock sampler come in a different order than one 5-sample batch's: tier 2 by construction",
       applies=_no_precondition,
       description=f"the diffusion samples one chunk at a time: configs.infer_setting.sample_diffusion_chunk_size = `samples` for each {UNIT_SITE} call",
       preconditions=(), settings=("samples",))
    def sample_chunk(ctx):
        n = ctx.setting("sample_chunk", "samples", 1, cast=int)
        if n < 1:
            raise mem.RefusalError(mem.refuse("sample_chunk", "samples", f"settings.sample_chunk.samples={n}: an integer >= 1"))
        _ST["settings"]["sample_chunk"] = n
        return mem.Applied(lever="sample_chunk", settings={"samples": n}, sites=(f"{UNIT_MODULE}.{UNIT_SITE}",))


# ---------------------------------------------------------------------------------------------------------------- composition
def refuse_words(environ) -> None:
    """A caller's ``OPENDDE_BIG_*`` variable is refused by name: the mode is one lever set — no switch and no setting is read from the
    environment (the offload unit and the DiT hoist follow their size gates ``MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`` /
    ``MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS``, the documented deployment parameters)."""
    stray = sorted(k for k in (environ or {}) if str(k).startswith(WORDS))
    if stray:
        raise BigRefusal(f"{', '.join(stray)}: the big mode reads no {WORDS}* variable — its lever set is fixed "
                           f"(pair_offload follows the size gate {OFFLOAD_GATE}, no_dit_hoist {NO_HOIST_GATE}); unset it")


def gated_off(name: str) -> dict:
    """{memory lever: its policy} for the levers of big line ``name`` that :func:`plan` left OUT: a size-gated lever whose largest item is
    below its gate, and the levers that follow that gate on this line (``GATE_FOLLOWERS``; a follower carries the policy it follows).
    ``{}`` without a plan or when every gate says on (the static row — also the memory-safe side when no size is known)."""
    pol = _ST["plan"]
    if not pol or name not in modes.BIG_MEM_LEVERS:
        return {}
    on_line = modes.BIG_MEM_LEVERS[name]
    out = {}
    for lever, p in (pol.get("policies") or {}).items():
        if lever in on_line and not p["on"]:
            out[lever] = p
            for f in GATE_FOLLOWERS.get(lever, ()):
                if f in on_line:
                    out.setdefault(f, p)
    return out


def switches(name: str) -> dict:
    """The switches the package sets itself for line ``name``: ``{lever: False}`` for every lever :func:`gated_off` names, in the line's
    order; ``{}`` otherwise."""
    off = gated_off(name)
    return {lever: False for lever in modes.BIG_MEM_LEVERS.get(name, ()) if lever in off}


def _selection(name: str, sw: Optional[dict] = None) -> "object":
    mem = _registry()
    sel = mem.selection(modes.BIG_MEM_LEVERS[name], switches=sw or None, allow_partial=False)
    if sel.refusals:
        raise BigRefusal("; ".join(str(r) for r in sel.refusals))
    return sel


def compose(name: str, environ: Optional[dict] = None) -> modes.Line:
    """The big line ``name``: the static row ``modes.LINES[name]``, or — when :func:`plan` decided a size gate leaves a lever out (the
    offload unit below ``MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS`` with sample_chunk, the hoist's removal below
    ``MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS``) — the row without those levers, resolved by the core's selection at the package's own
    switches (``modes.big_line``). A caller's ``OPENDDE_BIG_*`` variable in ``environ`` is refused by name (:func:`refuse_words`)."""
    if name not in modes.BIG_LINES:
        raise ValueError(f"{name!r} is not a big line ({', '.join(modes.BIG_LINES)})")
    env = os.environ if environ is None else environ
    refuse_words(env)
    _registry()                                                                  # the core's registry present and new enough, or refused by name — on every route, switch or none
    for lever in modes.BIG_MEM_LEVERS[name]:                                   # the line's size gates well-formed, or refused by name — on every route (`check` included), plan or none
        if lever in SIZE_GATES:
            size_gate_policy(lever, None, env)
    sw = switches(name)
    if not sw:
        return modes.LINES[name]
    sel = _selection(name, sw)
    note = (f"{name} composed on {modes.BIG_BASE_MODE} ({modes.BIG_BASE}) at its flags: "
            f"levers={list(sel.levers)} off_by_flag={list(sel.off_by_flag)} on_by_flag={list(sel.on_by_flag)}")
    return modes.big_line(name, tuple(sel.levers), {}, note=note)


# ---------------------------------------------------------------------------------------------------------------- the size gate
def count_tokens(jobs: list) -> dict:
    """Residue tokens per item of an upstream query = the sum over its protein / DNA / RNA chains of len(sequence) x count (the size
    ladder's unit: residues over the asymmetric unit; the model's ~1.93x structural-token expansion is the model's, not the bin's).
    Ligands / ions are not counted: ``ligands_uncounted`` names how many the item carries."""
    out = {}
    for j in jobs or []:
        n, lig = 0, 0
        for s in j.get("sequences") or []:
            for key in ("proteinChain", "dnaSequence", "rnaSequence"):
                c = s.get(key)
                if c:
                    n += len(c.get("sequence") or "") * int(c.get("count", 1))
            if "ligand" in s or "ion" in s:
                lig += 1
        out[j.get("name")] = {"residue_tokens": n, "ligands_uncounted": lig}
    return out


def card_memory_mib() -> Optional[int]:
    """Total memory of device 0 in MiB, read ONCE per process the way the package reads its GPU facts (``stack.gpu_info``: nvidia-smi's
    ``memory.total`` — the reader the activation report runs anyway; it imports nothing, so on the ``pred`` route, where torch is not yet
    imported when the line is resolved and planned, no CUDA context exists before the line's allocator policy binds at activation —
    ``opt_core.mem.torch_alloc.export`` refuses an initialised one); None with no GPU visible (a dry-run box) or when the probe fails."""
    if not _CARD["probed"]:
        _CARD["probed"] = True
        try:
            from . import stack as _stack                                        # late import: stack imports this module
            v = _stack.gpu_info().get("memory_mib")
            _CARD["memory_mib"] = int(v) if v else None
        except Exception:  # noqa: BLE001                                       # an unreadable probe = no GPU fact: the 80 GB card's defaults
            _CARD["memory_mib"] = None
    return _CARD["memory_mib"]


def gate_default(lever: str) -> str:
    """The package's default of ``lever``'s size gate on THIS card, used when the gate's variable is unset: under 64 GiB of device-0
    memory (``SMALL_CARD_MIB``: the 40 GB A100) ``SMALL_CARD_GATES[lever]``; on a card of 64 GiB and over (H100, A100 80 GB), with no
    GPU visible, or when the probe fails, ``SIZE_GATES[lever][1]``."""
    _var, default = SIZE_GATES[lever]
    mib = card_memory_mib()
    if mib is not None and 0 < mib < SMALL_CARD_MIB:
        return SMALL_CARD_GATES[lever]
    return default


def size_gate_policy(lever: str, tokens: Optional[dict], environ: Optional[dict] = None) -> dict:
    """The size gate of a memory lever in ``SIZE_GATES`` (``pair_offload``: ``MODEL_OPT_BIG_OFFLOAD_MIN_TOKENS``; ``no_dit_hoist``:
    ``MODEL_OPT_BIG_NO_DIT_HOIST_MIN_TOKENS``): ``{"gate_var", "gate", "max_tokens", "on", "source", "reason"}``. The lever is ON when the
    largest item counts ``>= gate`` residue tokens and OFF below; an unknown size turns it ON — the memory-safe side (source ``policy``, the
    reason named). The count is the query's token floor (:func:`count_tokens`: polymer residues; ligand / ion atoms add tokens it does not
    see). No caller variable decides it: the gate is the one parameter — its variable when set (configs/<gpu>.env or the caller's
    environment, which wins), else the package's default for the running card (:func:`gate_default`); malformed, it is a
    :class:`BigRefusal` naming it."""
    var, _default = SIZE_GATES[lever]
    env = os.environ if environ is None else environ
    raw = (env.get(var) or gate_default(lever)).strip()                         # the card is probed only when the variable is unset or empty
    if not raw.isdigit():
        raise BigRefusal(f"{var}={raw!r}: a non-negative integer (residue tokens; 0 = the lever on at every size) is required")
    gate = int(raw)
    mx = None if tokens is None else max((t["residue_tokens"] for t in tokens.values()), default=0)
    if mx is None:
        return {"gate_var": var, "gate": gate, "max_tokens": None, "on": True, "source": "policy",
                "reason": "input size unknown before activation: the lever on (the memory-safe side)"}
    on = mx >= gate
    return {"gate_var": var, "gate": gate, "max_tokens": mx, "on": on, "source": "policy",
            "reason": f"largest item {mx} residue tokens {'>=' if on else '<'} gate {gate}: the lever {'on' if on else 'off'}"}


def plan(res: modes.Resolution, query_path: Optional[str], environ: Optional[dict] = None) -> dict:
    """The pre-activation hook (``cli.pred`` calls it, then activates): for a big line, the size-gate decision of each of the line's gated
    levers (``SIZE_GATES`` on ``modes.BIG_MEM_LEVERS[line]``: pair_offload and no_dit_hoist on ``BIG_F``, no_dit_hoist on ``BIG_TP``)
    from the query's residue-token count — kept in this process (``_ST["plan"]``: :func:`compose` / :func:`apply` turn it into the switches
    the core's selection gets) and returned as ``{"line", "tokens", "policies": {lever: policy}}``; ``{}`` for every other line. Recorded
    (``refresh``: ``kit.big.<lever>_policy``). ``environ`` carries only the gates' own variables."""
    if res.line is None or res.line.name not in modes.BIG_LINES:
        return {}
    env = os.environ if environ is None else environ
    tokens = None
    if query_path:
        from . import inputs as _inputs
        tokens = count_tokens(_inputs.load_query(query_path))
    name = res.line.name
    pol = {"line": name, "tokens": tokens,
           "policies": {lever: size_gate_policy(lever, tokens, env) for lever in modes.BIG_MEM_LEVERS[name] if lever in SIZE_GATES}}
    _ST["plan"] = pol
    return pol


def policies() -> dict:
    """{lever: policy} of this process's plan ({} before :func:`plan` or outside a big line)."""
    return dict(((_ST["plan"] or {}).get("policies")) or {})


def gated_off_reason(row: str, rep: Optional[dict] = None) -> Optional[str]:
    """The LEVER-row reason of a registry row whose memory lever this process's plan left out below its size gate:
    ``below_gate:<largest item residue tokens>/<gate>`` (``BELOW_GATE``); None for every other row, and for any row when the activation
    report ``rep`` is not the planned line's (``report.lever_state``)."""
    line = (_ST["plan"] or {}).get("line")
    if not line or (rep is not None and not _is_line(rep, line)):
        return None
    for lever, p in gated_off(line).items():
        if row in modes.BIG_KIT_ROWS.get(lever, ()):
            return f"{BELOW_GATE}:{p['max_tokens']}/{p['gate']}"
    return None


# ---------------------------------------------------------------------------------------------------------------- apply / census
def _make_unit_wrapper(stock):
    """The census unit boundary of every big line: one (item, seed) prediction = one call of the stock runner's
    ``run_sample_diffusion_stage`` (keyword-only; the sampler call is built inside it from ``configs.infer_setting``). The kit units that
    wrap the same method (the offload unit's stage timer) wrap this wrapper or are wrapped by it — each calls through."""
    def run_sample_diffusion_stage(self, **kw):
        return _prediction(self, stock, kw)
    run_sample_diffusion_stage._big_stock = stock
    return run_sample_diffusion_stage


def _mark_sample_chunk(rec, unit, kw, model):
    """The lever's act: ``configs.infer_setting.sample_diffusion_chunk_size`` = ``samples`` for this call (the stock value restored
    after it). Returns the restore closure (None when nothing was set)."""
    n_sample = kw.get("N_sample")
    chunk = _ST["settings"].get("sample_chunk", 1)
    cfg = getattr(getattr(model, "configs", None), "infer_setting", None)
    if n_sample is None or cfg is None or not hasattr(cfg, "sample_diffusion_chunk_size"):
        rec.fallback("sample_chunk", "N_sample or configs.infer_setting.sample_diffusion_chunk_size not reachable at the stage call: the stock chunking ran", unit)
        return None
    if int(n_sample) <= chunk:
        rec.skip("sample_chunk", f"N_sample={n_sample} <= samples={chunk}: one chunk either way", unit)
        return None
    stock_chunk = cfg.sample_diffusion_chunk_size
    cfg.sample_diffusion_chunk_size = chunk
    rec.mark("sample_chunk", unit, f"N_sample={n_sample} in chunks of {chunk} (stock: {stock_chunk})")

    def restore():
        cfg.sample_diffusion_chunk_size = stock_chunk
    return restore


def _mark_pair_offload(rec, unit, kw):
    """An INSTALLED-state read at the unit: the offload unit patches the model classes once per process (``odde_offload._INSTALLED``,
    ``CFG["stages"]``); its per-run events (d2h / h2d GiB, host HWM, diffz images, the runtime fallback counters) are ``kit_stats.offload``."""
    m = sys.modules.get("odde_offload")
    if m is not None and getattr(m, "_INSTALLED", False):
        stages = "+".join(sorted((getattr(m, "CFG", None) or {}).get("stages") or [])) or "?"
        rec.mark("pair_offload", unit, f"stages={stages} (installed state; the unit's own counters are the events)")
    else:
        rec.fallback("pair_offload", "the offload unit is not installed in this process", unit)


def _mark_no_dit_hoist(rec, unit, kw):
    """An INSTALLED-state read at the unit: the DITFAST addon's switch absent from the model process AND no hoist lever active in the
    addon module (``odde_addon._ACTIVE``) — the addon's own counters (``kit_stats.ditfast``) are the events."""
    v = os.environ.get("ODDE_ADDON_LEVERS")
    m = sys.modules.get("odde_addon")
    active = sorted(x for x in (getattr(m, "_ACTIVE", None) or ()) if str(x).startswith("dit_")) if m is not None else []
    if v:
        rec.fallback("no_dit_hoist", f"ODDE_ADDON_LEVERS={v!r} is set in the model process (the hoist is in force)", unit)
    elif active:
        rec.fallback("no_dit_hoist", f"the DITFAST addon reports active levers {active}", unit)
    else:
        rec.mark("no_dit_hoist", unit, "ODDE_ADDON_LEVERS absent, no DITFAST lever active")


_MARKERS = {"pair_offload": _mark_pair_offload, "no_dit_hoist": _mark_no_dit_hoist}


def _prediction(model, stock, kw):
    """One (item, seed) prediction = one census unit: the levers mark themselves (in the line's order), the line's allocator is cross-checked
    present, the stage runs on the stock method."""
    rec = _ST["record"]
    if rec is None:
        return stock(model, **kw)
    with _LOCK:
        _ST["predictions"] += 1
        unit = f"pred-{_ST['predictions']}"
    rec.unit_begin(unit)
    restore = None
    try:
        for lever in _ST["levers"]:
            if lever == "sample_chunk":
                restore = _mark_sample_chunk(rec, unit, kw, model)
            else:
                _MARKERS[lever](rec, unit, kw)
        if modes.EXPANDABLE not in (os.environ.get(modes.ALLOCATOR) or ""):      # the line's allocator, cross-checked at every prediction
            with _LOCK:
                _ST["allocator_absent"] += 1
        return stock(model, **kw)
    finally:
        if restore is not None:
            restore()
        rec.unit_end(unit)


def _apply_mem(res: modes.Resolution):
    """``opt_core.mem.apply`` for a big line: the line's memory levers (already resolved against the flags by ``compose`` inside
    ``modes.resolve``) applied strictly — every lever's precondition checked against the RESOLUTION's exports (activation exports them
    right after this call), the census unit armed on the stock runner. A refusal raises :class:`BigRefusal` naming lever, precondition, reason."""
    mem = _registry()
    name = res.line.name
    refuse_words(os.environ)                                                     # a caller's OPENDDE_BIG_* variable refuses by name before anything is applied
    sw = switches(name)                                                          # the package's own switches: the levers their size gates leave out (plan)
    hooks = {lever: {"tree": res.tree, "exports": dict(res.exports), "line": name} for lever in modes.BIG_KIT_ROWS}
    ctx = mem.Ctx(PREFIX, TAG, framework="torch", hooks=hooks, environ=os.environ, graphs=False,
                  extra={"kit_line": name, "kit_drop": list(modes.BIG_DROP), "n_gpu": 1, "allocator_writer": ALLOCATOR_WRITER})
    try:                                                                         # the line's static lever list: the core's selection applies the switch (recorded: off_by_flag)
        rec = mem.apply(list(modes.BIG_MEM_LEVERS[name]), ctx, base=modes.BIG_BASE_MODE, strict=True, modules=(), switches=sw or None, allow_partial=False)
    except mem.Refused as e:
        raise BigRefusal(e.record.not_active_line(TAG)) from None
    from opt_core import autoload                                                # the core's per-site patch (the registry import above proved the core)
    try:
        patch = autoload.patch_attr_at_import(UNIT_MODULE, UNIT_SITE, _make_unit_wrapper, tag=TAG, name="big_unit")
    except autoload.PatchError as e:
        raise BigRefusal(f"the census unit cannot be armed: {e}") from None
    if patch.state not in ("armed", "installed"):
        raise BigRefusal(f"the census unit cannot be armed: {UNIT_MODULE}.{UNIT_SITE} patch state {patch.state!r} ({patch.error})")
    _ST["patch"] = patch
    with _LOCK:
        _ST.update({"record": rec, "line": name, "levers": tuple(rec.levers), "allocator_conf": res.exports.get(modes.ALLOCATOR)})
    return rec


def apply(res: modes.Resolution, jobs=None) -> Optional[dict]:
    """What the adapter installs before the line's exports: the process's allocator policy through the kit's one writer (``alloc.apply``:
    a big line's fixed expandable-segments policy refused by name when it cannot bind; the S-hook lines' token-aware alloc_auto on the
    ``pred`` route) — its facts are the return value — and, for a big line, the memory levers through ``opt_core.mem`` (:func:`_apply_mem`)."""
    from . import alloc
    facts = alloc.apply(res, jobs=jobs)
    if res.line is not None and res.line.name in modes.BIG_LINES:
        _apply_mem(res)
    return facts


def applied() -> Optional[dict]:
    """The apply-time summary of the record (None outside a big line): levers, exactness, the flags' effect, the ACTIVE line."""
    rec = _ST["record"]
    if rec is None:
        return None
    return {"line": _ST["line"], "levers": list(rec.levers), "exact": rec.exact, "active_line": rec.active_line(TAG),
            "off_by_flag": list(rec.off_by_flag), "on_by_flag": list(rec.on_by_flag), **_policy_block(),
            "allocator": {"writer": ALLOCATOR_WRITER, "conf": _ST["allocator_conf"]}}


def _policy_block() -> dict:
    """The plan's record: ``<lever>_policy`` per gated lever of the line (``kit.big.pair_offload_policy`` / ``no_dit_hoist_policy``), the
    query's per-item token counts (``size_gate_tokens``) and ``gated_off`` — the levers the gates left out, comma-joined in the line's order
    (None = none): the one scalar of the decision on the EXIT tally's ``big={…}`` group; {} without a plan."""
    pol = _ST["plan"]
    if not pol:
        return {}
    return {**{f"{lever}_policy": p for lever, p in (pol.get("policies") or {}).items()}, "size_gate_tokens": pol.get("tokens"),
            "gated_off": ",".join(switches(pol.get("line"))) or None}


def refresh(rep: dict) -> dict:
    """Fold the record into the activation report after a run: the manifest block (``rep["big"]``), the census (one unit per
    prediction; a lever that did not run on a unit is a fallback by name: PARTIAL — `pred --allow-partial` is the one opt-out, at the exit
    code, recorded), the size-gate decision, the allocator checks. A big process in which no prediction ran names every lever."""
    rec = _ST["record"]
    if rec is None or not _same_line(rep):                                             # the record belongs to the line this process activated
        if _ST["plan"] is not None and rec is None:
            rep.setdefault("big", {}).update(_policy_block())
        return rep
    census = rec.census()
    block = rec.manifest_block()
    block.update({"census": census, "predictions": _ST["predictions"], "kit_line": _ST["line"], "mode_line": rec.mode_line(),
                  "exact_per_lever": dict(rec.exact_per_lever), **_policy_block(), "n_gpu": 1,
                  "allocator_check": {"writer": ALLOCATOR_WRITER, "conf": _ST["allocator_conf"], "absent_at_predictions": _ST["allocator_absent"]},
                  "unit_site": _ST["patch"].record() if _ST["patch"] is not None else None})
    rep["big"] = block
    per = census.get("per_lever") or {}
    applied, inert = set(rep.get("levers_applied") or []), set(rep.get("levers_inert") or [])
    for m in ("no_dit_hoist", "sample_chunk"):                                  # this adapter's own registry rows: applied when the lever ran on a
        if m not in rec.levers:                                                 # prediction with no fallback; inert when its site only skipped
            continue                                                            # (the offload / XL units' rows are their own refresh blocks')
        rows, c = set(modes.BIG_KIT_ROWS[m]), per.get(m) or {}
        applied -= rows
        if not _ST["predictions"] or c.get("fallback") or c.get("absent"):
            continue
        if c.get("ran"):
            applied |= rows
        else:
            inert |= rows
    rep["levers_applied"] = sorted(applied)
    if inert:
        rep["levers_inert"] = sorted(inert)
    for lever, p in policies().items():                                          # each size gate's decision, named (the manifest's activation notes)
        rep.setdefault("notes", []).append(f"big: {lever} {('on' if p['on'] else 'off')} — {p['reason']}")
    for lever, p in gated_off(_ST["line"]).items():
        if lever not in policies():                                              # a follower left out with the gate it follows (sample_chunk with pair_offload)
            rep.setdefault("notes", []).append(f"big: {lever} off — follows {', '.join(g for g, fs in GATE_FOLLOWERS.items() if lever in fs)} below its size gate ({p['reason']})")
    fell = set(rep.get("levers_fallback") or [])
    new = set()
    if _ST["allocator_absent"]:
        new.add(f"allocator: {modes.ALLOCATOR} without {modes.EXPANDABLE} at {_ST['allocator_absent']} predictions")
    new |= {f"{lever}: {reason} ({', '.join(units)})" for (lever, reason), units in _partial_by_lever(census).items()}
    new |= {f"{n}: refused" for n in rec.refused_names}
    if _ST["predictions"] == 0 and rec.levers:
        new |= {f"{lever}: no prediction ran in this process (no census unit)" for lever in rec.levers}
    if new:                                                                      # named on the EXIT / PRED lines; `pred --allow-partial` is the one opt-out (the exit code), no variable
        rep["levers_fallback"] = sorted(fell | new)
        named = {ev.split(":", 1)[0].strip() for ev in new}                      # a lever named as a fallback leaves levers_applied (one bucket per lever, ran.partition):
        rows = set().union(*(set(modes.BIG_KIT_ROWS.get(nm, (nm,))) for nm in named))   # the memory lever's registry rows
        rep["levers_applied"] = sorted(set(rep.get("levers_applied") or []) - rows)
        rep["partial"] = True
    return rep


def _is_line(rep: dict, name: Optional[str]) -> bool:
    """The activation report ``rep`` is line ``name``'s: ``line_name`` (a line by name) or the described line ``<NAME>(hook=…)`` (a mode)."""
    line = str(rep.get("line") or "")
    return bool(name) and (rep.get("line_name") == name or line == name or line.startswith(name + "("))


def _same_line(rep: dict) -> bool:
    """The activation report is this record's line's."""
    return _is_line(rep, _ST["line"])


def _partial_by_lever(census: dict) -> dict:
    """(lever, reason) -> the units on which the lever fell back (its named reason) or left no mark (the census's per-unit rows, regrouped)."""
    out: dict = {}
    for unit, row in (census.get("units") or {}).items():
        fb = row.get("fallback") or {}
        for lever in fb:
            out.setdefault((lever, str(fb[lever]) if isinstance(fb, dict) else "fell back"), []).append(unit)
        for lever in row.get("absent") or ():
            out.setdefault((lever, "left no mark at the prediction (its site never ran)"), []).append(unit)
    return out


def lever_evidence(name: str) -> Optional[dict]:
    """The counters of one of this adapter's kit rows for its LEVER line (``report.lever_lines``): the census tallies of the memory
    lever behind the row (units / ran / skipped / fallback / absent), its exactness label and canonical strategy id."""
    rec = _ST["record"]
    if rec is None:
        return None
    mem_lever = next((m for m, rows in modes.BIG_KIT_ROWS.items() if name in rows and m in rec.levers), None)
    if mem_lever is None:
        return None
    c = (rec.census().get("per_lever") or {}).get(mem_lever) or {}
    out = {"units": _ST["predictions"], "ran": c.get("ran", 0), "skipped": c.get("skipped", 0), "fallback": c.get("fallback", 0) + c.get("absent", 0),
           "exact": dict(rec.exact_per_lever).get(mem_lever), "strategy": STRATEGY[mem_lever]}
    if mem_lever == "sample_chunk":
        out["samples"] = _ST["settings"].get("sample_chunk")
    return out


# ------------------------------------------------------------------------------------------------------------ the offload unit's state
def offload_runtime_fallbacks(planned) -> list:
    """The offload unit's RUNTIME fallback events (every path counts itself in STATS["fallbacks"] = {reason: n}), each a named
    row-level event `offload:<reason> x<n>`; a planned diffz that never built its LayerNorm image is one too (STATS["diffz_images"] == 0)."""
    m = sys.modules.get("odde_offload")
    if m is None or not getattr(m, "_INSTALLED", False):
        return []
    st = getattr(m, "STATS", {}) or {}
    out = [f"offload:{reason} x{n}" for reason, n in sorted((st.get("fallbacks") or {}).items()) if n]
    if "diffz" in planned and bool((getattr(m, "CFG", {}) or {}).get("diffz")) and "diffz_images" in st and not int(st["diffz_images"]) and st.get("stages"):
        out.append("offload:diffz never engaged (no LayerNorm image of pair_z built)")      # the counter present and zero after a run
    return out


def offload_uninstalled(planned) -> list:
    """The planned offload levers the unit did not install in this process (its shim swallows install exceptions: this names them)."""
    m = sys.modules.get("odde_offload")
    if m is None or not getattr(m, "_INSTALLED", False):
        return list(planned)
    cfg = getattr(m, "CFG", {}) or {}
    stages = set(cfg.get("stages", ()))
    tr = sys.modules.get("odde_offload_trunk")
    trunk_ok = bool(getattr(tr, "_TRUNK_INSTALLED", False)) if tr is not None else False
    bl = getattr(m, "_BIGLN", {}) or {}
    have = {"pair_offload_struct": "struct" in stages, "pair_offload_trunk": "trunk" in stages and trunk_ok, "pair_offload_conf": "conf" in stages and trunk_ok,
            "diffz": bool(cfg.get("diffz")), "bigln_guard": bool(bl.get("installed")), "free_templ": bool(cfg.get("free_templ")) and "struct" in stages}
    return [x for x in planned if not have.get(x, False)]


def kit_stats() -> dict | None:
    """The manifest's `kit_stats.offload` block, None when the unit is not loaded in this process."""
    m = sys.modules.get("odde_offload")
    if m is None:
        return None
    try:
        st = dict(getattr(m, "STATS", {}) or {})
        bl = dict(getattr(m, "_BIGLN", {}) or {})
        tr = sys.modules.get("odde_offload_trunk")
        recs = list(st.get("stages") or [])
        return {"installed": bool(getattr(m, "_INSTALLED", False)), "stages_cfg": "+".join(sorted(getattr(m, "CFG", {}).get("stages", ()))),
                "pinned": "on" if getattr(m, "CFG", {}).get("pin", True) else "off",              # ODDE_OFFLOAD_PIN: pinned (a refused pinned allocation raises) | pageable host buffers
                "trunk_installed": bool(getattr(tr, "_TRUNK_INSTALLED", False)) if tr is not None else None,
                "h2d_gib": round(st.get("h2d_bytes", 0) / 2 ** 30, 2), "d2h_gib": round(st.get("d2h_bytes", 0) / 2 ** 30, 2),
                "strided_gib": round(st.get("strided_bytes", 0) / 2 ** 30, 2), "n_stage_records": len(recs),
                "peak_reserved_gib": max((r.get("peak_reserved_gib") or 0 for r in recs), default=0),
                "host_hwm_gib": max((r.get("host_hwm_gib") or 0 for r in recs), default=0), "stages": recs,
                "bigln": f"{'on' if bl.get('installed') else 'off'}/limit={bl.get('limit')}/hits={bl.get('hits')}",   # one field: the tally caps its fields
                "fallbacks": dict(st.get("fallbacks") or {}), "diffz_images": int(st.get("diffz_images") or 0)}   # the unit's own runtime fallback counters
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def report(planned) -> dict:
    """The adapter's reading of its unit for the refresh: planned levers not installed, counted runtime fallbacks, the stats block."""
    return {"uninstalled": offload_uninstalled(planned), "runtime_fallbacks": offload_runtime_fallbacks(planned), "kit_stats": kit_stats()}
