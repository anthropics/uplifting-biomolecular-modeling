"""The package's lines on stderr: one activation line per process, one exit tally per process.

Activation line (printed once by ``stack.py`` when a mode or line is resolved, applied or refused):
    [opendde-opt] ACTIVE mode=exact line=S1(hook=ACCEL; ...) levers=... gpu=... opendde=<version> package=<version>
    [opendde-opt] NOT ACTIVE mode=fast reason=...
    [opendde-opt] DRY RUN mode=exact ...                       (check: resolved and gated, nothing applied)
    ``floor=below_gate:<tokens>/<gate>`` after ``levers=`` when the small-input floor (smalln.py) composed levers out of a `pred` call's line;
    those levers' LEVER rows read ``state=off reason=below_gate:<tokens>/<gate>``.

Exit tally (``register_exit_tally()``, atexit): the kits' own counters — the served-levers hook's STATE (installs, errors), the
ARM-T COUNTS, the DITFAST stats, the ACCEL_V2 STATS — read from the kit modules if they were loaded in this process; a process that
never loaded a lever module says so. A partial activation is never silent: ``PARTIAL`` names the levers that fell back.
"""
from __future__ import annotations

import os
import sys

from opt_core import report as _core_report

from . import big as _big

TAG = "opendde-opt"
PREFIX = _core_report.prefix(TAG)


def _tp_fields(n_gpu) -> str:
    """``n_gpu=P sharding=rowpair`` / ``n_gpu=1 sharding=none``: the core's token text (opt_core.mem.ngpu through tp.fields)."""
    from . import tp
    return tp.fields(n_gpu)


def _n_gpu() -> int:
    """The accepted ``--n_gpu`` of this process's activation (1 when nothing was activated)."""
    try:
        from . import stack
        return int((stack.status() or {}).get("n_gpu") or 1)
    except Exception:  # noqa: BLE001
        return 1
MISSING_REASON = "unspecified (defect: this refusal carried no reason; report it)"   # the reason= value of a refusal report without one — greppable, never blank
TALLY_ID_KEYS = ("installed", "armed", "active", "mode", "state", "n_gpu", "rank", "group")   # printed first in every module's EXIT tally block (handed to the core formatter)


def _short(v, n=80):
    s = str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def floor_word(rep: dict) -> str | None:
    """``below_gate:<tokens>/<gate>`` when the small-input floor composed levers out of this call's line (smalln.py), else None — the one word
    the ACTIVE / PRED lines print as ``floor=`` and the composed-out levers' LEVER rows print as ``state=off reason=``."""
    gated = rep.get("levers_gated_off") or {}
    return next((w for w in gated.values() if str(w).startswith(_big.BELOW_GATE)), None)


def ceiling_word(rep: dict) -> str | None:
    """``above_gate:<tokens>/<gate>`` when the chunk lever's ceiling composed chunk_lift out of this call's line (chunklift.py), else None — printed on
    the ACTIVE / PRED lines as ``ceiling=`` and as that LEVER row's ``state=off reason=``."""
    w = (rep.get("levers_gated_off") or {}).get("chunk_lift")
    return w if w and str(w).startswith("above_gate") and not str(w).startswith("above_gate:unknown") else None   # no query read: the LEVER row says it, the line stays as smalln's


def activation_line(rep: dict) -> str:
    mode = rep.get("mode")
    tag = "DRY RUN" if rep.get("dry_run") else ("ACTIVE" if rep.get("active") else "NOT ACTIVE")
    parts = [f"mode={mode}" if mode else f"line={rep.get('line_name')}"]
    if rep.get("line"):
        parts.append(f"line={rep['line']}")
    if rep.get("active") or rep.get("dry_run"):
        parts.append(f"levers={','.join(rep.get('levers_applied') or rep.get('levers_planned') or []) or 'none'}")
        if rep.get("levers_inert"):
            parts.append(f"inert={','.join(rep['levers_inert'])}")
        if rep.get("levers_fallback"):
            parts.append(f"fallback={','.join(rep['levers_fallback'])}")
        if rep.get("partial"):
            parts.append("PARTIAL")
        if rep.get("levers_refused_on_pin"):                                       # a lever the pin refuses for cause: named, never silently absent
            parts.append("refused_on_pin=" + ",".join(f"{k}:{v}" for k, v in sorted(rep["levers_refused_on_pin"].items())))
        fw = floor_word(rep)                                                       # the small-input floor composed levers out of this call's line: floor=below_gate:<tokens>/<gate>
        if fw:
            parts.append(f"floor={fw}")
        cw = ceiling_word(rep)                                                     # the chunk lever's ceiling composed chunk_lift out: ceiling=above_gate:<tokens>/<gate>
        if cw:
            parts.append(f"ceiling={cw}")
        if rep.get("ablated"):                                                      # MODEL_OPT_LEVERS_OFF left these levers out of this call's line, by name
            parts.append("ablated=" + ",".join(rep["ablated"]))
        if rep.get("stock_knobs"):                                                  # upstream's --triatt_kernel / --trimul_kernel stated: run as stated, the kit levers of that site aside by name
            parts.append(f"stock_knobs={rep['stock_knobs']}")
    if rep.get("active") or rep.get("dry_run"):
        parts.append(_tp_fields(rep.get("n_gpu", 1)))                            # n_gpu=P sharding=rowpair|none (opt_core.mem.ngpu's text)
    gpu = rep.get("gpu") or {}
    if isinstance(gpu, dict) and gpu.get("name"):
        parts.append(f"gpu={gpu.get('name')!r}/sm{gpu.get('sm', '?')}")
    if rep.get("opendde_version"):
        parts.append(f"opendde={rep['opendde_version']}")
    if rep.get("package_version"):
        parts.append(f"package={rep['package_version']}")
    if rep.get("route"):
        parts.append(f"route={rep['route']}")
    if rep.get("stack_mismatch"):                                                     # a torch / cuEquivariance stack other than the pin's: named on the line (every lever still engages)
        parts.append(f"stack_mismatch={str(rep['stack_mismatch']).split(':', 1)[-1]}")
    if rep.get("det"):
        parts.append(f"det={rep['det']}")
    if not rep.get("active") and not rep.get("dry_run"):                          # a refusal ALWAYS names its reason (fail-loud): right after the mode / line
        parts.insert(1, f"reason={rep.get('reason') or MISSING_REASON}")                      # field, before the (long) line description — never at the tail, never absent
    return f"{PREFIX} {tag} " + " ".join(parts)


def log_activation(rep: dict) -> None:
    try:
        sys.stderr.write(activation_line(rep) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


ARM_T_KEYS = ("version", "installed", "arm", "u2_trimul", "trimul_fast_calls", "trimul_bf16_calls", "trimul_stock_calls", "trimul_stock_reasons",
              "trimul_admission", "zt_block_calls", "castonce_calls", "att_mode", "att_prov_calls", "att_conf_calls", "att_conf_prov_calls", "triattn", "trimul_prov_calls", "trimul_bind")             # odde_arm_t COUNTS keys carried into kit_stats()["arm_t"] (manifest, EXIT tally, LEVER lines)
ARM_TRIMUL_LEVERS = ("arm_u", "arm_u23")                                # the registry levers whose forward is odde_arm_t's TriMul wrapper (one per arm; its LEVER line carries the census)
SAMPLER_SECTIONS = {"dit_attn_apb": ("apb_state_dit", "calls"), "dit_attn_exact": ("dit_attn_exact", "calls"), "atom_attn_apb": ("apb_state_atom", "calls"),   # SAMPLER lever -> (its census section
                    "dit_fused": ("dit_fused", "calls"), "dit_lowp": ("dit_fused", "lowp_calls"), "atom_fused": ("atom_fused", "calls")}                         #  under kit_stats()['accel_v2'], its call counter)


def trimul_lever(arm_t: dict) -> str:
    """The registry lever the arm's TriMul census belongs to (arm U, the one arm that binds a TriMul kernel): arm_u23 when the U2 cell is on, else arm_u."""
    return "arm_u23" if (arm_t or {}).get("u2_trimul") else "arm_u"


TRIMUL_FALLBACK_REASONS = ("exception",)                                       # odde_arm_t.FALLBACK_TRIMUL_REASONS: a kernel exception is a per-call fallback by name (the run PARTIAL)
TRIMUL_ADMISSION_REASONS = ("planes_exceed_free_memory", "free_memory_probe_unavailable")   # odde_arm_t.ADMISSION_TRIMUL_REASONS: the U2 admission gate's refusals are named
                                                                                  # STEP-ASIDES (stock TriMul by rule for that call, counted; the run complete) — tests lock the two pairs equal


def trimul_fallback_items(arm_t: dict | None) -> list[str]:
    """The levers_fallback item for TriMul calls that fell back on a kernel EXCEPTION, ``<lever>:trimul_stock(exception:<n>)``, or [] when none.
    The admission gate's refusals (TRIMUL_ADMISSION_REASONS) are not fallbacks: see trimul_aside_items."""
    deg = {k: v for k, v in ((arm_t or {}).get("trimul_degraded") or {}).items() if k in TRIMUL_FALLBACK_REASONS}
    if not deg:
        return []
    return [f"{trimul_lever(arm_t)}:trimul_stock({','.join(f'{k}:{v}' for k, v in sorted(deg.items()))})"]


def trimul_aside_items(arm_t: dict | None) -> dict:
    """{<lever>: 'trimul_admission=<reason>:<n>[,<reason>:<n>]'} for the U2 admission gate's named step-asides of this process ({} when none):
    those calls ran the stock TriMul by rule; the arm's LEVER line carries the same words (trimul_evidence) and the run stays complete."""
    adm = {k: v for k, v in ((arm_t or {}).get("trimul_degraded") or {}).items() if k in TRIMUL_ADMISSION_REASONS}
    if not adm:
        return {}
    return {trimul_lever(arm_t): "trimul_admission=" + ",".join(f"{k}:{v}" for k, v in sorted(adm.items()))}


def trimul_evidence(arm_t: dict | None) -> dict:
    """The TriMul census fields of an arm LEVER line: trimul_fast / trimul_bf16 / trimul_stock call counts, trimul_stock_reasons=<reason:n,…>
    (sorted), trimul_admission=<reason>:<refused>,N=<N>,needed_gib=<x>,free_gib=<y|unknown> from the FIRST refusal when the U2 admission gate refused
    a call, and u2_cell=<key> when the U2 cell that served is not the running device's own row (env | <arch>:sm_90_tiles; kit_stats puts it there)."""
    at = arm_t or {}
    if "trimul_stock_reasons" not in at:
        return {}
    ev = {"trimul_fast": at.get("trimul_fast_calls"), "trimul_bf16": at.get("trimul_bf16_calls"), "trimul_stock": at.get("trimul_stock_calls"),
          "trimul_stock_reasons": ",".join(f"{k}:{v}" for k, v in sorted((at.get("trimul_stock_reasons") or {}).items())) or None}
    adm = at.get("trimul_admission") or {}
    fr = adm.get("first_refusal")
    if isinstance(fr, dict):
        free = "unknown" if fr.get("free_gib") is None else fr.get("free_gib")
        ev["trimul_admission"] = f"{fr.get('reason')}:{adm.get('refused')},N={fr.get('N')},needed_gib={fr.get('needed_gib')},free_gib={free}"
    if at.get("u2_cell"):                                                           # the U2 cell served from a row that is not the device's own (odde_arm_t.u2_cell_key): <arch>:sm_90_tiles
        ev["u2_cell"] = at["u2_cell"]
    return ev


def kit_stats() -> dict:
    """The kits' own counters, from the modules loaded in this process (never imports a kit module that is not loaded)."""
    out = {}
    m = sys.modules.get("odde_served_levers")
    if m is not None:
        try:
            st = m.stats() if hasattr(m, "stats") else dict(getattr(m, "STATE", {}))
            out["served_levers"] = {k: v for k, v in st.items() if k in ("version", "active", "wrapped", "installs", "errors", "deterministic_forced")}
            out["served_levers"]["n_installs"] = len(st.get("installs") or [])
        except Exception as e:  # noqa: BLE001
            out["served_levers"] = {"error": repr(e)}
    m = sys.modules.get("odde_arm_t")
    if m is not None:
        try:
            c = dict(getattr(m, "COUNTS", {}))
            out["arm_t"] = {k: c.get(k) for k in ARM_T_KEYS if k in c}
            if hasattr(m, "stats"):                                                  # the arm's stats() adds its sibling units' census (triattn: the provider binding's describe())
                try:
                    st_all = m.stats()
                    for k in ("triattn",):
                        if k in st_all:
                            out["arm_t"][k] = st_all[k]
                except Exception as e:  # noqa: BLE001
                    out["arm_t"]["triattn"] = {"error": repr(e)[:160]}
            keyf = getattr(m, "u2_cell_key", None)                                  # the U2 cell's row (odde_arm_t.U2_CELLS): named ONLY when borrowed — a CUDA device without
            if callable(keyf):                                                      # a row of its own (`<arch>:sm_90_tiles`); a device with its own row (H100, A100) carries no such field
                key, borrowed = keyf()
                if borrowed:
                    out["arm_t"]["u2_cell"] = key
            degraded = getattr(m, "degraded_trimul", None)                          # the arm's TriMul calls that ran stock for a reason outside the lever's domain (a kernel
            if callable(degraded):                                                  # exception, the U2 admission gate): {reason: n}; non-empty = PARTIAL by name (stack.refresh)
                out["arm_t"]["trimul_degraded"] = dict(degraded())
        except Exception as e:  # noqa: BLE001
            out["arm_t"] = {"error": repr(e)}
    m = sys.modules.get("odde_trimul_bind")                                         # the TriMul provider binding (levers/ARMT; ARM U's U2 route and the exact line's FPF adapter both reach it): its own census
    if m is not None and getattr(m, "COUNTS", {}).get("active"):
        try:
            out["trimul_prov"] = dict(m.describe())
        except Exception as e:  # noqa: BLE001
            out["trimul_prov"] = {"error": repr(e)}
    m = sys.modules.get("odde_transition_bind")                                     # the transition provider binding (levers/ARMT; arms U and Z apply it under its word): its own census
    if m is not None and getattr(m, "COUNTS", {}).get("active"):
        try:
            out["transition_prov"] = dict(m.describe())
        except Exception as e:  # noqa: BLE001
            out["transition_prov"] = {"error": repr(e)}
    m = sys.modules.get("fpf_transition_odde")                                      # ARM U's pair-transition adapter (levers/ARMT/third_party): its own per-call census —
    if m is not None:                                                               # calls served by the kernel vs sent to the stock module by a named cell class (declared domain, not partial)
        try:
            st = dict(getattr(m, "STATS", {}))
            out["arm_u_transition"] = {"transition_calls": int(st.get("calls") or 0), "transition_cells_kernel": int(st.get("kernel_calls") or 0),
                                       "transition_cells_stock": int(st.get("fallback") or 0), "transition_stock_reasons": dict(st.get("fallback_reasons") or {})}
        except Exception as e:  # noqa: BLE001
            out["arm_u_transition"] = {"error": repr(e)}
    off = _big.kit_stats()                       # the memory mode's adapter reports its unit
    if off is not None:
        out["offload"] = off
    cp = cuda_peak()
    if cp is not None:
        out["cuda_peak"] = cp
    m = sys.modules.get("odde_xl")
    if m is not None:
        try:
            out["xl"] = {"version": getattr(m, "__version__", None), "installed": sorted(getattr(m, "_INSTALLED", ()) or ()), **dict(getattr(m, "_STATS", {}))}
        except Exception as e:  # noqa: BLE001
            out["xl"] = {"error": repr(e)}
    al = sys.modules.get("opendde_opt.alloc")
    if al is not None and al.kit_stats() is not None:                       # the allocator policy decision of this process and the allocator's read-back
        out["alloc"] = al.kit_stats()
    bm = sys.modules.get("opendde_opt.bondmask")
    if bm is not None and (bm.STATS["installed"] or bm.STATS["missing"]):     # armed alone = nothing happened yet (no featurizer import)
        out["drop_bond_mask"] = {k: v for k, v in bm.STATS.items() if k != "patch"}   # the AttrPatch object is state, not a counter
    cl = sys.modules.get("opendde_opt.chunklift")
    if cl is not None and (cl.STATS["installed"] or cl.STATS["calls"]):       # armed alone = nothing happened yet (no model import)
        out["chunk_lift"] = cl.kit_stats()
    sg = sys.modules.get("opendde_opt.stepgraph")
    if sg is not None and (sg.STATS["installed"] or sg.STATS["sampler_calls"]):        # armed alone = upstream's model module not imported yet
        out["stepgraph"] = sg.kit_stats()
    ss = sys.modules.get("opendde_opt.structoksync")
    if ss is not None and (ss.STATS["installed"] or ss.STATS["calls"]):        # armed alone = the expander module not imported yet
        out["structok_sync"] = ss.kit_stats()
    zp = sys.modules.get("opendde_opt.zprephoist")
    if zp is not None and (zp.STATS["installed"] or zp.STATS["calls"]):                # armed alone = the diffusion module not imported yet
        out["zprep_hoist"] = zp.kit_stats()
    sh = sys.modules.get("opendde_opt.schedhost")
    if sh is not None and (sh.STATS["installed"] or sh.STATS["sched_calls"] or sh.STATS["aug_calls"]):   # armed alone = the generator module not imported yet
        out["sched_host"] = sh.kit_stats()
    pw = sys.modules.get("opendde_opt.precision")
    if pw is not None and (pw.STATS["installed"] or pw.STATS["calls"]):                # armed alone = the runner module not imported yet
        out["sampler_amp"] = pw.kit_stats()
    lc = sys.modules.get("opendde_opt.lncore")                                          # the LayerNorm provider binding: its own census (only in a process where its word is live)
    if lc is not None and lc.COUNTS.get("active"):
        out["ln_core"] = lc.kit_stats()
    ls = sys.modules.get("opendde_opt.lnstream")
    if ls is not None and (ls.STATS["installed"] or ls.STATS["state"] in ("serving", "fallback")):   # armed alone = upstream's layer_norm not imported yet
        out["lnstream"] = ls.kit_stats()
    td = sys.modules.get("opendde_opt.tmpldedup")
    if td is not None and (td.STATS["installed"] or td.STATS["calls"] or td.STATS["bypass"]):   # armed alone = the pairformer module not imported yet
        out["tmpl_dedup"] = td.kit_stats()
    kp = sys.modules.get("opendde_opt.keeppool")
    if kp is not None and kp.STATS["installed"]:
        out["keep_pool"] = kp.kit_stats()
    wo = sys.modules.get("opendde_opt.writer_overlap")
    if wo is not None and (wo.STATS["installed"] or wo.STATS["items"]):        # armed alone = runner.inference not imported yet
        out["writer_overlap"] = wo.kit_stats()
    if wo is not None and (wo.JSTATS["installed"] or wo.JSTATS["files"]):
        out["json_oneshot"] = wo.kit_stats_json()
    ic = sys.modules.get("opendde_opt.prefetch")
    if ic is not None and (ic.PSTATS["installed"] or ic.PSTATS["calls"]):
        out["prefetch"] = ic.kit_stats_prefetch()
    if ic is not None and (ic.STATS["installed"] or ic.STATS["items"]):        # the item-boundary census (not a lever: the manifest's record of the loop's seconds outside the forward)
        out["item_census"] = ic.kit_stats()
    tpm = sys.modules.get("opendde_opt.tp")
    if tpm is not None and (tpm.STATS["installed"] or tpm.STATS["armed"] or tpm.STATS["n_gpu"] > 1):
        out["rowpair_tp"] = tpm.kit_stats()
    bg = sys.modules.get("opendde_opt.big")
    if bg is not None:
        b = bg.applied()
        if b:
            out["big"] = b                                                   # the memory mode's record: levers, exact, flags' effect, size-gate policy, allocator
    m = sys.modules.get("odde_addon")
    if m is not None:
        try:
            st = m.stats() if hasattr(m, "stats") else {}
            out["ditfast"] = {k: v for k, v in (st or {}).items() if not isinstance(v, (dict, list))} or {"active": sorted(getattr(m, "_ACTIVE", {}))}
        except Exception as e:  # noqa: BLE001
            out["ditfast"] = {"error": repr(e)}
    m = sys.modules.get("odde_accel_v2")
    if m is not None:
        try:
            st = m.stats() if hasattr(m, "stats") else dict(getattr(m, "STATS", {}))
            out["accel_v2"] = {k: (v if not isinstance(v, (dict, list)) else {kk: vv for kk, vv in v.items() if not isinstance(vv, (dict, list))} if isinstance(v, dict) else len(v))
                               for k, v in (st or {}).items()}
        except Exception as e:  # noqa: BLE001
            out["accel_v2"] = {"error": repr(e)}
    m = sys.modules.get("fpf_engines")
    if m is not None:
        try:
            out["fpf_engines"] = {"version": getattr(m, "__version__", None), "ops": sorted(op for _e, op in getattr(m, "_ORIG", {})),
                                  "stats": {op: {k: v for k, v in st.items() if isinstance(v, (int, float, str)) or k == "fallbacks"}     # fallbacks: {reason: n}, the kit's own count
                                            for op, st in getattr(m, "STATS", {}).items()}}
        except Exception as e:                                             # a partial record is named, never silent
            out["fpf_engines"] = {"error": repr(e)}
    return out


def cuda_peak() -> dict | None:
    """The allocator's own high-water marks of this process (the allocator's instrument beside an nvidia-smi sampler): max reserved /
    max allocated GiB on the current device; None before CUDA is initialised."""
    torch = sys.modules.get("torch")
    if torch is None or not getattr(getattr(torch, "cuda", None), "is_initialized", lambda: False)():
        return None
    try:
        from .phase import PEAK_PROCESS as _pp                                   # the maxima folded before each per-item reset (phase.py's PEAK line): the PROCESS high-water
        return {"max_reserved_gib": round(max(torch.cuda.max_memory_reserved() / 2 ** 30, _pp["reserved_gib"]), 3),
                "max_allocated_gib": round(max(torch.cuda.max_memory_allocated() / 2 ** 30, _pp["alloc_gib"]), 3), "device": torch.cuda.current_device()}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def tally_fields(stats: dict) -> list[str]:
    """Every kit lever module's scalar counters as ``mod={k=v,...}`` EXIT-line fields: the core's one non-truncating formatter (equality keys
    first, then the rest alphabetically; nothing dropped) — ``opt_core.report.tally_fields``."""
    return _core_report.tally_fields(stats, identity_keys=TALLY_ID_KEYS, max_fields=None)


def exit_tally_line(pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    stats = kit_stats()
    lever = {k: v for k, v in stats.items() if k != "cuda_peak"}             # the allocator high-water is an instrument, not a lever counter
    line = (f"{PREFIX} EXIT pid={pid} {_tp_fields(_n_gpu())} " + " ".join(tally_fields(stats)) if lever
            else f"{PREFIX} EXIT pid={pid} {_tp_fields(_n_gpu())} no lever counters: no kit lever module was loaded in this process" + (" " + " ".join(tally_fields(stats)) if stats else ""))
    try:                                                                            # a partial application is never silent
        from . import stack
        st = stack.status() or {}
        if st.get("partial") or st.get("levers_fallback"):
            line += f" PARTIAL fallbacks={','.join(st.get('levers_fallback') or []) or 'none'}"
    except Exception:  # noqa: BLE001
        pass
    return line


def lever_state(name: str, rep: dict) -> tuple:
    """(state, reason) of one registry lever in this process from the activation report: ``on`` = applied and live; ``skipped`` = in the
    line's lever set but not applied (fell back by name, or declared off) — reason mandatory; ``off`` = not in
    this mode's lever set (a lever a size gate left out of this call's line says why: ``below_gate:<largest item tokens>/<gate>`` — a big
    memory lever below its gate, ``big.gated_off_reason``; a trunk lever below the small-input floor, ``smalln.gated_off``)."""
    planned = set(rep.get("levers") or []) | set(rep.get("levers_planned") or [])
    if name not in planned:
        gated = rep.get("levers_gated_off") or {}                                  # a lever the small-input floor composed out of this call's line says so: below_gate:<tokens>/<gate>
        if name in gated:
            return "off", gated[name]
        from . import registry                                                     # a lever off every line BY THE PIN says why (refused / superseded / retired)
        st, why = registry.PIN_STATUS.get(name, ("tested", ""))
        if name in registry.OFF_LINE:
            return "off", f"{st}:{why.split(':', 1)[0] if st == 'refused' else why}"[:160]
        return "off", _big.gated_off_reason(name, rep)                           # a memory lever below its size gate: off BY THE GATE, named (None for every other row)
    if name in (rep.get("levers_applied") or []) and rep.get("active"):
        return "on", None
    if name in (rep.get("levers_inert") or []):                                    # inert by design on this call (outside its domain, no call reached its site, every sampler
        return "off", (rep.get("levers_inert_reasons") or {}).get(name) or "inert"     # call stepped aside by rule): the run is complete, the row says why
    fell = [f for f in (rep.get("levers_fallback") or []) if f == name or f.startswith(name + ":") or f.startswith(name + " ")]
    if fell:
        return "skipped", "fell back: " + "; ".join(fell)
    if name in (rep.get("levers_declared_off") or []):
        return "skipped", (rep.get("declared_off_reasons") or {}).get(name) or "declared off for this run"
    if not rep.get("active"):
        return "skipped", f"line not active: {rep.get('reason') or 'not applied'}"
    return "skipped", "planned, no install record in this process"


def _device_cc():
    """(major, minor) of CUDA device 0 when torch is imported and CUDA initialised in this process, else None (never initialises CUDA)."""
    t = sys.modules.get("torch")
    try:
        if t is None or not t.cuda.is_available() or not t.cuda.is_initialized():
            return None
        return tuple(t.cuda.get_device_capability(0))
    except Exception:  # noqa: BLE001
        return None


def lever_evidence(name: str, stats: dict) -> dict:
    """The lever's own counters for its LEVER line (a few scalars from ``kit_stats()``; empty when the lever's module kept none)."""
    ev = {}
    if name in ARM_TRIMUL_LEVERS:
        if name in ("arm_u", "arm_u23") and isinstance(stats.get("arm_u_transition"), dict):   # the U line's transition adapter: kernel vs stock cells, by its own counters
            tr = stats["arm_u_transition"]
            ev.update(transition_cells_stock=tr.get("transition_cells_stock"), transition_cells_kernel=tr.get("transition_cells_kernel"),
                      transition_stock_reasons=",".join(f"{k}:{v}" for k, v in sorted((tr.get("transition_stock_reasons") or {}).items())) or None)
        if name == "arm_u" and isinstance(stats.get("arm_t"), dict):                          # the U arm's triangle-attention census: K2B kernel calls vs stock calls by reason, the attention mode
            at = stats["arm_t"]
            ev.update(att_mode=at.get("att_mode"), att_prov_calls=at.get("att_prov_calls"), att_stock_calls=at.get("att_stock_calls"),
                      att_stock_reasons=",".join(f"{k}:{v}" for k, v in sorted((at.get("att_stock_reasons") or {}).items())) or None)
        if isinstance(stats.get("arm_t"), dict) and trimul_lever(stats["arm_t"]) == name:   # the arm's TriMul census on the lever it belongs to: kernel calls, stock calls by reason, the admission gate
            ev.update(trimul_evidence(stats["arm_t"]))
    elif name in SAMPLER_SECTIONS and isinstance((stats.get("accel_v2") or {}).get(SAMPLER_SECTIONS[name][0]), dict):   # the SAMPLER unit's levers: served calls + the named call-form asides
        sec = stats["accel_v2"][SAMPLER_SECTIONS[name][0]]                                     # (asides=<word>:<n>,… — those calls ran the module's stock forward by name)
        ev.update({SAMPLER_SECTIONS[name][1]: sec.get(SAMPLER_SECTIONS[name][1])})
        asd = {k[len("aside_"):]: v for k, v in sec.items() if k.startswith("aside_") and v}
        if asd:
            ev.update(asides=",".join(f"{k}:{v}" for k, v in sorted(asd.items())))
        if sec.get("word") is not None:                                                     # the provider-bound attention sites (odde_apb_bind COUNTS): the word, the rows served per call class,
            ev.update(word=sec.get("word"), rows=",".join(f"{k}:{v}" for k, v in sorted((sec.get("rows") or {}).items())) or None,   # the calls the cell named a stock op for, the refusals by word
                      stock_calls=sec.get("stock_calls"), plan_calls=sec.get("plan_calls") or None,   # (plan_calls: eager calls served from the graph column in stepgraph's window; absent when 0)
                      refusals=",".join(f"{k}:{v}" for k, v in sorted((sec.get("refusals") or {}).items())) or None,
                      notes=";".join(sec.get("notes") or ()) or None)
    elif name == "cueq_tuned_cache":                                                        # the cuEquivariance tuning-cache location the lines export: name the card and what actually serves there
        cc = _device_cc()
        if cc is not None:
            table = os.path.join(os.environ.get("CUEQ_TRITON_CACHE_DIR") or "", "fused_sigmoid_gated_dual_gemm_forward_kernel_wrapper.%d.%d.json" % tuple(cc))
            ev.update(card=f"sm{cc[0]}{cc[1]}",                                              # no table is shipped: a file here is the user's own tuned entries; absent it the
                      tiles=("user_table_sm%d%d" % tuple(cc)) if os.path.isfile(table) else "packaged_defaults(no_table_shipped)")   # library's packaged per-card table serves
    elif name == "alloc_auto" and isinstance(stats.get("alloc"), dict):
        al = stats["alloc"]
        ev.update(alloc=al.get("policy"), decision=al.get("decision"), effective=al.get("effective"))
    elif name == "chunk_lift" and isinstance(stats.get("chunk_lift"), dict):
        clst = stats["chunk_lift"]
        ev.update(calls=clst.get("calls"), items=clst.get("items"), admitted=clst.get("admitted"), clamped_by_memory=clst.get("clamped_by_memory"),
                  stock=clst.get("stock"), above_gate=clst.get("above_gate"), fixed=clst.get("fixed"), no_probe=clst.get("no_probe"))
    elif name == "tmpl_dedup" and isinstance(stats.get("tmpl_dedup"), dict):
        tdst = stats["tmpl_dedup"]
        ev.update(calls=tdst.get("calls"), slots=tdst.get("slots"), distinct=tdst.get("distinct"), saved=tdst.get("saved"), decisions=tdst.get("decisions"), bypass=tdst.get("bypass"))
    elif name == "keep_pool" and isinstance(stats.get("keep_pool"), dict):
        kpst = stats["keep_pool"]
        ev.update(kept=kpst.get("skipped"), passed=kpst.get("passed"), errors=kpst.get("errors"))
    elif name in ("transition_core", "transition_exact") and isinstance(stats.get("transition_prov"), dict):   # the transition provider binding's census: calls decided under the word, provider rows
        tb = stats["transition_prov"]                                                                              # served, the kit singleton's calls by name, the distinct cell decisions, asides, refusals
        ev.update(word=tb.get("word"), calls=tb.get("calls"), rows=",".join(f"{k}:{v}" for k, v in sorted((tb.get("rows") or {}).items())) or "none",
                  singleton=",".join(f"{k}:{v}" for k, v in sorted((tb.get("singleton") or {}).items())) or None,
                  cells=";".join(sorted((tb.get("cells") or {}).keys()))[:240] or None,
                  asides=",".join(f"{k}:{v}" for k, v in sorted((tb.get("asides") or {}).items())) or None,
                  refusals=",".join(f"{k}:{v}" for k, v in sorted((tb.get("refusals") or {}).items())) or None,
                  excluded=",".join(tb.get("excluded") or []) or None)
    elif name in ("trimul_core", "trimul_exact") and isinstance(stats.get("trimul_prov"), dict):          # the TriMul provider binding's census: rows served per cell / row-count bucket, precisions, asides, refusals
        tb = stats["trimul_prov"]
        ev.update(word=tb.get("word"), calls=tb.get("calls"), rows=",".join(f"{k}:{v}" for k, v in sorted((tb.get("rows") or {}).items())) or None,
                  precisions=",".join(f"{k}:{v}" for k, v in sorted((tb.get("precisions") or {}).items())) or None,
                  channels=",".join(f"{k}:{v}" for k, v in sorted((tb.get("channels") or {}).items())) or None,
                  sites=",".join(f"{k}:{v}" for k, v in sorted((tb.get("sites") or {}).items())) or None,
                  stock_calls=tb.get("stock_calls"), asides=",".join(f"{k}:{v}" for k, v in sorted((tb.get("asides") or {}).items())) or None,
                  refusals=",".join(f"{k}:{v}" for k, v in sorted((tb.get("refusals") or {}).items())) or None,
                  buckets=",".join(f"{k}:{v}" for k, v in sorted((tb.get("buckets") or {}).items())) or None,
                  excluded=",".join(tb.get("excluded") or []) or None, errors=",".join(sorted((tb.get("errors") or {}).keys())) or None,
                  resolved_from=tb.get("resolved_from"), memo_pops=int(tb.get("memo_pops") or 0),          # the provider's per-module cast memos released after each call (0 where no cast row serves)
                  cells=";".join(sorted((tb.get("cells") or {}).keys())) or None)
    elif name == "triattn_conf" and isinstance(stats.get("arm_t"), dict):                                  # the confidence-stack sub-lever: that stack's calls, the ones a provider row served
        at = stats["arm_t"]
        ev.update(conf_calls=at.get("att_conf_calls"), conf_prov_calls=at.get("att_conf_prov_calls"), word=((at.get("triattn") or {}).get("word") if isinstance(at.get("triattn"), dict) else None))
        from .registry import conf_chunk_census, conf_offload_calls
        offc, cc = conf_offload_calls(), conf_chunk_census()                                             # which path served the confidence head's triangle attention this pass: the stack's own forward
        if cc is not None and (cc.get("paths") or {}).get("offload_rows"):                               # (module) or the offload unit's host-streamed row blocks, marked per module by odde_conf_chunk:
            core, stock = (cc.get("core") or {}).get("offload_rows") or 0, (cc.get("stock") or {}).get("offload_rows") or 0
            ev.update(conf_path="offload_rows_core" if core else ("offload_rows_stock" if stock else "offload_rows_unmarked"),   # offload_rows_core = provider rows served the blocks (rows=<row-block size>:<n>,
                      rows=",".join(f"{k}:{v}" for k, v in (cc.get("rows") or {}).items()) or None, blocks=cc.get("blocks"))    # blocks=<n> seen); _stock = every block the stock op's by a named rule;
        elif offc or at.get("att_conf_calls"):                                                           # _unmarked = the arm's wrapper never marked them; offload_rows:<passes> = the mark unbound (inert by name)
            ev.update(conf_path=f"offload_rows:{offc}" if offc and not at.get("att_conf_calls") else "module")
        if cc is not None and cc.get("module_chunks"):                                                   # upstream's chunk loop under the stack's own forward: its chunks, marked the same way
            ev.update(chunks=cc.get("module_chunks"))
    elif name in ("triattn_core", "triattn_exact") and isinstance((stats.get("arm_t") or {}).get("triattn"), dict):   # the provider binding's census: rows served, asides, refusals by name
        tb = stats["arm_t"]["triattn"]
        ev.update(word=tb.get("word"), conf=tb.get("conf"), calls=tb.get("calls"), rows=",".join(f"{k}:{v}" for k, v in sorted((tb.get("rows") or {}).items())) or None,
                  stock_calls=tb.get("stock_calls"), asides=",".join(f"{k}:{v}" for k, v in sorted((tb.get("asides") or {}).items())) or None,
                  refusals=",".join(f"{k}:{v}" for k, v in sorted((tb.get("refusals") or {}).items())) or None,
                  excluded=",".join(tb.get("excluded") or []) or None, unavailable=",".join(f"{k}:{v}" for k, v in sorted((tb.get("unavailable") or {}).items())) or None,
                  errors=",".join(sorted((tb.get("errors") or {}).keys())) or None, word_served=(tb.get("word_served") if tb.get("word_served") not in (None, tb.get("word")) else None), cuda_prebuilt=tb.get("cuda_prebuilt"),
                  passed=",".join(f"{k}:{v}" for k, v in sorted((tb.get("passed") or {}).items())) or None,
                  forms=",".join(f"{k}:{v}" for k, v in sorted((tb.get("forms") or {}).items())) or None,
                  cells=";".join(sorted((tb.get("cells") or {}).keys())) or None)
    elif name == "sampler_amp" and isinstance(stats.get("sampler_amp"), dict):
        pw = stats["sampler_amp"]
        ev.update(calls=pw.get("calls"), engaged=pw.get("engaged"), above_gate=pw.get("above_gate"), upstream_amp=pw.get("upstream_amp") or None,
                  no_amp_dtype=pw.get("no_amp_dtype") or None, gate=pw.get("gate"), policy="skip_amp.sample_diffusion=False")
    elif name == "stepgraph" and isinstance(stats.get("stepgraph"), dict):
        sg = stats["stepgraph"]
        ev.update(captures=sg.get("captures"), replays=sg.get("replays"), recaptures=sg.get("recaptures"), sampler_calls=sg.get("sampler_calls"),
                  eager=(sg.get("eager_head") or 0) + (sg.get("eager_other") or 0), held=sg.get("held"), tol_checks=sg.get("tol_checks"),
                  poison_probe=sg.get("poison_probe") or "none", pool_mib=sg.get("pool_mib_max"), det=sg.get("det"),
                  segsum=(sg.get("segsum_calls") if sg.get("det") else None),
                  tol_rel=(f"{sg.get('tol_max_rel', 0.0):.1e}" if sg.get("tol_checks") else None), spread=(f"{sg.get('eager_spread_max', 0.0):.1e}" if sg.get("tol_checks") else None),
                  step_ms=sg.get("step_ms"), eager_ms=sg.get("eager_ms"), replay_host_ms=sg.get("replay_host_ms"), replay_gpu_ms=sg.get("replay_gpu_ms"),
                  aside=(",".join(f"{k}x{v}" for k, v in sorted((sg.get("aside") or {}).items())) or None),
                  refused=(",".join(f"{k}:{v}" for k, v in sorted((sg.get("refused") or {}).items())) or None), disabled=sg.get("disabled"))
    elif name == "structok_sync" and isinstance(stats.get("structok_sync"), dict):
        ssst = stats["structok_sync"]
        ev.update(calls=ssst.get("calls"), pairs=ssst.get("pairs"), empty=ssst.get("empty"), host_reads=ssst.get("host_reads"))
    elif name == "writer_overlap" and isinstance(stats.get("writer_overlap"), dict):
        wost = stats["writer_overlap"]
        ev.update(items=wost.get("items"), submitted=wost.get("submitted"), written=wost.get("written"), failed=wost.get("failed"), sync=wost.get("sync") or None,
                  d2h_s=wost.get("d2h_s"), submit_s=wost.get("submit_s"), write_s=wost.get("write_s"), drain_s=wost.get("drain_s"), host_mib=wost.get("bytes_mib"),
                  writer=wost.get("mode"), max_pending=wost.get("max_pending"),
                  aside=(",".join(f"{k}x{v}" for k, v in sorted((wost.get("aside") or {}).items())) or None))
    elif name == "prefetch" and isinstance(stats.get("prefetch"), dict):
        pfst = stats["prefetch"]
        ev.update(dataloaders=pfst.get("calls"), engaged=pfst.get("engaged"), workers=pfst.get("workers"), relayed=pfst.get("relayed"),
                  aside=(",".join(f"{k}x{v}" for k, v in sorted((pfst.get("aside") or {}).items())) or None))
    elif name == "json_oneshot" and isinstance(stats.get("json_oneshot"), dict):
        jost = stats["json_oneshot"]
        ev.update(files=jost.get("files"), mib=jost.get("mib"), encode_s=jost.get("encode_s"), write_s=jost.get("write_s"), indent_files=jost.get("indent_files"))
    elif name == "zprep_hoist" and isinstance(stats.get("zprep_hoist"), dict):
        zpst = stats["zprep_hoist"]
        ev.update(calls=zpst.get("calls"), hits=zpst.get("hits"), records=zpst.get("records"), passthrough=zpst.get("passthrough"), entries=zpst.get("entries"))
    elif name == "sched_host" and isinstance(stats.get("sched_host"), dict):
        shst = stats["sched_host"]
        ev.update(sched_calls=shst.get("sched_calls"), host_cmp=shst.get("host_cmp"), rot_copies=shst.get("rot_copies"),
                  aside=(",".join(f"{k}x{v}" for k, v in sorted((shst.get("aside") or {}).items())) or None))
    elif name == "ln_core" and isinstance(stats.get("ln_core"), dict):                       # the LayerNorm provider binding's census: word, calls served per row / cell, the extension's calls by reason
        lb = stats["ln_core"]
        ev.update(word=lb.get("word"), calls=lb.get("calls"), rows=",".join(f"{k}:{v}" for k, v in sorted((lb.get("rows") or {}).items())) or None,
                  cells=",".join(f"{k}:{v}" for k, v in sorted((lb.get("cells") or {}).items())) or None, stock_calls=lb.get("stock_calls"),
                  asides=",".join(f"{k}:{v}" for k, v in sorted((lb.get("asides") or {}).items())) or None,
                  refusals=",".join(f"{k}:{v}" for k, v in sorted((lb.get("refusals") or {}).items())) or None,
                  errors=",".join(f"{k}:{v}" for k, v in sorted((lb.get("errors") or {}).items())) or None, widths=lb.get("widths"), stack=lb.get("stack"))
    elif name == "lnstream" and isinstance(stats.get("lnstream"), dict):
        lsst = stats["lnstream"]
        ev.update(build=lsst.get("state"), bound=lsst.get("bound"), calls=(f"{lsst.get('calls')}+" if str(lsst.get("loader") or "").startswith("upstream") else lsst.get("calls")),
                  jit_s=lsst.get("jit_s"), sites=lsst.get("sites_patched"), binary=lsst.get("build"), prebuilt=lsst.get("prebuilt"))
    elif name == "drop_bond_mask" and isinstance(stats.get("drop_bond_mask"), dict):
        bmst = stats["drop_bond_mask"]
        ev.update(dropped=bmst.get("dropped"), missing=bmst.get("missing"), bytes_dropped=bmst.get("bytes_dropped"))
    elif name in ("fpf_trimul_exact", "fpf_transition") and isinstance((stats.get("fpf_engines") or {}).get("stats"), dict):
        ops = stats["fpf_engines"]["stats"]
        ev.update(calls=sum(int(st.get("calls") or 0) for st in ops.values()), kernel=sum(int(st.get("kernel") or 0) for st in ops.values()))
    elif name.startswith("pair_offload") or name in ("diffz", "bigln_guard", "free_templ"):
        off = stats.get("offload")
        if isinstance(off, dict):
            ev.update(installed=off.get("installed"), pinned=off.get("pinned"), h2d_gib=off.get("h2d_gib"))
    elif name == "rowpair_tp":                                                     # the core's n_gpu / sharding tokens lead, then this rank's layout facts and census
        from . import tp
        ev.update(dict(tp.evidence_pairs()))
    elif name == "tp_triatt":                                                      # the kernel word bound and this process's served / declined counts (tp_kernels + the core's ledger)
        from . import tp_kernels as _tpk
        tc = _tpk.triatt_census()
        ev.update(impl=tc.get("impl"), kernel=tc.get("kernel") or "-", calls=tc.get("calls", 0), served=tc.get("served", 0), declined=tc.get("fallback", 0),
                  declined_by=",".join(f"{k}:{v}" for k, v in sorted((tc.get("fallback_by") or {}).items())) or "none")
        if tc.get("door") or "door_off" in (tc.get("door_asides") or {}):          # the provider door ahead of the dispatch (tp_kernels): who served the row batches
            ev.update(door="off" if "door_off" in (tc.get("door_asides") or {}) else tc.get("door"), door_calls=tc.get("door_calls", 0),
                      door_rows=",".join(f"{k}:{v}" for k, v in sorted((tc.get("door_rows") or {}).items())) or "none",
                      door_asides=",".join(f"{k}:{v}" for k, v in sorted((tc.get("door_asides") or {}).items()) if k != "door_off") or "none")
    from . import modes as _modes
    if name in {r for rows in _modes.BIG_KIT_ROWS.values() for r in rows} - set(_modes._OFFLOAD_LEVERS):   # the memory mode's own rows: units / ran / skipped / fallback / exact / strategy
        from . import big as _big
        ev.update({k: v for k, v in (_big.lever_evidence(name) or {}).items() if k != "strategy"})   # the strategy id is report.STRATEGY's (one source)
    return {k: v for k, v in ev.items() if v is not None}


def lever_lines(rep: dict | None = None, stats: dict | None = None) -> list[str]:
    """One activation-evidence line per registry lever for this process (the core's pinned form ``opt_core.report.lever_line``:
    name, state, [reason], impl, origin lead): ``[opendde-opt] LEVER name=<lever> state=<on|off|skipped> [reason=…] mode=… line=… [counters]``."""
    from . import registry
    if rep is None:
        try:
            from . import stack
            rep = stack.refresh()
        except Exception:  # noqa: BLE001
            rep = {}
    rep = rep or {}
    stats = kit_stats() if stats is None else stats
    line = rep.get("line_name") or rep.get("line")
    out = []
    for name in registry.LEVERS:
        state, reason = lever_state(name, rep)
        lv = registry.LEVERS[name]
        origin = "core" if name in CORE_LEVERS else "kit"                    # the implementation's home: the core's primitive under a thin adapter, or kit/carried bytes
        fields = {"mode": rep.get("mode"), "line": line}
        if state == "on":
            ev = lever_evidence(name, stats)
            impl = ev.pop("impl", None) or lv.file
            fields.update(ev)
        else:
            impl = lv.file
        fields = {k: _token(v) for k, v in fields.items()}
        out.append(_core_report.lever_line(TAG, name, state, reason=_token(reason) if reason else None, impl=_token(impl), origin=origin,
                                           strategy=STRATEGY.get(name), **fields))
    return out


def _token(v):
    """A LEVER value is one blank-free token (the core splits lines on blanks): whitespace runs become '_'."""
    if v is None or isinstance(v, (int, float, bool)):
        return v
    return "_".join(str(v).split()) or "-"


CORE_LEVERS = {"alloc_auto": "opt_core.mem.torch_alloc", "fpf_trimul_exact": "opt_core.kernels.fpf_trimul", "writer_overlap": "opt_core.host.outputs",
               "tp_triatt": "opt_core.mem.rowpair.triatt", "triattn_core": "opt_core.kernels.triattn", "triattn_exact": "opt_core.kernels.triattn", "triattn_conf": "opt_core.kernels.triattn",
               "trimul_core": "opt_core.kernels.trimul", "trimul_exact": "opt_core.kernels.trimul",
               "ln_core": "opt_core.kernels.ln",
               "transition_core": "opt_core.kernels.transition", "transition_exact": "opt_core.kernels.transition"}   # levers whose implementation is the core's (origin=core; the transition binding names its kit singleton on the line)
STRATEGY = {"alloc_auto": "F7.expandable_segments", "fpf_trimul_exact": "F2.fpf_trimul_exact", "writer_overlap": "F6.output_overlap", "triattn_core": "F1.triattn_provider", "triattn_exact": "F2.triattn_provider_exact", "triattn_conf": "F1.triattn_provider", "trimul_core": "F2.trimul_provider", "trimul_exact": "F2.trimul_provider_exact", "ln_core": "F1.ln_provider", "transition_core": "F2.transition_provider", "transition_exact": "F2.transition_provider_exact",
            "cueq_tuned_cache": "F2.cueq_tuned_tiles", "arm_z": "F2.trimul_tperm_exact", "tp_triatt": "F1.flash_triatt",
            "pair_offload_struct": "F7.pair_offload", "pair_offload_trunk": "F7.pair_offload", "pair_offload_conf": "F7.pair_offload", "diffz": "F7.pair_offload", "free_templ": "F7.pair_offload", "bigln_guard": "F7.pair_offload", "no_dit_hoist": "F7.hoist_off", "sample_chunk": "F7.chunked_eval", "xl_hyb": "F7.chunked_eval", "rowpair_tp": "F7.tensor_parallel"}                       # registry lever -> canonical strategy id (opt_core/STRATEGIES.json); levers absent here print no strategy key
_LEVER_LINES_DONE = {"printed": False}


def emit_lever_lines(rep: dict | None = None) -> None:
    """Print the per-lever lines once per process (stderr, flushed): at exit for an applied line, at once for `off` or a refusal."""
    if _LEVER_LINES_DONE["printed"]:
        return
    _LEVER_LINES_DONE["printed"] = True
    try:
        if rep is None:
            from . import stack
            rep = stack.refresh()
        for ln in lever_lines(rep):
            sys.stderr.write(ln + "\n")
        if isinstance((rep or {}).get("numerics"), dict):                    # the process numerics signature, once (opt_core.precision.policy.numerics_signature)
            sys.stderr.write(f"{PREFIX} NUMERICS " + _core_report.kv(**{str(k): v for k, v in rep["numerics"].items()}) + "\n")
        if isinstance((rep or {}).get("core_kernels"), dict) and rep["core_kernels"]:
            sys.stderr.write(f"{PREFIX} CORE_KERNELS " + _core_report.kv(**{n: ("routed" if d.get("ok") else "REFUSED") for n, d in rep["core_kernels"].items()}) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        pass


def register_exit_tally() -> None:
    """Print the per-lever lines and the exit tally at interpreter exit (once per process): the core's atexit hook with this package's
    whole EXIT line, and the LEVER lines before it (atexit is LIFO: registered after, printed first)."""
    import atexit
    if _core_report.register_exit_tally(TAG, exit_tally_line):
        atexit.register(emit_lever_lines)
