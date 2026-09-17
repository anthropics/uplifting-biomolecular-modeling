"""Observability for esmfold2_opt: the activation line, the application line and the exit tally (all on stderr, prefixed
``[esmfold2-opt]``).

* activation — ``ACTIVE mode=<m> variant=<v> server_mode=<name[+overrides]> esm=<v> transformers=<v> gpu=<name(smNN)> n_gpu=<P>
  sharding=<rowpair|none> levers=<a,b,...> fallbacks=<x,...> applied=deferred|configured atom_attn=flash_attn|sdpa esmc_rope=… esmc_attn=…
  flash_attn=<version|absent> compile=n/a graph_lru_sampler=<n> graph_budgets=…|graph_capture=off`` (or ``DRY-RUN ... graph_lru_sampler=<n> …`` / ``NOT ACTIVE: <reason>``) — the attention words are attn.py's,
  read from upstream's FLASH_ATTN_AVAILABLE in this process (ACTIVE) and again per model from the forwards bound on its atom-attention modules
  (APPLIED ``atom_attn=… atom_forward=<census>``); a dry run imports nothing of torch and prints the metadata word only,
  formatted from the activation report returned by ``esmfold2_opt.enable()`` / ``status()``; the ``n_gpu=… sharding=…`` token is the
  release tree's (opt_core.mem.ngpu) — 1 / none on every in-process route, P / rowpair on the multi-GPU line (tp.py);
* application — ``APPLIED model#<i> ...`` once per model instance the kit's configure() ran on, with what the kit's own records
  show applied, fallen back, or substituted by its device policy;
* exit — ``EXIT pid=<pid> n_gpu=<P> sharding=<…> …``: the kit modules' own counters of this process (``ef2_opt.stats()``, ``ef2_w4.stats()``, ``ef2_msa.STATS``,
  ``ef2_mk_sampler.STATS``, ``ef2_opt.invariant_report()``), read in memory at interpreter exit; when no lever module was ever
  loaded the tally says so (never silent). ``register_exit_tally()`` is called before the kit modules are imported.
* the exit codes (``EXIT_*``, the one table every verb imports) and the exit rule of a run (``verdict``): a mode is all of its levers on a
  GPU class — a PARTIAL application (a lever of the mode's set the kit's own records show not applied on this device: ``levers_fallback``,
  the report's ``partial`` list) is refused by name, EXIT_NOT_ACTIVE (pred refuses before its first fold when the records already show
  it); the kit's declared guards (the MK hoist off at num_diffusion_samples > 1) are ``gated``, recorded, not partial; a pass whose
  outputs are short of the request is ``incomplete`` (EXIT_FAIL), never conflated with partial. A GPU class or library version with no
  cell row is never partial: the lever engages and its line names the class (cards: ``card_support=``).
"""
from __future__ import annotations

import os
import sys

TAG = "esmfold2-opt"                                    # the kit's tag (opt_core.report.prefix(TAG) == PREFIX)
PREFIX = f"[{TAG}]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE, EXIT_NOT_ACTIVE = 0, 1, 2, 3        # one code per condition: ok · failed or incomplete · usage · not active (a refusal by name, the partial refusal included)
N_GPU = {"P": 1}                                       # this process's --n_gpu (cli sets it; 1 on every in-process route): the ACTIVE / DRY-RUN / EXIT lines carry
                                                       # `n_gpu=P sharding=<rowpair|none>` from the release tree's one producer (opt_core.mem.ngpu via tp.active_fields)


def set_n_gpu(p: int) -> int:
    N_GPU["P"] = int(p)
    return N_GPU["P"]


def n_gpu_fields(p: int | None = None) -> str:
    from .tp import active_fields
    return active_fields(N_GPU["P"] if p is None else int(p))
OPT_KEY_WORDS = ("capture", "replay", "fallback", "oom", "miss", "hit", "generation", "clears", "pops", "reset", "evict", "cache", "verify_n",
                 "budget", "eager",                                     # the graph budget's counters (ef2_opt STATS graph_budget_eager_<site>) are tally fields, never filtered out
                 "launches")                                            # the fused pair-bias kernel's row-blocked launches past its 32-bit offset bound (ef2_opt STATS pair_bias_row_launches)


def _join(items) -> str:
    if not items:
        return "none"
    return ",".join(str(x) for x in items)


def gpu_label(gpu) -> str:
    if isinstance(gpu, dict):
        name = gpu.get("name")
        sm = gpu.get("sm")
        if not name:
            return "none"
        if sm in (None, ""):
            return str(name)
        sm = str(sm)
        return f"{name}({sm})" if sm.startswith("sm") else f"{name}(sm{sm})"
    return str(gpu) if gpu else "none"


def partial_phrase(rep: dict) -> str:
    g = rep.get("gpu") or {}
    return (f"(partial on {g.get('name')}, cc {g.get('cc')}: kernel levers unavailable — "
            f"{', '.join(rep.get('levers_unavailable') or rep.get('levers_fallback') or [])})")


def verdict(rc: int, rep: dict | None, incomplete: str | None = None) -> dict:
    """The exit rule of a run (pred, warm): ``rc`` = the run's own code; ``incomplete`` ("<n>/<m>", outputs short of the request)
    turns a 0 into EXIT_FAIL — the run's completeness verdict, first; ``partial`` (the report's list: levers of the mode's set the kit's
    records show not applied on this device — a kernel that could not compile or launch here —, the declared guards excluded:
    stack.settle_mk) turns a 0 into EXIT_NOT_ACTIVE: a mode is all of its levers on a GPU class and refuses by name, it never runs under
    its name with a subset (pred refuses before the first fold when the records already show it: cli; a lever whose application only
    the run's own counters can tell — mk — is judged here). A run that failed on its own (rc 1, or incomplete) keeps its own code.
    Returns {exit_code, partial, gated, incomplete, refused_partial}."""
    rep = rep or {}
    partial = list(rep.get("partial") or [])
    code = int(rc); refused = False
    if code == EXIT_OK and incomplete:
        code = EXIT_FAIL
    elif code == EXIT_OK and partial:
        code, refused = EXIT_NOT_ACTIVE, True
    return {"exit_code": code, "partial": partial, "gated": list(rep.get("gated") or []), "incomplete": incomplete, "refused_partial": refused}


PARTIAL_REFUSED = "{prefix} NOT ACTIVE: partial activation — {detail}: a mode is all of its levers on a GPU class, refused by name; exit " + str(EXIT_NOT_ACTIVE)   # the family grammar of the partial refusal (fixed parts byte-literal; `exit 3` = EXIT_NOT_ACTIVE)


def partial_detail(v: dict, reasons: dict | None = None) -> str:
    """``<detail>`` of the partial lines: the lever names first, then the kit's reason per lever."""
    why = "; ".join(f"{n}: {(reasons or {}).get(n) or 'no reason recorded'}" for n in v["partial"])
    return f"levers={_join(v['partial'])} ({why})"


def partial_line(v: dict, reasons: dict | None = None) -> str | None:
    """The one line of the partial refusal (every verb: pred — before its first fold when the records already show it, else after the
    run —, warm through pred): ``PARTIAL_REFUSED`` with ``<detail>`` = partial_detail (the levers with the kit's reasons); None when
    nothing is partial or the run failed on its own (EXIT_FAIL: its own line)."""
    if not v["partial"] or not v.get("refused_partial"):
        return None
    return PARTIAL_REFUSED.format(prefix=PREFIX, detail=partial_detail(v, reasons))


def attn_words(st: dict, forward: bool | None = None) -> str:
    """The attention-path words of a state dict (attn.words — the one formatter; report.py only places them on its lines)."""
    from .attn import words
    return words(st, forward=forward)


def _versions(rep: dict) -> str:
    up = rep.get("upstream") or {}
    return f"esm={up.get('esm')} transformers={up.get('transformers')}"


def not_active_line(reason: str) -> str:
    """The kit's refusal line for a reason without a mode head (the gates run.sh and the stock route print before any activation): the
    same `[esmfold2-opt] NOT ACTIVE: <reason>` grammar every route prints."""
    return f"{PREFIX} NOT ACTIVE: {reason}"


COMPILE_WORD = "compile=n/a"                            # the release tree's compile word on the ACTIVE / DRY-RUN line (compile=on|off:user|stepped_aside:<reason>|n/a): stock ESMFold2
                                                        # never calls torch.compile and this kit adds no compile step in any mode, so the word is the constant n/a here
                                                        # (MODEL_OPT_LEVERS_OFF=compile names no lever of this kit: ablation.foreign_note says so; nothing to turn off)


def graph_words(rep: dict | None) -> str:
    """`` compile=n/a graph_lru_sampler=<n> graph_budgets=<t>/<e>/<s>`` — the compile word (COMPILE_WORD), then the kit's sampler step-graph budget per graph
    generation as this process runs it (``effective.graph_lru_sampler``: mode > caller's EF2_GRAPH_LRU_SAMPLER > the package default 2;
    stack.effective_graph_settings) and the per-site graph budgets (trunk/encoder/sampler, 0 = no cap) — or ``graph_capture=off`` in their place when
    the capture policy is off (``effective.graph_capture`` == 0: the memory line captures no CUDA graph, so no budget governs anything); the last
    words of the ACTIVE / DRY-RUN line. The graph words are empty when the report carries no ``effective`` field."""
    eff = (rep or {}).get("effective") or {}
    out = " " + COMPILE_WORD
    out += f" graph_lru_sampler={eff['graph_lru_sampler']}" if "graph_lru_sampler" in eff else ""
    if str(eff.get("graph_capture", 1)) == "0":                            # the capture policy off (EF2_GRAPH_CAPTURE=0, the memory line): every graph site eager — the budgets govern nothing
        out += " graph_capture=off"
    elif "graph_budget_tokens_trunk_effective" in eff:                     # the per-site graph budgets that govern this process (ef2_opt._site_budget): trunk/encoder/sampler, 0 = no cap
        out += " graph_budgets={}/{}/{}".format(*(eff[f"graph_budget_tokens_{s}_effective"] for s in ("trunk", "encoder", "sampler")))
    return out


def route_word(rep) -> str:
    """' not_for_route=<a,b,...>' on the DRY-RUN / APPLIED lines of a row-sharded run (n_gpu > 1): the set's levers the multi-GPU line leaves off by
    its own table (modes.route_drop), each with a LEVER line ``state=skipped reason=not_for_route:n_gpu=<P>:<reason>``; '' at n_gpu 1."""
    if int((rep or {}).get("n_gpu") or N_GPU["P"] or 1) <= 1:
        return ""
    return " not_for_route=" + (",".join((rep or {}).get("levers_not_for_route") or {}) or "none")


def class_word(rep: dict | None) -> str:
    """' not_for_class=<a,b,...>' on the DRY-RUN / APPLIED lines when the box's compute-capability class leaves levers of the set to another
    class's kernels (modes.class_drop: t15 / t15msa / t10 on 9.0, where t16 serves; t16 on the other classes), each with a LEVER line
    ``state=skipped reason=not_for_class:<class>:<reason>``; '' when nothing left the set by class."""
    off = (rep or {}).get("levers_not_for_class") or {}
    return (" not_for_class=" + ",".join(off)) if off else ""


def ablate_word(rep: dict | None) -> str:
    """`` ablate=<tokens>`` when the ablation variable subtracted levers or set knobs in this process (ablation.py), else '' — a run under a
    mode's plain name never carries the word."""
    toks = (rep or {}).get("ablate_tokens") or []
    return (" ablate=" + ",".join(str(t) for t in toks)) if toks else ""


def activation_line(rep: dict | None) -> str:
    rep = rep or {}
    mode, variant = rep.get("mode"), rep.get("variant")
    head = f"mode={mode} variant={variant}" + (f" line={rep['line']}" if rep.get("line") else "")     # a mode with lines names the line resolved
    if rep.get("active"):
        axis = n_gpu_fields(rep.get("n_gpu"))                                                          # the --n_gpu axis token, after gpu= on the ACTIVE / DRY-RUN lines (n_gpu=1 sharding=none in-process)
        levers = rep.get("levers_applied") if rep.get("applied") == "configured" else rep.get("levers_planned")
        line = (f"{PREFIX} ACTIVE {head} server_mode={rep.get('server_line')} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))} {axis} "
                f"levers={_join(levers)} fallbacks={_join(rep.get('levers_fallback'))} applied={rep.get('applied')}{ablate_word(rep)}")
        if rep.get("attn"):                                                                             # the attention-path words (attn.py): upstream's flag in this process; the bound forwards once a model exists
            line += " " + attn_words(rep["attn"], forward=False)
        if rep.get("partial"):
            line += " " + partial_phrase(rep)
        return line + graph_words(rep)
    reason = rep.get("reason") or "no reason given by esmfold2_opt.enable()"
    if rep.get("dry_run") and rep.get("server_mode"):
        tg = rep.get("target_gpu")
        axis = n_gpu_fields(rep.get("n_gpu"))
        return (f"{PREFIX} DRY-RUN {head} server_mode={rep.get('server_line')} {_versions(rep)} gpu={gpu_label(rep.get('gpu'))} {axis}"
                + (f" target_gpu={tg}" if tg else "")
                + f" levers={_join(rep.get('levers_planned'))} not_for_variant={_join(rep.get('levers_not_for_variant'))}{class_word(rep)}{route_word(rep)}{ablate_word(rep)}"
                + f" kernel_key={rep.get('kernel_key') or 'unknown'} env_vars={len(rep.get('env') or {})}"
                + (f" {rep['attn_metadata']}" if rep.get("attn_metadata") else ""
                + graph_words(rep))
                + (f" would_refuse={rep['would_refuse']!r}" if rep.get("would_refuse") else "")
                + (" notes=" + "; ".join(rep["notes"]) if rep.get("notes") else ""))
    return f"{PREFIX} NOT ACTIVE: {reason}" + (f" ({head})" if mode else "")



def _seed_word(seed):
    """A fold's seed on the kit's per-fold lines: the integer, or ``none`` for upstream's unseeded default (an input with no ``seeds`` and no
    ``--seeds``: one unseeded fold — the line must print, not raise)."""
    return "none" if seed is None else int(seed)

def fold_line(name: str, seed: int, n_samples: int, wall_s: float) -> str:
    """The per-(item, seed) fold line: ``[esmfold2-opt] fold <name> s<seed> samples=<k> <s>s``
    — the stock subprocess prints the same grammar under its own prefix (stock_fold.py); a kit line that folds outside the server (the
    multi-GPU line, tp.py) prints it from the run's own record."""
    return f"{PREFIX} fold {name} s{_seed_word(seed)} samples={int(n_samples)} {float(wall_s):.2f}s"


def done_line(n_predictions: int, n_items: int, out_dir: str) -> str:
    """The end marker of a pred run: ``[esmfold2-opt] DONE predictions=<n> items=<m> out_dir=<dir>`` (stock_fold.py prints the same grammar)."""
    return f"{PREFIX} DONE predictions={int(n_predictions)} items={int(n_items)} out_dir={out_dir}"


def ready_line(variant, t_load_s: float) -> str:
    """ONE line at the moment the weights are on the device (every route prints it: pred, the stock caller), so a log reader can
    split cold start from steady state: ``[esmfold2-opt] ready variant=<v> t=<s>``."""
    return f"{PREFIX} ready variant={variant} t={float(t_load_s):.2f}"


def applied_line(rep: dict, rec: dict) -> str:
    sub = ",".join(f"{a}->{b}" for a, b in (rec.get("levers_substituted") or {}).items())
    line = (f"{PREFIX} APPLIED model#{rec.get('model_index')} {('msa_encoder' if rec.get('has_msa_encoder') else 'no_msa_encoder')} "
            f"mode={rep.get('mode')} variant={rep.get('variant')} server_mode={rep.get('server_line')} levers={_join(rec.get('levers_applied'))} "
            f"fallbacks={_join(rec.get('levers_fallback'))} substituted={sub or 'none'} not_for_variant={_join(rec.get('levers_not_for_variant'))}{class_word(rep)}{route_word(rep)}{ablate_word(rep)}")
    if rec.get("attn"):                                                                                  # atom_attn=<kernel> atom_forward=<the forward each SWA3DRoPEAttention instance resolves, counted> (attn.py, read on this model)
        line += " " + attn_words(rec["attn"], forward=True)
    if rec.get("levers_fallback"):
        line += " " + partial_phrase(dict(rep, levers_unavailable=rec["levers_fallback"]))
    return line


def _slug(text) -> str:
    """A blank-free token of a free-text reason (whitespace runs -> '_'), so a per-lever line stays k=v parseable."""
    import re as _re
    return _re.sub(r"\s+", "_", str(text).strip()) or "none"


def lever_lines(rep: dict, rec: dict, include_deferred: bool = True) -> list:
    """ONE evidence line per registry lever for this process's arm, from the kit's own records after configure() (the same classification the
    APPLIED line summarises), in the release tree's pinned per-lever grammar (opt_core.report.lever_line): ``[esmfold2-opt] LEVER name=<flag>
    state=<on|off|skipped> [reason=<kind:detail, blank-free>] impl=<kit file> origin=kit strategy=<F<k>.name|LOCAL.name> mode=<m>
    variant=<v> model=<i> tier_vs_stock=<T>``. ``on`` = applied and live on this model instance; ``off`` = not in this mode's lever set; ``skipped`` = in the set
    but not applied here — substituted by the device policy, fallen back (the record's reason), not for this variant, or not driven by this
    route — always with its reason; ``skipped reason=deferred:rowpair.install_rank`` = a row-chunking lever of the n_gpu > 1 route whose install
    runs after configure() (stack.settle_route prints its definitive line; apply_to passes ``include_deferred=False`` so a log carries one line per lever)."""
    from opt_core import arch
    from opt_core.report import lever_line
    from . import cards
    from .registry import LEVERS, STRATEGY
    g = rep.get("gpu") or {}
    sm = arch.sm_of(g.get("cc") or g.get("sm")) or arch.current_sm()[0]   # the activation report's device, else the box's (torch / nvidia-smi); None on a CPU host (word no_gpu)
    applied = set(rec.get("levers_applied") or ())
    fallback = set(rec.get("levers_fallback") or ())
    substituted = dict(rec.get("levers_substituted") or {})
    not_for_variant = set(rec.get("levers_not_for_variant") or ())
    out_of_scope = dict(rec.get("levers_out_of_scope") or {})
    ablated = set(rep.get("ablate") or ())
    dropped = set(rep.get("levers_dropped") or ())
    off_route = dict(rep.get("levers_not_for_route") or {})
    off_class = dict(rep.get("levers_not_for_class") or {})
    deferred = set(rec.get("levers_deferred") or ())
    from .registry import class_word as _class_word
    why = dict(rec.get("fallback_reasons") or {})
    planned = set(rep.get("levers_planned") or ())
    from .registry import FIELD_RC as _FIELD_RC
    out = []
    for name, lv in LEVERS.items():
        if lv.field == _FIELD_RC and name not in (planned | ablated | deferred | applied | fallback):
            continue                                                      # the multi-GPU route's row-chunking levers are in no n_gpu 1 set: no census line there (the n_gpu 1 census is the base tree's, line for line)
        if name in applied:
            state, reason = "on", None
        elif name in substituted:                                     # reason values are ONE blank-free token (the line is k=v parsed): kind:detail
            state, reason = "skipped", f"substituted_by:{substituted[name]}"
        elif name in fallback:
            state, reason = "skipped", "fallback:" + _slug(why.get(name) or "kit record unset after configure()")
        elif name in out_of_scope:
            state, reason = "skipped", "guard:" + _slug(out_of_scope[name])
        elif name in not_for_variant:
            state, reason = "skipped", f"not_for_variant:{rep.get('variant')}"
        elif name in off_route:                                       # the row-sharded route leaves it off at n_gpu > 1 (modes.route_drop: the multi-GPU line's own table, with its reason)
            state, reason = "skipped", f"not_for_route:n_gpu={rep.get('n_gpu') or N_GPU['P']}:{_slug(off_route[name])}"
        elif name in off_class:                                       # another compute-capability class's kernels serve this lever's op on this box (modes.class_drop, with its reason)
            state, reason = "skipped", f"not_for_class:{_class_word(rep.get('class_cc'))}:{_slug(off_class[name])}"
        elif name in deferred:                                        # a row-chunking lever of the n_gpu > 1 route: installed by rowpair.install_rank after configure(); settle_route prints its line
            if not include_deferred:
                continue
            state, reason = "skipped", "deferred:rowpair.install_rank"
        elif name in ablated:                                         # the ablation variable took it out of the set before configure() (ablation.py)
            state, reason = "off", "ablated"
        elif name in dropped:                                         # the resolved line leaves it off by design (modes.KitMode.drop)
            state, reason = "off", f"dropped_by_line:{rep.get('line') or rep.get('mode')}"
        else:
            state, reason = "off", None
        fields = cards.words_for(name, sm)                            # the arch registry's words for this class (sm= / card_support= / card= + card_reason=); the STATE stays the kit's own record of what ran
        if state == "on":
            fields.update(lever_evidence(name))                       # the module's own record of HOW it engaged (t15's tile row, xte's self-check, the precision knobs, the declared scopes)
        out.append(lever_line(TAG, name, state, reason=reason, impl=lv.kit_file, origin="kit", strategy=STRATEGY.get(name, "LOCAL.unfiled"),
                              mode=rep.get("mode"), variant=rep.get("variant"), model=rec.get("model_index"), tier_vs_stock=lv.tier_vs_stock, **fields))
    return out


def lever_evidence(name: str) -> dict:
    """Blank-free k=v evidence fields for an engaged lever's LEVER line, read from its module's own state in this process ({} when the module holds
    nothing to say): t15 / t15msa ``t15_row=sm90|sm80``; af ``gemm=<precision>``; dit ``gemm= cond=
    attn= attn_precision=``; ro ``scope=nds1_batch1``, kd / dit ``scope=ndsN_batchN``; glue ``sigmoid_check=ok:<n>``; rg ``max_tokens=<n>``; mh ``engages=per_fold``;
    t16 ``cubin=<sha12> cubin_source=shipped regs= spill_bytes= smem= arch=sm_90a canary= serves=``; tx ``word= row= cell= stack= abi= cc= [prefer=]
    refused= stock_classes=`` (+ ``why=`` when the word names the stock op; ``prefer=tx_sm90a:eager:N<=512`` under the big word on class 9.x); xln ``sites= floor_rows= core= kernels= cc= provider=``; xte ``word= row= core= cc= modules= bitwise_check= window= inplace= launch=``; x4 ``engine=pinned|pageable`` is the add-on's own word on its line (big.py)."""
    ev = {}
    try:
        if name in ("t15", "t15msa", "xtr"):                                # word=<tier word bound> provider= core= (+ rows=<kind:row=n,...> refused=<kind:reason=n,...> once calls ran)
            m = sys.modules.get("ef2_pair_v2")
            if m is not None and hasattr(m, "bind_words"):
                ev.update({k: _slug(v) for k, v in m.bind_words().items()})
        elif name == "af":
            m = sys.modules.get("ef2_atom")
            ev["gemm"] = str((getattr(m, "_CFG", {}) or {}).get("gemm")) if m else "unknown"
        elif name in ("ro", "kd", "dit"):
            m = sys.modules.get("ef2_dit")
            if m is not None:
                sc = (m.lever_scope(name) if hasattr(m, "lever_scope") else getattr(m, "SCOPE", {})) or {}   # ro = nds1_batch1 (the roll-out); kd / dit = ndsN_batchN (any sample count of one input)
                ev["scope"] = f"nds{sc.get('num_diffusion_samples')}_batch{sc.get('batch')}"
                if name == "dit":
                    ev.update({k: _slug(v) for k, v in m.knobs().items()})
                if name == "ro":
                    ev["kabsch"] = str((getattr(m, "_CFG", {}) or {}).get("kabsch"))
        elif name == "glue":
            m = sys.modules.get("ef2_hoist")
            ev["sigmoid_check"] = m.sigmoid_check_word() if m else "unknown"
        elif name == "rg":
            m = sys.modules.get("ef2_opt")
            ev["max_tokens"] = int(getattr(getattr(m, "CFG", None), "recycle_graph_max_tokens", 0)) if m else 0; ev["engages"] = "per_shape"
        elif name == "mh":
            ev["engages"] = "per_fold"                                # data-dependent: the fold's LEVERFOLD line says whether it served
        elif name == "xte":
            m = sys.modules.get("ef2_xte")
            if m is not None:
                ev.update({k: _slug(v) for k, v in m.evidence().items()})
        elif name in ("injrows", "confrows", "confbf16", "pdeskip", "confmem", "biasfree", "zbf16"):   # the row-chunking levers: the installer's knobs in force (block MiB, token floor)
            m = sys.modules.get("esmfold2_opt.rowchunk.install")
            if m is not None:
                ev.update({k: _slug(v) for k, v in m.evidence(name).items()})   # word= row= core= cc= modules= bitwise_check= check_rows= window= inplace= launch= provider= (+ refused=)
        elif name == "xln":
            m = sys.modules.get("ef2_xln")
            if m is not None:
                ev.update({k: _slug(v) for k, v in m.evidence().items()})   # sites= floor_rows= core= kernels= cc= provider=
        elif name == "t16":
            m = sys.modules.get("ef2_transition_cute")
            if m is not None:
                ev.update({k: _slug(v) for k, v in m.kernel_evidence().items()})
        elif name == "tx":                                            # ef2_w4's provider binding: the bound tier word, the row(s) the shared core's cell table served so far ('+'-joined;
            w4 = sys.modules.get("ef2_w4")                            # the canary's row before the first fold), the canary's cell, this process's stack word / prebuilt ABI key / class,
            st = (w4.describe() or {}).get("tx_state") if w4 is not None else None   # the refusal kinds counted so far, the classes the stock op serves by name (or why no kernel row serves)
            if st:
                rows = sorted((st.get("rows") or {}).items())
                ev["word"] = _slug(st.get("word")); ev["row"] = _slug("+".join(r for r, _ in rows) or st.get("row") or "upstream"); ev["cell"] = _slug(st.get("cell"))
                ev["stack"] = _slug(st.get("stack")); ev["abi"] = _slug(st.get("abi")); ev["cc"] = _slug(st.get("cc"))
                if st.get("prefer_rule"):                             # the binding's eager row preference under the big word (tx_sm90a:eager:N<=512); absent for fast / exact
                    ev["prefer"] = _slug(st.get("prefer_rule"))
                ev["refused"] = _slug(",".join(f"{k}:{v}" for k, v in sorted((st.get("refused") or {}).items())) or "none"); ev["stock_classes"] = int(st.get("stock_classes") or 0)
                if not st.get("on") and st.get("why"):
                    ev["why"] = _slug(st.get("why"))
    except Exception as e:  # noqa: BLE001
        ev["evidence_error"] = _slug(repr(e))[:80]
    return {k: v for k, v in ev.items() if v is not None}


def log_activation(rep: dict | None, stream=None) -> str:
    line = activation_line(rep)
    if not (rep or {}).get("logged"):
        print(line, file=stream or sys.stderr, flush=True)
    return line


# ----------------------------------------------------------------------------------------------------------------- exit tally
def _counters(mod, attr: str):
    m = sys.modules.get(mod)
    if m is None:
        return None
    v = getattr(m, attr, None)
    try:
        d = v() if callable(v) else v
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}
    return dict(d) if isinstance(d, dict) else None


def memory_stats() -> dict | None:
    out = {}
    opt = _counters("ef2_opt", "stats")
    if opt is not None:
        out["ef2_opt"] = {k: v for k, v in opt.items() if any(w in str(k) for w in OPT_KEY_WORDS)} or opt
        inv = _counters("ef2_opt", "invariant_report")
        if inv is not None:
            out["graph_invariant"] = inv
    w4 = _counters("ef2_w4", "stats")
    if w4 is not None:
        out["ef2_w4"] = {k: v for k, v in w4.items() if k not in ("probe",)}
    msa = _counters("ef2_msa", "STATS")
    if msa is not None:
        out["ef2_msa"] = msa
    mk = _counters("ef2_mk_sampler", "stats")
    if mk is not None:
        out["ef2_mk_sampler"] = mk
    xl = _counters("ef2_xl", "stats")                                          # the XL add-on (an xl mode): its engagement counters
    if xl is not None:
        out["ef2_xl"] = xl
    for mod in LEVER_MODULES_V2:                                                # the lever modules of the fourth field's groups: every scalar counter each holds
        d = _counters(mod, "stats")
        if d is not None:
            out[mod] = d
    rc = _counters("esmfold2_opt.rowchunk.install", "counters")                 # the multi-GPU route's row-chunking levers (n_gpu > 1 rank processes only): rowchunk={levers=a+b,<member>_<counter>=…}
    if rc is not None:
        out["rowchunk"] = rc
    return out or None


LEVER_MODULES_V2 = ("ef2_atom", "ef2_feats", "ef2_msa_v2", "ef2_pair_v2", "ef2_transition_cute", "ef2_xte", "ef2_hoist", "ef2_xln", "ef2_dit")   # driver modules with a stats() of counters, tallied at exit after the first five


CACHE_KEYS = ("feature_cache_hits", "feature_cache_stores", "esmc_cache_hits", "esmc_cache_stores")   # the kit's content-keyed cross-fold caches (ef2_opt fc / ec): a store = a miss computed inside the call
ENV_CACHE_SCOPE = "ESMFOLD2_OPT_CACHE_SCOPE"                                      # input (default: the caches hold the CURRENT input only — emptied when the fold loop moves to a new input) | global (the caches live for the process)
CACHE_SCOPES = ("global", "input")


def cache_scope(environ=None) -> str:
    """The kit caches' scope word (``ESMFOLD2_OPT_CACHE_SCOPE``): ``input`` (default: resident cross-input state is one input — the feature /
    ESM-C caches are emptied when the fold loop moves to a new input, and the captured-graph generation is kept only while the inputs' shape
    signature holds (graphgen_decide: reused across inputs of one signature, released before an input of another); the seeds of one input still hit
    and replay) or ``global`` (the caches and the generation live for the process); any other word is refused by name."""
    environ = os.environ if environ is None else environ
    w = (environ.get(ENV_CACHE_SCOPE) or "input").strip().lower()
    if w not in CACHE_SCOPES:
        raise ValueError(f"{ENV_CACHE_SCOPE}={environ.get(ENV_CACHE_SCOPE)!r}: one of {'|'.join(CACHE_SCOPES)}")
    return w


def cache_counts() -> dict | None:
    """The kit's feature / ESM-C cache counters now (ef2_opt.STATS), or None when the kit's lever module is not loaded in this process (the stock arm)."""
    m = sys.modules.get("ef2_opt")
    st = getattr(m, "STATS", None) if m is not None else None
    if st is None:
        return None
    return {k: int(st.get(k, 0)) for k in CACHE_KEYS}


def cache_clear() -> bool:
    """Empty the kit's feature and ESM-C caches (scope ``input``): every entry is dropped, so the cached CUDA tensors of earlier inputs are released
    (nothing else references them); False when the kit's lever module is not loaded."""
    m = sys.modules.get("ef2_opt")
    if m is None:
        return False
    for name in ("_FEATURE_CACHE", "_ESMC_CACHE"):
        c = getattr(m, name, None)
        if c is not None:
            c.clear()
    mh = sys.modules.get("ef2_msa_v2")                                          # the MSA-encoder hoist's held output (lever mh: one pair-sized tensor of the previous input) goes with them
    if mh is not None and callable(getattr(mh, "mh_clear", None)):
        mh.mh_clear()
    return True


def graphs_held() -> int:
    """How many captured CUDA graphs the kit's lever module holds right now (its graphed modules' per-shape graphs + the graphed samplers' step
    graphs); 0 when the module is not loaded or nothing is captured."""
    m = sys.modules.get("ef2_opt")
    if m is None:
        return 0
    n = sum(len(getattr(x, "_ef2opt_graphs", None) or ()) for x in list(getattr(m, "_GRAPHED_MODULES", None) or ()))
    n += sum(len(getattr(x, "_ef2opt_step_graphs", None) or ()) for x in list(getattr(m, "_GRAPHED_SAMPLERS", None) or ()))
    return n


def graphs_clear(reason: str = "new_input") -> int:
    """Release the kit's captured-graph generation before an input of a NEW shape signature (scope ``input``; graphgen_decide): ``ef2_opt.clear_graphs``
    — every captured graph, every static side buffer a graph reads (SWA masks, pair-bias buffers), a fresh graph pool, the caching allocator emptied
    (the kit's whole-generation protocol; never an individual graph). The graphs and their private pool are sized to the previous signature's shapes:
    kept alive they hold that footprint in reserve while the next input's LM encode and trunk allocate outside the pool — the memory that grows input
    over input. Returns the number of graphs released (0: the module is not loaded or held nothing — the first input, the stock arm)."""
    m = sys.modules.get("ef2_opt")
    n = graphs_held()
    if m is None or n == 0 or not callable(getattr(m, "clear_graphs", None)):
        return 0
    m.clear_graphs(reason=reason)
    return n


def graphs_line(item_id, released: int) -> str:
    """``GRAPHS item=<id> released=<n> reason=new_input`` — printed before an input whose shape signature differs from the live generation's (the
    generation is released first; graphgen_decide)."""
    return f"{PREFIX} GRAPHS item={item_id} released={int(released)} reason=new_input (the previous input's captured graphs, their static buffers and graph pool; this input re-captures)"


# ---- the captured-graph generation, keyed on the input's shape signature ----------------------------------------------------------------------
# What the kit's captures are specialised on (opt/forward/fast_inference/driver/ef2_opt.py; every key is a tensor SHAPE + dtype, never content —
# inputs are copied into static buffers before each replay, static side buffers are refreshed in place per fold):
#   tg  graphed trunks (folding_trunk, parcae_coda, confidence_head.folding_trunk; lm_encoder when lm_dropout == 0) — key (pair, pair mask) shapes,
#       pair = [1, tok, tok, C]  (_graphed_trunk_forward)                                                              -> tok
#   eg  graphed encoders (lm_encoder, msa_encoder) — key = the shapes of every argument (_graphed_generic_forward, _sig_args): token-level
#       [1, tok, ...] and, for the MSA encoder, the per-loop MSA block [rows, tok, ...]                                       -> tok, msa
#   pb  pair-bias static buffers of the diffusion token transformer — key = z shape [S, tok, tok, C]                    -> tok, nds
#   sg  the graphed sampler's diffusion STEP graph — key = (_tree_sig(kwargs), _tree_sig(inference_cache), nds) (_sample_v2): the atom-level
#       arguments x_noisy / ref_* [S, atom, ...], the token-level ones [S, tok, ...] and, with flash-attn, the varlen index tensors of the atom
#       layout inside inference_cache (their lengths follow the input's real atom count)                                     -> tok, atom, nds
# So (tok, msa, nds) decide the whole generation; the atom count decides only the sampler's step graph, of which the kit keeps up to CFG.lru_sampler
# (EF2_GRAPH_LRU_SAMPLER, default 8) per generation beside each other and resets the WHOLE generation itself when that budget is spent.
GENERATION = {"sig": None, "atoms": [], "reuse": 0, "rebuild": 0, "armed": None, "decided": None,
              "base_alloc": None, "prologue_seen": False, "prologue_need": {}, "headroom_events": [], "headroom_budget": None}

# ---- the headroom guard (memory policy of the reuse path) --------------------------------------------------------------------------------------
# A live generation holds its graphs' private pool AND the static input / output clones of every graphed region for the life of the generation
# (the LM, the graph pools, the static clones, the pair-bias / sampler statics).
# The FIRST fold of a signature runs its eager prologue (LM pass, LM shim SingleToPair, pair init, embedder: the largest transient of the fold)
# BEFORE any pool exists and then captures; the NEXT fold of the same signature runs that prologue ON TOP of the resident generation — the one
# region no fold has proven to fit (everything after the first capture ran with the pools resident on the first fold). The guard measures the
# prologue's transient on every fold (max_memory_allocated at the first graphed-region entry minus memory_allocated at the model call: probe hooks,
# graphgen_install) and, on the reuse path, compares it with what the driver can still hand out (mem_get_info free; the kit empties the caching
# allocator after every fold, so cached-free is ~0 and private-pool free space is rightly not counted). When it does not fit, the generation is
# RELEASED before the network allocates (graphs_clear reason=headroom) and ef2_opt's per-shape graph budget (CFG.graph_budget_tokens, the
# EF2_GRAPH_BUDGET_TOKENS switch) is lowered to tok-1 for the rest of the process: this size and larger run the graph levers eagerly (same kernels,
# no capture: bitwise the captured path's numerics), smaller inputs keep capturing. Never fires while the prologue fits (small N: need << free).
HEADROOM_MARGIN_FRAC = 0.03                                                # of the device's total memory, kept free on top of the measured need
HEADROOM_MARGIN_MIN = 512 * 2 ** 20                                        # bytes


def _cuda_mem():
    """(free, total, allocated, max_allocated) in bytes from torch.cuda, or None without CUDA statistics (CPU stub: the guard is inert)."""
    torch = sys.modules.get("torch")
    cuda = getattr(torch, "cuda", None) if torch is not None else None
    try:
        if cuda is None or not cuda.is_available():
            return None
        free, total = cuda.mem_get_info()
        return int(free), int(total), int(cuda.memory_allocated()), int(cuda.max_memory_allocated())
    except Exception:  # noqa: BLE001
        return None


def headroom_fold_begin() -> None:
    """At an input's model call (graphgen hook, before the network allocates): the fold's allocation baseline for the prologue probe."""
    m = _cuda_mem()
    GENERATION["base_alloc"] = None if m is None else (m[2], m[3])          # (allocated, running peak) at the call
    GENERATION["prologue_seen"] = False


def headroom_probe() -> None:
    """At the first graphed-region entry of a fold (pre-hook on lm_encoder / msa_encoder / folding_trunk): the prologue's transient =
    max_memory_allocated now - allocated at the model call, recorded per (tok, msa, nds) as the largest seen (the need the guard tests)."""
    if GENERATION["prologue_seen"] or GENERATION["base_alloc"] is None or GENERATION["sig"] is None:
        GENERATION["prologue_seen"] = True
        return
    GENERATION["prologue_seen"] = True
    m = _cuda_mem()
    if m is None:
        return
    base, peak0 = GENERATION["base_alloc"]
    peak = m[3]
    if peak <= peak0:                                                      # the running peak was set before this fold's prologue: not this prologue's measure
        return
    key = (GENERATION["sig"][0], GENERATION["sig"][2], GENERATION["sig"][3])
    need = max(0, peak - base)
    if need > GENERATION["prologue_need"].get(key, 0):
        GENERATION["prologue_need"][key] = need


def headroom_check(sig: tuple):
    """The reuse-path test: None when the guard has nothing to say (no CUDA statistics, no measured prologue for this signature, or it fits);
    else a dict(need, free, total, margin) — the live generation must be released before this input's network runs."""
    key = (sig[0], sig[2], sig[3])
    need = GENERATION["prologue_need"].get(key)
    m = _cuda_mem()
    if need is None or m is None:
        return None
    free, total = m[0], m[1]
    margin = max(HEADROOM_MARGIN_MIN, int(HEADROOM_MARGIN_FRAC * total))
    if need + margin <= free:
        return None
    return {"need": need, "free": free, "total": total, "margin": margin}


def headroom_apply(sig: tuple, verdict: dict) -> int:
    """Release the live generation (graphs_clear reason=headroom) and make this token count and larger run the graph levers eagerly for the rest of
    the process (ef2_opt.CFG.graph_budget_tokens = tok - 1, only ever lowered). Returns the number of graphs released."""
    tok = int(sig[0])
    released = graphs_clear(reason="headroom")
    m = sys.modules.get("ef2_opt")
    cfg = getattr(m, "CFG", None)
    budget = tok - 1
    try:
        cur = int(getattr(cfg, "graph_budget_tokens", 0) or 0)
    except (TypeError, ValueError):
        cur = 0
    if cfg is not None and (cur <= 0 or budget < cur):
        try:
            cfg.graph_budget_tokens = budget
        except Exception:  # noqa: BLE001
            pass
    else:
        budget = cur if cur > 0 else budget
    GENERATION["headroom_budget"] = budget
    GENERATION["headroom_events"].append(dict(tok=tok, released=released, budget=budget, **verdict))
    return released


def headroom_line(item_id, released: int, ev: dict) -> str:
    """``GRAPHS item=<id> released=<n> reason=headroom need_gib=<f> free_gib=<f> budget_tokens=<n> (...)`` — printed before an input of the live
    generation's own signature whose eager prologue would not fit beside the generation's pools and static buffers."""
    g = float(2 ** 30)
    return (f"{PREFIX} GRAPHS item={item_id} released={int(released)} reason=headroom need_gib={ev['need'] / g:.2f} free_gib={ev['free'] / g:.2f} "
            f"budget_tokens={int(ev['budget'])} (this input's eager prologue measured {ev['need'] / g:.2f} GiB on its first fold; the live generation's graph pools "
            f"and static buffers leave {ev['free'] / g:.2f} GiB: released before the network allocates, and inputs of {int(ev['budget']) + 1} tokens or more run the "
            f"graph levers eagerly from here — same kernels, no capture)")


def _dim(t, axis: int) -> int:
    shape = getattr(t, "shape", None)
    return int(shape[axis]) if shape is not None and len(shape) >= abs(axis) else 0


MSA_MAX_DEPTH_DEFAULT = 1024          # upstream ESMFold2Model.forward(msa_max_depth=1024, msa_subsample_at_inference=True): the per-recycle row draw
MSA_SUBSAMPLE_DEFAULT = True


def msa_rows(kwargs: dict) -> int:
    """The MSA row extent the network's captured shapes actually see: what ``maybe_subsample_msa`` hands the MSA encoder every recycle — exactly
    ``msa_max_depth`` rows when subsampling is on and the featurised depth exceeds it (two inputs deeper than it are both ``msa_max_depth`` rows inside
    the encoder: the same captured shapes, content restaged per replay), else the featurised depth itself; 0 without an MSA."""
    msa = kwargs.get("msa")
    if msa is None:
        return 0
    depth = _dim(msa, -2)
    max_depth = kwargs.get("msa_max_depth", MSA_MAX_DEPTH_DEFAULT)
    enabled = kwargs.get("msa_subsample_at_inference", MSA_SUBSAMPLE_DEFAULT)
    if enabled and max_depth is not None and depth > 1 and depth > int(max_depth):
        return int(max_depth)
    return depth


def shape_signature(kwargs: dict) -> tuple | None:
    """The shape signature of one model call — upstream's ``model(**features, num_diffusion_samples=…, …)`` in ``ESMFold2InputBuilder.fold`` — as
    ``(tok, atom, msa, nds)``: ``tok`` the token count (``token_attention_mask``'s length; upstream pads no tokens), ``atom`` the featurised atom
    count (``atom_attention_mask``'s length; upstream's prepare_input pads the atoms to a multiple of 32), ``msa`` the MSA rows PER RECYCLE
    (``msa_rows``: the subsample extent the encoder graphs are captured on, not the featurised depth; 0 without an MSA), ``nds``
    ``num_diffusion_samples``. dtypes, the device and the fold settings are one per process and not part of it. None when the call carries no
    token mask (not a fold call)."""
    tm = kwargs.get("token_attention_mask")
    if getattr(tm, "shape", None) is None:
        return None
    return (_dim(tm, -1), _dim(kwargs.get("atom_attention_mask"), -1), msa_rows(kwargs), int(kwargs.get("num_diffusion_samples") or 1))


def signature_word(sig: tuple) -> str:
    """``tok:<n>,atom:<n>,msa:<n>,nds:<n>`` — the GRAPHGEN line's ``shape`` field."""
    tok, atom, msa, nds = sig
    return f"tok:{tok},atom:{atom},msa:{msa},nds:{nds}"


def sampler_budget() -> int:
    """The kit's sampler step-graph budget per generation as the sampler applies it in this process (``ef2_opt.CFG.lru_sampler``; 8 when unread)."""
    m = sys.modules.get("ef2_opt")
    try:
        return max(1, int(getattr(getattr(m, "CFG", None), "lru_sampler", 8)))
    except (TypeError, ValueError):
        return 8


def sampler_graphs_held() -> int:
    """How many diffusion step graphs the graphed samplers hold in the live generation."""
    m = sys.modules.get("ef2_opt")
    if m is None:
        return 0
    return sum(len(getattr(x, "_ef2opt_step_graphs", None) or ()) for x in list(getattr(m, "_GRAPHED_SAMPLERS", None) or ()))


def graphgen_arm(item_id) -> None:
    """The fold loop names the input it moves to (scope ``input``): the next model call decides reuse / rebuild for it and prints its GRAPHGEN line."""
    GENERATION["armed"] = item_id


def graphgen_decide(sig: tuple) -> tuple:
    """Reuse or rebuild the live captured-graph generation for a model call of shape signature ``sig`` = (tok, atom, msa, nds); returns
    ``(decision, released)`` with decision ``"reuse"`` | ``"rebuild"`` | ``"headroom"`` and ``released`` the number of graphs released.

    * no live generation (the first fold of the process)                        -> rebuild (the fold captures; nothing to release)
    * (tok, msa, nds) differ from the live generation's                          -> rebuild: the generation is RELEASED FIRST (graphs_clear), then recorded
    * equal, and ``atom`` is one the generation has captured a step graph for    -> reuse: every graph replays
    * equal, a new ``atom``                                                      -> reuse: the trunk / encoder graphs replay; the step-graph sampler
                                                                                    (where ``ro`` steps aside) captures one step graph beside the live
                                                                                    ones and, its budget spent (sampler_graphs_held() >= budget), resets
                                                                                    its OWN sub-generation at that fold's first graphed step (ef2_opt
                                                                                    clear_sampler_graphs: the step graphs and their pool only); the
                                                                                    trunk graphs stay
    * equal (a reuse), but this signature's measured eager prologue does not     -> headroom: released first, ef2_opt's graph budget lowered to tok-1
      fit in the memory the live generation leaves free (headroom_check)            (this size and larger run eagerly from here; counted as a rebuild)
    The counts ``reuse`` / ``rebuild`` are process-cumulative."""
    tok, atom, msa, nds = sig
    live = GENERATION["sig"]
    key = (tok, msa, nds)
    if live is not None and (live[0], live[2], live[3]) == key:
        if True:                                                                # every atom count of a live (tok, msa, nds) reuses the generation (the sampler's step-graph budget is its own sub-generation)
            verdict = headroom_check(sig) if graphs_held() > 0 else None       # the memory policy of the reuse path (headroom guard, above)
            if verdict is not None:
                released = headroom_apply(sig, verdict)                        # released FIRST; this size and larger run eagerly from here
                GENERATION["sig"] = sig
                GENERATION["atoms"] = [atom]
                GENERATION["rebuild"] += 1
                return "headroom", released
            if atom not in GENERATION["atoms"]:
                GENERATION["atoms"].append(atom)
            GENERATION["sig"] = sig
            GENERATION["reuse"] += 1
            return "reuse", 0
    released = graphs_clear() if live is not None else 0                       # release FIRST, then record the new generation
    GENERATION["sig"] = sig
    GENERATION["atoms"] = [atom]
    GENERATION["rebuild"] += 1
    return "rebuild", released


def graphgen_line(item_id, sig: tuple) -> str:
    """``GRAPHGEN item=<id> reuse=<n> rebuild=<n> shape=tok:<n>,atom:<n>,msa:<n>,nds:<n>`` — one per input on a kit route, before its network runs:
    the process-cumulative reuse / rebuild counts after this input's decision and its shape signature."""
    return f"{PREFIX} GRAPHGEN item={item_id} reuse={GENERATION['reuse']} rebuild={GENERATION['rebuild']} shape={signature_word(sig)}"


def graphgen_observe(kwargs: dict) -> list:
    """The decision for one model call, as the lines to print (stderr): [] on the stock arm (no kit module in the process), under scope ``global``,
    for a call that is not a fold call, and for the later calls of an input already decided whose signature holds (its seeds: silent replay); for the
    armed input's first call the GRAPHS line (when a generation was released) and its GRAPHGEN line; for an unarmed call whose signature changed
    (a caller outside the fold loop) the same lines under the last item id."""
    if sys.modules.get("ef2_opt") is None or cache_scope() != "input":
        return []
    sig = shape_signature(kwargs)
    if sig is None:
        return []
    item = GENERATION["armed"] if GENERATION["armed"] is not None else GENERATION["decided"]
    first_call = GENERATION["armed"] is not None
    headroom_fold_begin()                                                      # every fold call: the prologue probe's baseline (the guard's measure)
    if not first_call and sig == GENERATION["sig"]:
        return []
    n_ev = len(GENERATION["headroom_events"])
    decision, released = graphgen_decide(sig)
    GENERATION["armed"], GENERATION["decided"] = None, item
    if decision == "headroom" and len(GENERATION["headroom_events"]) > n_ev:
        lines = [headroom_line(item, released, GENERATION["headroom_events"][-1])]
    else:
        lines = [graphs_line(item, released)] if released else []
    lines.append(graphgen_line(item, sig))
    return lines


def graphgen_install(model):
    """Install the generation decision on ``model`` as a forward pre-hook (upstream's fold() calls ``model(**features, …)``: the hook runs after the
    input is featurised and before the network allocates anything); returns the hook handle (``.remove()``), None when the object takes no hooks."""
    reg = getattr(model, "register_forward_pre_hook", None)
    if not callable(reg):
        return None

    def _hook(module, args, kwargs):
        for line in graphgen_observe(kwargs):
            sys.stderr.write(line + "\n")
        sys.stderr.flush()
        return None

    handles = [reg(_hook, with_kwargs=True)]
    for name in HEADROOM_PROBE_MODULES:                                        # the prologue probe: the first graphed region a fold enters
        sub = getattr(model, name, None)
        sreg = getattr(sub, "register_forward_pre_hook", None)
        if callable(sreg):
            handles.append(sreg(lambda _m, _a: headroom_probe()))
    return _Handles(handles)


HEADROOM_PROBE_MODULES = ("lm_encoder", "msa_encoder", "folding_trunk")     # the graph levers' first regions inside _run_one_loop, in call order


class _Handles:
    """The graphgen hook set as one handle (``.remove()`` removes every hook)."""
    def __init__(self, handles):
        self.handles = list(handles)

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self.handles = []


def cache_line(item_id, seed: int, before: dict | None, after: dict | None) -> str:
    """One line per timed fold: the kit caches' traffic INSIDE the timed window (counter deltas across builder.fold()) —
    ``CACHE item=<id> seed=<s> window=fold feature_cache_hits=<n> feature_cache_stores=<n> esmc_cache_hits=<n> esmc_cache_stores=<n> scope=<word>``;
    a pass over distinct inputs at one seed shows 0 hits and 1 store per cache per fold. ``caches=absent`` on the stock arm."""
    if before is None or after is None:
        return f"{PREFIX} CACHE item={item_id} seed={_seed_word(seed)} window=fold caches=absent"
    d = " ".join(f"{k}={int(after.get(k, 0)) - int(before.get(k, 0))}" for k in CACHE_KEYS)
    return f"{PREFIX} CACHE item={item_id} seed={_seed_word(seed)} window=fold {d} scope={cache_scope()}"


LEVERFOLD_KEYS = {                                                               # (module, counter) pairs whose per-fold deltas the LEVERFOLD line carries: the data-dependent engagements
    "ef2_msa_v2": ("mh_fills", "mh_hits", "mh_subsample_rows", "m15_fallthrough", "m16_fallthrough", "m17_fallthrough"),
    "ef2_opt": ("recycle_graph_captures", "recycle_graph_replays", "recycle_eager", "loop_static_calls", "graph_budget_eager_recycle", "sampler_graph_captures", "sampler_graph_clears_lru"),
    "ef2_dit": ("rollout_folds", "rollout_fallback", "roll_captures", "roll_replays", "roll_eager_boundaries", "dit_steps", "dit_fallback", "kabsch_device"),
    "ef2_atom": ("a2_prefix_folds", "a2_nonprefix_folds", "a5_rowprefix_folds", "a5_nonprefix_folds", "a5_layout_fallbacks", "a5_static_reregistered", "a6_sorted_folds", "a6_unsorted_folds"),
    "ef2_hoist": ("trimul_calls", "trimul_fallthrough", "glue_calls", "glue_fallthrough", "disto_swapped", "disto_waits"),
    "ef2_pair_v2": ("t15_calls", "t15_msa_pair_transition_calls", "t15_fallthrough", "xtr_calls", "xtr_fallthrough", "packs"),
    "ef2_transition_cute": ("face_calls", "face_refused", "transition_calls", "pair_transition_calls", "fallthrough_transition", "fallthrough_pair_transition"),
    "ef2_xte": ("xte_calls", "xte_fallthrough"),
    "ef2_w4": ("tx_calls", "tx_refused", "tx_class_stock", "tx_upstream_calls"),
    "ef2_feats": ("calls", "general_path", "fallback_non_ascii", "pair_calls", "pair_selfcheck_identical", "pair_selfcheck_mismatch", "pair_fallback"),
}


def leverfold_counts() -> dict | None:
    """Snapshot of the LEVERFOLD_KEYS counters of every loaded lever module ({"<module>.<key>": int}); None when none of them is loaded (the stock arm,
    a kit arm before configure)."""
    out = {}
    for mod, keys in LEVERFOLD_KEYS.items():
        m = sys.modules.get(mod)
        if m is None:
            continue
        st = getattr(m, "STATS", None)
        if st is None:
            continue
        for k in keys:
            try:
                out[f"{mod}.{k}"] = int(st.get(k, 0) or 0)
            except (TypeError, ValueError):
                continue
    return out or None


def leverfold_line(item_id, seed, before: dict | None, after: dict | None) -> str | None:
    """One line per timed fold with the data-dependent engagements INSIDE the window (counter deltas across builder.fold(), like the CACHE line):
    ``LEVERFOLD item=<id> seed=<s> <module>.<counter>=<delta> ...`` for every LEVERFOLD_KEYS counter of the loaded modules — e.g. ``ef2_msa_v2.mh_hits=20``
    (the MSA hoist served this fold's recycles) or ``=0`` with ``mh_subsample_rows=21`` (rows were subsampled: all recycles computed), ``ef2_opt.recycle_graph_replays``
    (the recycle graph engaged at this shape), ``ef2_dit.rollout_folds=1`` / ``rollout_fallback``, the atom layouts (``a5_nonprefix_folds``), the hoists'
    fall-through counters. None on the stock arm (no module loaded)."""
    if before is None and after is None:
        return None
    before, after = before or {}, after or {}
    keys = sorted(set(before) | set(after))
    d = " ".join(f"{k}={int(after.get(k, 0)) - int(before.get(k, 0))}" for k in keys)
    return f"{PREFIX} LEVERFOLD item={item_id} seed={_seed_word(seed)} window=fold {d}"


def tally_fields(stats: dict) -> list[str]:
    out = []
    for mod in ("ef2_opt", "ef2_w4", "ef2_msa", "ef2_mk_sampler", "ef2_xl") + LEVER_MODULES_V2 + ("rowchunk",):
        d = stats.get(mod)
        if isinstance(d, dict):
            items = sorted((str(k), v) for k, v in d.items() if not isinstance(v, (dict, list)))
            out.append(f"{mod}={{" + ",".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in items) + "}")   # every scalar counter the module holds (sorted by name; seconds to the ms)
    inv = stats.get("graph_invariant")
    if isinstance(inv, dict):
        out.append(f"graph_invariant={'ok' if inv.get('ok') else 'VIOLATED'}(pops={inv.get('individual_graph_pops')},generations={inv.get('graph_generations')})")
    return out


def exit_tally_line(pid: int | None = None) -> str:
    pid = os.getpid() if pid is None else pid
    stats = memory_stats()
    line = (f"{PREFIX} EXIT pid={pid} {n_gpu_fields()} source=memory " + " ".join(tally_fields(stats)) if stats
            else f"{PREFIX} EXIT pid={pid} {n_gpu_fields()} no lever counters: the kit's lever modules were never loaded in this process")
    try:                                                                            # a partial application is never silent
        from . import stack
        st = stack.status() or {}
        if st.get("partial") or st.get("levers_fallback"):
            line += f" PARTIAL fallbacks={','.join(st.get('levers_fallback') or []) or 'none'}"
    except Exception:  # noqa: BLE001
        pass
    return line


def register_exit_tally() -> None:
    """Print the exit tally at interpreter exit (once per process) — opt_core.report.register_exit_tally with this kit's whole line."""
    from opt_core.report import register_exit_tally as _register
    _register(TAG, exit_tally_line)
