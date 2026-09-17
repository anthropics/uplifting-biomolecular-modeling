"""fpf_clisampler.policy — the sampler-lever admission policy of every mode row: per item, the route of the diffusion sampler is the first
of GRAPH+HOIST -> HOIST_EAGER -> STOCK whose PROJECTED sampler-phase peak fits the row's limit; the levers' memory is RELEASED at the sampler's exit
(graph entries, private pool, the hoist's slots, its padding-bias cache) so nothing rides into the confidence head or the next item's trunk.

Why (H100 80 GB; the constants below are the fitted memory model this policy USES):
  * the DiT hoist carries almost all of the sampler levers' gain at large N; the CUDA graph without the hoist buys nothing there;
  * what the routes cost, as the sampler phase's peak above what is allocated at its entry (ADMIT_MODEL):
    hoist H(N) = 3.5e-6*N^2 of slots + the eager record step's transient (2e-7*N^2 where the DiT attention levers run, 1e-6*N^2 above their 2560-token
    envelope): 8 / 15 / 24 / 41 GiB; graph G = H + 6.2e-6*N^2 (private pool, static copies, capture; sized to the pool-chained build); the stock
    sampler's own transient T ~ 3.2e-6*N^2 (stock attention path; fitted at 3000 tokens: 28.5 GiB above entry once the fused stack's block biases are per block);
  * what sets the item's peak WITHOUT the extras: the trunk phase (measured per item) or the confidence head — allocated + 5.35e-6*N^2 while every
    sample's pae/pde stays on the device (N <= 2000: 15.9 GiB @1400, 29.8 @2000), allocated + 2.9e-6*N^2 above (stock offloads them per sample:
    30.6 @2530, 41.9 @3000; the trunk's 35.1 / 45.4 stands there).
Routes (ALL exact-by-construction w.r.t. the stock sampler: graph replay = the same kernels in the same order with the RNG drawn outside in stock order;
the hoist records the step-invariant tensors at step 0 and reuses them; the bypass IS the stock sampler):
  GRAPH        graphed denoiser step + hoist                  when allocated + G <= limit (and N_token <= PTX_SAMPLER_GRAPH_MAXTOK when a cap is exported)
  HOIST_EAGER  stock loop, hoist driven by its eager protocol  when allocated + H <= limit (PTX_SAMPLER_HOIST_EAGER=1 and a hoist is bound)
  STOCK        the stock eager sampler                         otherwise
  limit by word:  memory (exact, fast) = 0.85 x device total - 1 GiB: the speed rows spend free memory on the sampler;
                  item (big)         = the item's own peak WITHOUT the extras + 1 GiB = max(trunk-phase peak [measured], the confidence head's projected
                                         phase peak [N_token <= 2560, its unchunked regime], the stock sampler's projected phase peak) + 1: the memory
                                         row's printed item peak is never raised for speed.
Lines (this module prints them; nothing is decided from a caught exception):
  `[fpf_clisampler] ADMIT item N_token=… N_atom=…: GRAPH|HOIST_EAGER|STOCK — <projection vs limit, basis> [rule=…; allocated at entry …; model GiB: …; trunk-phase peak …; device …]`
  `[fpf_clisampler] SAMPLER-MEM item N_token=… route=… admit_ok | ADMIT_MISS: measured sampler-phase peak P > limit L + 0.5 GiB limit_gib=… trunk_peak_gib=…
   alloc_at_entry_gib=… projected_peak_gib=… item_peak_after_sampler_gib=… alloc_at_exit_gib=… sampler_wall_s=… released=n g2_cleared=n | kept(…) alloc_after_release_gib=…`
  and the process SUMMARY carries admit=graph:n,hoist_eager:n,stock:n releases=n misses=n.
SWITCHES — internal composition words set by protenix_opt.modes per mode row (exact/fast: admit=memory release=reach hoist_eager=1; big: admit=item
release=item hoist_eager=1); any other value raises ValueError naming the variable at import; unset = the kit-cap behaviour of the graphed sampler:
  PTX_SAMPLER_ADMIT=item|memory   PTX_SAMPLER_RELEASE=item|reach   PTX_SAMPLER_HOIST_EAGER=0|1
  release=item (big): everything after every item; release=reach (exact/fast): only what was admitted above the graph envelope (a HOIST_EAGER
  route's slots; a GRAPH entry only above the cap / REACH_FLOOR_TOKENS) — inside the envelope the cached graph stays and a same-shape next item
  (another seed, a duplicate size) replays it with no capture.
API for the graphed loop (infopt_graphs' lever sampler_reach calls it for items above its REACH_FLOOR_TOKENS — install()'s wrapper steps aside for exactly
those items; a loop that drives every item declares `drives_admission = True` on the class; either way ONE decision per item — and calls
admit() -> [its graph path | run_hoist_eager(loop, stock_fn, ...) | run_stock(loop, stock_fn, ...)] -> release() -> note_measured(); otherwise install()
wraps loop.sample and drives the same sequence over the loop's BYPASS branch): project(), limit(), admit(), note_measured(), release(), run_hoist_eager(),
run_stock(), install(), summary(). Units: GiB = 2^30, floats.
"""
from __future__ import annotations
import os, sys, time
from typing import Optional
from collections import namedtuple
import torch

POLICY_VERSION = "3"                                      # 3: stock-sampler transient 3.2e-6*N^2; the stock route names an over-limit peak (STOCK_OVER_LIMIT); SAMPLER-MEM carries the item's N_token / N_atom. 2: admission terms refit on the kit with the DiT attention levers (>= 0.3.19) — see ADMIT_MODEL
ADMIT_MODEL = {                                           # GiB; the ONE place for every constant of the rule; each is printed on the ADMIT line
    # sampler-phase peak above allocated-at-entry, measured on H100 80 GB with the kit's own SAMPLER-MEM / confidence probes, N_sample 5, items
    # fitted on multi-chain items at 1400-3000 tokens (11k-24k atoms); no atom-count dependence was measurable:
    "hoist_static_n2": 3.5e-6,                             #   the hoist's slots (tok pair-bias x24, tok z-norm, pair_z clone, cond/enc): 7.0 / 14.0 / 22.2 / 31.1 measured at exit
    "hoist_trans_n2": 2.0e-7, "hoist_trans_hi_n2": 1.0e-6, #   + the eager record step's transient above them: 1.0 GiB at 2530 where the DiT attention levers run (<= dit_maxtok),
    "dit_maxtok": 2560,                                    #     9.1 GiB at 3000 above their envelope (the stock attention path) -> peak 23.2 @2530, 40.2 @3000
    "hoist_atom": 0.0,                                     #   (kept for the record: no per-atom term at these sizes)
    "graph_n2": 6.2e-6,                                    #   graph route adds private pool + static copies + capture: 10.0 / 19.9 / 31.4 (5.1e-6 N^2) without pool chaining,
                                                           #   24.9 @2000 with the pool-chained build (sampler_prep): the constant covers the larger build
    "tstock_n2": 3.2e-6,                                   #   the STOCK sampler's own transient on the stock attention path: 5.8 @1400, 48.2 @4000 (an upper bound where the DiT levers run)
    "conf_n2": 5.35e-6, "conf_hi_n2": 2.9e-6, "conf_alltok": 2000,   # the confidence head above allocated: 10.45 @1400, 21.3 @2000 (all samples\' pae/pde on the device: N <= 2000),
                                                           #   18.5 @2530, 25.9 @3000 (stock offloads pae/pde per sample above 2000 tokens: confidence.py)
    "slack_gib": 1.0, "miss_gib": 0.5,                     # admit when projected <= limit (limit carries the 1 GiB slack); ADMIT_MISS when measured > limit + miss
    "reach_floor_tokens": 1536,                            #   the graph envelope when no cap is exported (= infopt_graphs sampler_prep REACH_FLOOR_TOKENS): release=reach keeps entries at or below it
    "headroom": 0.85, "reserve_gib": 1.0,                  # word memory (exact / fast rows): limit = headroom x device - reserve
    "n_sample_fit": 5,
}
ROUTES = ("graph", "hoist_eager", "stock")
ADMIT_VALUES, RELEASE_VALUES, FLAG_VALUES = ("", "item", "memory"), ("", "item", "reach"), ("", "0", "1")
WORDS = {"admit": (os.environ.get("PTX_SAMPLER_ADMIT", "") or "").strip().lower(),
         "release": (os.environ.get("PTX_SAMPLER_RELEASE", "") or "").strip().lower(),
         "hoist_eager": os.environ.get("PTX_SAMPLER_HOIST_EAGER", "0") not in ("", "0")}
for _k, _env, _vals in (("admit", "PTX_SAMPLER_ADMIT", ADMIT_VALUES), ("release", "PTX_SAMPLER_RELEASE", RELEASE_VALUES)):
    if WORDS[_k] not in _vals:                              # a word the mode rows never set: refused by name (protenix_opt.modes composes these; no user setting)
        raise ValueError(f"{_env}={os.environ.get(_env)!r} is not one of {'|'.join(v or 'unset' for v in _vals)}")
if os.environ.get("PTX_SAMPLER_HOIST_EAGER", "0") not in FLAG_VALUES:
    raise ValueError(f"PTX_SAMPLER_HOIST_EAGER={os.environ.get('PTX_SAMPLER_HOIST_EAGER')!r} is not 0|1")
ST = {"installed": False, "verdicts": {"graph": 0, "hoist_eager": 0, "stock": 0}, "releases": 0, "released_gib": 0.0, "misses": 0, "items": []}
Admission = namedtuple("Admission", "route limit_gib projected_gib why terms")


def _say(msg):
    print(f"[fpf_clisampler] {msg}", file=sys.stderr, flush=True)


def active() -> bool:
    return WORDS["admit"] in ("item", "memory") or WORDS["release"] in ("item", "reach") or WORDS["hoist_eager"]


def words() -> str:
    """The SAMPLER line's policy token: admit=item|memory|cap release=item|retain hoist_eager=1|0."""
    return f"admit={WORDS['admit'] or 'cap'} release={WORDS['release'] or 'retain'} hoist_eager={int(WORDS['hoist_eager'])}"


def project(n_tok: int, n_atom: int, allocated_gib: float, n_sample: int = 5, m=ADMIT_MODEL) -> dict:
    """Projected sampler-phase peaks (GiB) per route from the fitted constants (fits at N_sample 5, the model's setting; n_sample is recorded only)."""
    n2 = float(n_tok) * float(n_tok)
    Hs = m["hoist_static_n2"] * n2
    Ht = (m["hoist_trans_n2"] if n_tok <= m["dit_maxtok"] else m["hoist_trans_hi_n2"]) * n2
    H = Hs + Ht + m["hoist_atom"] * n_atom
    Gx = m["graph_n2"] * n2
    T = m["tstock_n2"] * n2
    Cx = (m["conf_n2"] if n_tok <= m["conf_alltok"] else m["conf_hi_n2"]) * n2
    return {"graph": allocated_gib + H + Gx, "hoist_eager": allocated_gib + H, "stock": allocated_gib + T,
            "need_hoist": H, "hoist_statics": Hs, "hoist_transient": Ht, "graph_extra": Gx, "stock_transient": T, "conf_extra": Cx,
            "conf_proj": allocated_gib + Cx, "allocated": allocated_gib, "n_sample": n_sample}


def limit(word: str, *, n_tok: int, allocated_gib: float, trunk_peak_gib: float, total_gib: float, m=ADMIT_MODEL):
    """-> (limit_gib, basis): the row's ceiling for the projected sampler-phase peak."""
    if word == "memory":
        return m["headroom"] * total_gib - m["reserve_gib"], f"{m['headroom']:.2f} x device {total_gib:.1f} - reserve {m['reserve_gib']:.1f}"
    if word == "item":
        p = project(n_tok, 0, allocated_gib, m=m)
        ref = max(trunk_peak_gib, p["conf_proj"], p["stock"])
        return ref + m["slack_gib"], (f"item peak without extras {ref:.1f} = max(trunk {trunk_peak_gib:.1f}, confidence proj {p['conf_proj']:.1f}, "
                                     f"stock-sampler proj {p['stock']:.1f}) + slack {m['slack_gib']:.1f}")
    raise ValueError(f"admission word {word!r} is not item|memory")


def admit(*, n_tok: int, n_atom: int, allocated_gib: float, trunk_peak_gib: float, total_gib: float, word=None, candidates=ROUTES, graph_cap: int = 0,
          graph_peak_gib=None, hoist_bound: bool = True, n_sample: int = 5, say: bool = True, m=ADMIT_MODEL) -> Admission:
    """The admission rule. candidates: the routes the caller can run, in preference order ('stock' is always admissible); graph_cap: a nonzero
    PTX_SAMPLER_GRAPH_MAXTOK still bounds GRAPH; graph_peak_gib: a caller-supplied projected sampler-phase peak for GRAPH+HOIST (overrides the fitted
    graph term). word=None -> the row's PTX_SAMPLER_ADMIT ('' = no admission rule: GRAPH under the cap, else STOCK). Prints the ADMIT line (say)."""
    word = WORDS["admit"] if word is None else word
    t = project(n_tok, n_atom, allocated_gib, n_sample, m)
    if graph_peak_gib is not None:
        t["graph"] = float(graph_peak_gib); t["graph_from_caller"] = True
    t["trunk_peak"] = trunk_peak_gib; t["total"] = total_gib; t["word"] = word; t["n_tok"] = int(n_tok); t["n_atom"] = int(n_atom)
    cands = [c for c in candidates if c in ROUTES]
    if not (hoist_bound and WORDS["hoist_eager"]):
        cands = [c for c in cands if c != "hoist_eager"]
    cap_ok = not (graph_cap and n_tok > graph_cap)
    if not cap_ok:
        cands = [c for c in cands if c != "graph"]
    gtag = "graph+hoist" + (" (caller proj)" if t.get("graph_from_caller") else "")
    if word not in ("item", "memory"):                                       # no admission rule: the kit-cap behaviour
        t["limit"] = float("nan")
        route = "graph" if "graph" in cands else "stock"
        why = "kit cap only (no admission rule)" if route == "graph" else (f"size: N_token {n_tok} > PTX_SAMPLER_GRAPH_MAXTOK {graph_cap}" if not cap_ok else "no graph candidate") + " (no admission rule)"
    else:
        lim, basis = limit(word, n_tok=n_tok, allocated_gib=allocated_gib, trunk_peak_gib=trunk_peak_gib, total_gib=total_gib, m=m)
        t["limit"] = lim
        route, why = "stock", None
        for c in cands:
            if c == "stock":
                break
            if t[c] <= lim:
                route = c
                why = f"projected sampler peak with {gtag if c == 'graph' else 'the hoist'} {t[c]:.1f} <= {lim:.1f} GiB ({basis})"
                if c == "hoist_eager":
                    why += f"; {gtag} {t['graph']:.1f} > {lim:.1f}" if ("graph" in cands) else (f"; graph: N_token {n_tok} > cap {graph_cap}" if not cap_ok else "; graph not a candidate")
                break
        if why is None:
            tried = " / ".join(f"{t[c]:.1f} ({'graph+hoist' if c == 'graph' else 'hoist'})" for c in cands if c != "stock") or "no lever route available"
            why = f"projected sampler peak {tried} > {lim:.1f} GiB ({basis})" + ("" if hoist_bound else " (no hoist bound)")
    ST["verdicts"][route] += 1
    adm = Admission(route, t["limit"], t.get(route, float("nan")) if route != "stock" else t["stock"], why, t)
    if say:
        _say(f"ADMIT item N_token={n_tok} N_atom={n_atom}: {route.upper()} — {why} [rule={word or 'none'}; allocated at entry {allocated_gib:.1f}; model v{POLICY_VERSION} GiB: "
             f"hoist slots {t['hoist_statics']:.1f} (= {m['hoist_static_n2']:g}*N^2) + record transient {t['hoist_transient']:.1f} "
             f"(= {(m['hoist_trans_n2'] if n_tok <= m['dit_maxtok'] else m['hoist_trans_hi_n2']):g}*N^2, {'<=' if n_tok <= m['dit_maxtok'] else '>'} {m['dit_maxtok']} tok), "
             f"graph pool+copies {t['graph_extra']:.1f} (= {m['graph_n2']:g}*N^2), stock-sampler transient {t['stock_transient']:.1f} (= {m['tstock_n2']:g}*N^2), "
             f"confidence proj {t['conf_proj']:.1f} (= allocated + {(m['conf_n2'] if n_tok <= m['conf_alltok'] else m['conf_hi_n2']):g}*N^2, N {'<=' if n_tok <= m['conf_alltok'] else '>'} {m['conf_alltok']}); "
             f"trunk-phase peak {trunk_peak_gib:.1f}; device {total_gib:.1f}]")
    return adm


def note_measured(adm: Admission, measured_peak_gib: float, *, n_tok: int = 0, n_atom: int = 0, alloc_at_exit_gib=None, sampler_wall_s=None,
                  released: str = "", alloc_after_release_gib=None, say: bool = True, m=ADMIT_MODEL) -> dict:
    """The audit after the sampler: measured = the item's allocator peak so far (above the trunk peak only if the sampler set it). Prints SAMPLER-MEM
    with admit_ok, or ADMIT_MISS in words when the measured peak exceeds the admitted limit by more than miss_gib. Returns the record."""
    lim = adm.limit_gib
    n_tok = n_tok or int(adm.terms.get("n_tok") or 0); n_atom = n_atom or int(adm.terms.get("n_atom") or 0)   # the item admit() decided (callers that do not repeat them)
    over = lim == lim and measured_peak_gib > lim + m["miss_gib"]              # the measured peak exceeds the admitted limit -- on ANY route (the stock route is named too, never `admit_ok`)
    miss = adm.route != "stock" and over
    if miss: ST["misses"] += 1
    if over and adm.route == "stock": ST["stock_over"] = ST.get("stock_over", 0) + 1
    trunk = adm.terms.get("trunk_peak", float("nan")); a0 = adm.terms.get("allocated", float("nan"))
    rec = {"n_tok": n_tok, "n_atom": n_atom, "route": adm.route, "limit_gib": round(lim, 2) if lim == lim else None, "trunk_peak_gib": round(trunk, 2), "stock_over_limit": bool(over and adm.route == "stock"),
           "alloc_at_entry_gib": round(a0, 2), "projected_peak_gib": round(adm.projected_gib, 2), "item_peak_after_sampler_gib": round(measured_peak_gib, 2),
           "admit_miss": bool(miss), "sampler_wall_s": None if sampler_wall_s is None else round(sampler_wall_s, 2), "released": released}
    ST["items"].append(rec); del ST["items"][:-16]
    if say:
        _say(f"SAMPLER-MEM item N_token={n_tok} N_atom={n_atom} route={adm.route} "
             f"{('ADMIT_MISS: measured sampler-phase peak %.2f > limit %.2f + %.1f GiB' % (measured_peak_gib, lim, m['miss_gib'])) if miss else (('STOCK_OVER_LIMIT: measured sampler-phase peak %.2f > limit %.2f + %.1f GiB on the stock route' % (measured_peak_gib, lim, m['miss_gib'])) if over else 'admit_ok')} "
             f"limit_gib={lim:.2f} trunk_peak_gib={trunk:.2f} alloc_at_entry_gib={a0:.2f} projected_peak_gib={adm.projected_gib:.2f} item_peak_after_sampler_gib={measured_peak_gib:.2f} "
             f"({'sampler above trunk peak' if measured_peak_gib > trunk + 0.01 else 'trunk peak stands'})"
             + ("" if alloc_at_exit_gib is None else f" alloc_at_exit_gib={alloc_at_exit_gib:.2f}")
             + ("" if sampler_wall_s is None else f" sampler_wall_s={sampler_wall_s:.2f}")
             + (f" {released}" if released else " kept(inside the graph envelope: entry + pool stay for the next item)") + ("" if alloc_after_release_gib is None else f" alloc_after_release_gib={alloc_after_release_gib:.2f}"))
    return rec


def should_release(adm: Admission, n_tok: int, graph_cap: int = 0, word: Optional[str] = None) -> bool:
    """The row's release rule after an item's sampler. `item` (big, memory first): always — every entry, the pool, the hoist's slots, the padding-bias
    cache leave before the confidence head. `reach` (exact/fast, speed rows): only what was admitted ABOVE the graph envelope — a HOIST_EAGER route's
    slots always, a GRAPH entry only when N_token exceeds the envelope (the kit's graph cap, else REACH_FLOOR_TOKENS) — so that inside the envelope the
    cached graph and its pool stay for the next item (a second seed / a same-shape input replays with no capture; the pool policy of sampler_prep governs
    there). '' : never."""
    word = WORDS["release"] if word is None else word
    if word == "item":
        return True
    if word != "reach":
        return False
    if adm.route == "hoist_eager":
        return True
    envelope = int(graph_cap) if graph_cap else int(ADMIT_MODEL["reach_floor_tokens"])
    return adm.route == "graph" and int(n_tok) > envelope


def release(loop) -> str:
    """Return the sampler levers' memory to the device at the sampler's exit: evict every graph entry (graph + private pool + static copies:
    graphed.py's `_evict_oldest` until `order` is empty; a pool-chained previous graph `_prep_chain` too), drop the hoist's slots (dit_hoist keeps
    `H.cur` bound to the last entry's slot dict — an eviction outside a new bind must release it) and its G2 padding-bias cache (`H._padbias`, a pure
    per-atom-count cache, up to 8 shapes), then empty the allocator cache. -> 'released=n g2_cleared=m'."""
    n = 0
    while getattr(loop, "order", None):
        loop._evict_oldest(); n += 1
    try: loop.entries.clear()
    except Exception: pass
    if getattr(loop, "_prep_chain", None) is not None:                       # sampler_prep[pool_chain]: the previous graph kept alive for its pool
        loop._prep_chain = None; n += 1
    H = getattr(loop, "biascache", None)
    g2 = 0
    if H is not None and hasattr(H, "release"):
        H.release()
        pb = getattr(H, "_padbias", None)
        if isinstance(pb, dict):
            g2 = len(pb); pb.clear()
    alloc1 = torch.cuda.memory_allocated()
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    try: loop.ec_guard.flush_if_safe()
    except Exception: pass
    ST["releases"] += 1; ST["released_gib"] += max(0.0, (alloc1 - torch.cuda.memory_allocated()) / 2**30)
    return f"released={n} g2_cleared={g2}"


def run_hoist_eager(loop, stock_fn, /, *a, **k):
    """The HOIST_EAGER executor: brackets a stock-sampler callable with dit_hoist's eager protocol (bind per x signature, record at step 0, hit
    after) and returns stock_fn(*a, **k). loop and stock_fn are positional-only (the graphed loop passes a `stock_fn=` keyword THROUGH to its own
    sample); nothing is injected into the arguments."""
    H = getattr(loop, "biascache", None)
    if H is not None:
        H._eager_driving = True; H._x_shape = None; H.step = -1
    try:
        return stock_fn(*a, **k)
    finally:
        if H is not None:
            H._eager_driving = False; H.set_mode("off")


def run_stock(loop, stock_fn, /, *a, **k):
    """The STOCK route: the stock sampler callable as is (the hoist stays off there)."""
    return stock_fn(*a, **k)


def _bypass(loop, cls_sample):
    """-> a stock-sampler callable for install()'s wrapper: the graphed loop's own BYPASS branch (N_token > loop.max_tokens -> the stock
    sample_diffusion), entered by lowering max_tokens around the call."""
    def fn(denoise_net, input_feature_dict, s_inputs, *a, **k):
        saved = loop.max_tokens
        loop.max_tokens = max(1, int(s_inputs.shape[-2]) - 1)
        try:
            return cls_sample(loop, denoise_net, input_feature_dict, s_inputs, *a, **k)
        finally:
            loop.max_tokens = saved
    return fn


def install(loop, hoist=None) -> str:
    """Wiring from clisampler._install_on_model: wrap this GraphedDenoiseLoop instance's `sample` and drive admit -> route -> release -> note_measured,
    unless the loop class declares `drives_admission = True` (it then calls this module's functions itself: single call site either way)."""
    if not active():
        return "policy:off(" + words() + ")"
    if getattr(type(loop), "drives_admission", False):
        ST["installed"] = True
        return "policy:on(" + words() + "; driven by the graphed loop)"
    if getattr(loop, "_fpf_policy", False):
        return "policy:already"
    cls_sample = type(loop).sample
    reach_floor = None                                                        # lever sampler_reach (infopt_graphs sampler_prep, PTX_SAMPLER_REACH=1): above its floor the graphed loop
    prep = getattr(loop, "prep", None)                                        # itself calls admit / run_hoist_eager / release / note_measured — this wrapper steps aside for those items
    if prep is not None and getattr(prep, "reach", False):
        from infopt_graphs.protenix import sampler_prep as _sp
        reach_floor = int(_sp.REACH_FLOOR_TOKENS)

    def sample(denoise_net, input_feature_dict, s_inputs, *a, **k):
        n_tok = int(s_inputs.shape[-2])
        if reach_floor is not None and n_tok > reach_floor:                  # sampler_reach decides, releases and audits this item through this module's API (one decision per item)
            return cls_sample(loop, denoise_net, input_feature_dict, s_inputs, *a, **k)
        try: n_atom = int(input_feature_dict["atom_to_token_idx"].size(-1))
        except Exception: n_atom = 0
        try: n_sample = int(k.get("N_sample", 5))
        except Exception: n_sample = 5
        dev = torch.cuda.current_device(); total = torch.cuda.get_device_properties(dev).total_memory / 2**30
        alloc0 = torch.cuda.memory_allocated() / 2**30
        trunk_peak = torch.cuda.max_memory_allocated() / 2**30            # the item's allocator peak so far = its trunk phase (phase_timing / the memory row open the window at the item's start; never reset here)
        H = getattr(loop, "biascache", None)
        adm = admit(n_tok=n_tok, n_atom=n_atom, allocated_gib=alloc0, trunk_peak_gib=trunk_peak, total_gib=total, graph_cap=int(getattr(loop, "max_tokens", 0) or 0),
                    hoist_bound=H is not None, n_sample=n_sample)
        t0 = time.time()
        try:
            if adm.route == "graph":
                return cls_sample(loop, denoise_net, input_feature_dict, s_inputs, *a, **k)
            if adm.route == "hoist_eager":
                return run_hoist_eager(loop, _bypass(loop, cls_sample), denoise_net, input_feature_dict, s_inputs, *a, **k)
            return run_stock(loop, _bypass(loop, cls_sample), denoise_net, input_feature_dict, s_inputs, *a, **k)
        finally:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 2**30; alloc1 = torch.cuda.memory_allocated() / 2**30
            rel = release(loop) if should_release(adm, n_tok, graph_cap=int(getattr(loop, "max_tokens", 0) or 0)) else ""
            note_measured(adm, peak, n_tok=n_tok, n_atom=n_atom, alloc_at_exit_gib=alloc1, sampler_wall_s=time.time() - t0, released=rel,
                          alloc_after_release_gib=(torch.cuda.memory_allocated() / 2**30) if rel else None)
    loop.sample = sample                                                     # instance attribute: graphed_sample_diffusion resolves loop.sample per call
    loop._fpf_policy = True
    ST["installed"] = True; ST["reach_floor"] = reach_floor
    return "policy:on(" + words() + (f"; items above {reach_floor} tokens admitted by sampler_reach" if reach_floor is not None else "") + ")"


def summary() -> dict:
    return {"words": words(), "verdicts": dict(ST["verdicts"]), "releases": ST["releases"], "released_gib": round(ST["released_gib"], 2),
            "admit_misses": ST["misses"], "items": ST["items"][-8:], "model": dict(ADMIT_MODEL)}
