"""fpf_stackgraph.stackgraph — layer G ("PTX_BLK_GRAPH"): CUDA-graph replay of the Protenix-v2 48-block PairformerStack for SMALL designs
(host-bound regime, N_token <= PTX_BLK_GRAPH_MAXTOK, default 448) in the stock `protenix pred` CLI / FlashPairformer kit path.  EXACT by construction
and by test record: a CUDA graph replays the SAME kernels with the SAME launch parameters on static buffers, so the only ways to differ from eager are
(1) a kernel that does not launch on the capture stream (Protenix fast_layernorm launches on the legacy default stream -> this REQUIRES the
stream-correct prebuilt LN, bitwise-equal in-process, else the lever stays off), (2) host-side data-dependent control flow inside the region (none in
PairformerStack.forward at inference: no .item(), no shape-dependent branches once N_token is fixed; asserted by the check step), (3) RNG (none in the
trunk at eval).  Protocol per (signature): eager reference call (its result is what the caller gets for that call) -> warm-up on a side stream -> capture
on static clones -> replay once on the same inputs -> torch.equal(s), torch.equal(z) vs the eager reference -> ARM; any mismatch/exception -> REFUSE the
signature forever (eager) and count it.  Replays serve every later call with the same signature: across recycles (9 of 10), across SEEDS of the same
design (all 10 cycles of seeds 2..5) and across designs of equal N_token in the same process.  Signature = (N_token, shapes/dtypes/strides of s and z,
autocast state, triangle_* flags, inplace/chunk/mask flags, id(model stack)).  Memory discipline: at most PTX_BLK_GRAPH_MAX (default 8; the stock CLI loops seeds OUTER / designs inner) live graphs (LRU;
graph private pools are freed on eviction), capture skipped when free memory < 2x the eager peak of one stack call (measured during the reference call).
Output handling: stock returns fresh (s, z); we return the graph's static outputs *without* cloning when PTX_BLK_GRAPH_NOCLONE=1 (safe in Protenix's
recycle loop: s,z are consumed by layernorm_z_cycle / layernorm_s before the next replay overwrites them, and the final-cycle outputs are consumed by the
diffusion/confidence modules before the next seed's first replay) — default is to clone (two N x N x 512-byte tensors) because 'safe by inspection'
is not 'safe by construction'.
Protocol: the capture-and-check discipline of infopt_graphs; this module adds cross-seed /
cross-design replay, LRU + memory guard, and the CLI/kit wiring (sitecustomize hook, PTX_LEVER_REPORT counters).
Counters (PTX_LEVER_REPORT line, key 'stackgraph'): installed, ln, captures, replays, eager, refused, check_fail_maxabs, evicted, oom_skips, sigs, capture_s.
PLAN-FOR-CAPTURE. The eager reference call, the side-stream warm-up and the capture of a first-sighted signature run inside
ONE planning window (planning() is True; counter planned_refs, printed in the SUMMARY line): a lever inside the stack whose provider keys its cells by timing column (eager | graph:
ptx_transition_core over opt_core.kernels.transition, protenix_opt.apb_core's pf_attn over opt_core.kernels.apb) selects the GRAPH column for all three under a TIER word
(fast | big), so the bitwise oracle and the graph serve the same rows; served different columns, the replay check would fail, the
signature would be refused and the whole trunk would run eager. The exact word is untouched by construction (every admissible exact row equals the
statement bitwise, so the oracle holds whichever column answers).
Process-global data_ptr-keyed lever caches (fpf_triatt_epi._WT_CACHE) cleared right before/after capture so every cached tensor the graph reads is produced inside the graph by construction (counter ptr_cache_clears).
OPT-IN second region PTX_BLK_GRAPH_TEMPL=1 = TemplateEmbedder's 2-block c=64 PairformerStack (s=None, pair_mask static input); same capture-and-check protocol; counters ST['templ']; default v0.6 behaviour unchanged when unset. MSA-module pair blocks are never captured (trunk-graphs job D: check2 refuses 12/12 in situ).
PTX_BLK_GRAPH_MEM_GB is reconciled with the XL policy ('' | auto=0.30*total | x GB | 0/off; honoured in fixed-count mode too; SUMMARY prints mode/max_sig/budget/both eager counters).
PTX_BLK_GRAPH_MAX=auto — the live-graph bound is no longer a count but a MEMORY BUDGET (PTX_BLK_GRAPH_MEM_GB; default = 25 % of the device memory free at the first eligible stack call, i.e. after model load): every distinct signature (padded N_token) seen is captured while the measured private-pool bytes fit the budget; when the next capture would not fit, that signature runs EAGER (exact) — counted budget_eager (and full_eager for the multisig verdict) — and NOTHING already captured is freed (the hazard #43/#57 class: never release a graph pool while another graph lives). Per-graph pool bytes are MEASURED (device-free delta across capture after empty_cache) and printed in the SUMMARY line (mode, budget_gb, pool_gb, distinct_ntok, budget_eager). An integer PTX_BLK_GRAPH_MAX keeps the v0.8 count policy unchanged.
HAZARD #43 — no destructive LRU eviction (cache full -> new signatures eager, counted 'full_eager'); every replay goes through _replay() which terminates the process (exit 70) on a CUDA fault instead of returning into Protenix's per-input catch-all (silent loss).
Failed-capture recovery = the shared fpf_trunkgraph/recovery.py (no local epilogues).
A failed capture => lever disabled for the process (fail-safe), capture_error_mode=thread_local; known incompatibility: host syncs inside the region (e.g. the timing run --fine per-module timers) make capture impossible -> run timing arms without --fine.
Warm-up + capture run under autocast(cache_enabled=False) — avoids the autocast weight-cast-cache address hazard (see the WEIGHT-CAST CACHE note below).
"""
from __future__ import annotations
from opt_core.oom import is_oom   # a broad handler that reroutes around a lever re-raises device out-of-memory first (opt_core.oom.is_oom)
import os, sys, time, json, atexit, collections
import torch

ST = {"installed": False, "why": None, "ln": None, "captures": 0, "replays": 0, "eager": 0, "eager_ineligible": 0, "refused": 0,
      "check_fail_maxabs": [], "evicted": 0, "oom_skips": 0, "sigs": 0, "capture_s": 0.0, "check_s": 0.0, "max_tok": None, "max_sig": None,
      "noclone": None, "by_ntok": {}}
_MAX_RAW = os.environ.get("PTX_BLK_GRAPH_MAX", "8").strip().lower()
_AUTO = _MAX_RAW in ("auto", "mem", "budget")          # memory-budgeted admission instead of a live-graph count
_MAX_SIG = (1 << 30) if _AUTO else int(_MAX_RAW)     # the stock CLI iterates SEEDS OUTER, designs inner -> one live graph per distinct N_token in the input dir avoids re-captures (pool 0.26-0.57 GB each at 153-278 tok; memory guard skips capture when short)
_MAX_TOK = int(os.environ.get("PTX_BLK_GRAPH_MAXTOK", "448"))
_MEM_GB_ENV = os.environ.get("PTX_BLK_GRAPH_MEM_GB", "").strip()      # budget for the SUM of graph private pools (GB); empty -> 25 % of free device memory at the first eligible call (auto mode only)
_MEM_FRAC = float(os.environ.get("PTX_BLK_GRAPH_MEM_FRAC", "0.25"))
def _parse_mem_gb(v: str):
    """ONE admission function, both knobs (PTX_BLK_GRAPH_MAX, PTX_BLK_GRAPH_MEM_GB) honoured:
    PTX_BLK_GRAPH_MEM_GB = ''        -> no explicit budget (auto mode then uses PTX_BLK_GRAPH_MEM_FRAC=0.25 of FREE device memory at the first eligible call; fixed mode = count only)
                         = 'auto'    -> 30 % of TOTAL device memory (XL policy default; resolved lazily at the first eligible call)
                         = '0'|'off' -> no explicit budget
                         = <x>       -> x GB.   In fixed-count mode (PTX_BLK_GRAPH_MAX=<n>) an explicit budget is honoured IN ADDITION to the count."""
    v = (v or "").strip().lower()
    if v in ("", "0", "off", "none"): return None, None
    if v == "auto": return "auto", "auto(0.30*total)"
    try: return float(v) * 2**30, "env"
    except Exception: return None, None
_MEM_BYTES0, _MEM_SRC0 = _parse_mem_gb(_MEM_GB_ENV)
_BUDGET = {"bytes": (None if _MEM_BYTES0 in (None, "auto") else _MEM_BYTES0), "src": _MEM_SRC0, "pool_total": 0, "per_tok2_max": 0.0, "overshoot": 0, "lazy_auto": _MEM_BYTES0 == "auto", "explicit": _MEM_BYTES0 is not None}
_MIN_TOK = int(os.environ.get("PTX_BLK_GRAPH_MINTOK", "17"))
_NOCLONE = os.environ.get("PTX_BLK_GRAPH_NOCLONE", "0") == "1"
_VERBOSE = os.environ.get("PTX_BLK_GRAPH_VERBOSE", "0") == "1"
_TEMPL = os.environ.get("PTX_BLK_GRAPH_TEMPL", "0") == "1"      # OPT-IN: also capture the TemplateEmbedder 2-block c=64 stack (s=None, pair_mask given); separate counters under ST["templ"]
ST["max_tok"], ST["max_sig"], ST["noclone"] = _MAX_TOK, ("auto" if _AUTO else _MAX_SIG), _NOCLONE
ST["mode"] = "auto" if _AUTO else "fixed"; ST["budget_eager"] = 0; ST["pool_bytes_by_ntok"] = {}
ST["templ"] = {"enabled": _TEMPL, "captures": 0, "replays": 0, "eager": 0, "refused": 0}

ST["planned_refs"] = 0
# ---- plan-for-capture: the window in which THIS module takes a first-sighted signature's eager reference, warms it up and captures it.  Levers inside the
# stack whose provider cells are keyed by timing column read planning() and select the GRAPH column for the whole window under a tier word (fast | big), so the
# reference (the bitwise oracle of the check step) and the graph serve the same rows; outside the window (replays need no selection; refused / ineligible / cache-full
# signatures run eager) nothing changes.  One dict, set and cleared by _planning() only (cleared in `finally`: an exception in the reference call never leaves it set).
_PLAN = {"sig": None}


def planning() -> bool:
    """True while the eager reference / warm-up / capture of a signature this module is about to capture runs (the plan-for-capture window)."""
    return _PLAN["sig"] is not None


class _planning:
    """Context of ONE first-sighting: planning() is True inside, False after (also on an exception); counts ST['planned_refs']."""
    __slots__ = ("sig",)

    def __init__(self, sig):
        self.sig = sig

    def __enter__(self):
        _PLAN["sig"] = self.sig if self.sig is not None else "anon"; ST["planned_refs"] = ST.get("planned_refs", 0) + 1
        return self

    def __exit__(self, *exc):
        _PLAN["sig"] = None
        return False


def report() -> dict:
    d = dict(ST); d["check_fail_maxabs"] = list(ST["check_fail_maxabs"])[:8]; return d


def _log(msg):
    if _VERBOSE:
        print(f"[fpf_stackgraph] {msg}", file=sys.stderr, flush=True)


# Process-global caches in composed levers that are keyed by data_ptr and evict (=> re-point) at run time.  A graph that HITS such an entry bakes in an address it
# does not own; a later eviction frees it -> silent corruption (root cause of the MSA-sub-stack replay failure under BLK2: fpf_triatt_epi._WT_CACHE,
# 108 distinct Wo > cap 64).  For the whole 48-block stack the 96 fixed-order lookups provably all MISS during capture (cap 64 < 96), so every WoT the graph
# reads is produced inside the graph — but we make that hold BY CONSTRUCTION: clear these caches right before and right after capture (numerics-neutral:
# they hold transposed copies of constant weights).
_GLOBAL_PTR_CACHES = (("fpf_triatt_epi.epilogue", "_WT_CACHE"),)


def _clear_global_ptr_caches():
    n = 0
    for modname, attr in _GLOBAL_PTR_CACHES:
        try:
            mod = sys.modules.get(modname)
            if mod is None:
                continue
            c = getattr(mod, attr, None)
            if isinstance(c, dict) and c:
                n += len(c); c.clear()
        except Exception:
            pass
    ST["ptr_cache_clears"] = ST.get("ptr_cache_clears", 0) + n
    return n


def _copy_in(ent, s, z, m):
    if ent.s_in is not None and s is not None:
        ent.s_in.copy_(s)
    ent.z_in.copy_(z)
    if getattr(ent, "m_in", None) is not None and m is not None:
        ent.m_in.copy_(m)


def _eq_opt(a, b):
    if a is None or b is None:
        return (a is None) and (b is None)
    return bool(torch.equal(a, b))


def _maxabs_opt(a, b):
    if a is None or b is None:
        return None
    return float((a.float() - b.float()).abs().max())


class _Entry:
    __slots__ = ("graph", "s_in", "z_in", "m_in", "s_out", "z_out", "armed", "n_tok", "uses", "pool_bytes")


_CACHE: "collections.OrderedDict[tuple, _Entry]" = collections.OrderedDict()


def _tdesc(t):
    return None if t is None else (tuple(t.shape), t.dtype, t.stride(), t.device.index)


def _sig(stack, s, z, kw):
    return (id(stack), _tdesc(s), _tdesc(z), _tdesc(kw.get("pair_mask")), torch.is_autocast_enabled(),
            torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else None,
            kw.get("triangle_multiplicative"), kw.get("triangle_attention"), bool(kw.get("inplace_safe")), kw.get("chunk_size"))



def _publish_ln_report(LN, r) -> None:
    """Composition rule: whichever component installs the stream-correct LN FIRST leaves its
    checking report where a SECOND installer in the same process finds it (the kit worker's guard refuses graphs unless bitwise.all_bitwise_equal):
    (1) cached on the protenix LN module (read by the bundle's fastln_prebuilt / fastln_stream 'already installed' branches), (2) mirrored into the kit's
    infopt_graphs.protenix.FASTLN_REPORT (never overwriting an existing bitwise entry) so the guard passes even when the kit's own unpatched fastln_prebuilt shadows ours."""
    try:
        if r.get("bitwise") and not getattr(LN, "_infopt_fastln_report", None):
            LN._infopt_fastln_report = dict(r)
    except Exception:
        pass
    try:
        mod = sys.modules.get("infopt_graphs.protenix")
        if mod is None:
            import importlib
            mod = importlib.import_module("infopt_graphs.protenix")
        FR = getattr(mod, "FASTLN_REPORT", None)
        if isinstance(FR, dict) and not FR.get("bitwise") and r.get("bitwise"):
            FR.update({k: v for k, v in r.items() if k in ("installed", "build_s", "patched_launches", "source_sha256", "patched_source_sha256", "side_stream", "bitwise", "reason", "load_s")})
    except Exception:
        pass

def ensure_stream_ln() -> bool:
    """Install the stream-correct fast-LN (prebuilt .so preferred; source rebuild fallback); both check bitwise + side-stream + graph
    replay in THIS process before patching.  Returns True iff protenix's fast LN now launches on the current stream."""
    if os.environ.get("LAYERNORM_TYPE", "") != "fast_layernorm":
        ST["ln"] = "torch_layernorm(no patch needed)"; return True
    try:
        import protenix.model.layer_norm.layer_norm as LN
        if getattr(LN, "_infopt_stream_patched", False):
            ST["ln"] = "prebuilt(already)"; return True
    except Exception as e:
        ST["why"] = f"protenix LN import: {e!r}"; return False
    pre = os.environ.get("INFOPT_FASTLN_PREBUILT", "")
    if pre:
        try:
            import fastln_prebuilt as FP
            from protenix_opt.binary_sums import manifest_binary_refusal      # the shipped .so is held to the SHA256SUMS beside it before it is mapped; refused by name -> the source rebuild below, as for an absent one
            held = manifest_binary_refusal(pre)
            r = FP.install_prebuilt_fastln(pre, require_bitwise=True) if held is None else {"installed": False, "reason": f"{os.path.basename(pre.rstrip('/'))}: {held}"}
            if r.get("installed"):
                _publish_ln_report(LN, r)
                ST["ln"] = "prebuilt:" + os.path.basename(pre.rstrip("/")); return True
            ST["why"] = f"prebuilt refused: {r.get('reason')}"[:300]
        except Exception as e:
            ST["why"] = f"prebuilt import/install: {e!r}"[:300]
    try:
        from infopt_graphs.protenix.fastln_stream import install_stream_correct_fastln
        try:                                   # print-only: say WHICH infopt_graphs copy serves fastln_stream
            import infopt_graphs as _ig; ST["infopt_graphs"] = f"{getattr(_ig, '__version__', '?')}@{os.path.dirname(getattr(_ig, '__file__', '?'))}"
            print(f"[fpf_stackgraph] infopt_graphs {ST['infopt_graphs']}", file=sys.stderr, flush=True)
        except Exception:
            pass
        r = install_stream_correct_fastln(require_bitwise=True)
        if r.get("installed", r.get("ok", False)):
            _publish_ln_report(LN, r)
            ST["ln"] = "rebuilt"; return True
        ST["why"] = (ST["why"] or "") + f" | fastln_stream refused: {str(r.get('reason', r))[:200]}"
    except Exception as e:
        ST["why"] = (ST["why"] or "") + f" | fastln_stream: {e!r}"[:200]
    return False



def _capture_ctx():
    """Autocast WEIGHT-CAST CACHE hazard: inside Protenix's per-predict torch.autocast(bf16) the eager reference call
    populates the autocast cast cache, so a capture made with the cache ON records no cast kernels and bakes in the ADDRESSES of the cached bf16 weight
    copies -- which torch frees when the outermost autocast exits (end of every predict()).  Cross-seed / cross-design replays would then read freed
    memory.  Fix (numerics-neutral: the cast is the same elementwise fp32->bf16 rounding either way): warm-up AND capture run under autocast with
    cache_enabled=False, so the cast kernels are recorded inside the graph and read the persistent fp32 parameters at replay time."""
    if torch.is_autocast_enabled():
        return torch.autocast("cuda", dtype=torch.get_autocast_gpu_dtype(), cache_enabled=False)
    import contextlib
    return contextlib.nullcontext()


def _unpoison(device_index: int, pool, failed_graph, why: str) -> dict:
    """Restore process-wide CUDA state after a FAILED capture — ONE shared routine for the bundle: fpf_trunkgraph/recovery.py
    (third_party/fpf_trunkgraph).  It ends a dangling capture, runs the allocator epilogue for our
    explicit pool (endAllocateToPool + releasePool exactly once), resets the CUDA generator capture flag via a trivial successful capture, probes allocator
    (sync + empty_cache + alloc/free) and RNG health, retries once and RAISES if still unhealthy (a poisoned generator would silently break MSA sampling)."""
    from fpf_trunkgraph.recovery import recover_after_failed_capture as _rec          # third_party/fpf_trunkgraph: the one copy
    via = "fpf_trunkgraph.recovery"
    rep = _rec(graph=failed_graph, pool=pool, device=device_index, capture_began=True, capture_ended=False, raise_if_unhealthy=True)
    out = {"via": via, "why": why[:160]}
    try:
        out.update(dict(rep))
    except Exception:
        out["result"] = str(rep)[:300]
    return out


_FAILED_GRAPHS = []


def _live_graphs() -> int:
    return sum(1 for x in _CACHE.values() if x.graph is not None)


def _evict_lru():
    """HAZARD #43: NON-DESTRUCTIVE. A captured graph is never destroyed while the process lives: destroying the LRU graph released its
    private pool (empty_cache unmapped the segments) while another armed graph could still hold an address produced inside that pool through a
    data_ptr-keyed global cache of a composed lever -> 'CUDA error: unspecified launch failure' at the next replay (observed: 18 inputs, 9 distinct
    N_token <= 448, crash right after the first eviction; Protenix's batch runner then logged 'Run inference failed' and exited 0 = silent loss).
    Policy now: at most PTX_BLK_GRAPH_MAX live graphs; once full, NEW signatures run EAGER (exact, slower) — counted in ST['full_eager'] and printed once per
    N_tok. Only graph-less bookkeeping entries are trimmed here."""
    dead = [k for k, x in _CACHE.items() if x.graph is None]
    while len(_CACHE) > (256 if _AUTO else 4 * _MAX_SIG) and dead:
        _CACHE.pop(dead.pop(0), None)


def _init_budget():
    """Fix the auto-mode budget at the FIRST eligible stack call (model is loaded, first features are on the device)."""
    if _BUDGET["bytes"] is None:
        free, total = torch.cuda.mem_get_info()
        if _BUDGET.get("lazy_auto"):                                  # PTX_BLK_GRAPH_MEM_GB=auto (XL policy): 30 % of TOTAL device memory
            _BUDGET["bytes"] = 0.30 * float(total); _BUDGET["src"] = f"auto=0.30*total({total/2**30:.0f}GB)"
        else:                                                         # unset: the default, 25 % of FREE memory at the first eligible call
            _BUDGET["bytes"] = float(_MEM_FRAC) * float(free); _BUDGET["src"] = f"{_MEM_FRAC:.2f}*free({free/2**30:.1f}GB of {total/2**30:.0f}GB)"
        ST["budget_gb"] = round(_BUDGET["bytes"] / 2**30, 2); ST["budget_src"] = _BUDGET["src"]
        print(f"[fpf_stackgraph] auto cache: budget {ST['budget_gb']} GB ({_BUDGET['src']}); signatures are captured while the measured graph pools fit, then run eager (nothing captured is freed)", file=sys.stderr, flush=True)
    elif "budget_gb" not in ST:
        ST["budget_gb"] = round(_BUDGET["bytes"] / 2**30, 2); ST["budget_src"] = _BUDGET["src"]


def _estimate_pool(ntok: int) -> float:
    """Predicted private-pool bytes of the next capture: the largest measured bytes/N_tok^2 so far scaled to ntok (pair activations dominate);
    before the first capture a conservative 0.75 GB * (ntok/256)^2 floor-ed at 0.25 GB."""
    if _BUDGET["per_tok2_max"] > 0:
        return _BUDGET["per_tok2_max"] * ntok * ntok
    return max(0.25 * 2**30, 0.75 * 2**30 * (ntok / 256.0) ** 2)


def _admit(ntok: int):
    """Admission of a NEW signature. fixed mode: live-graph count < PTX_BLK_GRAPH_MAX. auto mode: predicted pool of this capture fits the
    remaining memory budget AND the device has that much free right now. Never evicts (non-destructive by construction)."""
    if not _AUTO:
        lg = _live_graphs()
        if lg >= _MAX_SIG:
            return False, f"{lg}/{_MAX_SIG} live graphs"
        if not _BUDGET.get("explicit"):                               # fixed-count mode without an explicit PTX_BLK_GRAPH_MEM_GB: count only
            return True, f"{lg}/{_MAX_SIG} live graphs"
    _init_budget()
    est = _estimate_pool(ntok); left = _BUDGET["bytes"] - _BUDGET["pool_total"]
    try:
        free_now = torch.cuda.mem_get_info()[0] + (torch.cuda.memory_reserved() - torch.cuda.memory_allocated())
    except Exception:
        free_now = float("inf")
    ok = (est <= left) and (est * 1.5 <= free_now)
    why = (f"auto budget {_BUDGET['bytes']/2**30:.2f} GB: pools {_BUDGET['pool_total']/2**30:.2f} GB in {_live_graphs()} graphs, next N_tok={ntok} est {est/2**30:.2f} GB, "
           f"left {left/2**30:.2f} GB, device free {free_now/2**30:.1f} GB")
    return ok, why


def _account_pool(ent, pool_bytes: int):
    ntok = int(ent.n_tok); pool_bytes = max(int(pool_bytes), 0)
    ent.pool_bytes = pool_bytes
    _BUDGET["pool_total"] += pool_bytes
    if ntok > 0 and pool_bytes > 0:
        _BUDGET["per_tok2_max"] = max(_BUDGET["per_tok2_max"], pool_bytes / float(ntok * ntok))
    ST["pool_bytes_by_ntok"][ntok] = pool_bytes; ST["pool_gb"] = round(_BUDGET["pool_total"] / 2**30, 3)
    if _AUTO and _BUDGET["bytes"] is not None and _BUDGET["pool_total"] > _BUDGET["bytes"]:
        _BUDGET["overshoot"] += 1; ST["budget_overshoot"] = _BUDGET["overshoot"]
        print(f"[fpf_stackgraph] NOTE: measured pools {_BUDGET['pool_total']/2**30:.2f} GB exceed the budget {_BUDGET['bytes']/2**30:.2f} GB after N_tok={ntok} (estimate was low); "
              f"no further captures; captured graphs stay (never freed).", file=sys.stderr, flush=True)


def _replay(ent, where: str):
    """Replay with a HARD failure policy: a CUDA error during replay leaves the context unusable; returning to the caller would let Protenix's
    per-input catch-all log 'Run inference failed' and continue/exit 0. We print a FATAL line with the counters and terminate the process non-zero."""
    try:
        ent.graph.replay()
    except BaseException as ex:      # noqa: BLE001
        ST["replay_fail"] = ST.get("replay_fail", 0) + 1; ST["disabled"] = True
        msg = (f"[fpf_stackgraph] FATAL: graph replay failed at N_tok={getattr(ent, 'n_tok', '?')} ({where}): {ex!r}; live_graphs={_live_graphs()} "
               f"captures={ST.get('captures')} replays={ST.get('replays')} evicted={ST.get('evicted')} full_eager={ST.get('full_eager', 0)} -> terminating the process "
               f"with exit code 70 (a CUDA fault is sticky; continuing would silently drop the remaining inputs). Workaround: PTX_BLK_GRAPH=0.")
        print(msg, file=sys.stderr, flush=True); print(msg, flush=True)
        try:
            _write_report_line({"stackgraph_fatal": msg})
        except Exception:
            pass
        os._exit(70)


def _write_report_line(d: dict) -> None:
    p = os.environ.get("PTX_LEVER_REPORT")
    if p:
        import json as _json
        with open(p, "a") as fh:
            fh.write(_json.dumps(d, default=str) + "\n")


def install(model=None) -> str:
    """Patch protenix PairformerStack.forward (class-level) with the capture/replay wrapper.  Idempotent.  Returns a one-line status string."""
    import protenix.model.modules.pairformer as PF
    if getattr(PF.PairformerStack, "_fpf_stackgraph", False):
        return "already"
    if not torch.cuda.is_available():
        ST["why"] = "no cuda"; return "off(no cuda)"
    if not ensure_stream_ln():
        ST["installed"] = False
        return "off(no stream-safe LN: " + str(ST["why"])[:200] + ")"
    _orig = PF.PairformerStack.forward

    def forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        kw = dict(pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                  inplace_safe=inplace_safe, chunk_size=chunk_size)
        base = (not self.training) and (not torch.is_grad_enabled()) and chunk_size is None and isinstance(z, torch.Tensor) and z.is_cuda \
            and z.dim() == 3 and _MIN_TOK <= z.shape[-2] <= _MAX_TOK and inplace_safe
        main_region = base and pair_mask is None and isinstance(s, torch.Tensor) and s.dim() == 2 and len(self.blocks) >= 8 and getattr(self.blocks[0], "c_s", 0) > 0
        templ_region = base and _TEMPL and (s is None) and getattr(self.blocks[0], "c_s", -1) == 0 and isinstance(pair_mask, torch.Tensor) and pair_mask.is_cuda \
            and len(self.blocks) <= 4
        elig = main_region or templ_region
        region = "templ" if (templ_region and not main_region) else "main"
        if not elig or ST.get("disabled"):
            ST["eager_ineligible"] += 1
            return _orig(self, s, z, **kw)
        sig = _sig(self, s, z, kw)
        ent = _CACHE.get(sig)
        if ent is not None and ent.armed:
            _CACHE.move_to_end(sig)
            _copy_in(ent, s, z, pair_mask)
            _replay(ent, "armed")
            ent.uses += 1; ST["replays"] += 1
            if region == "templ": ST["templ"]["replays"] += 1
            d = ST["by_ntok"].setdefault(int(z.shape[-2]), {"replays": 0, "captures": 0}); d["replays"] += 1
            if _NOCLONE:
                return ent.s_out, ent.z_out
            return (None if ent.s_out is None else ent.s_out.clone()), ent.z_out.clone()
        if ent is not None and not ent.armed:
            ST["eager"] += 1
            return _orig(self, s, z, **kw)
        # ---- cache-full policy (HAZARD #43): never evict; beyond PTX_BLK_GRAPH_MAX live graphs new signatures run EAGER (exact)
        _adm_ok, _adm_why = _admit(int(z.shape[-2]))
        if not _adm_ok:
            ST["full_eager"] = ST.get("full_eager", 0) + 1
            if _AUTO or _BUDGET.get("explicit"): ST["budget_eager"] = ST.get("budget_eager", 0) + 1
            fe = ST.setdefault("full_eager_by_ntok", {}); nt = int(z.shape[-2]); fe[nt] = fe.get(nt, 0) + 1
            if fe[nt] == 1:
                print(f"[fpf_stackgraph] cache full ({_adm_why}): N_tok={nt} runs EAGER (exact, slower) for the rest of this process; "
                      f"{'raise PTX_BLK_GRAPH_MEM_GB' if _AUTO else 'raise PTX_BLK_GRAPH_MAX (or set it to auto)'} if memory allows (each live graph keeps its private pool; nothing captured is ever freed).", file=sys.stderr, flush=True)
            e = _Entry(); e.armed = False; e.graph = None; e.m_in = None; e.s_in = e.z_in = e.s_out = e.z_out = None; e.n_tok = nt; e.uses = 0; e.pool_bytes = 0
            _CACHE[sig] = e; _evict_lru()
            ST["eager"] += 1
            return _orig(self, s, z, **kw)
        # ---- first sighting of this signature: eager reference (this call's result), then capture + check
        # the reference, the warm-up and the capture below run inside ONE plan-for-capture window (planning() True) so timing-keyed levers in the stack select the
        # graph column for all three under a tier word -> the oracle and the graph serve the same rows; the window closes in __exit__ whatever happens inside.
        with _planning(sig):
            s0 = None if s is None else s.clone(); z0 = z.clone(); m0 = None if pair_mask is None else pair_mask.clone()
            torch.cuda.synchronize()
            free0, total = torch.cuda.mem_get_info()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            s_ref, z_ref = _orig(self, s, z, **kw)
            ST["eager"] += 1
            if region == "templ": ST["templ"]["eager"] += 1
            torch.cuda.synchronize()
            peak_extra = torch.cuda.max_memory_allocated() - base_alloc
            e = _Entry(); e.armed = False; e.graph = None; e.m_in = None; e.s_in = e.z_in = e.s_out = e.z_out = None; e.n_tok = int(z.shape[-2]); e.uses = 0; e.pool_bytes = 0
            try:
                free1, _ = torch.cuda.mem_get_info()
                need = 2 * peak_extra + 4 * (z0.numel() * z0.element_size() + (0 if s0 is None else s0.numel() * s0.element_size()))
                if free1 + (torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) < need:
                    ST["oom_skips"] += 1; ST["why"] = f"capture skipped: free {free1/2**30:.1f} GB < need {need/2**30:.1f} GB"
                    # hazard #29: a memory-driven skip must not be SILENT (the stack would run eager with no console line). Print once per N_tok (+count) to stderr.
                    _sk = ST.setdefault("oom_skips_by_ntok", {}); _sk[e.n_tok] = _sk.get(e.n_tok, 0) + 1
                    if _sk[e.n_tok] == 1:
                        print(f"[fpf_stackgraph] WARNING: {ST['why']} at N_tok={e.n_tok} (region={region}) -> this signature runs EAGER (exact, slower); oom_skips={ST['oom_skips']}. Speed rows must show oom_skips==0.", file=sys.stderr, flush=True)
                    _CACHE[sig] = e; _evict_lru(); return s_ref, z_ref
                t0 = time.perf_counter()
                e.s_in = None if s0 is None else s0.clone(); e.z_in = z0.clone(); e.m_in = None if m0 is None else m0.clone()
                kw_st = dict(kw)
                if e.m_in is not None:
                    kw_st["pair_mask"] = e.m_in                    # static mask buffer (template region); main region has pair_mask None
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side), _capture_ctx():      # warm-up on a side stream (allocator / cuBLAS handles / Triton JIT), per torch.cuda.graphs docs; no autocast cache
                    _orig(self, None if e.s_in is None else e.s_in.clone(), e.z_in.clone(), **kw_st)
                torch.cuda.current_stream().wait_stream(side)
                torch.cuda.synchronize()
                torch.cuda.empty_cache(); _free_c0 = torch.cuda.mem_get_info()[0]          # pool accounting (cached free blocks returned first so the delta is the private pool)
                g = torch.cuda.CUDAGraph(); pool = torch.cuda.graph_pool_handle(); dev_idx = z0.device.index if z0.device.index is not None else torch.cuda.current_device()
                _copy_in(e, s0, z0, m0)
                captured = False
                _clear_global_ptr_caches()          # every data_ptr-keyed cached tensor the graph reads must be PRODUCED inside the graph (see _GLOBAL_PTR_CACHES)
                try:
                    with _capture_ctx(), torch.cuda.graph(g, pool=pool, capture_error_mode="thread_local"):   # autocast cache DISABLED: weight casts recorded in the graph (read persistent fp32 params)
                        e.s_out, e.z_out = _orig(self, e.s_in, e.z_in, **kw_st)
                    captured = True
                    _clear_global_ptr_caches()      # eager code never holds graph-private-pool tensors through these caches
                finally:
                    if not captured:                                 # capture died inside the region: restore allocator + RNG state BEFORE anything else runs
                        ST["unpoison"] = _unpoison(dev_idx, pool, g, "capture failed")
                        print(f"[fpf_stackgraph] capture FAILED -> state restored: {ST['unpoison']}", file=sys.stderr, flush=True)
                e.graph = g
                torch.cuda.synchronize()
                try:
                    torch.cuda.empty_cache(); _account_pool(e, _free_c0 - torch.cuda.mem_get_info()[0])
                except Exception as _ex:
                    _log(f"pool accounting failed: {_ex!r}")
                ST["capture_s"] += time.perf_counter() - t0
                # ---- check: replay on the same inputs and compare with THIS call's eager result (bitwise)
                t1 = time.perf_counter()
                _copy_in(e, s0, z0, m0)
                _replay(e, "check"); torch.cuda.synchronize()
                ok = bool(torch.equal(e.z_out, z_ref)) and _eq_opt(e.s_out, s_ref)
                ST["check_s"] += time.perf_counter() - t1
                ST["captures"] += 1
                if region == "templ": ST["templ"]["captures"] += 1
                d = ST["by_ntok"].setdefault(e.n_tok, {"replays": 0, "captures": 0}); d["captures"] += 1
                if ok:
                    e.armed = True
                    _log(f"armed N_tok={e.n_tok} capture+check {ST['capture_s']:.2f}s peak_extra={peak_extra/2**30:.2f}GB")
                else:
                    ST["refused"] += 1
                    ST["check_fail_maxabs"].append([e.n_tok, float((e.z_out.float() - z_ref.float()).abs().max()), _maxabs_opt(e.s_out, s_ref), region])
                    if region == "templ": ST["templ"]["refused"] += 1
                    e.graph = None; e.s_out = e.z_out = e.s_in = e.z_in = None
                    _log(f"REFUSED N_tok={e.n_tok}: replay != eager (maxabs {ST['check_fail_maxabs'][-1]})")
                _CACHE[sig] = e; ST["sigs"] = len(_CACHE); _evict_lru()
            except Exception as ex:
                if is_oom(ex): raise
                # A failed capture (e.g. a host sync / .item() / profiler timer inside the region) can leave CUDA/RNG capture state inconsistent for the rest of
                # the process ("Offset increment outside graph capture").  Policy: disable the lever for the WHOLE process (all later calls eager), record why,
                # best-effort cleanup; the eager reference result of this call is still returned (it was computed before the capture attempt).
                ST["refused"] += 1; ST["why"] = f"capture error (lever disabled for this process): {ex!r}"[:400]; ST["disabled"] = True
                e.armed = False; e.graph = None; e.s_out = e.z_out = e.s_in = e.z_in = None
                try:
                    _CACHE[sig] = e
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    import gc; gc.collect(); torch.cuda.empty_cache()
                except Exception:
                    pass
                print(f"[fpf_stackgraph] WARNING: {ST['why']}", file=sys.stderr, flush=True)
            return s_ref, z_ref

    PF.PairformerStack.forward = forward
    PF.PairformerStack._fpf_stackgraph = True
    PF.PairformerStack._fpf_stackgraph_orig = _orig
    ST["installed"] = True
    _bd = (f"{_BUDGET['bytes']/2**30:g}GB" if _BUDGET["bytes"] else ("0.30*total" if _BUDGET.get("lazy_auto") else f"{_MEM_FRAC:.0%} of free at 1st call"))
    _ms = (f"auto(mem {_bd})" if _AUTO else (f"{_MAX_SIG}" + (f"+mem {_bd}" if _BUDGET.get("explicit") else "")))
    ST["mode_desc"] = _ms
    return f"on(PairformerStack.forward; ln={ST['ln']}; max_sig={_ms}; tok {_MIN_TOK}..{_MAX_TOK}; noclone={_NOCLONE}; templ={_TEMPL})"


def uninstall():
    import protenix.model.modules.pairformer as PF
    if getattr(PF.PairformerStack, "_fpf_stackgraph", False):
        PF.PairformerStack.forward = PF.PairformerStack._fpf_stackgraph_orig
        PF.PairformerStack._fpf_stackgraph = False
    _CACHE.clear(); torch.cuda.empty_cache()


def reset_cache():
    _CACHE.clear(); ST["sigs"] = 0; torch.cuda.empty_cache()


def _atexit():
    p = os.environ.get("PTX_LEVER_REPORT", "")
    if p and ST["installed"]:
        try:
            with open(p, "a") as f:
                f.write(json.dumps({"stackgraph": report(), "pid": os.getpid()}, default=str) + "\n")
        except Exception:
            pass


atexit.register(_atexit)


def _fpf_atexit_summary():                       # one summary line per process so eager-only runs are visible (hazard #29)
    try:
        if ST.get("installed") or ST.get("captures") or ST.get("eager") or ST.get("oom_skips"):
            print(f"[fpf_stackgraph] SUMMARY full_eager={ST.get('full_eager', 0)} live_graphs={_live_graphs()} replay_fail={ST.get('replay_fail', 0)} captures={ST.get('captures')} replays={ST.get('replays')} eager={ST.get('eager')} refused={ST.get('refused')} oom_skips={ST.get('oom_skips')} oom_skips_by_ntok={ST.get('oom_skips_by_ntok', {})} evicted={ST.get('evicted')} disabled={ST.get('disabled')} mode={ST.get('mode')} budget_gb={ST.get('budget_gb')} pool_gb={round(_BUDGET['pool_total']/2**30, 3)} distinct_ntok={len(set(int(getattr(x, 'n_tok', 0)) for x in _CACHE.values()))} budget_eager={ST.get('budget_eager', 0)} pool_bytes_by_ntok={ST.get('pool_bytes_by_ntok', {})} max_sig_desc={ST.get('mode_desc', ST.get('max_sig'))} budget_src={ST.get('budget_src')} planned_refs={ST.get('planned_refs', 0)}", file=sys.stderr, flush=True)
    except Exception:
        pass
try:
    import atexit as _atexit
    _atexit.register(_fpf_atexit_summary)
except Exception:
    pass
