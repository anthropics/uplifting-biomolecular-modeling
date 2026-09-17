"""fpf_clisampler.clisampler — the sampler levers (CUDA-graphed diffusion loop + step-invariant hoist) in the stock CLI path.

WHAT IS PORTED (credit — none of the mechanism is new here):
  * `infopt_graphs.protenix.install(model, sampler=True, pool=...)` replaces
    protenix.model.protenix.sample_diffusion by GraphedDenoiseLoop.sample — the 200-step diffusion sampler with the denoiser step captured in a CUDA
    graph per (N_atom, N_token, N_sample, dtype) signature; every random draw (rotation / translation / eps) is made OUTSIDE the graph in stock order,
    so the RNG stream consumed is identical to stock; RNGGuard asserts the captured body consumes no RNG; poison test per capture.
    Worker flags as pinned: --graphs 1 --graphs_pool private.
  * DiT-FAST add-on (dit_hoist.py, worker flag --biascache 2): step-invariant hoist of the pair-bias chain / conditioning terms out of the loop, driven by
    the graphed sampler's bind/record/hit/check/poison protocol.
WHAT THIS MODULE ADDS: only the wiring for the stock CLI (`protenix pred`): a meta-path hook on `runner.inference` wraps InferenceRunner.init_model so the
levers are installed on the constructed model (instance patches survive load_checkpoint), env switches, a refusal policy, and an at-exit SUMMARY/lever-report line.

SWITCHES (default OFF; read at import):
  PTX_SAMPLER_GRAPH=1          graphed sampler (requires the stream-correct fast-LN, exactly like fpf_stackgraph: refuses when protenix's fast LN is not
                               stream-patched in this process; ARM=E env.sh installs it via fpf_stackgraph before the model is built)
  PTX_SAMPLER_HOIST=1          + dit_hoist (requires PTX_SAMPLER_GRAPH=1 and the DiT-FAST add-on's dit_hoist.py on PYTHONPATH, i.e. the levers add-on src dir)
  PTX_SAMPLER_GRAPH_POOL=private|shared   (default private = worker line as pinned)
  PTX_SAMPLER_GRAPH_MAXTOK=<N> predictions above N tokens run the stock sampler (0 = graph every shape; default 0)
  PTX_SAMPLER_GRAPH_POOL_GB=<x> evict sampler-graph entries beyond this pool budget (infopt max_pool_bytes; default 0 = count policy, max_entries 8)
  Under the DET recipe (PTX_DET=1) the lever installs like anywhere else: the recipe's detref scatter (src/detref/det_segment_reduce.py) keeps
                               no state across calls and builds its gather table inside the captured step, so graph replay under DET is value-exact.
NUMERICS LABEL: Tier-2 unless a DET test record says otherwise (graph replay = same kernels; the hoist reorders where step-invariant terms are computed).
"""
from __future__ import annotations
import os, sys, json, atexit, time, contextlib

ST = {"requested": False, "installed": False, "why": None, "graph": False, "hoist": False, "pool": None, "max_tokens": 0, "pool_gb": 0.0,
      "install_s": None, "infopt_graphs": None, "dit_hoist": None, "det": os.environ.get("PTX_DET", "0") == "1"}
_H = {"loop": None, "hoist": None, "model": None}

_GRAPH = os.environ.get("PTX_SAMPLER_GRAPH", "0") not in ("", "0")
_HOIST = os.environ.get("PTX_SAMPLER_HOIST", "0") not in ("", "0")
_POOL = os.environ.get("PTX_SAMPLER_GRAPH_POOL", "private")
_MAXTOK = int(os.environ.get("PTX_SAMPLER_GRAPH_MAXTOK", "0") or 0)
_POOL_GB = float(os.environ.get("PTX_SAMPLER_GRAPH_POOL_GB", "0") or 0)


def _say(msg):
    print(f"[fpf_clisampler] {msg}", file=sys.stderr, flush=True)


def report() -> dict:
    d = dict(ST)
    try:
        lp = _H["loop"]
        if lp is not None and hasattr(lp, "summary"):
            s = lp.summary()
            d["sampler"] = {k: v for k, v in (s.items() if isinstance(s, dict) else []) if k != "events"}
            if hasattr(lp, "prep"):                      # lever sampler_prep: parts on + its counters
                from infopt_graphs.protenix import sampler_prep as _sp
                d["prep"] = lp.prep.describe(); d["reach"] = lp.prep.describe_reach(); d["prep_poison"] = _sp.poison_status(); d["prep_stats"] = _sp.report()
            if isinstance(s, dict) and s.get("events"):
                d["sampler_events_head"] = [str(e)[:200] for e in s["events"][:6]]
    except Exception as e:
        d["sampler_summary_err"] = repr(e)[:200]
    try:
        h = _H["hoist"]
        if h is not None:
            d["hoist_stats"] = {k: getattr(h, k) for k in ("installed", "n_token_blocks", "mode") if hasattr(h, k)}
            for k in ("stats", "STATS"):
                v = getattr(h, k, None)
                if isinstance(v, dict):
                    d["hoist_stats"].update({kk: vv for kk, vv in v.items() if isinstance(vv, (int, float, str, bool))})
    except Exception as e:
        d["hoist_summary_err"] = repr(e)[:200]
    try:
        from . import policy as _policy
        d["policy"] = _policy.summary()
    except Exception as e:
        d["policy_err"] = repr(e)[:200]
    return d


def _install_on_model(model) -> str:
    t0 = time.perf_counter()
    ST["requested"] = True; ST["graph"] = _GRAPH; ST["hoist"] = _HOIST; ST["pool"] = _POOL; ST["max_tokens"] = _MAXTOK; ST["pool_gb"] = _POOL_GB
    import sys as _sys, os as _os
    _main = _os.path.basename(getattr(_sys.modules.get("__main__"), "__file__", "") or (_sys.argv[0] if _sys.argv else ""))
    if _main.startswith("ptx_worker") and _os.environ.get("PTX_SAMPLER_GRAPH_IN_WORKER", "0") != "1":   # the kit worker owns its sampler graphs (--graphs/--biascache); never double-install from env.sh defaults
        ST["why"] = f"inside kit worker ({_main}): worker flags own the sampler graphs; fpf_clisampler is CLI-path only (PTX_SAMPLER_GRAPH_IN_WORKER=1 overrides)"; return "off(" + ST["why"] + ")"
    if not _GRAPH:
        ST["why"] = "PTX_SAMPLER_GRAPH not set" + (" (PTX_SAMPLER_HOIST needs it, like worker --biascache 2 needs --graphs 1)" if _HOIST else ""); return "off(" + ST["why"] + ")"
    try:
        import torch
        if not torch.cuda.is_available():
            ST["why"] = "no cuda"; return "off(no cuda)"
    except Exception as e:
        ST["why"] = f"torch: {e!r}"; return "off(torch)"
    # stream-correct LN guard (same precondition as fpf_stackgraph / the worker's fastln_guard): graphs replay only kernels launched on the capture stream
    if os.environ.get("LAYERNORM_TYPE", "") == "fast_layernorm":
        try:
            import protenix.model.layer_norm.layer_norm as LN
            ok_ln = bool(getattr(LN, "_infopt_stream_patched", False))
        except Exception as e:
            ok_ln = False; ST["why"] = f"LN import: {e!r}"
        if not ok_ln:
            try:                                   # ARM=E normally installed it already (fpf_stackgraph.ensure_stream_ln at pairformer import); try once more here
                from fpf_stackgraph.stackgraph import ensure_stream_ln
                ok_ln = bool(ensure_stream_ln()) and bool(getattr(LN, "_infopt_stream_patched", False) or os.environ.get("LAYERNORM_TYPE") != "fast_layernorm")
            except Exception as e:
                ST["why"] = f"ensure_stream_ln: {e!r}"[:200]
        if not ok_ln:
            ST["why"] = (ST["why"] or "") + " | stream-correct fast-LN not installed in this process -> refusing sampler graphs"
            return "off(" + ST["why"].strip(" |") + ")"
    try:
        import infopt_graphs
        from infopt_graphs.protenix import install as _gi
        ST["infopt_graphs"] = f"{getattr(infopt_graphs, '__version__', '?')}@{os.path.dirname(getattr(infopt_graphs, '__file__', '?'))}"
    except Exception as e:
        ST["why"] = f"infopt_graphs not importable (put the kit-of-record src dir on PYTHONPATH): {e!r}"[:300]; return "off(" + ST["why"] + ")"
    try:
        handles = _gi(model, sampler=True, trunk=False, pool=_POOL, max_pool_bytes=int(_POOL_GB * 1e9), max_tokens=_MAXTOK, fastln_stream_fix=False)
    except TypeError:                              # older infopt_graphs (bundle _fallback copy): no max_pool_bytes/max_tokens kwargs
        handles = _gi(model, sampler=True, trunk=False, pool=_POOL, fastln_stream_fix=False)
    loop = handles.get("sampler") if isinstance(handles, dict) else None
    if loop is None:
        ST["why"] = f"infopt_graphs.install returned no sampler handle ({list(handles) if isinstance(handles, dict) else handles})"; return "off(" + ST["why"] + ")"
    _H["loop"] = loop; _H["model"] = model
    hoist_s = "off"
    if _HOIST:
        try:
            import dit_hoist
            ST["dit_hoist"] = os.path.abspath(getattr(dit_hoist, "__file__", "?"))
            sys.modules.setdefault("biascache_static", dit_hoist)          # worker line: graphed.py drives the hoist through the biascache_static protocol name
            _H["hoist"] = dit_hoist.install(model, sampler=loop)
            hoist_s = f"on(n_token_blocks={getattr(_H['hoist'], 'n_token_blocks', '?')})"
        except Exception as e:
            ST["why"] = f"dit_hoist install failed -> graphs without hoist: {e!r}"[:300]; _say("WARNING: " + ST["why"]); hoist_s = "FAILED"
    try:                                                                  # lever sampler_admit: the admission policy (admission / release / eager hoist; inert unless its words are set)
        from . import policy as _policy
        pol_s = _policy.install(loop, _H["hoist"])
    except Exception as e:                                                # named on the SAMPLER: marker (the activation report moves sampler_admit to fallen back), never silent
        pol_s = f"policy:FAILED({e!r})"[:200]; ST["why"] = (ST["why"] or "") + " | " + pol_s
    ST["policy"] = pol_s
    ST["installed"] = True; ST["install_s"] = round(time.perf_counter() - t0, 2)
    prep_s = (loop.prep.describe() + "; reach=" + loop.prep.describe_reach()) if hasattr(loop, "prep") else "n/a"; ST["prep"] = prep_s
    return (f"on(sampler graph: infopt_graphs {ST['infopt_graphs']}; pool={_POOL}; max_tokens={_MAXTOK or 'all'}; pool_gb={_POOL_GB or 'count'}; "
            f"hoist={hoist_s}; prep={prep_s}; {pol_s}; det={'ALLOWED(test-only)' if ST['det'] else 'n/a'}; {ST['install_s']} s)")


def _wrap_runner_class(RI):
    if getattr(RI.InferenceRunner, "_fpf_clisampler", False):
        return
    orig = RI.InferenceRunner.init_model

    def init_model(self, *a, **k):
        r = orig(self, *a, **k)
        try:
            with contextlib.redirect_stdout(sys.stderr):
                msg = _install_on_model(self.model)
        except Exception as e:      # never break the CLI: stock sampler continues
            ST["why"] = f"install error: {e!r}"[:300]; msg = "off(" + ST["why"] + ")"
        _say("SAMPLER:" + msg)
        return r
    RI.InferenceRunner.init_model = init_model
    RI.InferenceRunner._fpf_clisampler = True


def install_hook() -> str:
    """Called from src/sitecustomize.py when PTX_SAMPLER_GRAPH / PTX_SAMPLER_HOIST is set."""
    if "runner.inference" in sys.modules:
        _wrap_runner_class(sys.modules["runner.inference"]); return "hook(direct)"
    import importlib.abc

    class _Finder(importlib.abc.MetaPathFinder):
        _done = False

        def find_spec(self, fullname, path, target=None):
            if fullname != "runner.inference" or _Finder._done:
                return None
            _Finder._done = True
            real = None
            for f in sys.meta_path:
                if f is self:
                    continue
                try:
                    real = f.find_spec(fullname, path, target)
                except Exception:
                    continue
                if real:
                    break
            if real is None or real.loader is None:
                _say("WARNING: runner.inference spec not found; sampler levers not applied"); return None
            _oexec = real.loader.exec_module

            def _exec(module, _orig=_oexec):
                _orig(module)
                if module.__name__ == "runner.inference":
                    try:
                        _wrap_runner_class(module)
                    except Exception as e:
                        _say(f"WARNING: could not wrap InferenceRunner.init_model: {e!r}")
            real.loader.exec_module = _exec
            return real
    sys.meta_path.insert(0, _Finder())
    return "hook(meta_path)"


def _atexit():
    try:
        if ST["requested"] or _GRAPH or _HOIST:
            r = report()
            s = r.get("sampler") or {}
            pol = r.get("policy") or {}
            _say(f"SUMMARY installed={ST['installed']} graph={ST['graph']} hoist={ST['hoist']} det={ST['det']} why={ST['why']} policy={pol.get('words', '-')} admit={','.join(f'{k}:{v}' for k, v in (pol.get('verdicts') or {}).items()) or '-'} releases={pol.get('releases', 0)} misses={pol.get('admit_misses', 0)} sampler={json.dumps(s, default=str)[:600]} hoist_stats={json.dumps(r.get('hoist_stats'), default=str)[:300]}")
            p = os.environ.get("PTX_LEVER_REPORT", "")
            if p:
                with open(p, "a") as fh:
                    fh.write(json.dumps({"clisampler": r, "pid": os.getpid()}, default=str) + "\n")
    except Exception:
        pass


atexit.register(_atexit)
