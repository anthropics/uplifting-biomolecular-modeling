"""sitecustomize.py — PYTHONPATH injector for the trunk levers into the STOCK `protenix pred` CLI process.
Active only if any PTX_T1_TRANS / PTX_T2_NOCOPY / PTX_ZT / PTX_T5_EINSUM env var is set.  Mechanism: a meta-path finder that lets the normal
finder locate protenix.model.modules.pairformer, then wraps that spec's loader.exec_module so that — right after the module body has executed
once — ptx_trunk2_levers.apply_from_env() installs the monkeypatches (before any model object is built).  Nothing on disk is modified.
Writes a JSON line to $PTX_LEVER_REPORT (if set) at interpreter exit: levers applied + call counters (evidence that the arm really ran them).
"""
import os, sys, contextlib
if os.environ.get("PTX_LEVER_REPORT"):                  # env.sh's default lives under the user's own cache directory: made private (0700) before any lever appends
    try:
        _rep_dir = os.path.dirname(os.path.abspath(os.environ["PTX_LEVER_REPORT"]))
        if not os.path.lexists(_rep_dir):
            os.makedirs(os.path.dirname(_rep_dir), exist_ok=True); os.mkdir(_rep_dir, 0o700)
    except OSError:
        pass
_KEYS = ("PTX_LAZY_INIT", "PTX_MK_PF", "PTX_T_TRIMUL", "PTX_GLUE_V2", "PTX_E_PAD8", "PTX_BLK_LN", "PTX_T1_TRANS", "PTX_T2_NOCOPY", "PTX_ZT", "PTX_T5_EINSUM", "PTX_OG", "PTX_NOMASK", "PTX_BLK", "FPF_OPS", "PF_TRIMUL", "PF_TRIATTN", "PTX_PWA_ZCACHE", "PTX_DEADSKIP", "PTX_V02_MSABLK", "PTX_V02_TMPL", "PTX_V02_CONF", "PTX_V02_GRAPH", "PTX_BLK_GRAPH", "PTX_V02_CENSUS")

# default-off: PTX_LAZY_INIT=1|recheck -> ptx_lazy_init (src/). Patches
# runner.inference.InferenceRunner.init_model so the 13 random-init fns are no-ops ONLY during model construction (restored after); load_checkpoint
# (strict=True) then overwrites every parameter/persistent buffer -> model state identical to stock (check mode proves it per tensor). Start-up lever only.
if os.environ.get("PTX_LAZY_INIT", "0") in ("1", "recheck"):
    try:
        import ptx_lazy_init as _pli
        _m = _pli.install()
        print(f"[sitecustomize] PTX_LAZY_INIT={os.environ.get('PTX_LAZY_INIT')} -> ptx_lazy_init.install() = {_m}", file=sys.stderr, flush=True)
    except Exception as _e:
        print(f"[sitecustomize] PTX_LAZY_INIT requested but ptx_lazy_init unavailable -> stock construction: {_e!r}", file=sys.stderr, flush=True)

if any(os.environ.get(k) for k in _KEYS):
    import importlib, importlib.abc, importlib.util, atexit, json

    class _Hook(importlib.abc.MetaPathFinder):
        _done = False
        def find_spec(self, fullname, path, target=None):
            if fullname != "protenix.model.modules.pairformer" or _Hook._done:
                return None
            _Hook._done = True
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
                sys.stderr.write("[sitecustomize] WARNING: could not locate protenix pairformer spec; levers NOT applied\n")
                return None
            _orig_exec = real.loader.exec_module
            def exec_module(module, _orig=_orig_exec):
                _orig(module)
                with contextlib.redirect_stdout(sys.stderr):   # rc4: lever import/apply-time prints never reach stdout (`python -c 'print(x)'` under env.sh prints exactly x)
                    _exec_module_body(module)
            def _exec_module_body(module):
                if module.__name__ != "protenix.model.modules.pairformer":
                    return
                here = os.path.dirname(os.path.abspath(__file__))
                if here not in sys.path:
                    sys.path.insert(0, here)
                import ptx_trunk2_levers as LEV
                rep = LEV.apply_from_env()
                try:
                    import ptx_fpf_v02 as V02; rep02 = V02.apply_from_env(); rep['applied'] = list(rep.get('applied', [])) + ['v02:' + str(a) for a in rep02.get('applied', [])]
                    sys.stderr.write(f"[sitecustomize] FPF v0.2 levers: {rep02.get('applied')}\n")
                except Exception as e:
                    sys.stderr.write(f"[sitecustomize] FPF v0.2 levers FAILED: {e!r}\n"); raise
                sys.stderr.write(f"[sitecustomize] trunk-II levers applied: {rep.get('applied')}\n")
                if os.environ.get("PF_TRIMUL") or os.environ.get("PF_TRIATTN"):     # PF-kernels pad/flash add-on: apply AFTER the trunk levers (it wraps module forwards and calls the original inside)
                    try:
                        import pf_protenix
                        pf_protenix.apply()
                        sys.stderr.write(f"[sitecustomize] pf_protenix applied: {pf_protenix.report() if hasattr(pf_protenix, 'report') else 'ok'}\n")
                        def _pfdump():
                            p = os.environ.get("PTX_LEVER_REPORT")
                            if p and hasattr(pf_protenix, "report"):
                                with open(p, "a") as fh: fh.write(json.dumps({"pf_protenix": pf_protenix.report(), "pid": os.getpid()}, default=str) + "\n")
                        atexit.register(_pfdump)
                    except Exception as e:
                        sys.stderr.write(f"[sitecustomize] pf_protenix apply FAILED: {e!r}\n"); raise
                if os.environ.get("FPF_OPS"):
                    try:
                        import fpf as _FPF
                        frep = _FPF.enable_from_env()
                        sys.stderr.write(f"[sitecustomize] FPF ops enabled: {frep}\n")
                        LEV._STATS["fpf"] = frep; LEV._STATS["fpf_calls"] = _FPF.STATS
                    except Exception as e:
                        sys.stderr.write(f"[sitecustomize] FPF enable FAILED: {e!r}\n"); raise
                def _dump():
                    p = os.environ.get("PTX_LEVER_REPORT")
                    if p:
                        try:
                            r = LEV.report(); r["pid"] = os.getpid()
                            try:
                                import ptx_fpf_v02 as _V; r["v02"] = _V.report()
                            except Exception as _e:
                                r["v02_err"] = repr(_e)
                            with open(p, "a") as fh:
                                fh.write(json.dumps(r, default=str) + "\n")
                        except Exception as e:
                            sys.stderr.write(f"[sitecustomize] report failed: {e}\n")
                atexit.register(_dump)
            real.loader.exec_module = exec_module
            return real
    sys.meta_path.insert(0, _Hook())

# ---- composition with the levers add-on (levers_addon/PTXV2_LEVERS_ADDON_v1): python imports only ONE sitecustomize module, so when
#      this file shadows theirs, so their env-gated import is replicated here (their modules must be on PYTHONPATH too, after this dir).
_fpf_redirect = contextlib.redirect_stdout(sys.stderr); _fpf_redirect.__enter__()   # rc4: sibling-kit imports and the direct-apply branch also print only to stderr
if os.environ.get("PTX_TEMPL_DEDUPE", "0") in ("1", "recheck"):
    try:
        import ptx_addon_levers  # noqa: F401  (sibling L2: template distinct-eval; applies at import)
    except Exception as e:
        print(f"[sitecustomize] ptx_addon_levers not installed: {e!r}", file=sys.stderr)

if os.environ.get("PTX_DET", "0") == "1":   # the Protenix 1.1 DET-mode hook (reference config; needs CUBLAS_WORKSPACE_CONFIG=:4096:8 + its scatter patch)
    try:
        import torch
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[sitecustomize] PTX_DET=1: torch.use_deterministic_algorithms(True, warn_only=True); CUBLAS_WORKSPACE_CONFIG=" + str(os.environ.get("CUBLAS_WORKSPACE_CONFIG")), file=sys.stderr)
    except Exception as e:
        print(f"[sitecustomize] PTX_DET failed: {e!r}", file=sys.stderr)
# if the sibling imports above already imported protenix.model.modules.pairformer, the meta-path hook never fires -> apply directly now
if any(os.environ.get(k) for k in _KEYS) and "protenix.model.modules.pairformer" in sys.modules and not getattr(sys.modules.get("ptx_trunk2_levers"), "_STATS", {}).get("applied"):
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        import ptx_trunk2_levers as LEV, atexit as _ax, json as _js
        rep = LEV.apply_from_env()
        try:
            import ptx_fpf_v02 as V02; rep02 = V02.apply_from_env(); rep['applied'] = list(rep.get('applied', [])) + ['v02:' + str(a) for a in rep02.get('applied', [])]
            sys.stderr.write(f"[sitecustomize] FPF v0.2 levers: {rep02.get('applied')}\n")
        except Exception as e:
            sys.stderr.write(f"[sitecustomize] FPF v0.2 levers FAILED: {e!r}\n"); raise
        sys.stderr.write(f"[sitecustomize] trunk-II levers applied (direct): {rep.get('applied')}\n")
        def _dump2():
            p = os.environ.get("PTX_LEVER_REPORT")
            if p:
                r = LEV.report(); r["pid"] = os.getpid()
                try:
                    import ptx_fpf_v02 as _V; r["v02"] = _V.report()
                except Exception as _e:
                    r["v02_err"] = repr(_e)
                open(p, "a").write(_js.dumps(r, default=str) + "\n")
        _ax.register(_dump2)
    except Exception as e:
        sys.stderr.write(f"[sitecustomize] FAILED direct apply: {e!r}\n"); raise

# ---- default-off: served-worker sampler levers in the CLI path — PTX_SAMPLER_GRAPH=1 (the graphed diffusion sampler)
#      [+ PTX_SAMPLER_HOIST=1: DiT-FAST add-on dit_hoist]; installed on the model right after InferenceRunner.init_model (third_party/fpf_clisampler). Tier-2 unless DET-tested;
#      installs under PTX_DET=1 too (the detref scatter v1.1 is stateless and CUDA-graph-safe).
if os.environ.get("PTX_SAMPLER_GRAPH", "0") not in ("", "0") or os.environ.get("PTX_SAMPLER_HOIST", "0") not in ("", "0"):
    try:
        import fpf_clisampler as _fcs
        sys.stderr.write("[sitecustomize] PTX_SAMPLER_GRAPH/HOIST -> fpf_clisampler." + _fcs.install_hook() + "\n")
    except Exception as _e:
        sys.stderr.write(f"[sitecustomize] PTX_SAMPLER_GRAPH requested but fpf_clisampler unavailable -> stock sampler: {_e!r}\n")
try:
    _fpf_redirect.__exit__(None, None, None)
except Exception:
    pass
