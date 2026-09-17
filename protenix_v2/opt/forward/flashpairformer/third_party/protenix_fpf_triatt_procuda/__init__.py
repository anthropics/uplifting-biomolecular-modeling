"""protenix_fpf_triatt_procuda — kit lever `triatt_prologue_cuda`.  EXACT-BY-CONSTRUCTION (same arithmetic as the F1 prologue cell it displaces; the first
call of every new shape is checked bitwise against that cell in-process and the lever REFUSES BY NAME on a mismatch).

WHAT: the tri-attention statement's fused prologue (LayerNorm -> q|k|v|g projections + pair bias; fpf_mkpf's F1 Triton cell on the BLK2 block path) is
replaced by one hand-written sm_90a kernel (csrc/triatt_procuda_sm90.cu, shipped prebuilt): persistent and warp-specialized — producer warps stream the
pair rows through a cp.async ring and apply the cell's LayerNorm arithmetic op for op (rn/ftz/approx exactly as the emulation of upstream's
fast_layernorm), two consumer warpgroups run the bf16 projections on the tensor cores (wgmma, fp32 accumulate, k ascending = the cell's order) and store
q/k/v/g through swizzled staging + TMA straight into the statement's layouts (row-major [I,H,J,D], or head-major for lever triatt_headsplit_exact; unpadded
or the padded PAD8 buffers, valid region only).  Measured per node call on H100 (graph replay): 768 tokens 1.05 -> 0.64 ms, 1536 tokens 4.11 -> 2.5 ms,
bitwise equal outputs.

ENVELOPE (compile-time): c_z = 256, 8 heads x 32, bf16 activations, the 'welford' F1 cell (cc 9.0).  Calls outside it (the template pair stack's c=64
modules, write_x requests) go to the displaced Triton function unchanged and are COUNTED (tally pass_shape / pass_wx) — the lever is engaged with 0
eligible calls there, never a fallback.  Devices other than cc 9.x: REFUSED BY NAME at install (the binary is sm_90a).

COUNTERS: ptx_trunk2_levers._STATS["procuda"] = live tally {cuda, rm, hm, pad, pass_shape, pass_wx, chk_eq, chk_ne, chk_defer, chk_pending} for the
kit's LEVER/EXIT line (chk_defer = new shapes first met while a CUDA graph was being captured: served, checked at their next eager call; chk_pending = how
many of those are still unchecked — at exit it must read 0, i.e. every served shape was compared bitwise with the F1 cell at least once).

SWITCH: PTX_TRIATT_PROCUDA=1 (set by the mode registry).  install() raises Refused (by name) outside the install-time envelope."""
import os, sys, json, atexit
import torch

__version__ = "0.9.1"
LEVER = "triatt_prologue_cuda"
SWITCH = "PTX_TRIATT_PROCUDA"
TESTED_CC = ((9, 0),)

_STATE = {"installed": False, "why": "", "patched": [], "cells": None, "manifest": None, "shapes": []}
_T = {"cuda": 0, "rm": 0, "hm": 0, "pad": 0, "pass_shape": 0, "pass_wx": 0, "chk_eq": 0, "chk_ne": 0, "chk_defer": 0, "chk_pending": 0}
_DEFERRED = set()


class Refused(RuntimeError):
    """Raised BY NAME when the lever cannot serve (install-time envelope) or when a bitwise shape check fails (run time). Never caught by the lever."""


def _say(msg):
    print(f"[FPF] protenix_fpf_triatt_procuda {msg}", file=sys.stderr, flush=True)


def requested():
    return (os.environ.get(SWITCH, "0") or "0") not in ("", "0")


def marker():
    if not _STATE["installed"]:
        return f"PROCUDA:unavailable({_STATE['why'] or 'not installed'})"
    man = _STATE["manifest"] or {}
    return f"PROCUDA:on(sm90a,so={str(man.get('so_sha256', '?'))[:10]},sites={'+'.join(_STATE['patched'])})"


def stats():
    return {"version": __version__, "lever": LEVER, "installed": _STATE["installed"], "why": _STATE["why"], "patched": list(_STATE["patched"]),
            "tally": dict(_T), "shapes": [list(s) for s in _STATE["shapes"]], "manifest": _STATE["manifest"], "cells": _STATE["cells"]}


def _write_report(path):
    try:
        with open(path, "a") as f:
            f.write(json.dumps({"protenix_fpf_triatt_procuda": stats(), "unit": "protenix_fpf_triatt_procuda", "pid": os.getpid()}, default=str) + "\n")
    except Exception as e:
        _say(f"report write failed: {e!r}")


def _shape_check(key, mine, ref_fn):
    """First call of a new shape key: recompute with the displaced Triton function and require bitwise equality of all five outputs."""
    if key in _STATE["shapes"]:
        return
    if torch.cuda.is_current_stream_capturing():
        if key not in _DEFERRED:
            _DEFERRED.add(key); _T["chk_defer"] += 1; _T["chk_pending"] += 1      # chk_pending = deferred shapes not (yet) checked; must read 0 at exit
        return
    ref = ref_fn()
    q, k, v, g, b = mine
    rq, rk, rv, rg, rb = ref[:5]
    if key[3]:   # padded: compare the valid region (the reference wrote into its own scratch buffers of the same padded geometry)
        NI, NJ = key[0], key[1]
        pairs = ((q[:NI, :, :NJ], rq[:NI, :, :NJ]), (k[:NI, :, :NJ], rk[:NI, :, :NJ]), (v[:NI, :, :NJ], rv[:NI, :, :NJ]), (g, rg),
                 ((b if b.dim() == 4 else b.unsqueeze(0))[:, :, :NI, :NJ], (rb if rb.dim() == 4 else rb.unsqueeze(0))[:, :, :NI, :NJ]))
    else:
        pairs = ((q, rq), (k, rk), (v, rv), (g, rg), (b, rb))
    eq = all(x.shape == y.shape and bool(torch.equal(x, y)) for x, y in pairs)
    del ref
    _STATE["shapes"].append(key); _T["chk_eq" if eq else "chk_ne"] += 1
    if key in _DEFERRED:
        _DEFERRED.discard(key); _T["chk_pending"] -= 1
    if not eq:
        raise Refused(f"protenix_fpf_triatt_procuda ({LEVER}): REFUSED — CUDA prologue != F1 cell (torch.equal False) at shape NI={key[0]} NJ={key[1]} "
                      f"ending={key[2]} padded={key[3]} head_major={key[4]}; run with MODEL_OPT_LEVERS_OFF={LEVER} and report this line")


def _scratch_like(out):
    """Padded reference buffers of the same geometry (zeros; bias pads copied) for the one-time shape check."""
    q = out["q"]; b = out["bias"]
    ref = {"q": torch.zeros_like(q), "k": torch.zeros_like(q), "v": torch.zeros_like(q), "bias": b.clone()}
    return ref


def _make_prologue(orig, B, head_major, HMmod=None):
    """Replacement for fpf_mkpf.kernels.prologue_ln (head_major=False) / protenix_fpf_triatt_headsplit.prologue_hm.prologue_ln_hm (head_major=True):
    identical signature and return layouts; serves the call with the CUDA kernel inside the envelope, else calls `orig` (counted)."""
    def prologue(module, z, cch, cfg, ending, ln_arith="fused", fma_flags=(True, True, True), out=None, write_x=False, **kw):
        if write_x or kw:
            _T["pass_wx"] += 1
            return orig(module, z, cch, cfg, ending, ln_arith=ln_arith, fma_flags=fma_flags, out=out, write_x=write_x, **kw) if not head_major else \
                   orig(module, z, cch, cfg, ending, ln_arith=ln_arith, fma_flags=fma_flags, out=out, **kw)
        if ln_arith != "welford" or tuple(bool(f) for f in fma_flags) != (True, True, True) or not B.fits(z, cch):
            _T["pass_shape"] += 1
            return orig(module, z, cch, cfg, ending, ln_arith=ln_arith, fma_flags=fma_flags, out=out)
        if out is not None and head_major and HMmod is not None:
            HMmod.mark_padded_layout(out, "hm")
        res = B.launch(z, cch, bool(ending), out=out, head_major=head_major)
        _T["cuda"] += 1; _T["hm" if head_major else "rm"] += 1
        if out is not None: _T["pad"] += 1
        NI, NJ = (int(z.shape[1]), int(z.shape[0])) if ending else (int(z.shape[0]), int(z.shape[1]))
        key = (NI, NJ, bool(ending), out is not None, bool(head_major), int(out["q"].shape[0]) if out is not None else 0)
        if key not in _STATE["shapes"]:
            def ref_fn():
                if out is None:
                    return orig(module, z, cch, cfg, ending, ln_arith=ln_arith, fma_flags=fma_flags)
                scratch = _scratch_like(out)
                r = orig(module, z, cch, cfg, ending, ln_arith=ln_arith, fma_flags=fma_flags, out=scratch)
                if head_major and HMmod is not None:      # keep the caller's tag: the reference call tagged only its scratch dict
                    pass
                return r
            _shape_check(key, res, ref_fn)
        return res
    prologue.__wrapped__ = orig
    prologue._procuda = "hm" if head_major else "rm"
    return prologue


def install(verbose=True):
    """Engage the lever (idempotent).  Raises Refused (by name) outside the install-time envelope."""
    if _STATE["installed"]:
        return stats()
    if os.environ.get("PTX_BLK", "0") != "2":
        raise Refused("protenix_fpf_triatt_procuda: requires the BLK2 block path (PTX_BLK=2, the kit's env.sh)")
    if not torch.cuda.is_available():
        raise Refused("protenix_fpf_triatt_procuda: CUDA not available")
    cc = torch.cuda.get_device_capability()
    if cc[0] != 9:
        raise Refused(f"protenix_fpf_triatt_procuda: no binary for cc {cc[0]}.{cc[1]} (the shipped kernel is sm_90a)")
    import fpf_mkpf as MKP
    st = getattr(MKP, "_STATE", {})
    if not st.get("installed") or "triatt_prologue" not in st.get("patched", ()):
        raise Refused("protenix_fpf_triatt_procuda: requires MK-PF F1 installed on the tri-attention prologue (PTX_MK_PF=F1) — the kernel reproduces the F1 cell's arithmetic")
    cells = st.get("cells") or MKP.resolve_cells()
    arith = os.environ.get("FPF_MKPF_LN", cells.get("ln_arith", "fused"))            # == fpf_mkpf.install()'s own expression
    fma = tuple(bool(int(f)) for f in cells.get("fma_flags", [1, 1, 1]))
    if arith != "welford" or fma != (True, True, True):
        raise Refused(f"protenix_fpf_triatt_procuda: the F1 cell here is arith={arith} fma={fma}; the kernel implements the 'welford' (1,1,1) cell only")
    from . import binding as B
    try:
        B.load()
    except Exception as e:
        raise Refused(f"protenix_fpf_triatt_procuda: {e}")
    _STATE["manifest"] = {k: B.manifest().get(k) for k in ("so_sha256", "src_sha256", "nvcc", "arch", "built_utc")}
    _STATE["cells"] = dict(f1=dict(cells["f1"]["cfg"]), ln_arith=arith, fma_flags=list(fma), mkpf_class=str(cells.get("class", "")))
    import fpf_mkpf.kernels as MK
    if not getattr(MK.prologue_ln, "_procuda", None):
        MK.prologue_ln = _make_prologue(MK.prologue_ln, B, head_major=False); _STATE["patched"].append("f1")
    HM = None
    for name in ("protenix_fpf_triatt_headsplit.prologue_hm", "fpf_triatt_headsplit.prologue_hm"):
        try:
            HM = __import__(name, fromlist=["prologue_ln_hm"]); break
        except ImportError:
            continue
    if HM is not None and not getattr(HM.prologue_ln_hm, "_procuda", None):
        HM.prologue_ln_hm = _make_prologue(HM.prologue_ln_hm, B, head_major=True, HMmod=HM); _STATE["patched"].append("f1hm")
    _STATE["installed"] = True
    _STATE["why"] = f"cc {cc[0]}.{cc[1]}; f1 cell {cells['f1']['cfg']} arith={arith}; binary {_STATE['manifest'].get('so_sha256', '?')[:12]}"
    try:
        import ptx_trunk2_levers as LEV
        LEV._STATS["applied"].append(marker()); LEV._STATS["procuda"] = _T
    except Exception:
        pass
    rep = os.environ.get("PTX_LEVER_REPORT")
    if rep:
        atexit.register(_write_report, rep)
    if verbose:
        _say(f"v{__version__} installed: {marker()} :: {_STATE['why']} patched={_STATE['patched']}")
    return stats()
