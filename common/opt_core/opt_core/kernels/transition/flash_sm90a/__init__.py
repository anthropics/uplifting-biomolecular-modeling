"""protenix_fpf_flash_transition (unit fpf_flash_transition) — the Protenix-v2 pair Transition (LayerNorm -> a|b projections -> SiLU(a)*b -> output projection, + residual) served by
one sm_90a CUDA kernel (csrc/flash_transition_sm90.cu) shipped as a prebuilt extension per (torch, CUDA) stack under prebuilt/<key>/.

Numerics class EXACT: the kernel reproduces the stock bf16-autocast chain bit for bit (bf16 x bf16 products accumulated in fp32 in ascending K,
the stock rounding points for a, silu(a), b*silu(a), the update and the residual add).  Lever `flash_transition` (PTX_FLASH_TRANSITION=1): the
LayerNorm is the stock module call (module.layernorm1) and the kernel consumes its output, exactly as in opt_core's fpf_transition.fn_residual which
this unit replaces on sm_90 for the (c_in=256, hidden=1024) cell.  Lever `flash_transition_ln` (PTX_FLASH_TRANSITION_LN=1, on top of the first):
the kernel consumes the RAW rows and computes the LayerNorm in its prologue with the arithmetic of Protenix's fast_layernorm kernel
(LayerNormForwardV2<bf16, float4>: per-lane sequential Welford, xor-butterfly merge, rcp/rsqrt approximations, flush-to-zero, bf16-cast weight/bias)
reproduced instruction for instruction -> the same bits as the FusedLayerNorm module (checked at load on the GPU with ln_rows(), and route 'module-ln'
whenever a module's LayerNorm is not that class / has no weight or bias); it removes one kernel launch and one bf16 round trip of the rows per call.

install() (called once by ptx_trunk2_levers._apply_blk2 when PTX_FLASH_TRANSITION=1, BEFORE the block path binds fn_residual):
  1. picks prebuilt/<torch>-cu<cuda>/ for the running torch, checks manifest.json (torch, CUDA, python tag, sha256 of the .so AND of csrc/*.cu),
  2. requires compute capability 9.0 and >= smem_bytes() of opt-in shared memory per block,
  3. loads the extension, runs the load-time numerical check on the GPU: (a) silu_table() == torch F.silu for all 65536 bf16 inputs, (b) torch.equal of
     fn_residual against the stock chain on synthetic tiles (M = 8, 333 and 65536 rows: partial tiles, odd tile count, a model-class row count;
     both residual forms), (c) with the LN
     lever: ln_rows() == the FusedLayerNorm module on the same rows and the fused call == the stock chain,
  4. rebinds fn_residual / fn of opt_core's fpf_transition.transition under BOTH import names (fpf_transition.transition and
     opt_core.kernels.fpf_transition.transition are distinct module objects) to this unit's functions (originals stay reachable as CORE_FN_*),
  and returns stats(); any failure raises RuntimeError("flash_transition: ...") so the mode refuses by name.  Nothing happens at import time.

Envelope (decided from shapes before the call, counted in stats()['calls']): CUDA bf16 input with x.shape[-1] == module.c_in == 256, hidden 1024
(NH % 64 == 0 is what the kernel needs; 1024 is the cell that was checked), eval mode, plain LinearNoBias projections, sm_90 device.  Anything else is
routed to the opt_core function that was bound before install() (route 'core', counted with its reason) — a declared route, never an exception fallback.
CUDA-graph capture: no host synchronisation, no data-dependent shapes, launches on the current stream; the packed weights are cached on the module
(kept alive for graph replays)."""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict, Optional

import torch

__version__ = "1.1.0"
UNIT = "fpf_flash_transition"
MARK = "FLASH_TR:"                                   # marker family appended by ptx_trunk2_levers: FLASH_TR:on(...) | FLASH_TR:refused(...) | FLASH_TR:unavailable(...)
HERE = os.path.dirname(os.path.abspath(__file__))
CELL = (256, 1024)                                    # (c_in, hidden) served by the kernel
LOADCHECK_TILES = (8, 333, 65536)                     # synthetic row counts compared against the stock chain at load (see load_check)
_EXT = None                                           # the loaded extension module
SWITCH_LN = "PTX_FLASH_TRANSITION_LN"                  # lever flash_transition_ln: "1" -> LayerNorm in the kernel prologue (install(ln=None) reads it)
_STATE: Dict[str, Any] = {"installed": False, "why": "not installed", "prebuilt": None, "marker": None, "loadcheck": None, "ln": False,
                          "calls": {"flash": 0, "flash_rows": 0, "flash_ln": 0, "module_ln": 0, "module_ln_reasons": {}, "core": 0, "core_reasons": {}}}
_FUSED_LN_CLS = None                                  # protenix FusedLayerNorm class (resolved at install when the LN lever is on)
CORE_FN_RESIDUAL = None                               # opt_core's functions as bound before install()
CORE_FN = None
CORE_MODULES = ("fpf_transition.transition", "opt_core.kernels.fpf_transition.transition")   # the same file is importable under both names (two module objects)
CORE_ORIGINALS: Dict[str, Any] = {}                   # id(function) -> function, every fn_residual/fn object seen before the rebind (for callers that check bindings)
_CC: Dict[str, tuple] = {}


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def stack_key() -> str:
    tv = torch.__version__.split("+")[0]
    return f"torch{tv}-cu{(torch.version.cuda or 'none').replace('.', '')}"


def prebuilt_dir() -> str:
    return os.path.join(HERE, "prebuilt", stack_key())


def _fail(msg: str):
    _STATE["installed"] = False; _STATE["why"] = msg
    raise RuntimeError(f"flash_transition: {msg}")


def load(check_source: bool = True):
    """Load the prebuilt extension for this stack after manifest checks; returns the module.  Raises RuntimeError('flash_transition: ...')."""
    global _EXT
    if _EXT is not None:
        return _EXT
    d = prebuilt_dir()
    man_p = os.path.join(d, "manifest.json")
    if not os.path.isfile(man_p):
        _fail(f"no prebuilt for stack {stack_key()} (looked for {man_p})")
    man = json.load(open(man_p))
    so = os.path.join(d, man["so"])
    if not os.path.isfile(so):
        _fail(f"prebuilt .so missing: {so}")
    if man.get("torch") != torch.__version__ or man.get("cuda") != torch.version.cuda:
        _fail(f"prebuilt was built for torch {man.get('torch')} / CUDA {man.get('cuda')}, running torch {torch.__version__} / CUDA {torch.version.cuda}")
    pytag = f"cp{sys.version_info[0]}{sys.version_info[1]}"
    if man.get("python_tag") != pytag:
        _fail(f"prebuilt python tag {man.get('python_tag')} != {pytag}")
    if _sha(so) != man.get("so_sha256"):
        _fail("sha256 of the prebuilt .so does not match manifest.json")
    if check_source:
        for fn, digest in (man.get("source_sha256") or {}).items():
            p = os.path.join(HERE, "csrc", fn)
            if not os.path.isfile(p) or _sha(p) != digest:
                _fail(f"csrc/{fn} does not match the source the prebuilt was built from (manifest source_sha256)")
    name = man["module_name"]
    loader = importlib.machinery.ExtensionFileLoader(name, so)
    spec = importlib.util.spec_from_file_location(name, so, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    for sym in ("flash_transition", "silu_table", "smem_bytes", "hidden_chunk"):
        if not hasattr(mod, sym):
            _fail(f"prebuilt module lacks symbol {sym}")
    _EXT = mod
    _STATE["prebuilt"] = {"dir": os.path.relpath(d, HERE), "so": man["so"], "so_sha256": man["so_sha256"][:12], "built_from": man.get("source_commit", "?")}
    return mod


def _device_ok(device) -> tuple:
    key = str(device)
    if key not in _CC:
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(idx)
        props = torch.cuda.get_device_properties(idx)
        smem = getattr(props, "shared_memory_per_block_optin", None)
        ok = (cc == (9, 0)) and (smem is None or _EXT is None or smem >= _EXT.smem_bytes())
        _CC[key] = (ok, f"cc{cc[0]}{cc[1]}" + ("" if ok else "-unsupported"))
    return _CC[key]


def _pack(module, device):
    """bf16 weight packs for the kernel, cached on the module and keyed on the parameters' identity/version (a reload or rebinding re-packs)."""
    wa, wb, wo = module.linear_no_bias_a.weight, module.linear_no_bias_b.weight, module.linear_no_bias.weight
    ln = module.layernorm1; lw, lb = getattr(ln, "weight", None), getattr(ln, "bias", None)
    key = (wa.data_ptr(), wb.data_ptr(), wo.data_ptr(), wa._version, wb._version, wo._version, str(device),
           None if lw is None else (lw.data_ptr(), lw._version), None if lb is None else (lb.data_ptr(), lb._version))
    c = getattr(module, "_flash_transition_pack", None)
    if isinstance(c, dict) and c.get("key") == key:
        return c
    bh = _EXT.hidden_chunk()
    with torch.no_grad():
        wa16 = wa.detach().to(device=device, dtype=torch.bfloat16); wb16 = wb.detach().to(device=device, dtype=torch.bfloat16)
        nh, c_in = wa16.shape
        w1p = torch.stack([wa16.reshape(nh // bh, bh, c_in), wb16.reshape(nh // bh, bh, c_in)], dim=1).reshape(2 * nh, c_in).contiguous()
        wo16 = wo.detach().to(device=device, dtype=torch.bfloat16).contiguous()          # [c_in, nh]
        lnw = None if lw is None else lw.detach().to(device=device, dtype=torch.bfloat16).contiguous()   # the module casts weight/bias to the input dtype
        lnb = None if lb is None else lb.detach().to(device=device, dtype=torch.bfloat16).contiguous()
    keep = getattr(module, "_flash_transition_pack", None)
    c = {"key": key, "w1p": w1p, "wo": wo16, "nh": nh, "lnw": lnw, "lnb": lnb, "eps": float(getattr(ln, "eps", 1e-5)), "prev": keep}   # 'prev' keeps an older pack referenced (captured graphs may hold its pointers)
    module._flash_transition_pack = c
    return c


def eligible(module, x) -> tuple:
    """(True, '') when the kernel serves this call; else (False, reason).  Pure function of shapes / dtypes / flags / device class."""
    if _EXT is None:
        return False, "not-loaded"
    if not (torch.is_tensor(x) and x.is_cuda):
        return False, "not-cuda"
    if x.dtype != torch.bfloat16:
        return False, f"dtype-{x.dtype}"
    if module.training:
        return False, "training"
    if x.dim() < 2 or x.numel() == 0:
        return False, "shape"
    c = int(module.c_in)
    if x.shape[-1] != c or c != CELL[0]:
        return False, f"c_in-{c}"
    nh = module.linear_no_bias.weight.shape[1]
    if int(nh) != CELL[1]:
        return False, f"hidden-{nh}"
    for lin in (module.linear_no_bias_a, module.linear_no_bias_b, module.linear_no_bias):
        if getattr(lin, "precision", None) is not None or getattr(lin, "bias", None) is not None:
            return False, "linear-variant"
    ok, word = _device_ok(x.device)
    if not ok:
        return False, word
    return True, ""


def ln_eligible(module) -> tuple:
    """(True, '') when the in-kernel LayerNorm serves module.layernorm1 (lever flash_transition_ln); else (False, reason) -> the module's own call."""
    if not _STATE["ln"]:
        return False, "lever-off"
    ln = getattr(module, "layernorm1", None)
    if _FUSED_LN_CLS is None or type(ln) is not _FUSED_LN_CLS:
        return False, f"class-{type(ln).__name__}"
    if getattr(ln, "weight", None) is None or getattr(ln, "bias", None) is None:
        return False, "no-affine"
    if tuple(getattr(ln, "normalized_shape", ())) != (CELL[0],):
        return False, "shape"
    return True, ""


def _forward(module, x, residual: bool):
    ok, why = eligible(module, x)
    if not ok:
        calls = _STATE["calls"]; calls["core"] += 1; calls["core_reasons"][why] = calls["core_reasons"].get(why, 0) + 1
        core = CORE_FN_RESIDUAL if residual else CORE_FN
        if core is None:
            raise RuntimeError(f"flash_transition: call outside the envelope ({why}) and no opt_core function bound")
        return core(module, x)
    other_dims = x.shape[:-1]
    c = x.shape[-1]
    size = x.shape[-2]
    x2 = x.reshape(-1, c)
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != c) or (x2.data_ptr() % 16) != 0:
        x2 = x2.contiguous()
    pack = _pack(module, x2.device)
    chunk_num = 1 if size < 3200 else 8                  # the stock Transition chunking (rows are independent: numerically irrelevant; kept for memory parity)
    outputs = torch.empty((x2.shape[0], c), dtype=torch.bfloat16, device=x2.device)
    start = 0
    ln_ok, ln_why = ln_eligible(module)
    calls = _STATE["calls"]
    if ln_ok:                                            # lever flash_transition_ln: LayerNorm inside the kernel prologue (same bits as the module call)
        for chunk in torch.chunk(x2, chunk_num, dim=-2):
            n = chunk.shape[0]
            if n == 0:
                continue
            _EXT.flash_transition_ln(chunk, pack["w1p"], pack["wo"], chunk if residual else None, outputs[start:start + n], pack["lnw"], pack["lnb"], pack["eps"])
            start += n
        calls["flash"] += 1; calls["flash_rows"] += int(x2.shape[0]); calls["flash_ln"] += 1
        return outputs.reshape(*other_dims, c)
    if _STATE["ln"]:
        calls["module_ln"] += 1; calls["module_ln_reasons"][ln_why] = calls["module_ln_reasons"].get(ln_why, 0) + 1
    for chunk in torch.chunk(x2, chunk_num, dim=-2):
        n = chunk.shape[0]
        if n == 0:
            continue
        y = module.layernorm1(chunk)                       # the stock LayerNorm call
        if y.dtype != torch.bfloat16:
            y = y.to(torch.bfloat16)
        if y.stride(-1) != 1 or (n > 1 and y.stride(0) != c) or (y.data_ptr() % 16) != 0:
            y = y.contiguous()
        _EXT.flash_transition(y, pack["w1p"], pack["wo"], chunk if residual else None, outputs[start:start + n])
        del y
        start += n
    calls["flash"] += 1; calls["flash_rows"] += int(x2.shape[0])
    return outputs.reshape(*other_dims, c)


def ln_rows(module, x2):
    """LayerNorm of the rows of x2 [M, 256] (bf16) with the in-kernel arithmetic (load-time check / tools)."""
    pack = _pack(module, x2.device)
    y = torch.empty((x2.shape[0], x2.shape[1]), dtype=torch.bfloat16, device=x2.device)
    _EXT.ln_rows(x2, pack["lnw"], pack["lnb"], y, pack["eps"])
    return y


def fn_residual(module, x):
    """x + transition(x), the residual add inside the kernel epilogue: bf16(fp32(x) + fp32(bf16(update))) == the stock `z += transition(z)`."""
    return _forward(module, x, residual=True)


def fn(module, x):
    """transition(x) (the caller adds)."""
    return _forward(module, x, residual=False)


def load_check(device=None) -> Dict[str, Any]:
    """Load-time numerical load-time check on the GPU.  (a) SiLU: the kernel's bf16 silu for every one of the 65536 bf16 bit patterns == torch F.silu on
    bf16.  (b) Whole op: fn_residual / fn on synthetic tiles == the stock bf16-autocast chain (torch.equal).  Returns a dict; 'ok' False names the failure."""
    import torch.nn.functional as F
    from protenix.model.modules.primitives import Transition
    dev = torch.device(device if device is not None else "cuda")
    rep: Dict[str, Any] = {"ok": False}
    bits = torch.arange(65536, dtype=torch.int32, device=dev).to(torch.int16).view(torch.bfloat16)
    want = F.silu(bits)
    got = torch.empty(65536, dtype=torch.bfloat16, device=dev); _EXT.silu_table(got)
    both_nan = want.isnan() & got.isnan()
    n_bad = int(((want.view(torch.int16) != got.view(torch.int16)) & ~both_nan).sum().item())
    rep["silu_mismatch"] = n_bad
    if n_bad:
        rep["why"] = f"silu table differs from torch on {n_bad} of 65536 bf16 inputs"; return rep
    g = torch.Generator(device="cpu").manual_seed(20260911)
    mod = Transition(c_in=CELL[0], n=CELL[1] // CELL[0]).to(dev).eval()
    with torch.no_grad():
        for lin in (mod.linear_no_bias_a, mod.linear_no_bias_b, mod.linear_no_bias):
            lin.weight.copy_(torch.randn(lin.weight.shape, generator=g) * 0.06)
        mod.layernorm1.weight.copy_(1.0 + 0.1 * torch.randn(mod.layernorm1.weight.shape, generator=g))
        mod.layernorm1.bias.copy_(0.05 * torch.randn(mod.layernorm1.bias.shape, generator=g))
    cases = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for m_rows in LOADCHECK_TILES:                                                        # 65536: a row count of the model's class, so a GEMM-library
            # heuristic that changed the stock result at large M would be caught here by name, in-process; 333 / 8: partial tiles, odd tile counts
            x = (torch.randn((m_rows, CELL[0]), generator=g) * 1.5).to(dev, torch.bfloat16)
            y = mod.layernorm1(x)
            a = F.linear(y, mod.linear_no_bias_a.weight); a = F.silu(a); b = F.linear(y, mod.linear_no_bias_b.weight); b = b * a
            u = F.linear(b, mod.linear_no_bias.weight)
            want_r = x.clone(); want_r += u
            got_r = fn_residual(mod, x); got_u = fn(mod, x)
            torch.cuda.synchronize()
            case = {"M": m_rows, "residual_equal": bool(torch.equal(got_r, want_r)), "update_equal": bool(torch.equal(got_u, u)),
                    "max_abs": float((got_r.float() - want_r.float()).abs().max())}
            if _STATE["ln"]:                                                                  # the in-kernel LayerNorm alone vs the module, and which route served the tile
                case["ln_rows_equal"] = bool(torch.equal(ln_rows(mod, x), y)); case["ln_route"] = "fused" if ln_eligible(mod)[0] else "module:" + ln_eligible(mod)[1]
            cases.append(case)
    for k in ("flash", "flash_rows", "flash_ln", "module_ln"):
        _STATE["calls"][k] = 0                                                                # the load-time check calls are not model calls
    _STATE["calls"]["module_ln_reasons"] = {}
    rep["cases"] = cases
    bad = [c for c in cases if not (c["residual_equal"] and c["update_equal"] and c.get("ln_rows_equal", True))]
    if bad:
        rep["why"] = f"synthetic tile mismatch: {bad}"; return rep
    if _STATE["ln"] and any(c.get("ln_route") != "fused" for c in cases):
        rep["why"] = f"LN lever on but the load-time tiles were not served by the fused LayerNorm: {[c.get('ln_route') for c in cases]}"; return rep
    rep["ok"] = True
    return rep


def install(verbose: bool = True, ln: Optional[bool] = None) -> Dict[str, Any]:
    """Load + load-time check + rebind opt_core's fpf_transition.fn_residual / fn.  ln=True (or env PTX_FLASH_TRANSITION_LN=1 when ln is None) also
    engages lever flash_transition_ln (LayerNorm in the kernel prologue); it requires Protenix's FusedLayerNorm to be the LayerNorm class in use
    (LAYERNORM_TYPE=fast_layernorm, the kit's layernorm_fast lever) and refuses by name otherwise.  Raises RuntimeError('flash_transition: <reason>')."""
    global CORE_FN_RESIDUAL, CORE_FN, _FUSED_LN_CLS
    t0 = time.time()
    if ln is None:
        ln = os.environ.get(SWITCH_LN, "0") == "1"
    if not torch.cuda.is_available():
        _fail("no CUDA device")
    load()
    if ln:
        try:
            from protenix.model.layer_norm.layer_norm import FusedLayerNorm as _F
        except Exception as e:                                    # noqa: BLE001
            _fail(f"flash_transition_ln: protenix FusedLayerNorm not importable ({type(e).__name__}: {e})")
        from protenix.model.triangular import layers as _layers            # protenix's LayerNorm factory: FusedLayerNorm iff LAYERNORM_TYPE=fast_layernorm at import
        if not getattr(_layers, "fastln_is_installed", False):
            _fail("flash_transition_ln: protenix LayerNorm modules are not FusedLayerNorm in this process (LAYERNORM_TYPE != fast_layernorm; lever layernorm_fast)")
        _FUSED_LN_CLS = _F
    _STATE["ln"] = bool(ln)
    ok, word = _device_ok(torch.device("cuda", torch.cuda.current_device()))
    if not ok:
        _fail(f"device {torch.cuda.get_device_name()} ({word}): the prebuilt kernel is sm_90a only")
    import importlib
    mods = []
    for name in CORE_MODULES:                                     # ptx_trunk2_levers imports `fpf_transition.transition`; tools may import the opt_core-qualified name
        try:
            m = importlib.import_module(name)
        except ImportError:
            continue
        if all(m is not q for q in mods):
            mods.append(m)
    if not mods:
        _fail("opt_core fpf_transition.transition is not importable")
    for m in mods:
        for attr in ("fn_residual", "fn"):
            f = getattr(m, attr, None)
            if f is not None and f is not fn_residual and f is not fn:
                CORE_ORIGINALS[id(f)] = f
    if CORE_FN_RESIDUAL is None:
        CORE_FN_RESIDUAL, CORE_FN = mods[0].fn_residual, mods[0].fn
    rep = load_check()
    _STATE["loadcheck"] = rep
    if not rep.get("ok"):
        _fail("load-time check failed: " + str(rep.get("why")))
    for m in mods:
        m.fn_residual = fn_residual
        m.fn = fn
    _STATE["rebound_modules"] = [m.__name__ for m in mods]
    _STATE["installed"] = True; _STATE["why"] = ""
    _STATE["marker"] = (f"on(sm90a,prebuilt={_STATE['prebuilt']['dir']},cell={CELL[0]}x{CELL[1]},class=EXACT,ln={'fused' if _STATE['ln'] else 'module'},"
                        f"loadcheck=silu65536+tiles({','.join(str(m) for m in LOADCHECK_TILES)}){'+lnrows' if _STATE['ln'] else ''},{time.time() - t0:.1f}s)")
    if verbose:
        print(f"[{UNIT}] installed: {_STATE['marker']} (opt_core fpf_transition.fn_residual/fn rebound; other cells -> opt_core)", file=sys.stderr, flush=True)
    return stats()


def stats() -> Dict[str, Any]:
    return {"unit": UNIT, "version": __version__, "installed": _STATE["installed"], "ln": _STATE["ln"], "why": _STATE["why"], "marker": _STATE["marker"],
            "prebuilt": _STATE["prebuilt"], "calls": _STATE["calls"], "loadcheck": _STATE["loadcheck"], "rebound_modules": _STATE.get("rebound_modules")}


def report() -> Dict[str, Any]:
    """Record for the shared lever report: unit, pid, calls (flash / rows / core + reasons)."""
    c = _STATE["calls"]
    return {"unit": UNIT, "pid": os.getpid(), "installed": _STATE["installed"], "ln": _STATE["ln"], "calls": c["flash"], "rows": c["flash_rows"],
            "routes": {"flash": c["flash"], "flash_ln": c["flash_ln"], "module_ln": c["module_ln"], "module_ln_reasons": dict(c["module_ln_reasons"]),
                       "core": c["core"], "core_reasons": dict(c["core_reasons"])}}
