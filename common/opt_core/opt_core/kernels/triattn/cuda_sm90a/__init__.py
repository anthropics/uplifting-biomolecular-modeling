"""fpf_triattn_cuda — triangle-attention forward for sm_90 (H100/H200): a hand-written CUDA kernel (csrc/triattn_mw.cu, mma.sync
m16n8k16 bf16 -> fp32, TMA-fed shared-memory ring, streaming softmax) behind the cuEquivariance / K2B `triangle_attention` call convention.

    out = attn(q, k, v, bias, mask=None, scale=None)

  q, k, v : [B, N, H, S, D] bf16 CUDA tensors (rank < 5 -> leading singleton dims); D = 32; N (pair rows) may differ from S (keys ==
            queries); read in place when the last stride is 1, the other strides are non-negative multiples of 8 elements and the base is
            16-byte aligned (contiguous tensors, row slices q[:, i0:i1] and the block-core prologue's outputs all qualify); anything else is
            copied once, counted in COUNTS["copy"].
  bias    : [B, 1, H, S, S] fp32 | bf16 | fp16 (rank < 5 -> leading singleton dims), any strides; shared by the N pair rows.  Staged once
            per call to an fp32 [B, H, ceil128(S), ceil64(S)] copy pre-divided by `scale` (fp32 arithmetic).
  mask    : [B, N, 1, 1, S] bool (True = attend) or None; any pattern; a pair row with no attended key attends uniformly to all S keys
            (mean of v), as cuEquivariance does.
  scale   : softmax scale, default D ** -0.5.
  returns : new contiguous [B, N, H, S, D] bf16 tensor.  Deterministic; capturable in a CUDA graph after the first call of a process
            (the first call sizes a per-device int32 scratch buffer); launches on the current stream; no host synchronisation.

Numerics: Q.K^T and P.V on the tensor cores from bf16 operands with fp32 accumulation, bias added in fp32, softmax in fp32 (base 2,
ex2.approx) with a per-row power-of-two offset instead of a running maximum, P rounded to bf16 for the P.V product, row sums in fp32
from the same bf16 P, output bf16.  Same class as the K2B Triton kernel and the cuEquivariance kernel (bf16 tensor-core attention with
fp32 softmax); not bitwise equal to either.

Envelope: compute capability 9.0 only (the prebuilt binary is sm_90a); D = 32; 1 <= S <= 65536; B*H <= 65535; bf16 operands; masks of
shape [B, N, 1, 1, S] only.  Inputs outside the envelope raise `Refused` naming the reason -- this module never substitutes another kernel.

Loading: `install()` loads prebuilt/<stack key>/<so> for the running stack (stack key = torch version + python SOABI + sm, checked
against manifest.json: sha256 of the binary, torch and CUDA versions, sha256 of every file under csrc/), then runs the load-time
load-check: the kernel recomputes the small deterministic cases of `loadcheck_cases()` and must reproduce prebuilt/<key>/loadcheck.pt
bit for bit (and stay within a fixed bound of an fp32 evaluation).  A missing binary, a manifest mismatch, a load error or a load-check
mismatch raise `Refused`.  `attn` calls `install()` on first use; callers that want the check at start-up call `install()` themselves.
`build_prebuilt.py` (nvcc through torch.utils.cpp_extension) produces the prebuilt directory for a stack.

Device memory per call besides the output (transient, allocated through the caching allocator on the current stream): the staged bias
B*H*ceil128(S)*ceil64(S)*4 bytes (H = 8: 8 MiB at S = 512, 32 MiB at 1024, 128 MiB at 2048, 288 MiB at 3072, 512 MiB at 4096); with a
mask, B*N*S + 9*B*N*ceil(S/64) + B*ceil64(S) bytes of tables; a per-device int32 tile list of 12*ceil(S/128)*ceil(N/3)*B*H bytes (at least
256 KiB, kept).  No [rows, S, S] logits are materialised.
"""
from __future__ import annotations

import hashlib
import importlib.machinery
import importlib.util
import json
import math
import os
import sys
import sysconfig
import time
from typing import Any, Dict, Optional

import torch

__all__ = ["attn", "install", "report", "build_ext", "stack_key", "Refused", "COUNTS", "PROVIDER", "reference", "stage_bias", "loadcheck_cases"]
__version__ = "1.2.0"

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXT = None
_INSTALL_REPORT: Dict[str, Any] = {}
EXT_NAME = "triattn_cuda_sm90"                     # python module name of the extension (PYBIND11_MODULE / TORCH_EXTENSION_NAME)
PROVIDER = {"name": "triattn_cuda", "label": "TIER2", "kv_pad": 1, "wants_kv_len": False, "out_layouts": ("bnhsd", "bhnsd", "bnshd"), "accepts_out": True}   # ptx_trunk2_levers provider meta (unpadded layout)


class Refused(NotImplementedError):
    """The entry cannot serve: no prebuilt for this stack, failed load-check, or an input outside what the kernel serves.  Typed and
    named; callers never receive another kernel's result from this module."""


class _Counts(dict):
    """Named slow paths, counted (device censuses folded in on read; reading synchronises):
       copy            q/k/v operand copied to a TMA-legal layout (0 on the Protenix prologue's tensors and their row slices);
       fix_tiles       CTA tiles recomputed by the max-tracking pass (logit swings beyond ~2^100 inside a row; 0 on model data);
       list_rowgroups  row groups computed by the general (per-row mask) pass because a row's mask differs from the batch OR."""
    def _sync(self):
        tiles = rows = 0
        for b in _FIX.values():
            c = b[:2].tolist(); tiles += int(c[0]); rows += int(c[1])
        dict.__setitem__(self, "fix_tiles", tiles); dict.__setitem__(self, "list_rowgroups", rows)
    def __getitem__(self, k): self._sync(); return dict.__getitem__(self, k)
    def get(self, k, d=None): self._sync(); return dict.get(self, k, d)
    def items(self): self._sync(); return dict.items(self)
    def __iter__(self): self._sync(); return dict.__iter__(self)
    def __repr__(self): self._sync(); return dict.__repr__(self)
    def copy(self): self._sync(); return dict(dict.items(self))


COUNTS = _Counts(copy=0, fix_tiles=0, list_rowgroups=0, calls=0)
_FIX: Dict[Any, torch.Tensor] = {}   # device -> int32 [3 + 3 * CTA tiles]: [0] fix-tile census, [1] list row-group census, [2] this call's count, [3:] tile list
PRESIZE_TOKENS = 4096          # install() sizes the tile-list scratch for square calls up to this many tokens at B*H = 8 (4 MiB); see install()


# ----------------------------------------------------------------------------------------------------------------- build / load ---
def stack_key() -> str:
    """'<torch version>-<SOABI>-sm<cc>' of the running process, e.g. torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90."""
    cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    return f"torch{torch.__version__}-{sysconfig.get_config_var('SOABI')}-sm{cc[0]}{cc[1]}"


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def source_sha256() -> Dict[str, str]:
    d = os.path.join(_HERE, "csrc")
    return {fn: _sha(os.path.join(d, fn)) for fn in sorted(os.listdir(d)) if fn.endswith((".cu", ".h", ".cuh"))}


NVCC_FLAGS = ["-O3", "-std=c++17", "-gencode", "arch=compute_90a,code=sm_90a", "--expt-relaxed-constexpr", "-DNDEBUG", "-Xptxas", "-v"]


def build_ext(build_dir: Optional[str] = None, verbose: bool = False):
    """Build the extension with nvcc (torch.utils.cpp_extension.load) and import it.  Used by build_prebuilt.py and install(allow_build=True).
    The sources are compiled from a neutral staging copy (<build dir>/protenix_fpf_triattn_cuda/csrc) with the staging prefix mapped away,
    so the binary carries no build-machine paths (no -lineinfo either)."""
    import shutil, tempfile
    from torch.utils.cpp_extension import load
    bdir = build_dir or tempfile.mkdtemp(prefix="ptx_ext_")
    os.makedirs(bdir, exist_ok=True)
    stage = os.path.join(bdir, "protenix_fpf_triattn_cuda", "csrc")
    if os.path.exists(stage): shutil.rmtree(stage)
    shutil.copytree(os.path.join(_HERE, "csrc"), stage)
    pmap = f"-ffile-prefix-map={bdir}/="
    return load(name=EXT_NAME, sources=[os.path.join(stage, "triattn_mw.cu")], extra_include_paths=[stage],
                extra_cuda_cflags=list(NVCC_FLAGS) + ["-Xcompiler", pmap], extra_cflags=["-O3", "-std=c++17", pmap], verbose=verbose,
                build_directory=bdir)


def _load_so(path: str, name: str = EXT_NAME):
    loader = importlib.machinery.ExtensionFileLoader(name, path)
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[name] = mod
    return mod


def prebuilt_dir(root: Optional[str] = None) -> str:
    return os.path.join(root or os.path.join(_HERE, "prebuilt"), stack_key())


def install(prebuilt_root: Optional[str] = None, allow_build: bool = False) -> Dict[str, Any]:
    """Load + load-check once per process; returns the install record (also kept in _INSTALL_REPORT).  Raises Refused.
    allow_build=True builds the extension with nvcc when no prebuilt directory exists for the stack (development; the kit never passes it)."""
    global _EXT
    if _EXT is not None:
        return _INSTALL_REPORT
    if not torch.cuda.is_available():
        raise Refused("triattn_cuda: CUDA is not available")
    cc = torch.cuda.get_device_capability()
    if cc != (9, 0):
        raise Refused(f"triattn_cuda: compute capability {cc[0]}.{cc[1]}: the kernel is built for sm_90a (H100/H200) only")
    t0 = time.time()
    rep: Dict[str, Any] = {"stack_key": stack_key(), "route": None}
    pdir = prebuilt_dir(prebuilt_root)
    mod = None
    man_path = os.path.join(pdir, "manifest.json")
    if os.path.exists(man_path):
        man = json.load(open(man_path))
        so = os.path.join(pdir, man["so"])
        checks = {"so_exists": os.path.exists(so)}
        if checks["so_exists"]:
            from opt_core.gates import binary_refusal  # noqa: PLC0415
            why = binary_refusal(so)
            if why:
                raise Refused(f"triattn_cuda: prebuilt {so} refused: {why}")
            checks["so_sha256"] = _sha(so) == man["so_sha256"]
        checks["torch"] = torch.__version__ == man["torch"]
        checks["cuda"] = torch.version.cuda == man["cuda"]
        checks["sources"] = (source_sha256() == man.get("source_sha256")) if os.path.isdir(os.path.join(_HERE, "csrc")) else True
        rep["prebuilt"] = {"dir": pdir, "checks": checks, "built_utc": man.get("built_utc"), "so_sha256": man.get("so_sha256")}
        if not all(checks.values()):
            raise Refused(f"triattn_cuda: prebuilt at {pdir} does not match this process: {json.dumps(checks)}")
        try:
            mod = _load_so(so, man.get("module_name", EXT_NAME))
        except Exception as e:  # noqa: BLE001
            raise Refused(f"triattn_cuda: loading {so} failed: {e!r}") from e
        rep["route"] = "prebuilt"
    elif allow_build:
        mod = build_ext()
        rep["route"] = "built (allow_build=True)"
    else:
        raise Refused(f"triattn_cuda: no prebuilt extension for stack {stack_key()} under {os.path.dirname(pdir)}")
    for sym in ("fwd", "stage_bias", "stage_mask", "fix_elems", "smem_bytes"):
        if not hasattr(mod, sym):
            raise Refused(f"triattn_cuda: extension lacks symbol {sym}")
    _EXT = mod
    sc = _loadcheck(pdir, required=(rep["route"] == "prebuilt"))
    rep["loadcheck"] = sc
    if not sc["ok"]:
        _EXT = None
        raise Refused(f"triattn_cuda: load-check failed: {sc}")
    # Pre-size the per-device tile-list scratch so that no later call up to PRESIZE_TOKENS (square, B*H = 8) has to grow it: the kit captures
    # CUDA graphs of the block stack only far below that size, so every allocation the op makes inside a capture is then a plain per-call
    # transient (output, staged bias, mask tables) from the capturing allocator, never a kept buffer.  Larger calls grow it eagerly (no graphs there).
    dev = torch.device("cuda", torch.cuda.current_device())
    _fix_buffer(dev, int(_EXT.fix_elems(1, PRESIZE_TOKENS, 8, PRESIZE_TOKENS)))
    rep["scratch_presized_tokens"] = PRESIZE_TOKENS; rep["scratch_bytes"] = int(_FIX[dev].numel()) * 4
    rep["load_s"] = round(time.time() - t0, 2)
    _INSTALL_REPORT.clear(); _INSTALL_REPORT.update(rep)
    return _INSTALL_REPORT


# ------------------------------------------------------------------------------------------------------------------- load-check ---
def loadcheck_cases(device="cuda"):
    """The baked load-check inputs: small, deterministic (CPU generator), exercising the mask-free pass with ragged S, the batch-OR
    mask fold, the per-row (list) pass with an interior hole and a fully-masked row, N != S, B = 2 and a row slice view."""
    g = torch.Generator(device="cpu").manual_seed(20260911)
    def rnd(*sh, s=1.0):
        return (torch.randn(*sh, generator=g) * s)
    cases = []
    # 1. unmasked, ragged S (2 q tiles, 3 key tiles + 8), N != multiple of 3, H = 2
    B, N, H, S, D = 1, 7, 2, 200, 32
    q, k, v = (rnd(B, N, H, S, D).to(torch.bfloat16) for _ in range(3)); bias = rnd(B, 1, H, S, S, s=2.0)
    cases.append(dict(name="nomask_S200", q=q, k=k, v=v, bias=bias, mask=None))
    # 2. B=2, padding mask shared by all rows of element 0 (batch-OR fold), element 1: one interior hole row + one fully masked row (list pass)
    B, N, H, S = 2, 5, 2, 150
    q, k, v = (rnd(B, N, H, S, D).to(torch.bfloat16) for _ in range(3)); bias = rnd(B, 1, H, S, S, s=3.0).to(torch.bfloat16)
    mask = torch.ones(B, N, 1, 1, S, dtype=torch.bool)
    mask[0, :, 0, 0, 131:] = False
    mask[1, 2, 0, 0, 40:77] = False
    mask[1, 4, 0, 0, :] = False
    cases.append(dict(name="mask_B2_S150", q=q, k=k, v=v, bias=bias, mask=mask))
    # 3. N (4 rows, a slice view of a 9-row tensor) != S, fp32 bias with large dynamic range, prefix mask of odd half-tile length
    B, N0, H, S = 1, 9, 2, 260
    q0, k0, v0 = (rnd(B, N0, H, S, D).to(torch.bfloat16) for _ in range(3)); bias = rnd(B, 1, H, S, S, s=12.0)
    mask = (torch.arange(S) < 225)[None, None, None, None, :].expand(B, 4, 1, 1, S).contiguous()
    cases.append(dict(name="rows2to6_S260_prefix", q=q0[:, 2:6], k=k0[:, 2:6], v=v0[:, 2:6], bias=bias, mask=mask))
    out = []
    for c in cases:
        out.append({kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for kk, vv in c.items()})
    return out


def _loadcheck(pdir: str, required: bool = True) -> Dict[str, Any]:
    ref_path = os.path.join(pdir, "loadcheck.pt")
    if not os.path.exists(ref_path):
        if required:
            return {"ok": False, "expected": None, "cases": {}, "mode": f"missing {ref_path}"}
        ref_path = None
    cases = loadcheck_cases()
    res: Dict[str, Any] = {"expected": ref_path, "cases": {}}
    ok = True
    baked = torch.load(ref_path, map_location="cuda") if ref_path else None
    for c in cases:
        o = attn(c["q"], c["k"], c["v"], c["bias"], mask=c["mask"])
        r32 = reference(c["q"], c["k"], c["v"], c["bias"], c["mask"])
        err = (o.float() - r32).abs().max().item()
        entry = {"max_abs_vs_fp32": err, "finite": bool(torch.isfinite(o.float()).all())}
        good = entry["finite"] and err < 2e-2
        if baked is not None:
            entry["bitwise"] = bool(c["name"] in baked and torch.equal(o, baked[c["name"]]))
            good = good and entry["bitwise"]
        entry["ok"] = good; ok = ok and good
        res["cases"][c["name"]] = entry
    res["ok"] = ok
    res["mode"] = "bitwise vs loadcheck.pt + fp32 bound" if baked is not None else "fp32 bound only (no loadcheck.pt for this stack)"
    return res


def report() -> Dict[str, Any]:
    """Per-process record for the caller's accounting: install route + checks and the named-route counters."""
    r = {"unit": __name__, "version": __version__, "pid": os.getpid(), "installed": _EXT is not None}
    r.update({k: _INSTALL_REPORT.get(k) for k in ("stack_key", "route", "load_s", "scratch_presized_tokens", "scratch_bytes")})
    if "prebuilt" in _INSTALL_REPORT:
        r["so_sha256"] = _INSTALL_REPORT["prebuilt"].get("so_sha256")
    if "loadcheck" in _INSTALL_REPORT:
        r["loadcheck_ok"] = _INSTALL_REPORT["loadcheck"].get("ok"); r["loadcheck_mode"] = _INSTALL_REPORT["loadcheck"].get("mode")
    r["counts"] = COUNTS.copy() if _EXT is not None else dict(dict.items(COUNTS))
    return r


def bake_loadcheck(path: str) -> Dict[str, str]:
    """Write the loaded kernel's outputs on loadcheck_cases() to `path` (build step, on the target GPU); returns short sha256 per case."""
    outs = {c["name"]: attn(c["q"], c["k"], c["v"], c["bias"], mask=c["mask"]).cpu() for c in loadcheck_cases()}
    torch.save(outs, path)
    return {k: hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()[:16] for k, v in outs.items()}


# --------------------------------------------------------------------------------------------------------------------- the op ---
def _ext():
    if _EXT is None:
        install()
    return _EXT


def _dims(t: torch.Tensor, n: int) -> torch.Tensor:
    while t.dim() < n:
        t = t.unsqueeze(0)
    return t


def _tma_ok(t: torch.Tensor) -> bool:      # K, V: TMA descriptors -- d stride 1, other strides multiples of 8 elements, 16-byte base
    return t.stride(-1) == 1 and all(int(s) % 8 == 0 and int(s) >= 0 for s in t.stride()[:-1]) and t.data_ptr() % 16 == 0


def _ldq_ok(t: torch.Tensor) -> bool:      # Q: 4-byte loads -- d stride 1, other strides even, 4-byte base
    return t.stride(-1) == 1 and all(int(s) % 2 == 0 and int(s) >= 0 for s in t.stride()[:-1]) and t.data_ptr() % 4 == 0


def _fix_buffer(device, n: int) -> torch.Tensor:
    b = _FIX.get(device)
    if b is None or b.numel() < n:
        nb = torch.zeros(max(n, 1 << 16), dtype=torch.int32, device=device)
        if b is not None:
            nb[:2] = b[:2]
        _FIX[device] = b = nb
    return b


def stage_mask(mask: torch.Tensor, B: int, N: int, S: int):
    """[B,N,1,1,S] (True = attend) -> (mask_u8 [B,N,S], keyany [B,S64] u8, rowkind [B,N] u8, irregular row-group list, ktend [B],
    mtab int32 [B,N,nkt,2] / ctab u8 [B,N,nkt] / ktendr int32 [B,N] per-row tables of the general passes)."""
    if mask.dim() == 5 and (mask.shape[2] != 1 or mask.shape[3] != 1):
        raise Refused(f"triattn_cuda: mask shape {tuple(mask.shape)}: per-head / per-query masks are not served (expect [B,N,1,1,S])")
    if mask.numel() != B * N * S and not (mask.dim() == 5 and mask.shape[0] == B and mask.shape[1] == N and mask.shape[4] == S):
        raise ValueError(f"triangle_attention: input mask must have shape {(B, N, 1, 1, S)} but got: {tuple(mask.shape)}")
    m = mask.reshape(B, N, S)
    if m.dtype == torch.bool:
        m = m.contiguous().view(torch.uint8)
    elif m.dtype != torch.uint8:
        m = (m != 0).view(torch.uint8)
    m = m.contiguous()
    keyany, rowkind, irr, ktend, mtab, ctab, ktendr = _ext().stage_mask(m)
    return m, keyany, rowkind, irr, ktend, mtab, ctab, ktendr


def stage_bias(bias: torch.Tensor, scale: float, keyany: Optional[torch.Tensor] = None) -> torch.Tensor:
    """[B,1,H,S,S] -> fp32 [B,H,ceil128(S),ceil64(S)] / scale in mma C-fragment order (-inf on keys >= S and on keys no row attends)."""
    return _ext().stage_bias(_dims(bias, 5)[:, 0], float(scale), keyany)


OUT_LAYOUTS = ("bnhsd", "bhnsd", "bnshd")      # physical order of the output buffer; the returned tensor is always the logical [B,N,H,S,D] view of it


def attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias: torch.Tensor, mask: Optional[torch.Tensor] = None,
         scale: Optional[float] = None, *, bias_staged: Optional[torch.Tensor] = None, out: Optional[torch.Tensor] = None,
         out_layout: str = "bnhsd", _dbg: int = 0) -> torch.Tensor:
    """Triangle attention forward, cuEquivariance `triangle_attention` convention (see module docstring).

    Output placement (for fused epilogues; values are identical bit for bit in every layout, only the strides differ):
      out_layout="bnhsd" (default): a new contiguous [B,N,H,S,D] tensor (= cuEquivariance / K2B).
      out_layout="bhnsd": head-major buffer [B,H,N,S,D]; returned as its [B,N,H,S,D] permuted view (o.permute(0,2,1,3,4) is contiguous).
      out_layout="bnshd": buffer [B,N,S,H,D] = [rows, queries, H*D], the layout an output projection consumes without a permute;
                          returned as its [B,N,H,S,D] view (o.permute(0,1,3,2,4).reshape(B,N,S,H*D) is free).
      out=<tensor>: caller-provided bf16 CUDA tensor of logical shape [B,N,H,S,D] with d-stride 1 and even, non-negative other strides
                    (any of the above, or a slice of a larger buffer); written in place and returned."""
    ext = _ext()
    if out_layout not in OUT_LAYOUTS:
        raise ValueError(f"triattn_cuda: out_layout {out_layout!r} not in {OUT_LAYOUTS}")
    q = _dims(q, 5); k = _dims(k, 5); v = _dims(v, 5); bias = _dims(bias, 5)
    if mask is not None:
        mask = _dims(mask, 5)
    B, N, H, S, D = q.shape
    if tuple(k.shape) != (B, N, H, S, D) or tuple(v.shape) != (B, N, H, S, D):
        if k.dim() == 5 and k.shape[3] != S and tuple(k.shape[:3]) == (B, N, H):
            raise Refused(f"triattn_cuda: S_q != S_kv ({S} vs {k.shape[3]}) is not served")
        raise ValueError(f"triangle_attention: q/k/v shape mismatch {tuple(q.shape)} {tuple(k.shape)} {tuple(v.shape)}")
    if tuple(bias.shape) != (B, 1, H, S, S):
        raise ValueError(f"triangle_attention: input bias must have shape {(B, 1, H, S, S)} but got: {tuple(bias.shape)}")
    if mask is not None and tuple(mask.shape) != (B, N, 1, 1, S):
        if mask.dim() == 5 and mask.shape[2:4] != (1, 1):
            raise Refused(f"triattn_cuda: mask shape {tuple(mask.shape)}: per-head / per-query masks are not served")
        raise ValueError(f"triangle_attention: input mask must have shape {(B, N, 1, 1, S)} but got: {tuple(mask.shape)}")
    if torch.is_autocast_enabled():                                 # cuEq semantics: operands follow the autocast dtype
        adt = torch.get_autocast_dtype("cuda")
        q, k, v = (t if t.dtype == adt else t.to(adt) for t in (q, k, v))
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise Refused(f"triattn_cuda: dtype {q.dtype}/{k.dtype}/{v.dtype}: bf16 operands only")
    if D != 32:
        raise Refused(f"triattn_cuda: head dim {D}: D = 32 only")
    if S > 65536:
        raise Refused(f"triattn_cuda: S = {S} > 65536 is not served")
    if not q.is_cuda:
        raise Refused("triattn_cuda: CUDA tensors only")
    if bias.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        raise Refused(f"triattn_cuda: bias dtype {bias.dtype}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    if out is not None:
        if tuple(out.shape) != (B, N, H, S, D) or out.dtype != torch.bfloat16 or out.device != q.device:
            raise ValueError(f"triattn_cuda: out must be a bf16 [B,N,H,S,D] = {(B, N, H, S, D)} tensor on {q.device}; got {tuple(out.shape)} {out.dtype} {out.device}")
        if out.stride(4) != 1 or any((st % 2) or st < 0 for st in out.stride()[:4]):
            raise ValueError(f"triattn_cuda: out strides {out.stride()}: d-stride 1 and even non-negative strides required")
    elif out_layout == "bnhsd":
        out = torch.empty(B, N, H, S, D, dtype=torch.bfloat16, device=q.device)
    elif out_layout == "bhnsd":
        out = torch.empty(B, H, N, S, D, dtype=torch.bfloat16, device=q.device).permute(0, 2, 1, 3, 4)
    else:  # "bnshd"
        out = torch.empty(B, N, S, H, D, dtype=torch.bfloat16, device=q.device).permute(0, 1, 3, 2, 4)
    if out.numel() == 0:
        return out
    ops = []
    for t, ok in ((q, _ldq_ok(q)), (k, _tma_ok(k)), (v, _tma_ok(v))):
        if not ok:
            COUNTS["copy"] = dict.get(COUNTS, "copy") + 1; t = t.contiguous()
        ops.append(t)
    q_, k_, v_ = ops
    mask_u8 = keyany = rowkind = irr = ktend = mtab = ctab = ktendr = None
    if mask is not None:
        mask_u8, keyany, rowkind, irr, ktend, mtab, ctab, ktendr = stage_mask(mask, B, N, S)
    if bias_staged is None:
        bias_staged = ext.stage_bias(bias[:, 0], float(scale), keyany)
    fix = _fix_buffer(q.device, ext.fix_elems(B, N, H, S))
    ext.fwd(q_, k_, v_, bias_staged, mask_u8, rowkind, irr, ktend, mtab, ctab, ktendr, float(scale), out, fix, int(_dbg), None)
    dict.__setitem__(COUNTS, "calls", dict.get(COUNTS, "calls") + 1)
    return out


def reference(q, k, v, bias, mask=None, scale=None, dtype=torch.float32):
    """Plain PyTorch evaluation of the op (fp32 by default) with cuEquivariance's fully-masked-row convention (uniform over all keys)."""
    q = _dims(q, 5); k = _dims(k, 5); v = _dims(v, 5); bias = _dims(bias, 5)
    B, N, H, S, D = q.shape
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.einsum("bnhqd,bnhkd->bnhqk", q.to(dtype), k.to(dtype)) * scale + bias.to(dtype)
    if mask is not None:
        keep = _dims(mask, 5).reshape(B, N, 1, 1, S).to(torch.bool)
        allmasked = ~keep.any(dim=-1, keepdim=True)
        s = torch.where(keep | allmasked, s, torch.full_like(s, float("-inf")))
        s = torch.where(allmasked.expand_as(s), torch.zeros_like(s), s)
    return torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(s, dim=-1), v.to(dtype))
