"""kernels.trimul_xla -- the triangle multiplication (pair update, bias-free form) for JAX / XLA programs on the SEALED trimul_native kernels:
pre-compiled cubins (pkg/v1/build/sm_90a/*.cubin, shipped, never built at install) launched through the XLA-FFI launcher of kernels/triattn_xla
(generic target `xla_cubin_call`, launcher 1.5): K1 (input LayerNorm + the four gated projections + mask -> channel-major bf16 planes) | the
contraction as ONE strided-batched cuBLAS GEMM on the two halves of the plane buffer (launcher target `xla_cublas_bgemm_nt`: bf16 operands, fp32 compute,
bf16 out -- the call the package issues through torch.bmm; XLA's own dot when that target is unavailable) |
K3 (output LayerNorm + output projection x sigmoid(gate) [+ residual]).  Row word: `native_xla`.

Op boundary (== trimul_native's, == the bias-free TriangleMultiplication of the AF3 / Boltz families):
    z [N,N,c_z] | [B,N,N,c_z] pre-LayerNorm pair tensor (bf16; or fp32 = the `f32z` form: fp32-resident z under bf16 compute), mask [N,N] | [B,N,N] 0/1 or None
    x = LN_in(z);  a = mask*sigmoid(x W_ag^T)*(x W_ap^T);  b = mask*sigmoid(x W_bg^T)*(x W_bp^T)
    X[i,j,:] = sum_k a[i,k,:] b[j,k,:] (outgoing) | sum_k a[k,i,:] b[k,j,:] (incoming)
    update = sigmoid(x W_og^T) * (LN_out(X) W_o^T);  returns update, or z + update when residual=True (dtype of z)
Weights: the ten canonical tensors of the package (torch Linear layout [out, in]): ln_in_w ln_in_b [c_z] | w_ag w_ap w_bg w_bp [c_h, c_z] | ln_out_w ln_out_b [c_h] |
w_o [c_z, c_h] | w_og [c_z, c_z]; `weights_from_math_layout()` maps kernels.pallas' trimul_params (left_w [C,Ch] ... = [in, out]) onto them.  Modules WITH
projection / gate / output BIASES (the AF2 family's TriangleMultiplication) are outside this op: refused BY NAME (`biases`), fallback named.

FORWARD ONLY.  jax.grad / vjp / jvp through `triangle_multiplication` raise `Refused` naming the fallback (`kernels.pallas cd_trimul`, which has a backward);
no VJP is defined.  vmap: a per-sample rule (jax.custom_batching) where this jax has it, else the ffi calls' sequential rule; report()["vmap"] says which.

Cards: cc 9.0 -> the sm_90a units carried here (c_z = c_hidden = 128: the pairformer width; c_z = c_hidden = 64: the template pair stack); cc 8.x and every
other capability -> refused by name (`arch`: the package's sm_80 kernels are not carried in this version), fallback named.  Sizes: 16 <= N <= 4096.
Selection data: CELLS.json ("units" = the tile table of the carried units, restated from the package; "measured" = this row against the JAX arms).
Equality: K1's planes and K3's output reproduce the package's own torch launch of the same cubins bit for bit on identical inputs (vectors.json);
the contraction between them is the same cuBLAS call the package makes through torch.bmm, and the END-TO-END output reproduces
the torch op bit for bit on every recorded case (vectors.json; a different cuBLAS release may order that GEMM's sums differently -- then equal up to
bf16 rounding of the same fp32 sums).  Hazard 11 (opt-in): a program reaches this only by binding it.
Ablation: MODEL_OPT_LEVERS_OFF words `trimul_xla` / `trimul_xla:native_xla` switch the row off (refused by name)."""
import hashlib
import json
import os

from ... import cell_census as _CENSUS                   # stdlib-only: the coverage census (one record per decided call class; pure observation)
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

__version__ = "1.0.0"
PACKAGE = "trimul_native"
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.join(HERE, "pkg", "v1")
ROOT_NAME = "trimul_xla"                 # the launcher root under which the cubin paths below are resolved (registered at load(), in every process)
ROW = "native_xla"
ROWS = (ROW,)
FALLBACK = "kernels.pallas cd_trimul (or the stock XLA statement)"
NTHREADS = 384
EPS_DEFAULT = 1e-5
N_MIN, N_MAX = 16, 4096
COUNTS: Dict[str, int] = {ROW: 0, "refused": 0}
_STATE: Dict[str, Any] = {}

with open(os.path.join(HERE, "CELLS.json"), encoding="utf-8") as _f:
    CELLS: Dict[str, Any] = json.load(_f)
UNITS: Dict[str, Dict[str, Any]] = CELLS["units"]          # "<c_z>x<c_h>" -> {cubin, b: {k1, k3, by_n}, f: {k1, k3, k3_mode}}


class Refused(RuntimeError):
    """The row cannot serve this call; the message names the reason word and the fallback.  `.kind` = the reason word."""

    def __init__(self, kind: str, msg: str):
        super().__init__("trimul_xla %s refused by name (%s): %s; fallback: %s" % (ROW, kind, msg, FALLBACK))
        self.kind = kind


# ----------------------------------------------------------------------------------------------------------------------------------- package data
def manifest() -> Dict[str, Any]:
    if "manifest" not in _STATE:
        with open(os.path.join(PKG_DIR, "build", "manifest.json"), encoding="utf-8") as f:
            _STATE["manifest"] = json.load(f)
    return _STATE["manifest"]


def package_version() -> str:
    with open(os.path.join(PKG_DIR, "VERSION"), encoding="utf-8") as f:
        return f.read().split()[-1]


def sha256sums() -> Dict[str, str]:
    if "sums" not in _STATE:
        out = {}
        with open(os.path.join(PKG_DIR, "SHA256SUMS"), encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) == 2:
                    out[parts[1].lstrip("./")] = parts[0]
        _STATE["sums"] = out
    return _STATE["sums"]


def cubin_entry(unit: str) -> Dict[str, Any]:
    """{path (relative to PKG_DIR), sha256, sha16, kernels} of a carried unit; the file digest is checked against SHA256SUMS once per process."""
    key = "cubin:" + unit
    if key not in _STATE:
        rel = UNITS[unit]["cubin"]
        full = os.path.join(PKG_DIR, rel)
        if not os.path.exists(full):
            raise Refused("carry", "cubin %s is not in the package" % rel)
        with open(full, "rb") as f:
            have = hashlib.sha256(f.read()).hexdigest()
        want = sha256sums().get(rel)
        if want is not None and want != have:
            raise Refused("carry", "cubin %s sha256 %s != SHA256SUMS %s" % (rel, have[:16], want[:16]))
        munit = manifest()["units"].get(UNITS[unit]["manifest_unit"], {})
        _STATE[key] = {"path": rel, "sha256": have, "sha16": have[:16], "kernels": sorted(munit.get("kernels", {}))}
    return _STATE[key]


# ----------------------------------------------------------------------------------------------------------------------------------- environment words
def levers_off() -> Tuple[bool, str]:
    words = [w.strip() for w in os.environ.get("MODEL_OPT_LEVERS_OFF", "").replace(",", " ").split() if w.strip()]
    for w in words:
        if w in ("trimul_xla", "trimul_xla:" + ROW, ROW):
            return True, w
    return False, ""


def device_cc(cc: Optional[str] = None) -> str:
    if cc:
        return str(cc)
    env = os.environ.get("TRIMUL_XLA_CC") or os.environ.get("TRIATTN_XLA_CC")
    if env:
        return env
    try:
        import jax  # noqa: WPS433
        d = jax.devices()[0]
        v = getattr(d, "compute_capability", None)
        return str(v) if v else "cpu"
    except (ImportError, RuntimeError, AttributeError, IndexError):
        return "?"


# ----------------------------------------------------------------------------------------------------------------------------------- tile table + launch geometry
def _ceil(a: int, b: int) -> int:
    return -(-a // b)


def unit_key(c_z: int, c_h: int) -> str:
    return "%dx%d" % (int(c_z), int(c_h))


def tile_cfg(unit: str, form: str, n: int) -> Dict[str, Any]:
    """{k1: (BI,BJ,NSLOT,SKCH), k3: (BI,BJ,NSLOT,NACC), k3_mode, covered} for (unit, form 'b'|'f', N): the unit's defaults overridden by the smallest by_n bucket >= N."""
    u = UNITS[unit].get(form)
    if u is None:
        raise Refused("form", "unit %s has no %s entry" % (unit, {"b": "bf16", "f": "fp32-resident"}[form]))
    out = {"k1": tuple(u["k1"]), "k3": tuple(u["k3"]), "k3_mode": u.get("k3_mode", "b" if form == "b" else "f"), "covered": True}
    by_n = {int(k): v for k, v in u.get("by_n", {}).items()}
    if by_n:
        fit = [b for b in sorted(by_n) if b >= n]
        b = fit[0] if fit else max(by_n)
        out["covered"] = bool(fit)
        for k in ("k1", "k3"):
            if k in by_n[b]:
                out[k] = tuple(by_n[b][k])
    return out


def k1_name(c_z: int, c_h: int, cfg: Sequence[int], masked: bool, zf32: bool) -> str:
    bi, bj, nslot, skch = cfg
    lnm = UNITS[unit_key(c_z, c_h)]["lnm"]["k1_f" if zf32 else "k1_b"]
    return "tmn_k1_z%d_h%d_%s_t%dx%d_s%dk%d_m%d_l%d_v0" % (c_z, c_h, "f" if zf32 else "b", bi, bj, nslot, skch, int(masked), lnm)


def k3_name(c_z: int, c_h: int, cfg: Sequence[int], mode: str) -> str:
    bi, bj, nslot, nacc = cfg
    lnm = UNITS[unit_key(c_z, c_h)]["lnm"]["k3"]
    return "tmn_k3_z%d_h%d_%s_t%dx%d_s%da%d_l%d" % (c_z, c_h, mode, bi, bj, nslot, nacc, lnm)


def k1_smem(c_z: int, zbytes: int, nslot: int, skch: int) -> int:
    """== K1Cfg::SMEM (pkg csrc/tmn_kernels.cuh; the release's kernel.py k1_smem)."""
    nkca = c_z // (128 // zbytes); nbar = 2 + 2 * nslot
    return nkca * 16384 + nslot * skch * 8192 + 16384 + 2 * c_z * 4 + ((nbar * 8 + 127) // 128) * 128


def k3_smem(c_z: int, c_h: int, zbytes: int, bi: int, bj: int, nslot: int) -> int:
    """== K3Cfg::SMEM (modes b / f)."""
    bmt = bi * bj; nkcz = c_z // (128 // zbytes); slot = max(c_z, c_h) * 64; ob = 16 * 64 * zbytes
    nbar = 2 * nkcz + 2 + 2 * nslot
    return (bmt // 64) * c_h * 128 + nkcz * bmt * 128 + nslot * slot + 8 * ob + (2 * c_z + 2 * c_h) * 4 + ((nbar * 8 + 127) // 128) * 128


K1_PARAMS_SIZE, K3_PARAMS_SIZE = 384, 640        # csrc/tmn_kernels.cuh K1Params / K3Params (by-value, 64-byte aligned: they open with CUtensorMap fields)


def k1_launch(c_z: int, c_h: int, n: int, zf32: bool, masked: bool, transpose: bool, eps: float) -> Dict[str, Any]:
    """Everything the K1 launch needs, as launcher tokens.  Buffers: in 0 z [N,N,c_z] | 1 w1 [4c_h,c_z] bf16 | 2 ln_in_w f32 | 3 ln_in_b f32 | 4 mask f32 [N,N] (masked);
    out 0 ab planes [2c_h, Np, Np] bf16 (zero pads written by the kernel)."""
    from ..triattn_xla import _launch as L  # noqa: WPS433
    unit = unit_key(c_z, c_h); cfg = tile_cfg(unit, "f" if zf32 else "b", n)
    bi, bj, nslot, skch = cfg["k1"]
    esz = 4 if zf32 else 2; np_ = _ceil(n, 16) * 16
    tiles_i, tiles_j = _ceil(np_, bi), _ceil(np_, bj); num_tiles = tiles_i * tiles_j
    ms_i, ms_j = (1, n) if transpose else (n, 1)
    s_fast, s_slow = c_z * esz, n * c_z * esz
    if transpose:
        s_fast, s_slow = s_slow, s_fast
    tm_z = L.tmap_token("f32" if zf32 else "bf16", [c_z, n, n], [s_fast, s_slow], [128 // esz, bj, bi], swizzle=128, l2=128, oob=0)
    tm_w = L.tmap_token("bf16", [c_z, 4 * c_h], [c_z * 2], [64, 64], swizzle=128, l2=256, oob=0)
    fields = ["tb0:0", "tb1:1", ("b4" if masked else "n"), "b2", "b3", "o0", "n", "n",
              "i%d" % n, "i%d" % np_, "i%d" % tiles_j, "i%d" % num_tiles, "i%d" % int(np_ % 8 == 0), "i%d" % ms_i, "i%d" % ms_j, L.f32_token(eps),
              "i%d" % (ms_i * c_z), "i%d" % (ms_j * c_z), "i0", "i0", "z32"]
    return {"kname": k1_name(c_z, c_h, cfg["k1"], masked, zf32), "shared": k1_smem(c_z, esz, nslot, skch), "grid": (num_tiles, 1, 1), "grid_rule": 1,
            "params": [L.struct_token(K1_PARAMS_SIZE, fields)], "tmaps": [tm_z, tm_w], "np": np_, "cfg": cfg}


def k3_launch(c_z: int, c_h: int, n: int, zf32: bool, residual: bool, eps: float) -> Dict[str, Any]:
    """K3 tokens.  Buffers: in 0 z [N,N,c_z] (gate operand rows + residual) | 1 x [c_h, Np, Np] bf16 | 2 wg [c_z,c_z] bf16 | 3 wo [c_z,c_h] bf16 | 4 ln_in_w | 5 ln_in_b |
    6 ln_out_w | 7 ln_out_b (f32); out 0 [N,N,c_z] in z's dtype."""
    from ..triattn_xla import _launch as L  # noqa: WPS433
    unit = unit_key(c_z, c_h); cfg = tile_cfg(unit, "f" if zf32 else "b", n)
    mode = cfg["k3_mode"]
    if mode not in ("b", "f"):
        raise Refused("form", "K3 operand mode %r of unit %s is not restated in this version (b | f are)" % (mode, unit))
    bi, bj, nslot, nacc = cfg["k3"]
    esz = 4 if zf32 else 2; np_ = _ceil(n, 16) * 16
    tiles_i, tiles_j = _ceil(n, bi), _ceil(n, bj); num_tiles = tiles_i * tiles_j
    tm_z = L.tmap_token("f32" if zf32 else "bf16", [c_z, n, n], [c_z * esz, n * c_z * esz], [128 // esz, bj, bi], swizzle=128, l2=128, oob=0)
    tm_x = L.tmap_token("bf16", [np_, np_, c_h], [np_ * 2, np_ * np_ * 2], [64, 1, 64], swizzle=128, l2=128, oob=0)
    tm_wg = L.tmap_token("bf16", [c_z, c_z], [c_z * 2], [64, 32], swizzle=128, l2=256, oob=0)
    tm_wo = L.tmap_token("bf16", [c_h, c_z], [c_h * 2], [64, 32], swizzle=128, l2=256, oob=0)
    fields = ["tb0:0", "tb1:1", "tb2:2", "tb3:3", "b4", "b5", "b6", "b7", "b0", "o0", "n",
              "i%d" % n, "i%d" % np_, "i%d" % tiles_j, "i%d" % num_tiles, "i%d" % int(bool(residual)), "i0", L.f32_token(eps), "i0", "z40"]
    return {"kname": k3_name(c_z, c_h, cfg["k3"], mode), "shared": k3_smem(c_z, c_h, esz, bi, bj, nslot), "grid": (num_tiles, 1, 1), "grid_rule": 1,
            "params": [L.struct_token(K3_PARAMS_SIZE, fields)], "tmaps": [tm_z, tm_x, tm_wg, tm_wo], "np": np_, "cfg": cfg}


# ----------------------------------------------------------------------------------------------------------------------------------- selection / refusal
def cannot_serve(cc: str, dtype: Any, c_z: int, c_h: int, n: int, *, biases: bool = False, batch: int = 1) -> Optional[Tuple[str, str]]:
    """None when the row serves the call class, else (reason word, text).  Pure table logic (no GPU)."""
    off, word = levers_off()
    if off:
        return "levers_off", "MODEL_OPT_LEVERS_OFF names %r" % word
    if biases:
        return "biases", "the module carries projection / gate / output biases; the sealed kernels implement the bias-free update (the six bias vectors cannot be folded into the LayerNorm affine or the weights)"
    ccs = str(cc)
    if not ccs.startswith("9.0"):
        return "arch", "compute capability %s: this version carries the sm_90a units only (cc 9.0); the package's sm_80 kernels are not restated here" % ccs
    unit = unit_key(c_z, c_h)
    if unit not in UNITS:
        return "width", "no unit for c_z=%d, c_hidden=%d (carried: %s)" % (c_z, c_h, ", ".join(sorted(UNITS)))
    dw = _dtype_word(dtype)
    if dw not in ("bf16", "f32"):
        return "dtype", "z dtype %s (bf16, or fp32 = the fp32-resident form under bf16 compute)" % dw
    if dw == "f32" and "f" not in UNITS[unit]:
        return "dtype", "unit %s has no fp32-resident entry" % unit
    if n < N_MIN or n > N_MAX:
        return "size", "N=%d outside %d..%d" % (n, N_MIN, N_MAX)
    if (n * c_z * (4 if dw == "f32" else 2)) % 16:
        return "size", "row stride N*c_z*elem = %d bytes is not a multiple of 16 (TMA)" % (n * c_z * (4 if dw == "f32" else 2))
    return None


def _dtype_word(dtype: Any) -> str:
    s = str(getattr(dtype, "name", dtype)).replace("jnp.", "").replace("numpy.", "")
    return {"bfloat16": "bf16", "float32": "f32", "float16": "f16"}.get(s, s)


def select(cc: Optional[str], dtype: Any, c_z: int, c_h: int, n: int, *, biases: bool = False, batch: int = 1) -> str:
    """The row word that serves the call class, or raise Refused (by name, fallback named)."""
    ccs = device_cc(cc)
    why = cannot_serve(ccs, dtype, c_z, c_h, n, biases=biases, batch=batch)
    key = {"cc": ccs, "dtype": _dtype_word(dtype), "shape": "c%dx%d" % (int(c_z), int(c_h)), "bucket": _CENSUS.bucket_word(n), "form": "biases" if biases else "bias_free", "word": ROW}
    if why is not None:
        COUNTS["refused"] += 1
        _CENSUS.record("trimul_xla", key, "named_fallback", "caller:" + FALLBACK.split(" ")[1], cell_id="by_cc:" + ccs.split(".")[0] + ".x", refused="%s:%s" % (ROW, why[0]), note="raised")
        raise Refused(*why)
    _CENSUS.record("trimul_xla", key, "opt_in", ROW, cell_id="by_cc:9.0")
    return ROW


def measured(cc: Optional[str] = None) -> Sequence[Dict[str, Any]]:
    """CELLS.json "measured": this row against the JAX arms per (cc, jax line, dtype, width, direction, N)."""
    ccs = device_cc(cc)
    return [r for r in CELLS.get("measured", []) if str(r.get("cc")) == ccs or cc is None]


# ----------------------------------------------------------------------------------------------------------------------------------- weights
WEIGHT_KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
MATH_BIAS_KEYS = ("left_b", "right_b", "left_gate_b", "right_gate_b", "out_b", "gate_b")


def weights_from_math_layout(params: Dict[str, Any], direction: str) -> Dict[str, Any]:
    """kernels.pallas' trimul_params (math layout: left_w / right_w / left_gate_w / right_gate_w [C, Ch] = [in, out], ln_c_* [Ch], out_w [Ch, C], gate_w [C, C],
    ln_in_scale / ln_in_offset [C]) -> the package's ten canonical tensors ([out, in]).  The einsum statements those parameters belong to are
    'ikc,jkc->ijc' (outgoing: X_ij = sum_k left_ik right_jk -> a = left, b = right) and 'kjc,kic->ijc' (incoming: X_ij = sum_k left_kj right_ki, while the
    kernels' incoming is sum_k a_ki b_kj -> a = RIGHT, b = LEFT).  Raises Refused(`biases`) when any of the six biases is present."""
    import jax.numpy as jnp  # noqa: WPS433
    present = [k for k in MATH_BIAS_KEYS if params.get(k) is not None]
    if present:
        COUNTS["refused"] += 1
        raise Refused("biases", "params carry %s" % ", ".join(present))
    if direction not in ("outgoing", "incoming"):
        raise ValueError("direction must be 'outgoing' or 'incoming'")
    t = jnp.swapaxes
    a, b = ("left", "right") if direction == "outgoing" else ("right", "left")
    return {"ln_in_w": params["ln_in_scale"], "ln_in_b": params["ln_in_offset"], "w_ag": t(params[a + "_gate_w"], 0, 1), "w_ap": t(params[a + "_w"], 0, 1),
            "w_bg": t(params[b + "_gate_w"], 0, 1), "w_bp": t(params[b + "_w"], 0, 1), "ln_out_w": params["ln_c_scale"], "ln_out_b": params["ln_c_offset"],
            "w_o": t(params["out_w"], 0, 1), "w_og": t(params["gate_w"], 0, 1)}


def pack_weights(weights: Any) -> Dict[str, Any]:
    """The kernels' operand forms (jnp, traced with the program): w1 [4c_h, c_z] bf16 = 32-row blocks interleaving [w_ag; w_bg] and [w_ap; w_bp]; wg = w_og bf16;
    wo = w_o bf16; LayerNorm vectors fp32.  == trimul_native pack_weights."""
    import jax.numpy as jnp  # noqa: WPS433
    g = (lambda k: weights[k]) if isinstance(weights, dict) else (lambda k: getattr(weights, k))
    w = {k: jnp.asarray(g(k)) for k in WEIGHT_KEYS}
    ch, cz = int(w["w_ag"].shape[0]), int(w["w_ag"].shape[1])
    bf = jnp.bfloat16
    gate = jnp.concatenate([w["w_ag"], w["w_bg"]], 0).astype(jnp.float32)
    proj = jnp.concatenate([w["w_ap"], w["w_bp"]], 0).astype(jnp.float32)
    nb = (2 * ch) // 32
    w1 = jnp.stack([gate.reshape(nb, 32, cz), proj.reshape(nb, 32, cz)], 1).reshape(4 * ch, cz).astype(bf)      # block b: rows gate[32b:32b+32] then proj[32b:32b+32]
    return {"c_z": cz, "c_h": ch, "w1": w1, "wg": w["w_og"].astype(bf), "wo": w["w_o"].astype(bf),
            "ln_in_w": w["ln_in_w"].astype(jnp.float32), "ln_in_b": w["ln_in_b"].astype(jnp.float32), "ln_out_w": w["ln_out_w"].astype(jnp.float32), "ln_out_b": w["ln_out_b"].astype(jnp.float32)}


# ----------------------------------------------------------------------------------------------------------------------------------- launches
def load() -> None:
    """Load the launcher (kernels/triattn_xla), register this package's root; cached.  Raises Refused by name (jax without jax.ffi, no launcher build, ...)."""
    if _STATE.get("loaded"):
        return
    from ..triattn_xla import _launch as L, Refused as TRefused  # noqa: WPS433
    try:
        L.add_root(ROOT_NAME, PKG_DIR)
    except TRefused as e:
        raise Refused("launcher", str(e))
    _STATE["loaded"] = True


def _calls(c_z: int, c_h: int, n: int, zf32: bool, masked: bool, transpose: bool, residual: bool, eps: float, out_dtype: Any):
    """(k1_call, k3_call) ffi callables for one static plane shape; cached."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from ..triattn_xla import _launch as L  # noqa: WPS433
    key = (c_z, c_h, n, zf32, masked, transpose, residual, float(eps), str(out_dtype))
    cache = _STATE.setdefault("calls", {})
    if key in cache:
        return cache[key]
    load()
    unit = unit_key(c_z, c_h); ent = cubin_entry(unit)
    k1 = k1_launch(c_z, c_h, n, zf32, masked, transpose, eps); k3 = k3_launch(c_z, c_h, n, zf32, residual, eps)
    for k in (k1, k3):
        if ent["kernels"] and k["kname"] not in ent["kernels"]:
            raise Refused("carry", "kernel %s is not in %s" % (k["kname"], ent["path"]))
    np_ = k1["np"]
    lab1 = "trimul_xla K1 %s N=%d%s%s" % (unit, n, " f32z" if zf32 else "", " T" if transpose else "")
    lab3 = "trimul_xla K3 %s N=%d%s%s" % (unit, n, " f32z" if zf32 else "", " +res" if residual else "")
    c1 = L.cubin_call(root=ROOT_NAME, path=ent["path"], sha=ent["sha16"], kname=k1["kname"], shared=k1["shared"], block=NTHREADS, grid=k1["grid"], grid_rule=k1["grid_rule"],
                      params=k1["params"], tmaps=k1["tmaps"], out_types=[jax.ShapeDtypeStruct((2 * c_h, np_, np_), jnp.bfloat16)], label=lab1)
    c3 = L.cubin_call(root=ROOT_NAME, path=ent["path"], sha=ent["sha16"], kname=k3["kname"], shared=k3["shared"], block=NTHREADS, grid=k3["grid"], grid_rule=k3["grid_rule"],
                      params=k3["params"], tmaps=k3["tmaps"], out_types=[jax.ShapeDtypeStruct((n, n, c_z), out_dtype)], label=lab3)
    cache[key] = (c1, c3)
    return c1, c3


CUBLAS_WORKSPACE_BYTES = 32 << 20          # the workspace the GEMM call is given (torch's default per-handle workspace class on this card generation)


def contraction(ab, c_h: int, how: Optional[str] = None):
    """x [c_h, Np, Np] bf16 = a . b^T per channel from the K1 planes ab [2 c_h, Np, Np]: through the launcher's cuBLAS strided-batched GEMM on the two halves
    of the ONE plane buffer (no slices materialised; fp32 compute, bf16 out -- the call the package issues through torch.bmm), or -- `how="xla"` / a
    launcher without that target -- XLA's own dot on the sliced planes (same arithmetic class; XLA materialises the two slices).  report()["contraction"]."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from ..triattn_xla import _launch as L, Refused as TRefused  # noqa: WPS433
    np_ = int(ab.shape[-1])
    how = how or os.environ.get("TRIMUL_XLA_CONTRACTION") or "cublas"
    if how == "cublas" and _cublas_selfcheck() is not None:
        how = "selfcheck"
    if how == "cublas":
        try:
            key = ("bgemm", c_h, np_)
            cache = _STATE.setdefault("calls", {})
            if key not in cache:
                cache[key] = L.cublas_bgemm_nt([jax.ShapeDtypeStruct((c_h, np_, np_), jnp.bfloat16), jax.ShapeDtypeStruct((CUBLAS_WORKSPACE_BYTES,), jnp.uint8)],
                                               a_buf=0, a_off=0, b_buf=0, b_off=c_h * np_ * np_, batch=c_h, m=np_, n=np_, k=np_, ws=CUBLAS_WORKSPACE_BYTES, label="trimul_xla bgemm c_h=%d Np=%d" % (c_h, np_))
            _STATE["contraction"] = "cublas (launcher target xla_cublas_bgemm_nt)"
            return cache[key](ab)[0]
        except TRefused as e:
            _STATE["contraction"] = "xla dot (%s)" % str(e)[:120]
    elif how == "selfcheck":
        _STATE["contraction"] = "xla dot (the cuBLAS target failed its self-check in this process: %s)" % _STATE.get("selfcheck")
    else:
        _STATE["contraction"] = "xla dot (TRIMUL_XLA_CONTRACTION=%s)" % how
    return jnp.matmul(ab[:c_h], jnp.swapaxes(ab[c_h:], 1, 2))


def _cublas_selfcheck() -> Optional[str]:
    """Once per process: one small bgemm through the launcher target against jnp's dot, run EAGERLY.  None = usable (or not decidable here: called while
    tracing, no eager execution possible -- then the target is used and a failing call raises by name at run time; TRIMUL_XLA_CONTRACTION=xla is the
    switch); a string = the reason the target is unusable in this process (then every program traced afterwards takes XLA's dot).  `selfcheck()` runs it
    from an eager context (tests, bring-up)."""
    if "selfcheck" in _STATE:
        return _STATE["selfcheck"]
    import numpy as np  # noqa: WPS433
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    from ..triattn_xla import _launch as L, Refused as TRefused  # noqa: WPS433
    c, n = 2, 64
    host = (((np.arange(2 * c * n * n) % 13) - 6).reshape(2 * c, n, n).astype(np.float32) / 8)
    try:
        f = L.cublas_bgemm_nt([jax.ShapeDtypeStruct((c, n, n), jnp.bfloat16), jax.ShapeDtypeStruct((CUBLAS_WORKSPACE_BYTES,), jnp.uint8)],
                              a_buf=0, a_off=0, b_buf=0, b_off=c * n * n, batch=c, m=n, n=n, k=n, ws=CUBLAS_WORKSPACE_BYTES, label="trimul_xla bgemm self-check")
        with jax.ensure_compile_time_eval():
            t = jnp.asarray(host).astype(jnp.bfloat16)
            y = jax.jit(lambda a: f(a)[0])(t)
            ref = jnp.matmul(t[:c], jnp.swapaxes(t[c:], 1, 2))
            if isinstance(y, jax.core.Tracer) or isinstance(ref, jax.core.Tracer):
                _STATE["selfcheck_note"] = "deferred (first use was inside a trace; the target is used, a failing call raises by name)"
                return None                                   # not cached as decided: an eager caller may still run it
            ok = bool(np.array_equal(np.asarray(jax.device_get(y)).view(np.uint16), np.asarray(jax.device_get(ref)).view(np.uint16)))
        _STATE["selfcheck"] = None if ok else "result differs from jnp.matmul on an exactly representable case"
    except TRefused as e:
        _STATE["selfcheck"] = "refused: %s" % str(e)[:160]
    except Exception as e:      # noqa: BLE001  (a tracer leaking out of the eager attempt = deferred; a launch / library failure = take the XLA dot; recorded in report())
        name = type(e).__name__
        if "Tracer" in name or "Concretization" in name:
            _STATE["selfcheck_note"] = "deferred (%s while tracing)" % name
            return None
        _STATE["selfcheck"] = "%s: %s" % (name, str(e)[:160])
    return _STATE.get("selfcheck")


def selfcheck() -> Optional[str]:
    """Run the cuBLAS-target self-check now (eager context) and return None (usable) or the reason it is not."""
    _STATE.pop("selfcheck", None)
    return _cublas_selfcheck()


def _plane(z, mask, pw: Dict[str, Any], *, outgoing: bool, residual: bool, eps: float, save: Optional[Dict[str, Any]] = None):
    """One [N,N,c_z] plane: K1 -> XLA dot -> K3."""
    import jax.numpy as jnp  # noqa: WPS433
    n, c_z = int(z.shape[0]), int(z.shape[-1]); c_h = int(pw["c_h"])
    zf32 = _dtype_word(z.dtype) == "f32"
    masked = mask is not None
    k1c, k3c = _calls(c_z, c_h, n, zf32, masked, not outgoing, residual, eps, z.dtype)
    args = [z, pw["w1"], pw["ln_in_w"], pw["ln_in_b"]] + ([mask.astype(jnp.float32)] if masked else [])
    ab = k1c(*args)[0]                                                                   # [2c_h, Np, Np] bf16 (transposed planes for incoming)
    x = contraction(ab, c_h)                                                             # X[c,i,j] = sum_k a[c,i,k] b[c,j,k]; fp32 accumulate, bf16 planes
    out = k3c(z, x, pw["wg"], pw["wo"], pw["ln_in_w"], pw["ln_in_b"], pw["ln_out_w"], pw["ln_out_b"])[0]
    if save is not None:
        save["ab"] = ab; save["x"] = x
    return out


def _forward(z, mask, pw, outgoing: bool, residual: bool, eps: float, save=None):
    import jax.numpy as jnp  # noqa: WPS433
    if z.ndim == 3:
        return _plane(z, mask, pw, outgoing=outgoing, residual=residual, eps=eps, save=save)
    outs = [_plane(z[i], None if mask is None else mask[i], pw, outgoing=outgoing, residual=residual, eps=eps, save=save) for i in range(int(z.shape[0]))]
    return jnp.stack(outs)


def _guarded(outgoing: bool, residual: bool, eps: float) -> Callable:
    """The forward wrapped (1) forward-only: differentiation raises Refused by name; (2) with a per-sample vmap rule where jax.custom_batching exists."""
    import jax  # noqa: WPS433
    key = ("guard", outgoing, residual, float(eps))
    cache = _STATE.setdefault("guards", {})
    if key in cache:
        return cache[key]

    def raw(z, mask, w1, wg, wo, ln_in_w, ln_in_b, ln_out_w, ln_out_b):
        pw = {"c_z": int(z.shape[-1]), "c_h": int(wo.shape[1]), "w1": w1, "wg": wg, "wo": wo, "ln_in_w": ln_in_w, "ln_in_b": ln_in_b, "ln_out_w": ln_out_w, "ln_out_b": ln_out_b}
        return _forward(z, mask, pw, outgoing, residual, eps)

    def raw_nomask(z, w1, wg, wo, ln_in_w, ln_in_b, ln_out_w, ln_out_b):
        pw = {"c_z": int(z.shape[-1]), "c_h": int(wo.shape[1]), "w1": w1, "wg": wg, "wo": wo, "ln_in_w": ln_in_w, "ln_in_b": ln_in_b, "ln_out_w": ln_out_w, "ln_out_b": ln_out_b}
        return _forward(z, None, pw, outgoing, residual, eps)

    def fwd_only(fn):
        f = jax.custom_vjp(fn)

        def f_fwd(*args):
            return fn(*args), None

        def f_bwd(_res, _ct):
            raise Refused("forward_only", "this row defines no VJP (jax.grad / vjp / jvp through it); the differentiated triangle multiplication is kernels.pallas cd_trimul")
        f.defvjp(f_fwd, f_bwd)
        return f

    try:
        from ..triattn_xla import _batching as B  # noqa: WPS433
        wrap = lambda fn: B.per_sample(fn, 1) if B.custom_vmap_available() is None else fn     # noqa: E731
        vm = "custom_vmap per-sample rule" if B.custom_vmap_available() is None else "ffi sequential rule (%s)" % B.custom_vmap_available()
    except ImportError as e:
        wrap = lambda fn: fn      # noqa: E731
        vm = "none (%s)" % e
    _STATE["vmap"] = vm

    def unwrap1(fn):
        def g(*a):
            r = fn(*a)
            return r[0] if isinstance(r, (tuple, list)) else r
        return g
    out = (fwd_only(unwrap1(wrap(lambda *a: (raw(*a),)))), fwd_only(unwrap1(wrap(lambda *a: (raw_nomask(*a),)))))
    cache[key] = out
    return out


def triangle_multiplication(z, mask=None, *, weights: Any, direction: str, residual: bool = False, eps: float = EPS_DEFAULT, cc: Optional[str] = None,
                          impl: str = "auto", return_row: bool = False):
    """update = TriangleMultiplication(z) (bias-free form) on the sealed kernels; z [N,N,c_z] or [B,N,N,c_z] (bf16 | fp32), mask [.., N, N] 0/1 or None,
    weights = the ten canonical tensors (dict / attributes) or already `pack_weights(...)`, direction 'outgoing' | 'incoming', residual -> z + update.
    Raises Refused BY NAME (arch / width / dtype / size / biases / levers_off / launcher / forward_only) naming the fallback.  impl: 'auto' | 'native_xla'."""
    if direction not in ("outgoing", "incoming"):
        raise ValueError("direction must be 'outgoing' or 'incoming'")
    if impl not in ("auto", ROW):
        raise Refused("word", "impl %r (words: auto, %s)" % (impl, ROW))
    pw = weights if (isinstance(weights, dict) and "w1" in weights) else pack_weights(weights)
    n, c_z = int(z.shape[-2]), int(z.shape[-1]); c_h = int(pw["c_h"])
    if z.ndim not in (3, 4) or int(z.shape[-3]) != n:
        raise Refused("shape", "z must be [N,N,c_z] or [B,N,N,c_z], got %s" % (tuple(z.shape),))
    if int(pw["c_z"]) != c_z:
        raise Refused("shape", "weights are for c_z=%d, z has c_z=%d" % (int(pw["c_z"]), c_z))
    if mask is not None and tuple(mask.shape) != tuple(z.shape[:-1]):
        raise Refused("shape", "mask %s must match z's token dims %s" % (tuple(mask.shape), tuple(z.shape[:-1])))
    select(cc, z.dtype, c_z, c_h, n, batch=(1 if z.ndim == 3 else int(z.shape[0])))
    f_mask, f_nomask = _guarded(direction == "outgoing", bool(residual), float(eps))
    common = (pw["w1"], pw["wg"], pw["wo"], pw["ln_in_w"], pw["ln_in_b"], pw["ln_out_w"], pw["ln_out_b"])
    out = f_mask(z, mask, *common) if mask is not None else f_nomask(z, *common)
    COUNTS[ROW] += 1
    return (out, ROW) if return_row else out


def forward_with_intermediates(z, mask=None, *, weights: Any, direction: str, residual: bool = False, eps: float = EPS_DEFAULT, cc: Optional[str] = None):
    """(out, {"ab": planes, "x": contraction}) -- the unguarded forward exposing K1's planes and the XLA contraction (for stage-wise bitwise comparison)."""
    pw = weights if (isinstance(weights, dict) and "w1" in weights) else pack_weights(weights)
    n, c_z = int(z.shape[-2]), int(z.shape[-1])
    select(cc, z.dtype, c_z, int(pw["c_h"]), n)
    save: Dict[str, Any] = {}
    out = _forward(z, mask, pw, direction == "outgoing", bool(residual), float(eps), save=save)
    return out, save


def k3_only(z, x, *, weights: Any, residual: bool = False, eps: float = EPS_DEFAULT):
    """K3 alone on a given contraction x [c_h, Np, Np] bf16 (for comparing K3 bitwise on the torch op's own contraction)."""
    pw = weights if (isinstance(weights, dict) and "w1" in weights) else pack_weights(weights)
    n, c_z = int(z.shape[0]), int(z.shape[-1]); c_h = int(pw["c_h"])
    _k1, k3c = _calls(c_z, c_h, n, _dtype_word(z.dtype) == "f32", False, False, bool(residual), float(eps), z.dtype)
    return k3c(z, x, pw["wg"], pw["wo"], pw["ln_in_w"], pw["ln_in_b"], pw["ln_out_w"], pw["ln_out_b"])[0]


def reference(z, mask, weights: Any, *, direction: str, residual: bool = False, eps: float = EPS_DEFAULT, precision="highest"):
    """The op statement in jnp, fp32 throughout (the identity band's reference)."""
    import jax  # noqa: WPS433
    import jax.numpy as jnp  # noqa: WPS433
    g = (lambda k: weights[k]) if isinstance(weights, dict) else (lambda k: getattr(weights, k))
    w = {k: jnp.asarray(g(k)).astype(jnp.float32) for k in WEIGHT_KEYS}
    f32 = jnp.float32; zf = z.astype(f32)

    def ln(v, gam, bet):
        mu = v.mean(-1, keepdims=True); var = ((v - mu) ** 2).mean(-1, keepdims=True)
        return (v - mu) / jnp.sqrt(var + eps) * gam + bet
    pr = jax.lax.Precision.HIGHEST if precision == "highest" else None
    x = ln(zf, w["ln_in_w"], w["ln_in_b"])
    mm = lambda t, W: jnp.einsum("...c,hc->...h", t, W, precision=pr)      # noqa: E731
    m = 1.0 if mask is None else mask.astype(f32)[..., None]
    a = m * jax.nn.sigmoid(mm(x, w["w_ag"])) * mm(x, w["w_ap"]); b = m * jax.nn.sigmoid(mm(x, w["w_bg"])) * mm(x, w["w_bp"])
    if direction == "outgoing":
        X = jnp.einsum("...ikc,...jkc->...ijc", a, b, precision=pr)
    else:
        X = jnp.einsum("...kic,...kjc->...ijc", a, b, precision=pr)
    upd = jax.nn.sigmoid(mm(x, w["w_og"])) * mm(ln(X, w["ln_out_w"], w["ln_out_b"]), w["w_o"])
    return (zf + upd) if residual else upd


def report() -> Dict[str, Any]:
    """What this package can say without a GPU call: version, package release, carried units + digests, launcher state, counts, vmap rule, cells summary."""
    out: Dict[str, Any] = {"version": __version__, "package": PACKAGE, "package_version": None, "row": ROW, "units": {}, "counts": dict(COUNTS), "vmap": _STATE.get("vmap"),
                           "loaded": bool(_STATE.get("loaded")), "contraction": _STATE.get("contraction"), "cublas_selfcheck": _STATE["selfcheck"] if "selfcheck" in _STATE else _STATE.get("selfcheck_note", "not run"), "fallback": FALLBACK, "forward_only": True, "cards": {"9.0": "sm_90a units", "8.x": "refused by name (arch)"}}
    try:
        out["package_version"] = package_version()
    except OSError as e:
        out["package_version"] = "? (%s)" % e
    for u, d in UNITS.items():
        try:
            e = cubin_entry(u); out["units"][u] = {"cubin": e["path"], "sha256": e["sha256"], "kernels": len(e["kernels"]), "forms": [f for f in ("b", "f") if f in d]}
        except Refused as e:
            out["units"][u] = {"refused": str(e)}
    try:
        from ..triattn_xla import _launch as L  # noqa: WPS433
        out["launcher"] = {k: v for k, v in L.status().items() if k in ("loaded", "path", "ffi_api_version", "generic_target", "roots", "refused")}
    except (ImportError, RuntimeError, OSError) as e:
        out["launcher"] = {"error": "%s: %s" % (type(e).__name__, str(e)[:200])}
    out["measured_rows"] = len(CELLS.get("measured", []))
    return out
