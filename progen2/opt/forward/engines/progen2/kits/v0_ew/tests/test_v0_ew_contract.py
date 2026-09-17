"""v0_ew kit — CPU contract tests (no GPU, no nvcc, no progen package): the kit surface and its sha pin, counters == expectations
and the refusals, the rounding chains the kernels implement (checked against torch's own fp16/fp32 CPU semantics), the head /
column mapping of the fused rotary kernel (a mirror vs the stock's reshape/split/_split_heads arithmetic), capture safety of
the hot path (no host syncs), and the extension loader's refusals.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
from engines.progen2.kits import v0_ew  # noqa: E402
from engines.progen2.kits.v0_ew import ext, patches as P  # noqa: E402

KIT_DIR = os.path.dirname(os.path.abspath(v0_ew.__file__))


def test_surface():
    assert v0_ew.KIT == "v0_ew" and v0_ew.LEVERS == ("gelu", "rotary", "residual", "glue", "ln")
    for fn in ("apply", "release", "stamp", "expected_per_forward", "assert_counted_path", "counters_delta"):
        assert callable(getattr(v0_ew, fn))


def _fake_model(n_layer):
    return types.SimpleNamespace(config=types.SimpleNamespace(n_layer=n_layer))


def test_expected_counts_per_lever_set():
    m = _fake_model(27)
    assert P.expected_counts(m, ("gelu",)) == {"gelu_new": 27}
    assert P.expected_counts(m, ("rotary",)) == {"rotary_split_qkv": 27}
    assert P.expected_counts(m, ("residual",)) == {"residual_add2": 27}
    assert P.expected_counts(m, ("glue",)) == {"attn_glue": 27}
    assert P.expected_counts(m, ("ln",)) == {"layer_norm": 28}
    assert P.expected_counts(m, v0_ew.LEVERS) == {"gelu_new": 27, "rotary_split_qkv": 27, "residual_add2": 27, "attn_glue": 27, "layer_norm": 28}


def test_assert_counts_refuses_a_stock_path_and_a_shadow_mismatch():
    exp = {"gelu_new": 3, "layer_norm": 4}
    P.assert_counts({"gelu_new": 6, "layer_norm": 8}, exp, 2)
    with pytest.raises(P.KitRefused):
        P.assert_counts({"gelu_new": 5, "layer_norm": 8}, exp, 2)
    with pytest.raises(P.KitRefused):
        P.assert_counts({"gelu_new": 6, "layer_norm": 8, "shadow_mismatch_gelu_new": 1}, exp, 2)


def test_stamp_and_counted_path_refuse_when_not_applied():
    with pytest.raises(P.KitRefused):
        v0_ew.stamp()
    with pytest.raises(P.KitRefused):
        v0_ew.assert_counted_path({"forwards": 1}, 1)


def test_hot_path_has_no_host_sync():
    """Capture safety: no .item()/.tolist()/.cpu()/.numpy()/torch.equal/synchronize on the kit's forward path."""
    src = open(os.path.join(KIT_DIR, "patches.py")).read()
    tree = ast.parse(src)
    hot = ("_gelu_ew", "qkv_heads_rotary", "_attn_forward_ew", "_attn_ew", "_block_forward_ew", "_ln_forward_ew", "_tables")
    fns = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in hot}
    assert len(fns) == len(hot)
    for name, fn in fns.items():
        calls = {ast.unparse(c.func).split(".")[-1] for c in ast.walk(fn) if isinstance(c, ast.Call)}
        assert not calls & {"item", "tolist", "cpu", "numpy", "equal", "synchronize"}, (name, calls)
    src_ext = open(os.path.join(KIT_DIR, "ext.py")).read()
    assert "synchronize" not in src_ext and ".item()" not in src_ext


def test_gelu_scalars_are_the_fp32_constants_for_both_dtypes():
    """The wrapped Python double is read as an opmath fp32 for fp16 and fp32 tensors alike (pinned by an exhaustive sweep:
    the fp16-rounded alternative fails 156 of 65,536 patterns)."""
    c0h, c1h = P.gelu_scalars(torch.float16)
    c0f, c1f = P.gelu_scalars(torch.float32)
    assert c0f == float(np.float32(0.044715)) and c1f == float(np.float32(math.sqrt(2.0 / math.pi)))
    assert (c0h, c1h) == (c0f, c1f)
    assert c0h != float(np.float16(0.044715))


def test_expected_counts_autocast_names_the_autocast_chain():
    m = _fake_model(12)
    assert P.expected_counts(m, ("gelu",), autocast=True) == {"gelu_new_autocast": 12}


def _chain_fp16_numpy(x16: np.ndarray, c0: float, c1: float) -> np.ndarray:
    """The kernel's fp16 chain in numpy (fp32 opmath, one RN to fp16 per stock op; pow = (x*x)*x in Half)."""
    f = np.float32
    h = lambda v: v.astype(np.float16)  # noqa: E731
    xf = x16.astype(f)
    t1 = h(f(0.5) * xf).astype(f)
    p1 = h(xf * xf).astype(f)
    p = h(p1 * xf).astype(f)
    t3 = h(f(c0) * p).astype(f)
    t4 = h(xf + t3).astype(f)
    t5 = h(f(c1) * t4).astype(f)
    t6 = h(np.tanh(t5)).astype(f)
    t7 = h(t6 + f(1.0)).astype(f)
    return h(t1 * t7)


def test_gelu_fp16_chain_matches_torch_cpu_on_the_finite_range():
    """torch's CPU fp16 gelu_new (fp32 opmath, RN per op) vs the chain the kernel encodes; tanh differs only by libm vs
    numpy's tanh, so the check is over values where both agree (the GPU sweep is exhaustive)."""
    A = pytest.importorskip("transformers.activations")
    x = torch.linspace(-8, 8, 4001).to(torch.float16)
    ref = A.gelu_new(x).numpy().view(np.int16)
    c0, c1 = P.gelu_scalars(torch.float16)
    got = _chain_fp16_numpy(x.numpy(), c0, c1).view(np.int16)
    assert (ref == got).mean() > 0.995


def test_rotary_head_mapping_mirrors_the_stock_split():
    """The kernel's column arithmetic (m = h // (H/8), g = h % (H/8); column m*3E/8 + which*E/8 + g*hd + d) vs the stock's
    reshape(mp_num=8) / split(local_dim) / _split_heads on an integer-valued qkv."""
    for (E, H) in ((1024, 16), (1536, 16), (2560, 32), (4096, 16)):
        hd = E // H
        B, L = 2, 3
        qkv = torch.arange(B * L * 3 * E, dtype=torch.float32).reshape(B, L, 3 * E)
        qkv_split = qkv.reshape(qkv.shape[:-1] + (8, -1))
        local_dim = hd * H // 8
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)

        def split_heads(x):
            r = x.reshape(x.shape[:-1] + (H // 8, hd))
            return r.reshape(x.shape[:-2] + (-1,) + r.shape[-1:])
        sq, sk, sv = split_heads(query), split_heads(key), split_heads(value)
        Hg, Eg = H // 8, E // 8
        mq = torch.empty(B, L, H, hd)
        mk = torch.empty(B, L, H, hd)
        mv = torch.empty(B, L, H, hd)
        for h in range(H):
            m, g = h // Hg, h % Hg
            base = m * 3 * Eg + g * hd
            mq[:, :, h, :] = qkv[:, :, base: base + hd]
            mv[:, :, h, :] = qkv[:, :, base + Eg: base + Eg + hd]
            mk[:, :, h, :] = qkv[:, :, base + 2 * Eg: base + 2 * Eg + hd]
        assert torch.equal(mq, sq) and torch.equal(mk, sk) and torch.equal(mv, sv), (E, H)


def test_rotary_pair_formula_matches_the_stock_ops_on_cpu():
    """out[2j] = RN(RN(x0*c) + RN((-x1)*s)), out[2j+1] = RN(RN(x1*c) + RN(x0*s)) in fp32 with fp16 x promoted — the stock's
    (x*cos) + (rotate_every_two(x)*sin) with repeat_interleave, evaluated by torch on the CPU (fp32, no FMA in these ops)."""
    torch.manual_seed(0)
    B, L, H, rd = 2, 5, 4, 32
    x = (torch.randn(B, L, H, rd) * 3).to(torch.float16)
    tab = torch.randn(L, rd // 2)
    cos, sin = torch.cos(tab), torch.sin(tab)
    cos_il, sin_il = cos[None, :, None, :].repeat_interleave(2, 3), sin[None, :, None, :].repeat_interleave(2, 3)
    x1, x2 = x[..., ::2], x[..., 1::2]
    rot = torch.stack((-x2, x1), dim=-1).flatten(-2)
    ref = (x * cos_il) + (rot * sin_il)
    xf = x.float()
    x0, xo = xf[..., ::2], xf[..., 1::2]
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    e0 = (x0 * c) + ((-xo) * s)
    e1 = (xo * c) + (x0 * s)
    got = torch.stack((e0, e1), dim=-1).flatten(-2)
    assert ref.dtype == torch.float32 and torch.equal(got.view(torch.int32), ref.view(torch.int32))


def test_residual_chain_dtype_promotion():
    torch.manual_seed(1)
    a = (torch.randn(1000) * 4).to(torch.float16)
    f = (torch.randn(1000) * 4).to(torch.float16)
    r = torch.randn(1000) * 4
    ref = a + f + r
    got = (a.float() + f.float()).to(torch.float16).float() + r
    assert ref.dtype == torch.float32 and torch.equal(got.view(torch.int32), ref.view(torch.int32))
    assert torch.promote_types(torch.float16, torch.float32) == torch.float32


def test_glue_scalars_use_the_module_scale_and_the_reciprocal_rule():
    attn = types.SimpleNamespace(scale_attn=torch.sqrt(torch.tensor(96, dtype=torch.float32)).to(torch.float16), masked_bias=torch.tensor(-1e9))
    inv, mv = P.glue_scalars(attn, torch.float32)
    assert attn.scale_attn.item() == 9.796875 and inv == (torch.tensor(1.0) / torch.tensor(9.796875)).item()
    assert mv == float(np.float32(-1e9))
    inv16, mv16 = P.glue_scalars(attn, torch.float16)
    assert inv16 == inv and mv16 == float("-inf")
    P._state["scalars"].clear()


def test_ext_load_refuses_missing_binary_or_another_cuda_runtime(tmp_path, monkeypatch):
    """The load's stack check is the CUDA runtime MAJOR the library binds (BUILD.json nvcc release / the +cu tag), never torch's version
    string: a library built for another runtime major refuses by name with the rebuild command; any torch on the same major passes the
    check (here it then reaches ctypes, which rejects the fake bytes — the proof the version check let it through)."""
    monkeypatch.setattr(ext, "_lib", None)
    with pytest.raises(ext.KitRefused, match="missing"):
        ext.load(path=str(tmp_path / "libew_progen2.so"))
    so = tmp_path / "libew_progen2.so"
    so.write_bytes(b"not a library")
    assert ext.cuda_major_of_build({"nvcc": ["Cuda compilation tools, release 12.8, V12.8.93"], "torch": "2.8.0+cu128"}) == 12
    assert ext.cuda_major_of_build({"torch": "2.4.1+cu118"}) == 11 and ext.cuda_major_of_build({"torch": "2.8.0"}) is None
    monkeypatch.setattr(ext, "cuda_major_of_torch", lambda: 12)
    json.dump({"torch": "2.4.1+cu118", "nvcc": ["Cuda compilation tools, release 11.8, V11.8.89"]}, open(tmp_path / "BUILD.json", "w"))
    with pytest.raises(ext.KitRefused, match="binds the CUDA 11 runtime.*rebuilds it on this stack"):
        ext.load(path=str(so))
    json.dump({"torch": "2.9.1+cu128", "nvcc": ["Cuda compilation tools, release 12.8, V12.8.93"]}, open(tmp_path / "BUILD.json", "w"))   # another torch on the same runtime major: the check passes
    monkeypatch.setattr(ext, "_cudart_preload", lambda: None)
    with pytest.raises(OSError):
        ext.load(path=str(so))
    assert ext._lib is None


def test_kernel_image_gate_by_compute_capability():
    """BUILD.json's gencode list decides which image of the library serves a card: a SASS image of the card's major at or below its minor
    (sm_80 serves 8.0 / 8.6 / 8.9 — binary compatible within a major; sm_90 9.0; sm_100 10.0 / 10.3), else the PTX image the driver
    JIT-compiles for a newer card (compute_90 serves 12.0: engaged, named as a note), else none (7.5 is older than every image: the levers
    cannot run there — named, never silent). A record without a gencode list is `unknown` (engage; a launch failure would name itself)."""
    build = json.load(open(os.path.join(KIT_DIR, "BUILD.json")))
    assert sorted(g for g in build["gencode"]) == ["-gencode=arch=compute_100,code=sm_100", "-gencode=arch=compute_80,code=sm_80", "-gencode=arch=compute_90,code=compute_90", "-gencode=arch=compute_90,code=sm_90"]
    for cc, want in (((8, 0), ("sass", (8, 0))), ((8, 6), ("sass", (8, 0))), ((8, 9), ("sass", (8, 0))), ((9, 0), ("sass", (9, 0))),
                     ((10, 0), ("sass", (10, 0))), ((10, 3), ("sass", (10, 0))), ((12, 0), ("ptx", (9, 0)))):
        assert ext.kernel_image(build, cc) == want, cc
    kind, why = ext.kernel_image(build, (7, 5))
    assert kind is None and why.startswith("no kernel image of libew_progen2.so runs on sm_75 (SASS sm_80, sm_90, sm_100: same major, minor at or below the card's; PTX compute_90: a card at or above it) — "), why
    assert ext.kernel_image({}, (9, 0)) == ("unknown", None) and ext.kernel_image({"gencode": []}, (7, 0)) == ("unknown", None)
    assert ext.build_record(os.path.join(KIT_DIR, ext.BINARY)) == build and ext.build_record("/nonexistent/dir/libew_progen2.so") == {}


def test_apply_names_the_untested_and_refuses_what_cannot_run_by_lever(monkeypatch):
    """The lever policy at apply (CPU, the model and the library stubbed): a running stack off the tested one and transformers' gelu_new off the
    bytes the kernel was tested against are NOTES — untested, every lever engaged; a library that cannot load (missing / another CUDA runtime
    major / bytes that do not load) or a card with no kernel image is a KitRefused naming the five levers (`ew:gelu, ew:rotary, ew:residual,
    ew:glue, ew:ln cannot run: <reason>`) — the caller's mode refuses, nothing bound; a cc 12.0 card engages from the PTX image, named."""
    dev = types.SimpleNamespace(type="cpu")
    model = types.SimpleNamespace(transformer=types.SimpleNamespace(wte=types.SimpleNamespace(weight=types.SimpleNamespace(device=dev))), config=types.SimpleNamespace(n_layer=2),
                                  register_forward_pre_hook=lambda fn, with_kwargs=False: types.SimpleNamespace(remove=lambda: None))
    monkeypatch.setattr(v0_ew, "stock_source_pins", lambda: {"modeling_progen.py": "ab"})
    stack_note = "kit v0_ew: the running stack differs from the tested one — torch 2.9.1+cu128 (tested on 2.8.0+cu128)"
    monkeypatch.setattr(v0_ew, "running_stack", lambda: ({"torch": "2.9.1+cu128", "transformers": "4.16.2"}, [stack_note]))
    gelu_note = "kit v0_ew: transformers/activations.py sha256 0badf00d != the tested 6065ad15: the gelu kernel was tested against other bytes of gelu_new (engaged)"
    monkeypatch.setattr(v0_ew, "stock_bytes", lambda: ({"activations.py": "0badf00d"}, [gelu_note]))
    bound = []
    monkeypatch.setattr(v0_ew, "_apply_patches", lambda m, levers, shadow, ln_variant: (bound.append(levers), {"ln_variant": ln_variant, "shadow": shadow})[1])
    monkeypatch.setattr(ext, "load", lambda *a, **k: object())
    monkeypatch.setattr(ext, "record", lambda: {"path": "/x/libew_progen2.so", "build": {"ln_variant": 1, "gencode": ["-gencode=arch=compute_90,code=sm_90"]}})
    monkeypatch.setattr(v0_ew, "_state", dict(v0_ew._state))
    rec = v0_ew.apply(model, size="progen2-small", require_gpu=False)
    assert rec["levers"] == list(v0_ew.LEVERS) and bound == [tuple(v0_ew.LEVERS)] and rec["notes"] == [stack_note, gelu_note] and rec["ln_variant"] == 1 and "off" not in rec and "applied" not in rec
    with pytest.raises(v0_ew.KitRefused, match="already applied"):                 # a second apply in one process stays a refusal (a caller's bug, not the box)
        v0_ew.apply(model, size="progen2-small", require_gpu=False)
    names = "ew:gelu, ew:rotary, ew:residual, ew:glue, ew:ln"
    # the library cannot load: the five levers refused by name with the loader's own words, nothing bound
    monkeypatch.setattr(v0_ew, "_state", dict(v0_ew._state, applied=False))
    bound.clear()
    def missing(*a, **k): raise ext.KitRefused("kit v0_ew: prebuilt library missing at /x/libew_progen2.so — " + ext.REBUILD)
    monkeypatch.setattr(ext, "load", missing)
    with pytest.raises(v0_ew.KitRefused) as ei:
        v0_ew.apply(model, size="progen2-small", require_gpu=False)
    assert str(ei.value) == f"kit v0_ew: {names} cannot run: kit v0_ew: prebuilt library missing at /x/libew_progen2.so — {ext.REBUILD}" and bound == []
    def unloadable(*a, **k): raise OSError("/x/libew_progen2.so: invalid ELF header")
    monkeypatch.setattr(ext, "load", unloadable)
    with pytest.raises(v0_ew.KitRefused, match=f"^kit v0_ew: {names} cannot run: libew_progen2.so does not load \\(OSError: /x/libew_progen2.so: invalid ELF header\\) — "):
        v0_ew.apply(model, size="progen2-small", require_gpu=False)
    with pytest.raises(v0_ew.KitRefused, match=f"^kit v0_ew: {names} cannot run: no CUDA device"):   # require_gpu on a box without one: the same words
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        v0_ew.apply(model, size="progen2-small")
    # a card with no kernel image (the model on a cc 7.5 device): refused by name; a cc 12.0 card: engaged from PTX, named; the house cards: silent
    monkeypatch.setattr(ext, "load", lambda *a, **k: object())
    monkeypatch.setattr(ext, "record", lambda: {"path": "/x/libew_progen2.so", "build": json.load(open(os.path.join(KIT_DIR, "BUILD.json")))})
    dev.type = "cuda"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    with pytest.raises(v0_ew.KitRefused, match=f"^kit v0_ew: {names} cannot run: no kernel image of libew_progen2.so runs on sm_75 "):
        v0_ew.apply(model, size="progen2-small", require_gpu=False)
    assert bound == []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (12, 0))
    rec = v0_ew.apply(model, size="progen2-small", require_gpu=False)
    assert rec["levers"] == list(v0_ew.LEVERS) and "ew kernels JIT-compiled from PTX (compute_90) on sm_120: no prebuilt image for this card in libew_progen2.so" in rec["notes"]
    for cc in ((8, 0), (8, 6), (8, 9), (9, 0), (10, 0), (10, 3)):                    # the house cards and their binary-compatible kin: engaged, nothing to note about the image
        monkeypatch.setattr(v0_ew, "_state", dict(v0_ew._state, applied=False))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None, cc=cc: cc)
        rec = v0_ew.apply(model, size="progen2-small", require_gpu=False)
        assert rec["levers"] == list(v0_ew.LEVERS) and not any("PTX" in n for n in rec["notes"]), cc


def test_kernel_chains_use_only_rn_intrinsics_and_no_fast_math():
    """Every float assignment inside the arithmetic kernels goes through an explicit *_rn intrinsic, a conversion, or a library
    transcendental (tanhf, rsqrtf); the build never passes fast-math."""
    src = open(os.path.join(KIT_DIR, "kernels.cu")).read()
    body = src.split("namespace {", 1)[1].split("}  // namespace")[0]
    allowed = re.compile(r"__f(mul|add|sub|div|maf)_rn|tof<|rt<|h2f|tanhf|rsqrtf|__shfl_down_sync|meansigmabuf|countbuf|costab|sintab|\bc\.|dA\.|dB\.|\bwd\.|\bo\.|\bx\[|\bw\[|\bqkv\[|masked_value|inv_scale|amask\[")
    for line in body.splitlines():
        s = line.strip()
        if not s.startswith("float ") or "=" not in s or s.startswith("float*"):
            continue
        rhs = s.split("=", 1)[1]
        if re.search(r"[\w\)\]]\s*[*+/-]\s*[\w\(]", rhs) and not allowed.search(rhs):
            raise AssertionError(f"bare arithmetic on an opmath value: {s}")
    from engines.progen2.kits.v0_ew import build
    assert not any("fast_math" in f or "fmad" in f or "ftz" in f or "prec-div" in f for f in build.FLAGS + build.GENCODE)
