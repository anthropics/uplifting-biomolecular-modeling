"""The K2B / K2 flash triangle-attention Triton kernels the kit binds BY NAME (opt_core ``kernels/fpf_triatt_k2b``: the Tier-2 attention
slot of ``src/ptx_trunk2_levers.py`` -- the block core's ``PTX_BLK_ATT=k2b`` word (the slot's kernel on the cc-10.x rows; the cc-9.0
rows put ``triattn_native`` / ``triattn_cuda`` there and the cc-8.0 block core runs the stock statement), the lean / row-chunked statements,
the generic provider slot's below-threshold branch -- and the registry op ``fpf_triatt_k2b:fn`` over the stock module path) form their IN-ROW
element offsets in 32-bit arithmetic: the query tile ``(S-1)*position_stride + D``, the key-tile bases ``start_n*position_stride`` of
k / v, the key index times the mask's key stride, the bias staging pass over the caller's bias strides, and the staged fp32 bias
``S*ceil16(S)`` (opt_core 0.5.26.5's audit; its face ``kernels.triattn`` refuses BY NAME ``int32_offset:...`` -- the kit's direct
binding of ``fpf_triatt_k2b.attn_k2b`` does not pass through that face, so the kit's safety is its operand LAYOUT).  Batch, pair-row
and head bases are 64-bit in the kernels.

This test pins the kit's side of that contract on the CPU (integer arithmetic through opt_core's public ``int32_offset_terms`` /
``int32_offsets_ok`` / ``position_stride_bound``; no GPU, milliseconds):
  1. every kit call site hands the kernel operands whose position stride is ``D`` (the prologue's freshly written ``[I,H,J,D]`` q / k / v,
     opt_core ``kernels/fpf_triatt_pro/prologue.py``: ``torch.empty((NI, H, NJ, D))``) or ``H*D`` (the stock module's ``_prep_qkv``
     view ``[I,J,H*D] -> view [I,J,H,D] -> transpose(-2,-3)``): through the big x1 reach ceiling (3,036 tokens) and far past it
     every 32-bit term stays below 2**31 (largest at 3,036 tokens, H=8, D=32: q position term 776,992; the bias plane terms 9.2e6);
     those layouts cross at 8.4 M (stride H*D) / 67 M (stride D) tokens, the ``[S,S]`` bias plane terms from 46,337;
  2. the layout the kit never passes -- the ending direction served as a transposed VIEW of ``[B,N,H,S,D]`` storage (position stride
     ``N*H*D``) -- crosses exactly where opt_core says (1,449 tokens @ H16/D64, 2,897 @ H4/D64, 4,097 @ H4/D32) and at 2,897 for this
     model's H8/D32 pair stacks, i.e. INSIDE big's reach: the reason the x-frame convention below is load-bearing;
  3. the widening arithmetic: the same products in int64 are the true element offsets where int32 wraps;
  4. the glue's x-frame convention by source: every prologue call in ``ptx_trunk2_levers.py`` passes ``ending=False`` (the ending
     direction is ``z.transpose(-2,-3)`` handed to a prologue that WRITES the transposed frame -- never served through strides).
"""
import os
import re

import numpy as np
import pytest

from protenix_opt import stack
from opt_core.kernels import triattn as T

INT32_LIMIT = 2 ** 31
PAIR_H, PAIR_D = 8, 32            # Protenix 2.0 pairformer / MSA-module / confidence-head pair stacks (src/ptx_native_core.py HEAD_DIM, HEADS)
TMPL_H, TMPL_D = 2, 32            # the c=64 template pair stack
REACH_BIG_X1 = 3036             # big x1 reach ceiling on the 80 GB card (kit README reach row)
SIZES = (20, 33, 199, 300, 400, 800, 1200, 1340, 2048, 2530, REACH_BIG_X1, 4096, 5120, 8192)
CROSSINGS = ((1449, 16, 64), (2897, 4, 64), (4097, 4, 32), (2897, PAIR_H, PAIR_D))   # opt_core 0.5.26.5's three + this model's pair stacks
FPF_SRC = os.path.join(stack.kit_home(), "src", "ptx_trunk2_levers.py")


def _terms(n, h, d, pos_stride, mask_key_stride=0):
    """opt_core's named 32-bit terms for an [.., S=n, D=d] call: q/k/v at ``pos_stride``, the kit's bias ``[1,1,H,S,S]`` fp32 contiguous
    (query stride S, key stride 1), mask absent (the kit's rows pass mask=None: lever ``nomask``) unless a key stride is given."""
    return T.int32_offset_terms(n, n, d, q_pos_stride=pos_stride, k_pos_stride=pos_stride, v_pos_stride=pos_stride,
                                mask_key_stride=mask_key_stride, bias_q_stride=n, bias_k_stride=1)


@pytest.mark.parametrize("h,d", [(PAIR_H, PAIR_D), (TMPL_H, TMPL_D)])
@pytest.mark.parametrize("n", SIZES)
def test_kit_layouts_stay_below_the_int32_bound(n, h, d):
    for name, ps in (("prologue [I,H,J,D] contiguous", d), ("stock _prep_qkv view", h * d)):
        ok, why = T.int32_offsets_ok(_terms(n, h, d, ps, mask_key_stride=1))
        assert ok, f"{name} at {n} tokens (H={h}, D={d}): {why}"
        assert T.position_stride_bound(n, d) >= ps, (name, n, T.position_stride_bound(n, d))


def test_largest_terms_at_the_big_reach_ceiling_are_the_documented_numbers():
    n, h, d = REACH_BIG_X1, PAIR_H, PAIR_D
    view = _terms(n, h, d, h * d); pro = _terms(n, h, d, d)
    assert pro["q_pos"] == (n - 1) * d + d == 97_152 and view["q_pos"] == (n - 1) * h * d + d == 776_992
    assert view["bias_src"] == n * n - 1 == 9_217_295 and view["bias_staged"] == n * 3040 == 9_229_440       # ceil16(3036) = 3040
    assert max(max(view.values()), max(pro.values())) < INT32_LIMIT // 200                                     # two orders of magnitude of headroom
    # where those layouts WOULD cross: position stride H*D from 8,388,609 tokens, stride D from 67,108,864; the [S,S] bias plane from 46,341
    assert (INT32_LIMIT - d) // (h * d) + 1 == 8_388_608 and _terms(8_388_608, h, d, h * d)["q_pos"] < INT32_LIMIT <= _terms(8_388_609, h, d, h * d)["q_pos"]
    assert _terms(46_340, h, d, d)["bias_src"] < INT32_LIMIT <= _terms(46_341, h, d, d)["bias_src"]
    first_refused = next(n_ for n_ in range(46_300, 46_400) if not T.int32_offsets_ok(_terms(n_, h, d, d))[0])
    assert first_refused == 46_337 > 15 * REACH_BIG_X1                                                        # the staged bias S*ceil16(S) (the term the wrapper checks) is first
    assert T.int32_offsets_ok(_terms(46_336, h, d, d)) == (True, "ok") and T.int32_offsets_ok(_terms(46_337, h, d, d))[1] == "int32_offset:bias_staged=2147812624>=2**31"


@pytest.mark.parametrize("n,h,d", CROSSINGS)
def test_transposed_view_crossings_are_opt_cores_numbers(n, h, d):
    """The ending direction served through strides: q / k / v are [B,S,H,N,D] views of [B,N,H,S,D] storage -> position stride N*H*D."""
    at = T.int32_offsets_ok(_terms(n, h, d, n * h * d)); below = T.int32_offsets_ok(_terms(n - 1, h, d, (n - 1) * h * d))
    assert not at[0] and at[1].startswith("int32_offset:q_pos=") and below == (True, "ok"), (n, h, d, at, below)
    assert T.position_stride_bound(n, d) < n * h * d <= T.position_stride_bound(n - 1, d) + h * d


def test_this_models_view_crossing_is_inside_bigs_reach():
    """H8/D32 has the element count of H4/D64: a strided ending view would cross at 2,897 tokens < the 3,036-token big reach -- so the
    kit's contiguous x-frame operands (test 4) are what keeps every big row inside the kernel's arithmetic, not the size."""
    h, d = PAIR_H, PAIR_D
    first_bad = next(n for n in range(2800, 3100) if not T.int32_offsets_ok(_terms(n, h, d, n * h * d))[0])
    assert first_bad == 2897 < REACH_BIG_X1
    assert not T.int32_offsets_ok(_terms(REACH_BIG_X1, h, d, REACH_BIG_X1 * h * d))[0]          # a view at the reach ceiling: refused / would wrap
    assert T.int32_offsets_ok(_terms(REACH_BIG_X1, h, d, d))[0]                                   # what the kit passes there: fine


@pytest.mark.parametrize("n,h,d", CROSSINGS + ((REACH_BIG_X1, PAIR_H, PAIR_D),))
def test_int64_products_are_the_true_offsets_where_int32_wraps(n, h, d):
    """The kernels' two position products on a transposed view, mirrored: q tile ``offs_mc*sqq + d`` at the last query row and the
    key-tile base ``start_n*skk`` at the last tile start (BLOCK_N 32), evaluated in wrapping int32 (what the un-widened kernel computes)
    and in int64 (the widening): int64 == the exact integer everywhere; int32 differs exactly when the exact value is >= 2**31."""
    ps = n * h * d
    cases = {"q_pos": (n - 1, ps, d - 1), "k_base": ((n // 32) * 32 if n % 32 else n - 32, ps, 0)}
    for name, (idx, stride, add) in cases.items():
        exact = idx * stride + add
        with np.errstate(over="ignore"):
            i32 = int((np.array([idx], dtype=np.int32) * np.array([stride], dtype=np.int32) + np.int32(add))[0])
        i64 = int(np.int64(idx) * np.int64(stride) + np.int64(add))
        assert i64 == exact, (name, n, h, d)
        assert (i32 != exact) == (exact >= INT32_LIMIT), (name, n, h, d, i32, exact)
    assert (n - 1) * ps + d - 1 >= INT32_LIMIT                                                   # every listed shape wraps the query term


def test_glue_writes_the_transposed_frame_never_serves_it_by_strides():
    src = open(FPF_SRC, encoding="utf-8").read()
    calls = re.findall(r"(?:prologue|_pro_pad)\(module[^\n]*", src)
    assert len(calls) >= 4, calls
    assert all("ending=False" in c for c in calls), [c[:80] for c in calls if "ending=False" not in c]
    assert "x = z.transpose(-2, -3) if ending else z" in src                                          # the x-frame view handed to the prologue
    body = src[src.index("def _triatt_block_pro_epi("):src.index("def _blk2_triatt_or_stock(")]
    for stmt in re.findall(r"\n\s*o(?:_r)? = (?:att|att_lean)\((.*?)\)", body):                       # the K2B-slot statements take the prologue's q/k/v as is
        assert "transpose" not in stmt and "permute" not in stmt and ".T" not in stmt, stmt


def test_layout_strides_with_meta_tensors():
    torch = pytest.importorskip("torch")
    i = j = 257; h, d = PAIR_H, PAIR_D
    assert torch.empty((i, h, j, d), device="meta").stride() == (h * j * d, j * d, d, 1)                              # prologue q/k/v: position stride D
    assert torch.empty((i, j, h * d), device="meta").view(i, j, h, d).transpose(-2, -3).stride() == (j * h * d, d, h * d, 1)   # stock _prep_qkv: H*D
    n = 2897
    view = torch.empty((1, n, h, n, d), device="meta").transpose(1, 3)                                              # an ending view of [B,N,H,S,D] storage
    assert view.shape == (1, n, h, n, d) and view.stride(3) == h * n * d and (n - 1) * view.stride(3) + d >= INT32_LIMIT
