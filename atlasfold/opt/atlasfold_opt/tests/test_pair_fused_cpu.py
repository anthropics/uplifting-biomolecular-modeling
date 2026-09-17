"""CPU tests (no GPU, no weights) of the fused pair levers `triatt_block` (hooks/triatt_block.py) and `pair_transition` (hooks/pair_transition_fused.py):
the per-call gate words, the registry's expected vocabulary, the installers table, the fast row's ORDER contract, the core cell verdicts on named
stacks, and the weight PACKING convention — stock fp32 modules against a plain-torch fp32 restatement of the fused math built from the packed views."""
import os, re
import pytest

torch = pytest.importorskip("torch")
HERE = os.path.dirname(os.path.abspath(__file__)); PKG = os.path.dirname(HERE)
DEFECT_WORDS = ("heads:", "mask-shape", "mask-batch", "bias:")        # words the hooks CAN emit that are defects by design: NOT in the registry's expected tuples (they refuse the gate)


def _stock():
    tu = pytest.importorskip("atlasfold.model.network.primitives.triangle_update")
    tr = pytest.importorskip("atlasfold.model.network.primitives.transition")
    return tu, tr


def _randomize(m, seed=0):
    """Random bf16-representable parameters (the stock 'final'/'gating' inits are zeros: a vacuous comparison otherwise); LN weight near 1."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            base = 1.0 if (name.endswith("weight") and p.dim() == 1) else 0.0
            p.copy_((base + torch.randn(p.shape, generator=g) * 0.08).to(torch.bfloat16).float())
    return m.eval()


# ----------------------------------------------------------------------------------------------------------------------------- gate words
def test_triatt_block_gate_words_cpu():
    tu, _ = _stock()
    from atlasfold_opt.hooks import triatt_block as H, pair_cells as PC
    from atlasfold_opt.registry import LEVERS
    m = tu.TriangleAttentionStartingNode(128, 4).eval()
    z32 = torch.zeros(1, 16, 16, 128); mk = torch.ones(1, 16, 16, dtype=torch.bool)
    zb = z32.to(torch.bfloat16)
    assert H.refusal(m, zb, mk, "torch") == "backend_torch"
    assert H.refusal(m, z32, mk) == "dtype:float32"
    assert H.refusal(m, zb, mk) == "below_min_tokens"
    assert H.refusal(m, zb[0], mk[0]) == "below_min_tokens"                                  # 3-D z [L, L, C] + [L, L] mask is a served rank
    assert H.refusal(m, zb.reshape(1, 16, 2, 128 * 8)[..., :128], mk) == "rank"               # not a pair tensor
    big = torch.zeros(1, PC.MIN_TOKENS, PC.MIN_TOKENS, 128, dtype=torch.bfloat16); bigm = torch.ones(1, PC.MIN_TOKENS, PC.MIN_TOKENS, dtype=torch.bool)
    assert H.refusal(m, big, bigm) == "cpu"                                                    # everything structural passes: only the device is left
    assert H.refusal(m, big, bigm[:, :5]) == "mask-shape"
    assert H.refusal(m, big.expand(2, -1, -1, -1), bigm.expand(3, -1, -1)) == "mask-batch"
    m64 = tu.TriangleAttentionEndingNode(64, 4).eval()
    assert H.refusal(m64, torch.zeros(1, 16, 16, 64, dtype=torch.bfloat16), mk) == "c:64"      # the template pair stack's width
    m.train()
    assert H.refusal(m, zb, mk) == "training"
    words = LEVERS["triatt_block"]["expected"]
    for w in ("backend_torch", "dtype:float32", "below_min_tokens", "rank", "cpu", "c:64", "training", "no-cell:fpf:prologue:128x4x32:7.5|3.3+off(not-measured)"):
        assert PC.expected(w, words), w
    for w in ("mask-shape", "mask-batch", "heads:2", "bias:linear_qkv", "stride", "device", "mask-rank"):
        assert not PC.expected(w, words), w                                                    # defects refuse the gate


def test_pair_transition_gate_words_cpu():
    _, tr = _stock()
    from atlasfold_opt.hooks import pair_transition_fused as H, pair_cells as PC
    from atlasfold_opt.registry import LEVERS
    m = tr.Transition(128, 4).eval()
    x32 = torch.zeros(1, 16, 16, 128); xb = x32.to(torch.bfloat16)
    assert H.geometry(m) == (128, 4)
    assert H.refusal(m, x32) == "dtype:float32"
    assert H.refusal(m, xb) == "cpu"                                                                                   # no kit size floor: the provider's cells decide per size
    assert H.refusal(m, xb.reshape(-1, 128)) == "rank"
    assert H.refusal(m, torch.zeros(1, 2, PC.MIN_TOKENS, 128, dtype=torch.bfloat16)) == "cpu"
    assert H.refusal(tr.Transition(384, 4).eval(), torch.zeros(1, 16, 384, dtype=torch.bfloat16)) == "c:384"      # the single track
    assert H.refusal(tr.Transition(128, 2).eval(), xb) == "factor:2"                                                  # the diffusion module's factor-2 transitions
    assert H.geometry(tr.Transition(128, 2)) == (128, 2)
    m.train()
    assert H.refusal(m, xb) == "training"
    words = LEVERS["pair_transition"]["expected"]
    for w in ("dtype:float32", "rank", "cpu", "c:384", "c:768", "factor:2", "training", "stock_row:n256", "refused:no_cell"):
        assert PC.expected(w, words), w
    for w in ("bias:swiglu", "stride", "hidden", "device", "below_min_tokens", "no-cell:fpf:transition:128x512:7.5|3.3"):
        assert not PC.expected(w, words), w


def test_every_literal_reason_word_is_expected_or_a_declared_defect():
    """Every word the two hooks' refusal() can return is either expected by the registry (prefix rule) or one of DEFECT_WORDS — read off the source."""
    from atlasfold_opt.registry import LEVERS
    from atlasfold_opt.hooks import pair_cells as PC
    for fname, lever in (("triatt_block.py", "triatt_block"), ("pair_transition_fused.py", "pair_transition")):
        src = open(os.path.join(PKG, "hooks", fname)).read()
        body = src[src.index("def refusal("):]; body = body[:body.index("\ndef ", 1)]
        lits = re.findall(r'return f?"([A-Za-z_\-]+:?)', body)
        assert lits, fname
        for w in lits:
            assert PC.expected(w, LEVERS[lever]["expected"]) or any(w.startswith(d.rstrip(":")) for d in DEFECT_WORDS), (fname, w)
        assert 'Unsupported("bias:' in src or 'Refusal("bias:' in src                          # the pack refuses a biased linear by name


# ----------------------------------------------------------------------------------------------------------------------- tables & order
def test_installers_and_registry_cover_the_pair_levers():
    from atlasfold_opt.hooks import installers, triatt_block, pair_transition_fused
    from atlasfold_opt.registry import LEVERS
    from atlasfold_opt import modes
    t = installers()
    assert t["triatt_block"] is triatt_block.install and t["pair_transition"] is pair_transition_fused.install
    for lever, words in (("triatt_block", ("no-cell:", "below_min_tokens")), ("pair_transition", ("stock_row:", "refused:"))):
        d = LEVERS[lever]
        assert d["cls"] == "fast" and d["strategy"] == f"LOCAL.atlasfold.{lever}" and all(w in d["expected"] for w in words), lever
        assert lever in modes.MODES["fast"] and lever not in modes.MODES["exact"]


def test_fast_row_order_contract():
    """pair_transition is OUTERMOST over the one transition_chunk wrapper (installed after both chunk levers); triatt_block installs after flash_triattn
    (its stock fallback reaches the flash core through the module-global cueq_tri_attn)."""
    from atlasfold_opt import modes
    row, bg = modes.MODES["fast"], modes.MODES["big"]
    assert row.index("pair_transition") > row.index("pair_transition_chunk") > row.index("conf_transition_chunk")      # the fused transition installs after (outside) the chunk wrapper in fast
    assert bg.index("pair_transition") > bg.index("pair_transition_chunk") > bg.index("conf_transition_chunk")         # ... and in big
    assert row.index("triatt_block") > row.index("flash_triattn")


def test_cell_verdicts_on_named_stacks(monkeypatch):
    pytest.importorskip("opt_core.attn.pair_fused")
    from opt_core.attn import pair_fused as PF
    from atlasfold_opt.hooks import pair_cells as PC
    from atlasfold_opt.registry import LEVERS
    for stack in (("9.0", "3.3"), ("9.0", "3.7"), ("8.0", "3.4"), ("10.0", "3.4")):
        for lever in ("triatt_block", "pair_transition"):
            v = PC.verdict(lever, stack=stack)                                                   # the CLASS contract: served with the card's cells, or unserved by a NAMED reason the
            if v["served"]:                                                                      # registry expects (which capability the core's table measures off is the core's to change)
                assert v["cells"] == f"{stack[0]}|*" and v["settings"] is None and v["reason"] == "", (stack, lever, v)
                assert len(v["cell_rows"].split("+")) == len(PC.PIECES[lever]), v
            else:
                assert v["reason"].startswith("no-cell:") and PC.expected(v["reason"], LEVERS[lever]["expected"]), (stack, lever, v)
    assert PC.verdict("triatt_block", stack=("9.0", "3.3"))["served"] is True                      # the H100 cells this kit was measured on serve
    for lever in ("triatt_block", "pair_transition"):                                          # a capability without rows: the core's SAFE settings serve -> engaged
        v = PC.verdict(lever, stack=("7.0", "3.3"))
        assert v["served"] is True and v["cells"] == "safe" and v["settings"].startswith("safe:no_cell:"), v
    # a capability the table lists as MEASURED OFF: unserved -> the lever installs all the same, calls counted with the core's word
    table = {"schema": "test", "version": 0, "rows": [], "named_off": [{"id": "offT", "impl": "fpf", "piece": "*", "key": "*", "cc": "7.7", "off": "not-measured", "evidence": "test"}]}
    monkeypatch.setattr(PF, "_CELLS", table)
    v = PC.verdict("triatt_block", stack=("7.7", "3.3"))
    assert v["served"] is False and v["cells"] is None and v["reason"].startswith("no-cell:fpf:prologue:128x4x32:7.7|3.3"), v
    from atlasfold_opt.registry import LEVERS
    assert PC.expected(v["reason"], LEVERS["triatt_block"]["expected"])


def test_unserved_capability_installs_anyway(monkeypatch):
    tu, tr = _stock()
    from atlasfold_opt import TAG
    from atlasfold_opt.hooks import pair_cells as PC, triatt_block as HT, pair_transition_fused as HP
    unserved = {"served": False, "cells": None, "cell_rows": None, "settings": None, "reason": "no-cell:fpf:prologue:128x4x32:7.7|3.3+off(not-measured)"}
    monkeypatch.setattr(PC, "verdict", lambda lever, device=None, stack=None: dict(unserved))
    keep = {c: getattr(tu, c).forward for c, _ in HT.CLASSES}; keep_t = tr.Transition.forward
    try:
        for H in (HT,):                                                                         # pair_transition binds the transition provider by tier word: no pair_fused cell verdict to stub
            ins = H.install("fast", TAG, {})
            assert ins.applied and ins.facts["served"] is False and ins.facts["cells"] is None
            line = ins.lines[0]()
            assert " cells=none" in line and " min_tokens=512" in line and "state=skipped reason=no_calls" in line, line
            assert ins.gates[0]().ok                                                            # no calls: a legitimate (empty) census
    finally:
        for c, f in keep.items():
            setattr(getattr(tu, c), "forward", f)
        tr.Transition.forward = keep_t


def test_gate_prefix_semantics():
    from opt_core.counters import Ledger
    from atlasfold_opt.hooks import pair_cells as PC
    words = ("dtype:", "below_min_tokens", "no-cell:")
    L = Ledger("LOCAL.atlasfold.triatt_block", impl="x", expected=words)
    L.fallback("dtype:float32"); L.fallback("no-cell:fpf:prologue:128x4x32:7.7|3.3+off(not-measured)"); L.fallback("below_min_tokens")
    assert PC.gate_for(L, words)().ok                                                          # all-expected fallbacks, served 0: a legitimate stock run
    L.fallback("mask-shape")
    g = PC.gate_for(L, words)()
    assert not g.ok and "mask-shape" in g.reason and "dtype" not in g.reason
    L2 = Ledger("LOCAL.atlasfold.pair_transition", impl="x", expected=words); L2.serve("1x512"); L2.error("RuntimeError")
    assert not PC.gate_for(L2, words)().ok                                                     # a kernel error refuses


# ---------------------------------------------------------------------------------------------------------------- the packing convention
def _restate_triattn(W, z, mask, ending: bool, inf: float = 1e9):
    """Plain-torch fp32 statement of the fused block from the PACKED views (W.wqkvg rows q|k|v|g head-major, W.wb, W.wo16, W.lnw32/lnb32):
    the attention runs in the frame x = z^T for the ending node; logits[b,i,h,q,k] = scale q.k + bias[b,h,q,k] - inf (1 - mask[b,i,k])."""
    F = torch.nn.functional
    x = z.transpose(-2, -3) if ending else z
    m = mask.transpose(-1, -2) if ending else mask
    ln = F.layer_norm(x, (W.C,), W.lnw32, W.lnb32, W.eps)
    HD = W.H * W.D
    q, k, v, g = (ln @ W.wqkvg.float().t()).split(HD, dim=-1)
    sh = lambda t: t.unflatten(-1, (W.H, W.D)).permute(0, 1, 3, 2, 4)                          # [B,I,J,(h d)] -> [B,I,H,J,D]
    q, k, v = sh(q), sh(k), sh(v)
    bias = (ln @ W.wb.float().t()).permute(0, 3, 1, 2)[:, None]                                # [B,I,J,H] -> [B,1,H,I,J]
    a = (q * W.D ** -0.5) @ k.transpose(-1, -2) + bias - inf * (~m)[:, :, None, None, :].float()
    o = (a.softmax(-1) @ v).permute(0, 1, 3, 2, 4).reshape(*x.shape[:-1], HD)
    u = (torch.sigmoid(g) * o) @ W.wo16.float().t()
    return u.transpose(-2, -3) if ending else u


@pytest.mark.parametrize("B,L", [(1, 16), (2, 64)])
def test_triattn_packing_matches_stock_fp32_cpu(B, L):
    tu, _ = _stock()
    pytest.importorskip("opt_core.attn.pair_fused")
    from atlasfold_opt.hooks import triatt_block as H
    g = torch.Generator().manual_seed(10 + L)
    z = torch.randn(B, L, L, 128, generator=g)
    mask = torch.rand(B, L, L, generator=g) > 0.15
    for cls_name, ending in H.CLASSES:
        m = _randomize(getattr(tu, cls_name)(128, 4), seed=3 + int(ending))
        with torch.no_grad():
            ref = m(z.clone(), mask, kernel_backend="torch")
            W = H.weights(m)
            assert W.C == 128 and W.H == 4 and W.D == 32 and m._afo_triatt_w is W
            got = _restate_triattn(W, z, mask, ending)
        err = float((got - ref).abs().max())
        assert err <= 1e-5, (cls_name, B, L, err, float(ref.abs().max()))
        for ones in (True,):
            with torch.no_grad():
                full = torch.ones(B, L, L, dtype=torch.bool)
                err1 = float((_restate_triattn(W, z, full, ending) - m(z.clone(), full, kernel_backend="torch")).abs().max())
            assert err1 <= 1e-5, (cls_name, "all-ones mask", err1)


@pytest.mark.parametrize("B,L", [(1, 16), (2, 64)])
def test_transition_packing_matches_stock_fp32_cpu(B, L):
    _, tr = _stock()
    KT = pytest.importorskip("opt_core.kernels.transition")
    from atlasfold_opt.hooks import pair_transition_fused as H, transition_exact as X
    m = _randomize(tr.Transition(128, 4), seed=7)
    x = torch.randn(B, L, L, 128, generator=torch.Generator().manual_seed(20 + L))
    with torch.no_grad():
        ref = torch.nn.Sequential.forward(m, x)                                                # the stock statement itself, whatever wraps Transition.forward in this process
        W = H.kt_weights(m)
        assert (W.c, W.hidden) == (128, 512) and m._afo_kt_transition_w is W and X.weights(m, KT) is W   # one pack per module, shared by the two levers
        err = float((KT.reference(x, W, dtype=torch.float32) - ref).abs().max())             # the provider's statement from the PACKED weights (rows a then b = the stock chunk order)
    assert err <= 1e-5, (B, L, err, float(ref.abs().max()))
