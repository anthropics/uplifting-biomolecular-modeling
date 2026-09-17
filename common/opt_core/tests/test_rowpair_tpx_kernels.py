"""The fused-kernel seams of the row-pair family on CPU rank-threads (the kernels themselves need CUDA; their GPU numerics are an op-level box):

* ``trimul_update_`` with a plain ``TriMulFns`` (no hooks) == the dense statement, and a provider whose ``proj_into`` / ``tile_epilogue`` hooks
  DECLINE every unit gives byte-identical shards — the hook-less path is the path every existing adapter runs;
* a provider implementing the two hooks with the SAME torch statements (projection written into the GEMM block; the tile epilogue through the
  staged column window, as the K3 launch does) gives byte-identical shards at P in {2, 3}, outgoing / incoming, add on / off, with masks;
* ``fused_trimul_fns`` below its size gate / opted out declines by name and the update equals the plain one; where the lever cannot run
  (CPU tensors) it raises ``RowpairRefused``; no cells row for a card = the lever's safe settings; a build failure = safe, then refusal;
  a wrong weight vocabulary is refused;
* ``attention_core``: ``kernel="torch"`` == the engine core through ``attend_query_blocks`` for every query block, counted ``kernel_torch``;
  ``flash_triattn`` on CPU tensors falls back BY NAME to the engine core (equal values, the reason in the ledger), also through the row-window
  split and the ``bias_elems>int32`` refusal (``ROWPAIR_TRIATT_INT32_GUARD=1``); a plane above the int32 bias bound reaches the kernel (a recording
  stand-in here) PER QUERY BLOCK on the caller's bias view, one serve per row window; an unknown kernel word and a bad layout word are refused;
  ``ROWPAIR_TRIATT_CORE`` overrides.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused                            # noqa: E402
from opt_core.mem.rowpair import trimul as TM, trimul_fused as RF, triatt as RA   # noqa: E402
from opt_core.mem.rowpair.dist import Layout, all_gather_rows                # noqa: E402
from opt_core.testing import run_ranks                                       # noqa: E402
from opt_core import trimul as CORE_TRIMUL                                   # noqa: E402

N, C, C_H = 40, 16, 8
F = torch.nn.functional


def _weights(g):
    def r(*shape, s=0.3):
        return torch.randn(*shape, generator=g) * s
    return dict(ln_in_w=1.0 + r(C, s=0.1), ln_in_b=r(C, s=0.1), w_ag=r(C_H, C), w_ap=r(C_H, C), w_bg=r(C_H, C), w_bp=r(C_H, C),
                ln_out_w=1.0 + r(C_H, s=0.1), ln_out_b=r(C_H, s=0.1), w_o=r(C, C_H), w_og=r(C, C))


def _stock_fns(w):
    """The AF3-family statements (what every adapter hands the driver today)."""
    def proj(zb, mb, is_a):
        x = F.layer_norm(zb, (C,), w["ln_in_w"], w["ln_in_b"], 1e-5)
        wg, wp = (w["w_ag"], w["w_ap"]) if is_a else (w["w_bg"], w["w_bp"])
        return torch.sigmoid(F.linear(x, wg)) * F.linear(x, wp) * mb

    def out(x):
        return F.linear(F.layer_norm(x, (C_H,), w["ln_out_w"], w["ln_out_b"], 1e-5), w["w_o"])

    def gate(zb):
        return torch.sigmoid(F.linear(F.layer_norm(zb, (C,), w["ln_in_w"], w["ln_in_b"], 1e-5), w["w_og"]))

    return TM.TriMulFns(proj, out, gate, C_H)


class _Declining(TM.TriMulFns):
    """Carries the hooks; every unit declined -> the driver's torch statements (the fallback branch of a fused provider)."""

    def __init__(self, fns):
        TM.TriMulFns.__init__(self, fns.proj, fns.out, fns.gate, fns.C_h)
        self.declined = 0

    def proj_into(self, dst, z_block, mask_block, is_a):
        self.declined += 1
        return False

    def tile_epilogue(self, T, z_block, add):
        self.declined += 1
        return False


class _TorchHooks(TM.TriMulFns):
    """The two hooks implemented with the same torch statements: the projection written straight into its channel-major block, the tile epilogue
    through the staged contiguous column window (copy in, update in place, copy out) — the protocol of the fused provider without its kernels."""

    def __init__(self, fns):
        TM.TriMulFns.__init__(self, fns.proj, fns.out, fns.gate, fns.C_h)
        self.k1 = self.k3 = 0

    def proj_into(self, dst, z_block, mask_block, is_a):
        assert tuple(dst.shape) == (self.C_h, int(z_block.shape[0]), int(z_block.shape[1])) and dst.stride(2) == 1
        dst.copy_(self.proj(z_block, mask_block.unsqueeze(-1), is_a).permute(2, 0, 1))
        self.k1 += 1
        return True

    def tile_epilogue(self, T, z_block, add):
        x = self.out(T.permute(1, 2, 0))
        g = self.gate(z_block)
        x = x.mul_(g) if x.is_contiguous() else x * g
        zs = z_block if z_block.is_contiguous() else z_block.contiguous()      # staged window
        if add:
            zs += x
        else:
            zs.copy_(x)
        if zs.data_ptr() != z_block.data_ptr():
            z_block.copy_(zs)
        self.k3 += 1
        return True


def _dense_update(z, mask, fns, outgoing, add):
    """The dense statement: z (+)= gate(z) * out(contraction(proj_a, proj_b))."""
    a = fns.proj(z, mask.unsqueeze(-1), True)
    b = fns.proj(z, mask.unsqueeze(-1), False)
    x = TM.trimul_dense(a, b, outgoing)
    y = fns.out(x) * fns.gate(z)
    return z + y if add else y


def _rank(rank, P, kind, outgoing, add, B):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(N, N, C, generator=g)
    mask = (torch.rand(N, N, generator=g) > 0.15).float()
    w = _weights(g)
    stock = _stock_fns(w)
    fns = {"plain": stock, "declining": _Declining(stock), "hooks": _TorchHooks(stock), "fused_cpu": None, "fused_cpu_envtorch": None, "fused_cpu_gate0": None}[kind]
    if kind == "fused_cpu":
        fns = RF.fused_trimul_fns(w, stock)                                 # default size gate: N=40 < 2048 -> below_gate
    elif kind == "fused_cpu_envtorch":
        os.environ[RF.ENV_KERNELS] = "torch"                                # the opt-out (read per call, on this rank thread too)
        fns = RF.fused_trimul_fns(w, stock, min_tokens=0)
    elif kind == "fused_cpu_gate0":
        fns = RF.fused_trimul_fns(w, stock, min_tokens=0)                   # no gate, CPU tensors: the lever cannot run -> RowpairRefused
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone()                                          # a shard COPY (rank 0's slice of a contiguous z is z's own storage)
    TM.trimul_update_(fns, zs, mask[lay.r0:lay.r1].clone(), lay, outgoing=outgoing, add=add, inplace_chunk=8, RB=8)
    full = all_gather_rows(zs, lay)
    ref = _dense_update(z, mask, stock, outgoing, add)
    extra = {}
    if kind == "declining":
        extra["declined"] = fns.declined
    if kind == "hooks":
        extra["k1"], extra["k3"] = fns.k1, fns.k3
    if kind in ("fused_cpu", "fused_cpu_envtorch", "fused_cpu_gate0"):
        extra["fallback_by"] = dict(fns.ledger.fallbacks)
    return full, ref, extra


@pytest.mark.parametrize("P,B", [(2, 8), (3, 8), (2, 20)])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("add", [True, False])
def test_hook_seam_is_byte_identical(P, B, outgoing, add):
    plain = run_ranks(P, _rank, "plain", outgoing, add, B)
    decl = run_ranks(P, _rank, "declining", outgoing, add, B)
    hooks = run_ranks(P, _rank, "hooks", outgoing, add, B)
    full_plain, ref, _ = plain[0]
    assert torch.allclose(full_plain, ref, rtol=1e-4, atol=1e-5), float((full_plain - ref).abs().max())     # the sharded schedule == the dense statement
    for r in range(P):
        assert torch.equal(plain[r][0], full_plain)
        assert torch.equal(decl[r][0], full_plain), "a declining provider must run today's statements byte for byte"
        assert decl[r][2]["declined"] > 0
        assert torch.equal(hooks[r][0], full_plain), float((hooks[r][0] - full_plain).abs().max())
        assert hooks[r][2]["k1"] > 0 and hooks[r][2]["k3"] > 0


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("kind,reason", [("fused_cpu", "below_gate"), ("fused_cpu_envtorch", "env_torch")])
def test_fused_provider_declines_by_name_on_cpu(outgoing, kind, reason, monkeypatch):
    monkeypatch.delenv(RF.ENV_KERNELS, raising=False)
    RF.LEDGER.clear()
    try:
        res = run_ranks(2, _rank, kind, outgoing, True, 8)
    finally:
        os.environ.pop(RF.ENV_KERNELS, None)
    plain = run_ranks(2, _rank, "plain", outgoing, True, 8)
    for r in range(2):
        assert torch.equal(res[r][0], plain[r][0])
        assert res[r][2]["fallback_by"].get(reason, 0) > 0 and len(res[r][2]["fallback_by"]) == 1, res[r][2]


def test_fused_provider_refuses_where_it_cannot_run():
    """No size gate and CPU tensors: the lever cannot run in this process → ``RowpairRefused`` naming the lever and the opt-out (raised from
    ``operand_dtype`` before the driver touches z — the shard is left as it was), never a silent plain path."""
    g = torch.Generator().manual_seed(3)
    w = _weights(g)
    fns = RF.fused_trimul_fns(w, _stock_fns(w), min_tokens=0)
    with pytest.raises(RowpairRefused, match=r"F2\.trimul_rows .*not a CUDA device.*ROWPAIR_TRIMUL_KERNELS=torch"):
        fns.operand_dtype(torch.zeros(2, N, C))                             # what the driver asks first, before it touches z
    with pytest.raises(Exception, match=r"not a CUDA device"):             # through the driver on rank threads (the hub re-raises a rank's exception)
        run_ranks(2, _rank, "fused_cpu_gate0", True, True, 8)


def test_resolve_cells_no_row_serves_safe_settings(capsys, monkeypatch):
    """A card without a cells row is served with the lever's SAFE settings (one stderr line, census word), a tuned row or an explicit mapping
    wins, and a capability without safe settings is the lever's refusal."""
    from opt_core.kernels import safe_settings as SAFE
    net = SAFE.SafeNet(RF.SAFE_LEVER, refused=RowpairRefused)
    tuned = {"k1": {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    assert RF.resolve_cells(net, "8.0", None, tuned, "8.0|3.7", {}) == (tuned, "8.0|3.7") and not net.on
    assert RF.resolve_cells(net, "8.0", {"k1": {}, "k3": {}}, None, None, {}) == ({"k1": {}, "k3": {}}, "explicit") and not net.on
    cfg, key = RF.resolve_cells(net, "8.0", None, None, None, {}, triton_mm="3.7")
    assert key == "safe" and cfg == SAFE.SAFE_ROWS[RF.SAFE_LEVER]["8.0"]["settings"] and net.on and net.word() == "safe:no_cell:rows"
    err = capsys.readouterr().err
    assert err.count("safe settings served (no_cell:rows, cc 8.0, triton 3.7)") == 1, err
    assert RF.resolve_cells(net, "8.0", None, None, None, {})[1] == "safe" and capsys.readouterr().err == ""     # idempotent: no second line
    net120 = SAFE.SafeNet(RF.SAFE_LEVER, refused=RowpairRefused)                              # a capability no engine certified: the any-capability SAFE row, by name
    cfg, key = RF.resolve_cells(net120, "12.0", None, None, None, {}, triton_mm="3.7")
    assert key == "safe" and cfg == SAFE.SAFE_ROWS[RF.SAFE_LEVER]["*"]["settings"] and net120.word() == "safe:no_cell:rows"
    assert capsys.readouterr().err.count("safe settings served (no_cell:rows, cc 12.0, triton 3.7)") == 1
    monkeypatch.setitem(SAFE.SAFE_ROWS, RF.SAFE_LEVER, {k: v for k, v in SAFE.SAFE_ROWS[RF.SAFE_LEVER].items() if k != "*"})   # a lever WITHOUT an any-capability row: case (iii)
    net75 = SAFE.SafeNet(RF.SAFE_LEVER, refused=RowpairRefused)
    with pytest.raises(RowpairRefused, match=r"no cells row and no safe settings for cc 7\.5"):
        RF.resolve_cells(net75, "7.5", None, None, None, {})
    assert not net75.on


def test_launch_build_failure_serves_safe_then_refuses(capsys, monkeypatch):
    """A triton BUILD failure of the tuned settings switches the process to the safe settings (one line, census ``settings=safe:build_failed:…``)
    and redoes the launch; the safe settings failing to build too is the lever's refusal; a capability without safe settings likewise."""
    g = torch.Generator().manual_seed(4)
    w = _weights(g)
    fns = RF.fused_trimul_fns(w, _stock_fns(w))

    class K:                                                                # resolve_cfg of the kernels module: (k1, k3) of a row
        @staticmethod
        def resolve_cfg(cfg, C_z, C_h, has_bias):
            return dict(cfg["k1"]), dict(cfg["k3"])

    pack = {"has_bias": False}
    tuned = {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 3}
    seen = []

    def launch(c):
        seen.append(dict(c))
        if c.get("num_stages") == 3:
            raise RuntimeError("PassManager::run failed (simulated triton build failure)")

    fns._launch("k1", launch, tuned, "8.0", K, pack)
    from opt_core.kernels import safe_settings as SAFE
    safe_k1 = SAFE.SAFE_ROWS[RF.SAFE_LEVER]["8.0"]["settings"]["k1"]
    assert seen == [tuned, safe_k1] and fns.net.on and fns.net.word() == "safe:build_failed:RuntimeError"
    assert fns.facts["settings"] == "safe:build_failed:RuntimeError" and fns.facts["cells"] == "safe" and fns.facts["k1"] == RF._cell_key(safe_k1)
    assert "safe settings served (build_failed:RuntimeError, cc 8.0" in capsys.readouterr().err
    fns._launch("k1", launch, tuned, "8.0", K, pack)                        # from here on the safe settings launch directly
    assert seen[-1] == safe_k1

    def launch_bad(c):
        raise RuntimeError("PassManager::run failed (safe too)")
    with pytest.raises(RowpairRefused, match=r"safe settings cannot build/run either"):
        fns._launch("k1", launch_bad, tuned, "8.0", K, pack)

    from opt_core.kernels import safe_settings as _S
    monkeypatch.setitem(_S.SAFE_ROWS, RF.SAFE_LEVER, {k: v for k, v in _S.SAFE_ROWS[RF.SAFE_LEVER].items() if k != "*"})
    other = RF.fused_trimul_fns(w, _stock_fns(w))                           # a lever without safe settings on the capability: the tuned settings failing is the refusal
    with pytest.raises(RowpairRefused, match=r"no safe settings for cc 7\.5"):
        other._launch("k1", launch, tuned, "7.5", K, pack)

    def launch_oom(c):
        raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
    with pytest.raises(RuntimeError, match="out of memory"):                # not a build failure: propagates untouched, no safe settings
        RF.fused_trimul_fns(w, _stock_fns(w))._launch("k1", launch_oom, tuned, "8.0", K, pack)
    assert RF.LEDGER.served == 0 and RF.LEDGER.state == "skipped"
    line = RF.LEDGER.line("t")
    assert "name=F2.trimul_rows" in line and "state=skipped" in line and "origin=core" in line and "fallback_by=" in line, line


def test_fused_provider_weight_vocabulary():
    g = torch.Generator().manual_seed(1)
    w = _weights(g)
    stock = _stock_fns(w)
    fns = RF.fused_trimul_fns(w, stock)
    assert fns.C_h == C_H and fns.C_z == C
    assert fns.operand_dtype(torch.zeros(2, N, C)) is None                   # declined on CPU -> the statements' dtype
    assert fns.min_tokens == RF.DEFAULT_MIN_TOKENS == 2048
    assert set(CORE_TRIMUL.WEIGHT_KEYS) <= set(w)
    bad = dict(w)
    bad.pop("w_og")
    with pytest.raises(ValueError, match="weights_missing=w_og"):
        RF.fused_trimul_fns(bad, stock)
    worse = dict(w, w_extra=w["w_o"])
    with pytest.raises(ValueError, match="weights_unknown=w_extra"):
        RF.fused_trimul_fns(worse, stock)


# ----------------------------------------------------------------------------------------------------------------- attention cores
H, D, S = 2, 8, 24


def _eager_core(q, k, v, biases):
    """An engine's eager attention core (OpenFold statement): softmax(q k^T * scale + sum(biases)) v."""
    a = torch.einsum("...qd,...kd->...qk", q * (D ** -0.5), k)
    for b in biases:
        a = a + b
    return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v)


def _inputs(rows=5, g=None):
    g = g or torch.Generator().manual_seed(3)
    q, k, v = (torch.randn(1, rows, H, S, D, generator=g) for _ in range(3))
    keep = torch.rand(1, rows, 1, 1, S, generator=g) > 0.2
    mask_bias = torch.zeros(1, rows, 1, 1, S).masked_fill(~keep, float("-inf"))
    tb = torch.randn(1, 1, H, S, S, generator=g)
    return q, k, v, [mask_bias, tb]


class _Calls(object):
    def __init__(self):
        self.n = 0

    def __call__(self, q, k, v, biases):
        self.n += 1
        return _eager_core(q, k, v, biases)


def test_attention_core_torch_word_equals_stock_and_counts():
    q, k, v, biases = _inputs()
    stock = _Calls()
    L = RA.core_ledger()
    L.clear()
    core = RA.attention_core(stock, kernel="torch")
    assert core.kernel == "torch"
    for qblock in (None, 4, 8, 24):
        o = RA.attend_query_blocks(core, q, k, v, biases, qblock)
        assert torch.equal(o, RA.attend_query_blocks(_eager_core, q, k, v, biases, qblock))
    assert stock.n > 0 and L.served == 0 and L.fallbacks.get("kernel_torch", 0) == stock.n
    line = L.line("t")
    assert "name=F1.flash_triattn" in line and "state=skipped" in line and "fallback_by=kernel_torch:" in line and "kernel=torch" in line and "rowpair=1" in line, line


def test_attention_core_flash_refuses_where_it_cannot_run():
    """CPU tensors: the flash lever cannot run in this process → ``RowpairRefused`` naming the lever and the opt-out (never the engine core silently)."""
    q, k, v, biases = _inputs(rows=2)
    core = RA.attention_core(_eager_core, kernel="flash_triattn")
    with pytest.raises(RowpairRefused, match=r"F1\.flash_triattn .*not a CUDA device.*ROWPAIR_TRIATT_CORE=torch"):
        core(q, k, v, biases)
    assert torch.equal(RA.attention_core(_eager_core, kernel="torch")(q, k, v, biases), _eager_core(q, k, v, biases))   # the opt-out runs


def test_attention_core_flash_falls_back_by_name_on_cpu(monkeypatch):
    monkeypatch.setattr(RA, "_lever_ready", lambda F1, word, q: None)      # readiness aside (CPU box): the serve layer's own named gates below
    q, k, v, biases = _inputs(rows=6)
    stock = _Calls()
    L = RA.core_ledger()
    L.clear()
    core = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8)
    o = core(q, k, v, biases)
    assert torch.equal(o, _eager_core(q, k, v, biases)), float((o - _eager_core(q, k, v, biases)).abs().max())
    fb = L.fallbacks
    assert L.served == 0 and sum(fb.values()) == 1 and (list(fb)[0].startswith("kernel_unavailable:") or list(fb)[0].startswith("unsupported:")), fb
    # the row-window split (rows*H*S*D >= the int32 bound) and the bias refusal, with a small bound
    L.clear()
    monkeypatch.setattr(RA, "INT32_MAX", H * S * D * 2 + 1)                 # 2 rows per launch -> 3 windows of a 6-row batch; H*S*S = 1152 > the bound 2*8*24*2+1=769: a "big" plane
    monkeypatch.setenv(RA.ENV_TRIATT_INT32_GUARD, "1")                      # the <= 0.5.18.7 refusal restored by name (read at bind)
    core_g = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8)
    o = core_g(q, k, v, biases)
    assert torch.equal(o, _eager_core(q, k, v, biases))
    assert L.fallbacks == {"unsupported:bias_elems>int32": 1}, L.fallbacks
    L.clear()
    monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)           # default (0.5.18.8): the big plane goes to the flash kernel per query block — on this CPU box the serve
    core_d = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8)   #  layer's own gate names the fallback per row window (3 windows), never `bias_elems>int32`
    o = core_d(q, k, v, biases)
    assert torch.equal(o, _eager_core(q, k, v, biases))
    fb = L.fallbacks
    assert L.served == 0 and sum(fb.values()) == 3 and "unsupported:bias_elems>int32" not in fb, fb
    assert all(r.startswith("kernel_unavailable:") or r.startswith("unsupported:") for r in fb), fb
    L.clear()
    monkeypatch.setattr(RA, "INT32_MAX", H * S * S + 1)                     # bias fits; rows per launch = (H*S*S+1)//(H*S*D) = 3 -> 2 windows
    o = core(q, k, v, biases)
    assert torch.allclose(o, _eager_core(q, k, v, biases), rtol=0, atol=0)
    assert sum(L.fallbacks.values()) == 2, L.fallbacks                       # one named fallback per window
    # engine layout [B, H, rows, S, D]
    L.clear()
    core_h = RA.attention_core(lambda q_, k_, v_, b_: _eager_core(q_, k_, v_, b_), kernel="flash_triattn", layout="bhnsd")
    qh, kh, vh = (t.transpose(1, 2) for t in (q, k, v))
    bh = [biases[0].transpose(1, 2), biases[1].transpose(1, 2)]              # [B, 1, rows, 1, S] key-mask bias, [B, H, 1, S, S] triangle bias
    oh = core_h(qh, kh, vh, bh)
    assert torch.equal(oh, _eager_core(qh, kh, vh, bh))


def test_attention_core_words_and_override(monkeypatch):
    with pytest.raises(RowpairRefused, match="kernel 'bogus'"):
        RA.attention_core(_eager_core, kernel="bogus")
    with pytest.raises(RowpairRefused, match="layout"):
        RA.attention_core(_eager_core, kernel="torch", layout="nope")
    monkeypatch.setenv(RA.ENV_TRIATT_CORE, "torch")
    core = RA.attention_core(_eager_core, kernel="flash_triattn")
    assert core.kernel == "torch"
    from opt_core.mem.rowpair.evidence import schedule
    sch = schedule()
    assert sch.get("triatt_core") == "torch" and sch.get("triatt_core_src") == "env", sch


FP32_TOO = ("bfloat16", "float16", "float32")                              # the fp32 test tensors served (an fp32 line's adapter says so the same way)


class _FakeFlash(object):
    """A stand-in for the carried kernel module (the serve layer reaches it through ``kernel_module()``): ``flash_supported`` says yes on this CPU box and
    ``flash_triangle_attention`` is the materialised statement on the call's tensors (cuEq signature: q/k/v [B, N, H, S_q, D], bias [B, 1, H, S_q, S_k],
    bool key mask [B, N, 1, 1, S_k] True = keep), recording every call's shapes — so the ROUTING of ``attention_core`` above the int32 bias bound is
    tested where the kernel itself cannot run."""
    BuildFailed = ()

    def __init__(self):
        self.calls = []

    def flash_supported(self, q, k, v, bias, mask=None):
        return True, "ok"

    def flash_triangle_attention(self, q, k, v, bias, mask=None, scale=None, **kw):
        self.calls.append(dict(q=tuple(q.shape), k=tuple(k.shape), bias=tuple(bias.shape), mask=None if mask is None else tuple(mask.shape),
                               bias_contig=bool(bias.is_contiguous()), bias_dtype=bias.dtype, bias_strides=tuple(bias.stride()[-2:])))
        a = torch.einsum("...qd,...kd->...qk", q * float(scale), k) + bias.to(q.dtype)
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v)


@pytest.mark.parametrize("lim", ["kernel_strided", "copy"])
@pytest.mark.parametrize("form", ["view", "transposed"])
@pytest.mark.parametrize("qblock", [8, 5, 24, 100])
def test_attention_core_big_plane_routes_to_the_kernel_per_query_block(qblock, form, lim, monkeypatch):
    """0.5.18.8: a plane whose bias has more than INT32_MAX elements (the bound patched small here) is SERVED by the flash kernel per query block of
    ``ROWPAIR_TRIATT_FLASH_QBLOCK`` queries on the caller's bias view (slices ``[1, 1, H, qblock, S]``, k/v/mask whole), ONE serve per row window, no
    ``bias_elems>int32`` event, values == the engine core; ``ROWPAIR_TRIATT_INT32_GUARD=1`` refuses it by name instead (the kernel never called)."""
    from opt_core.kernels import flash_triattn_serve as F1
    fake = _FakeFlash()
    monkeypatch.setattr(F1, "_KMOD", fake)
    monkeypatch.setattr(RA, "_lever_ready", lambda F1_, word, q: None)
    monkeypatch.setattr(RA, "INT32_MAX", H * S * D * 2 + 1)
    if lim == "copy":
        monkeypatch.setattr(F1, "INPLANE_INT32_LIM", 8)                     # every block's strided view "overflows": the copy branch
                 # = 769: 2 rows per launch -> windows (0,2),(2,4),(4,6) of a 6-row batch; H*S*S = 1152 > 769 -> "big"
    monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, str(qblock))
    monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)
    q, k, v, biases = _inputs(rows=6)
    tb_cl = biases[1][0, 0].permute(1, 2, 0).contiguous()                 # the channel-last [S, S, H] gathered bias of the tp line
    if form == "view":                                                      # the TP kits' starting node (and one kit's ending node): movedim(-1, 0) — a NON-contiguous view
        tb_view = tb_cl.movedim(-1, 0)[None, None]
        assert torch.equal(tb_view, biases[1])
    else:                                                                   # a TP kit's ENDING node (transpose_bias=True): the two token dims swapped — a transposed view
        tb_view = tb_cl.movedim(-1, 0).transpose(-1, -2)[None, None]
    assert not tb_view.is_contiguous()
    biases = [biases[0], tb_view]
    view_strides = tuple(tb_view.stride()[-2:])                             # a block handed AS IS keeps the caller's in-plane strides (no copy)
    stock = _Calls()
    L = RA.core_ledger()
    L.clear()
    core = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8, serve_dtypes=FP32_TOO)
    o = core(q, k, v, biases)
    nb = len(F1.query_blocks(S, qblock))
    assert stock.n == 0 and L.served == 3 and not L.fallbacks, (stock.n, L.served, L.fallbacks)
    assert len(fake.calls) == 3 * nb, (len(fake.calls), nb)
    for c in fake.calls:
        rows_w, qb = c["q"][1], c["q"][3]
        assert c["q"] == (1, rows_w, H, qb, D) and c["k"] == (1, rows_w, H, S, D) and c["bias"] == (1, 1, H, qb, S) and c["mask"] == (1, rows_w, 1, 1, S), c
        assert qb <= max(qblock, 0) or nb == 1, c
        if lim == "copy":                                                    # the in-plane int32 source bound patched below this block: the small fp32 CONTIGUOUS copy branch
            assert c["bias_dtype"] == torch.float32 and c["bias_contig"]
        else:                                                                # the block's strided view handed AS IS (its in-plane offsets fit int32): no copy, the caller's strides
            assert c["bias_dtype"] == torch.float32 and c["bias_strides"] == view_strides
    assert sorted({c["q"][1] for c in fake.calls}) == [2]
    ref_blocks = RA.attend_query_blocks(_eager_core, q, k, v, [biases[0], biases[1]], None)
    assert torch.allclose(o, ref_blocks, rtol=0, atol=1e-5), float((o - ref_blocks).abs().max())
    from opt_core.mem.rowpair.evidence import schedule
    assert schedule().get("triatt_flash_qblock") == qblock, schedule()
    # the opt-out: refused by name, the kernel never reached
    fake.calls.clear(); L.clear()
    monkeypatch.setenv(RA.ENV_TRIATT_INT32_GUARD, "1")
    core_g = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8, serve_dtypes=FP32_TOO)
    og = core_g(q, k, v, biases)
    assert not fake.calls and stock.n > 0 and L.served == 0 and L.fallbacks == {"unsupported:bias_elems>int32": 1}, (fake.calls, stock.n, L.fallbacks)
    assert torch.equal(og, RA.attend_query_blocks(_eager_core, q, k, v, biases, 8))
    # below the bound in the same process: ONE kernel call per row window over all queries on the whole-plane fp32 contiguous bias (the 0.5.18.7 path)
    fake.calls.clear(); L.clear()
    monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)
    monkeypatch.setattr(RA, "INT32_MAX", 2 ** 31 - 1)
    core_s = RA.attention_core(stock, kernel="flash_triattn", stock_qblock=8, serve_dtypes=FP32_TOO)
    os_ = core_s(q, k, v, biases)
    assert len(fake.calls) == 1 and fake.calls[0]["bias"] == (1, 1, H, S, S) and fake.calls[0]["q"] == (1, 6, H, S, D) and fake.calls[0]["bias_contig"], fake.calls
    assert torch.allclose(os_, o, rtol=0, atol=1e-5)


def test_flash_qblock_env_words(monkeypatch):
    monkeypatch.delenv(RA.ENV_TRIATT_FLASH_QBLOCK, raising=False)
    assert RA.flash_qblock() == RA.FLASH_QBLOCK_DEFAULT == 2048
    monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, "512")
    assert RA.flash_qblock() == 512
    for bad in ("0", "-4", "abc"):
        monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, bad)
        with pytest.raises(RowpairRefused, match="ROWPAIR_TRIATT_FLASH_QBLOCK"):
            RA.flash_qblock()
    monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)
    assert RA.int32_guard() is False
    for on in ("1", "yes"):
        monkeypatch.setenv(RA.ENV_TRIATT_INT32_GUARD, on)
        assert RA.int32_guard() is True
    monkeypatch.setenv(RA.ENV_TRIATT_INT32_GUARD, "0")
    assert RA.int32_guard() is False


def test_serve_query_blocks_words():
    from opt_core.kernels import flash_triattn_serve as F1
    assert F1.query_blocks(10, 4) == [(0, 4), (4, 8), (8, 10)]
    assert F1.query_blocks(10, 10) == [(0, 10)] and F1.query_blocks(10, 0) == [(0, 10)] and F1.query_blocks(10, 64) == [(0, 10)]
    assert F1.query_blocks(8, 4) == [(0, 4), (4, 8)]


def test_bias_hnn_memo():
    tb = torch.randn(S, S, H)
    a = RA.bias_hnn(tb, torch.float32)
    assert tuple(a.shape) == (1, 1, H, S, S) and a.is_contiguous()
    assert RA.bias_hnn(tb, torch.float32) is a                                # one copy per gathered-bias tensor
    tb2 = torch.randn(S, S, H)
    assert RA.bias_hnn(tb2, torch.float32) is not a
    assert torch.equal(RA.bias_hnn(tb2)[0, 0].permute(1, 2, 0), tb2)


def test_pointer_row_rule():
    tma = {"k1": {"impl": "tma", "BM": 128, "BN": 32, "num_warps": 4, "num_stages": 3}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}}
    ptr = {"status": "CERTIFIED_OP", "k1": {"BM": 128, "BN": 128, "num_warps": 8, "num_stages": 1}, "k3": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1},
           "overrides": {"_doc": "x", "k1_C128_D128": {"BM": 128, "BN": 32, "num_warps": 4, "num_stages": 1}}}
    table = {"9.0|3.7": dict(tma, status="CERTIFIED_OP"), "9.0|*": ptr}
    cfg, key = RF.pointer_row(table, "9.0", "9.0|3.7", tma)
    assert key == "9.0|*" and cfg["k1"]["BN"] == 128 and "impl" not in cfg["k1"] and list(cfg["overrides"]) == ["k1_C128_D128"]
    cfg, key = RF.pointer_row({"9.0|3.7": tma}, "9.0", "9.0|3.7", tma)                  # no pointer row -> the tma row's numbers
    assert key == "9.0|3.7" and cfg is tma
    cfg, key = RF.pointer_row(dict(table, **{"9.0|*": dict(ptr, status="TESTED_OP")}), "9.0", "9.0|3.7", tma)   # uncertified pointer row: not preferred
    assert key == "9.0|3.7"
    exact_ptr = {"k1": {"BM": 64, "BN": 64, "num_warps": 4, "num_stages": 1}, "k3": tma["k3"]}
    assert RF.pointer_row(table, "9.0", "9.0|3.3", exact_ptr) == (exact_ptr, "9.0|3.3")   # a pointer row resolved exactly stays
    assert RF.pointer_row(table, "9.0", None, None) == (None, None)


# ------------------------------------------------------------------------------- provider present, every unit declined, MIXED dtypes (autocast engines)
def _stock_fns_dtype_sensitive(w):
    """Statements whose arithmetic runs in the INPUT's dtype (an engine's own LayerNorm under autocast): what they return depends on the dtype of
    the tile / block they are handed — the class of engine on which a declined unit must receive exactly the plain path's tensors."""
    def ln(x, wt, b):
        mu = x.mean(-1, keepdim=True)
        xc = x - mu
        var = (xc * xc).mean(-1, keepdim=True)
        return xc / torch.sqrt(var + 1e-5) * wt.to(x.dtype) + b.to(x.dtype)

    def proj(zb, mb, is_a):
        x = ln(zb, w["ln_in_w"], w["ln_in_b"])
        wg, wp = (w["w_ag"], w["w_ap"]) if is_a else (w["w_bg"], w["w_bp"])
        return torch.sigmoid(F.linear(x, wg)) * F.linear(x, wp) * mb

    def out(x):
        return F.linear(ln(x, w["ln_out_w"], w["ln_out_b"]), w["w_o"])

    def gate(zb):
        return torch.sigmoid(F.linear(ln(zb, w["ln_in_w"], w["ln_in_b"]), w["w_og"]))

    return TM.TriMulFns(proj, out, gate, C_H)


def _rank_autocast(rank, P, kind, outgoing, add, B):
    g = torch.Generator().manual_seed(7)
    z = torch.randn(N, N, C, generator=g)                                   # fp32 pair values
    mask = (torch.rand(N, N, generator=g) > 0.15).float()
    w = _weights(g)
    stock = _stock_fns_dtype_sensitive(w)
    if kind == "plain":
        fns = stock
    elif kind == "below_gate":
        fns = RF.fused_trimul_fns(w, stock)                                 # N=40 < 2048: every unit declined (below_gate)
    else:
        os.environ[RF.ENV_KERNELS] = "torch"                                # the opt-out: every unit declined (env_torch)
        fns = RF.fused_trimul_fns(w, stock, min_tokens=0)
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone()
    with torch.autocast("cpu", dtype=torch.bfloat16):                       # the engine's mixed-precision region: GEMMs bf16, z fp32
        TM.trimul_update_(fns, zs, mask[lay.r0:lay.r1].clone(), lay, outgoing=outgoing, add=add, inplace_chunk=8, RB=8)
    return all_gather_rows(zs, lay)


@pytest.mark.parametrize("kind", ["below_gate", "env_torch"])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("add", [True, False])
def test_declined_shard_under_autocast_is_the_plain_path(kind, outgoing, add, monkeypatch):
    monkeypatch.delenv(RF.ENV_KERNELS, raising=False)
    """Condition (1) extended to 'provider PRESENT, every unit declined' with fp32 z under autocast (bf16 GEMMs): the shards are byte-identical to
    the hook-less path's — a declined shard hands the statements exactly the plain path's tiles and operands (no dtype cast on its behalf)."""
    plain = run_ranks(2, _rank_autocast, "plain", outgoing, add, 8)
    try:
        dec = run_ranks(2, _rank_autocast, kind, outgoing, add, 8)
    finally:
        os.environ.pop(RF.ENV_KERNELS, None)
    for r in range(2):
        assert plain[r].dtype == torch.float32
        assert torch.equal(dec[r], plain[r]), float((dec[r] - plain[r]).abs().max())


# ------------------------------------------------------------------------------------------------ import closure of the row-pair modules
ROWPAIR_MODULES = ("opt_core.mem.rowpair.triatt", "opt_core.mem.rowpair.trimul", "opt_core.mem.rowpair.trimul_fused")
OLD_UNIT = "opt_core.kernels.fpf_trimul"                                    # the earlier TriMul unit: its engine-adapter module rides with it


def test_rowpair_modules_do_not_import_the_engine_adapters():
    """Importing the row-pair triangle modules in a fresh interpreter leaves the earlier fpf_trimul unit (its engine-adapter
    module), the single-GPU provider ladder that names kernel units, and the Triton modules out of ``sys.modules`` — a
    kit's minimal bundle carries the row-pair modules without them."""
    import subprocess
    code = ("import importlib, json, sys\n"
            "for m in %r: importlib.import_module(m)\n"
            "print(json.dumps(sorted(m for m in sys.modules if m.startswith('opt_core'))))" % (ROWPAIR_MODULES,))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=os.path.dirname(HERE),
                       env={**os.environ, "PYTHONPATH": os.path.dirname(HERE)})
    assert r.returncode == 0, r.stderr[-2000:]
    import json as _json
    loaded = set(_json.loads(r.stdout.strip().splitlines()[-1]))
    assert set(ROWPAIR_MODULES) <= loaded, loaded
    leaked = [m for m in loaded if m == OLD_UNIT or m.startswith(OLD_UNIT + ".") or m in ("opt_core.trimul", "opt_core.kernels.fpf_trimul_v4.kernels")]
    assert not leaked, leaked                                                # the Triton modules load at the first SERVED call, never at import


def test_rowpair_modules_name_no_kernel_unit_by_top_level_name():
    """Static form of the same property (what a bundler's import-closure scan sees): the three modules import no ``opt_core.trimul`` /
    ``opt_core.kernels.fpf_trimul`` module and carry no string literal naming a carried kernel unit by its top-level name (``<unit>`` or
    ``<unit>.sub[:attr]``) — such a name pulls the whole unit, engine adapters included, into every row-pair kit's closure."""
    import ast
    import re as _re
    core = os.path.join(os.path.dirname(HERE), "opt_core")
    units = sorted(d for d in os.listdir(os.path.join(core, "kernels")) if os.path.isdir(os.path.join(core, "kernels", d)) and not d.startswith("_"))
    assert "fpf_trimul" in units and "fpf_trimul_v4" in units, units
    for mod in ROWPAIR_MODULES:
        path = os.path.join(core, *mod.split(".")[1:]) + ".py"
        tree = ast.parse(open(path, encoding="utf-8").read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:                                              # relative: resolve against opt_core.mem.rowpair
                    parts = mod.split(".")[:-node.level] if node.level > 0 else mod.split(".")
                    base = ".".join(parts + ([node.module] if node.module else []))
                names = [base] + [base + "." + a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                s_ = node.value.strip()
                assert s_ not in units, (mod, node.lineno, s_)
                assert not (_re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?::[A-Za-z_]\w*)?", s_) and s_.split(".")[0] in units), (mod, node.lineno, s_)
                continue
            else:
                continue
            for n in names:
                assert n != "opt_core.trimul" and not n.startswith("opt_core.trimul."), (mod, node.lineno, n)
                assert not n.startswith("opt_core.kernels.fpf_trimul.") and n != "opt_core.kernels.fpf_trimul", (mod, node.lineno, n)
                assert n.split(".")[0] not in units, (mod, node.lineno, n)   # `import fpf_trimul_v4` by routed top-level name


def test_weight_vocabulary_is_one():
    from opt_core import trimul as P1, trimul_weights as V
    assert P1.WEIGHT_KEYS is V.WEIGHT_KEYS and P1.BIAS_KEYS is V.BIAS_KEYS
    assert RF.KERNEL == "fpf_trimul_v4"                                     # the census word impl=<unit>@<version> is the unit's own name


class _FakeFlash32(_FakeFlash):
    """As :class:`_FakeFlash` but with the real kernel's numerics class: fp32 accumulation of bf16 / fp16 operands, one rounding of the output."""
    def flash_triangle_attention(self, q, k, v, bias, mask=None, scale=None, **kw):
        self.calls.append(dict(q=tuple(q.shape), bias=tuple(bias.shape), bias_dtype=bias.dtype))
        a = torch.einsum("...qd,...kd->...qk", q.float() * float(scale), k.float()) + bias.float()
        if mask is not None:
            a = a.masked_fill(~mask, float("-inf"))
        return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v.float()).to(q.dtype)


def _ref32(q, k, v, biases):
    """The statement in fp32 on the call's (possibly bf16) tensors: the reference every path is measured against."""
    return _eager_core(q.float(), k.float(), v.float(), [b.float() for b in biases])


def test_attention_core_tier_word_above_the_int32_bound_takes_the_flash_path_by_name(monkeypatch):
    """The tier door (``kernel="tier:<word>"``) on a pair plane above the int32 bias bound (patched small here): the flash_triattn path serves
    it per query block BY NAME (schedule word ``triatt_core_tier_big``) -- never the engine statement (0.5.181 and before fell to the statement
    there: `unsupported:bias_elems>int32`, the numerics class of the engine's eager arithmetic instead of the kernels'), and no provider row is
    asked (none stages a whole-plane fp32 bias); numerics = the flash kernel's class (fp32 accumulation: maxerr <= 2e-3 in bf16 against the
    fp32 statement); ``ROWPAIR_TRIATT_INT32_GUARD=1`` still refuses by name to the statement (the explicit <= 0.5.18.7 opt-in); below the
    bound the provider row serves as before."""
    from opt_core.kernels import flash_triattn_serve as F1
    import opt_core.attn.pair_fused as PF
    from opt_core.mem.rowpair.evidence import schedule
    fake = _FakeFlash32()
    monkeypatch.setattr(F1, "_KMOD", fake)
    monkeypatch.setattr(RA, "_lever_ready", lambda F1_, word, q: None)
    monkeypatch.setattr(RA, "INT32_MAX", H * S * D * 2 + 1)                 # H*S*S = 1152 > 769: "big"; 2 rows per launch -> 3 windows of a 6-row batch
    monkeypatch.setenv(RA.ENV_TRIATT_FLASH_QBLOCK, "8")
    monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)
    pf_calls = []

    def provider(q, k, v, bias, mask5=None, *, core=None, scale=None):   # kernels.triattn's row through the pair block: records, serves the fp32 statement
        pf_calls.append((tuple(q.shape), bias.dtype))
        a = torch.einsum("...qd,...kd->...qk", q.float() * float(scale), k.float()) + bias.float()
        if mask5 is not None:
            a = a.masked_fill(~mask5, float("-inf"))
        return torch.einsum("...qk,...kd->...qd", torch.softmax(a, dim=-1), v.float()).to(q.dtype)
    monkeypatch.setattr(PF, "core_attention", provider)
    nb = len(F1.query_blocks(S, 8))
    for dt, tol in ((torch.float32, 1e-5), (torch.bfloat16, 2e-3)):
        g = torch.Generator().manual_seed(11)
        q, k, v = (torch.randn(1, 6, H, S, D, generator=g).to(dt) for _ in range(3))
        v = (v.float() * 0.2).to(dt)                                          # |o| < ~0.5: one bf16 rounding of the output is below 2e-3
        keep = torch.rand(1, 6, 1, 1, S, generator=g) > 0.2
        mask_bias = torch.zeros(1, 6, 1, 1, S).masked_fill(~keep, float("-inf")).to(dt)
        tb = torch.randn(1, 1, H, S, S, generator=g).to(dt)
        biases = [mask_bias, tb]
        ref = _ref32(q, k, v, biases)
        stock = _Calls()
        L = RA.core_ledger(); L.clear(); fake.calls.clear(); del pf_calls[:]
        core = RA.attention_core(stock, kernel="tier:big", stock_qblock=8, serve_dtypes=FP32_TOO)
        o = core(q, k, v, biases)
        assert stock.n == 0 and not pf_calls and L.served == 3 and "unsupported:bias_elems>int32" not in L.fallbacks and not L.fallbacks, (dt, stock.n, pf_calls, L.served, L.fallbacks)
        assert len(fake.calls) == 3 * nb and all(c["bias"] == (1, 1, H, c["q"][3], S) for c in fake.calls), fake.calls[:2]     # per query block on the caller's bias view
        assert all(c["bias_dtype"] == dt for c in fake.calls) or dt == torch.float32, {c["bias_dtype"] for c in fake.calls}       # no whole-plane fp32 staging of a bf16 bias
        err = float((o.float() - ref).abs().max())
        assert o.dtype == dt and err <= tol, (dt, err)
        assert schedule().get("triatt_core_tier_big") == "flash_triattn:bias_elems>int32", schedule()
        # the explicit opt-in keeps its by-name refusal to the statement
        monkeypatch.setenv(RA.ENV_TRIATT_INT32_GUARD, "1"); L.clear(); fake.calls.clear()
        og = RA.attention_core(stock, kernel="tier:big", stock_qblock=8, serve_dtypes=FP32_TOO)(q, k, v, biases)
        assert not fake.calls and not pf_calls and stock.n > 0 and L.fallbacks == {"unsupported:bias_elems>int32": 1}, (fake.calls, pf_calls, stock.n, L.fallbacks)
        assert torch.equal(og, RA.attend_query_blocks(_eager_core, q, k, v, biases, 8))
        monkeypatch.delenv(RA.ENV_TRIATT_INT32_GUARD, raising=False)
    # below the bound in the same process: the provider row serves every row window (the door unchanged there)
    monkeypatch.setattr(RA, "INT32_MAX", 2 ** 31 - 1)
    g = torch.Generator().manual_seed(12)
    q, k, v = (torch.randn(1, 6, H, S, D, generator=g).bfloat16() for _ in range(3))
    v = (v.float() * 0.2).bfloat16()
    biases = [torch.zeros(1, 6, 1, 1, S).bfloat16(), torch.randn(1, 1, H, S, S, generator=g).bfloat16()]
    stock = _Calls(); L = RA.core_ledger(); L.clear(); fake.calls.clear(); del pf_calls[:]
    o = RA.attention_core(stock, kernel="tier:big", stock_qblock=8, serve_dtypes=FP32_TOO)(q, k, v, biases)
    assert pf_calls and not fake.calls and stock.n == 0 and L.served == 1, (pf_calls, fake.calls, stock.n, L.served)
    assert float((o.float() - _ref32(q, k, v, biases)).abs().max()) <= 2e-3
