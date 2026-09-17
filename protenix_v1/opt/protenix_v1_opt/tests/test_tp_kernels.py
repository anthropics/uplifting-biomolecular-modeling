"""The xP line's triangle kernels (tp.TP_TRIMUL_KERNELS / tp.TP_TRIATT_CORE): ON = the core's fused row kernels constructed with the module's
weight vocabulary and today's torch fns / stock attention core as their named fallback; forced to the torch words = today's statements with
nothing imported and nothing printed; a selected kernel the installed core lacks = a NAMED fallback (NOTE + census word); and the attention-core
adapter reproduces the stock module's forward bit for bit whenever the core hands the call to the stock statement (which it does on the CPU).
CPU; the protenix parts skip when the stock wheel is not importable."""
import os
import sys

import pytest

os.environ.setdefault("LAYERNORM_TYPE", "torch")
torch = pytest.importorskip("torch")
from protenix_v1_opt import report as R, tp as TP                     # noqa: E402
from opt_core import trimul as OT                                      # noqa: E402
from opt_core.mem.rowpair import triatt as RA, trimul_fused as RF      # noqa: E402

try:
    from protenix.model.triangular import triangular as TRI            # noqa: E402
    HAVE_PTX = True
except Exception:                                                      # a CPU box without the stock wheel: the module-level tests below skip by name
    HAVE_PTX = False
needs_ptx = pytest.mark.skipif(not HAVE_PTX, reason="protenix (the pinned stock wheel) is not importable in this interpreter")


@pytest.fixture
def fresh(monkeypatch):
    """Every test resolves the kernels afresh (tp caches the resolution per process)."""
    monkeypatch.setattr(TP, "_TPX", {})
    TP._NOTED.discard("tp_kernels")
    yield
    TP._TPX.clear(); TP._NOTED.discard("tp_kernels")


def _randomize(mod, seed=0, scale=0.3):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in mod.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * scale)
    return mod.eval()


def test_the_xP_line_selects_the_fused_row_kernels(fresh):
    assert TP.TP_TRIMUL_KERNELS == "fpf_v4" and TP.TP_TRIATT_CORE == "tier:big"
    tk = TP.tp_kernels()
    assert tk["RF"] is RF and tk["RA_core"] is RA.attention_core and tk["reason"] is None          # the installed core carries both
    assert tk["trimul_word"] == "rowpair_fpf_v4" and tk["triatt_word"] == "tier:big"


def test_the_torch_words_are_todays_statements_and_import_nothing(fresh, monkeypatch, capsys):
    monkeypatch.setattr(TP, "TP_TRIMUL_KERNELS", "torch"); monkeypatch.setattr(TP, "TP_TRIATT_CORE", None)
    tk = TP.tp_kernels()
    assert tk["RF"] is None and tk["RA_core"] is None and tk["reason"] is None
    assert tk["trimul_word"] == "rowpair_torch" and tk["triatt_word"] is None
    assert TP.emit_tp_kernel_lines() == [] and "NOTE" not in capsys.readouterr().err


@needs_ptx
def test_the_torch_words_give_the_plain_fns_and_the_whole_module(fresh, monkeypatch):
    monkeypatch.setattr(TP, "TP_TRIMUL_KERNELS", "torch"); monkeypatch.setattr(TP, "TP_TRIATT_CORE", None)
    fns = TP.trimul_fns(_randomize(TRI.TriangleMultiplicationOutgoing(c_z=16, c_hidden=16)))
    assert type(fns) is TP.C("TM").TriMulFns                                           # not the fused provider: trimul_update_ runs today's path
    assert TP.triatt_core(_randomize(TRI.TriangleAttention(c_in=16, c_hidden=8, no_heads=2)), "torch") is None


@needs_ptx
def test_selected_kernels_the_core_lacks_are_a_named_fallback(fresh, monkeypatch, capsys):
    import opt_core
    from opt_core.mem import rowpair as RPK
    monkeypatch.delattr(RA, "attention_core"); monkeypatch.delitem(sys.modules, "opt_core.mem.rowpair.trimul_fused"); monkeypatch.delattr(RPK, "trimul_fused", raising=False)
    real_import = __import__

    def no_fused(name, *a, **k):                                                        # a core without the module
        if name == "opt_core.mem.rowpair" and len(a) >= 3 and a[2] and "trimul_fused" in a[2]:
            raise ImportError("no trimul_fused in this core")
        return real_import(name, *a, **k)
    monkeypatch.setattr("builtins.__import__", no_fused)
    tk = TP.tp_kernels()
    assert tk["RF"] is None and tk["RA_core"] is None and tk["reason"] == f"core_{opt_core.__version__}_lacks_row_kernels"
    assert tk["trimul_word"] == "rowpair_torch" and tk["triatt_word"] is None
    err = capsys.readouterr().err
    assert err.count(f"{R.PREFIX} NOTE row-sharded fused kernels not in the installed core") == 1 and "trimul='fpf_v4' triatt_core='tier:big' requested" in err
    TP.tp_kernels(); assert "NOTE" not in capsys.readouterr().err                      # announced once per process
    assert type(TP.trimul_fns(_randomize(TRI.TriangleMultiplicationOutgoing(c_z=16, c_hidden=16)))) is TP.C("TM").TriMulFns
    assert TP.triatt_core(_randomize(TRI.TriangleAttention(c_in=16, c_hidden=8, no_heads=2)), "torch") is None


@needs_ptx
def test_the_fused_trimul_provider_gets_the_weight_vocabulary_and_todays_fns(fresh):
    mod = _randomize(TRI.TriangleMultiplicationOutgoing(c_z=16, c_hidden=16))
    fns = TP.trimul_fns(mod)
    assert isinstance(fns, RF.FusedTriMulFns) and fns.min_tokens == RF.DEFAULT_MIN_TOKENS == 2048   # the core's size gate, its default
    w = TP.trimul_weights(mod)                                                         # the mapping the provider accepted (it validates the vocabulary)
    assert tuple(w) == OT.WEIGHT_KEYS
    assert w["ln_in_w"] is mod.layer_norm_in.weight and w["w_ap"] is mod.linear_a_p.weight and w["w_bg"] is mod.linear_b_g.weight
    assert w["ln_out_b"] is mod.layer_norm_out.bias and w["w_o"] is mod.linear_z.weight and w["w_og"] is mod.linear_g.weight
    assert fns.eps == float(mod.layer_norm_in.eps) and fns.C_z == 16 and fns.C_h == 16
    assert fns.proj.__qualname__.startswith("trimul_fns.") and fns.out.__qualname__.startswith("trimul_fns.") and fns.gate.__qualname__.startswith("trimul_fns.")   # today's torch statements ride along as the fallback


@needs_ptx
@pytest.mark.parametrize("word", ["torch"])
def test_the_attention_core_adapter_reproduces_the_stock_module_bit_for_bit(fresh, monkeypatch, word):
    """When the core hands a call to the stock statement (the `torch` word: by name; on a GPU also the flash core's dtype / shape declines)
    the module's forward around the core must equal the module — the fallback receives exactly the adapter's q, k, v, biases. (On CPU the
    flash word refuses outright: test_the_flash_core_on_cpu_tensors_refuses_by_name.)"""
    monkeypatch.setattr(TP, "TP_TRIATT_CORE", word)
    R_, N_, C_, H_, D_ = 5, 24, 16, 2, 8
    att = _randomize(TRI.TriangleAttention(c_in=C_, c_hidden=D_, no_heads=H_))
    fns = TP.triatt_fns(att, "torch")
    g = torch.Generator().manual_seed(1)
    x = torch.randn(R_, N_, C_, generator=g); mask = (torch.rand(R_, N_, generator=g) < 0.9).float(); tb_full = torch.randn(N_, N_, H_, generator=g)
    got = fns.attend(x, mask, tb_full, None)
    mask_bias = (att.inf * (mask - 1))[..., :, None, None, :]
    want = att.mha(q_x=x, kv_x=x, biases=[mask_bias, TP._tb_operand(tb_full)], triangle_attention="torch")
    assert got.shape == want.shape == (R_, N_, C_) and torch.equal(got, want)
    d = RA.describe_core()
    assert d["core_served"] == 0 and d["core_fallback"] >= 1                          # counted, by reason, on the core's ledger
    assert TP.tp_kernels()["triatt_word"] == word


@needs_ptx
def test_the_shapes_handed_to_the_core(fresh, monkeypatch):
    seen = []

    def attention_core(stock, **kw):
        seen.append(kw)

        def core(q, k, v, biases):
            seen.append((tuple(q.shape), tuple(k.shape), tuple(v.shape), [tuple(b.shape) for b in biases]))
            return stock(q, k, v, biases)
        return core
    monkeypatch.setattr(RA, "attention_core", attention_core)
    R_, N_, C_, H_, D_ = 3, 20, 16, 2, 8
    att = _randomize(TRI.TriangleAttention(c_in=C_, c_hidden=D_, no_heads=H_))
    fns = TP.triatt_fns(att, "torch")
    assert seen[0] == dict(kernel="tier:big", min_tokens=0, ledger=None, scale=None, layout="bnhsd", mask_from="bias0", tri_bias="bias1")
    g = torch.Generator().manual_seed(2)
    fns.attend(torch.randn(R_, N_, C_, generator=g), None, torch.randn(N_, N_, H_, generator=g), None)
    assert seen[1] == ((1, R_, H_, N_, D_),) * 3 + ([(1, R_, 1, 1, N_), (1, 1, H_, N_, N_)],)     # bnhsd with a leading batch of 1; mask bias; triangle bias
    fns.attend(torch.randn(R_, TP.SMALL_Q, C_, generator=g), None, torch.randn(TP.SMALL_Q, TP.SMALL_Q, H_, generator=g), None)
    assert len(seen) == 2                                                               # the module's own rule: Q <= 16 -> the whole module, the core not called


@needs_ptx
def test_log_rowpair_prints_the_core_lines_after_the_rowpair_line(fresh, capsys):
    TP.trimul_fns(_randomize(TRI.TriangleMultiplicationOutgoing(c_z=16, c_hidden=16)))
    TP.triatt_fns(_randomize(TRI.TriangleAttention(c_in=16, c_hidden=8, no_heads=2)), "torch")
    R.log_rowpair()
    lines = [l for l in capsys.readouterr().err.splitlines() if " LEVER " in l]
    assert [l.split(" name=")[1].split()[0] for l in lines] == ["rowpair", "F2.trimul_rows", "F1.flash_triattn"]
    assert all(l.startswith(R.PREFIX + " LEVER ") for l in lines) and "origin=core" in lines[1] and "rowpair=1" in lines[2]


def test_log_rowpair_prints_only_the_rowpair_line_on_the_torch_words(fresh, monkeypatch, capsys):
    monkeypatch.setattr(TP, "TP_TRIMUL_KERNELS", "torch"); monkeypatch.setattr(TP, "TP_TRIATT_CORE", None)
    TP.tp_kernels(); R.log_rowpair()
    lines = [l for l in capsys.readouterr().err.splitlines() if " LEVER " in l]
    assert len(lines) == 1 and " name=rowpair " in lines[0]


@needs_ptx
def test_a_run_kernel_without_a_row_block_core_refuses_by_name(fresh):
    att = _randomize(TRI.TriangleAttention(c_in=16, c_hidden=8, no_heads=2))
    with pytest.raises(TP.C("D").RowpairRefused, match="--triatt_kernel 'deepspeed' has no row-block core"):
        TP.triatt_fns(att, "deepspeed")


@needs_ptx
def test_the_stock_fallback_runs_the_modules_own_shapes(fresh, monkeypatch):
    """A declined unit hands the stock callable the adapter's 5-D tensors; the callable strips the leading batch dim so the module's kernel
    wrapper / eager statement see exactly the 4-D shapes today's whole-module call gives them."""
    from protenix.model.triangular import layers as LY
    seen = []
    real = LY._attention
    monkeypatch.setattr(LY, "_attention", lambda q, k, v, biases: (seen.append((tuple(q.shape), [tuple(b.shape) for b in biases])), real(q, k, v, biases))[1])
    monkeypatch.setenv("ROWPAIR_TRIATT_CORE", "torch")                                  # CPU: the core's engineering word hands every unit to the stock callable
    R_, N_, C_, H_, D_ = 3, 20, 16, 2, 8
    att = _randomize(TRI.TriangleAttention(c_in=C_, c_hidden=D_, no_heads=H_))
    fns = TP.triatt_fns(att, "torch")
    g = torch.Generator().manual_seed(3)
    fns.attend(torch.randn(R_, N_, C_, generator=g), None, torch.randn(N_, N_, H_, generator=g), None)
    assert seen == [((R_, H_, N_, D_), [(R_, 1, 1, N_), (1, H_, N_, N_)])]


@needs_ptx
@pytest.mark.parametrize("core_word", ["torch"])
def test_a_cuequivariance_run_declined_to_stock_returns_the_modules_rank(fresh, monkeypatch, core_word):
    """cuequivariance_ops_torch.triangle_attention prepends singleton dims to a 4-D call and returns 5-D (ensure_dims; o = empty_like(q5)).
    A declined unit of a `--triatt_kernel cuequivariance` run (the core's `torch` word: every unit)
    must still hand triatt_update_ the module's [rows, N, C] — and equal the stock module, which absorbs the kernel's rank the same way."""
    from protenix.model.triangular import layers as LY

    def fake_cueq(q, k, v, bias, mask, scale):                                           # the op's rank behaviour + its statement (scale inside)
        while q.dim() < 5:
            q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        while bias.dim() < 5:
            bias = bias.unsqueeze(0)
        while mask.dim() < 5:
            mask = mask.unsqueeze(0)
        return LY._attention(q * scale, k, v, [torch.where(mask, 0.0, -1e9).to(q.dtype), bias.to(q.dtype)])
    monkeypatch.setattr(LY, "cuequivariance_triangular_attn", fake_cueq)
    monkeypatch.setenv("ROWPAIR_TRIATT_CORE", core_word)
    R_, N_, C_, H_ = 3, 20, 16, 2
    att = _randomize(TRI.TriangleAttention(c_in=C_, c_hidden=8, no_heads=H_))
    g = torch.Generator().manual_seed(5)
    x, tb = torch.randn(R_, N_, C_, generator=g), torch.randn(N_, N_, H_, generator=g)
    out = TP.triatt_fns(att, "cuequivariance").attend(x, None, tb, None)
    ref = att.mha(q_x=x, kv_x=x, biases=[(att.inf * (x.new_ones(R_, N_) - 1))[..., :, None, None, :], TP._tb_operand(tb)], triangle_attention="cuequivariance")
    assert out.shape == (R_, N_, C_) == ref.shape and torch.equal(out, ref)


@needs_ptx
def test_the_flash_core_on_cpu_tensors_refuses_by_name(fresh, monkeypatch):
    """A lever that cannot run at all (q not on a CUDA device) is a RowpairRefused naming the lever and its opt-out — the core's rule; the
    kit's rank process turns it into `ITEM event=failed … error=RowpairRefused:…` and a non-zero EXIT (never a silent plain path)."""
    monkeypatch.setattr(TP, "TP_TRIATT_CORE", "flash_triattn")
    att = _randomize(TRI.TriangleAttention(c_in=16, c_hidden=8, no_heads=2))
    fns = TP.triatt_fns(att, "torch")
    g = torch.Generator().manual_seed(4)
    with pytest.raises(TP.C("D").RowpairRefused, match="F1.flash_triattn .*cannot run in this process"):
        fns.attend(torch.randn(3, 20, 16, generator=g), None, torch.randn(20, 20, 2, generator=g), None)
