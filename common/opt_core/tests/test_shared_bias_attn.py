"""opt_core.attn.shared_bias_attn: the tested-cell gate refuses BY NAME before any launch; a tested cell reaches the kernel call."""
import pytest

torch = pytest.importorskip("torch")
from opt_core.attn import shared_bias_attn as SBA


class _Launched(Exception):
    pass


class _FakeKernel:
    calls = 0

    @staticmethod
    def flash_triangle_attention(q, k, v, bias, **kw):
        _FakeKernel.calls += 1
        raise _Launched("kernel reached")


def _inputs(S=2, H=4, L=32, Lk=128, D=32, G=3):
    q = torch.zeros(G, S, H, L, D); k = torch.zeros(G, S, H, Lk, D); v = torch.zeros(G, S, H, Lk, D); b = torch.zeros(G, 1, H, L, Lk)
    return q, k, v, b


@pytest.fixture
def fake(monkeypatch):
    """A fake kernel that records launches; the device check is neutralised so a CPU box can prove which refusals precede the launch."""
    _FakeKernel.calls = 0
    monkeypatch.setattr(SBA, "kernel", lambda: _FakeKernel)
    monkeypatch.setattr(SBA, "triton2_dot_shim", lambda: "t2shim")
    if not torch.zeros(1).is_cuda:
        try:
            monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
        except Exception as e:                      # pragma: no cover — a torch whose Tensor class refuses the attribute
            pytest.skip(f"cannot neutralise the device check on this torch: {e}")
    return _FakeKernel


def test_uncertified_atom_cell_on_triton2_is_refused_without_launch(fake, monkeypatch):
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    q, k, v, b = _inputs(D=32)                                   # (2, float32, 32): the triton-2.3.1 SIGABRT cell — absent from the table
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, k, v, b, scale=1.0, input_precision="tf32")
    assert e.value.event.startswith("cell_uncertified:t2:float32:d32"), e.value.event
    assert fake.calls == 0


@pytest.mark.parametrize("kw, suffix", [(dict(input_precision="tf32x3"), "ip_tf32x3"),
                                        (dict(config=dict(BLOCK_M=64, BLOCK_N=64, ROWS=2, num_warps=4, num_stages=2, ORDER=0)), "config")])
def test_uncertified_precision_or_config_refused_without_launch(fake, monkeypatch, kw, suffix):
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    q, k, v, b = _inputs(D=48, L=216, Lk=216, G=1)              # the token cell (pads to 64) but an untested precision / launch configuration
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, k, v, b, scale=1.0, **kw)
    assert e.value.event == "cell_uncertified:t2:float32:d64:" + suffix, e.value.event
    assert fake.calls == 0


def test_short_lengths_refused_without_launch(fake, monkeypatch):
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    q, k, v, b = _inputs(D=48, L=100, Lk=100, G=1)
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, k, v, b, scale=1.0)
    assert e.value.event == "cell_uncertified:t2:float32:d64:len_lt_216"
    assert fake.calls == 0


def test_no_triton_is_cell_t0_refused(fake, monkeypatch):
    monkeypatch.setattr(SBA, "triton_major", lambda: None)
    q, k, v, b = _inputs(D=48, L=216, Lk=216, G=1)
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, k, v, b, scale=1.0)
    assert e.value.event.startswith("cell_uncertified:t0:float32:d64")
    assert fake.calls == 0


def test_certified_token_cell_reaches_the_launch(fake, monkeypatch):
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    q, k, v, b = _inputs(D=48, L=216, Lk=216, G=1)              # (2, float32, 64 after pad), tf32, table config, L >= 216: tested
    with pytest.raises(SBA.Refused) as e:                        # the fake kernel raises -> worded launch:_Launched (the launch WAS reached)
        SBA.attention(q, k, v, b, scale=1.0, input_precision="tf32")
    assert e.value.event == "launch:_Launched", e.value.event
    assert fake.calls == 1


def test_probe_switch_reaches_the_launch(fake, monkeypatch):
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    q, k, v, b = _inputs(D=32)
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, k, v, b, scale=1.0, allow_uncertified=True)
    assert e.value.event == "launch:_Launched" and fake.calls == 1


def test_shared_axis_forms(fake, monkeypatch):
    """rank-4 inputs, an expand()ed (stride-0) sample axis on the bias, and a real per-sample bias (refused unless per_sample='rows1')."""
    monkeypatch.setattr(SBA, "triton_major", lambda: 2)
    S, H, L, D = 3, 16, 216, 48
    q = torch.zeros(S, H, L, D); b1 = torch.zeros(1, H, L, L)
    cq, ck, cv, cb = SBA.canonical(q, q, q, b1.expand(S, H, L, L))          # stride-0 sample axis == shared
    assert tuple(cb.shape) == (1, 1, H, L, L)
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, q, q, torch.zeros(S, H, L, L), scale=1.0)           # a materialised per-sample bias shares nothing
    assert e.value.event == "bias_per_sample" and fake.calls == 0
    with pytest.raises(SBA.Refused) as e:
        SBA.attention(q, q, q, torch.zeros(S, H, L, L), scale=1.0, per_sample="rows1")
    assert e.value.event == "launch:_Launched"


def test_table_rows_are_well_formed():
    for key, row in SBA.CERTIFIED_CELLS.items():
        assert isinstance(key[0], int) and key[1] in ("float32", "bfloat16", "float16") and key[2] in SBA.SERVED_HEAD_DIMS
        assert set(row["ips"]) <= set(SBA.IP_WORDS) and None in row["configs"] and row["evidence"] and int(row["min_len"]) >= 1
    assert (2, "float32", 32) not in SBA.CERTIFIED_CELLS                     # the known-fatal triton-2.3.1 cell stays out


def test_cc80_wildcard_row_leads_on_an_sm80_device_for_every_triton(monkeypatch):
    """On a cc 8.0 device the "8.0|*" (float32, 64) row certifies whatever the triton version: word t<major>:float32:d64:cc8.0, tf32 only,
    single-stage configurations only (an H100-measured multi-stage t3 configuration is refused there by name)."""
    monkeypatch.setattr(SBA, "_current_cc", lambda: "8.0")
    for major, mm in ((2, "2.3"), (3, "3.3"), (3, "3.7")):
        SBA._CERT_MEMO.clear()
        monkeypatch.setattr(SBA, "_triton_mm", lambda mm=mm: mm)
        assert SBA.cc_row("8.0", mm, "float32", 64)[0] == "8.0|*"
        assert SBA.certify(major, "float32", 64, "tf32", 216, 216, None) == "t%d:float32:d64:cc8.0" % major
        assert SBA.certify(major, "float32", 64, "tf32", 384, 384, dict(BLOCK_M=64, BLOCK_N=64, ROWS=1, num_warps=4, num_stages=1, ORDER=0)) == "t%d:float32:d64:cc8.0" % major
        for kw, suffix in ((dict(ip="ieee", config=None), "ip_ieee"), (dict(ip="tf32", config=dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=2, ORDER=0)), "config")):
            if major >= 3:                                                            # triton >= 3: SERVED and named, never refused for want of a record
                assert SBA.certify(major, "float32", 64, kw["ip"], 216, 216, kw["config"]) == "t%d:float32:d64:cc8.0:unverified(%s)" % (major, suffix)
                continue
            with pytest.raises(SBA.Refused) as e:                                     # triton 2: refused before launch (untested cells can abort that compiler)
                SBA.certify(major, "float32", 64, kw["ip"], 216, 216, kw["config"])
            assert e.value.event == "cell_uncertified:t%d:float32:d64:cc8.0:%s" % (major, suffix), e.value.event
            assert "SIGABRT" in e.value.reason
    SBA._CERT_MEMO.clear()
    assert SBA.cc_row("8.0", "3.7", "float32", 32) == (None, None)                     # no cc row for that cell: the capability-free table decides
    assert sorted(SBA.CERTIFIED_CELLS_BY_CC) == ["8.0|*"] and all(len(k) == 3 for k in SBA.CERTIFIED_CELLS)


def test_cc90_or_no_device_keeps_the_capability_free_rows(monkeypatch):
    monkeypatch.setattr(SBA, "_triton_mm", lambda: "3.3")
    for cc in ("9.0", None):
        SBA._CERT_MEMO.clear()
        monkeypatch.setattr(SBA, "_current_cc", lambda cc=cc: cc)
        assert SBA.cc_row(cc, "3.3", "float32", 64) == (None, None)
        assert SBA.certify(2, "float32", 64, "ieee", 216, 216, dict(BLOCK_M=64, BLOCK_N=32, ROWS=2, num_warps=4, num_stages=2, ORDER=0)) == "t2:float32:d64"
        assert SBA.certify(3, "float32", 32, "tf32", 32, 128, None) == "t3:float32:d32"
    SBA._CERT_MEMO.clear()


def test_triton3_uncertified_combinations_are_served_and_named(monkeypatch):
    """P12: on triton >= 3 a (dtype, D) cell / precision / length / launch configuration without a record is SERVED; certify returns the cell word
    with `:unverified(<what>)` (the event's suffix); on triton 2 / no triton the same combinations are refused by name."""
    monkeypatch.setattr(SBA, "_triton_mm", lambda: "3.7")
    monkeypatch.setattr(SBA, "_current_cc", lambda: "9.0")
    SBA._CERT_MEMO.clear()
    assert SBA.certify(3, "bfloat16", 64, "tf32", 384, 384, None) == "t3:bfloat16:d64:unverified(cell)"          # no bf16 cell recorded
    assert SBA.certify(3, "float32", 64, "tf32", 100, 100, None) == "t3:float32:d64:unverified(len_lt_216)"     # below the measured lengths
    assert SBA.certify(3, "float32", 32, "tf32x3", 32, 128, None) == "t3:float32:d32:unverified(ip_tf32x3)"
    assert SBA.certify(3, "float32", 32, "tf32", 32, 128, dict(BLOCK_M=16, BLOCK_N=16, ROWS=1, num_warps=2, num_stages=1, ORDER=0)) == "t3:float32:d32:unverified(config)"
    assert SBA.certify(3, "float32", 32, "tf32", 32, 128, None) == "t3:float32:d32"                             # the recorded cell: unchanged
    assert SBA.certify(3, "bfloat16", 64, "tf32", 384, 384, None) == "t3:bfloat16:d64:unverified(cell)"          # memoised: the same word
    for major in (2, 0):
        with pytest.raises(SBA.Refused) as e:
            SBA.certify(major, "bfloat16", 64, "tf32", 384, 384, None)
        assert e.value.event == "cell_uncertified:t%d:bfloat16:d64" % major
    SBA._CERT_MEMO.clear()

