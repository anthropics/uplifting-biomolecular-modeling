"""GPU tests of opt_core.attn.shared_bias_attn (run on a CUDA box with Triton: `python -m pytest common/opt_core/tests/gpu/test_shared_bias_attn_gpu.py -q -s`;
skipped by name elsewhere). The tested token cell (H 16, head dim 48 zero-padded to 64, L >= 216, S samples sharing one bias) is SERVED
by the carried flash_triattn kernel and agrees with the materialised float64 reference to the precision class asked for (tf32 / ieee),
run-to-run bit-exact; the AF3 atom-window cell (G windows x S samples, H 4, D 32, Lq 32 / Lk 128) is served on Triton >= 3 and REFUSED BEFORE
LAUNCH (`cell_uncertified:...`) on Triton 2 (where compiling it aborts the process); lengths below the cell's floor are refused before launch.
Every numerics row is printed as one JSON line (`SBA_ROW {...}`) for the box record."""
import json
import math

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
if not torch.cuda.is_available():
    pytest.skip("needs a CUDA device", allow_module_level=True)
from opt_core.attn import shared_bias_attn as SBA  # noqa: E402
from opt_core.attn.sdpa_bias import Refused  # noqa: E402

TM = SBA.triton_major()
if not TM:
    pytest.skip("needs Triton", allow_module_level=True)
DEV = torch.device("cuda")
TOL = {"tf32": 3e-3, "ieee": 3e-5, "tf32x3": 3e-5}         # rel-RMS vs float64 of softmax(QK^T s + b) V: tensor-core tf32 products vs fp32 products


def _token(S, L, H=16, D=48, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(DEV)
    return r(1, S, H, L, D), r(1, S, H, L, D), r(1, S, H, L, D), 0.5 * r(1, 1, H, L, L)


def _atom(G, S, H=4, D=32, Lq=32, Lk=128, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g).to(DEV)
    return r(G, S, H, Lq, D), r(G, S, H, Lk, D), r(G, S, H, Lk, D), 0.5 * r(G, 1, H, Lq, Lk)


def _relrms(a, ref):
    a, ref = a.double(), ref.double()
    return float(((a - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()))


def _row(**kw):
    print("SBA_ROW " + json.dumps(kw), flush=True)


@pytest.mark.parametrize("S,L", [(10, 216), (40, 216), (40, 416), (80, 616)])
@pytest.mark.parametrize("ip", ["tf32", "ieee"])
def test_token_cell_is_served_accurate_and_bitwise_repeatable(S, L, ip):
    q, k, v, bias = _token(S, L)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out, ev = SBA.attention(q, k, v, bias, scale=scale, input_precision=ip)
    assert ev.startswith("served:flash_triattn:" + ip) and "pad48to64" in ev, ev
    assert out.shape == q.shape and out.dtype == q.dtype
    ref = SBA.reference(q, k, v, bias, scale=scale, dtype=torch.float64)
    rr = _relrms(out, ref); mx = float((out.double() - ref).abs().max())
    out2, ev2 = SBA.attention(q, k, v, bias, scale=scale, input_precision=ip)
    bit = bool(torch.equal(out, out2))
    _row(cell="token", triton_major=TM, S=S, L=L, H=16, D=48, ip=ip, event=ev, rel_rms_vs_fp64=rr, max_abs_vs_fp64=mx, repeat_bitwise=bit,
         gpu=torch.cuda.get_device_name(), torch=torch.__version__)
    assert rr < TOL[ip], (ip, rr)
    assert bit, "run-to-run bitwise"
    assert torch.isfinite(out).all()


def test_expanded_sample_axis_is_read_as_shared():
    q, k, v, bias = _token(10, 216)
    scale = 1.0 / math.sqrt(48)
    out_a, ev_a = SBA.attention(q, k, v, bias, scale=scale)
    out_b, ev_b = SBA.attention(q, k, v, bias.expand(1, 10, 16, 216, 216), scale=scale)      # stride-0 sample axis: shared, no copy
    assert ev_a == ev_b and torch.equal(out_a, out_b)


@pytest.mark.parametrize("G,S", [(54, 10), (154, 40)])
def test_atom_window_cell_served_on_triton3_refused_before_launch_on_triton2(G, S):
    q, k, v, bias = _atom(G, S)
    scale = 1.0 / math.sqrt(32)
    if TM < 3:
        with pytest.raises(Refused) as e:
            SBA.attention(q, k, v, bias, scale=scale, input_precision="tf32")
        assert e.value.event.startswith("cell_uncertified:t%d:float32:d32" % TM), e.value.event
        _row(cell="atom", triton_major=TM, G=G, S=S, event=e.value.event, refused_before_launch=True)
        return
    out, ev = SBA.attention(q, k, v, bias, scale=scale, input_precision="tf32")
    assert ev.startswith("served:flash_triattn:tf32") and "pad" not in ev, ev
    ref = SBA.reference(q, k, v, bias, scale=scale, dtype=torch.float64)
    rr = _relrms(out, ref)
    bit = bool(torch.equal(out, SBA.attention(q, k, v, bias, scale=scale, input_precision="tf32")[0]))
    _row(cell="atom", triton_major=TM, G=G, S=S, H=4, D=32, Lq=32, Lk=128, ip="tf32", event=ev, rel_rms_vs_fp64=rr, repeat_bitwise=bit)
    assert rr < TOL["tf32"] and bit


def test_lengths_below_the_cell_floor_are_refused_before_launch_on_triton2_and_served_named_on_triton3():
    q, k, v, bias = _token(10, 100)                      # L 100 < the token cell's floor (216): no measurement record for these lengths
    if TM is not None and TM >= 3:                       # triton >= 3: served and NAMED unverified(len_lt_216) — a compile failure would be the kernel's own named event
        out, ev = SBA.attention(q, k, v, bias, scale=1.0 / math.sqrt(48))
        assert ":unverified(len_lt_" in ev and bool(torch.isfinite(out).all()), ev
        _row(cell="token_short", triton_major=TM, L=100, event=ev, refused_before_launch=False)
        return
    with pytest.raises(Refused) as e:                    # triton 2: refused before launch (an untested cell can abort that compiler)
        SBA.attention(q, k, v, bias, scale=1.0 / math.sqrt(48))
    assert e.value.event.startswith("cell_uncertified:"), e.value.event
    _row(cell="token_short", triton_major=TM, L=100, event=e.value.event, refused_before_launch=True)


def test_a_real_per_sample_bias_is_refused_by_name_or_served_as_rows1():
    q, k, v, bias = _token(4, 216)
    b4 = bias.repeat(1, 4, 1, 1, 1).contiguous()         # a materialised sample axis: nothing shared
    with pytest.raises(Refused) as e:
        SBA.attention(q, k, v, b4, scale=1.0 / math.sqrt(48))
    assert e.value.event == "bias_per_sample", e.value.event
    out, ev = SBA.attention(q, k, v, b4, scale=1.0 / math.sqrt(48), per_sample="rows1")
    assert "rows1" in ev.split("served:flash_triattn:")[1].split(":"), ev      # worded among the event's suffix words (e.g. served:flash_triattn:tf32:rows1:pad48to64[:t2shim])
    out_s, _ = SBA.attention(q, k, v, bias, scale=1.0 / math.sqrt(48))
    assert (out - out_s).abs().max().item() < 1e-3       # the same bias per sample either way (ROWS-per-tile grouping differs -> same class, not bit-exact required)
