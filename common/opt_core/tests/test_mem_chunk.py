"""``opt_core.mem.chunk`` held to the un-chunked statements of the synthetic pair stack (torch.equal where ``bitwise`` is declared), to its
refusals by name, to the registry contract (the five chunk levers registered at import; ``applies`` / ``apply`` / ``undo`` on synthetic
modules through ``opt_core.mem.apply``) and to torch-free import (the lever modules are discovered in every kit's process). Skipped by
name when torch is absent, except the torch-free import test."""
import os
import subprocess
import sys

import pytest

from opt_core.mem import registry as R

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)



def _band_equal(y, ref, what=""):
    """A ``band``-labelled chunked path is held to fp32 rounding of the stock statement (bit-exact on some CPU GEMM stacks, a few ulp on
    others: the row split changes the GEMM shape); ``bitwise``-labelled paths are held with torch.equal elsewhere in this file."""
    if torch.equal(y, ref):
        return
    err = (y - ref).abs().max().item()
    scale = ref.abs().max().item()
    assert err <= 1e-5 * max(scale, 1.0), (what, err, scale)

def test_import_without_torch_registers_the_levers_and_refuses_by_name():
    """A process without torch (a JAX kit) imports the module, sees the five levers registered and gets ``torch_absent`` on use."""
    code = ("import sys; sys.modules['torch'] = None\n"
            "from opt_core.mem import registry, chunk\n"
            "assert set(chunk.LEVERS_HERE) <= set(registry.LEVERS), sorted(registry.LEVERS)\n"
            "assert all(registry.LEVERS[n].family == 'chunk' for n in chunk.LEVERS_HERE)\n"
            "try:\n    chunk.chunk_rows(lambda b: b, object(), 0, 1, exact='bitwise', reason='r')\nexcept chunk.ChunkRefusal as e:\n    assert e.name == 'torch_absent', e.name\nelse:\n    raise SystemExit('no refusal')\n"
            "print('OK')\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=PKG_ROOT, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": PKG_ROOT})
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stderr[-2000:]


torch = pytest.importorskip("torch", reason="the chunked ops need torch; the rest of the core's suite does not")

from opt_core import mem                      # noqa: E402
from opt_core.mem import chunk as C          # noqa: E402
from tests import synthetic_pair as S        # noqa: E402

N, Cz, H, D, CH, BINS, SEQ = 24, 16, 4, 8, 16, 12, 10
CHUNKS = (7, 24, 100)                        # ragged blocks, one block covering the axis, a block larger than the axis


@pytest.fixture(autouse=True)
def _fresh_record():
    C.LOG.clear()
    torch.manual_seed(0)
    yield
    C.LOG.clear()


def pair(batch=2, n=N, c=Cz, dtype=torch.float32):
    x = torch.randn(batch, n, n, c, generator=S.seeded(1)).to(dtype)
    mask = (torch.rand(batch, n, n, generator=S.seeded(2)) > 0.1).to(dtype)
    return x, mask


# ------------------------------------------------------------------------------------------------------------ chunk_rows


def test_chunk_rows_reorders_independent_rows_bitwise():
    x, _ = pair()
    fn = lambda b: torch.tanh(b) * 2 + b.sum(-1, keepdim=True)          # row-local (per position)
    ref = fn(x)
    for chunk in CHUNKS:
        y = C.chunk_rows(fn, x, 1, chunk, exact="bitwise", reason="row-local")
        assert torch.equal(y, ref)
        e = C.LOG[-1]
        assert e["site"] == "chunk_rows" and e["exact"] == "bitwise" and e["n_chunks"] == (4 if chunk == 7 else 1) and e["rows"] == N and e["dim"] == 1 and e["decision"] == "chunked"
    y = C.chunk_rows(fn, x, -2, 5, exact="bitwise", reason="column blocks")
    assert torch.equal(y, ref) and C.LOG[-1]["dim"] == 2 and C.LOG[-1]["n_chunks"] == 5


def test_chunk_rows_one_block_is_the_statement_itself():
    x, _ = pair()
    y = C.chunk_rows(lambda b: b * 3, x, 1, 24, exact="bitwise", reason="one block")
    assert torch.equal(y, x * 3) and C.LOG[-1]["n_chunks"] == 1


def test_chunk_rows_out_and_accumulate():
    x, _ = pair()
    fn = lambda b: b * 0.5
    z = x.clone()
    y = C.chunk_rows(fn, x, 1, 7, exact="bitwise", reason="residual", out=z, accumulate=True)
    assert y is z and torch.equal(z, x + x * 0.5)
    o = torch.empty_like(x)
    assert C.chunk_rows(fn, x, 1, 7, exact="bitwise", reason="into out", out=o) is o and torch.equal(o, x * 0.5)
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(fn, x, 1, 7, exact="bitwise", reason="r", accumulate=True)
    assert e.value.name == "accumulate_without_out"
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(fn, x, 1, 7, exact="bitwise", reason="r", out=torch.empty(2, 23, N, Cz))
    assert e.value.name == "bad_out"


def test_chunk_rows_with_offsets_and_band_label():
    x, _ = pair()
    seen = []

    def fn(b, i0, i1):
        seen.append((i0, i1))
        return b.cumsum(1)                                                 # a reduction along the axis: split by the chunk -> band

    y = C.chunk_rows(fn, x, 1, 10, exact="band", reason="cumsum split across blocks", with_offsets=True)
    assert seen == [(0, 10), (10, 20), (20, 24)] and C.LOG[-1]["exact"] == "band"
    assert not torch.equal(y, x.cumsum(1)) and torch.equal(y[:, :10], x.cumsum(1)[:, :10])


@pytest.mark.parametrize("bad, name", [
    (dict(x="no"), "not_a_tensor"), (dict(fn=3), "not_callable"), (dict(dim=4), "bad_axis"), (dict(chunk=0), "bad_chunk"),
    (dict(chunk=True), "bad_chunk"), (dict(exact="exact"), "bad_exact_label"), (dict(exact="measured"), "bad_exact_label"), (dict(reason=""), "no_reason"),
])
def test_chunk_rows_refuses_by_name(bad, name):
    x, _ = pair()
    kw = dict(fn=lambda b: b, x=x, dim=1, chunk=7, exact="bitwise", reason="r")
    kw.update(bad)
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(kw.pop("fn"), kw.pop("x"), kw.pop("dim"), kw.pop("chunk"), **kw)
    assert e.value.name == name and isinstance(e.value, R.RefusalError) and e.value.refusal.precondition == name and e.value.refusal.lever == "chunk_rows"


def test_chunk_rows_refuses_bad_fn_output():
    x, _ = pair()
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(lambda b: 1, x, 1, 7, exact="bitwise", reason="r")
    assert e.value.name == "fn_output_not_tensor"
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(lambda b: b.sum(1), x, 1, 7, exact="bitwise", reason="r")
    assert e.value.name == "fn_output_shape"
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(lambda b: b[:, :1], x, 1, 7, exact="bitwise", reason="r")
    assert e.value.name == "fn_output_shape"


def test_record_sink_forms():
    """The per-call sink: an object with ``call``, a bare callable, a record with ``note`` only (its note + the standalone log), a bad one refused."""
    x, _ = pair()
    got = []
    C.chunk_rows(lambda b: b, x, 1, 7, exact="bitwise", reason="r", record=lambda lever, site, exact, reason, **d: got.append((lever, site, exact, d["n_chunks"])))
    assert got == [("chunk_rows", "chunk_rows", "bitwise", 4)]

    class FourOnly:                                                        # a sink that takes no details
        def __init__(self):
            self.seen = []

        def call(self, lever, site, exact, reason):
            self.seen.append((lever, site, exact, reason))

    s = FourOnly()
    C.chunk_rows(lambda b: b, x, 1, 7, exact="bitwise", reason="four", record=s)
    assert s.seen == [("chunk_rows", "chunk_rows", "bitwise", "four")]

    class NoteOnly:
        def __init__(self):
            self.notes = []

        def note(self, t):
            self.notes.append(t)

    n = NoteOnly()
    C.chunk_rows(lambda b: b, x, 1, 7, exact="bitwise", reason="noted", record=n)
    assert n.notes == ["chunk_rows chunk_rows exact=bitwise decision=chunked: noted"] and C.LOG[-1]["reason"] == "noted"
    with pytest.raises(C.ChunkRefusal) as e:
        C.chunk_rows(lambda b: b, x, 1, 7, exact="bitwise", reason="r", record=3)
    assert e.value.name == "bad_record"


# ------------------------------------------------------------------------------------------------------------ pair transition


def test_pair_transition_bf16_site_is_chunked_bitwise():
    x, _ = pair(dtype=torch.bfloat16)
    m = S.PairTransition(Cz).to(torch.bfloat16)
    with torch.no_grad():
        ref = m(x)
        for chunk in CHUNKS:
            y = C.pair_transition_chunked(x, m, chunk=chunk)
            assert torch.equal(y, ref), chunk
            e = C.LOG[-1]
            assert e["site"] == "pair_transition" and e["exact"] == "bitwise" and e["decision"] == "chunked" and e["fp32_site"] is False
        z = x.clone()
        C.pair_transition_chunked(x, m, chunk=7, out=z, accumulate=True)
        assert torch.equal(z, x + ref)


def test_pair_transition_fp32_site_named_decisions():
    x, _ = pair()
    m = S.PairTransition(Cz)
    with torch.no_grad():
        ref = m(x)
        y = C.pair_transition_chunked(x, m, chunk=7)                        # default: passthrough, recorded
        assert torch.equal(y, ref)
        e = C.LOG[-1]
        assert e["decision"] == "passthrough" and e["n_chunks"] == 1 and e["exact"] == "bitwise"
        z = x.clone()
        assert C.pair_transition_chunked(x, m, chunk=7, out=z, accumulate=True) is z and torch.equal(z, x + ref)
        with pytest.raises(C.ChunkRefusal) as err:
            C.pair_transition_chunked(x, m, chunk=7, out=torch.empty(2, 23, N, Cz))
        assert err.value.name == "bad_out"
        y = C.pair_transition_chunked(x, m, chunk=7, fp32="band")
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
        assert C.LOG[-1]["decision"] == "chunked" and C.LOG[-1]["exact"] == "band" and C.LOG[-1]["n_chunks"] == 4 and C.LOG[-1]["fp32_site"] is True
        with pytest.raises(C.ChunkRefusal) as err:
            C.pair_transition_chunked(x, m, chunk=7, fp32="refuse")
        assert err.value.name == "fp32_site"
        with pytest.raises(C.ChunkRefusal) as err:
            C.pair_transition_chunked(x, m, chunk=7, fp32="maybe")
        assert err.value.name == "bad_fp32_policy"
        with torch.autocast("cpu", dtype=torch.bfloat16):                  # autocast: the GEMMs run bf16 -> a chunked site, bit-exact
            ref_ac = m(x)
            y = C.pair_transition_chunked(x, m, chunk=7)
        assert torch.equal(y, ref_ac) and C.LOG[-1]["decision"] == "chunked" and C.LOG[-1]["exact"] == "bitwise"
    s = C.summary()
    assert s["chunk_pair_transition"]["passthrough"] == 2 and s["chunk_pair_transition"]["chunked"] == 2 and s["chunk_pair_transition"]["sites"]["pair_transition"] == {"bitwise": 3, "band": 1}
    assert C.active_fragment() == "chunk: chunk_pair_transition=4/2"
    with pytest.raises(C.ChunkRefusal) as err:
        C.pair_transition_chunked(x[0, 0], m, chunk=7)
    assert err.value.name == "not_pair_shaped"


# ------------------------------------------------------------------------------------------------------------ triangle attention


@pytest.mark.parametrize("starting", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
def test_triangle_attention_two_pass_equals_stock(starting, with_mask):
    x, mask = pair()
    mask = mask if with_mask else None
    m = S.TriangleAttention(Cz, D, H, starting)
    with torch.no_grad():
        ref = m(x, mask)
        assert torch.equal(m(x, mask, chunk_size=5), ref)                  # stock's own row-chunked path is the same statement
        for chunk in CHUNKS:
            y = C.triangle_attention_chunked(x, mask, m.parts(), chunk=chunk, starting=starting)
            assert torch.equal(y, ref), (starting, with_mask, chunk)
        sites = [e["site"] for e in C.LOG.entries]
        assert sites[-2:] == ["triangle_attention_bias", "triangle_attention"] and C.LOG[-1]["node"] == ("starting" if starting else "ending")
        z = x.clone()
        C.triangle_attention_chunked(x, mask, m.parts(), chunk=7, starting=starting, out=z, accumulate=True)
        assert torch.equal(z, x + ref)


def test_triangle_attention_refusals():
    x, mask = pair()
    m = S.TriangleAttention(Cz, D, H, True)
    for bad, name in [(dict(mask=mask[..., :5]), "bad_mask"), (dict(parts=m), "bad_parts"), (dict(chunk=-1), "bad_chunk"),
                      (dict(x=x[..., :5, :]), "not_square"), (dict(out=x[..., :1]), "bad_out"), (dict(x=x[0, 0]), "not_pair_shaped")]:
        kw = dict(x=x, mask=mask, parts=m.parts(), chunk=7)
        kw.update(bad)
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_attention_chunked(kw.pop("x"), kw.pop("mask"), kw.pop("parts"), **kw)
        assert e.value.name == name, bad
    assert not hasattr(C.TriAttnParts, "bias_row_dim")


# ------------------------------------------------------------------------------------------------------------ triangle multiplication


@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("with_mask", [True, False])
@pytest.mark.parametrize("arrange", [True, False])
def test_triangle_multiplication_rows_equals_stock(outgoing, with_mask, arrange):
    """fp32 modules: the product is an fp32 site -> the default decision is band (chunked, labelled band): equal to stock within fp32 rounding."""
    x, mask = pair()
    mask = mask if with_mask else None
    m = S.TriangleMultiplication(Cz, CH, outgoing)
    with torch.no_grad():
        ref = m(x, mask)
        for chunk in CHUNKS:
            y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=chunk, arrange=arrange)
            _band_equal(y, ref, (outgoing, with_mask, chunk))
        assert [e["site"] for e in C.LOG.entries[-2:]] == ["triangle_multiplication_operand", "triangle_multiplication"]
        e = C.LOG[-1]
        assert e["mode"] == "rows" and e["exact"] == "band" and e["fp32_site"] is True and e["decision"] == "chunked" and e["arranged"] is arrange
        z = x.clone()
        C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7, out=z, accumulate=True, arrange=arrange)
        _band_equal(z, x + ref, "accumulate")


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_bf16_product_is_bitwise(outgoing):
    x, mask = pair(dtype=torch.bfloat16)
    m = S.TriangleMultiplication(Cz, CH, outgoing).to(torch.bfloat16)
    with torch.no_grad():
        ref = m(x, mask)
        for mode, chunk in (("rows", 7), ("rows", 100), ("channels", 5)):
            y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=chunk, mode=mode, hidden=CH)
            assert torch.equal(y, ref), (outgoing, mode, chunk)
            assert C.LOG[-1]["exact"] == "bitwise" and C.LOG[-1]["fp32_site"] is False


def test_triangle_multiplication_fp32_decisions():
    x, mask = pair()
    m = S.TriangleMultiplication(Cz, CH, True)
    with torch.no_grad():
        ref = m(x, mask)
        y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7, fp32="passthrough")
        assert torch.equal(y, ref) and C.LOG[-1]["decision"] == "passthrough" and C.LOG[-1]["n_chunks"] == 1 and C.LOG[-1]["exact"] == "bitwise"
        z = x.clone()
        assert C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7, fp32="passthrough", out=z, accumulate=True) is z and torch.equal(z, x + ref)
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7, fp32="refuse")
        assert e.value.name == "fp32_site"
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7, fp32="never")
        assert e.value.name == "bad_fp32_policy"
        y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=5, mode="channels", hidden=CH)                # band in channels mode too
        assert torch.equal(y, ref)
        prod = [e for e in C.LOG.entries if e["site"] == "triangle_multiplication_product"]
        assert prod[-1]["exact"] == "band" and prod[-1]["fp32_site"] is True and "wall_s" in prod[-1]
        with torch.autocast("cpu", dtype=torch.bfloat16):                  # autocast: the product runs bf16 -> bit-exact class
            ref_ac = m(x, mask)
            y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=7)
        assert torch.equal(y, ref_ac) and C.LOG[-1]["exact"] == "bitwise" and C.LOG[-1]["fp32_site"] is False
    s = C.summary()["chunk_triangle_multiplication"]
    assert s["passthrough"] == 2 and s["chunked"] > 0


@pytest.mark.parametrize("outgoing", [True, False])
def test_triangle_multiplication_channels_equals_stock(outgoing):
    x, mask = pair()
    m = S.TriangleMultiplication(Cz, CH, outgoing)
    with torch.no_grad():
        ref = m(x, mask)
        for chunk in (5, 16, 40):
            y = C.triangle_multiplication_chunked(x, mask, m.parts(), chunk=chunk, mode="channels", hidden=CH, row_block=7)
            assert torch.equal(y, ref), (outgoing, chunk)
        prod = [e for e in C.LOG.entries if e["site"] == "triangle_multiplication_product"]
        assert prod[0]["n_chunks"] == 4 and prod[0]["mode"] == "channels" and prod[-1]["n_chunks"] == 1 and prod[0]["row_block"] == 7
        parts = m.parts()
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, parts, chunk=5, mode="channels")
        assert e.value.name == "bad_hidden"
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, parts, chunk=5, mode="channels", hidden=CH + 1)
        assert e.value.name == "operand_output_shape"
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, parts, chunk=5, mode="cols")
        assert e.value.name == "bad_mode"
        with pytest.raises(C.ChunkRefusal) as e:
            C.triangle_multiplication_chunked(x, mask, parts, chunk=5, mode="channels", hidden=CH, row_block=0)
        assert e.value.name == "bad_chunk"


def test_arrange_fixed_keeps_the_logical_tensor():
    b = torch.randn(2, 5, 6, 3)
    for outgoing in (True, False):
        a = C._arrange_fixed(b, outgoing)
        assert a.shape == b.shape and torch.equal(a, b) and not a.is_contiguous()
        perm = (0, 3, 2, 1) if outgoing else (0, 3, 1, 2)
        assert a.permute(*perm).is_contiguous()


# ------------------------------------------------------------------------------------------------------------ confidence head


def test_confidence_head_rows_equal_stock():
    logits = torch.randn(2, N, N, BINS, generator=S.seeded(3))
    m = S.ConfidenceStatement(BINS)
    ref = m(logits)
    for chunk in CHUNKS:
        rows = C.confidence_head_chunked(logits, m.per_rows, chunk=chunk)
        out = S.finish_confidence(rows)
        assert out.keys() == ref.keys() and all(torch.equal(out[k], ref[k]) for k in ref), chunk
        assert C.LOG[-1]["outputs"] == ("err", "contact", "tm_row") and C.LOG[-1]["n_chunks"] == (4 if chunk == 7 else 1)
    with pytest.raises(C.ChunkRefusal) as e:
        C.confidence_head_chunked(logits, lambda b: b.sum(), chunk=7)
    assert e.value.name == "statement_output"
    with pytest.raises(C.ChunkRefusal) as e:
        C.confidence_head_chunked(logits, lambda b: {"tm": b.float().mean(1)}, chunk=7)     # the across-row reduction inside the statement
    assert e.value.name == "statement_output_rows"


# ------------------------------------------------------------------------------------------------------------ MSA rows


def test_msa_rows_transition_bitwise_and_opm_band():
    m = torch.randn(2, SEQ, N, Cz, generator=S.seeded(4))
    t = S.MSATransition(Cz)
    with torch.no_grad():
        ref = t(m)
        for chunk in (3, 10, 50):
            y = C.msa_rows_chunked(t, m, chunk=chunk, exact="bitwise", reason="MSA transition is per position")
            assert torch.equal(y, ref), chunk
        assert C.LOG[-1]["site"] == "msa_rows" and C.LOG[-1]["dim"] == 1
    opm = lambda b: torch.einsum("bsic,bsjd->bijcd", b, b)[..., 0]                            # an outer-product mean: its output has no S axis
    with pytest.raises(C.ChunkRefusal) as e:
        C.msa_rows_chunked(opm, m, chunk=3, exact="bitwise", reason="wrong claim")            # -> shape refusal: a reduction over S cannot be chunked over S
    assert e.value.name == "fn_output_shape"
    with pytest.raises(C.ChunkRefusal) as e:
        C.msa_rows_chunked(t, m[0, 0], chunk=3, exact="bitwise", reason="r")
    assert e.value.name == "not_msa_shaped"


def test_msa_rows_with_offsets_masked_statement_and_lever_rebind():
    """A masked MSA statement slices its mask to the block's rows: through the generic (with_offsets) and through the lever (a rebind taking (t, i0, i1))."""
    m = torch.randn(2, SEQ, N, Cz, generator=S.seeded(5))
    mask = (torch.rand(2, SEQ, N, generator=S.seeded(6)) > 0.2).float()
    t = S.MSATransition(Cz)
    with torch.no_grad():
        ref = t(m) * mask[..., None]
        y = C.msa_rows_chunked(lambda b, i0, i1: t(b) * mask[:, i0:i1, :, None], m, chunk=3, exact="bitwise", reason="masked transition, mask by row offsets", with_offsets=True)
        assert torch.equal(y, ref) and C.LOG[-1]["n_chunks"] == 4
        with pytest.raises(TypeError):                                     # without with_offsets a 3-argument fn is the caller's error, never a silent misuse
            C.msa_rows_chunked(lambda b, i0, i1: b, m, chunk=3, exact="bitwise", reason="r")

    class MaskedMSA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.t = S.MSATransition(Cz)

        def forward(self, m, mask):
            return self.t(m) * mask[..., None]

    def call(module, args, kwargs):
        m_, mask_ = args

        def rebind(t_, i0, i1):
            return (t_, mask_[:, i0:i1]), {}
        return m_, mask_, rebind

    stock = MaskedMSA.forward
    mod = MaskedMSA()
    with torch.no_grad():
        ref = mod(m, mask)
    ctx = _ctx({"chunk_msa_rows": {"cls": MaskedMSA, "call": call, "exact": "bitwise", "reason": "masked per-position transition"}}, settings={"chunk_msa_rows": {"rows": 4}})
    rec = mem.apply(["chunk_msa_rows"], ctx, base="exact")
    C.LOG.clear()
    try:
        rec.unit_begin("u")
        with torch.no_grad():
            y = mod(m, mask)
        rec.unit_end()
        assert torch.equal(y, ref) and rec.units["u"].ran == ["chunk_msa_rows"]
        n_chunks = (next(e["detail"]["n_chunks"] for e in rec.units["u"].events if e.get("kind") == "call") if hasattr(rec, "call") else C.LOG[-1]["n_chunks"])
        assert n_chunks == 3
    finally:
        mem.undo(rec)
    assert MaskedMSA.forward is stock and C._takes_offsets(lambda t_: None) is False and C._takes_offsets(lambda *a: None) is True


# ------------------------------------------------------------------------------------------------------------ policy


def test_parse_setting():
    assert C.parse_setting(None) == "off" and C.parse_setting("0") == "off" and C.parse_setting(" OFF ") == "off" and C.parse_setting(-3) == "off"
    assert C.parse_setting("Auto") == "auto" and C.parse_setting("always") == "always" and C.parse_setting("2304") == 2304 and C.parse_setting(1024) == 1024
    with pytest.raises(C.ChunkRefusal) as e:
        C.parse_setting("big")
    assert e.value.name == "bad_setting"


def test_auto_calibration_quadratic_threshold():
    cal = C.AutoCalibration(base_gb=4.0, full_gb_at_ref=45.0, ref_ntok=1981, headroom=0.85)   # test values: the XL policy's audited cell
    assert cal.threshold(80) == 2304 and cal.threshold(141) == 3072 and cal.threshold(178) == 3584
    assert cal.threshold(4) == 0
    assert C.AutoCalibration(4.0, 45.0, 1981, clamp=2048).threshold(178) == 2048
    assert C.AutoCalibration(4.0, 45.0, 1981, floor=1024).threshold(4) == 1024 and C.AutoCalibration(4.0, 45.0, 1981, floor=1024, clamp=2299).threshold(80) == 2299
    with pytest.raises(C.ChunkRefusal) as e:
        cal.threshold(None)
    assert e.value.name == "no_device_total"
    with pytest.raises(C.ChunkRefusal) as e:
        C.AutoCalibration(4.0, 0.0, 1981)
    assert e.value.name == "bad_calibration"
    with pytest.raises(C.ChunkRefusal) as e:
        C.AutoCalibration(4.0, 45.0, 1981, floor=3000, clamp=2048)
    assert e.value.name == "bad_calibration"


def test_chunk_policy_decisions_are_recorded_with_their_source():
    off = C.ChunkPolicy("trans", "off", rows=256)
    d = off.decide(5000)
    assert d == C.ChunkDecision(False, None, None, "off", 5000, "trans: off")
    always = C.ChunkPolicy("trans", "always", rows=256)
    assert always.decide(1).engaged and always.decide(1).threshold == 0 and always.decide(1).source == "always" and not always.decide(0).engaged
    explicit = C.ChunkPolicy("trans", "1024", rows=256)
    assert explicit.decide(1024).engaged is False and explicit.decide(1024).source == "explicit"
    d = explicit.decide(1025)
    assert d.engaged and d.rows == 256 and d.threshold == 1024 and d.source == "explicit" and d.as_record()["n_tokens"] == 1025
    stepped = C.ChunkPolicy("triatt", 1024, table=((1024, 512), (1536, 256), (2048, 128)))
    assert stepped.decide(1500).rows == 512 and stepped.decide(1537).rows == 256 and stepped.decide(4000).rows == 128 and stepped.decide(4000).source == "table"
    gap = C.ChunkPolicy("triatt", 512, table=((1024, 512),))
    assert not gap.decide(600).engaged and "no rows for this size" in gap.decide(600).detail
    auto = C.ChunkPolicy("trans", "auto", rows=128, calibration=C.AutoCalibration(4.0, 45.0, 1981))
    assert auto.decide(2304, total_gb=80).engaged is False and auto.decide(2305, total_gb=80) == C.ChunkDecision(True, 128, 2304, "auto", 2305, "trans: 2305 tokens > threshold 2304 (auto); rows=128")
    with pytest.raises(C.ChunkRefusal) as e:
        C.ChunkPolicy("trans", "auto", rows=128).decide(3000, total_gb=80)
    assert e.value.name == "no_calibration"
    with pytest.raises(C.ChunkRefusal) as e:
        auto.decide(3000)
    assert e.value.name == "no_device_total"
    for bad, name in [(dict(rows=0), "bad_chunk"), (dict(table=((2048, 128), (1024, 256))), "bad_table"), (dict(), "no_rows"), (dict(setting="huge", rows=1), "bad_setting"),
                      (dict(rows=1, calibration=3), "bad_calibration")]:
        kw = dict(setting="1024")
        kw.update(bad)
        with pytest.raises(C.ChunkRefusal) as e:
            C.ChunkPolicy("x", **kw)
        assert e.value.name == name, bad
    with pytest.raises(C.ChunkRefusal) as e:
        explicit.decide(-1)
    assert e.value.name == "bad_tokens"
    assert auto.describe()["calibration"]["ref_ntok"] == 1981 and stepped.describe()["table"] == [(1024, 512), (1536, 256), (2048, 128)]


def test_device_memory_is_read_through_the_one_reader(monkeypatch):
    """The ``auto`` threshold's device total (chunk) and the allocator-aware free figure (budget) both read the driver figures through
    ``opt_core.arch.device_memory``: a fake torch whose ``mem_get_info`` is the only memory source proves the route (no second reader)."""
    import types

    from opt_core import arch
    from opt_core.mem import budget

    calls = []
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: True, current_device=lambda: 3,
                                      mem_get_info=lambda index=0: calls.append(("mem_get_info", index)) or (10 * 10 ** 9, 80 * 10 ** 9),
                                      memory_reserved=lambda device=None: 4 * 10 ** 9, memory_allocated=lambda device=None: 1 * 10 ** 9)
    fake.device = lambda s: types.SimpleNamespace(type=str(s).split(":")[0], index=int(str(s).split(":")[1]) if ":" in str(s) else None)
    monkeypatch.setitem(sys.modules, "torch", fake)
    assert arch.device_memory(1) == {"total_bytes": 80 * 10 ** 9, "free_bytes": 10 * 10 ** 9, "source": "torch"}
    cuda_x = types.SimpleNamespace(device=types.SimpleNamespace(type="cuda", index=1))
    cpu_x = types.SimpleNamespace(device=types.SimpleNamespace(type="cpu", index=None))
    assert C._device_total_gb(cuda_x) == 80.0 and C._device_total_gb(cpu_x) is None
    assert budget.device_free_bytes() == 13 * 10 ** 9 and budget.device_free_bytes("cuda:2") == 13 * 10 ** 9 and budget.device_free_bytes(5) == 13 * 10 ** 9
    assert calls == [("mem_get_info", 1), ("mem_get_info", 1), ("mem_get_info", 3), ("mem_get_info", 2), ("mem_get_info", 5)]
    auto = C.ChunkPolicy("trans", "auto", rows=128, calibration=C.AutoCalibration(4.0, 45.0, 1981))
    assert auto.decide(2305, total_gb=C._device_total_gb(cuda_x)).threshold == 2304


# ------------------------------------------------------------------------------------------------------------ the registered levers


def test_levers_registered_with_the_contract():
    for name in C.LEVERS_HERE:
        lv = R.get(name)
        assert lv.family == "chunk" and lv.frameworks == ("torch",) and "rows" in lv.settings and "tok" in lv.settings and lv.module == "opt_core.mem.chunk"
        assert lv.exact == ("band" if name.endswith("_band") else "bitwise"), name       # the fp32-capable mechanisms come as a bit-exact / band lever pair
    assert "fp32" in R.get("chunk_pair_transition").settings and set(R.get("chunk_triangle_multiplication").settings) == {"rows", "tok", "mode", "hidden", "channels", "fp32"}
    assert len(C.LEVERS_HERE) == 7 and all(n in R.LEVERS for n in C.LEVERS_HERE)


def _ctx(hooks, settings=None, environ=None):
    return R.Ctx(prefix="SYN", tag="syn-opt", framework="torch", hooks=hooks, settings=settings or {}, environ=environ or {})


def test_applies_refuses_by_name_without_hooks_or_with_bad_settings():
    lv = R.get("chunk_pair_transition")
    r = lv.applies(_ctx({}))
    assert isinstance(r, R.Refusal) and r.lever == "chunk_pair_transition" and r.precondition == "hooks.cls"
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition, "attr": "nope"}}))
    assert r.precondition == "hooks.attr"
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"rows": "zero"}}))
    assert r.precondition == "settings.chunk_pair_transition.rows"
    assert lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, environ={"SYN_BIG_CHUNK_PAIR_TRANSITION_ROWS": "zero"})) is None   # the environment is never read
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"tok": "off"}}))
    assert r.precondition == "setting.tok" and "switches={'chunk_pair_transition': False}" in r.reason
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"fp32": "never"}}))
    assert r.precondition == "bad_fp32_policy"
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"fp32": "band"}}))
    assert r.precondition == "setting.fp32" and "chunk_pair_transition_band" in r.reason     # band is the twin lever, never a setting of the bit-exact one
    r = R.get("chunk_pair_transition_band").applies(_ctx({"chunk_pair_transition_band": {"cls": S.PairTransition}}, settings={"chunk_pair_transition_band": {"fp32": "refuse"}}))
    assert r.precondition == "setting.fp32"
    assert R.get("chunk_pair_transition_band").applies(_ctx({"chunk_pair_transition_band": {"cls": S.PairTransition}})) is None
    r = lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"tok": "auto"}}))
    assert r.precondition == "no_calibration"
    assert lv.applies(_ctx({"chunk_pair_transition": {"cls": S.PairTransition}})) is None
    assert R.get("chunk_triangle_attention").applies(_ctx({"chunk_triangle_attention": {"cls": S.TriangleAttention}})).precondition == "hooks.parts"
    r = R.get("chunk_triangle_multiplication").applies(_ctx({"chunk_triangle_multiplication": {"cls": S.TriangleMultiplication, "parts": S.TriangleMultiplication.parts}},
                                                            settings={"chunk_triangle_multiplication": {"mode": "channels"}}))
    assert r.precondition == "setting.hidden"
    assert R.get("chunk_confidence_head").applies(_ctx({"chunk_confidence_head": {"cls": S.ConfidenceStatement, "logits": lambda m, a, k: a[0], "statement": lambda m: m.per_rows}})).precondition == "hooks.finish"
    assert R.get("chunk_msa_rows").applies(_ctx({"chunk_msa_rows": {"cls": S.MSATransition, "exact": "exactly"}})).precondition == "bad_exact_label"


def test_apply_patches_the_synthetic_stack_through_mem_apply_and_undo_restores():
    stock = {cls: cls.forward for cls in (S.PairTransition, S.TriangleAttention, S.TriangleMultiplication, S.ConfidenceStatement, S.MSATransition)}
    x, mask = pair(dtype=torch.bfloat16)
    logits = torch.randn(2, N, N, BINS, generator=S.seeded(3))
    msa = torch.randn(2, SEQ, N, Cz, generator=S.seeded(4)).to(torch.bfloat16)
    mods = {"trans": S.PairTransition(Cz).to(torch.bfloat16), "att": S.TriangleAttention(Cz, D, H, False).to(torch.bfloat16),
            "mul": S.TriangleMultiplication(Cz, CH, False).to(torch.bfloat16), "conf": S.ConfidenceStatement(BINS), "msa": S.MSATransition(Cz).to(torch.bfloat16)}
    with torch.no_grad():
        ref = {"trans": mods["trans"](x), "att": mods["att"](x, mask), "mul": mods["mul"](x, mask), "conf": mods["conf"](logits), "msa": mods["msa"](msa)}
    hooks = {
        "chunk_pair_transition": {"cls": S.PairTransition},
        "chunk_triangle_attention": {"cls": S.TriangleAttention, "parts": S.TriangleAttention.parts, "starting": lambda m: m.starting},
        "chunk_triangle_multiplication": {"cls": S.TriangleMultiplication, "parts": S.TriangleMultiplication.parts},
        "chunk_confidence_head": {"cls": S.ConfidenceStatement, "logits": lambda m, a, k: a[0], "statement": lambda m: m.per_rows, "finish": lambda m, rows, a, k: S.finish_confidence(rows)},
        "chunk_msa_rows": {"cls": S.MSATransition, "exact": "bitwise", "reason": "per position"},
    }
    ctx = _ctx(hooks, settings={"chunk_pair_transition": {"rows": 7}, "chunk_triangle_attention": {"rows": 7}, "chunk_triangle_multiplication": {"rows": 7},
                               "chunk_confidence_head": {"rows": "5"}, "chunk_msa_rows": {"rows": "3"}},
               environ={"SYN_BIG_CHUNK_CONFIDENCE_HEAD_ROWS": "9", "SYN_BIG_CHUNK_MSA_ROWS_ROWS": "9"})       # the environment is never read
    line = [n for n in C.LEVERS_HERE if not n.endswith("_band")]                          # the bit-exact line of a bf16 stack
    rec = mem.apply(line, ctx, base="fast")
    try:
        assert rec.levers == tuple(line) and rec.exact == "bitwise" and not rec.refused
        applied = {a.lever: a for a in rec.applied}
        assert applied["chunk_pair_transition"].settings == {"rows": 7, "tok": "always", "fp32": "passthrough"} and applied["chunk_pair_transition"].sites == ("tests.synthetic_pair.PairTransition.forward",)
        assert applied["chunk_pair_transition"].exact == "bitwise" and applied["chunk_triangle_multiplication"].settings["fp32"] == "refuse"
        assert applied["chunk_triangle_multiplication"].settings["mode"] == "rows" and applied["chunk_triangle_multiplication"].exact == "bitwise" and applied["chunk_confidence_head"].settings["rows"] == 5 and applied["chunk_msa_rows"].settings["rows"] == 3
        assert all(a.undo is not None for a in rec.applied) and S.PairTransition.forward.big_lever == "chunk_pair_transition"
        assert any(s["lever"] == "chunk_confidence_head" and s["key"] == "rows" and s["source"] == "ctx.settings" and s["value"] == 5 for s in rec.settings)
        rec.unit_begin("item-1")
        with torch.no_grad():
            got = {"trans": mods["trans"](x), "att": mods["att"](x, mask), "mul": mods["mul"](x, mask), "conf": mods["conf"](logits), "msa": mods["msa"](msa)}
        rec.unit_end()
        for k in ref:
            if k == "conf":
                assert all(torch.equal(got["conf"][kk], ref["conf"][kk]) for kk in ref["conf"])
            else:
                assert torch.equal(got[k], ref[k]), k
        u = rec.units["item-1"]
        assert sorted(u.ran) == sorted(line) and not u.fallback and not u.skipped
        assert rec.census()["ok"] and rec.exit_gate(0)["exit_code"] == 0
        if hasattr(rec, "call"):                                           # the record's per-call sink takes the events (the standalone log stays empty by design)
            assert [e["site"] for e in u.events if e.get("kind") == "call" and e["lever"] == "chunk_triangle_attention"] == ["triangle_attention_bias", "triangle_attention"]
            assert not [e for e in C.LOG.entries if e["lever"] == "chunk_triangle_attention"]
        else:                                                              # a record without it: the note line + the standalone log
            assert [e["site"] for e in C.LOG.entries if e["lever"] == "chunk_triangle_attention"] == ["triangle_attention_bias", "triangle_attention"]
            assert any(n.startswith("chunk_triangle_multiplication triangle_multiplication exact=bitwise") for n in rec.notes)
        # a skip by name: stock's own chunk_size set on the call; a below-threshold call
        rec.unit_begin("item-2")
        with torch.no_grad():
            mods["att"](x, mask, chunk_size=5)
        rec.unit_end()
        assert rec.units["item-2"].skipped["chunk_triangle_attention"] == "stock chunk_size set on the call"
    finally:
        undone = mem.undo(rec)
    assert sorted(undone) == sorted(line) and all(cls.forward is fn for cls, fn in stock.items())


def test_apply_fp32_passthrough_is_a_fallback_event_and_band_is_the_twin_lever():
    stock = S.PairTransition.forward
    x, _ = pair()                                                          # fp32 input, no autocast: an fp32 site
    m = S.PairTransition(Cz)
    with torch.no_grad():
        ref = m(x)
    ctx = _ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"rows": 7}})
    rec = mem.apply(["chunk_pair_transition"], ctx, base="exact")
    try:
        rec.unit_begin("u")
        with torch.no_grad():
            assert torch.equal(m(x), ref)
        rec.unit_end()
        assert "passthrough" in rec.units["u"].fallback["chunk_pair_transition"]
        v = rec.exit_gate(0)
        assert v["exit_code"] != 0 and "chunk_pair_transition" in v["partial"]                # the census sees the passthrough: fail-closed
    finally:
        mem.undo(rec)
    assert S.PairTransition.forward is stock
    ctx = _ctx({"chunk_pair_transition_band": {"cls": S.PairTransition}}, settings={"chunk_pair_transition_band": {"rows": 7}})
    rec = mem.apply(["chunk_pair_transition_band"], ctx, base="exact")
    try:
        a = rec.applied[0]                                                 # the band twin: registered band, fp32 sites chunked under it
        assert a.exact == "band" and a.settings["fp32"] == "band" and rec.exact == "band"
        with torch.no_grad():
            y = m(x)
        assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5)
        assert rec.units[mem.PROCESS_UNIT].ran == ["chunk_pair_transition_band"] and not rec.units[mem.PROCESS_UNIT].fallback
    finally:
        mem.undo(rec)
    ctx = _ctx({"chunk_pair_transition": {"cls": S.PairTransition}}, settings={"chunk_pair_transition": {"rows": 7, "fp32": "refuse"}})
    rec = mem.apply(["chunk_pair_transition"], ctx, base="exact")
    try:
        with pytest.raises(C.ChunkRefusal) as e:
            with torch.no_grad():
                m(x)
        assert e.value.name == "fp32_site"
    finally:
        mem.undo(rec)
    assert S.PairTransition.forward is stock


def test_trimul_bitwise_lever_refuses_fp32_products_and_the_band_twin_chunks_them():
    """chunk_triangle_multiplication (bit-exact) refuses an fp32 product by default (or passes it through as a fallback); the _band twin chunks it."""
    hooks = {"chunk_triangle_multiplication": {"cls": S.TriangleMultiplication, "parts": S.TriangleMultiplication.parts}}
    x, mask = pair()
    m = S.TriangleMultiplication(Cz, CH, True)
    with torch.no_grad():
        ref = m(x, mask)
    rec = mem.apply(["chunk_triangle_multiplication"], _ctx(hooks, settings={"chunk_triangle_multiplication": {"rows": 7}}), base="fast")
    try:
        assert rec.applied[0].exact == "bitwise" and rec.exact == "bitwise" and rec.applied[0].settings["fp32"] == "refuse"
        with pytest.raises(C.ChunkRefusal) as e:                           # an fp32 product under the bit-exact lever: refused by name, never chunked band under a bit-exact label
            with torch.no_grad():
                m(x, mask)
        assert e.value.name == "fp32_site"
    finally:
        mem.undo(rec)
    assert R.get("chunk_triangle_multiplication").applies(_ctx(hooks, settings={"chunk_triangle_multiplication": {"rows": 7, "fp32": "band"}})).precondition == "setting.fp32"
    hooks_b = {"chunk_triangle_multiplication_band": hooks["chunk_triangle_multiplication"]}
    rec = mem.apply(["chunk_triangle_multiplication_band"], _ctx(hooks_b, settings={"chunk_triangle_multiplication_band": {"rows": 7}}), base="fast")
    try:
        assert rec.applied[0].exact == "band" and rec.exact == "band" and rec.applied[0].settings["fp32"] == "band"
        rec.unit_begin("u")
        with torch.no_grad():
            y = m(x, mask)
        rec.unit_end()
        _band_equal(y, ref, "band twin")
        assert rec.units["u"].ran == ["chunk_triangle_multiplication_band"] and not rec.units["u"].fallback
    finally:
        mem.undo(rec)


def test_selection_switches_reach_the_chunk_levers():
    sel = R.selection(["chunk_pair_transition", "chunk_msa_rows"], switches={"chunk_msa_rows": False, "chunk_triangle_multiplication": True})
    assert sel.levers == ("chunk_pair_transition", "chunk_triangle_multiplication") and sel.off_by_flag == ("chunk_msa_rows",) and not sel.refusals
