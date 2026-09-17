"""of3_sampler.dit_rows: the provider word door (set_apb_word / apb_word / ProviderCore / resolve_apb_core). CPU-only: the words, the
resolution and the calling-convention refusals; the rows themselves are the provider's tests (test_apb_provider_cells)."""
import pytest

from opt_core.of3_sampler import dit_rows as DR
from opt_core.kernels import apb as P


def setup_function(_f):
    DR.set_apb_word(None)


def teardown_function(_f):
    DR.set_apb_word(None)


def test_words():
    assert DR.apb_word() is None
    for w in ("fast", "big", "exact", "apb_attn", "fpf_apb", "sdpa:cudnn", "dtk_loop", " FAST "):
        assert DR.set_apb_word(w) == w.strip().lower() and DR.apb_word() == w.strip().lower()
    for bad in ("flash", "sdpa:nope", "tier"):
        with pytest.raises(ValueError):
            DR.set_apb_word(bad)
    assert DR.set_apb_word("") is None and DR.apb_word() is None and DR.set_apb_word(None) is None


def test_resolve():
    assert DR.resolve_apb_core("dtk") == (None, "pinned:dtk")
    core, why = DR.resolve_apb_core("auto")
    assert not isinstance(core, DR.ProviderCore)                       # nothing bound: the direct entry (or None when its kernel is not importable here)
    DR.set_apb_word("fast", source="test")
    core, why = DR.resolve_apb_core("auto")
    assert isinstance(core, DR.ProviderCore) and why == "" and core.word == "fast" and core.kernel() is P
    pc, why = DR.resolve_apb_core("sdpa")
    assert isinstance(pc, DR.ProviderCore) and pc.word == "sdpa"
    bad, why = DR.resolve_apb_core("no_such_row")
    assert bad is None and why.startswith("provider:")


def test_calling_convention_refusals_cpu():
    torch = pytest.importorskip("torch")
    core = DR.ProviderCore("fast")
    S, H, D, N = 2, 16, 24, 8
    q = torch.zeros(S, H, N, D); pb = torch.zeros(H, N, N); km = torch.ones(1, N)
    with pytest.raises(core.Unsupported) as ei:                        # CPU tensors: refused by name (the rows serve CUDA tensors), the caller's stock path
        core.pair_bias_attention(q, q, q, pb, km, num_samples=S, num_heads=H, layout="shnd")
    assert ei.value.event == "device"
    with pytest.raises(core.Unsupported) as ei:
        core.pair_bias_attention(q, q, q, torch.zeros(3, H, N, N), km, num_samples=S, num_heads=H, layout="shnd")
    assert ei.value.event == "bias_shape"
    with pytest.raises(core.Unsupported) as ei:
        core.pair_bias_attention(q, q, q, pb, torch.ones(3, N), num_samples=S, num_heads=H, layout="shnd")
    assert ei.value.event == "mask_shape"
    with pytest.raises(core.Unsupported) as ei:
        core.pair_bias_attention(q.reshape(S * H * N, D), q, q, pb, layout="rows")
    assert ei.value.event == "args"
    with pytest.raises(core.Unsupported) as ei:
        core.pair_bias_attention(q, q, q, pb, layout="bshnd")
    assert ei.value.event == "layout"
    c = core.census()
    assert c["served"] == 0 and c["word"] == "fast"
