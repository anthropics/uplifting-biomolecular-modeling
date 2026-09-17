"""dit_apb on CPU (class contracts): row / registry / installer membership (fast, big AND exact; after sampler_hoist / atom_sdpa /
atom_kdedup), the word handed to the provider IS the mode's tier word (no per-card / per-row kit table in the module), the expected words ==
the module's literals, sampler_hoist recognises the wrapper, the folded statement through a provider double equals the stock statement
(B = 1 and B = 2, shared bias), rank-3 calls / per-sample masks / AFO_DIT_APB_WORD=0 take the statement below by name, a refusing row follows
the provider's NAMED fallback row once and a refusing word steps aside by name, the exact tier serves nothing unless the provider vouches an
exact-class row against this engine's statement (stock_row:<arm> by name, bitwise stock), and the scratch A/B switch."""
import importlib
import math
import types

import pytest
import torch

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import installers


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


def test_row_registry_installer():
    for mode in (modes.FAST, modes.BIG, modes.EXACT):
        row = modes.MODES[mode]
        assert "dit_apb" in row, mode
        assert "dit_sdpa" not in row, mode                                       # the kit SDPA construction left: the provider's stock-family rows are that kernel
        for earlier in ("sampler_hoist", "atom_kdedup") + (("atom_sdpa",) if mode != modes.EXACT else ()):
            assert row.index(earlier) < row.index("dit_apb"), (mode, earlier)
    assert registry.LEVERS["dit_apb"]["cls"] == "fast"
    assert "dit_apb" in installers() and "dit_sdpa" not in installers() and "dit_sdpa" not in registry.LEVERS


def test_the_word_is_the_modes_tier_word_no_kit_table(monkeypatch):
    from atlasfold_opt.hooks import dit_apb as H
    src = open(H.__file__).read()
    for gone in ("WORD_BY_CARD", "DEFAULT_EAGER_MIN_TOKENS", "AFO_DIT_APB_EAGER_MIN_TOKENS", "eager_small", "(9, 0)", "(8, 0)"):
        assert gone not in src, gone                                             # no per-card word table, no kit size floor: the provider's cells choose
    monkeypatch.delenv(H.ENV, raising=False); monkeypatch.setitem(H._STATE, "override", None)
    assert H.tier_word("fast") == "fast" and H.tier_word("big") == "big" and H.tier_word("exact") == "exact"
    assert H.word("fast") == "fast" and H.word("big") == "big" and H.word("exact") == "exact"
    monkeypatch.setenv(H.ENV, "l3a"); assert H.word("fast") == "l3a"           # the developer override names one provider row
    monkeypatch.setenv(H.ENV, "0"); assert H.word("big") == ""                # off by word
    monkeypatch.setenv(H.ENV, " "); assert H.word("big") == "big"           # blank = unset


def test_expected_words_are_the_module_literals():
    from atlasfold_opt.hooks import dit_apb
    assert tuple(registry.LEVERS["dit_apb"]["expected"]) == dit_apb.EXPECTED
    src = open(dit_apb.__file__).read()
    for word in dit_apb.EXPECTED:
        if word.endswith(":"):
            assert f'"{word}" +' in src or f"'{word}' +" in src or f'"{word}"' in src, word   # parametric prefixes (stock_row:<arm>, refused:<row>)
        else:
            assert f'ledger.fallback("{word}")' in src, word


def test_sampler_hoist_knows_the_wrapper():
    from atlasfold_opt.hooks import sampler_hoist
    assert "Attention.forward[atlasfold_opt:dit_apb]" in sampler_hoist.DIT_BIAS_WRAPPERS
    assert "Attention.forward[atlasfold_opt:dit_apb]" in sampler_hoist.ATOM_BIAS_WRAPPERS


class _Refusal(Exception):
    def __init__(self, kind, row=None, fallback=None):
        super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback


ROWS_OF_TIER = {"fast": "fpf_apb", "big": "apb_attn"}                          # the double's cell winners per tier word (any names: the kit must not care)


def _double(refuse_rows=(), refuse_words=(), exact_row=None):
    """A provider double with the face's words: select() resolves a tier word to the double's cell winner (a row word to itself),
    pair_bias_attention() = softmax(q k^T / sqrt(d) + bias) v in fp32 on [N, H, L, D] ('shnd'); rows in `refuse_rows` refuse AT LAUNCH naming
    fallback 'sdpa'; words in `refuse_words` refuse AT SELECTION; the exact word resolves to `exact_row` (default: the floor stock arm)."""
    calls, selects = [], []
    TIER_WORDS = ("fast", "exact", "faithful", "big"); EXACT_ROWS = ("dit_exact",); STOCK_ROWS = ("sdpa", "sdpa_upcast", "naive")

    def dtype_word(dt):
        s = str(dt).replace("torch.", "")
        return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "float64": "fp64"}.get(s, s)

    def _sel(word):
        if word in refuse_words:
            raise _Refusal(f"unknown_word:{word}", row=word, fallback=None)
        if word == "exact":
            row, variant = (exact_row or ("sdpa", "auto"))
            return types.SimpleNamespace(row=row, variant=variant, cls=("exact" if row in EXACT_ROWS else "stock"), word=word,
                                         exact_vs=("sdpa_upcast" if row in EXACT_ROWS else None), reason="cell_hit; exact vouch not recorded on TEST:stack" if row not in EXACT_ROWS else "cell_hit")
        row = ROWS_OF_TIER.get(word, word)
        r, _, v = row.partition(":")
        return types.SimpleNamespace(row=r, variant=(v or None), cls=("tol" if r not in STOCK_ROWS else "stock"), word=word, exact_vs=None, reason="cell_hit")

    def select(cc, dtype, cell, n_tokens, *, word, samples=1, capture=False, head_dim=None, heads=None, stack=None, abi=None, **kw):
        assert cell == "dit_h16d48" and isinstance(cc, tuple) and dtype in ("bf16", "fp16", "fp32")
        selects.append((word, dtype, n_tokens, samples, capture, stack))
        return _sel(word)

    def pair_bias_attention(q, k, v, bias, key_mask=None, gate=None, *, word=None, selection=None, layout="shnd", cell=None, cache=None, capture=False, **kw):
        sel = selection or _sel(word)
        arm = sel.row + ((":" + sel.variant) if sel.variant else "")
        if sel.row in refuse_rows or arm in refuse_rows:
            raise _Refusal("cc:0.0_not_admitted", row=sel.row, fallback="sdpa")
        assert layout == "shnd" and key_mask is None and gate is None
        calls.append((tuple(q.shape), tuple(bias.shape), arm, word))
        s = torch.einsum("nhqd,nhkd->nhqk", q.float(), k.float()) / math.sqrt(q.shape[-1]) + bias.float()
        out = torch.einsum("nhqk,nhkd->nhqd", torch.softmax(s, -1), v.float()).to(q.dtype)
        return out, sel
    return types.SimpleNamespace(pair_bias_attention=pair_bias_attention, select=select, dtype_word=dtype_word, Refusal=_Refusal, SERVES_CPU=True,
                                 TIER_WORDS=TIER_WORDS, EXACT_ROWS=EXACT_ROWS, STOCK_ROWS=STOCK_ROWS, stack_word=lambda dev=None: "TEST:stack",
                                 dit_exact_abi=lambda: "torchT-cuC-sm0", calls=calls, selects=selects)


def _install_with(monkeypatch, double, mode="fast"):
    import opt_core.kernels as K
    from atlasfold_opt.hooks import dit_apb
    monkeypatch.setattr(K, "apb", double, raising=False)                    # `from opt_core.kernels import apb` reads the package attribute first
    return dit_apb.install(mode, "atlasfold-opt", {})


def _dit_inputs(B, N, L, C, H, dtype=torch.bfloat16):
    a = torch.randn(B, N, L, C).to(dtype)
    mask = torch.ones(B, 1, 1, L, dtype=torch.bool); mask[..., L - 3:] = False          # SelfAttention hands [B, 1, 1, L]
    pb = torch.randn(B, 1, H, L, L).to(dtype)                                            # one block's pair bias, shared over N
    return a, mask, pb


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_tier_word_serves_the_folded_statement_equal_to_stock(monkeypatch, mode):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    stock = att.Attention.forward
    dbl = _double()
    ins = _install_with(monkeypatch, dbl, mode)
    try:
        assert ins.applied and ins.lever == "dit_apb" and ins.facts["word"] == mode and ins.facts["tier"] == mode
        torch.manual_seed(0)
        C, H, L = 64, 4, 24
        m = att.Attention(C, H).to(torch.bfloat16).eval()
        for B, N in ((1, 3), (2, 2)):
            a, mask, pb = _dit_inputs(B, N, L, C, H)
            with torch.no_grad():
                ref = stock(m, a, a, mask, pb)
                out = m(a, a, mask, pb)
            assert out.shape == ref.shape == (B, N, L, C) and out.dtype == ref.dtype
            assert (out.float() - ref.float()).abs().max().item() <= 2e-2 * ref.float().abs().max().item()
        want = ROWS_OF_TIER[mode]
        assert dbl.calls and all(c[1] == (1, H, L, L) and c[2] == want and c[3] == mode for c in dbl.calls) and len(dbl.calls) == 1 + 2   # B=1: one launch; B=2: one per batch element
        assert all(s[0] == mode for s in dbl.selects) and len(dbl.selects) == 2          # one pure selection per call class (S=3, S=2), the tier word verbatim
        line = ins.lines[0]()
        for tok in ("name=LOCAL.atlasfold.dit_apb", "served=2", f"word={mode}", f"row={want}", f"rows={want}:2", f"plan=24e:{want}", "refused=none", "cls=tol"):
            assert tok in line, (tok, line)
        assert "eager_min_tokens" not in line and "eager_small" not in line
        # rank-3 (Pairformer) call: below by name
        s = torch.randn(1, L, C).to(torch.bfloat16); m3 = torch.ones(1, 1, L, dtype=torch.bool)
        with torch.no_grad():
            assert torch.equal(m(s, s, m3, None), stock(m, s, s, m3, None))
        assert "rank:1" in ins.lines[0]()
        # a per-sample mask ([B, N, 1, L]): not the sampler's statement -> below by name, bitwise stock
        a, mask, pb = _dit_inputs(1, 2, L, C, H); maskn = torch.rand(1, 2, 1, L) > 0.3; maskn[..., 0] = True
        with torch.no_grad():
            assert torch.equal(m(a, a, maskn, pb), stock(m, a, a, maskn, pb))
        assert "bias_form:1" in ins.lines[0]()
        # fp32 activations are a provider cell too (fp32 cells), served through the same word
        m32 = att.Attention(C, H).eval(); a32, mask32, pb32 = _dit_inputs(1, 2, L, C, H, torch.float32)
        with torch.no_grad():
            out32 = m32(a32, a32, mask32, pb32); ref32 = stock(m32, a32, a32, mask32, pb32)
        assert (out32 - ref32).abs().max().item() <= 1e-4 * max(1.0, ref32.abs().max().item()) and "24e:fp32" in ins.lines[0]()
        monkeypatch.setenv("AFO_DIT_APB_WORD", "0")
        with torch.no_grad():
            assert torch.equal(m(a, a, mask, pb), stock(m, a, a, mask, pb))
        assert "disabled:1" in ins.lines[0]()
        assert ins.gates[0]().ok
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_a_row_refusing_at_launch_follows_the_providers_named_fallback_once(monkeypatch):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    stock = att.Attention.forward
    dbl = _double(refuse_rows=("fpf_apb",))                                    # the fast cell's winner refuses here; the provider names 'sdpa' instead
    ins = _install_with(monkeypatch, dbl, "fast")
    try:
        C, H, L = 64, 4, 16
        m = att.Attention(C, H).to(torch.bfloat16).eval()
        a, mask, pb = _dit_inputs(1, 2, L, C, H)
        with torch.no_grad():
            for _ in range(3):
                out = m(a, a, mask, pb); ref = stock(m, a, a, mask, pb)
                assert (out.float() - ref.float()).abs().max().item() <= 2e-2 * ref.float().abs().max().item()
        line = ins.lines[0]()
        assert "served=3" in line and "rows=sdpa:3" in line and "refused=fpf_apb:cc:0.0_not_admitted->sdpa" in line and "plan=16e:sdpa" in line, line
        assert all(c[2] == "sdpa" for c in dbl.calls) and ins.gates[0]().ok
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_a_refused_word_steps_aside_by_name(monkeypatch):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.setenv("AFO_DIT_APB_WORD", "dit_exact")                        # a developer row word the double refuses at launch with fallback 'sdpa', and 'sdpa' refuses too
    dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    stock = att.Attention.forward
    ins = _install_with(monkeypatch, _double(refuse_rows=("dit_exact", "sdpa")), "fast")
    try:
        C, H, L = 64, 4, 16
        m = att.Attention(C, H).to(torch.bfloat16).eval()
        a, mask, pb = _dit_inputs(1, 2, L, C, H)
        with torch.no_grad():
            for _ in range(3):
                assert torch.equal(m(a, a, mask, pb), stock(m, a, a, mask, pb))
        line = ins.lines[0]()
        for tok in ("fallback_by=refused:dit_exact:3", "refused=dit_exact:cc:0.0_not_admitted->sdpa,sdpa:cc:0.0_not_admitted", "served=0", "word=dit_exact",
                    "state=skipped reason=all_fallback:refused:dit_exact", "plan=16e:refused/dit_exact", "rows=none"):
            assert tok in line, (tok, line)
        assert " row=" not in line                                           # nothing was served: no row= / cls= tokens
        assert ins.gates[0]().ok                                            # a named step-aside: the run is the statement below, the line says why
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_exact_tier_serves_nothing_unvouched_bitwise_stock(monkeypatch):
    """--mode exact: the provider's exact word resolves to a stock-family arm (no exact-class row vouched on this stack) -> every call class takes
    the module's own statement BY NAME (stock_row:<arm>), no launch through the provider, outputs bitwise the stock statement; an exact-class
    row that replicates ANOTHER statement (dit_exact vs sdpa_upcast) is not this engine's statement either."""
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    for exact_row, why in ((None, "vouch_not_recorded_on:TEST:stack"), (("dit_exact", None), "dit_exact_replicates:sdpa_upcast")):
        dit_apb._STATE["override"] = None
        _restore(att.Attention, "forward")
        stock = att.Attention.forward
        dbl = _double(exact_row=exact_row)
        ins = _install_with(monkeypatch, dbl, "exact")
        try:
            assert ins.applied and ins.facts["word"] == "exact" and ins.facts["tier"] == "exact"
            C, H, L = 64, 4, 16
            m32 = att.Attention(C, H).eval()
            for B, N in ((1, 3), (2, 2)):
                a, mask, pb = _dit_inputs(B, N, L, C, H, torch.float32)
                with torch.no_grad():
                    assert torch.equal(m32(a, a, mask, pb), stock(m32, a, a, mask, pb))
            assert not dbl.calls                                                # nothing launched through the provider
            assert dbl.selects and all(s[0] == "exact" and s[5] == "TEST:stack" for s in dbl.selects)   # judged on THIS stack
            line = ins.lines[0]()
            arm = "sdpa:auto" if exact_row is None else "dit_exact"
            for tok in ("served=0", f"state=skipped reason=all_fallback:stock_row:{arm}", "word=exact", f"plan=16e:fp32:stock_row/{arm}", f"refused=exact_word:{why}"):
                assert tok in line, (tok, line)
            assert ins.gates[0]().ok                                            # an exact run whose DiT calls are all the stock statement is a legitimate exact run
        finally:
            _restore(att.Attention, "forward")
            dit_apb._STATE["override"] = None


def test_exact_mode_refuses_a_tolerance_word_by_name(monkeypatch):
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    _restore(att.Attention, "forward")
    try:
        for w in ("l3a", "fast", "sdpa:cudnn"):
            monkeypatch.setenv("AFO_DIT_APB_WORD", w); dit_apb._STATE["override"] = None
            ins = _install_with(monkeypatch, _double(), "exact")
            assert not ins.applied and ins.reason == f"not_an_exact_word:{w}"
            assert getattr(att.Attention.forward, "__wrapped_stock__", None) is None   # nothing bound
        monkeypatch.setenv("AFO_DIT_APB_WORD", "0"); dit_apb._STATE["override"] = None
        ins = _install_with(monkeypatch, _double(), "exact")
        assert ins.applied and ins.facts["word"] == "off"                       # off by word: installed, inert (the AFO_* convention)
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_graph_context_selects_the_graph_cells(monkeypatch):
    """Inside denoiser_graph's warm-up / capture (graph_context) the provider is asked with capture=True (its graph cells), on the default
    stream with capture=False (eager cells): the row the warm-up compiles is the row the capture records."""
    from atlasfold_opt.hooks import dit_apb
    att = importlib.import_module(dit_apb.TARGET)
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False); dit_apb._STATE["override"] = None
    _restore(att.Attention, "forward")
    dbl = _double()
    state = {"g": False}
    monkeypatch.setattr(dit_apb, "graph_context", lambda t: state["g"])
    ins = _install_with(monkeypatch, dbl, "fast")
    try:
        C, H, L = 64, 4, 16
        m = att.Attention(C, H).to(torch.bfloat16).eval()
        a, mask, pb = _dit_inputs(1, 2, L, C, H)
        with torch.no_grad():
            m(a, a, mask, pb); state["g"] = True; m(a, a, mask, pb); m(a, a, mask, pb)
        caps = [s[4] for s in dbl.selects]
        assert caps == [False, True], caps                                      # one pure selection per (class, timing form); the third call is the cached graph class
        line = ins.lines[0]()
        assert "plan=16e:fpf_apb,16g:fpf_apb" in line and "served=3" in line, line
        assert ins.gates[0]().ok
    finally:
        _restore(att.Attention, "forward")
        dit_apb._STATE["override"] = None


def test_bench_arm_switch(monkeypatch):
    from atlasfold_opt.hooks import dit_apb
    monkeypatch.delenv("AFO_DIT_APB_WORD", raising=False)
    try:
        dit_apb._STATE["override"] = None; dit_apb._STATE["mode"] = "big"
        assert dit_apb.word() == "big"
        dit_apb.bench_arm("A"); assert dit_apb.word() == ""
        dit_apb.bench_arm("B"); assert dit_apb.word() == "big"
        dit_apb.bench_arm("apb_attn"); assert dit_apb.word() == "apb_attn"
        dit_apb.bench_arm(None)
        monkeypatch.setenv("AFO_DIT_APB_WORD", "off"); assert dit_apb.word() == ""
    finally:
        dit_apb._STATE["override"] = None; dit_apb._STATE["mode"] = None
