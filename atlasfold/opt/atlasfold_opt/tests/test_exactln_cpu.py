"""exactln — LayerNorm.forward bound BY TIER WORD to opt_core.kernels.ln (CPU: rows / registry / ORDER after ln_bf16, the word per mode, no kit-side
row floor left, the provider's cell answers for atlasfold's LayerNorm call classes on both cards read through the provider's own select() — pair
rows, single / token / atom rows, the three dtype forms, an unknown width —, install -> a CPU call takes the statement below by name with the stock
output unchanged, AFO_EXACTLN=0 -> `disabled`, the in-process override)."""
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import exactln as E

STACK_H100 = "H100:torch2.7.1+cu128/3.3.1/cueq0.10.0"          # the kit's pinned stack word as the provider spells it


def _restore(cls, attr):
    fn = getattr(cls, attr)
    while getattr(fn, "__wrapped_stock__", None) is not None:
        fn = fn.__wrapped_stock__
    setattr(cls, attr, fn)


def test_rows_registry_order_and_words():
    for m in ("exact", "fast", "big"):
        row = list(modes.levers_of(m))
        assert "exactln" in row and row.index("ln_bf16") < row.index("exactln"), m       # outermost on LayerNorm.forward: a stock-row call runs ln_bf16's statement below
        assert E.word(m) == m                                                              # the mode's tier word
    r = registry.LEVERS["exactln"]
    assert r["cls"] == "exact" and r["module"] == "atlasfold_opt.hooks.exactln" and r["strategy"] == E.NAME
    assert tuple(r["expected"]) == E.EXPECTED
    for w in ("stock_row:", "refused:", "no_affine:", "disabled", "cpu", "param_form"):
        assert w in r["expected"]
    for gone in ("small", "eager_small", "unsupported"):                                    # the kit row floors and the row-word binding's refusal word are gone
        assert gone not in r["expected"]
    for name in ("MIN_ROWS_ENV", "EAGER_MIN_ROWS_ENV", "DEFAULT_MIN_ROWS", "DEFAULT_EAGER_MIN_ROWS", "min_rows", "eager_min_rows", "WORD"):
        assert not hasattr(E, name), name


def test_word_env(monkeypatch):
    monkeypatch.setenv(E.WORD_ENV, "exactln")
    assert E.word("fast") == "exactln"
    monkeypatch.setenv(E.WORD_ENV, " ")
    assert E.word("big") == "big"


def test_provider_answers_for_atlasfold_call_classes():
    """What the lever resolves per call class, read off the provider: the row count picks the cell family (pair N^2 / atom 8N / dit 5N / single N),
    the exact word names a bitwise-recorded row on the running stack or ATen, fast / big name the cells' winners, a width without a family names
    the stock row.  Nothing here pins a row."""
    LN = pytest.importorskip("opt_core.kernels.ln")
    torch = pytest.importorskip("torch")
    bf16, f32 = torch.bfloat16, torch.float32
    assert set(E.NO_AFFINE_ROWS) <= set(LN.ROW_NAMES) and not set(E.NO_AFFINE_ROWS) & set(LN.STOCK_ROWS)
    for cc in ("9.0", "8.0"):
        for n in (256, 640, 896, 1280):
            rows = n * n
            cw = LN.cell_for_rows(rows, n, 128)
            assert cw is not None and str(cw).startswith("pair_c128"), (cc, n, cw)
            for xdt, widen, out in ((f32, False, None), (bf16, False, None), (bf16, True, "bf16")):
                for wd in ("fast", "big"):
                    s = LN.select(cc, xdt, cw, n, word=wd, widen=widen, out=out, C=128, has_triton=True, rows=rows)
                    assert s.row in LN.ROW_NAMES and s.word == wd, (cc, n, xdt, wd, s)
                e = LN.select(cc, xdt, cw, n, word="exact", widen=widen, out=out, C=128, has_triton=True, rows=rows, stack=STACK_H100 if cc == "9.0" else None)
                assert e.row in LN.STOCK_ROWS or e.row in LN.EXACT_ROWS, (cc, n, xdt, e)   # exact: a bitwise row recorded on this stack, else ATen by name — never a tolerance row
        for C, rows in ((384, 896), (768, 5 * 896), (128, 8 * 896 * 4)):                # single c384 token rows, the DiT's 5N token rows, atom-shaped rows
            cw = LN.cell_for_rows(rows, 896, C)
            s = LN.select(cc, f32, cw, 896, word="fast", C=C, has_triton=True, rows=rows)
            assert s.row in LN.ROW_NAMES, (cc, C, rows, s)
        assert LN.cell_for_rows(4096, 64, 201) is None                                      # PairConditioning's C=201 concat norm: no family -> the word's stock row, by name
        s = LN.select(cc, f32, None, 64, word="fast", C=201, has_triton=True, rows=4096)
        assert s.row in LN.STOCK_ROWS, s


def test_install_cpu_paths(monkeypatch):
    torch = pytest.importorskip("torch")
    nm = pytest.importorskip(E.TARGET)
    monkeypatch.delenv(E.ENV, raising=False); monkeypatch.delenv(E.WORD_ENV, raising=False)
    E._STATE["override"] = None
    _restore(nm.LayerNorm, "forward")
    stock = nm.LayerNorm.forward
    try:
        ins = E.install("fast", "[t]", {"mode": "fast"})
        assert ins.applied, ins.reason
        assert getattr(nm.LayerNorm.forward, "__wrapped_stock__", None) is stock
        if ins.facts["provider"] is None:
            pytest.skip(f"provider absent here: {ins.facts['refusal']}")
        assert ins.facts["word"] == "fast" and ins.facts["pinned"] is False
        m = nm.LayerNorm(128)
        x = torch.randn(2, 64, 128)
        with torch.no_grad():
            assert torch.equal(m(x), stock(m, x))                                          # CPU tensor: the statement below, by name
        line = ins.lines[0]()
        for tok in ("name=LOCAL.atlasfold.exactln", "state=skipped reason=all_fallback:cpu", " word=fast", " pinned=0", " rows=none", " plan=none", "origin=core"):
            assert tok in line, line
        for gone in (" min_rows=", " eager_min_rows="):
            assert gone not in line, line
        monkeypatch.setenv(E.ENV, "0")
        with torch.no_grad():
            assert torch.equal(m(x), stock(m, x))
        assert "disabled:1" in ins.lines[0]() and ins.gates[0]().ok
    finally:
        _restore(nm.LayerNorm, "forward")
        E._STATE["override"] = None
    monkeypatch.setenv(E.WORD_ENV, "aten")
    try:
        ins = E.install("exact", "[t]", {})
        assert not ins.applied and ins.reason == "word_names_no_kernel:aten"
        monkeypatch.setenv(E.WORD_ENV, "nope")
        ins = E.install("exact", "[t]", {})
        assert not ins.applied and ins.reason == "unknown_word:nope"
    finally:
        _restore(nm.LayerNorm, "forward")


def test_bench_arm_switch_overrides_the_env_word(monkeypatch):
    monkeypatch.setenv(E.ENV, "0")
    try:
        E._STATE["override"] = None
        assert E.enabled() is False
        E.bench_arm("B"); assert E.enabled() is True
        E.bench_arm("A"); assert E.enabled() is False
    finally:
        E._STATE["override"] = None
