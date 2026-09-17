"""kernels.trimul row `of3_form` (the OpenFold-family module statement issued whole-tensor): registration, admission and refusals by name, the
table's row entry and in_form records, module-level import hygiene -- pure checks (no GPU); the serving refusals run when torch is importable."""
import ast
import json
import os

import pytest

from opt_core.kernels import trimul as T

HERE = os.path.dirname(os.path.abspath(T.__file__))


def test_row_is_registered_exact_class_forward_only_and_a_tier_value_only_under_its_form():
    assert "of3_form" in T.ROW_NAMES and "of3_form" in T.EXACT_ROWS and "of3_form" in T.MODULE_EXACT_ROWS
    assert "of3_form" not in T.BACKWARD_ROWS and "of3_form" not in T.STOCK_ROWS and "of3_form" not in T.NEEDS_ESM_IMAGE
    t = T.table()
    row = t["rows"]["of3_form"]
    assert row["class"] == "exact" and row["backward"] is False and row["fallback"] == "torch_math"
    assert "module" in row["exact_vs"].lower() and "not the cuequivariance op" in row["exact_vs"].lower()
    assert "of3_form" in t["tiers"]["exact"] and "of3_form" in t["tiers"]["fast"]
    for key, cell in t["cells"].items():                                   # a MODULE-exact row is no cell's tier value (the exact rule is against the stock op)
        assert cell.get("exact") != "of3_form" and cell.get("fast") != "of3_form", key
        for st, w in (cell.get("exact_per_stack") or {}).items():
            assert w != "of3_form", (key, st)


def test_admission_by_name():
    T.admits("of3_form", (9, 0), "bf16", 64, 64, 400)
    T.admits("of3_form", "9.0", "bf16", 64, 128, 1200)                      # c_hidden != c_z served (unfused projections)
    T.admits("of3_form", (8, 0), "fp32", 128, 128, 800, tf32=True)
    T.admits("of3_form", (9, 0), "bf16", 128, 128, 384, residency="fp32")   # f32z_bf16
    with pytest.raises(T.Refusal) as r:
        T.admits("of3_form", (9, 0), "fp16", 64, 64, 400)
    assert r.value.fallback == "torch_math" and "dtype" in r.value.kind
    with pytest.raises(T.Refusal) as r:
        T.admits("of3_form", (7, 5), "bf16", 64, 64, 400)
    assert r.value.fallback == "torch_math" and r.value.kind.startswith("cc:")
    with pytest.raises(T.Refusal):
        T.admits("of3_form", (9, 0), "bf16", 64, 64, 400, backward=True)


def test_row_word_selects_it_and_tier_words_without_form_never_do():
    sel = T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="of3_form", stack="H100:2.10.0+cu128/3.6.0/cueq0.10.0")   # vouched (stack, N) only
    assert sel.row == "of3_form" and sel.cls.startswith("bitwise")
    with pytest.raises(T.Refusal):
        T.select((9, 0), "bf16", 64, 64, 400, "outgoing", word="of3_form")                                                # no stack word: not vouched
    for n in (400, 800, 1200):
        for d in ("outgoing", "incoming"):
            assert T.select((9, 0), "bf16", 64, 64, n, d, word="exact").row != "of3_form"
            assert T.select((9, 0), "bf16", 64, 64, n, d, word="fast").row != "of3_form"
            assert T.select((9, 0), "bf16", 128, 128, n, d, word="exact").row != "of3_form"


def test_in_form_records_on_the_openfold3_cells():
    t = T.table()
    assert "in_form_note" in t and "PAIR" in t["in_form_note"]
    seen_template, seen_trunk = 0, 0
    for n in (400, 800, 1200):
        for d in ("out", "in"):
            c64 = t["cells"]["9.0|bf16|C64|H64|N<=%d|%s|fwd" % (n, d)]
            c128 = t["cells"]["9.0|bf16|C128|H128|N<=%d|%s|fwd" % (n, d)]
            for cell, form in ((c64, "openfold3_template"), (c128, "openfold3_pairformer")):
                assert "in_form" in cell, (n, d, form)
                recs = cell["in_form"]["stacks"]
                assert recs, (n, d)
                for st, forms in recs.items():
                    assert st in t["stacks"]
                    if "openfold3_template" not in forms and "openfold3_pairformer" not in forms:
                        continue                                          # another engine's records on its own stack (af3t_*): tests/test_trimul_forms_af3t.py
                    rec = forms[form]
                    assert rec["measured_at"] <= n and rec["capture"] and rec["in_engine"] and rec["states"]
                    as_cap = [s for s in rec["states"] if s.startswith("as_captured")]
                    assert as_cap
                    rows = rec["states"][as_cap[0]]["rows"]
                    assert rows["statement"]["class"].startswith("bitwise") and rows["statement"]["e2e_ms"] > 0
                    if form == "openfold3_template":                       # the module statement: of3_form bitwise and at or above parity; the library-exact rows are not bitwise
                        assert rows["of3_form"]["class"].startswith("bitwise") and rows["of3_form"]["x_e2e_vs_statement"] >= 1.0
                        assert not rows["tmk3_exact"]["class"].startswith("bitwise") and not rows["cueq"]["class"].startswith("bitwise")
                        assert rec["states"][as_cap[0]]["exact_in_form"].startswith("of3_form")
                        assert rec["capture"]["engine_flags"]["use_cueq_triangle_kernels"] is False
                        seen_template += 1
                    else:                                                  # the library statement: the library-exact rows bitwise, of3_form tolerance
                        assert rows["tmk3_exact"]["class"].startswith("bitwise") and rows["cueq"]["class"].startswith("bitwise")
                        assert not rows["of3_form"]["class"].startswith("bitwise")
                        assert rec["capture"]["engine_flags"]["use_cueq_triangle_kernels"] is True
                        seen_trunk += 1
    assert seen_template == 6 and seen_trunk == 6


def test_row_module_imports_only_the_standard_library_at_module_level():
    src = open(os.path.join(HERE, "of3_form.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert top == [], [ast.dump(n) for n in top]                            # torch and the LayerNorm provider are imported inside functions only
    meta = json.load(open(os.path.join(os.path.dirname(HERE), "META", "trimul.json"), encoding="utf-8"))
    assert "of3_form.py" in meta["core_added"]


def test_serving_refusals_with_tensors():
    torch = pytest.importorskip("torch")
    from opt_core.kernels.trimul import of3_form as OF
    w = {k: torch.zeros((64, 64)) if k.startswith("w_") else torch.zeros(64) for k in T.WEIGHT_KEYS}
    z = torch.zeros((1, 8, 8, 64))
    assert OF.supported(z, None, w).startswith("device:")                  # CPU tensors refused by name (the statements are CUDA kernels)
    if torch.cuda.is_available():
        zc = z.cuda().half()
        assert OF.supported(zc, None, {k: v.cuda() for k, v in w.items()}).startswith("dtype:")


def test_weight_casts_are_memoised_without_version_counters_under_inference_mode():
    """The engines call under torch.inference_mode; tensors made there (an adapter's copies) are inference tensors with no version counter:
    the rows' caches key on storage / shape / stride / dtype / device only (reading `_version` raised on torch >= 2.13 images)."""
    torch = pytest.importorskip("torch")
    from opt_core.kernels.trimul import of3_form as OF, af3t_form as AF
    src = open(os.path.join(HERE, "of3_form.py"), encoding="utf-8").read() + open(os.path.join(HERE, "af3t_form.py"), encoding="utf-8").read()
    assert src.count("._version") == 2 and src.count("Inference tensors do not track version counter") >= 2   # read ONLY inside the guarded _wsig helpers
    w = torch.randn(64, 64)
    for ctx in (torch.inference_mode, torch.no_grad):
        with ctx():
            wi = w.clone()                                                # created inside the mode
            cache = {}
            a = OF._cast(cache, "w_ap", wi, torch.bfloat16)
            assert a.dtype == torch.bfloat16 and torch.equal(a, wi.to(torch.bfloat16))
            assert OF._cast(cache, "w_ap", wi, torch.bfloat16) is a          # memoised on the same storage
            assert OF._cast(cache, "w_ap", torch.randn(64, 64), torch.bfloat16) is not a   # another storage: recast
            assert OF._cast(cache, "w_ap", wi, torch.float32) is wi           # already in dtype: the tensor itself
            c2 = {}
            wc = w.clone()
            t = AF._w2(c2, "p", wi, wc, torch.bfloat16)
            assert tuple(t.shape) == (128, 64) and AF._w2(c2, "p", wi, wc, torch.bfloat16) is t          # same source tensors: the memo
            assert AF._w2(c2, "p", wi, wi, torch.bfloat16) is not t                                     # other sources at the same shapes: rebuilt (multi-weight)
    assert "hm_fused" in OF.CONFIGS and OF.CONFIGS["hm_fused"]["fused"] is True and OF.CONFIGS["upstream"]["fused"] is False


def test_multi_weight_caches_key_on_weight_identity_never_on_shape_alone():
    """FLEET AUDIT (multi-weight): one cache dict shared across modules, weight sets A, B, A, C interleaved at the same shapes -> every memoised
    cast / pack is the one for THAT weight set (bitwise); an address recycled after free, or an in-place update, never serves a stale value."""
    torch = pytest.importorskip("torch")
    from opt_core.kernels.trimul import of3_form as OF, af3t_form as AF
    g = torch.Generator().manual_seed(0)
    A, B, C = [torch.randn(64, 64, generator=g) for _ in range(3)]
    cache = {}
    for w in (A, B, A, C, B):
        assert torch.equal(OF._cast(cache, "w_ap", w, torch.bfloat16), w.to(torch.bfloat16))
    D = torch.randn(64, 64, generator=g); OF._cast(cache, "w_ap", D, torch.bfloat16); del D          # freed while cached: the entry holds it, no address collision
    E = torch.randn(64, 64, generator=g)
    ge = OF._cast(cache, "w_ap", E, torch.bfloat16); assert torch.equal(ge, E.to(torch.bfloat16))
    E.add_(1.0)                                                                                          # in-place update: the version counter moves
    ge2 = OF._cast(cache, "w_ap", E, torch.bfloat16); assert torch.equal(ge2, E.to(torch.bfloat16)) and ge2 is not ge
    assert OF._cast(cache, "w_ap", E, torch.bfloat16) is ge2                                            # memo hit on the same tensor
    c2 = {}
    for we, wo in ((A, B), (B, C), (A, B), (C, A)):
        t = AF._w2(c2, "wp_out", we, wo, torch.bfloat16)
        ref = torch.empty(128, 64, dtype=torch.bfloat16); ref[0::2] = we.to(torch.bfloat16); ref[1::2] = wo.to(torch.bfloat16)
        assert torch.equal(t, ref)
    assert AF._w2(c2, "wp_out", C, A, torch.bfloat16) is AF._w2(c2, "wp_out", C, A, torch.bfloat16)
    g1 = AF._cached(c2, "wo_bf16", lambda: A.to(torch.bfloat16), A); g2 = AF._cached(c2, "wo_bf16", lambda: B.to(torch.bfloat16), B)
    assert torch.equal(g1, A.to(torch.bfloat16)) and torch.equal(g2, B.to(torch.bfloat16))
    src = open(os.path.join(HERE, "af3t_form.py"), encoding="utf-8").read()
    assert "def _cached(cache, key, make, *srcs)" in src and src.count("_cached(cache, ") == src.count("_cached(cache, ") and "lambda: w[\"w_o\"].detach().to(gdt2).contiguous(), w[\"w_o\"]" in src
