"""Levers trimul / tmpl_trimul ask the shared core's TriMul provider (opt_core.kernels.trimul) for the MODE's tier word (exact | fast | big) at c=128 and at the
template stack's c=64, and lever trimul_exact asks its `exact` word in this engine's module form; every row is served through the provider face, a stock row or a
refusal is the module's own statement by name; the kit carries no TriMul cell table and names no row. The model process names the tier before build_model.
CPU-only: the adapter's selection helpers are compiled from af3_kernels.py's source against a mocked provider and a stub torch (no GPU, no Triton)."""
import ast
import os
import types

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AK_PATH = os.path.join(KIT, "opt", "forward", "af3t", "kernels", "af3_kernels.py")


def _read(*parts):
    with open(os.path.join(KIT, *parts)) as f:
        return f.read()


def _functions(names):
    """The named top-level functions of af3_kernels.py as source text (compiled below against stubs)."""
    src = _read("opt", "forward", "af3t", "kernels", "af3_kernels.py")
    tree = ast.parse(src)
    segs = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            segs.append(ast.get_source_segment(src, node))
    assert len(segs) == len(names), [n for n in names]
    return "\n\n".join(segs)


class _Refusal(Exception):
    def __init__(self, kind, row=None, fallback=None):
        super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback


class _Sel:
    def __init__(self, row, word):
        self.row, self.word = row, word


def _namespace(rows_by_word, cc=(9, 0)):
    """A namespace holding the adapter helpers with: a provider mock whose select() records its kwargs and answers rows_by_word[word] (a row name, or None =
    refused by name); a stub torch naming cc."""
    calls = []

    def select(cc_, dtype, c_z, c_hidden, n, direction, **kw):
        calls.append(dict(cc=cc_, dtype=dtype, c_z=c_z, c_hidden=c_hidden, n=n, direction=direction, **kw))
        row = rows_by_word.get(kw["word"])
        if row is None:
            raise _Refusal("form_vouch_not_recorded(shape=...)" if kw["word"] == "exact" else "no_row", "af3t_form" if kw["word"] == "exact" else None, "torch_math")
        return _Sel(row, kw["word"])
    TPm = types.SimpleNamespace(select=select, Refusal=_Refusal, STOCK_ROWS=("cueq", "torch_math"))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: cc))
    counts = {}
    ns = {"torch": torch, "_TIER": {"word": "fast"}, "_EXACT_FORM": "af3t_module", "_count": lambda lever, w: counts.__setitem__((lever, w), counts.get((lever, w), 0) + 1),
          "_TRIMUL_PROV": {"mod": TPm, "stack": "H100:2.13.0+cu130/3.7.1/nocueq", "sel": {}, "refused": None, "cache": {}, "has_cueq": False, "why": None}}
    exec(compile(_functions(["set_tier", "_trimul_select"]), AK_PATH, "exec"), ns)
    return ns, calls, counts


def test_the_tier_word_is_the_modes_and_naming_it_drops_memoised_selections():
    ns, calls, _ = _namespace({"fast": "native", "big": "v4", "exact": "af3t_form"})
    assert ns["_TIER"]["word"] == "fast"                                       # default tier (a build_model caller that never named one)
    sel = ns["_trimul_select"]("trimul", ns["_TIER"]["word"], 448, 128, "outgoing")
    assert calls[-1]["word"] == "fast" and calls[-1]["cc"] == "9.0" and calls[-1]["dtype"] == "bf16" and calls[-1]["residency"] is None and calls[-1]["form"] is None
    assert (calls[-1]["c_z"], calls[-1]["c_hidden"], calls[-1]["stack"], calls[-1]["has_cueq"]) == (128, 128, "H100:2.13.0+cu130/3.7.1/nocueq", False)
    assert sel.row == "native"                                                 # whatever the provider's cell names: the kit passes the word, never a row
    for word in ("big", "exact", "fast"):
        assert ns["set_tier"](word) == word and ns["_TIER"]["word"] == word and ns["_TRIMUL_PROV"]["sel"] == {}   # naming the tier drops memoised selections
    ns["set_tier"]("big")
    sel = ns["_trimul_select"]("trimul", ns["_TIER"]["word"], 448, 128, "outgoing")
    assert calls[-1]["word"] == "big" and sel.row == "v4"                    # big asks big (never the fast word)
    n = len(calls); ns["_trimul_select"]("trimul", "big", 448, 128, "outgoing"); assert len(calls) == n           # memoised per (word, N, C, direction, precision)
    ns["_trimul_select"]("trimul", "big", 448, 128, "outgoing", "f32z_bf16")  # the confidence head's fp32 pair inside the bf16 autocast region: its own cell (residency fp32)
    assert len(calls) == n + 1 and calls[-1]["residency"] == "fp32" and calls[-1]["dtype"] == "bf16"
    ns["_trimul_select"]("tmpl_trimul", "big", 448, 64, "incoming")           # the template stack's c=64 asks the same tier word at (64, 64)
    assert (calls[-1]["word"], calls[-1]["c_z"], calls[-1]["c_hidden"], calls[-1]["direction"]) == ("big", 64, 64, "incoming")


def test_set_tier_refuses_anything_but_a_provider_tier_word():
    ns, _, _ = _namespace({"fast": "native"})
    for bad in ("off", "native", "v4", "tier", "", None):
        try:
            ns["set_tier"](bad)
        except ValueError:
            continue
        raise AssertionError("set_tier accepted %r" % (bad,))


def test_the_exact_word_is_asked_in_the_module_form_and_a_refusal_is_none_counted_once():
    ns, calls, counts = _namespace({"exact": "af3t_form", "fast": "native"})
    sel = ns["_trimul_select"]("trimul_exact", "exact", 800, 128, "outgoing")
    assert calls[-1]["word"] == "exact" and calls[-1]["form"] == "af3t_module" and sel.row == "af3t_form"      # served where the provider's table vouches the row
    ns8, calls8, counts8 = _namespace({"fast": "v4"}, cc=(8, 0))                # a stack with no vouch record (the mock refuses `exact`): None, by name, counted once per shape
    assert ns8["_trimul_select"]("trimul_exact", "exact", 800, 128, "outgoing") is None
    assert counts8 == {("trimul_exact", "refused:af3t_form:form_vouch_not_recorded"): 1} and ns8["_TRIMUL_PROV"]["refused"] == "af3t_form:form_vouch_not_recorded"
    assert ns8["_trimul_select"]("trimul_exact", "exact", 800, 128, "outgoing") is None and len(calls8) == 1     # memoised: the provider is not asked again for that shape
    assert ns8["_trimul_select"]("trimul", "fast", 800, 128, "outgoing").row == "v4" and calls8[-1]["form"] is None   # the tier words carry no form


def test_no_provider_is_no_selection():
    ns, calls, _ = _namespace({"fast": "native"})
    ns["_TRIMUL_PROV"]["mod"] = None
    assert ns["_trimul_select"]("trimul", "fast", 448, 128, "outgoing") is None and calls == []


def test_the_forward_serves_every_row_through_the_face_and_steps_aside_by_name():
    ak = _read("opt", "forward", "af3t", "kernels", "af3_kernels.py")
    body = ak.split("def _trimul_serve")[1].split("\ndef _trimul_forward")[0]
    assert 'prec = "bf16" if z.dtype == torch.bfloat16 else "f32z_bf16"' in body and "sel = _trimul_select(lever_c, word, N, C, direction, prec)" in body
    assert "TPm.triangle_multiplication(z, mask, direction=direction, weights=w10, selection=sel" in body      # the face, for every row
    assert "fpf_trimul_v4" not in ak and "G.trimul(" not in ak and "trimul_v4_forward" not in ak              # no direct kernel call, no kit cell path
    assert 'if sel.row in TPm.STOCK_ROWS:' in body and '_fallback(lever_c, "stock_row:%s" % sel.row)' in body   # a stock row = the module's own statement, by name
    assert '_fallback(lever_c, "unvouched" if word == "exact" else "refused")' in body                        # a refusal = the module, by name (exact: unvouched)
    assert '_count(lever_c, "served:%s:%s" % (served, direction))' in body and '_count(lever_c, "stepaside:%s->%s" % (sel.row, served))' in body
    assert "except TPm.Refusal as r:" in body
    fwd = ak.split("def _trimul_forward")[1].split("\ndef ")[0]; ex = ak.split("def _trimul_exact_forward")[1].split("\ndef ")[0]
    assert 'return _trimul_serve("trimul", _TIER["word"], self, pair, mask, residual)' in fwd                # trimul: the MODE's word
    assert 'return _trimul_serve("trimul_exact", "exact", self, pair, mask, residual)' in ex                 # trimul_exact: the exact word, whatever the mode's
    aside = ak.split("def _trimul_forward_exact_aside")[1].split("\ndef ")[0]
    assert '_count("trimul_exact", "fallback:superseded:trimul")' in aside                                   # fast / big: trimul owns the class, trimul_exact steps aside by name
    install = ak.split("def _install():")[1].split("\ndef ")[0]
    assert '(_trimul_forward_exact_aside if "trimul_exact" in _ON else _trimul_forward) if "trimul" in _ON' in install
    assert '(_trimul_exact_forward if "trimul_exact" in _ON else _ORIG["trimul"])' in install


def test_the_model_process_names_the_tier_before_build_model_and_the_api_hands_it_to_the_adapter():
    fwd = _read("opt", "af3_torch_opt", "forward.py")
    i = fwd.index('A.set_provider_tier("big" if big_levers else "fast")'); j = fwd.index("model = A.build_model(a.params, levers=levers, **kw)")
    assert fwd.index("big_levers = [t.strip()") < i < j
    api = _read("opt", "forward", "af3t", "af3_torch", "af3_torch_api.py")
    assert "def set_provider_tier(word):" in api and "return K.set_tier(word)" in api.split("def set_provider_tier(word):")[1].split("\ndef ")[0]


def test_the_kit_carries_no_trimul_cell_table_and_names_no_row():
    """The provider's measured cells name the row: no kit cells file, no per-arch table key read for the TriMul, no routed fpf_trimul_v4 name, no row word in the adapter's binding."""
    from af3_torch_opt import registry
    assert not os.path.exists(os.path.join(KIT, "opt", "forward", "af3t", "kernels", "third_party", "fpf_trimul_v4"))
    assert "fpf_trimul_v4" not in registry.KERNEL_ROUTES
    ak = _read("opt", "forward", "af3t", "kernels", "af3_kernels.py")
    for word in ("trimul_row", "trimul64_row", "trimul64_cell", "fpf_trimul_v4", "_TRIMUL64", "FPF_TRIMUL_V4_CELLS"):   # no per-arch table key or kit cell file is read for the TriMul
        assert word not in ak, word
    assert registry.IMPL["trimul"] == ("opt_core.kernels.trimul", "core") and registry.IMPL["tmpl_trimul"] == ("opt_core.kernels.trimul", "core")
    assert registry.IMPL["trimul_exact"] == ("opt_core.kernels.trimul:exact+af3t_module", "core") and registry.STRATEGY["trimul_exact"] == "F2.fpf_trimul_exact"
    assert "trimul_exact" in registry.EXACT and "trimul_exact" not in registry.NOT_BITWISE and {"trimul", "tmpl_trimul"} <= set(registry.NOT_BITWISE)
    assert registry.EXPECTED_FALLBACKS["trimul_exact"] == ("fallback:unvouched", "fallback:superseded:trimul", "fallback:stock_row:{row}")
    sel = ak.split("def _trimul_select")[1].split("\ndef ")[0]
    assert 'word=word' in sel and 'form=(_EXACT_FORM if word == "exact" else None)' in sel and '_EXACT_FORM = "af3t_module"' in ak
