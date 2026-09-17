"""Lever triattn asks the shared core's triangle-attention provider (opt_core.kernels.triattn) for the MODE's tier word -- fast in fast, big in
big -- at every (card, dtype, head dim, heads, size) cell: the trunk / MSA / confidence pair stacks' head dim 32 and the template stack's head dim
16 alike; the provider names the row, the kit names none.  A refusal by name binds the fallback row the PROVIDER names, else the stock statement;
a stock row is never a kernel the kit routes the trunk to.  CPU-only: the adapter's selection helpers are compiled from af3_kernels.py's source
against a stub provider and a stub torch (no GPU, no Triton)."""
import ast
import os
import re
import types

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AK_PATH = os.path.join(KIT, "opt", "forward", "af3t", "kernels", "af3_kernels.py")


def _src():
    with open(AK_PATH, encoding="utf-8") as f:
        return f.read()


def _functions(names):
    src = _src()
    tree = ast.parse(src)
    segs = [ast.get_source_segment(src, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(segs) == len(names), names
    return "\n\n".join(segs)


class _Refusal(Exception):
    def __init__(self, kind, row=None, fallback=None):
        super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback


class _Sel:
    def __init__(self, row, word):
        self.row, self.word = row, word


def _namespace(answer, tier="fast", cc=(9, 0), tier_words=("fast", "exact", "big")):
    """answer(word, D, N) -> row name | _Refusal instance (raised).  Returns (namespace, calls, counts)."""
    calls, counts = [], {}

    def select(cc_, dtype, head_dim, heads, n, direction="fwd", **kw):
        calls.append(dict(cc=cc_, dtype=dtype, D=head_dim, H=heads, N=n, direction=direction, **kw))
        r = answer(kw["word"], head_dim, n)
        if isinstance(r, _Refusal):
            raise r
        return _Sel(r, kw["word"])
    TP = types.SimpleNamespace(select=select, Refusal=_Refusal, TIER_WORDS=tier_words, STOCK_ROWS=frozenset({"cueq", "ds4sci", "sdpa", "stock"}), NEEDS_STOCK=frozenset({"cueq", "ds4sci", "exact_headsplit", "stock"}))
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(get_device_capability=lambda: cc))
    ns = {"torch": torch, "_TIER": {"word": tier}, "_count": lambda lever, w: counts.__setitem__((lever, w), counts.get((lever, w), 0) + 1),
          "_TRIATTN_PROV": {"mod": TP, "stack": "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90", "sel": {}, "refused": None}}
    exec(compile(_functions(["_triattn_word", "_triattn_select", "_triattn_bind", "_TRIATTN_STOCK_ROWS"]), AK_PATH, "exec"), ns)
    return ns, calls, counts


def test_the_mode_word_is_asked_at_every_head_dim_with_the_kit_form():
    for tier in ("fast", "big"):
        ns, calls, counts = _namespace(lambda word, D, N: "rowA" if D == 32 else "rowB", tier=tier)
        s32 = ns["_triattn_select"](448, 4, 32); s16 = ns["_triattn_select"](448, 4, 16)
        assert (s32.row, s32.word, s16.row, s16.word) == ("rowA", tier, "rowB", tier)
        assert [(c["word"], c["D"], c["H"], c["N"], c["dtype"], c["cc"], c["form"], c["direction"]) for c in calls] == [(tier, 32, 4, 448, "bf16", "9.0", "keypad", "fwd"), (tier, 16, 4, 448, "bf16", "9.0", "keypad", "fwd")]
        assert all(c["stack"] == ns["_TRIATTN_PROV"]["stack"] for c in calls) and not counts
        ns["_triattn_select"](448, 4, 32)                                   # memoised per (N, H, D, word): no second provider call
        assert len(calls) == 2


def test_a_refusal_binds_the_provider_named_fallback_row_and_counts_it():
    def answer(word, D, N):
        return _Refusal("no_prebuilt", "rowP", "rowT") if word == "fast" else ("rowT" if word == "rowT" else _Refusal("unknown", word, None))
    ns, calls, counts = _namespace(answer)
    s = ns["_triattn_select"](832, 4, 32)
    assert s.row == "rowT" and [c["word"] for c in calls] == ["fast", "rowT"]       # the fallback row is bound by the name the provider gave, nothing the kit chose
    assert counts == {("triattn", "refused:rowP:no_prebuilt"): 1} and ns["_TRIATTN_PROV"]["refused"] == "rowP:no_prebuilt"


def test_no_cell_or_a_stock_row_means_the_stock_statement_by_name():
    ns, calls, counts = _namespace(lambda word, D, N: _Refusal("no_cell:bf16_D32", word, "cueq"), cc=(10, 0))   # a card the provider has no cell for: its named fallback is a stock row
    assert ns["_triattn_select"](448, 4, 32) is None and [c["word"] for c in calls] == ["fast"]
    assert counts == {("triattn", "refused:fast:no_cell:bf16_D32"): 1}
    assert ns["_triattn_select"](448, 4, 32) is None and len(calls) == 1                 # memoised: the shape is not asked again
    ns2, calls2, counts2 = _namespace(lambda word, D, N: "sdpa")                         # the word resolved to a stock row: never a kernel this kit routes the trunk to
    assert ns2["_triattn_select"](448, 4, 32) is None and counts2 == {("triattn", "refused:sdpa:stock_row"): 1}


def test_no_provider_means_none():
    ns, calls, counts = _namespace(lambda word, D, N: "rowA")
    ns["_TRIATTN_PROV"]["mod"] = None
    assert ns["_triattn_select"](448, 4, 32) is None and not calls


def test_the_adapter_names_no_provider_row_and_reads_no_kit_cell_table():
    """The binding is by tier word only: no provider row name appears as a word the adapter asks, and the deleted cell table is not read."""
    src = _src()
    tri = src[src.index("def _triattn_word"):src.index("# lever 'transition'")]
    for row in ("triattn_native", "cuda_sm90a", "k2b", "\"flash\"", "'flash'", "exact_tier_row"):
        assert row not in tri, row
    body = src.split("def _apply_arch_cells")[1].split("def arch_info")[0]
    assert 'os.path.join(os.path.dirname(os.path.abspath(__file__)), "af3_arch_cells.json")' not in body and "_json.load(open(path))" not in body   # the table's loader is gone
    assert not os.path.exists(os.path.join(KIT, "opt", "forward", "af3t", "kernels", "af3_arch_cells.json"))
    assert re.search(r'word=sel\.word, selection=sel\)', tri)


def test_folded_names_are_not_levers():
    src = _src()
    levers = ast.literal_eval(re.search(r"^LEVERS = (\(.*?\))\s+#", src, flags=re.M).group(1))
    assert "triattn" in levers and "attn_epi" in levers and not {"k2b", "cuda_attn", "native_attn"} & set(levers)
    from af3_torch_opt import registry
    api = open(os.path.join(KIT, "opt", "forward", "af3t", "af3_torch", "af3_torch_api.py"), encoding="utf-8").read()
    fastest = ast.literal_eval(re.search(r'"fastest": (\(.*?\)),\n', api, flags=re.S).group(1))
    assert "triattn" in fastest and not {"k2b", "cuda_attn", "native_attn"} & set(fastest)
    for table in (registry.LEVERS, registry.IMPL, registry.STRATEGY, registry.EVIDENCE):
        assert not {"k2b", "cuda_attn", "native_attn"} & set(table)
    assert registry.IMPL["triattn"] == ("opt_core.kernels.triattn", "core") and set(registry.KERNEL_ROUTES["fpf_triatt_k2b"]["levers"]) == {"triattn"}
