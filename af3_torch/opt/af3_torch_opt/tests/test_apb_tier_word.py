"""Lever apb binds the shared core's pair-bias attention provider (opt_core.kernels.apb) by the MODE's tier word (fast | big): the Pairformer's
and the diffusion transformer's attention core through its CORE op, the Pairformer's pair logits through its PRODUCER op, the sample-batched DTK
step's one-launch attention through the same face with the caller's word -- no kit-side row pin or cell table; a call class the word's row refuses
by name is served by the kit's own statement, counted, never silent; exact / off keep xfold's statements by name.
CPU-only: the adapter's helpers are compiled from af3_kernels.py's source against a mocked provider (no GPU, no Triton); the provider's own pure
selection is exercised on its shipped cell table for this kit's geometries on both cards."""
import ast
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
AK_PATH = os.path.join(KIT, "opt", "forward", "af3t", "kernels", "af3_kernels.py")
DTK_PATH = os.path.join(KIT, "opt", "forward", "dtk", "dtk_modules.py")


def _read(path):
    with open(path) as f:
        return f.read()


def _functions(names):
    src = _read(AK_PATH)
    tree = ast.parse(src)
    segs = [ast.get_source_segment(src, node) for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(segs) == len(names), names
    return "\n\n".join(segs)


class _Refusal(Exception):
    def __init__(self, kind, row=None, fallback=None):
        super().__init__(kind); self.kind, self.row, self.fallback = kind, row, fallback


class _Sel:
    def __init__(self, row, word, variant=None):
        self.row, self.word, self.variant = row, word, variant


class _T:
    """A stand-in tensor: the adapter only views / slices / reshapes what it hands the provider."""
    def __init__(self, tag, shape=None):
        self.tag, self.shape = tag, shape

    def unflatten(self, dim, sizes):
        return _T(self.tag + ".unflatten", None)

    def __getitem__(self, item):
        return _T(self.tag + "[]", None)

    def reshape(self, *shape):
        return _T(self.tag + ".reshape", tuple(shape))


def _namespace(row_by_word, refuse=(), graph_cells=True, capturing=False):
    """The apb helpers with a provider mock: pair_bias_attention records its kwargs and answers (out, Sel(row_by_word[word])) or raises Refusal for words in `refuse`."""
    calls, counts, logs = [], {}, []

    def cell_word(kind, heads=None, head_dim=None, c_z=None, op=None):
        if kind == "pf":
            return "pf_h16d24" if (heads, head_dim) == (16, 24) else None
        if kind == "dit":
            return "dit_h16d48" if (heads, head_dim) == (16, 48) else None
        if kind == "bias":
            return "bias_c%dh%d" % (c_z, heads)
        return None

    def pair_bias_attention(q, k, v, bias, key_mask=None, gate=None, **kw):
        calls.append(dict(kw, key_mask=key_mask, gate=gate))
        if kw["word"] in refuse:
            raise _Refusal("cell:refused_for_test", "fpf_apb", "sdpa")
        return _T("out", None), (kw["selection"] or _Sel(row_by_word[kw["word"]], kw["word"]))

    KA = types.SimpleNamespace(cell_word=cell_word, pair_bias_attention=pair_bias_attention, Refusal=_Refusal,
                               arm_word=lambda row, variant=None: row if not variant else "%s:%s" % (row, variant))
    ns = {"_TIER": {"word": "fast"}, "_APB_PROV": {"mod": KA, "tried": True, "sel": {}, "refused": None, "graph_cells": graph_cells},
          "_capturing": lambda: capturing, "_count": lambda lever, w: counts.__setitem__((lever, w), counts.get((lever, w), 0) + 1),
          "_log": logs.append}
    exec(compile(_functions(["set_graph_cells", "_apb_face", "provider_binding", "_apb_refused", "_apb_attention"]), AK_PATH, "exec"), ns)
    return ns, calls, counts, logs


def _attn(ns, H, Dh, N=448):
    return ns["_apb_attention"](_T("q"), _T("k"), _T("v"), _T("g"), _T("am"), N, H, Dh)


def test_the_pairformer_attention_asks_the_provider_for_the_modes_tier_word():
    ns, calls, counts, _ = _namespace({"fast": "sdpa", "big": "fpf_apb"})
    o = _attn(ns, 16, 24)
    assert o.shape == (448, 384), o.shape                                   # the served rows come back [N, H*Dh]
    c = calls[-1]
    assert (c["word"], c["cell"], c["layout"], c["capture"], c["key_mask"]) == ("fast", "pf_h16d24", "snhd", False, None), c
    assert abs(c["scale"] - 24 ** -0.5) == 0 and c["gate"] is not None and c["selection"] is None
    _attn(ns, 16, 24)                                                        # the class resolved once: the second call rides the cached Selection
    assert calls[-1]["selection"] is not None and calls[-1]["selection"].row == "sdpa"
    assert counts[("apb", "served:selfattn:c=384:sdpa")] == 2
    ns["_TIER"]["word"] = "big"                                            # big asks big -- never the fast word
    _attn(ns, 16, 24)
    assert calls[-1]["word"] == "big" and calls[-1]["selection"] is None and counts[("apb", "served:selfattn:c=384:fpf_apb")] == 1
    rows = ns["provider_binding"]()["rows"]
    assert rows == {"selfattn:pf_h16d24:S1:N448:fast:eager": "sdpa", "selfattn:pf_h16d24:S1:N448:big:eager": "fpf_apb"}, rows


def test_the_diffusion_transformer_geometry_asks_the_graph_replay_cells_when_the_sampler_captures():
    ns, calls, _, _ = _namespace({"fast": "fpf_apb"}, graph_cells=True)
    _attn(ns, 16, 48)
    assert (calls[-1]["cell"], calls[-1]["capture"]) == ("dit_h16d48", True)  # stepgraph: the warm-up steps already launch the row the capture replays
    ns["set_graph_cells"](False)                                             # hoist only (big): the eager cells
    _attn(ns, 16, 48)
    assert calls[-1]["capture"] is False and calls[-1]["selection"] is None
    _attn(ns, 16, 24)
    assert (calls[-1]["cell"], calls[-1]["capture"]) == ("pf_h16d24", False)  # the Pairformer follows the stream's actual state
    ns2, calls2, _, _ = _namespace({"fast": "fpf_apb"}, graph_cells=False, capturing=True)
    _attn(ns2, 16, 24)
    assert calls2[-1]["capture"] is True


def test_a_refusal_by_name_hands_the_call_class_to_the_kit_statement_once():
    ns, calls, counts, logs = _namespace({"fast": "fpf_apb"}, refuse=("fast",))
    assert _attn(ns, 16, 24) is None                                         # None: _selfattn_forward runs the kit's SDPA statement for this call
    assert counts[("apb", "refused:selfattn:fpf_apb:cell:refused_for_test")] == 1 and len(logs) == 1
    assert _attn(ns, 16, 24) is None and len(calls) == 1                     # the class stays with the kit statement: no second table walk, no second line
    assert _attn(ns, 16, 24, N=832) is None and len(calls) == 2              # another call class asks again
    b = ns["provider_binding"]()
    assert b["refused"] == "selfattn:fpf_apb:cell:refused_for_test" and b["rows"]["selfattn:pf_h16d24:S1:N448:fast:eager"] == "fpf_apb:cell:refused_for_test", b
    ns3, calls3, _, _ = _namespace({"fast": "fpf_apb"})
    ns3["_APB_PROV"]["mod"] = None                                           # no provider in the process: the kit statement, no call
    assert _attn(ns3, 16, 24) is None and not calls3
    assert ns3["_apb_attention"](_T("q"), _T("k"), _T("v"), _T("g"), None, 448, 16, 24) is None   # no bias to carry: the kit statement


def test_the_selfattention_hook_serves_the_kit_statement_only_where_the_provider_did_not():
    src = _read(AK_PATH)
    seg = src[src.index("def _selfattn_forward("):src.index("def _apb_planes(")]
    assert "o = _apb_attention(q, k, v, g, am, N, H, Dh)" in seg and "if o is None:" in seg and "F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=Dh ** -0.5)" in seg
    assert '_count(lever, "served:selfattn:c=%d" % (H * Dh))' in seg          # the kit statement's own census word is unchanged
    seg = src[src.index("def _pair_logits_fused("):src.index("def _pwa_msa_weights(")]
    assert "b16 = _apb_planes(block, c, pair, C, H)" in seg and "import lnl_fused as RFU" in seg   # the producer row first, the kit's fused kernel by name after a refusal
    seg = src[src.index("def _apb_planes("):src.index("def _pair_logits_fused(")]
    assert 'KA.pair_bias_planes(' in seg and 'word=word' in seg and 'out_layout="hij"' in seg and "out_dtype=torch.bfloat16" in seg
    assert "APB_PROVIDER_WORDS" not in src and "provider_word_for" not in src


def test_the_batched_dtk_step_asks_the_callers_tier_word_not_a_pinned_row():
    dtk = _read(DTK_PATH)
    assert "APB_PROVIDER_WORDS" not in dtk and "def provider_word_for" not in dtk          # no kit-side row pin / cell table
    seg = dtk[dtk.index("    def _attn_provider("):dtk.index("    def apb_rows(")]
    assert "word=self.apb_word" in seg and "capture=cap" in seg and "selection=sel" in seg and 'cell=cell' in seg
    assert "except KA.Refusal" in seg and "self.batched_attn = KA.arm_word(sel2.row, sel2.variant)" in seg
    assert "\n    apb_word = None " in dtk and "\n    apb_graph = None " in dtk
    fwd = _read(os.path.join(KIT, "opt", "af3_torch_opt", "forward.py"))
    assert "self.fused.apb_word = _provider_tier_word()" in fwd and 'dtk.fused.apb_graph = bool(getattr(model.diffusion_head, "use_step_graph", True))' in fwd
    assert "def _provider_tier_word()" in fwd and 'getattr(K, "provider_tier"' in fwd
    api = _read(os.path.join(KIT, "opt", "forward", "af3t", "af3_torch", "af3_torch_api.py"))
    assert 'K.set_graph_cells("stepgraph" in levers)' in api and "def provider_tier():" in api
    reg = _read(os.path.join(KIT, "opt", "af3_torch_opt", "registry.py"))
    assert '"apb": ("opt_core.kernels.apb:tier", "core")' in reg and '"apb_attn": {"levers": ("sbatch", "apb")' in reg


def test_the_provider_names_a_row_for_every_cell_this_kit_asks_on_both_cards():
    """The face's pure selection on its shipped cell table: the Pairformer (pf_h16d24, S1), the diffusion transformer serial (dit_h16d48, S1) and
    sample-batched (S5), the pair-logits producer (bias_c128h16), at the padded ladder sizes, eager and graph replay, fast and big, cc 9.0 and 8.0:
    a named row every time (an uncovered cell serves the nearest measured row and says so -- never a refusal); the exact word names the provider's
    stock SDPA floor on this stack (no row is byte-vouched against xfold's eager-softmax statement), which is why exact / off keep xfold's statements."""
    apb = pytest.importorskip("opt_core.kernels.apb")
    rows = set(apb.ROW_NAMES)
    for cc in ("9.0", "8.0"):
        for cell, S, hd in (("pf_h16d24", 1, 24), ("dit_h16d48", 1, 48), ("dit_h16d48", 5, 48)):
            for n in (448, 832, 1216):
                for word in ("fast", "big"):
                    for timing in ("eager", "graph"):
                        sel = apb.select(cc, "bf16", cell, n, word=word, samples=S, timing=timing, heads=16, head_dim=hd, capture=(timing == "graph"))
                        assert sel.row in rows and sel.word == word, (cc, cell, S, n, word, timing, sel)
                ex = apb.select(cc, "bf16", cell, n, word="exact", samples=S, timing="eager", heads=16, head_dim=hd)
                assert ex.row == "sdpa", (cc, cell, n, apb.arm_word(ex.row, ex.variant))
        for n in (448, 832, 1216):
            for word in ("fast", "big"):
                sel = apb.select(cc, "bf16", "bias_c128h16", n, word=word, c_z=128, heads=16)
                assert sel.row in rows, (cc, n, word, sel)
