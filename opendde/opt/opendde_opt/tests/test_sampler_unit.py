"""The SAMPLER unit (levers/SAMPLER; the diffusion-sampler kernel levers bound for OpenDDE): its table rows, switch words, ablation rules and carried packages — CPU only
(the kernels themselves are the core provider's Triton rows / prebuilt CUDA .so: their numerics are GPU-side and established on the kit image, not here)."""
import hashlib
import importlib
import json
import os
import sys

import pytest

from opendde_opt import modes, ran, registry

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
UNIT = os.path.join(OPT, "forward", "fast_inference", "levers", "SAMPLER")
TP = os.path.join(UNIT, "third_party")
LEVERS = ("dit_attn_apb", "atom_attn_apb", "cond_dedupe", "dit_fused", "dit_lowp", "atom_fused", "dit_attn_exact")


def test_unit_files_and_notices_are_carried():
    for rel in ("odde_sampler.py", "odde_apb_bind.py", "LICENSE", "NOTICE", "third_party/opendde_fpf_ditfast/NOTICE"):
        assert os.path.isfile(os.path.join(UNIT, rel)), rel
    for gone in ("third_party/opendde_fpf_apb", "third_party/opendde_fpf_dit_attn_exact", "third_party/opendde_fpf_ditfast/CELLS.json"):   # 0.2.57: the row-import binding, the exact-row binding and the
        assert not os.path.exists(os.path.join(UNIT, gone)), gone                                                                        # donor's measurement table are gone (the provider serves by word)


def test_every_unit_lever_has_registry_pin_ran_and_ablation_rows():
    assert tuple(modes.SAMPLER_LEVERS) == LEVERS
    for lv in LEVERS:
        assert lv in registry.LEVERS and registry.LEVERS[lv].kit == "fast_inference" and registry.LEVERS[lv].file.startswith(registry.SM), lv
        assert registry.PIN_STATUS[lv][0] == "tested", lv
        assert lv in ran.COUNTERS and lv not in ran.UNCOUNTED, lv          # every lever counted by its own live counter (odde_accel_v2.STATS sections)
        assert lv in modes.LEVER_SWITCHES and modes.LEVER_SWITCHES[lv], lv  # individually ablatable (MODEL_OPT_LEVERS_OFF)
    assert registry.LEVERS["dit_attn_exact"].tier == "exact"
    assert all(registry.LEVERS[lv].tier == "tier2" for lv in LEVERS if lv != "dit_attn_exact")


def test_switch_words_are_in_the_pool_and_the_probe_is_a_test_hook():
    for k in ("ODDE_DIT_ATTN_FP16", "ODDE_ATOM_ATTN", "ODDE_COND_DEDUPE", "ODDE_DIT_FUSED", "ODDE_DIT_LOWP", "ODDE_ATOM_FUSED", "ODDE_DIT_ATTN_EXACT"):
        assert k in modes.SAMPLER_SWITCHES and k in modes._ALL_SWITCHES, k
    assert "ODDE_SAMPLER_PROBE" in modes.TEST_HOOKS and "ODDE_SAMPLER_PROBE" in registry.KNOBS
    for ln in modes.LINES.values():
        assert "ODDE_SAMPLER_PROBE" not in ln.exports, ln.name


def test_dependents_leave_with_their_lever():
    ln = modes.LINES["LSTAR2A"]
    out = modes.line_without(ln, ("dit_attn_apb",))
    assert not {"dit_attn_apb", "dit_attn_fp16", "dit_fused", "dit_lowp"} & set(out.levers)
    for k in ("ODDE_DIT_ATTN", "ODDE_DIT_ATTN_FP16", "ODDE_DIT_FUSED", "ODDE_DIT_LOWP"):
        assert k not in out.exports and k in out.unset, k
    assert {"atom_attn_apb", "atom_fused", "cond_dedupe"} <= set(out.levers)
    out = modes.line_without(ln, ("atom_attn_apb",))
    assert not {"atom_attn_apb", "atom_fused"} & set(out.levers) and "dit_fused" in out.levers
    out = modes.line_without(ln, ("dit_lowp",))
    assert "dit_fused" in out.levers and "dit_lowp" not in out.levers and out.exports.get("ODDE_DIT_FUSED") == "1" and "ODDE_DIT_LOWP" not in out.exports
    out = modes.line_without(ln, ("cond_dedupe",))
    assert "cond_dedupe" not in out.levers and "ODDE_COND_DEDUPE" not in out.exports and "dit_fused" in out.levers


@pytest.fixture
def sampler_module(monkeypatch):
    for k in modes.SAMPLER_SWITCHES + ("ODDE_DIT_ATTN", "ODDE_SAMPLER_PROBE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(UNIT)
    sys.modules.pop("odde_sampler", None)
    m = importlib.import_module("odde_sampler")
    yield m
    sys.modules.pop("odde_sampler", None)


def test_requested_reads_the_kit_words(sampler_module, monkeypatch):
    m = sampler_module
    assert m.requested() == {}
    monkeypatch.setenv("ODDE_DIT_ATTN", "bf16")                       # the ACCEL lever's own word: not this unit's
    assert m.requested() == {}
    for k, v in modes._SP_FAST_EXPORTS.items():
        monkeypatch.setenv(k, v)
    assert set(m.requested()) == set(modes._SP_FAST_LEVERS)
    monkeypatch.setenv("ODDE_DIT_LOWP", "off")
    assert "dit_lowp" not in m.requested() and "dit_fused" in m.requested()
    monkeypatch.setenv("ODDE_DIT_ATTN", "exact")                                                # one switch, the word picks the site: exact -> the exact lever, never the apb lever
    assert "dit_attn_exact" in m.requested() and "dit_attn_apb" not in m.requested() and m.requested()["dit_attn_exact"] == "exact"
    monkeypatch.delenv("ODDE_DIT_ATTN"); monkeypatch.setenv("ODDE_DIT_ATTN_EXACT", "1")            # the pre-0.2.57 spelling still reads as the exact word
    assert m.requested().get("dit_attn_exact") == "exact"
    monkeypatch.setenv("ODDE_DIT_ATTN", "apb")                                                  # ... and `apb` as the fast word
    assert m.requested().get("dit_attn_apb") == "fast" and "dit_attn_exact" not in m.requested()
    monkeypatch.setenv("ODDE_DIT_ATTN", "fpf_apb:fp16")                                         # a row pin is the apb lever's word verbatim
    assert m.requested().get("dit_attn_apb") == "fpf_apb:fp16"


def test_words_that_cannot_compose_refuse_by_name(sampler_module, monkeypatch):
    m = sampler_module
    monkeypatch.setenv("ODDE_DIT_LOWP", "fp16")
    with pytest.raises(RuntimeError, match="dit_lowp: requires lever dit_fused"):
        m.install(object())
    monkeypatch.delenv("ODDE_DIT_LOWP"); monkeypatch.setenv("ODDE_DIT_FUSED", "1")
    with pytest.raises(RuntimeError, match="dit_fused: requires lever dit_attn_apb"):
        m.install(object())
    monkeypatch.setenv("ODDE_DIT_ATTN", "exact")                                                # the exact word is the exact lever's: the fused stack still has no attention site
    with pytest.raises(RuntimeError, match="dit_fused: requires lever dit_attn_apb"):
        m.install(object())
    monkeypatch.delenv("ODDE_DIT_FUSED"); monkeypatch.delenv("ODDE_DIT_ATTN"); monkeypatch.setenv("ODDE_ATOM_FUSED", "1")
    with pytest.raises(RuntimeError, match="atom_fused: requires lever atom_attn_apb"):
        m.install(object())
    monkeypatch.delenv("ODDE_ATOM_FUSED"); monkeypatch.setenv("ODDE_DIT_ATTN", "bogus_word")
    with pytest.raises(RuntimeError, match="dit_attn_apb: ODDE_DIT_ATTN=bogus_word: not a provider word"):
        m.install(object())


def test_lines_never_set_two_token_attention_routes():
    from opt_core.kernels import apb
    for ln in modes.LINES.values():
        e = ln.exports
        assert "ODDE_DIT_ATTN_EXACT" not in e and "ODDE_DIT_ATTN_FP16" not in e, ln.name           # the retired spellings: exported by no line
        w = e.get("ODDE_DIT_ATTN")
        if w is not None:
            assert w in apb.TIER_WORDS, (ln.name, w)                                                # a line binds the provider by a TIER word (row pins are for ablation by hand)
            assert (w == "exact") == (ln.tier == "exact"), (ln.name, w, ln.tier)                    # the exact word on the exact line only; fast | big on the tolerance lines
        if e.get("ODDE_DIT_FUSED"):
            assert w in ("fast", "big"), ln.name
        if e.get("ODDE_ATOM_FUSED"):
            assert e.get("ODDE_ATOM_ATTN") in ("fast", "big"), ln.name
        if e.get("ODDE_DIT_LOWP"):
            assert e.get("ODDE_DIT_FUSED") == "1", ln.name
        if ln.name in modes.BIG_LINES and w is not None:
            assert w == modes.SAMPLER_BIG_WORD and e.get("ODDE_ATOM_ATTN") in (None, modes.SAMPLER_BIG_WORD), ln.name   # the big lines say big (the provider's own peak-measured cells)


def test_carried_kernel_files_are_the_donor_bytes_where_declared():
    """The packages declare which files are the donor's bytes; when the donor tree is beside this kit (a full checkout), they must match."""
    donor = os.path.join(OPT, "..", "..", "protenix_v2", "opt", "forward", "flashpairformer", "third_party")
    if not os.path.isdir(donor):
        pytest.skip("the donor kit's tree is not beside this kit in this checkout")
    same = {"opendde_fpf_ditfast": ("protenix_fpf_ditfast", ["NOTICE"]),                          # 0.2.58: the row kernels are imported from opt_core.kernels.apb.ditfast (no kernel file carried)
            }                      # opendde_fpf_dit_attn_exact carries the binding only since 0.2.41 (kernel source + prebuilt live in opt_core.kernels.apb):
                                   # its LICENSE_NOTE.md is this kit's own attribution note, not the donor's bytes
    for mine, (theirs, files) in same.items():
        for f in files:
            a = open(os.path.join(TP, mine, f), "rb").read(); b = open(os.path.join(donor, theirs, f), "rb").read()
            assert a == b, f"{mine}/{f} differs from the donor's bytes"


def test_the_binding_imports_no_provider_row_and_carries_no_table():
    """0.2.57: the sampler's attention sites ask the provider's select() BY WORD and serve through its serving functions; no row module,
    prebuilt loader or cell table is imported / carried by the kit."""
    src = open(os.path.join(UNIT, "odde_apb_bind.py")).read()
    for banned in ("apb_triton", "atom_triton", "pf_triton", "carried_module(", "load_prebuilt", "ExtensionFileLoader", "CELLS"):
        assert banned not in src, banned
    for used in ("KA.select(", "KA.pair_bias_attention(", "KA.atom_attention(", "KA.TIER_WORDS", "KA.STOCK_ROWS"):
        assert used in src, used
    assert sorted(os.listdir(TP)) == ["opendde_fpf_ditfast"], os.listdir(TP)                     # the one third-party package left: the fused schedules (its row kernels are the shared copy's)
    pack = os.path.join(TP, "opendde_fpf_ditfast")
    for gone in ("kernels.py", "atom_kernels.py", "CELLS.json"):                                # 0.2.58: no kernel file or cell table in the package
        assert not os.path.exists(os.path.join(pack, gone)), gone
    for f, imp in (("dit_fast.py", "from opt_core.kernels.apb.ditfast import kernels as K"), ("atom_fast.py", "from opt_core.kernels.apb.ditfast import atom_kernels as K2")):
        assert imp in open(os.path.join(pack, f)).read(), (f, imp)                              # the schedules import the shared core's row kernels
    from opt_core.kernels.apb import ditfast as core_ditfast
    assert all(os.path.isfile(os.path.join(os.path.dirname(core_ditfast.__file__), f)) for f in ("kernels.py", "atom_kernels.py")), core_ditfast.__file__
    from opt_core.kernels import apb
    assert {"fast", "big", "exact"} <= set(apb.TIER_WORDS)
    for row in ("fpf_apb", "fpf_atom", "dit_exact"):
        assert row in apb.ROW_NAMES, row
    kit_stack = "torch2.7.1-cu126-sm90"                                                           # stock/PINS.json torch 2.7.1 + cu126 on sm_90: the exact line's row is vouched for it
    assert kit_stack in apb.DIT_EXACT_ABIS, (kit_stack, apb.DIT_EXACT_ABIS)


@pytest.fixture
def bind_module(monkeypatch):
    for k in modes.SAMPLER_SWITCHES + ("ODDE_DIT_ATTN", "ODDE_SAMPLER_PROBE"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.syspath_prepend(UNIT)
    for name in ("odde_apb_bind", "odde_sampler"):
        sys.modules.pop(name, None)
    B = importlib.import_module("odde_apb_bind")
    B.reset()
    yield B
    B.reset()
    for name in ("odde_apb_bind", "odde_sampler"):
        sys.modules.pop(name, None)


def test_dit_attn_exact_steps_aside_by_name_when_the_provider_names_the_stock_op(bind_module, monkeypatch):
    """Class contract (kit doctrine: a lever that cannot engage STEPS ASIDE BY NAME -- never silent, never a refusal): where the provider's exact
    tier names the STOCK op for the running card / stack (cc 8.0 today: no vouched kernel row) install_exact() raises nothing, installs nothing
    (the stock fp32 SDPA statement serves: exact by construction) and names the aside; the sampler unit records it under REPORT['aside'], not
    'installed'; the registry's engagement predicate reads it back as the LEVER row's reason (`state=off reason=aside:provider_exact_names_stock:...`)."""
    B = bind_module
    KA = B.KA
    monkeypatch.setattr(B.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(B, "_cc", lambda: (8, 0))
    asked = []

    def stock_naming_select(cc, dtype, cell, n_tokens, *, word, **kw):                           # the provider's answer on a card / stack with no vouched exact kernel row
        asked.append((cc, dtype, cell, n_tokens, word, kw.get("samples"), kw.get("stack")))
        return KA.Selection("sdpa", "auto", word, "8.0|fp32|dit_h16d48|S5|N<=800|eager|fwd", True, kw.get("stack"), True, "stock", None, True, True, 1.0, None,
                            "exact: no vouched kernel row on this stack: the stock op", None)
    monkeypatch.setattr(KA, "select", stock_naming_select)
    monkeypatch.setattr(KA, "stack_word", lambda *a, **k: "A100:torch2.7.1+cu126/3.3.1/cueq0.10.0")
    monkeypatch.setattr(KA, "dit_exact_abi", lambda: "torch2.7.1-cu126-sm80")
    monkeypatch.setenv("ODDE_DIT_ATTN", "exact")
    prim = sys.modules.get("opendde.model.modules.primitives")
    orig_attention = getattr(prim, "_attention", None)
    marker = B.install_exact()                                                                     # no raise
    assert marker.startswith("DITATTN:aside(provider_exact_names_stock:sdpa:auto@sm80") and "stock fp32 SDPA statement serves" in marker, marker
    assert asked and asked[0][4] == "exact" and asked[0][2] == "dit_h16d48" and asked[0][6] is not None       # asked by the exact WORD, at the token cell, WITH the running stack
    assert B.COUNTS["exact"]["aside"] == "provider_exact_names_stock:sdpa:auto@sm80" and B.COUNTS["exact"]["installed"] is False
    if prim is not None:
        assert prim._attention is orig_attention                                                   # nothing bound: the stock statement serves
    assert B.install_exact() == marker                                                             # idempotent: asked again, the same named aside
    monkeypatch.delenv("ODDE_DIT_ATTN"); monkeypatch.setenv("ODDE_DIT_ATTN_EXACT", "1")               # the pre-0.2.57 spelling reaches the same route
    assert B.word("dit") == "exact"
    # the sampler unit: recorded as an aside, not as installed, no raise
    S = importlib.import_module("odde_sampler")
    monkeypatch.setitem(sys.modules, "odde_sampler", S)
    rep = S.install(model=None)
    assert "dit_attn_exact" not in rep["installed"] and rep["aside"] == {"dit_attn_exact": "provider_exact_names_stock:sdpa:auto@sm80"}, rep
    # the registry's engagement predicate names it (ran.engagement -> LEVER state=off reason=aside:...)
    from opendde_opt import ran
    monkeypatch.setitem(sys.modules, "odde_apb_bind", B)
    engaged, why = ran.engagement("dit_attn_exact", {"det": True, "n_gpu": 1, "token_floors": {"item": 600}})
    assert engaged is False and why.startswith("aside:provider_exact_names_stock:sdpa:auto@sm80"), why


def test_apb_dit_attention_call_forms_the_rows_do_not_serve_step_aside_by_name_to_the_stock_forward(bind_module, monkeypatch):
    """Class contract (0.2.53, kept through the provider binding): the DiT token-attention replacement never raises on a call form -- a
    local-attention call (n_queries / trunked bias set), attn_bias=None, a per-sample [S,1,N,N] bias or another bias layout run the module's STOCK
    forward with the original arguments, counted per word (asides / aside_<word> on the lever's section, printed on the LEVER line); the registry's
    engagement predicate names a run whose every call stepped aside (never lever_never_ran). CPU: a tiny nn stub stands in for primitives.Attention;
    the provider is never reached."""
    import types
    import torch
    B = bind_module

    class StubAttention(torch.nn.Module):                                                      # the stock module's call surface (primitives.Attention.forward's signature)
        def __init__(self):
            super().__init__()
            self.num_heads, self.c_hidden, self.gating = 16, 48, True
            self.linear_q = torch.nn.Linear(8, 16 * 48, bias=False); self.linear_k = self.linear_v = self.linear_g = self.linear_q; self.linear_o = torch.nn.Linear(16 * 48, 8, bias=False)
            self.seen = []

        def forward(self, q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
            self.seen.append(dict(n_queries=n_queries, bias=None if attn_bias is None else tuple(attn_bias.shape), trunked=trunked_attn_bias is not None))
            return q_x * 2.0 + (0.0 if attn_bias is None else float(attn_bias.sum()))         # a recognisable stock output

    att = StubAttention()
    st = B.COUNTS["dit"]
    fwd = B._make_dit_forward(att)
    q = torch.randn(5, 32, 8)
    out = fwd(q, q, attn_bias=torch.zeros(1, 16, 32, 32), n_queries=32, n_keys=128)             # (a) a local-attention call form
    assert torch.equal(out, att(q, q, attn_bias=torch.zeros(1, 16, 32, 32), n_queries=32, n_keys=128)) and att.seen[-1]["n_queries"] == 32
    out = fwd(q, q, attn_bias=None)                                                              # (c) no pair bias
    assert torch.equal(out, q * 2.0) and att.seen[-1]["bias"] is None
    b = torch.ones(5, 1, 32, 32)
    assert torch.equal(fwd(q, q, attn_bias=b), att(q, q, attn_bias=b))                           # (b) a per-sample [S,1,N,N] bias
    assert torch.equal(fwd(q, q, attn_bias=torch.ones(1, 7, 32, 32)), att(q, q, attn_bias=torch.ones(1, 7, 32, 32)))   # (d) another bias layout
    kv = torch.randn(3, 32, 8)
    assert torch.equal(fwd(q, kv, attn_bias=torch.zeros(1, 16, 32, 32)), att(q, kv, attn_bias=torch.zeros(1, 16, 32, 32)))   # q / kv sample dims differ
    assert B.asides("dit") == {"local_attention": 1, "no_bias": 1, "per_sample_bias": 1, "bias_shape": 1, "qkv_sample_dims": 1} and st["calls"] == 0 and st["asides"] == 5
    assert set(B.asides("dit")) <= set(B.ASIDE_WORDS["dit"])
    # the atom replacement: a call that is not the windowed local form steps aside the same way
    att2 = StubAttention(); att2.num_heads, att2.c_hidden = 4, 32
    afwd = B._make_atom_forward(att2)
    assert torch.equal(afwd(q, q, attn_bias=torch.zeros(1, 4, 32, 32)), att2(q, q, attn_bias=torch.zeros(1, 4, 32, 32))) and B.asides("atom") == {"call_form": 1}
    # the kit side: the census rides in kit_stats()['accel_v2'] sections and the LEVER line prints it; a run whose every call stepped aside is named by the engagement predicate
    from opendde_opt import ran, report
    acc = types.ModuleType("odde_accel_v2"); acc.STATS = {"dit_attn": {"bf16_calls": 0}, "apb_state_dit": st, "apb_state_atom": B.COUNTS["atom"], "apb_launch": B.LAUNCHES}
    monkeypatch.setitem(sys.modules, "odde_accel_v2", acc)
    engaged, why = ran.engagement("dit_attn_apb", {"det": False, "n_gpu": 1, "token_floors": {"item": 600}})
    assert engaged is False and why.startswith("asides=bias_shape:1,local_attention:1,no_bias:1,per_sample_bias:1,qkv_sample_dims:1 ("), why
    ev = report.lever_evidence("dit_attn_apb", {"accel_v2": {"apb_state_dit": dict(st)}})
    assert ev.get("calls") == 0 and ev.get("asides") == "bias_shape:1,local_attention:1,no_bias:1,per_sample_bias:1,qkv_sample_dims:1", ev
    monkeypatch.setitem(st, "calls", 3); monkeypatch.setitem(st, "word", "fast"); monkeypatch.setitem(st, "rows", {"fpf_apb:fp16": 3})   # some calls served by a row: engaged; the LEVER line names the word and the rows
    assert ran.engagement("dit_attn_apb", {"det": False, "n_gpu": 1, "token_floors": {"item": 600}}) == (True, None)
    ev = report.lever_evidence("dit_attn_apb", {"accel_v2": {"apb_state_dit": dict(st)}})
    assert ev.get("word") == "fast" and ev.get("rows") == "fpf_apb:fp16:3" and ev.get("calls") == 3, ev


def test_selection_keys_a_dtype_without_a_measured_family_on_the_fp32_statement_cell_and_names_it(bind_module, monkeypatch):
    """The fused token stack's 16-bit activations (dit_lowp=fp16): the provider's DiT cells are keyed on the statement dtype; a dtype with no
    measured family asks the fp32 cell of the same geometry and the census notes it (never an unnamed fallback, never a stock op by surprise)."""
    B = bind_module
    KA = B.KA
    monkeypatch.setattr(B, "_cc", lambda: (9, 0))
    seen = []

    def sel(cc, dtype, cell, n_tokens, *, word, **kw):
        seen.append((dtype, cell, n_tokens, word, kw.get("samples"), kw.get("timing"), kw.get("stack")))
        return KA.Selection("fpf_apb", "fp16", word, "9.0|fp32|dit_h16d48|S5|N<=800|eager|fwd", True, None, True, "tolerance", None, False, True, 3.0, None, "fast winner", None)
    monkeypatch.setattr(KA, "select", sel)
    monkeypatch.setattr(KA, "cell_family", lambda cc, dtype, cell, timing="eager": {} if KA.dtype_word(dtype) == "fp16" else {(5, 800): "k"})
    B.COUNTS["dit"]["word"] = "fast"
    s1 = B.selection("dit", 5, 800, "fp16", False)
    assert seen[-1][:4] == ("fp32", "dit_h16d48", 800, "fast") and seen[-1][6] is None            # fp16 -> the fp32 statement cell; a fast word asks WITHOUT the stack (the provider's own convention)
    assert any(n.startswith("cell_dtype=fp32(for fp16 operands") for n in B.COUNTS["dit"]["notes"])
    assert B.selection("dit", 5, 800, "fp16", False) is s1 and len(seen) == 1                      # memoised per call class
    B.selection("dit", 5, 800, "fp16", True)
    assert seen[-1][5] == "graph"                                                                  # a capturing stream asks the graph-timed cells
    B.COUNTS["atom"]["word"] = "big"
    B.selection("atom", 5, 4096, "fp32", False)
    assert seen[-1][:5] == ("fp32", "atom_h4d32w32x128", 512, "big", 5)                          # the windowed cells are keyed on tokens ~ atoms // 8
