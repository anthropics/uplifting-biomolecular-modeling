"""triattn_core — the fused block's attention core bound to opt_core.kernels.triattn by the MODE's tier word (CPU: no kernel launches): row
membership and order, the word per mode / the developer override, the picker's per-call words (the tier core string / index_2p31), the served-core
census read off attn.pair_fused, pair_cells' picker plumbing, install's named skips, the exit gate's expectations, no kit row table."""
import types
import pytest

from atlasfold_opt import modes, registry
from atlasfold_opt.hooks import triattn_core as K, pair_cells as PC

STACK_271 = "torch2.7.1+cu128-cpython-311-x86_64-linux-gnu-sm90"


class FakeLedger:
    def __init__(self): self.ev = []
    def serve(self, shape): self.ev.append(("serve", shape))
    def fallback(self, w): self.ev.append(("fallback", w))
    def error(self, w): self.ev.append(("error", w))


class Z:                                     # a stand-in for z4 [B, N, N, C]: only .shape is read
    def __init__(self, B, N, C=128): self.shape = (B, N, N, C)


MOD = types.SimpleNamespace(num_heads=4, channel_hidden=32)


def _fake_site(monkeypatch, block_installed=True):
    """A stand-in for atlasfold's triangle_update module when the stock package is not importable here: `cueq_tri_attn` + the two node classes
    (their forward carrying triatt_block's qualname marker when `block_installed`)."""
    import sys, types
    name = "atlasfold.model.network.primitives.triangle_update"
    try:
        import importlib; return importlib.import_module(name)
    except Exception:
        pass
    tu = types.ModuleType(name)
    def cueq_tri_attn(q, k, v, bias, mask, scale):
        return q.clone()
    tu.cueq_tri_attn = cueq_tri_attn
    for cls_name in ("TriangleAttentionStartingNode", "TriangleAttentionEndingNode"):
        def forward(self, z, mask=None, kernel_backend="cuequiv"):
            return z
        forward.__qualname__ = f"{cls_name}.forward[atlasfold_opt:triatt_block]" if block_installed else f"{cls_name}.forward"
        setattr(tu, cls_name, type(cls_name, (), {"forward": forward}))
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg = ".".join(parts[:i])
        if pkg not in sys.modules:
            m = types.ModuleType(pkg); m.__path__ = []; monkeypatch.setitem(sys.modules, pkg, m)
    monkeypatch.setitem(sys.modules, name, tu)
    return tu


def test_row_membership_and_order():
    fast = modes.MODES["fast"]
    assert fast.index("triattn_core") == fast.index("triatt_block") + 1 and "triattn_core" not in modes.MODES["exact"]
    assert "triattn_core" in modes.MODES["big"]
    row = registry.LEVERS["triattn_core"]
    assert row["cls"] == "fast" and row["requires"] == ("triatt_block",) and "opt_core.kernels.triattn" in row["provider"]
    from atlasfold_opt.hooks import installers
    assert installers()["triattn_core"] is K.install
    assert "triattn_core" not in PC.KERNELS                                                  # the provider routes its own carried kernels; the kit names none for the core


def test_the_word_is_the_modes_tier_word(monkeypatch):
    monkeypatch.delenv(K.WORD_ENV, raising=False)
    assert K.word("fast") == "fast" and K.word("big") == "big"                            # never the fast word under big
    monkeypatch.setenv(K.WORD_ENV, "cuda_sm90a"); assert K.word("fast") == "cuda_sm90a" and K.word("big") == "cuda_sm90a"   # a developer override, by name
    monkeypatch.setenv(K.WORD_ENV, "  "); assert K.word("big") == "big"
    T = pytest.importorskip("opt_core.kernels.triattn")
    assert {"fast", "big", "exact"} <= set(T.TIER_WORDS)                                    # the provider carries every mode word (opt_core >= 0.5.114)


def test_no_kit_row_table():
    for name in ("PREFER", "prefer_for", "PREFER_ENV", "MIN_TOKENS_ENV", "MIN_TOKENS", "DEFAULT_WORD", "Binding"):
        assert not hasattr(K, name), name                                                   # no preference table, no row floor, no kit-side selection: the provider's cells decide


def test_index_refusal_is_the_2p31_element_guard():
    assert K.index_refusal(1, 896, 4, 32) is None and K.index_refusal(1, 3072, 4, 32) is None
    assert K.index_refusal(1, 4096, 4, 32) == "index_2p31:1x4096"                            # q|k|v [1,4096,4,4096,32] = 2**31 elements
    assert K.index_refusal(2, 2900, 4, 32) == "index_2p31:2x2900" and K.index_refusal(2, 2048, 4, 32) is None


def test_picker_words_and_outcomes():
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    core = PF.TIER_CORE_PREFIX + "big"
    L = FakeLedger(); p = K.Picker(PF, L, core)
    assert p.min_tokens() == PC.MIN_TOKENS
    assert p.pick(MOD, Z(1, 512), False) == core and p.pick(MOD, Z(1, 1024), True) == core and L.ev == []   # a pick alone counts nothing: the block reports the outcome
    assert p.pick(MOD, Z(1, 4096), False) == PC.CORE and L.ev[-1] == ("fallback", "index_2p31:1x4096")       # by name: the provider default core serves that shape
    p.result(core, "served", None, "1x1024:end"); p.result(core, "served", None, "2x512:start"); p.result(core, "error", "RuntimeError", "s")
    assert L.ev[-3:] == [("serve", "1x1024:end"), ("serve", "2x512:start"), ("error", "RuntimeError")] and p.sizes == {512, 1024}
    plan = p.plan_line((9, 0))
    assert plan.startswith("512:") and ",1024:" in plan                                       # the provider's own resolution per served size (a row name or refused:<kind>), informational
    T = pytest.importorskip("opt_core.kernels.triattn")
    for tok in plan.split(","):
        n, row = tok.split(":", 1)
        assert row in T.ROW_NAMES or row.startswith("refused:"), tok


def test_served_core_census_is_read_off_the_provider():
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    saved = dict(PF.SERVED_CORES)
    try:
        PF.SERVED_CORES.clear()
        PF.SERVED_CORES.update({"tier:fast=triattn_native": 1000, "tier:fast=k2b": 100, "tier:fast=flash_triattn(refused:no_prebuilt:v11@x)": 84,
                                "default=tier:fast=triattn_native": 7, "tier:big=cuda_sm90a": 5})
        rows, refused = K.served_cores(PF, "fast")
        assert rows == {"triattn_native": 1000, "k2b": 100, "flash_triattn": 84} and refused == {"no_prebuilt:v11@x->flash_triattn": 84}
        assert K.served_cores(PF, "big") == ({"cuda_sm90a": 5}, {})                          # another word's / the default core's entries are not this lever's
    finally:
        PF.SERVED_CORES.clear(); PF.SERVED_CORES.update(saved)


def test_row_state_probes_are_informational_words():
    T = pytest.importorskip("opt_core.kernels.triattn")
    assert K.cuda_row_state(T, (8, 0), None) == "arch" and K.native_row_state(T, (7, 5), None) != "ok"
    s = K.cuda_row_state(T, (9, 0), STACK_271); assert s == "ok" or s.startswith("no_prebuilt"), s
    s = K.native_row_state(T, (9, 0), STACK_271); assert isinstance(s, str) and " " not in s


def test_pair_cells_picker_plumbing():
    try:
        assert PC.core_pick(MOD, Z(1, 1024), False) == PC.CORE                                # no picker: the provider default core, always
        core = "tier:fast"
        class P:
            def __init__(self): self.out = []
            def pick(self, m, z4, ending): return core if z4.shape[1] >= 512 else PC.CORE
            def result(self, c, outcome, word, shape): self.out.append((outcome, word, shape))
        p = P(); PC.set_core_picker(p)
        assert PC.core_pick(MOD, Z(1, 512), True) == core and PC.core_pick(MOD, Z(1, 256), True) == PC.CORE
        PC.core_result(core, "served", None, "1x512:end"); PC.core_result(PC.CORE, "served", None, "x")   # the default core's outcomes are the block's, not the picker's
        assert p.out == [("served", None, "1x512:end")]
        class Boom:
            def pick(self, *a): raise RuntimeError("x")
            def result(self, *a): raise RuntimeError("y")
        PC.set_core_picker(Boom()); assert PC.core_pick(MOD, Z(1, 1024), False) == PC.CORE    # a picker that raises never costs the call
        PC.core_result("tier:fast", "served", None, "s")
    finally:
        PC.set_core_picker(None)


def test_install_steps_aside_by_name_without_the_block():
    pytest.importorskip("torch"); pytest.importorskip("opt_core")
    try:
        import atlasfold.model.network.primitives.triangle_update  # noqa: F401
    except Exception:
        pytest.skip("stock atlasfold not importable here")
    ins = K.install("fast", "atlasfold-opt", {"mode": "fast", "det": 0})
    assert ins.applied is False and (ins.reason == "requires:triatt_block" or ins.reason.startswith("core:triattn_not_in_opt_core@"))


def test_install_binds_the_tier_core_word_per_mode(monkeypatch):
    pytest.importorskip("torch"); PF = pytest.importorskip("opt_core.attn.pair_fused")
    monkeypatch.delenv(K.WORD_ENV, raising=False)
    tu = _fake_site(monkeypatch)
    marked = []
    for cls_name in ("TriangleAttentionStartingNode", "TriangleAttentionEndingNode"):          # the block's marker on the node forwards (row order): the real module gets it for the test's duration
        f = getattr(tu, cls_name).forward
        if "triatt_block" not in f.__qualname__:
            def fwd(self, *a, **k): return None
            fwd.__qualname__ = f"{cls_name}.forward[atlasfold_opt:triatt_block]"; monkeypatch.setattr(getattr(tu, cls_name), "forward", fwd); marked.append(cls_name)
    try:
        for mode in ("fast", "big"):
            ins = K.install(mode, "[t]", {"mode": mode})
            assert ins.applied, ins.reason
            assert ins.facts["core"] == f"tier:{mode}" and ins.facts["word"] == mode and PC.core_pick(MOD, Z(1, 896), False) == f"tier:{mode}"
            line = ins.lines[0]()
            for tok in (f"name={K.NAME}", f"impl=opt_core.kernels.triattn@", f":{mode} ", f"core=tier:{mode}", "form=mask_bias", "index_limit=2147483648", "plan=none", "refused=none", "rows=none", f"word={mode}", f"min_tokens={PC.MIN_TOKENS}"):
                assert tok in line, (tok, line)
            assert " prefer=" not in line                                                    # no kit preference row on the census
            assert ins.gates[0]().ok                                                          # no calls: nothing to refuse
        monkeypatch.setenv(K.WORD_ENV, "nonsense")
        ins = K.install("fast", "[t]", {}); assert not ins.applied and ins.reason == "unknown_word:nonsense"
    finally:
        PC.set_core_picker(None)


def test_gate_expectations_by_prefix():
    words = registry.LEVERS["triattn_core"]["expected"]
    assert PC.expected("index_2p31:1x4096", words) and PC.expected("no_calls", words)
    assert not PC.expected("import:fpf_triatt_k2b", words) and not PC.expected("unknown_word:x", words) and not PC.expected("below_min_tokens", words)   # below the floor is the BLOCK's word, not the core's


def test_default_core_word_follows_the_provider_default():
    """Without the core lever the fused block runs pair_cells.CORE = the provider's `default` word — its own per-call-class table — never a
    literal cell name; the provider accepts it and the tier strings."""
    assert PC.CORE == "default" and PC.IMPL == "fpf" and PC.LN == "fused"
    PF = pytest.importorskip("opt_core.attn.pair_fused")
    assert PC.CORE == PF.DEFAULT_CORE and PF._core_ok("tier:fast") and PF._core_ok("tier:big")
    for cc in ("9.0", "8.0"):
        for n in (256, 512, 896, 1280):
            w = PF.default_core_for(cc, "bf16", 32, 4, n)
            assert w == "flash_triattn" or w.startswith("tier:"), (cc, n, w)                  # class contract: the flash cell or a tier word (which, and from which size, is the provider's table)
