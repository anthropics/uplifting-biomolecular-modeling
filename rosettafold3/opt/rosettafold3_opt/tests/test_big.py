"""The big mode on CPU: the composition and the flags (no torch), the window index arithmetic, the source derivation, and the two
chunked statements against their whole-tensor forms on synthetic modules (torch.equal on CPU: the chunk levers change no arithmetic per
element). The window-form prefix and the offload need the kit's patched classes and a CUDA device: they are exercised on a GPU only."""
import os
import sys
import types

import pytest

from rosettafold3_opt import _core, big, modes, registry as kit_registry


def _env(**kw):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ROSETTAFOLD3_BIG_")}
    env.update(kw)
    return env


def test_mode_table_and_line():
    """big = the fast row + every ported memory lever (one composition, parametric on the base); the FPF arm applied is fast's
    minus the components the memory levers disengage by lever property (the trunk CUDA graph as a memory holder); the fused triangle
    attention stays size-gated (triatt_chunk above big.TRI_GFLASH_GATE tokens) and the fused transition holds its site (transition_chunk
    site_owned:fpf_ttr)."""
    assert "big" in modes.MODES and modes.KIT_MODES["big"].switches == modes.BIG_SWITCHES == {**modes.KIT_MODES["fast"].switches, "RF3_CUDAGRAPH": "0"} and modes.KIT_MODES["fast"].switches == modes.KIT_MODES["exact"].switches
    assert modes.KIT_MODES["big"].fpf_arm is None and modes.KIT_MODES["big"].tree_state == "patched"   # the table entry is base-agnostic: kit_mode resolves the arm
    km = modes.kit_mode("big")
    assert big.BASE == "fast" and km.fpf_arm == big.fpf_arm_for(modes.KIT_MODES["fast"].fpf_arm)[0]
    arm, gone = big.fpf_arm_for(modes.FPF_ARM)
    base_parts = modes.FPF_ARM.partition("@")[0].split("+")
    assert set(gone) == ({"tg"} & set(base_parts)) and set(gone) <= set(big.DISENGAGED) and "gflash" not in big.DISENGAGED and "gflash" in arm.split("@")[0].split("+"), (arm, gone)
    reword = {"msa": "msa.pwa", "fast.fast": "fast.big", "apb.fast": "apb.big"}                       # UNIT_WORD (msa -> msa.pwa) + TIER_WORD (a provider's fast tier word -> its big tier word)
    assert arm == "+".join(reword.get(p, p) for p in base_parts if p not in gone)  and "ttr" in arm.split("@")[0].split("+") and "ttr" not in big.DISENGAGED and set(gone) == {"tg"}, (arm, gone)   # fast's fused pair transition STAYS in big's arm (faster, no memory cost); transition_chunk is then site_owned:fpf_ttr (ARM_SERVES)
    assert big.fpf_arm_for("fast.fast+apb.fast+xln@L1")[0] == "fast.big+apb.big+xln"                     # a provider's fast TIER word asks its big tier word in the memory row (TIER_WORD); row words (fast.v4, apb.sdpa) and other components ride unchanged
    assert big.fpf_arm_for("fast.v4+apb.sdpa@L1")[0] == "fast.v4+apb.sdpa"
    assert big.fpf_arm_for(None) == (None, {}) and big.fpf_arm_for("fast+apb@L1") == ("fast+apb", {})   # the lever step leaves with the graph (DISENGAGED_SWITCHES)
    table, line = big.compose(_env())
    assert table.modes == ("off", "exact", "fast", "big") and table.default == modes.DEFAULT_MODE
    assert line.base == "fast" and line.levers == big.LEVERS and line.base_rule == "fast-else-exact"   # the core's own rule names fast: no override
    assert big.LEVERS == ("triatt_chunk", "transition_chunk", "atom_pair_local", "opm_chunk", "cond_chunk", "confidence_offload", "feature_park")
    assert set(big.NEUTRAL_VS_FAST) < set(big.LEVERS)
    km_exact = modes.big_mode("exact")                                                             # the parametric form: the graph arm minus the trunk graph (a memory holder)
    assert km_exact.fpf_arm == "sapb+xmul.eager+xln+smsa" and km_exact.switches == {**modes.KIT_MODES["exact"].switches, "RF3_CUDAGRAPH": "0"} and km_exact.tree_state == "patched"
    assert km_exact.kit_levers == tuple(lv for lv in modes.KIT_MODES["exact"].kit_levers if lv not in big.DISENGAGED_KIT) == ("confhoist", "hostlean", "prefetch", "awrite")   # exact's package levers minus the memory holders (xtr): the confidence-head hoist and the non-forward levers ride every base
    with pytest.raises(ValueError):
        modes.big_mode("off")


def test_modes_literal_is_validated_not_computed(monkeypatch):
    """B16: modes.MODES is a literal tuple carrying "big" (an external reader regexes it); compose validates it against the core's composition."""
    import ast, inspect
    src = inspect.getsource(modes)
    node = next(n for n in ast.parse(src).body if isinstance(n, (ast.Assign, ast.AnnAssign)) and getattr(getattr(n, "targets", [n])[0] if isinstance(n, ast.Assign) else n.target, "id", None) == "MODES")
    val = node.value
    assert isinstance(val, ast.Tuple) and [ast.literal_eval(e) for e in val.elts] == ["off", "exact", "fast", "big"]
    assert not (set(big.LEVERS) & set(modes.MODES))              # B14: levers are never modes
    monkeypatch.setattr(modes, "MODES", ("off", "exact", "fast"))
    with pytest.raises(big.BigError):
        big.compose(_env())


def test_autoload_modes_match():
    from rosettafold3_opt import _autoload
    assert tuple(_autoload.MODES) == tuple(modes.MODES)


def test_selection_is_the_whole_line_with_explicit_switches():
    """The line is one lever set: the selection reads no lever word from the environment (one there is an undeclared name, refused at start);
    the only switches are the site-owned levers under n_gpu > 1, passed explicitly; the partial-unit opt-out is the environment route's word."""
    arm_served = set(big.site_owned(1))                                                                         # n_gpu 1: only the levers whose site a component of the applied arm serves (ARM_SERVES: transition_chunk under ttr)
    assert arm_served == {lv for lv, (comp, _why) in big.ARM_SERVES.items() if comp in modes.kit_mode("big").fpf_arm.partition("@")[0].split("+")}
    sel = big.selection(_env())
    assert sel.levers == tuple(lv for lv in big.LEVERS if lv not in arm_served) and not sel.refusals and set(sel.off_by_flag) == arm_served \
        and sel.flags == big.switches_for(1) == {lv: False for lv in arm_served} and sel.allow_partial is False
    sel = big.selection(_env(ROSETTAFOLD3_BIG_COND_CHUNK="0", ROSETTAFOLD3_BIG_OPM_CHUNK_ROWS="128"))       # not read: the selection is unchanged
    assert sel.levers == tuple(lv for lv in big.LEVERS if lv not in arm_served) and set(sel.off_by_flag) == arm_served and not sel.refusals
    sel = big.selection(_env(), n_gpu=2)
    assert set(sel.off_by_flag) == set(big.site_owned(2)) and sel.flags == big.switches_for(2) and not sel.refusals
    assert big.selection(_env(**{big.ALLOW_PARTIAL_ENV: "1"})).allow_partial is True


def test_kit_registry_names_the_big_levers():
    for lv in big.LEVERS:
        e = kit_registry.LEVERS["big_" + lv]
        assert e.switch == f"{big.PREFIX}_BIG_{lv.upper()}" and e.probe == ("big", lv)


def test_state_before_install():
    assert big.state() is None


torch = pytest.importorskip("torch")
from rosettafold3_opt import levers  # noqa: E402


def _ref_windows(L, qbatch=32, kbatch=128):
    nq = (L + qbatch - 1) // qbatch
    Cs = torch.arange(nq) * qbatch + qbatch // 2
    iQ = torch.clamp(Cs[:, None] + (torch.arange(qbatch) - qbatch // 2)[None, :], 0, L - 1)
    iK = torch.clamp(Cs[:, None] + (torch.arange(kbatch) - kbatch // 2)[None, :], 0, L - 1)
    return iQ, iK


@pytest.mark.parametrize("L", [1, 31, 32, 33, 100, 257])
def test_windows_match_the_atom_transformer_arithmetic(L):
    iQ, iK = levers.windows(L, torch.device("cpu"))
    rQ, rK = _ref_windows(L)
    assert torch.equal(iQ, rQ) and torch.equal(iK, rK)
    assert iQ.shape == ((L + 31) // 32, 32) and iK.shape == ((L + 31) // 32, 128)
    assert torch.equal(iQ[:, :][iQ < L].unique(), torch.arange(L))       # every atom is a query exactly where the stock puts it


def test_derive_replaces_one_statement():
    mod = types.ModuleType("synthetic_mod")
    src = "class K:\n    def m(self, x):\n        y = x + 1\n        return y * 2\n"
    exec(src, mod.__dict__)
    mod.__file__ = __file__
    import inspect, linecache
    lines = src.splitlines(True)
    linecache.cache["<syn>"] = (len(src), None, lines, "<syn>")
    code = compile(src, "<syn>", "exec")
    ns = {}
    exec(code, ns)
    K = ns["K"]
    mod.K = K
    mod.__dict__.update(ns)
    fn = levers._derive(K, "m", "        y = x + 1\n", "        y = x + 10\n", mod)
    assert fn(None, 1) == 22 and K.m(None, 1) == 4
    assert getattr(fn, levers.DERIVED_ATTR) == {"cls": "K", "method": "m", "old": "        y = x + 1\n", "new": "        y = x + 10\n"}
    with pytest.raises(levers.RefusalError):
        levers._derive(K, "m", "        nothing here\n", "x", mod)
    orig_m = K.m
    K.m = fn                                                                       # the lever applied: the class now carries the derived function (no source file)
    try:
        with pytest.raises(levers.RefusalError, match="already the function this kit derived"):   # a second derivation is refused BY NAME, never an OSError
            levers._derive(K, "m", "        y = x + 1\n", "        y = x + 10\n", mod)
        ns2 = {}
        exec(compile("def m(self, x):\n    return x\n", "<nowhere>", "exec"), ns2)        # a function with no readable source at all: refused by name too
        K.m = ns2["m"]
        with pytest.raises(levers.RefusalError, match="cannot be read"):
            levers._derive(K, "m", "        y = x + 1\n", "        y = x + 10\n", mod)
    finally:
        K.m = orig_m


class _OPM(torch.nn.Module):
    """OuterProductMean_AF3's statements (outer_product.py) on small widths."""

    def __init__(self, c_in=6, c_outer=3, c_out=5):
        super().__init__()
        self.norm = torch.nn.LayerNorm(c_in)
        self.proj_left = torch.nn.Linear(c_in, c_outer, bias=False)
        self.proj_right = torch.nn.Linear(c_in, c_outer, bias=False)
        self.proj_out = torch.nn.Linear(c_outer * c_outer, c_out)

    def forward(self, msa):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = torch.einsum("bsli,bsmj->blmij", left, right).reshape(B, L, L, -1)
        return self.proj_out(out)


@pytest.mark.parametrize("rows", [1, 3, 7, 64])
def test_opm_chunked_equals_whole(rows):
    torch.manual_seed(0)
    m = _OPM()
    msa = torch.randn(1, 4, 11, 6)
    levers.STATE["opm_rows"] = rows
    levers.STATE["chunk_entries"].clear()
    with torch.no_grad():
        whole = m(msa)
        chunked = levers.opm_forward_chunked(m, msa)
    assert chunked.shape == whole.shape and chunked.dtype == whole.dtype
    if rows >= 11:
        assert torch.equal(chunked, whole)                        # one block: the statement's own output
    else:                                                          # row blocks: the GEMM at another M — ULP-level on CPU (the box's torch.equal decides the GPU label)
        assert torch.allclose(chunked, whole, rtol=1e-5, atol=1e-6), (chunked - whole).abs().max()
    e = levers.STATE["chunk_entries"][-1]
    assert e["lever"] == "opm_chunk" and e["n_chunks"] == (1 if rows >= 11 else -(-11 // rows))


class _Cond(torch.nn.Module):
    """DiffusionConditioning's pair statements (the kit's _pair): cat[Z.float(), relpos] -> to_zii -> two transitions."""

    def __init__(self, c_z=4, c_rel=3, c_out=5):
        super().__init__()
        self.rel = torch.nn.Linear(2, c_rel)
        self.to_zii = torch.nn.Sequential(torch.nn.LayerNorm(c_z + c_rel), torch.nn.Linear(c_z + c_rel, c_out, bias=False))
        self.transition_1 = torch.nn.ModuleList([torch.nn.Sequential(torch.nn.LayerNorm(c_out), torch.nn.Linear(c_out, c_out)) for _ in range(2)])

    def relative_position_encoding(self, f):
        return self.rel(f["rel"])

    def pair(self, f, Z):
        Z_II = torch.cat([Z.float(), self.relative_position_encoding(f)], dim=-1)
        Z_II = self.to_zii(Z_II)
        for b in range(2):
            Z_II = Z_II + self.transition_1[b](Z_II)
        return Z_II


@pytest.mark.parametrize("rows", [1, 4, 5, 512])
def test_cond_chunked_equals_whole(rows):
    torch.manual_seed(1)
    m = _Cond()
    I = 9
    f = {"rel": torch.randn(I, I, 2)}
    Z = torch.randn(I, I, 4, dtype=torch.bfloat16)
    levers.STATE["cond_rows"] = rows
    with torch.no_grad():
        whole = m.pair(f, Z)
        chunked = levers.pair_chunked(m, f, Z, lambda: m.pair(f, Z))
    assert chunked.shape == whole.shape and chunked.dtype == whole.dtype
    if rows >= I:
        assert torch.equal(chunked, whole)
    else:
        assert torch.allclose(chunked, whole, rtol=1e-5, atol=1e-6), (chunked - whole).abs().max()
    with torch.no_grad():                                          # a batched trunk: the named fallback runs the stock pair
        rec = _FakeRecord()
        levers.STATE["ctx"] = types.SimpleNamespace(record=rec)
        out = levers.pair_chunked(m, f, Z[None], lambda: "stock")
        assert out == "stock" and rec.fallbacks and rec.fallbacks[0][0] == "cond_chunk"
        levers.STATE["ctx"] = None


class _FakeRecord:
    def __init__(self):
        self.marks, self.fallbacks = [], []

    def mark(self, lever, unit=None, detail=None):
        self.marks.append((lever, detail))

    def fallback(self, lever, reason, unit=None):
        self.fallbacks.append((lever, reason))


def test_chunk_summary_shape():
    s = levers.chunk_summary()
    assert set(s) >= {"calls", "kept"} and s["calls"] >= 1


def _ref_triangle_attention(query, key, value, bias, scale):
    """A pure-torch cuEquivariance stand-in: ``[b, n, h, s, d]`` q/k/v, bias ``[b, 1, h, s, s]``, softmax over the key axis per (row n, query s)."""
    torch = pytest.importorskip("torch")
    logits = torch.einsum("bnhqd,bnhkd->bnhqk", query, key) * scale + bias
    return torch.einsum("bnhqk,bnhkd->bnhqd", torch.softmax(logits, dim=-1), value)


class _TriAttn:
    """The stock TriangleAttention's members (attention.py) on CPU."""
    def __init__(self, torch, c=16, h=2, d=8, start_node=True):
        nn = torch.nn
        self.norm = nn.LayerNorm(c); self.to_q = nn.Linear(c, h * d, bias=False); self.to_k = nn.Linear(c, h * d, bias=False); self.to_v = nn.Linear(c, h * d, bias=False)
        self.to_b = nn.Linear(c, h, bias=False); self.to_g = nn.Linear(c, h * d); self.to_out = nn.Linear(h * d, c)
        for m in (self.to_q, self.to_k, self.to_v, self.to_b, self.to_g, self.to_out):
            nn.init.normal_(m.weight, std=0.3)
        self.scaling = 1 / d ** 0.5; self.h = h; self.dim = d; self.start_node = start_node; self.use_cuequivariance = True

    def forward_stock(self, pair):
        """attention.py: forward + _forward_cuequivariance, the cuEq call replaced by the stand-in."""
        from einops import rearrange
        torch = pytest.importorskip("torch")
        pair = self.norm(pair); bias = self.to_b(pair)
        if not self.start_node:
            pair = rearrange(pair, "b i j d -> b j i d")
        gate = torch.sigmoid(self.to_g(pair))
        query = rearrange(self.to_q(pair), "b i j (h d) -> b i h j d", h=self.h); key = rearrange(self.to_k(pair), "b i k (h d) -> b i h k d", h=self.h)
        value = rearrange(self.to_v(pair), "b i k (h d) -> b i h k d", h=self.h); bias_cueq = rearrange(bias, "b i j h -> b 1 h i j")
        out = rearrange(_ref_triangle_attention(query, key, value, bias_cueq, self.scaling), "b i h j d -> b i j (h d)")
        out = gate * out
        if not self.start_node:
            out = rearrange(out, "b i j d -> b j i d")
        return self.to_out(out)


@pytest.mark.parametrize("start_node", [True, False])
def test_triatt_chunked_matches_stock_statement(monkeypatch, start_node):
    """The lean line's triatt_chunk: the core's two-pass triangle_attention_chunked with the stock module's parts equals the stock statement
    (both nodes; the ending node's bias orientation is the risk) on CPU with a pure-torch stand-in for the cuEquivariance kernel."""
    torch = pytest.importorskip("torch"); pytest.importorskip("einops")
    fake = types.ModuleType("rf3.model.layers.attention"); fake.SHOULD_USE_CUEQUIVARIANCE = True
    fake.cuet = types.SimpleNamespace(triangle_attention=lambda q, k, v, bias, scale: _ref_triangle_attention(q, k, v, bias, scale))
    monkeypatch.setitem(sys.modules, "rf3.model.layers.attention", fake)
    torch.manual_seed(0)
    mod = _TriAttn(torch, start_node=start_node)
    pair = torch.randn(1, 40, 40, 16)
    with torch.no_grad():
        ref = mod.forward_stock(pair)
        monkeypatch.setitem(levers.STATE["orig"], "tri_forward", _TriAttn.forward_stock)
        monkeypatch.setitem(levers.STATE, "tri_rows", 16)
        marks = []
        monkeypatch.setattr(levers, "mark", lambda lever, what: marks.append((lever, what)))
        monkeypatch.setattr(levers, "chunk_record", lambda *a, **k: None)
        out = levers.tri_forward_chunked(mod, pair)
        assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5, rtol=1e-5), (out - ref).abs().max()
        assert marks and marks[-1][0] == levers.L_TRI and ("start" if start_node else "end") in marks[-1][1]
        # at or below the block size the stock forward runs (named as whole)
        marks.clear()
        small = torch.randn(1, 12, 12, 16)
        assert torch.equal(levers.tri_forward_chunked(mod, small), mod.forward_stock(small)) and "whole" in marks[-1][1]


def test_every_line_lever_has_a_hook_entry(monkeypatch):
    """Every lever of the composition has an entry in big._hooks(); a lever without a 'module' hook (e.g. triatt_chunk) would be refused at apply."""
    monkeypatch.setattr(big, "_mod", lambda name: None)
    hooks = big._hooks()
    for lv in big.LEVERS:
        assert lv in hooks and isinstance(hooks[lv], dict) and hooks[lv], lv


def test_triatt_apply_swaps_the_forward_and_undoes(monkeypatch):
    """The lever's apply path on a core Ctx (the per-lever hook table the trigger reads is ctx.hooks[<lever>],
    not ctx.hooks['module']): the class's forward is swapped for tri_forward_chunked, the rows setting read, undo restores the original."""
    mem = big._mem()
    class Fake:
        def forward(self, pair): return "stock"
        def _forward_cuequivariance(self, pair, bias): return "stock"
    orig = Fake.forward
    ctx = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRI: {"module": Fake}}, settings={levers.L_TRI: {"rows": 256}}, environ=_env(), graphs=True, extra={})
    assert levers._tri_applies(ctx) is None
    applied = levers._tri_apply(ctx)
    try:
        assert Fake.forward is levers.tri_forward_chunked and applied.settings == {"rows": 256} and levers.STATE["tri_rows"] == 256
    finally:
        applied.undo()
    assert Fake.forward is orig
    # under an FPF arm that engages the fused kernel the lever applies with the arm's gate policy; no gate set = gate 0 = the fused kernel never holds the site
    ctx2 = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRI: {"module": Fake}}, settings={}, environ=_env(), graphs=True, extra={"fpf_arm": "fast+gflash@L1"})
    assert levers._tri_applies(ctx2) is None
    # with the size gate set (big.TRI_GFLASH_GATE: 0 today = row-block / stock at every size) the lever applies under the same arm and reports the policy
    ctx3 = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRI: {"module": Fake}}, settings={levers.L_TRI: {"gate": big.TRI_GFLASH_GATE}}, environ=_env(), graphs=True, extra={"fpf_arm": "fast+gflash@L1"})
    assert levers._tri_applies(ctx3) is None
    applied = levers._tri_apply(ctx3)
    try:
        assert Fake.forward is levers.tri_forward_chunked and applied.settings == {"rows": 512, "gate": big.TRI_GFLASH_GATE, "below_gate": "gflash"} and levers.STATE["tri_gate"] == big.TRI_GFLASH_GATE
    finally:
        applied.undo(); levers.STATE["tri_gate"] = 0


def test_transition_chunked_matches_stock_statement(monkeypatch):
    """The lean line's transition_chunk: chunk_rows over the leading dim of a [I, J, c] pair equals the stock Transition statement (allclose on
    CPU: the GEMM at another M), the stock forward whole below the block, and the apply path swaps / undoes the class forward."""
    torch = pytest.importorskip("torch")
    nn = torch.nn
    class Transition(nn.Module):
        def __init__(self, n, c):
            super().__init__(); self.layer_norm_1 = nn.LayerNorm(c); self.linear_1 = nn.Linear(c, n * c, bias=False); self.linear_2 = nn.Linear(c, n * c, bias=False); self.linear_3 = nn.Linear(n * c, c, bias=False)
        def forward(self, X):
            X = self.layer_norm_1(X); A = self.linear_1(X); B = self.linear_2(X); return self.linear_3(torch.nn.functional.silu(A) * B)
    torch.manual_seed(0); mod = Transition(4, 16); X = torch.randn(40, 24, 16)
    mem = big._mem()
    ctx = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRANS: {"module": Transition}}, settings={levers.L_TRANS: {"rows": 16}}, environ=_env(), graphs=True, extra={})
    assert levers._trans_applies(ctx) is None
    with torch.no_grad():
        ref = Transition.forward(mod, X)
        applied = levers._trans_apply(ctx)
        try:
            marks = []
            monkeypatch.setattr(levers, "mark", lambda lever, what: marks.append((lever, what))); monkeypatch.setattr(levers, "chunk_record", lambda *a, **k: None)
            out = mod(X)
            assert out.shape == ref.shape and torch.allclose(out, ref, atol=1e-5, rtol=1e-5) and marks[-1] == (levers.L_TRANS, "rows=16")
            small = torch.randn(8, 24, 16); assert torch.equal(mod(small), Transition.forward(mod, small)) and "whole" in marks[-1][1]
            batched = torch.randn(1, 40, 24, 16); assert torch.allclose(mod(batched), Transition.forward(mod, batched), atol=1e-5, rtol=1e-5)
        finally:
            applied.undo()
    assert Transition.forward.__name__ == "forward" and Transition.forward is not levers.transition_forward_chunked
    ctx2 = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRANS: {"module": Transition}}, settings={}, environ=_env(), graphs=True, extra={"fpf_arm": "fast+ttr@L1"})
    r = levers._trans_applies(ctx2); assert r is not None and getattr(r, "precondition", None) == "site_owned"



def test_site_owned_is_by_arm_component_not_by_arm_presence():
    """transition_chunk refuses when the APPLIED arm engages its site (ttr); triatt_chunk applies under a gflash arm with the arm's gate policy
    (gate 0 = the fused kernel never holds the site) and otherwise leaves the stock forward, so it applies on every base."""
    mem = big._mem()
    class Fake:
        def forward(self, pair): return "stock"
        def _forward_cuequivariance(self, pair, bias): return "stock"
    for arm, tri_refused, trans_refused in (("fast+gflash+ttr+apb+tg+res@L1", False, True), ("fast+apb+res@L1", False, False), (None, False, False), ("fast+sapb+ttr@L1", False, True)):
        ctx = mem.Ctx(prefix=big.PREFIX, tag="t", framework="torch", hooks={levers.L_TRI: {"module": Fake}, levers.L_TRANS: {"module": Fake}}, settings={}, environ=_env(), graphs=True, extra={"fpf_arm": arm})
        r_tri = levers._tri_applies(ctx); r_trans = levers._trans_applies(ctx)
        assert (getattr(r_tri, "precondition", None) == "site_owned") == tri_refused, (arm, r_tri)
        assert (getattr(r_trans, "precondition", None) == "site_owned") == trans_refused, (arm, r_trans)


def test_feature_park_round_trip_on_the_cpu_line():
    """The park / unpark bookkeeping on the library's CPU test line: the cast originals parked when referenced elsewhere, the per-recycle
    tensor parked between steps and returned before each, every trunk-only tensor parked after the trunk; values byte-identical."""
    torch = pytest.importorskip("torch")
    from opt_core.mem import offload as OFF

    class Trunk:
        def __init__(self): self.n = 0
        def trunk_forward_with_recycling(self, f, n_recycles):
            for i in range(n_recycles):
                f["msa"] = f["msa_stack"][i]
                self.n += 1
                yield {"Z": f["msa"].float().sum() + f["distogram_condition"].sum()}

    class Model(Trunk):
        def parameters(self): return iter([torch.zeros(1)])
        def forward(self, input):
            outs = list(self.trunk_forward_with_recycling(input["f"], 3))
            return outs[-1]["Z"]

    n = 300                                                   # 300 x 300 x 64 fp32 = 23 MB per tensor (above FP_MIN_BYTES with 4 stacks)
    orig_stack = torch.randn(3, 8, n, 64); dc = torch.randn(n, n, 64) * 0 + 1.5
    keep = orig_stack                                         # the engine's own reference (the batch it keeps)
    f = {"msa_stack": orig_stack, "distogram_condition": dc, "has_distogram_condition": torch.ones(n, n), "small": torch.zeros(4)}
    ref = Model().forward({"f": {k: v.clone() for k, v in f.items()}})
    levers.FP_MIN_BYTES_SAVED = levers.FP_MIN_BYTES
    levers.FP_MIN_BYTES = 256 << 10                           # the test's threshold: the cast msa_stack (921 KB), its slice (307 KB) and has_distogram (360 KB) qualify; `small` does not
    levers.STATE["fp_settings"] = OFF.Settings(pin_max_gb=4.0, min_tokens=0, cols="rowloop", device="cpu")
    levers.STATE["fp_pool"] = OFF.PinPool(levers.STATE["fp_settings"])
    levers.STATE["fp"] = None
    levers.STATE["fp_counters"] = {"items": 0, "orig_parked": 0, "orig_dropped": 0, "step_parks": 0, "post_parks": 0, "unparks": 0, "refused": 0, "h2d_gb": 0.0, "d2h_gb": 0.0, "steps": 0}
    levers.STATE["orig"]["fp_forward"] = Model.forward; levers.STATE["orig"]["fp_trunk"] = Trunk.trunk_forward_with_recycling
    Model.forward = levers.fp_forward; Trunk.trunk_forward_with_recycling = levers.fp_trunk
    marks = []; unit_marks = []
    levers.STATE["ctx"] = type("C", (), {"record": type("R", (), {"fallback": staticmethod(lambda *a, **k: marks.append(a)), "mark": staticmethod(lambda lever, detail=None: unit_marks.append((lever, detail or "")))})()})()
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = Model().forward({"f": f})
        with torch.autocast("cpu", dtype=torch.bfloat16):
            ref2 = Model().forward({"f": {"msa_stack": keep.clone(), "distogram_condition": dc.clone(), "has_distogram_condition": torch.ones(n, n), "small": torch.zeros(4)}})
    finally:
        Model.forward = levers.STATE["orig"]["fp_forward"]; Trunk.trunk_forward_with_recycling = levers.STATE["orig"]["fp_trunk"]
        levers.FP_MIN_BYTES = levers.FP_MIN_BYTES_SAVED; levers.STATE["ctx"] = None
    c = levers.STATE["fp_counters"]
    assert c["items"] == 2 and c["steps"] == 6 and c["refused"] == 0 and not marks, (c, marks)
    assert [m[0] for m in unit_marks] == [levers.L_FP, levers.L_FP] and "steps=3" in unit_marks[0][1], unit_marks   # the unit mark per item (the census counts the lever as ran)
    assert set(c["feature_census"]) == {"msa_stack", "distogram_condition", "has_distogram_condition"}, c["feature_census"]
    assert c["step_parks"] == 6 and c["unparks"] == 6 and c["post_parks"] == 8 and c["orig_parked"] == 1 and c["orig_dropped"] == 1, c   # distogram_condition parked per step and unparked before every next() (the exhausting one included); after the trunk: distogram_condition, msa_stack (the cast copy), msa (its slice view), has_distogram_condition; the fp32 original parked on the first item (`keep` references it) and dropped on the second (nothing else does)
    assert torch.equal(out, ref2)                                           # the levered forward equals the same statements without the lever
    assert f["msa_stack"].dtype == torch.bfloat16 and torch.equal(f["distogram_condition"], dc)   # values byte-identical after the round trips
    assert torch.equal(keep, orig_stack)                                     # the engine's fp32 original: intact (parked and released, its object re-pointed)



def test_fpf_silent_lever_on_the_big_row_is_a_named_disengagement(monkeypatch):
    """A fast kernel silent because its own admission declined every call (fpf_res above its size gate at >= ~2900 tokens, fpf_trimul where the core declines, e.g. a 3,000-token fold) is a lever property the
    census names on the big row — `disengaged_by_property` on the tally, ok stays True; on the fast row the same silence is the named failure."""
    from rosettafold3_opt import report, registry as kit_registry
    import sys, types
    adp = types.ModuleType("fpf_rf3_adapter")
    adp.describe = lambda: {"mode": "fast", "served": 0, "fallback": {"declined": 10}, "errors": 0}                     # the trimul kernel served nothing: the core declined every call (counted by its word)
    adp.describe_v2 = lambda: {"counts": {"apb": {"served:triton": 480}}, "res": {"fused": 0, "unfused": 480}, "trunk_graph": {"replays": 0, "fallbacks": {}}}
    monkeypatch.setitem(sys.modules, "fpf_rf3_adapter", adp)
    levers = ["graph", "hoist", "fpf_trimul", "fpf_apb", "fpf_res"]
    assert all(kit_registry.LEVERS[lv].probe[0].startswith("fpf_") for lv in levers if lv.startswith("fpf_"))
    rep = {"fpf": {"arm": "fast+apb+res@L1", "applied": "configured"}, "levers": levers, "mode": "big"}
    t = report.fpf_tally(rep)
    assert t["levers_acted"] == ["fpf_apb"] and t["levers_silent"] == ["fpf_trimul", "fpf_res"], t
    assert t["disengaged_by_property"] == ["fpf_trimul", "fpf_res"] and t["ok"] is True and "silent lever" not in (t.get("reason") or ""), t
    assert "disengaged_by_property=fpf_trimul,fpf_res" in report.fpf_tally_line(t)
    t2 = report.fpf_tally({**rep, "mode": "fast"})
    assert t2["ok"] is False and "silent lever(s): fpf_trimul,fpf_res" in t2["reason"] and "disengaged_by_property" not in t2


def test_offload_levers_settings_are_built_through_the_cores_grammar(monkeypatch):
    """confidence_offload / feature_park read their host-park settings through ``opt_core.mem.offload.Settings.from_ctx`` (the pinned core's
    API; the kit's stated pin budget and engagement threshold are the kit settings — ``ctx.settings["host_park"]`` overrides them, a site's own
    ``cols`` comes last; nothing is read from the environment) — exercised on CPU (the levers' applies() refuse by name without CUDA before they
    reach the settings, so this is the statement that proves the API)."""
    from .. import _core, big, levers
    mem = _core.load("mem")
    offload = _core.load("mem.offload")
    assert hasattr(offload.Settings, "from_ctx") and not hasattr(offload.Settings, "from_env")          # the core's 0.4.0 API: Settings.from_ctx, no from_env
    env = {}
    ctx = mem.Ctx(prefix=big.PREFIX, tag="rosettafold3-opt", framework="torch", hooks={}, settings={}, environ=env, graphs=True)
    s = levers._park_settings(ctx)
    assert isinstance(s, offload.Settings) and s.pin_max_gb == float(levers.PIN_MAX_GB_DEFAULT) and s.min_tokens == int(levers.MIN_TOKENS_DEFAULT)
    assert levers._park_settings(ctx, cols="rowloop").cols == "rowloop"                                # feature_park's site override
    ctx2 = mem.Ctx(prefix=big.PREFIX, tag="rosettafold3-opt", framework="torch", hooks={}, settings={"host_park": {"min_tokens": 123}}, environ=env, graphs=True)
    assert levers._park_settings(ctx2).min_tokens == 123                                                # the kit's ctx.settings value wins over the stated default
    env[f"{big.PREFIX}_BIG_HOST_PARK_MIN_TOKENS"] = "7"                                             # an environment word is not read
    ctx3 = mem.Ctx(prefix=big.PREFIX, tag="rosettafold3-opt", framework="torch", hooks={}, settings={}, environ=env, graphs=True)
    assert levers._park_settings(ctx3).min_tokens == int(levers.MIN_TOKENS_DEFAULT)


def test_levers_whose_site_rowpair_takes_are_off_by_property_under_n_gpu_gt_1(monkeypatch):
    """--n_gpu > 1: triatt_chunk, opm_chunk, cond_chunk and confidence_offload are the row-sharding adapter's statements (rowpair.py installs
    after the levers on the same trigger) — off BY PROPERTY, named (big.site_owned; LEVER reason site_owned:rowpair), never a silent hole in
    the census; n_gpu 1: all seven."""
    from .. import big, report
    for k in [k for k in os.environ if k.startswith(big.PREFIX + "_BIG_")]:
        monkeypatch.delenv(k)
    arm_served = big.site_owned(1)                                                                              # n_gpu 1: the arm-served sites only (ARM_SERVES — transition_chunk is site_owned:fpf_ttr while ttr rides big's arm)
    assert set(arm_served) <= set(big.ARM_SERVES) and all(v.startswith("site_owned:fpf_") for v in arm_served.values())
    assert big.expected_levers(n_gpu=1) == [lv for lv in big.LEVERS if lv not in arm_served]
    exp2 = big.expected_levers(n_gpu=2)
    assert set(big.ROWPAIR_OWNS) == {"triatt_chunk", "opm_chunk", "cond_chunk", "confidence_offload"}
    assert set(big.LEVERS) - set(exp2) == set(big.ROWPAIR_OWNS) | set(arm_served) == set(big.site_owned(2))
    assert all(v.startswith("site_owned:rowpair — ") for lv, v in big.site_owned(2).items() if lv in big.ROWPAIR_OWNS)
    rep = {"levers": ["graph"] + [lv for lv in exp2] + ["rowpair"], "mode": "big", "big": {"site_owned": big.site_owned(2)}}
    t = {"mode": "big", "graph_flags_imported": True, "n_gpu": 2, "sharding": "rowpair"}
    st = report.lever_states(rep, t)
    for lv, why in big.site_owned(2).items():
        assert st["big_" + lv]["reason"] == why.partition(" — ")[0] and st["big_" + lv]["state"] == "off", (lv, st["big_" + lv])   # site_owned:rowpair / site_owned:fpf_ttr — named, never a silent hole


def test_install_passes_the_site_owned_levers_as_explicit_off_switches_under_n_gpu_gt_1(monkeypatch):
    from .. import big
    from .. import stack as _stack
    monkeypatch.setattr(_stack, "_install_watch", lambda *a, **k: None)
    monkeypatch.setattr(big.atexit, "register", lambda *a, **k: None)
    rep = {"n_gpu": 2, "mode": "big", "mem": {"policy": "capped"}}
    blk = big.install(rep, arm=True)
    assert blk["site_owned"] == big.site_owned(2) and not (set(blk["expected"]) & set(big.site_owned(2)))
    assert blk["flags"] == {lv: False for lv in big.site_owned(2)} and blk["off_by_flag"] == []          # explicit switches; nothing exported
    assert not any(k.startswith(big.PREFIX + "_BIG_") for k in os.environ)
    big.STATE.update({"installed": False, "line": None, "selection": None})


@pytest.mark.parametrize("n_gpu", [1, 2])
def test_the_activation_path_reaches_a_closed_census_with_the_model_stubbed(monkeypatch, n_gpu):
    """The exact big activation path of a rank process, on CPU with the levers' mechanisms stubbed: install(rep) (the n_gpu axis on the
    report → the site-owned levers as explicit off switches) → apply(rep) (opt_core.mem.apply with the same switches: the applied set) → one census
    unit in which every APPLIED lever marks itself → the exit verdict 0. Under n_gpu 2 the census expects the three levers the adapter leaves
    in place minus the arm-served one (triatt_chunk / opm_chunk / cond_chunk / confidence_offload are site_owned:rowpair; transition_chunk is site_owned:fpf_ttr) — the regression lock for: the axis reaching
    run_kit, the rowpair watch chaining after the levers' watch, the site-owned switches reaching the core."""
    from .. import big, levers, report, stack as _stack
    registry = _core.load("mem.registry")
    for k in [k for k in os.environ if k.startswith(big.PREFIX + "_BIG_")]:
        monkeypatch.delenv(k)
    monkeypatch.setattr(_stack, "_install_watch", lambda *a, **k: None)
    monkeypatch.setattr(big.atexit, "register", lambda *a, **k: None)
    monkeypatch.setattr(big, "_hooks", lambda: {})
    monkeypatch.setattr(levers, "install_units", lambda ctx: None)
    big.STATE.update({"installed": False, "record": None, "line": None, "ctx": None, "units": 0, "trigger_fired": False, "reason": None, "exit": None})
    for name in big.LEVERS:                                                          # the mechanisms need CUDA; the census does not: stub applies/apply per lever
        lv = registry.get(name)
        orig = (lv.applies, lv.apply)
        object.__setattr__(lv, "applies", lambda ctx, _n=name: None)
        monkeypatch.setattr(big, "_restore_" + name, lambda lv=lv, orig=orig: (object.__setattr__(lv, "applies", orig[0]), object.__setattr__(lv, "apply", orig[1])), raising=False)
        object.__setattr__(lv, "apply", lambda ctx, _n=name: registry.Applied(lever=_n, undo=lambda: None))
    rep = {"n_gpu": n_gpu, "sharding": "rowpair" if n_gpu > 1 else "none", "mode": "big", "row": "big", "mem": {"policy": "capped"}, "levers": ["graph"], "fpf": None}
    try:
        _census_path(big, levers, report, registry, rep, n_gpu)
    finally:
        for name in big.LEVERS:
            getattr(big, "_restore_" + name)()
        big.STATE.update({"installed": False, "record": None, "line": None, "ctx": None, "units": 0, "trigger_fired": False, "reason": None, "exit": None})


def _census_path(big, levers, report, registry, rep, n_gpu):
    blk = big.install(rep, arm=True)
    expected = list(blk["expected"])
    assert (set(big.LEVERS) - set(expected)) == (set(big.site_owned(n_gpu))) and len(expected) == len(big.LEVERS) - len(big.site_owned(n_gpu)), (n_gpu, expected)   # 6 at n_gpu 1 (transition_chunk is site_owned:fpf_ttr), 2 under rowpair
    trig = types.ModuleType(big.TRIGGER)                                             # the executed trigger module the watch hands to apply
    big.apply(rep, trig)                                                             # opt_core.mem.apply: reads the flags from os.environ — must agree with `expected`
    rec = big.STATE["record"]
    assert set(rec.levers) == set(expected), (sorted(rec.levers), sorted(expected))
    blk = big.apply(rep, trig)                                                       # ONCE per process: the callback running again is a named no-op —
    assert blk["reapply"] == "skipped:already_applied" and big.STATE["record"] is rec and blk["applied"] == "configured"   # nothing re-derived, the record kept
    assert big.apply(rep)["reapply"] == "skipped:already_applied" and big.STATE["record"] is rec                        # (with or without the module object)
    with pytest.raises(big.BigError, match="executed a second time"):                                                   # the trigger EXECUTED AGAIN (another module
        big.apply(rep, types.ModuleType(big.TRIGGER))                                                                   # object): refused by name, never re-derived
    assert rep["big"]["reapply"] == "refused:trigger_reexecuted" and big.STATE["record"] is rec
    big.STATE["reason"] = None
    rec.unit_begin("fold1")
    for lv in expected:
        rec.mark(lv)
    rec.unit_end("fold1")
    big.STATE["units"] = 1
    st = big.state()
    assert st["exit"]["exit_code"] == 0 and st["census"]["ok"], (st["exit"], st["census"])
    if n_gpu > 1:
        assert blk["site_owned"] == big.site_owned(2) and blk["flags"] == big.switches_for(2) and not any(k.startswith(big.PREFIX + "_BIG_") for k in os.environ)
        states = report.lever_states({"levers": rep["levers"] + expected + ["rowpair"], "mode": "big", "big": blk}, {"mode": "big", "graph_flags_imported": True, "n_gpu": 2, "sharding": "rowpair", "big": st})
        assert states["big_triatt_chunk"]["reason"] == "site_owned:rowpair" and states["big_opm_chunk"]["reason"] == "site_owned:rowpair"
    # and the census DOES fail by name when an expected lever never marks (a silent lever is a failure, not a pass)
    rec.unit_begin("fold2"); rec.unit_end("fold2")
    big.STATE["units"] = 2; big.STATE["exit"] = None
    assert big.state()["exit"]["exit_code"] == 3
    big.STATE.update({"installed": False, "record": None, "line": None, "ctx": None, "units": 0, "trigger_fired": False, "reason": None, "exit": None})


def test_offload_consumers_after_a_closed_unit_are_attributed_per_unit_and_never_raise(monkeypatch):
    """The engine's post-processing (`compile_af3_style_confidence_outputs`) and its metrics (`compute_ptm`) run AFTER `model.forward`'s unit
    closed. An above-gate item (samples offloaded) followed by a below-gate item (logits stay on the device) in ONE fold process is the ordinary
    case: the below-gate item's device logits are the declared policy (counter `n_compile_device`), attributed by that unit's own head decision
    (`unit_offloaded`), NOT by the process-cumulative `n_offloaded` — the census record raises by name on a unit-scope event outside an open unit
    (opt_core.mem.record: "no unit is open"), so the consumers never emit one there. Device logits although the last unit's samples WERE offloaded
    are a named PROCESS-scope event: counter `n_device_after_offload` + a record note; still no raise."""
    mem = _core.load("mem")
    rec = mem.AppliedRecord(prefix="ROSETTAFOLD3", base="fast", expected=("confidence_offload",))

    class Dev:
        def __init__(self, typ): self.type = typ
    class T:                                                        # a logits stand-in: only `.device.type` and `.shape` are read on these paths
        def __init__(self, typ, n_tokens=8): self.device, self.shape = Dev(typ), (1, n_tokens, n_tokens, 4)
    class St:                                                       # the _Staging counters the consumers touch (no pinned pool on the CPU line)
        n_offloaded = 3; n_compile_device = 0; n_device_after_offload = 0; n_compile = 0; n_ptm = 0; unit_offloaded = False
    st = St()
    calls = []
    monkeypatch.setitem(levers.STATE, "offload", st)
    monkeypatch.setitem(levers.STATE, "orig", {**levers.STATE.get("orig", {}), "compute_ptm": lambda pae, tc, *a, **k: calls.append("ptm") or "PTM",
                                                "compile": lambda pl, pae, pde, *a, batch_idx=0, **k: calls.append("compile") or {"plddt": None}})
    monkeypatch.setitem(levers.STATE, "ctx", types.SimpleNamespace(record=rec))
    rec.unit_begin("fold1"); rec.mark("confidence_offload", detail="head:pae_logits,pde_logits"); rec.unit_end("fold1")    # item 1: above the gate, offloaded
    rec.unit_begin("fold2"); rec.mark("confidence_offload", detail="head:below_gate:N=8"); rec.unit_end("fold2")           # item 2: below the gate; st.unit_offloaded False
    assert levers.ptm_offload(T("cuda"), None) == "PTM" and calls == ["ptm"]                     # no unit open, device logits, n_offloaded > 0: NO raise (was ValueError at record._unit)
    assert st.n_compile_device == 1 and st.n_device_after_offload == 0
    st.unit_offloaded = True                                                                     # the inconsistency: the last unit DID offload, yet device logits arrive
    assert levers.ptm_offload(T("cuda"), None) == "PTM"
    assert st.n_device_after_offload == 1 and any("n_device_after_offload=1" in n for n in rec.notes)
    assert all(not u.fallback for u in rec.units.values())                                       # no unit carries a fallback: the event is process-scope, named in the note + counter
