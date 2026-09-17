"""opt_core.of3_sampler (+ its rollout_memo) on CPU: the per-rollout store's epoch / address-stability / cap contract, dtk_kernels.segments
on CPU tensors, dit_rows.lead_period, and the adapters' switch grammar through configure(). GPU numerics are the kits' tests/test_dit_cells.py
and tests/gpu/test_dtk_v02_gpu.py."""
import pytest

torch = pytest.importorskip("torch")

from opt_core.of3_sampler import rollout_memo as RM  # noqa: E402
from opt_core.of3_sampler import apb_trunk, atom_hoist, dit_glue, dit_rows, token_agg  # noqa: E402


def _dtk():
    pytest.importorskip("triton")
    from opt_core.kernels import route
    route("dtk_kernels")
    import importlib
    return importlib.import_module("dtk_kernels")


def test_lead_period():
    assert dit_rows.lead_period((1, 1, 7), (1, 5, 7)) == 7            # s [1,1,N] serves a [1,S,N]: period N
    assert dit_rows.lead_period((1, 5, 7), (1, 5, 7)) == 35           # equal shapes: direct
    assert dit_rows.lead_period((7,), (2, 3, 7)) == 7 and dit_rows.lead_period((), (4, 7)) == 1
    assert dit_rows.lead_period((5, 1), (5, 7)) is None                 # interleaved broadcast: the caller expands


def test_switch_grammar_through_configure():
    """Unbound modules request nothing; bound to a kit's names they parse its switches and refuse unknown values by name."""
    saved = {m: {k: getattr(m, k) for k in m.CONFIGURABLE} for m in (atom_hoist, dit_glue, token_agg, apb_trunk)}
    try:
        for m in (atom_hoist, dit_glue, token_agg, apb_trunk):
            m.configure(ENV=None)
            assert m.requested({"X_ATOM_HOIST": "1", "X_DIT_GLUE": "1", "X_TOKEN_AGG": "seg_reduce", "X_APB_TRUNK": "1"}) is False
        with pytest.raises(KeyError):
            dit_glue.configure(NOT_A_SETTING=1)
        with pytest.raises(KeyError):
            apb_trunk.configure(ENV_CORE="X")
        with pytest.raises(ValueError):
            apb_trunk.configure(HIGH_PRECISION="sometimes")
        apb_trunk.configure(ENV="X_APB_TRUNK", ENV_MIN="X_APB_TRUNK_MIN_TOKENS", ENV_HIGH_PRECISION="X_APB_TRUNK_HP", ENV_SCOPE="X_APB_TRUNK_SCOPE",
                            CONFLICT_ENV="X_T_APB", HIGH_PRECISION="override", M_CONF=None)
        assert apb_trunk.requested({}) is False and apb_trunk.requested({"X_APB_TRUNK": "1", "X_APB_TRUNK_MIN_TOKENS": "384"}) is True
        assert apb_trunk.HIGH_PRECISION == "override" and "high_precision=override scope=trunk+confidence" in apb_trunk.census_line()
        assert apb_trunk.requested({"X_APB_TRUNK": "1", "X_APB_TRUNK_HP": "honour"}) is True          # the run-time word over the kit's
        for bad in ({"X_APB_TRUNK": "yes"}, {"X_APB_TRUNK": "1", "X_APB_TRUNK_MIN_TOKENS": "-5"}, {"X_APB_TRUNK": "1", "X_T_APB": "cueq"},
                    {"X_APB_TRUNK": "1", "X_APB_TRUNK_HP": "fp32"}, {"X_APB_TRUNK": "1", "X_APB_TRUNK_SCOPE": "confidence"},
                    {"X_APB_TRUNK": "1", "X_APB_TRUNK_SCOPE": "trunk"}):                              # scope=trunk needs M_CONF bound
            with pytest.raises(ValueError):
                apb_trunk.requested(bad)
        apb_trunk.configure(M_CONF="some.engine.heads")
        assert apb_trunk.requested({"X_APB_TRUNK": "1", "X_APB_TRUNK_SCOPE": "trunk"}) is True
        atom_hoist.configure(ENV="X_ATOM_HOIST", ENV_MAX_GB="X_ATOM_HOIST_MAX_GB", ENV_INV="X_ATOM_HOIST_INV")
        dit_glue.configure(ENV="X_DIT_GLUE", ENV_MIN="X_DIT_GLUE_MIN_TOKENS", ENV_CORE="X_DIT_GLUE_CORE", ENV_ROWS="X_DIT_GLUE_ROWS")
        token_agg.configure(ENV="X_TOKEN_AGG")
        assert dit_glue.requested({"X_DIT_GLUE": "1", "X_DIT_GLUE_ROWS": "block"}) is True and dit_glue.requested({"X_DIT_GLUE": "1", "X_DIT_GLUE_ROWS": "all"}) is True
        assert atom_hoist.requested({}) is False and atom_hoist.requested({"X_ATOM_HOIST": "1", "X_ATOM_HOIST_INV": "bias", "X_ATOM_HOIST_MAX_GB": "0.5"}) is True
        assert dit_glue.requested({"X_DIT_GLUE": "1", "X_DIT_GLUE_CORE": "dtk", "X_DIT_GLUE_MIN_TOKENS": "512"}) is True
        assert token_agg.requested({"X_TOKEN_AGG": "seg_reduce"}) is True
        for m, bad in ((atom_hoist, {"X_ATOM_HOIST": "yes"}), (atom_hoist, {"X_ATOM_HOIST": "1", "X_ATOM_HOIST_INV": "all"}),
                       (atom_hoist, {"X_ATOM_HOIST": "1", "X_ATOM_HOIST_MAX_GB": "-1"}), (dit_glue, {"X_DIT_GLUE": "on"}),
                       (dit_glue, {"X_DIT_GLUE": "1", "X_DIT_GLUE_CORE": "sdpa"}), (dit_glue, {"X_DIT_GLUE": "1", "X_DIT_GLUE_MIN_TOKENS": "-3"}), (dit_glue, {"X_DIT_GLUE": "1", "X_DIT_GLUE_ROWS": "atoms"}),
                       (token_agg, {"X_TOKEN_AGG": "scatter"})):
            with pytest.raises(ValueError):
                m.requested(bad)
    finally:
        for m, kv in saved.items():
            m.configure(**kv)


class _Sampler:
    pass


def test_lead_period():
    assert dit_rows.lead_period((1, 1, 7), (1, 5, 7)) == 7            # s [1,1,N] serves a [1,S,N]: period N
    assert dit_rows.lead_period((1, 5, 7), (1, 5, 7)) == 35           # equal shapes: direct
    assert dit_rows.lead_period((7,), (2, 3, 7)) == 7 and dit_rows.lead_period((), (4, 7)) == 1
    assert dit_rows.lead_period((5, 1), (5, 7)) is None                 # interleaved broadcast: the caller expands


def test_memo_store_epochs_and_addresses():
    """Inside a rollout a key answers from the current epoch only; a later rollout of the same layout refreshes the SAME buffer in place while
    graphs are active (address-stable), replaces it otherwise; the eager route drops the store at the boundary's exit; the cap skips by name."""
    RM.STORE.drop_all(); RM.STATE.update(fills=0, hits=0, cap_skips={}, bytes=0); RM._GRAPHS["active"] = False
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return (torch.full((4, 3), float(calls["n"])),)
    s = _Sampler()
    RM._enter_rollout(s)
    assert getattr(s, RM.EPOCH_ATTR) == RM.epoch() and RM.inside()
    a = RM.memo_call("q", ("q", 1), compute)[0]
    b = RM.memo_call("q", ("q", 1), compute)[0]
    assert calls["n"] == 1 and b is a and RM.STATE["hits"] == 1 and RM.STATE["fills"] == 1
    RM._exit_rollout()
    assert not RM.inside() and RM.STATE["entries"] == 0                                   # eager route: dropped at exit
    RM._GRAPHS["active"] = True                                                            # graphed route: entries survive, refreshed in place next rollout
    try:
        RM._enter_rollout(s); e1 = RM.epoch()
        c = RM.memo_call("q", ("q", 1), compute)[0]; ptr = c.data_ptr()
        RM._exit_rollout()
        RM._enter_rollout(s)
        assert RM.epoch() == e1 + 1 and RM.STORE.get(("q", 1)) is None                      # stale epoch: no answer
        d = RM.memo_call("q", ("q", 1), compute)[0]
        assert d.data_ptr() == ptr and float(d[0, 0]) == calls["n"]                          # same address, new contents
        big = RM.memo_call("big", ("big", 1), lambda: (torch.zeros(8),))                     # a new key under the cap: kept
        assert RM.STORE.get(("big", 1)) is not None and big[0].numel() == 8
        RM.set_cap_gb(0.0)
        skipped = RM.memo_call("huge", ("huge", 1), lambda: (torch.zeros(16),))[0]           # over the cap: computed, not kept, counted
        assert skipped.numel() == 16 and RM.STORE.get(("huge", 1)) is None and RM.STATE["cap_skips"] == {"huge": 1}
        RM._exit_rollout()
    finally:
        RM._GRAPHS["active"] = False; RM.set_cap_gb(1.5); RM.STORE.drop_all(); RM._EPOCH["depth"] = 0


def _segments_ref(idx, mask, n):
    starts = [0] * n; counts = [0] * n
    first = {}; last = {}
    for p, (i, m) in enumerate(zip(idx.tolist(), mask.tolist())):
        if m > 0:
            first.setdefault(i, p); last[i] = p
    for j in range(n):
        if j in first:
            starts[j] = first[j]; counts[j] = last[j] - first[j] + 1
    return starts, counts


def test_segments_on_cpu():
    dk = _dtk()
    idx = torch.tensor([0, 0, 0, 1, 2, 2, 2, 2, 4, 4, 0, 0])          # token 3 empty; the two trailing atoms are padding (masked) pointing at token 0
    mask = torch.tensor([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]).float()
    st, ct, mx, ok = dk.segments(idx, mask, 6)
    rs, rc = _segments_ref(idx, mask, 6)
    assert ok and st.tolist() == rs and ct.tolist() == rc and int(mx) == 4 and ct.tolist()[3] == 0 and ct.tolist()[5] == 0
    st, ct, mx, ok = dk.segments(torch.tensor([0, 2, 1]), torch.ones(3), 3)
    assert not ok and ct.tolist() == [0, 0, 0]                          # interleaved runs: reported, never served
    st, ct, mx, ok = dk.segments(torch.tensor([0, 2, 1]), torch.tensor([1.0, 0.0, 1.0]), 3)
    assert ok and ct.tolist() == [1, 1, 0]                              # the out-of-order atom is masked: the unmasked layout is sorted
    st, ct, mx, ok = dk.segments(torch.tensor([0, 7]), torch.ones(2), 3)
    assert not ok                                                       # an index outside [0, N): refused


def _reset_store():
    RM.STORE.drop_all(); RM.STATE.update(fills=0, hits=0, cap_skips={}, bytes=0, errors={}); RM._GRAPHS["active"] = False; RM._EPOCH["depth"] = 0; RM.set_cap_gb(1.5)


def test_frozen_store_under_graphs_never_readdresses_what_a_graph_may_hold():
    """Under graphs: a key that met the cap in a shape's first rollout re-enters put() at the next same-shape rollout while later producers'
    entries are still stale-epoch — nothing is evicted (the live graph reads them), the skip is sticky and counted, every stored entry is
    refreshed at its recorded address; an entry last touched two rollouts ago (another shape's) IS evictable; a same-key value of another layout
    raises AddressError instead of being re-addressed."""
    _reset_store(); s = _Sampler(); RM._GRAPHS["active"] = True
    val = {"n": 0.0}

    def prod(shape, dtype=torch.float32):
        val["n"] += 1.0
        return lambda: (torch.full(shape, val["n"], dtype=dtype),)
    try:
        RM.set_cap_gb((2 * 64 * 4 + 8) / 2 ** 30)                                              # room for A and C (64 floats each), never for B (4096)
        RM._enter_rollout(s)
        a = RM.memo_call("A", ("A", 64), prod((64,)))[0]; b = RM.memo_call("B", ("B", 4096), prod((4096,)))[0]; c = RM.memo_call("C", ("C", 64), prod((64,)))[0]
        assert RM.STORE.get(("A", 64)) is not None and RM.STORE.get(("B", 4096)) is None and RM.STORE.get(("C", 64)) is not None and b.numel() == 4096
        pa, pc = a.data_ptr(), c.data_ptr()
        assert RM.STATE["cap_skips"] == {"B": 1} and ("B", 4096) in RM.STORE.skipped
        RM._exit_rollout()
        RM._enter_rollout(s)                                                                     # rollout 2, same shapes: the graph captured in rollout 1 is live
        a2 = RM.memo_call("A", ("A", 64), prod((64,)))[0]
        assert a2.data_ptr() == pa and float(a2[0]) == val["n"]                                  # refreshed in place
        RM.memo_call("B", ("B", 4096), prod((4096,)))                                            # re-enters over cap: sticky skip, and C — stale-epoch right now — is NOT evicted
        assert ("C", 64) in RM.STORE.ent and RM.STORE.ent[("C", 64)]["t"][0].data_ptr() == pc and RM.STATE["cap_skips"] == {"B": 2}
        c2 = RM.memo_call("C", ("C", 64), prod((64,)))[0]
        assert c2.data_ptr() == pc and float(c2[0]) == val["n"]
        RM._exit_rollout()
        RM._enter_rollout(s)                                                                     # rollout 3, another shape: A/C (touched in rollout 2) stay; nothing older exists
        RM.set_cap_gb((3 * 64 * 4 + 8) / 2 ** 30)
        d = RM.memo_call("D", ("D", 64), prod((64,)))[0]
        assert RM.STORE.get(("D", 64)) is not None and ("A", 64) in RM.STORE.ent and ("C", 64) in RM.STORE.ent
        RM._exit_rollout()
        RM._enter_rollout(s)                                                                     # rollout 4, shape D again: A/C last touched two rollouts ago -> evictable under cap pressure
        RM.memo_call("D", ("D", 64), prod((64,)))
        e = RM.memo_call("E", ("E", 128), prod((128,)))[0]                                       # needs room: A and C go (no live graph can hold them), D (touched now) stays
        assert RM.STORE.get(("E", 128)) is not None and ("A", 64) not in RM.STORE.ent and ("C", 64) not in RM.STORE.ent and ("D", 64) in RM.STORE.ent and e.numel() == 128
        RM._exit_rollout()
        RM._enter_rollout(s)
        with pytest.raises(RM.AddressError):                                                     # same key, another dtype: not refreshable in place under a graph
            RM.memo_call("D", ("D", 64), prod((64,), dtype=torch.float64))
        RM._exit_rollout()
    finally:
        _reset_store()


def test_configure_refuses_a_second_different_binding():
    prev = dict(RM._BOUND)
    try:
        RM._BOUND.pop("BOUNDARY_METHODS", None)
        RM.configure(BOUNDARY_METHODS=("forward", "_sample_rollout"))
        RM.configure(BOUNDARY_METHODS=("forward", "_sample_rollout"))                            # the same binding again: fine (every adapter of a kit calls the one binder)
        with pytest.raises(ValueError):
            RM.configure(BOUNDARY_METHODS=("forward",))
        with pytest.raises(KeyError):
            RM.configure(NOT_A_WORD=1)
    finally:
        RM._BOUND.clear(); RM._BOUND.update(prev); RM.BOUNDARY_METHODS = ("forward", "_sample_rollout")


def test_dit_glue_rows_patch_dispatch_on_cpu():
    """The class-wide conditioned-transition / AdaLN rows: patched once (idempotent), pass through when off, refuse CPU tensors BY NAME with the
    stock method's output returned bitwise and the refusal counted, and report themselves on the census line."""
    torch = pytest.importorskip("torch")
    import types
    import sys as _sys

    class AdaLN(torch.nn.Module):                       # the AF3-family AdaLN layout dit_rows reads (LN_a no affine, LN_s scale only, linear_g with bias, linear_s without)
        def __init__(self, c_a, c_s):
            super().__init__(); self.c_a, self.c_s = c_a, c_s
            self.layer_norm_a = torch.nn.LayerNorm(c_a, elementwise_affine=False); self.layer_norm_s = torch.nn.LayerNorm(c_s, bias=False)
            self.linear_g = torch.nn.Linear(c_s, c_a); self.linear_s = torch.nn.Linear(c_s, c_a, bias=False)

        def forward(self, a, s):
            return torch.sigmoid(self.linear_g(self.layer_norm_s(s))) * self.layer_norm_a(a) + self.linear_s(self.layer_norm_s(s))

    class ConditionedTransitionBlock(torch.nn.Module):
        def __init__(self, c_a, c_s, n=2):
            super().__init__(); self.c_a, self.c_s = c_a, c_s; self.layer_norm = AdaLN(c_a, c_s)
            self.swiglu = torch.nn.Module(); self.swiglu.linear_a = torch.nn.Linear(c_a, n * c_a, bias=False); self.swiglu.linear_b = torch.nn.Linear(c_a, n * c_a, bias=False)
            self.linear_g = torch.nn.Linear(c_s, c_a); self.linear_out = torch.nn.Linear(n * c_a, c_a, bias=False)

        def forward(self, a, s, mask=None, chunk_size=None):
            x = self.layer_norm(a, s); b = torch.nn.functional.silu(self.swiglu.linear_a(x)) * self.swiglu.linear_b(x)
            u = torch.sigmoid(self.linear_g(s)) * self.linear_out(b)
            return u if mask is None else u * mask.unsqueeze(-1)

    mod = types.ModuleType("x_engine.diffusion_transformer"); mod.ConditionedTransitionBlock = ConditionedTransitionBlock
    home = _sys.modules[ConditionedTransitionBlock.__module__]                # the class's defining module must carry AdaLN beside it (as the engine's transition module does)
    had = hasattr(home, "AdaLN"); prev = getattr(home, "AdaLN", None); home.AdaLN = AdaLN
    stock_c, stock_a = ConditionedTransitionBlock.forward, AdaLN.forward
    try:
        assert dit_glue._patch_rows(types.ModuleType("empty")).startswith("no ConditionedTransitionBlock")
        assert dit_glue._patch_rows(mod) == "" and dit_glue._patch_rows(mod) == ""                     # idempotent
        assert ConditionedTransitionBlock.forward.__wrapped__ is stock_c and AdaLN.forward.__wrapped__ is stock_a
        torch.manual_seed(0)
        m = ConditionedTransitionBlock(8, 4).eval(); a = torch.randn(2, 5, 8); s = torch.randn(1, 5, 4); mask = torch.ones(2, 5)
        with torch.no_grad():
            want_c = stock_c(m, a, s, mask=mask); want_a = stock_a(m.layer_norm, a, s)
            dit_glue.STATE["rows_state"] = "off"
            assert torch.equal(m(a, s, mask=mask), want_c) and dit_glue.STATE["cond_fallback"] == {} and dit_glue.STATE["cond_served"] == 0
            dit_glue.STATE["rows_state"] = "on"
            got_c = m(a, s, mask=mask); got_a = m.layer_norm(a, s); got_chunk = m(a, s, mask=mask, chunk_size=4)
        assert torch.equal(got_c, want_c) and torch.equal(got_a, want_a) and torch.equal(got_chunk, want_c)
        assert dit_glue.STATE["cond_fallback"] == {"not_cuda": 1, "chunked": 1} and dit_glue.STATE["adaln_fallback"].get("not_cuda", 0) >= 1, (dit_glue.STATE["cond_fallback"], dit_glue.STATE["adaln_fallback"])
        assert dit_glue.STATE["cond_served"] == 0 and dit_glue.STATE["adaln_served"] == 0
        line = dit_glue.census_line()
        assert " rows=all cond_served=0 cond_fallback=chunked:1,not_cuda:1 cond_modes=none adaln_served=0 adaln_fallback=not_cuda:" in line, line
    finally:
        ConditionedTransitionBlock.forward, AdaLN.forward = stock_c, stock_a
        if had:
            home.AdaLN = prev
        else:
            delattr(home, "AdaLN")
        dit_glue.STATE.update(rows_state="off", cond_served=0, adaln_served=0, cond_first=None)
        dit_glue.STATE["cond_fallback"].clear(); dit_glue.STATE["adaln_fallback"].clear(); dit_glue.STATE["cond_modes"].clear()
