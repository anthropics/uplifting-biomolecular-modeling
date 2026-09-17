"""opt_core.of3_trunk (castcache, apb_hoist, trunk_graph) and of3_sampler.post_release on CPU: the memo keys refresh on weight edits, the
key-mask bias memo hits on the same mask and refills on another, the stack capture refuses by name off the GPU, the release gate's words, the
adapters' switch grammar through configure(); and attn.pair_fused.lookup_cell's memo keyed on the full argument set + table identity. GPU numerics
(bitwise vs the engine) are the kits' GPU records."""
import copy
import types

import pytest

torch = pytest.importorskip("torch")

from opt_core.attn import pair_fused as PF  # noqa: E402
from opt_core.of3_sampler import post_release  # noqa: E402
from opt_core.of3_trunk import apb_hoist, castcache, trunk_graph  # noqa: E402


def test_switch_grammar_through_configure():
    for m, env in ((castcache, "X_CASTCACHE"), (apb_hoist, "X_APB_HOIST"), (trunk_graph, "X_TRUNK_GRAPH"), (post_release, "X_POST_RELEASE")):
        keep = {k: getattr(m, k) for k in m.CONFIGURABLE}
        try:
            with pytest.raises(KeyError):
                m.configure(NOT_A_SETTING=1)
            m.configure(ENV=env)
            assert m.requested({}) is False and m.requested({env: "1"}) is True and m.requested({env: " "}) is False
            with pytest.raises(ValueError):
                m.requested({env: "yes"})
        finally:
            m.configure(**keep)
    keep = {k: getattr(trunk_graph, k) for k in trunk_graph.CONFIGURABLE}
    try:
        trunk_graph.configure(ENV="X_TRUNK_GRAPH", ENV_NMAX="X_TRUNK_GRAPH_NMAX")
        assert trunk_graph.nmax({}) == trunk_graph.NMAX_DEFAULT and trunk_graph.nmax({"X_TRUNK_GRAPH_NMAX": "2048"}) == 2048
        for bad in ("many", "-1"):
            with pytest.raises(ValueError):
                trunk_graph.nmax({"X_TRUNK_GRAPH_NMAX": bad})
    finally:
        trunk_graph.configure(**keep)
    assert post_release._knob({"K": "2400"}, "K", 1, int) == 2400 and post_release._knob({}, "K", 16.0, float) == 16.0
    with pytest.raises(ValueError):
        post_release._knob({"K": "lots"}, "K", 1, int)


def test_castcache_memo_refreshes_on_weight_edit():
    lin = torch.nn.Linear(8, 4)
    n0 = castcache.STATE["modules"]
    w1, b1 = castcache.cached16(lin, torch.bfloat16)
    assert w1.dtype is torch.bfloat16 and torch.equal(w1, lin.weight.detach().to(torch.bfloat16)) and torch.equal(b1, lin.bias.detach().to(torch.bfloat16))
    w2, b2 = castcache.cached16(lin, torch.bfloat16)
    assert w2 is w1 and b2 is b1 and castcache.STATE["modules"] == n0 + 1                       # a hit hands back the same copies
    with torch.no_grad():
        lin.weight.add_(1.0)                                                                   # in-place edit: version counter moves -> refresh
    r0 = castcache.STATE["refresh"]
    w3, _ = castcache.cached16(lin, torch.bfloat16)
    assert w3 is not w1 and torch.equal(w3, lin.weight.detach().to(torch.bfloat16)) and castcache.STATE["refresh"] == r0 + 1
    nb = lin.__dict__[castcache.ATTR][3]
    assert nb == (8 * 4 + 4) * 2                                                                # bf16 bytes accounted
    assert castcache._verify(castcache.cached16, castcache.LINEAR_SOURCE, "Linear.forward")     # a function without the served statements is refused (a reason string)
    assert castcache._verify(test_switch_grammar_through_configure, (), "x") is None                # nothing wanted: nothing missing


def test_apb_hoist_memo_hits_same_mask_refills_other():
    A = types.SimpleNamespace(permute_final_dims=lambda t, dims: t.permute(*([i for i in range(t.dim() - 3)] + [t.dim() - 3 + d for d in dims])))
    calls = {"orig": 0}

    def orig(self, a=None, z=None, mask=None):
        calls["orig"] += 1
        return ["orig"]

    class M:
        inf = 1e9
        layer_norm_z = staticmethod(lambda z: z)
        linear_z = staticmethod(lambda z: z)
    prep = apb_hoist._make_prep_bias(orig, A)
    apb_hoist.release()
    a = torch.zeros(1, 5, 7, 16); z = torch.zeros(1, 7, 7, 4); mask = torch.ones(1, 7)
    h0, f0 = apb_hoist.STATE["hit"], apb_hoist.STATE["fill"]
    out1 = prep(M(), a, z, mask); out2 = prep(M(), a, z, mask)
    assert apb_hoist.STATE["fill"] == f0 + 1 and apb_hoist.STATE["hit"] == h0 + 1 and out2[0] is out1[0]
    assert out1[0].shape == (1, 5, 1, 1, 7) and torch.equal(out1[0], (1e9 * (mask.expand(1, 5, 7) - 1))[..., None, None, :]) and out1[1].shape == (1, 4, 7, 7)
    mask2 = torch.ones(1, 7); mask2[0, -1] = 0
    out3 = prep(M(), a, z, mask2)
    assert apb_hoist.STATE["fill"] == f0 + 2 and out3[0][0, 0, 0, 0, -1] == -1e9
    assert prep(M(), a, z, None) == ["orig"] and calls["orig"] == 1 and apb_hoist.STATE["fallback"]["no_mask"] >= 1   # no mask: the engine's statement, counted


def test_trunk_graph_refuses_by_name_off_gpu():
    calls = {"n": 0}

    def orig(self, *a, **k):
        calls["n"] += 1
        return (k.get("s", a[0] if a else None), k.get("z"))
    trunk_graph.STATE["nmax"] = 1024; trunk_graph.STATE["state"] = "on"
    try:
        fwd = trunk_graph._make_forward(orig)
        stack = types.SimpleNamespace(training=False, blocks=[])
        s_, z_ = torch.zeros(1, 9, 8), torch.zeros(1, 9, 9, 4)
        f0 = dict(trunk_graph.STATE["fallback"])
        out = fwd(stack, s=s_, z=z_, single_mask=torch.ones(1, 9), pair_mask=torch.ones(1, 9, 9), chunk_size=None, use_deepspeed_evo_attention=False)
        assert out[0] is s_ and calls["n"] == 1 and trunk_graph.STATE["fallback"].get("cpu", 0) == f0.get("cpu", 0) + 1
        fwd(stack, s=s_, z=z_, single_mask=None, pair_mask=torch.ones(1, 9, 9))
        assert trunk_graph.STATE["fallback"].get("non_tensor_arg", 0) == f0.get("non_tensor_arg", 0) + 1 and calls["n"] == 2
        assert trunk_graph.release() == 0
    finally:
        trunk_graph.STATE["state"] = "off"; trunk_graph.STATE["nmax"] = None


def test_post_release_gate_words():
    post_release.STATE.update(ntok=2400, min_free_gb=16.0, total=80 * 2 ** 30)
    fake = types.SimpleNamespace(cuda=types.SimpleNamespace(memory_reserved=lambda: 10 * 2 ** 30, current_device=lambda: 0,
                                                            get_device_properties=lambda d: types.SimpleNamespace(total_memory=80 * 2 ** 30)))
    assert post_release._gate(fake, 3036) == "n_tok_3036_ge_2400"
    assert post_release._gate(fake, 400) is None and post_release._gate(fake, None) is None
    fake.cuda.memory_reserved = lambda: 70 * 2 ** 30
    assert post_release._gate(fake, 400).startswith("free_10.0GB_lt_16GB")
    assert post_release._n_tok({"token_mask": torch.ones(1, 77)}) == 77 and post_release._n_tok({}) is None and post_release._n_tok(None) is None


def test_lookup_cell_memo_keys_on_arguments_and_table_identity():
    t = PF.cells()
    row = next(r for r in t["rows"] if r["status"] == "certified")
    stack = (row["cc"], row["triton"] if row["triton"] != "*" else "9.9")
    PF.lookup_memo_clear()
    h0, m0 = PF.LOOKUP_MEMO_STATS["hit"], PF.LOOKUP_MEMO_STATS["miss"]
    d1 = PF.lookup_cell(row["impl"], row["piece"], row["key"], stack=stack, variant=row.get("variant"))
    d2 = PF.lookup_cell(row["impl"], row["piece"], list(row["key"]), stack=stack, variant=row.get("variant"))
    assert d1.row is not None and d2 is d1 and PF.LOOKUP_MEMO_STATS["miss"] == m0 + 1 and PF.LOOKUP_MEMO_STATS["hit"] == h0 + 1
    d3 = PF.lookup_cell(row["impl"], row["piece"], row["key"], stack=(row["cc"], "0.0" if row["triton"] != "*" else "9.8"), variant=row.get("variant"))
    assert PF.LOOKUP_MEMO_STATS["miss"] == m0 + 2 and d3 is not d1                            # another stack word: another key
    t["rows"].append(copy.deepcopy(row))                                                       # a row appended at run time: the table identity changes -> miss
    try:
        d4 = PF.lookup_cell(row["impl"], row["piece"], row["key"], stack=stack, variant=row.get("variant"))
        assert PF.LOOKUP_MEMO_STATS["miss"] == m0 + 3 and d4 == d1
    finally:
        t["rows"].pop()
        PF.lookup_memo_clear()
    unk = PF.lookup_cell("no_such_impl", "piece", [1, 2, 3], stack=stack)
    assert unk.row is None and unk.reason.startswith("no-cell:no_such_impl:piece:1x2x3:")
    assert PF.lookup_cell("no_such_impl", "piece", [1, 2, 3], stack=stack) is unk                # unserved decisions memoised too


def test_castcache_serves_each_primitive_on_its_own(monkeypatch, tmp_path):
    """An engine whose LayerNorm statement differs (the 0.5.x fork upcasts bf16 input to fp32) keeps that primitive as it is, named
    `skipped=layernorm:statement_differs`, while Linear is served; neither matching is the by-name refusal."""
    import importlib
    import sys
    (tmp_path / "fake_cc_linear.py").write_text(
        "import torch\nimport torch.nn as nn\ndeepspeed_is_initialized = False\n\n\nclass Linear:\n    precision = None\n\n    def forward(self, input):\n"
        "        d = input.dtype\n"
        "        if self.precision is not None:\n            return None\n"
        "        if d is torch.bfloat16 and not deepspeed_is_initialized:\n"
        "            bias = self.bias.to(dtype=d) if self.bias is not None else None\n"
        "            return nn.functional.linear(input, self.weight.to(dtype=d), bias)\n"
        "        return nn.functional.linear(input, self.weight, self.bias)\n")
    assert all(w in (tmp_path / "fake_cc_linear.py").read_text() for w in castcache.LINEAR_SOURCE)
    (tmp_path / "fake_cc_norm.py").write_text("class LayerNorm:\n    def forward(self, x):\n        return x.float()\n")      # upcasts: no per-call bf16 weight cast
    (tmp_path / "fake_cc_norm2.py").write_text("class Linear:\n    precision = None\n\n    def forward(self, input):\n        return input\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    for n in ("fake_cc_linear", "fake_cc_norm", "fake_cc_norm2"):
        sys.modules.pop(n, None)
    keep = {k: getattr(castcache, k) for k in castcache.CONFIGURABLE}; prev = {k: (list(v) if isinstance(v, list) else v) for k, v in castcache.STATE.items()}
    try:
        castcache.configure(ENV="X_CASTCACHE", M_LINEAR="fake_cc_linear", M_NORM="fake_cc_norm")
        castcache.STATE.update(installed=False, state="off", reason="", patched=[], skipped=[])
        st = castcache.install({"X_CASTCACHE": "1"})
        assert st["state"] == "on" and st["patched"] == ["Linear.forward"] and st["skipped"] == ["layernorm:statement_differs"], dict(st)
        assert " primitives=linear skipped=layernorm:statement_differs " in castcache.census_line()
        assert getattr(importlib.import_module("fake_cc_linear").Linear.forward, "_of3opt_castcache", False)
        castcache.configure(M_LINEAR="fake_cc_norm2", M_NORM="fake_cc_norm")                    # neither statement matches: refused by name, nothing patched
        castcache.STATE.update(installed=False, state="off", reason="", patched=[], skipped=[])
        st = castcache.install({"X_CASTCACHE": "1"})
        assert st["state"] == "refused" and st["reason"] == "statements_differ" and st["patched"] == []
    finally:
        castcache.configure(**keep); castcache.STATE.clear(); castcache.STATE.update(prev)


def test_trunk_graph_storage_signature_follows_the_parameters():
    """The captured stack's parameter / buffer addresses are the replay guard: a moved / re-created weight changes the signature."""
    m = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    m.register_buffer("b0", torch.zeros(3))
    sig = trunk_graph._storage_signature(m)
    assert sig == trunk_graph._storage_signature(m) and len(sig) == 4 + 1
    m[0].weight = torch.nn.Parameter(m[0].weight.detach().clone())                               # a reloaded weight: another storage
    assert trunk_graph._storage_signature(m) != sig


def test_post_release_counts_an_unreadable_token_count():
    prev = {k: (dict(v) if isinstance(v, dict) else v) for k, v in post_release.STATE.items()}
    try:
        fb = post_release.STATE["fallback"]; n0, n1 = fb.get("n_tok_unreadable:no_token_mask", 0), fb.get("n_tok_unreadable:AttributeError", 0)
        assert post_release._n_tok({"token_mask": torch.ones(2, 7)}) == 7
        assert post_release._n_tok({}) is None and fb.get("n_tok_unreadable:no_token_mask") == n0 + 1
        assert post_release._n_tok({"token_mask": 3}) is None and fb.get("n_tok_unreadable:AttributeError") == n1 + 1
    finally:
        post_release.STATE.clear(); post_release.STATE.update(prev)


class _Tuner:                                                                                     # upstream ChunkSizeTuner's two cached fields
    def __init__(self, cached=None, chunk=None):
        self.cached_arg_data, self.cached_chunk_size = cached, chunk


class _Stack:
    def __init__(self, tuner):
        self.chunk_size_tuner = tuner


def test_trunk_graph_captures_only_with_a_settled_tuner():
    """A capture may only record a call the engine's chunk-size tuner will not tune inside: its cached record must equal this call's."""
    s400, z400 = torch.zeros(5, 40, 8), torch.zeros(5, 40, 40, 4)                               # batched: samples folded into the leading dim
    s800, z800 = torch.zeros(1, 80, 8), torch.zeros(1, 1, 80, 80, 4)                            # per sample: batch dims kept
    st = _Stack(_Tuner(cached=(s400.shape, z400.shape), chunk=4))                               # the 3.x engine records the shapes
    assert trunk_graph._tuner_settled(st, s400, z400, 4) and not trunk_graph._tuner_settled(st, s800, z800, 4)
    assert trunk_graph._tuner_settled(st, s800, z800, None) and trunk_graph._tuner_settled(_Stack(None), s800, z800, 4)   # no chunking / no tuner: never tunes
    ob0 = _Stack(_Tuner(cached=((s800.shape, s800.dtype.itemsize), (z800.shape, z800.dtype.itemsize)), chunk=4))   # the 0.5.x fork records (shape, itemsize)
    assert trunk_graph._tuner_settled(ob0, s800, z800, 4) and not trunk_graph._tuner_settled(ob0, s800.half(), z800.half(), 4)
    n_unk = trunk_graph.STATE["fallback"].get("tuner_format_unknown", 0)
    odd = _Stack(_Tuner(cached=(s400.shape,), chunk=4))                                        # a record form this module does not know: never captured on, named
    assert not trunk_graph._tuner_settled(odd, s400, z400, 4) and trunk_graph.STATE["fallback"]["tuner_format_unknown"] == n_unk + 1
    assert not trunk_graph._tuner_settled(_Stack(_Tuner(cached=None, chunk=None)), s400, z400, 4)   # nothing cached yet: the engine tunes on this call


def test_tuner_guard_answers_changed_for_records_of_different_structure(monkeypatch, tmp_path):
    """The engine's comparison raises on a rank change (strict zip) / a type change (assert); the lever answers False so the tuner re-tunes."""
    import sys
    from opt_core.of3_trunk import tuner_guard
    (tmp_path / "fake_chunk_utils.py").write_text(
        "class ChunkSizeTuner:\n"
        "    def _compare_arg_caches(self, ac1, ac2):\n"
        "        consistent = True\n"
        "        for a1, a2 in zip(ac1, ac2, strict=True):\n"
        "            assert type(a1) is type(a2)\n"
        "            if isinstance(a1, (list, tuple)):\n"
        "                consistent &= self._compare_arg_caches(a1, a2)\n"
        "            else:\n"
        "                consistent &= a1 == a2\n"
        "        return consistent\n")
    monkeypatch.syspath_prepend(str(tmp_path)); sys.modules.pop("fake_chunk_utils", None)
    keep = {k: getattr(tuner_guard, k) for k in tuner_guard.CONFIGURABLE}; prev = {k: (dict(v) if isinstance(v, dict) else v) for k, v in tuner_guard.STATE.items()}
    try:
        tuner_guard.configure(ENV="X_TUNER_GUARD", M_CHUNK="fake_chunk_utils")
        tuner_guard.STATE.update(installed=False, state="off", reason="", resets={})
        assert tuner_guard.install({}) is tuner_guard.STATE and tuner_guard.STATE["state"] == "off"      # not requested: nothing patched
        st = tuner_guard.install({"X_TUNER_GUARD": "1"})
        assert st["state"] == "on"
        import fake_chunk_utils as F
        t = F.ChunkSizeTuner()
        b400 = (torch.Size([5, 40, 8]), torch.Size([5, 40, 40, 4])); p800 = (torch.Size([1, 80, 8]), torch.Size([1, 1, 80, 80, 4]))
        assert t._compare_arg_caches(b400, b400) is True and t._compare_arg_caches(b400, (torch.Size([5, 60, 8]), torch.Size([5, 60, 60, 4]))) is False
        assert st["resets"] == {}                                                                # comparable records: the engine's own answer
        assert t._compare_arg_caches(b400, p800) is False and st["resets"] == {"ValueError": 1}   # a rank change: 'changed' instead of ValueError
        assert t._compare_arg_caches((torch.Size([1]), 3), (torch.Size([1]), "x")) is False and st["resets"]["AssertionError"] == 1
        assert " state=on resets=AssertionError:1,ValueError:1" in tuner_guard.census_line()
        assert getattr(F.ChunkSizeTuner._compare_arg_caches, "_of3opt_tuner_guard", False)
        tuner_guard.STATE["installed"] = False; tuner_guard.install({"X_TUNER_GUARD": "1"})    # idempotent: not wrapped twice
        assert F.ChunkSizeTuner._compare_arg_caches.__wrapped__.__name__ == "_compare_arg_caches" and not hasattr(F.ChunkSizeTuner._compare_arg_caches.__wrapped__, "_of3opt_tuner_guard")
        tuner_guard.configure(M_CHUNK="no_such_engine_module"); tuner_guard.STATE.update(installed=False, state="off", reason="")
        assert tuner_guard.install({"X_TUNER_GUARD": "1"})["state"] == "refused" and tuner_guard.STATE["reason"].startswith("no_tuner:")
    finally:
        tuner_guard.configure(**keep); tuner_guard.STATE.clear(); tuner_guard.STATE.update(prev)
