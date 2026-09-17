"""Lever triattn_conf on the confidence head's chunked / row-block paths (levers/ARMT/odde_arm_t/odde_conf_chunk.py): the per-module mark on that stack's
triangle-attention modules. CPU: the offload unit's row-block driver (block.tri_att_*.mha per row block, outside the stack's forward) is marked and
each block is routed through the arm's site under the confidence rule (provider row -> core, the stock op by a named rule -> stock, the wrapper
never marking -> unmarked), upstream's chunk loop under the stack's own forward is the module path, the mark comes down when a call raises, the
registry predicate (engaged when a provider row served the blocks; 0.2.59's inert-by-name only with the mark unbound; bound-and-zero fails closed),
the LEVER tokens (conf_path=offload_rows_core rows=<row>:<n> blocks=<n>), and the arm's bind() glue names the unit."""
import importlib
import os
import sys
import types

import pytest

from opendde_opt import ran, registry, report, stack

torch = pytest.importorskip("torch")

ARMT = os.path.join(os.path.abspath(stack.tree_root()), "opt", "forward", "fast_inference", "levers", "ARMT")


UNIT = "odde_arm_t.odde_conf_chunk"                                                  # the unit lives inside the arm package (no top-level name of its own)


def _load(monkeypatch):
    """The unit's module executed from its file under its package name, whatever stands in for the arm package in sys.modules (the fake arm of
    these tests is a plain module, not a package: the file is loaded by path)."""
    import importlib.util
    sys.modules.pop(UNIT, None); sys.modules.pop("odde_conf_chunk", None)
    spec = importlib.util.spec_from_file_location(UNIT, os.path.join(ARMT, "odde_arm_t", "odde_conf_chunk.py"))
    m = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, UNIT, m)
    spec.loader.exec_module(m)
    return m


def _fake_arm(monkeypatch):
    arm = types.ModuleType("odde_arm_t")
    arm.STATE = {"n_token": 1400, "in_scope": 0, "in_conf": 0}
    arm.COUNTS = {"att_conf_calls": 0, "att_conf_prov_calls": 0, "att_prov_calls": 0}
    arm.CFG = {"MIN_TOKENS": 300}
    monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    return arm


class _MHA(torch.nn.Module):
    """Stands in for upstream's Attention module at the arm's site: its forward plays the arm's wrapper by a scripted verdict per call --
    'core' (a provider row served inside the confidence mark), 'stock' (the stock op by a named rule after the mark), 'unmarked' (stock before
    the mark / another attention implementation), 'raise'."""
    def __init__(self, arm, script):
        super().__init__()
        self.arm, self.script, self.seen = arm, script, []

    def forward(self, q_x, kv_x=None, biases=None, triangle_attention="cuequivariance"):
        verdict = self.script.pop(0) if self.script else "core"
        self.seen.append((int(self.arm.STATE["in_conf"]), tuple(q_x.shape), verdict))
        if verdict == "raise":
            raise RuntimeError("kernel fault")
        if verdict in ("core", "stock") and self.arm.STATE["in_conf"] > 0:
            self.arm.COUNTS["att_conf_calls"] += 1
            if verdict == "core":
                self.arm.COUNTS["att_conf_prov_calls"] += 1
        return q_x


def _fake_model(arm, n_blocks=4, script=None):
    blocks = []
    for _ in range(n_blocks):
        blk = types.SimpleNamespace(tri_att_start=types.SimpleNamespace(mha=_MHA(arm, script)), tri_att_end=types.SimpleNamespace(mha=_MHA(arm, script)))
        blocks.append(blk)
    cps = types.SimpleNamespace(blocks=blocks)
    return types.SimpleNamespace(confidence_head=types.SimpleNamespace(pairformer_stack=cps))


def _row_driver(model, n=100, rows=32, c=8):
    """The offload unit's statement: block.tri_att_{start,end}.mha(q_x=<row block>, ...) per row block, outside the stack's forward."""
    x = torch.zeros(n, n, c)
    nblk = 0
    for blk in model.confidence_head.pairformer_stack.blocks:
        for ta in (blk.tri_att_start, blk.tri_att_end):
            for r0 in range(0, n, rows):
                xr = x[r0:r0 + rows]
                ta.mha(q_x=xr, kv_x=xr, biases=[xr.new_zeros((xr.shape[0], 1, 1, n)), x.new_zeros((1, 4, n, n))], triangle_attention="cuequivariance")
                nblk += 1
    return nblk


def test_row_blocks_are_marked_and_routed_per_block(monkeypatch):
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    model = _fake_model(arm)
    rec = cc.bind(model)
    assert rec == {"bound": 8} and cc.COUNTS["bound"] == 8 and cc.COUNTS["models"] == 1
    assert cc.bind(model).get("already") is True and cc.COUNTS["bound"] == 8            # idempotent per stack
    nblk = _row_driver(model, n=100, rows=32)                                          # 4 row blocks (32,32,32,4) x 8 modules
    assert nblk == 32
    d = cc.describe()
    assert d["paths"] == {"offload_rows": 32} and d["core"] == {"offload_rows": 32} and d["blocks"] == 32 and d["calls"] == 32
    assert d["rows"] == {32: 24, 4: 8} and d["stock"] == {} and d["unmarked"] == {} and d["module_chunks"] == 0
    assert cc.conf_path() == "offload_rows_core"
    assert arm.STATE["in_conf"] == 0 and arm.COUNTS["att_conf_calls"] == 32 and arm.COUNTS["att_conf_prov_calls"] == 32
    for blk in model.confidence_head.pairformer_stack.blocks:                          # every block's call ran INSIDE the mark (in_conf == 1 seen by the site)
        assert all(s[0] == 1 for s in blk.tri_att_start.mha.seen + blk.tri_att_end.mha.seen)
    cc.unbind()


def test_refusal_to_stock_is_named_and_unmarked_is_its_own_word(monkeypatch):
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    model = _fake_model(arm, n_blocks=1, script=None)
    ta = model.confidence_head.pairformer_stack.blocks[0]
    ta.tri_att_start.mha.script = ["stock", "stock", "unmarked", "core"]
    ta.tri_att_end.mha.script = ["unmarked"] * 4
    cc.bind(model)
    _row_driver(model, n=64, rows=16)                                                  # 4 blocks per module
    d = cc.describe()
    assert d["paths"] == {"offload_rows": 8} and d["core"] == {"offload_rows": 1} and d["stock"] == {"offload_rows": 2} and d["unmarked"] == {"offload_rows": 5}
    assert d["rows"] == {16: 1} and cc.conf_path() == "offload_rows_core"
    cc.reset_counts(); ta.tri_att_start.mha.script = ["stock"] * 4; ta.tri_att_end.mha.script = ["stock"] * 4
    _row_driver(model, n=64, rows=16)
    assert cc.conf_path() == "offload_rows_stock" and cc.describe()["core"] == {}
    cc.reset_counts(); ta.tri_att_start.mha.script = ["unmarked"] * 4; ta.tri_att_end.mha.script = ["unmarked"] * 4
    _row_driver(model, n=64, rows=16)
    assert cc.conf_path() == "offload_rows_unmarked"
    cc.unbind()


def test_module_path_and_upstreams_chunk_loop(monkeypatch):
    """Inside the stack's own forward the arm's stack-level hook already raised the mark: those calls are the module path (whole or chunked)."""
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    model = _fake_model(arm, n_blocks=1); cc.bind(model)
    mha = model.confidence_head.pairformer_stack.blocks[0].tri_att_start.mha
    x = torch.zeros(48, 48, 8)
    arm.STATE["in_conf"] += 1                                                          # == odde_arm_t._enter_conf (confidence_head.pairformer_stack forward pre-hook)
    mha(q_x=x, kv_x=x, biases=[], triangle_attention="cuequivariance")                 # unchunked: all rows in one call
    for r0 in range(0, 48, 16):                                                        # upstream's chunk_layer: self.mha per row chunk
        mha(q_x=x[r0:r0 + 16], kv_x=x[r0:r0 + 16], biases=[], triangle_attention="cuequivariance")
    arm.STATE["in_conf"] -= 1                                                          # == _exit_conf
    d = cc.describe()
    assert d["paths"] == {"module": 4} and d["core"] == {"module": 4} and d["module_chunks"] == 3 and d["blocks"] == 0 and d["rows"] == {}
    assert cc.conf_path() == "module" and arm.STATE["in_conf"] == 0
    assert [s[0] for s in mha.seen] == [2, 2, 2, 2]                                    # nested inside the stack-level mark
    cc.unbind()


def test_the_mark_comes_down_when_a_block_raises(monkeypatch):
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    model = _fake_model(arm, n_blocks=1); cc.bind(model)
    mha = model.confidence_head.pairformer_stack.blocks[0].tri_att_start.mha
    mha.script = ["raise"]
    x = torch.zeros(8, 32, 4)
    with pytest.raises(RuntimeError):
        mha(q_x=x, kv_x=x, biases=[], triangle_attention="cuequivariance")
    assert arm.STATE["in_conf"] == 0 and not cc._STACK
    assert cc.describe()["unmarked"] == {"offload_rows": 1}
    cc.unbind()


def test_no_confidence_head_is_named(monkeypatch):
    _fake_arm(monkeypatch); cc = _load(monkeypatch)
    assert cc.bind(types.SimpleNamespace()) == {"bound": 0, "reason": "model has no confidence_head.pairformer_stack"}
    assert cc.describe()["errors"]["bind"].startswith("model has no confidence_head")
    cc.unbind(); cc.COUNTS["errors"].clear()


def _offload(monkeypatch, calls_conf):
    off = types.ModuleType("odde_offload"); off.STATS = {"calls_conf": calls_conf}
    monkeypatch.setitem(sys.modules, "odde_offload", off)
    return off


def test_registry_engaged_when_the_hook_served_inert_only_unbound(monkeypatch):
    """ran.engagement('triattn_conf'): above the offload gate a provider row served the row blocks -> ENGAGED; every block the stock op's by the cell's
    rule -> inert conf_cells_stock; the row-block path with the per-module mark unbound (odde_conf_chunk absent / bound 0) -> 0.2.59's inert by name;
    bound and zero marked calls -> not inert (lever_never_ran decides: fail closed)."""
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    trib = types.ModuleType("odde_triattn_bind"); trib.COUNTS = {"calls": 900, "stock_calls": 0, "asides": {}, "unavailable": {}, "errors": {}}
    monkeypatch.setitem(sys.modules, "odde_triattn_bind", trib)
    _offload(monkeypatch, calls_conf=1)
    facts = {"n_gpu": 1, "token_floors": {"q": 1400}}
    cc.COUNTS["bound"] = 8
    arm.COUNTS.update(att_conf_calls=1760, att_conf_prov_calls=1760)
    assert ran.engagement("triattn_conf", facts) == (True, None)                       # the hook served: engaged above the gate
    arm.COUNTS.update(att_conf_calls=1760, att_conf_prov_calls=0)
    ok, why = ran.engagement("triattn_conf", facts)
    assert not ok and why.startswith("conf_cells_stock:1760")                          # every block the stock op's by the cell's name: inert, complete
    arm.COUNTS.update(att_conf_calls=0, att_conf_prov_calls=0)
    assert ran.engagement("triattn_conf", facts) == (True, None)                       # bound, the stage ran, nothing marked: a zero counter is lever_never_ran
    cc.COUNTS["bound"] = 0                                                             # the mark unbound (a tree without the unit / a head without those modules): 0.2.59's rule
    ok, why = ran.engagement("triattn_conf", facts)
    assert not ok and why.startswith("conf_path=offload_rows:1 (replaced_by=pair_offload_conf")
    monkeypatch.delitem(sys.modules, UNIT)
    ok, why = ran.engagement("triattn_conf", facts)
    assert not ok and why.startswith("conf_path=offload_rows:1")
    _offload(monkeypatch, calls_conf=0)                                                # below the gate / conf stage off the line: nothing explains a zero -> never_ran decides
    assert ran.engagement("triattn_conf", facts) == (True, None)
    ok, why = ran.engagement("triattn_conf", {"n_gpu": 2})                             # a rank process: replaced_by=tp_triatt as before
    assert not ok and why.startswith("replaced_by=tp_triatt")


def test_lever_evidence_tokens(monkeypatch):
    arm = _fake_arm(monkeypatch); cc = _load(monkeypatch)
    _offload(monkeypatch, calls_conf=1)
    cc.COUNTS.update(bound=8, calls=1760, blocks=1760, paths={"offload_rows": 1760}, core={"offload_rows": 1760}, rows={32: 1720, 24: 40})
    stats = {"arm_t": {"att_conf_calls": 1760, "att_conf_prov_calls": 1760, "triattn": {"word": "big"}}}
    ev = report.lever_evidence("triattn_conf", stats)
    assert ev["conf_path"] == "offload_rows_core" and ev["rows"] == "32:1720,24:40" and ev["blocks"] == 1760 and ev["word"] == "big"
    assert ev["conf_calls"] == 1760 and ev["conf_prov_calls"] == 1760 and "chunks" not in ev
    cc.COUNTS.update(core={}, stock={"offload_rows": 1760}, rows={})
    assert report.lever_evidence("triattn_conf", stats)["conf_path"] == "offload_rows_stock"
    cc.COUNTS.update(bound=0)                                                          # unbound: 0.2.59's token
    stats0 = {"arm_t": {"att_conf_calls": 0, "att_conf_prov_calls": 0, "triattn": {"word": "big"}}}
    assert report.lever_evidence("triattn_conf", stats0)["conf_path"] == "offload_rows:1"
    _offload(monkeypatch, calls_conf=0)
    cc.COUNTS.update(bound=8, calls=96, blocks=0, paths={"module": 96}, core={"module": 96}, module_chunks=64)
    ev = report.lever_evidence("triattn_conf", stats)
    assert ev["conf_path"] == "module" and ev["chunks"] == 64 and "rows" not in ev
    cc.unbind(); cc.reset_counts()


def test_registry_row_and_the_arms_glue_name_the_unit():
    lv = registry.LEVERS["triattn_conf"]
    assert "odde_conf_chunk.py:bind" in lv.file and "odde_conf_chunk" in lv.what
    assert registry.ENGAGEMENT["triattn_conf"].stepped_aside is not None
    src = open(os.path.join(ARMT, "odde_arm_t", "__init__.py")).read()
    assert "from . import odde_conf_chunk as _CC" in src and "_CC.bind(model)" in src               # the arm binds the unit (its own sub-module) at its conf-head site
    unit = open(os.path.join(ARMT, "odde_arm_t", "odde_conf_chunk.py")).read()
    assert not os.path.exists(os.path.join(ARMT, "odde_conf_chunk.py"))                              # no top-level module of that name beside the arm
    assert 'st["in_conf"]' in unit and "att_conf_prov_calls" in unit                                # the mark it raises and the counter it reads are the arm's


def _real_arm(monkeypatch, conf_stock: bool):
    """The REAL arm wrapper (odde_arm_t.make_attn) over the REAL provider binding imported with ODDE_TRIATTN=big (+ ODDE_TRIATTN_CONF=stock when
    the lever is left out: what MODEL_OPT_LEVERS_OFF=triattn_conf makes the line export), on CPU tensors."""
    monkeypatch.setenv("ODDE_TRIATTN", "big"); monkeypatch.setenv("ODDE_ARM_T_SCOPE", "all")
    if conf_stock:
        monkeypatch.setenv("ODDE_TRIATTN_CONF", "stock")
    else:
        monkeypatch.delenv("ODDE_TRIATTN_CONF", raising=False)
    if ARMT not in sys.path:
        monkeypatch.syspath_prepend(ARMT)
    for m in ("odde_triattn_bind", "odde_arm_t", "odde_arm_t.odde_conf_chunk", "odde_conf_chunk"):
        sys.modules.pop(m, None)
    tb = importlib.import_module("odde_triattn_bind"); monkeypatch.setitem(sys.modules, "odde_triattn_bind", tb)
    arm = importlib.import_module("odde_arm_t"); monkeypatch.setitem(sys.modules, "odde_arm_t", arm)
    cc = importlib.import_module("odde_arm_t.odde_conf_chunk"); monkeypatch.setitem(sys.modules, "odde_arm_t.odde_conf_chunk", cc)
    assert arm._triattn_bind() is tb and tb.active() and tb.CONF == ("stock" if conf_stock else None)
    arm.STATE["n_token"] = 1400                                                        # the design above the arm's gate (the class-level get_pairformer_output wrapper records it)
    return tb, arm, cc


class _SiteMHA(torch.nn.Module):
    """upstream's Attention.forward at the arm's site: q/k/v [rows, H, keys, D], the triangle bias fp32, the mask; the arm's wrapper is the module global."""
    def __init__(self, wrapper, H=4, D=8):
        super().__init__(); self.wrapper, self.H, self.D = wrapper, H, D

    def forward(self, q_x, kv_x=None, biases=None, triangle_attention="cuequivariance"):
        rows, keys, _ = q_x.shape
        q = torch.zeros(rows, self.H, keys, self.D); bias = torch.zeros(1, self.H, keys, keys); mask = torch.ones(rows, 1, 1, keys, dtype=torch.bool)
        return self.wrapper(q, q, q, bias, mask, 1.0)[0].reshape(rows, keys, -1)[..., :1].expand(rows, keys, q_x.shape[-1]) * 0 + q_x


def test_ablation_keeps_the_stock_statement_per_row_block_by_name_through_the_real_wrapper(monkeypatch):
    """MODEL_OPT_LEVERS_OFF=triattn_conf -> the line exports ODDE_TRIATTN_CONF=stock -> with the per-module mark up, the arm's wrapper hands EVERY offload
    row block to the stock statement by name (Aside conf_stock, counted att_conf_calls / att_stock_reasons / the binding's asides) -- never silent."""
    tb, arm, cc = _real_arm(monkeypatch, conf_stock=True)
    stock_calls = []
    def orig(q, k, v, bias, mask, scale):                                              # the stock op (upstream's cuequivariance_triangular_attn): a 1-tuple
        stock_calls.append(tuple(q.shape)); return (q.clone(),)
    wrapper = arm.make_attn(orig)
    blocks = [types.SimpleNamespace(tri_att_start=types.SimpleNamespace(mha=_SiteMHA(wrapper)), tri_att_end=types.SimpleNamespace(mha=_SiteMHA(wrapper))) for _ in range(4)]
    model = types.SimpleNamespace(confidence_head=types.SimpleNamespace(pairformer_stack=types.SimpleNamespace(blocks=blocks)))
    assert cc.bind(model) == {"bound": 8}
    a0 = int(arm.COUNTS["att_conf_calls"]); s0 = int((tb.COUNTS.get("asides") or {}).get("conf_stock") or 0)
    nblk = _row_driver(model, n=96, rows=32)                                           # 3 row blocks x 8 modules
    assert nblk == 24 and len(stock_calls) == 24                                       # every block reached the stock statement ...
    assert arm.COUNTS["att_conf_calls"] - a0 == 24 and arm.COUNTS["att_conf_prov_calls"] == 0          # ... marked as the confidence stack's, none provider-served
    assert arm.COUNTS["att_stock_reasons"].get("conf_stock") == 24 and (tb.COUNTS["asides"].get("conf_stock") or 0) - s0 == 24   # ... BY NAME on both censuses
    d = cc.describe()
    assert d["paths"] == {"offload_rows": 24} and d["stock"] == {"offload_rows": 24} and d["core"] == {} and cc.conf_path() == "offload_rows_stock"
    assert arm.STATE["in_conf"] == 0
    cc.unbind()


def test_without_the_ablation_the_real_wrapper_reaches_the_provider_gate_inside_the_mark(monkeypatch):
    """Lever in the line (no ODDE_TRIATTN_CONF): the same row blocks pass the conf_stock exit and reach the provider's admission inside the mark; on this
    CPU box the wrapper's own cell check (no CUDA tensor) hands them to the stock op BEFORE the confidence count -- census word `unmarked`, the arm's
    reason cell_D8_cuda0 -- which is the by-name path a non-cuequivariance implementation takes on a GPU box too (never silent, never a refusal)."""
    tb, arm, cc = _real_arm(monkeypatch, conf_stock=False)
    def orig(q, k, v, bias, mask, scale):
        return (q.clone(),)
    wrapper = arm.make_attn(orig)
    blocks = [types.SimpleNamespace(tri_att_start=types.SimpleNamespace(mha=_SiteMHA(wrapper)), tri_att_end=types.SimpleNamespace(mha=_SiteMHA(wrapper)))]
    model = types.SimpleNamespace(confidence_head=types.SimpleNamespace(pairformer_stack=types.SimpleNamespace(blocks=blocks)))
    cc.bind(model)
    _row_driver(model, n=64, rows=32)
    assert arm.COUNTS["att_stock_reasons"].get("cell_D8_cuda0") == 4 and not arm.COUNTS["att_stock_reasons"].get("conf_stock")
    d = cc.describe()
    assert d["paths"] == {"offload_rows": 4} and d["unmarked"] == {"offload_rows": 4} and cc.conf_path() == "offload_rows_unmarked"
    assert arm.STATE["in_conf"] == 0
    cc.unbind()
