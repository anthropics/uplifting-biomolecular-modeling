"""The trunk CUDA-graph lever (opt/forward/graph/boltz_graph_trunk.py + boltz2_opt.graph): the torch-free / CPU-checkable surface —
spec parsing, the stock-signature binder, the fail-closed verdict, the adapter's LEVER line, the mode table's words and attachments, and the
run-report evidence rule. The capture / replay mechanics are GPU-only (proven on a GPU by bit-comparing every output file with stock's)."""
import importlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "graph"))


@pytest.fixture()
def M(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.syspath_prepend(GRAPH_DIR)
    sys.modules.pop("boltz_graph_trunk", None)
    m = importlib.import_module("boltz_graph_trunk")
    # a clean slate per test (module-level state)
    m.STATS.update({"census": {}, "captures": [], "errors": {}, "generations_evicted": 0, "pool_gib_max": 0.0, "static_gib": 0.0, "prime_rng_checks": 0})
    m._STATE.update({"applied": False, "units": (), "orig": {}, "refused": {}, "templ_patched": []})
    yield m
    sys.modules.pop("boltz_graph_trunk", None)


def test_parse_units(M):
    assert M.parse_units(None) == () and M.parse_units("") == () and M.parse_units("off") == () and M.parse_units("0") == ()
    assert M.parse_units("all") == M.UNITS == ("pf", "pfnoseq", "msa", "templ", "sovl")
    assert M.parse_units("pf, msa") == ("pf", "msa") and M.parse_units("pf,pf,templ") == ("pf", "templ")
    assert M.parse_units("pf,tailgraph") == ("pf", "tailgraph")      # a lowercase word outside UNITS: a unit another lever registers (register_unit)
    with pytest.raises(ValueError):
        M.parse_units("pf,Bogus-Word")


def test_register_unit_contract(M):
    """register_unit: pending before apply, refused by name on a source pin mismatch, the gate names a listed-but-never-registered unit."""
    class Toy:
        def forward(self, x, feats=None, flag=1):
            return x
    assert M.register_unit("toy", Toy, "forward", tensor_args=("x",), dict_args=("feats",), tokens_of=lambda b: 1) == "pending"
    assert "toy" in M._REGISTERED and M._REGISTERED["toy"]["params"] == ("self", "x", "feats", "flag") and M._REGISTERED["toy"]["defaults"] == {"feats": None, "flag": 1}
    assert M.register_unit("toy2", Toy, "forward", tensor_args=("x",), tokens_of=lambda b: 1, src_sha256="0" * 64) == "refused:source_mismatch"
    assert M._STATE["refused"].get("toy2") == "source_mismatch"
    M._STATE["refused"].pop("toy2"); M._REGISTERED.clear()
    # listed in the row but never registered -> the verdict names it
    M._STATE.update(applied=True, units=("pf",), wanted=("pf", "ghost"))
    try:
        v = M.verdict()
        assert v["ok"] is False and v["reason"] == "listed_not_registered:ghost"
    finally:
        M._STATE.update(applied=False, units=(), wanted=())


def test_bind_uses_the_pinned_parameter_names_not_the_wrapped_callable(M):
    P = ("s", "z", "mask", "pair_mask", "use_kernels"); D = {"use_kernels": False}
    b = M._bind(P, D, "SELF", (1, 2, 3, 4), {})
    assert list(b) == ["self", "s", "z", "mask", "pair_mask", "use_kernels"] and b["use_kernels"] is False and b["self"] == "SELF"
    assert M._bind(P, D, "S", (1, 2), {"mask": 3, "pair_mask": 4, "use_kernels": True})["use_kernels"] is True
    assert M._bind(P, D, "S", (1, 2, 3), {}) is None                      # pair_mask missing, no default
    assert M._bind(P, D, "S", (1, 2, 3, 4, 5, 6), {}) is None             # too many positionals
    assert M._bind(P, D, "S", (1, 2, 3, 4), {"z": 9}) is None             # duplicate
    assert M._bind(P, D, "S", (1, 2, 3, 4), {"chunk": 9}) is None         # unknown name


def test_verdict_is_fail_closed(M):
    assert M.verdict()["ok"] is False                                     # nothing applied
    M._STATE.update({"applied": True, "units": ("pf", "msa", "templ")})
    v = M.verdict(); assert v["ok"] and v["idle"]                          # applied, no call yet: idle (installed behind its gate)
    M._cnt("pf", "eager:above_max_tokens", 8); v = M.verdict(); assert v["ok"] and v["idle"]
    M._cnt("pf", "eager:warm", 1); M._cnt("pf", "captured", 1); M._cnt("pf", "replayed", 6)
    v = M.verdict(); assert v["ok"] and not v["idle"]
    M.STATS["errors"]["pf:RuntimeError"] = 1
    v = M.verdict(); assert not v["ok"] and v["reason"].startswith("capture_error:pf:RuntimeError")
    M.STATS["errors"].clear(); M._STATE["refused"]["MSAModule.forward"] = "source_mismatch:abc"
    assert not M.verdict()["ok"] and "refused_at_apply" in M.verdict()["reason"]
    M._STATE["refused"].clear()
    # engaged (many warm calls) and never replayed: refused
    M.STATS["census"] = {"pf": {"eager:warm": 9}}
    assert M.verdict() == {"ok": False, "idle": False, "reason": "engaged_never_replayed"}


def test_source_digests_are_pinned_and_match_the_vendored_stock(M):
    """The digests the lever checks at apply equal sha256[:16] of the def blocks in stock/src (what inspect.getsource returns in the worker)."""
    import ast, hashlib
    assert all(len(v) == 16 for v in M.SRC_SHA.values()), M.SRC_SHA
    assert set(M.SRC_SHA) == {"TemplateV2Module.forward", "TemplateModule.forward", "PairformerModule.forward", "PairformerNoSeqModule.forward", "MSAModule.forward", "PairformerLayer.forward"}
    stock = os.path.join(HERE, "..", "..", "..", "stock", "src", "boltz", "model")
    want = {}
    for rel, classes in (("layers/pairformer.py", ("PairformerModule", "PairformerNoSeqModule", "PairformerLayer")), ("modules/trunkv2.py", ("TemplateV2Module", "TemplateModule", "MSAModule"))):
        src = open(os.path.join(stock, rel)).read().splitlines(keepends=True); tree = ast.parse("".join(src))
        for cls in [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in classes]:
            fn = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "forward"][0]
            want[cls.name + ".forward"] = hashlib.sha256("".join(src[fn.lineno - 1:fn.end_lineno]).encode()).hexdigest()[:16]
    assert want == M.SRC_SHA
    # unit sovl knows the stock layer forward and the resid lever's restatement (boltz_trunk_levers._pf_layer_forward), by digest
    lev = open(os.path.join(HERE, "..", "..", "forward", "trunk_levers", "boltz_trunk_levers.py")).read().splitlines(keepends=True)
    fn = [n for n in ast.parse("".join(lev)).body if isinstance(n, ast.FunctionDef) and n.name == "_pf_layer_forward"][0]
    sha = hashlib.sha256("".join(lev[fn.lineno - 1:fn.end_lineno]).encode()).hexdigest()[:16]
    assert M.SOVL_LAYER_FORWARDS == {M.SRC_SHA["PairformerLayer.forward"]: "stock", sha: "boltz_trunk_levers._pf_layer_forward"}


def test_sovl_requires_pf_and_its_skips_refuse_the_gate(M):
    M._STATE.update({"applied": True, "units": ("pf", "sovl"), "sovl": True})
    M._cnt("pf", "replayed", 4); M._cnt("sovl", "calls", 4)
    assert M.verdict()["ok"]
    M._cnt("sovl", "skipped:unknown_layer_forward:f:0123", 1)
    v = M.verdict(); assert not v["ok"] and v["reason"].startswith("sovl_skipped:")


def test_adapter_line_and_report_without_torch_state(M, monkeypatch):
    from boltz2_opt import graph as G
    monkeypatch.setitem(G._STATE, "mod", M); monkeypatch.setitem(G._STATE, "applied", ["graph_trunk"])
    M._STATE.update({"applied": True, "units": ("pf", "msa")}); M._cnt("pf", "replayed", 3); M._cnt("pf", "captured", 1); M._cnt("pf", "eager:warm", 1); M._cnt("msa", "eager:above_max_tokens", 2)
    line = G.line()
    assert line.startswith("[boltz2-opt] LEVER name=graph_trunk state=on units=pf,msa replayed=3 captured=1 eager=3 eager_by=above_max_tokens:2,warm:1")
    assert "gate=ok" in line and "errors=none" in line
    r = G.report()
    assert r["applied"] == ["graph_trunk"] and r["units"] == ["pf", "msa"] and r["gate"]["ok"] and r["census"]["pf"]["replayed"] == 3
    assert G.requested({"BOLTZ_GRAPH_TRUNK": "pf"}) and not G.requested({"BOLTZ_GRAPH_TRUNK": "off"}) and not G.requested({})


def test_mode_rows_carry_the_lever_at_one_gpu_and_name_it_off_at_n_gpu_above_one():
    from boltz2_opt import modes, registry, worker_launch
    for m in ("exact", "fast"):
        env = modes.env_row(m)
        want = {"pf", "pfnoseq", "templ"}                                       # both rows: under the fast row's PAIRFUSE driver pf / pfnoseq capture the DRIVER's bodies (BODIES table)
        assert set(env["BOLTZ_GRAPH_TRUNK"].split(",")) == want and int(env["BOLTZ_GRAPH_TRUNK_MAX_TOKENS"]) >= 200, env
        assert "graph" in modes.attachments(m) and "graph_trunk" in modes.resolve(m)["levers"]
    env = modes.env_row("big")                                                 # big TABLES the lever as fast does and names it off by rule (no CUDA graphs in the memory mode):
    rb = modes.resolve("big")                                                    # its words and the `graph` attachment leave the resolved row BY NAME, recorded under `off`
    assert "graph_trunk" in modes.MODES["big"]["levers"] and "graph_trunk" not in rb["levers"] and "graph" not in rb["attach"] and not [k for k in env if k.startswith("BOLTZ_GRAPH_TRUNK")]
    assert rb["off"]["graph_trunk"].startswith("rule:") and modes.off_levers(rb)["graph_trunk"] == rb["off"]["graph_trunk"]
    try:                                                                           # at n_gpu > 1 the Pairformer stacks are the row-sharded driver's statements: the lever leaves BY NAME (modes.TP_DROPS)
        modes.set_n_gpu(2)
        r2 = modes.resolve("big")
        assert "graph_trunk" not in r2["levers"] and "graph" not in r2["attach"] and "BOLTZ_GRAPH_TRUNK" not in r2["env"] and r2["tp_off"] == dict(modes.TP_DROPS["big"]) and "graph_trunk" not in r2.get("off", {}) and set(r2["tp_off"]) == {"graph_trunk", "rollout", "dit_hoist", "align_jacobi64", "dit_fused", "templ_skip", "pairblock", "pairblock_c64", "fused_transition", "pairfuse", "atom_fused", "atom_gemm", "writer_overlap"} and "sampler" not in r2["attach"] and "BOLTZ_SAMPLER_DIT" not in r2["env"]
        assert not ({"pairblock", "fused_transition", "pairfuse"} & set(r2["levers"])) and not ({"transition", "pairblock", "pairfuse"} & set(r2["attach"])) and not ({"BOLTZ_PAIRBLOCK", "BOLTZ_TRANSITION", "BOLTZ_PAIRFUSE"} & set(r2["env"])) and "flash_triattn" in r2["levers"]   # the fused pair track leaves the xP line by name; the flash core stays: the row-sharded trunk's row-block attention (TP_EXPORTS ROWPAIR_TRIATT_CORE)
        assert modes.off_levers(r2)["graph_trunk"].startswith("replaced_by_rowpair:")
    finally:
        modes.set_n_gpu(1)
    assert "tp_off" not in modes.resolve("big")
    L = registry.LEVERS["graph_trunk"]
    assert L["switch"] == "BOLTZ_GRAPH_TRUNK" and L["tier"] == 1 and L["origin"] == "kit" and L["file"] == "forward/graph/boltz_graph_trunk.py"
    assert worker_launch.ATTACH["graph"] == {"module": "boltz2_opt.graph", "trigger": "boltz.model.models.boltz2", "report_key": "graph_report"}
    assert os.path.isfile(os.path.join(GRAPH_DIR, "boltz_graph_trunk.py"))


def test_the_switch_family_is_stripped_from_callers():
    import json
    pins = json.load(open(os.path.join(HERE, "..", "..", "..", "stock", "PINS.json")))
    prefixes = pins["stock_environment"]["must_be_absent_prefixes"]
    assert any("BOLTZ_GRAPH_TRUNK".startswith(p) for p in prefixes), prefixes
