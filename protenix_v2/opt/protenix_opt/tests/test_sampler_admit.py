"""CPU tests of the sampler admission policy (fpf_clisampler.policy): the routes engage where the fitted model says they fit under each row's word,
the words are validated by name, and the mode rows carry the levers and the words."""
import importlib, os, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(KIT, "opt", "forward", "flashpairformer", "src"))


def _policy(admit="item", release="item", hoist_eager="1"):
    os.environ["PTX_SAMPLER_ADMIT"] = admit; os.environ["PTX_SAMPLER_RELEASE"] = release; os.environ["PTX_SAMPLER_HOIST_EAGER"] = hoist_eager
    sys.modules.pop("fpf_clisampler.policy", None)
    import fpf_clisampler.policy as P
    return importlib.reload(P)


@pytest.mark.parametrize("n_tok,n_atom,alloc,trunk_peak,expect", [
    (1400, 11224, 5.4, 14.8, "hoist_eager"),     # ref = confidence 5.4+10.5 = 15.9 (+1); hoist 5.4+6.9+0.4 = 12.7 fits; graph +12.2 = 24.8 does not
    (2000, 16056, 8.5, 24.5, "hoist_eager"),     # ref = confidence 8.5+21.4 = 29.9 (+1); hoist 8.5+14.0+0.8 = 23.3 fits (graph: cap)
    (2530, 18760, 12.1, 35.1, "hoist_eager"),    # ref = trunk 35.1 (+1) [confidence 12.1+18.6 = 30.7]; hoist 12.1+22.4+1.3 = 35.8 <= 36.1 (measured 35.3)
    (3000, 23658, 16.0, 45.35, "stock"),         # ref = trunk 45.35 (+1); hoist 16+31.5+9.0 = 56.5 > 46.35 (measured 56.1) -> the stock sampler
    (4000, 32088, 26.3, 65.9, "stock"),
])
def test_item_word_h100(n_tok, n_atom, alloc, trunk_peak, expect):
    """The big row (word item): HOIST_EAGER, HOIST_EAGER, STOCK, STOCK at 1400/2000/3000/4000 tokens."""
    P = _policy(admit="item")
    adm = P.admit(n_tok=n_tok, n_atom=n_atom, allocated_gib=alloc, trunk_peak_gib=trunk_peak, total_gib=79.2, graph_cap=0, hoist_bound=True, say=False)
    assert adm.route == expect, adm
    t = adm.terms
    assert t["graph"] > t["hoist_eager"] > alloc and adm.limit_gib >= trunk_peak + P.ADMIT_MODEL["slack_gib"] - 1e-9


@pytest.mark.parametrize("n_tok,n_atom,alloc,expect", [
    (1400, 11224, 5.4, "graph"),                  # 0.85 x 79.2 - 1 = 66.3: graph+hoist 24.9 fits
    (2000, 16056, 8.5, "graph"),                  # graph+hoist 48.1 fits (cap lifted in this test)
    (2530, 18760, 12.1, "hoist_eager"),           # graph+hoist 75.5 > 66.3; hoist 35.8 fits
    (3000, 23658, 16.0, "hoist_eager"),           # hoist 56.5 fits
    (4000, 32088, 26.3, "stock"),                 # hoist 26.3+56+16 = 98 > 66.3
])
def test_memory_word_h100(n_tok, n_atom, alloc, expect):
    P = _policy(admit="memory")
    adm = P.admit(n_tok=n_tok, n_atom=n_atom, allocated_gib=alloc, trunk_peak_gib=10.0, total_gib=79.2, graph_cap=0, hoist_bound=True, say=False)
    assert adm.route == expect, adm
    assert adm.limit_gib == pytest.approx(0.85 * 79.2 - 1.0)


def test_graph_cap_bounds_graph_only():
    P = _policy(admit="memory")
    adm = P.admit(n_tok=2000, n_atom=16056, allocated_gib=8.5, trunk_peak_gib=26.0, total_gib=141.0, graph_cap=1536, hoist_bound=True, say=False)   # memory fits the graph, the cap does not
    assert adm.route == "hoist_eager" and "cap" in adm.why


def test_candidates_and_hoist_bound():
    P = _policy(admit="item")
    assert P.admit(n_tok=2000, n_atom=16056, allocated_gib=8.5, trunk_peak_gib=26.0, total_gib=79.2, hoist_bound=False, say=False).route == "stock"
    assert P.admit(n_tok=2000, n_atom=16056, allocated_gib=8.5, trunk_peak_gib=26.0, total_gib=79.2, candidates=("graph", "stock"), say=False).route == "stock"
    P0 = _policy(admit="item", hoist_eager="0")
    assert P0.admit(n_tok=1400, n_atom=11224, allocated_gib=5.4, trunk_peak_gib=13.7, total_gib=79.2, say=False).route == "stock"


def test_caller_graph_projection_overrides():
    P = _policy(admit="memory")
    adm = P.admit(n_tok=3000, n_atom=24060, allocated_gib=15.9, trunk_peak_gib=40.0, total_gib=79.2, graph_peak_gib=60.0, say=False)   # the caller's pool fit says the graph fits
    assert adm.route == "graph" and adm.terms.get("graph_from_caller") and adm.projected_gib == 60.0


def test_note_measured_flags_a_miss_in_words(capsys):
    P = _policy(admit="item")
    adm = P.admit(n_tok=1400, n_atom=11224, allocated_gib=5.4, trunk_peak_gib=13.7, total_gib=79.2, say=False)
    assert adm.route == "hoist_eager"
    ok = P.note_measured(adm, adm.limit_gib + 0.4, n_tok=1400, n_atom=11224)
    miss = P.note_measured(adm, adm.limit_gib + 0.6, n_tok=1400, n_atom=11224)
    err = capsys.readouterr().err
    assert not ok["admit_miss"] and miss["admit_miss"] and "ADMIT_MISS" in err and "admit_ok" in err and P.summary()["admit_misses"] == 1


def test_words_unset_is_kit_cap_only():
    P = _policy(admit="", release="", hoist_eager="0")
    assert not P.active()
    assert P.admit(n_tok=3000, n_atom=24060, allocated_gib=19.0, trunk_peak_gib=48.0, total_gib=79.2, graph_cap=1536, say=False).route == "stock"
    assert P.admit(n_tok=1400, n_atom=11224, allocated_gib=5.0, trunk_peak_gib=15.9, total_gib=79.2, graph_cap=1536, say=False).route == "graph"


def test_unknown_words_raise_by_name():
    with pytest.raises(ValueError) as e:
        _policy(admit="peak")
    assert "PTX_SAMPLER_ADMIT" in str(e.value)
    with pytest.raises(ValueError):
        _policy(release="unit")
    _policy()   # leave a valid module behind


def test_rows_carry_the_lever_and_the_words():
    sys.path.insert(0, os.path.join(KIT, "opt"))
    from protenix_opt import modes
    for mode in ("exact", "fast", "big"):
        assert "sampler_admit" in modes.MODES[mode] and "pred_release" in modes.MODES[mode], mode
    assert modes.PACKAGE_POST["fast"]["PTX_SAMPLER_ADMIT"] == "memory" == modes.PACKAGE_POST["exact"]["PTX_SAMPLER_ADMIT"]
    assert modes.PACKAGE_POST["fast"]["PTX_SAMPLER_RELEASE"] == "reach" == modes.PACKAGE_POST["exact"]["PTX_SAMPLER_RELEASE"] and modes.PACKAGE_POST["fast"]["PTX_SAMPLER_HOIST_EAGER"] == "1"
    assert modes.PACKAGE_POST["big"]["PTX_SAMPLER_RELEASE"] == "item"
    assert modes.PACKAGE_POST["big"]["PTX_SAMPLER_ADMIT"] == "item" and "PTX_SAMPLER_ADMIT" not in modes.BIG_POST
    for lv in ("sampler_graph", "sampler_graph_cache_policy", "sampler_admit"):
        assert lv in modes.MODES["big"], lv


def test_resolved_rows_export_the_words():
    """The words as the CLI exports them (modes.resolve on an H100 kernel key): big item, fast/exact memory; release=item, hoist_eager=1 and
    pred_release=1 on all three."""
    sys.path.insert(0, os.path.join(KIT, "opt"))
    from protenix_opt import modes
    fpf = os.path.join(KIT, "opt", "forward", "flashpairformer")
    for mode, word, rel in (("big", "item", "item"), ("fast", "memory", "reach"), ("exact", "memory", "reach")):
        r = modes.resolve(mode, {"PATH": os.environ.get("PATH", "")}, fpf, compute_cap="9.0", triton="3.7.1", memory_mib=81559, probe_gpu=False)
        ex = r.exports
        assert ex.get("PTX_SAMPLER_ADMIT") == word, (mode, ex.get("PTX_SAMPLER_ADMIT"))
        assert ex.get("PTX_SAMPLER_RELEASE") == rel and ex.get("PTX_SAMPLER_HOIST_EAGER") == "1" and ex.get("PTX_PRED_RELEASE") == "1", mode


def test_run_hoist_eager_brackets_the_callable():
    P = _policy(admit="item")
    class H:
        _eager_driving = False; _x_shape = 1; step = 5; mode = "?"
        def set_mode(self, m): self.mode = m
    class Loop: biascache = H()
    loop = Loop(); seen = {}
    def stock_fn(x, y=0):
        seen["during"] = (loop.biascache._eager_driving, loop.biascache._x_shape, loop.biascache.step); return x + y
    assert P.run_hoist_eager(loop, stock_fn, 1, y=2) == 3 and seen["during"] == (True, None, -1)
    assert loop.biascache._eager_driving is False and loop.biascache.mode == "off"
    assert P.run_stock(loop, stock_fn, 2, y=2) == 4
    passthru = lambda *a, **k: k.get("stock_fn")           # a keyword named stock_fn belongs to the callee (graphed.py's sample takes one)
    assert P.run_hoist_eager(loop, passthru, 1, stock_fn="kw") == "kw" and P.run_stock(loop, passthru, stock_fn="kw") == "kw"


def test_release_words():
    """big (item) releases after every item; exact/fast (reach) keep a graph entry inside the envelope (a second seed of the same input replays it,
    no capture) and release only what was admitted above it (a HOIST_EAGER route's slots; a GRAPH entry above the cap / floor)."""
    P = _policy(admit="memory", release="reach")
    g400 = P.Admission("graph", 66.3, 5.0, "", {}); g2000 = P.Admission("graph", 66.3, 48.0, "", {}); h2000 = P.Admission("hoist_eager", 66.3, 29.0, "", {})
    st = P.Admission("stock", 66.3, 20.0, "", {})
    assert P.should_release(g400, 400, graph_cap=1536) is False          # inside the kit cap: kept (same-shape replay stays free)
    assert P.should_release(g400, 400, graph_cap=0) is False             # no cap: inside REACH_FLOOR_TOKENS: kept
    assert P.should_release(h2000, 2000, graph_cap=1536) is True         # hoist slots never ride into the confidence head / next trunk
    assert P.should_release(g2000, 2000, graph_cap=0) is True            # a graph admitted above the envelope is released
    assert P.should_release(st, 3000, graph_cap=1536) is False
    P = _policy(admit="item", release="item")
    assert P.should_release(g400, 400, graph_cap=1536) is True and P.should_release(st, 3000) is True
