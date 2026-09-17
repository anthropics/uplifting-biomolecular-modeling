"""The trunk-side fused-kernel adapter (opt/forward/v05_addon/ptxfpf/trunk2_ptx1.py, lever words pfattn / opm_fused / pwa_fused over the carried
kernel packages opt_core.kernels.apb.fpf_apb + lib/protenix_fpf_msa): importable without torch; its routing decisions are pure functions of shapes, dtype
names, the autocast state and the card's capability — a call or a card the cells do not serve steps aside BY NAME (`stock:<word>` / a named
condition), never silently, never as a refusal; and the words are wired through the kit's one mechanism (levers_ptx1 grammar, modes registry,
report strategy ids, the stock route's module prefixes, the row-sharded line's replaced sites). No GPU needed."""
import ast
import os
import sys

import pytest

from .conftest import KIT

PTXFPF = os.path.join(KIT, "ptxfpf")
MSA_PKG = os.path.join(KIT, "lib", "protenix_fpf_msa")
import opt_core.kernels.apb as _APB
APB_PKG = os.path.join(os.path.dirname(_APB.__file__), "fpf_apb")                    # the shared core's carried fpf_apb (kit >= 0.2.21; before: lib/protenix_fpf_apb)


@pytest.fixture(scope="module")
def T():
    sys.path.insert(0, PTXFPF)
    try:
        sys.modules.pop("trunk2_ptx1", None)
        import trunk2_ptx1 as mod
    finally:
        sys.path.pop(0)
    return mod


def test_adapter_imports_without_torch():
    with open(os.path.join(PTXFPF, "trunk2_ptx1.py"), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = {a.name.split(".")[0] for n in top for a in n.names} | {n.module.split(".")[0] for n in top if isinstance(n, ast.ImportFrom) and n.module}
    assert not names & {"torch", "triton", "protenix", "protenix_fpf_apb", "protenix_fpf_msa", "opt_core"}, names


def test_words_and_packages(T):
    assert T.LEVERS == ("pfattn", "opm_fused", "pwa_fused") and T.PACKAGES == ("opt_core.kernels.apb", "protenix_fpf_msa")
    assert set(T.KIND.values()) == {"pf", "opm", "pwa"} and set(T.CELLS) == {"opm", "pwa"}        # pfattn owns no kit cell: the provider's table selects its rows
    for kind in T.CELLS:
        assert T.REFERENCE_CC in T.CELLS[kind]
    for pkg in (MSA_PKG,):                                                            # the carried package ships with its README and notice (the apb one is the core's)
        assert os.path.isfile(os.path.join(pkg, "README.md")) and os.path.isfile(os.path.join(pkg, "NOTICE.md")), pkg
        assert "PROVENANCE" in open(os.path.join(pkg, "README.md"), encoding="utf-8").read()
    for rel in ("__init__.py", "CELLS.json", "VECTORS.json", "loadcheck.py", "install.py"):
        assert os.path.isfile(os.path.join(MSA_PKG, rel)), rel
    assert os.path.isfile(os.path.join(APB_PKG, "pf_triton.py"))                       # one of the provider's producer rows (fpf_pf_bias) is the carried apb package's


def test_card_cell_measured_named_and_aside(T):
    assert T.card_cell("pf", None) == (None, None, "no CUDA device")
    cell, key, named = T.card_cell("opm", (7, 5))
    assert cell is None and key is None and named.startswith("arch=sm_75")             # below Ampere: the lever installs on nothing, named
    assert T.card_cell("opm", (9, 0)) == (T.CELLS["opm"]["9.0"], "9.0", None)
    for kind in ("opm", "pwa"):
        for cc in ((8, 0), (10, 0), (12, 0), ("8", "6")):                              # unmeasured cards >= 8.0 ENGAGE the 9.0 cell and are named
            cell, key, named = T.card_cell(kind, cc)
            assert cell == T.CELLS[kind]["9.0"] and key == "9.0" and "sm_%d%d" % (int(cc[0]), int(cc[1])) in named and "cell=9.0" in named


def test_geometry_words(T):
    assert T.pf_geometry_word(16, 24, True, True, 128) is None                          # the 1.1.0 pairformer: 16 heads x 24, gated, q bias, c_z 128
    assert T.pf_geometry_word(16, 24, True, True, 96) == "geometry=16x24:c_z96"
    assert T.pf_geometry_word(8, 24, True, True, 128).startswith("geometry=8x24")
    assert T.pf_geometry_word(16, 24, False, True, 128).endswith(":nogate")
    assert T.opm_geometry_word(64, 32, 128, 1024) is None and T.opm_geometry_word(64, 32, 128, 512) is not None
    assert T.pwa_geometry_word(64, 8, 32, 128) is None and T.pwa_geometry_word(64, 8, 12, 128) is not None


def test_pf_route_served_and_aside(T):
    N = 384
    assert T.pf_route((N, 384), True, (N, N, 128), False) is None                       # the trunk call: q [N, c_s], one pair tensor
    assert T.pf_route((1, N, 384), True, (1, N, N, 128), False) is None                 # size-1 leading dims
    assert T.pf_route((5, N, 384), True, (1, N, N, 128), False) is None                 # q rows sharing one pair tensor
    assert T.pf_route((5, N, 384), True, (5, N, N, 128), False) == "stock:z_batch"      # a batched pair tensor: the stock statement's
    assert T.pf_route((N, 384), True, (N, N, 128), True) == "stock:efficient_fusion"
    assert T.pf_route((N, 384), False, (N, N, 128), False) == "stock:cross"
    assert T.pf_route((N, 384), True, (N, N, 128), False, dtype="float32") is None                 # fp32-stored activations under bf16 autocast: served (stock's Linears round to bf16 there too)
    assert T.pf_route((N, 384), True, (N, N, 128), False, dtype="float16") == "stock:dtype=float16"
    assert T.pf_route((N, 384), True, (N, N, 128), False, autocast=False) == "stock:no_autocast"
    assert T.pf_route((N, 384), True, (N, N, 128), False, is_cuda=False) == "stock:device"
    assert T.pf_route((N, 384), True, (N, N, 128), False, training=True) == "stock:training"
    assert T.pf_route((N, 384), True, (N // 2, N, 128), False) == "stock:z_shape"       # a row shard is not this lever's (the row-sharded line owns it)
    assert T.pf_route((N, 384), True, (N + 8, N + 8, 128), False) == "stock:z_len"


def test_msa_routes_served_and_aside(T):
    S, N = 2048, 384
    assert T.opm_route((S, N, 64), False) is None
    assert T.opm_route((S, N, 64), True) == "stock:mask"
    assert T.opm_route((1, S, N, 64), False) == "stock:m_dim=4"
    assert T.opm_route((S, N, 64), False, fp16=True) == "stock:fp16"
    assert T.opm_route((S, N, 64), False, dtype="float16") == "stock:dtype=float16" and T.opm_route((S, N, 64), False, autocast=False) == "stock:no_autocast"
    assert T.pwa_route((S, N, 64), (N, N, 128)) is None
    assert T.pwa_route((S, N, 64), (N, N + 1, 128)) == "stock:z_shape"
    assert T.pwa_route((S, N, 64), (N // 2, N // 2, 128)) == "stock:z_len"
    assert T.pwa_route((S, N, 64), (N, N, 128), z_dtype="float64") == "stock:z_dtype=float64" and T.pwa_route((S, N, 64), (N, N, 128), dtype="float32") is None


def test_install_without_cuda_is_named_not_silent(T):
    pytest.importorskip("torch")
    import torch
    if torch.cuda.is_available():
        pytest.skip("CPU-only check")
    d = T.apply(object(), pfattn=True, opm_fused=True, pwa_fused=True)                  # no device: nothing installed, each word named
    for kind in ("pf", "opm", "pwa"):
        assert d[kind]["engaged"] is False and d[kind]["named"] == "no CUDA device"
    for word in T.LEVERS:
        f = T.lever_facts(word)
        assert f["state"] == "skipped" and f["reason"] == "no CUDA device" and f["served"] == 0
    d = T.apply(object(), pfattn=False, opm_fused=False, pwa_fused=False)
    assert T.lever_facts("pfattn")["reason"] in ("no CUDA device", "not_requested")


def test_lever_facts_grammar(T):
    T.COUNTS["opm"].clear(); T.STATE["opm"].update(installed_on=4, found=4, cell_key="9.0", named=None, calls=0)
    try:
        assert T.lever_facts("opm_fused")["aside"] == "no_call"                          # installed, never entered (no MSA pass): by design, named
        T.COUNTS["opm"]["t2:opm@9.0"] += 8
        f = T.lever_facts("opm_fused")
        assert f["state"] == "on" and f["served"] == 8 and "aside" not in f and f["cell"] == "opm@9.0"
        T.COUNTS["opm"]["stock:mask"] += 2; T.COUNTS["opm"]["error:RuntimeError"] += 1
        f = T.lever_facts("opm_fused")
        assert f["gated"] == {"stock:mask": 2} and f["fallback"] == {"error:RuntimeError": 1}
    finally:
        T.COUNTS["opm"].clear(); T.STATE["opm"].update(installed_on=0, found=0, cell_key=None, calls=0)


def test_words_wired_through_the_kits_one_mechanism(T):
    from protenix_v1_opt import kit as K, modes as M, report as R, stock_pred as SP, big as B
    trimuls, levers = K.lever_grammar(KIT)
    for w in T.LEVERS:
        assert w in levers, w                                                            # levers_ptx1.LEVER_NAMES (the grammar)
        assert M.LEVERS[w]["class"] == "tolerance" and "trunk2_ptx1.py" in M.LEVERS[w]["file"]   # the registry row
        assert R.STRATEGY_IDS[w] == "LOCAL.protenix_v1.%s" % w                          # the LEVER line's strategy id: this kit's local form
        assert R.lever_impl(w) == ("ptxfpf/trunk2_ptx1.py", "kit")
        assert w in B.ROWPAIR_REPLACES                                                   # the row-sharded line names the site it replaces (never `served 0` partial under --n_gpu P>1)
        for m in ("fast", "big"):
            assert w in K.parse_arm(M.KIT_MODES[m].arm, KIT)[1], (m, w)                  # fast and big compose the words; exact does not (tolerance class)
        assert w not in K.parse_arm(M.KIT_MODES["exact"].arm, KIT)[1]
    for name in ("trunk2_ptx1", "protenix_fpf_msa"):
        assert name.startswith(SP.KIT_MODULES), name                                    # the stock route's clean child refuses the adapter and its packages by name


def test_kit_evidence_reads_the_account(T):
    from protenix_v1_opt import report as R
    levers = {"cfg": {"trimul": "fast", "pfattn": True, "opm_fused": True, "pwa_fused": True},
              "counts": {"trimul": {"fast": 40}, "pf": {"t2:pf@9.0": 96, "stock:z_batch": 4}, "opm": {"t2:opm@9.0": 8}, "pwa": {}},
              "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96},
                         "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8},
                         "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 0}}}
    ev = R.kit_evidence(levers, "fast", ("pfattn", "opm_fused", "pwa_fused"))
    assert ev["pfattn"]["served"] == 96 and ev["pfattn"]["gated"] == {"stock:z_batch": 4} and ev["pfattn"]["fallback"] == {} and "aside" not in ev["pfattn"]
    assert ev["opm_fused"]["served"] == 8 and ev["opm_fused"]["cell"] == "opm@9.0"
    assert ev["pwa_fused"]["served"] == 0 and ev["pwa_fused"]["aside"] == {"state": "aside", "word": "no_call"}   # installed, no MSA pass this run: by design
    assert R.partial_of(ev) == ([], None)
    levers["trunk2"]["opm"] = {"engaged": False, "named": "no CUDA device", "installed_on": 0, "calls": 0}; levers["counts"]["opm"] = {}
    ev = R.kit_evidence(levers, "fast", ("pfattn", "opm_fused", "pwa_fused"))
    assert ev["opm_fused"]["fallback"] == {"engaged": "not installed: no CUDA device"} and R.partial_of(ev)[0] == ["opm_fused"]   # a word of the mode that did not install: partial, named
    levers["counts"]["pf"]["error:RuntimeError"] = 1
    ev = R.kit_evidence(levers, "fast", ("pfattn",))
    assert ev["pfattn"]["fallback"] == {"error:RuntimeError": 1} and R.partial_of(ev)[0] == ["pfattn"]


def test_tier_word_is_the_kit_mode(T):
    assert T.tier_word({"PROTENIX_V1_OPT": "fast"}) == "fast" and T.tier_word({"PROTENIX_V1_OPT": "big"}) == "big" and T.tier_word({"PROTENIX_V1_OPT": "exact"}) == "exact"
    assert T.tier_word({}) == "fast" and T.tier_word({"PROTENIX_V1_OPT": "off"}) == "fast"          # no mode variable / a non-tier mode word reads fast


def test_pf_has_no_kit_cell_and_engages_by_card(T):
    cell, key, named = T.card_cell("pf", (9, 0))
    assert cell == {} and key == "9.0" and named is None
    cell, key, named = T.card_cell("pf", (8, 0))
    assert cell == {} and key == "8.0" and named is None                                             # the provider selects per card: nothing to name on cc 8.0
    assert T.card_cell("pf", (7, 5))[0] is None and T.card_cell("pf", None)[0] is None

