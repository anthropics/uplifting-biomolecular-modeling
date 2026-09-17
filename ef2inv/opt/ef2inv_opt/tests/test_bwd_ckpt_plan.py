"""The memory plan prices a kept block-pass with the kernel set that ACTUALLY serves the grad-mode pair block (k/ef2_bwd_ckpt.py kernels_of /
kernel_set): the three measured rows for the three full sets, the stock row (an upper bound) for every mixed set an ablation or a lever stepping
aside leaves — so `MODEL_OPT_LEVERS_OFF=trimul` (the stock cuEquivariance triangle multiplication beside the lean transition) checkpoints MORE than
the shipped composition and its estimate stays inside the budget, where pricing it at `agk3` overran an 80 GB card at 431 tokens. CPU: the module is
imported with a stand-in for ef2_autograd_kernels (its kernels need the GPU stack); plan() is arithmetic."""
import importlib.util
import os
import sys
import types

import pytest

from ._paths import KDIR

GIB = float(2 ** 30)


@pytest.fixture(scope="module")
def bc():
    pytest.importorskip("torch")
    saved, dont = sys.modules.get("ef2_autograd_kernels"), sys.dont_write_bytecode
    sys.modules["ef2_autograd_kernels"] = types.ModuleType("ef2_autograd_kernels")      # stand-in: kernels_of/kernel_set/plan never call into it
    sys.dont_write_bytecode = True                                                     # no bytecode under the kit tree (the design kit's k/ carries none)
    try:
        spec = importlib.util.spec_from_file_location("ef2_bwd_ckpt_under_test", os.path.join(KDIR, "ef2_bwd_ckpt.py"))
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        yield m
    finally:
        sys.dont_write_bytecode = dont
        if saved is None:
            sys.modules.pop("ef2_autograd_kernels", None)
        else:
            sys.modules["ef2_autograd_kernels"] = saved


def test_the_kernel_set_is_the_one_that_serves(bc):
    cfg = lambda trimul, transition: types.SimpleNamespace(trimul=trimul, transition=transition)
    fused, tiles = types.SimpleNamespace(variant="fused"), types.SimpleNamespace(variant="cueq_tiles")
    assert bc.kernel_set(cfg(None, "refround_lean"), fused) == "fused3"           # fast / big as shipped: ef2_trimul fused + K-D3 lean transition
    assert bc.kernel_set(cfg("bmm2", "refround_lean")) == "agk3"                 # the wired alternative: K-A2 + K-D3
    assert bc.kernel_set(cfg(None, None), None, cueq=True) == "stock" == bc.kernel_set(cfg(None, None), tiles, cueq=True)   # exact: the stock pair block on the cuEquivariance kernels (policy only)
    assert bc.kernel_set(cfg(None, "refround_lean"), None, cueq=True) == "stock"  # cuEquivariance beside the lean transition: the stock row bounds it
    assert bc.kernel_set(cfg(None, None), fused) == "stock"                      # MODEL_OPT_LEVERS_OFF=agk_transition: the fused triangle multiplication beside the stock transition (8.9 KiB measured <= 11.2)
    assert bc.kernel_set(cfg("bmm2", None)) == "stock"
    assert bc.kernel_set(cfg(None, "refround_lean")) == "ref3"                   # MODEL_OPT_LEVERS_OFF=trimul on fast / big: the fork's REFERENCE triangle multiplication (backend 'fused' is a no-grad path) beside the lean transition — was priced 'agk3'
    assert bc.kernel_set(cfg(None, "refround_lean"), tiles) == "ref3"            # a cueq_tiles handle is a table, not a serving kernel: the backend decides
    assert bc.kernel_set(cfg(None, None)) == "refstock"                          # trimul AND agk_transition off on the loader's backend: the stock compiled block on the reference triangle multiplication
    k = bc.KEPT_BYTES_PER_POS
    assert k["ref3"] > k["refstock"] > k["stock"] > k["agk3"] > k["fused3"] and k["ref3"] >= 3 * k["fused3"]   # eager reference > the compiled block on it > the compiled block on cuEquivariance > the kit sets


def test_trimul_ablated_at_431_tokens_checkpoints_more_and_stays_inside_the_budget(bc):
    """A 431-token plan (fast / big, one model, two passes, budget 71.26 GiB, base 20.4): priced at the reference row the plan keeps far
    fewer blocks than at 'agk3' (and than the shipped 'fused3' composition, which keeps all 24) and its estimate is inside the budget."""
    kw = dict(budget_bytes=71.26 * GIB, base_bytes=20.4 * GIB, total_bytes=79.6 * GIB)
    shipped, was, now = (bc.plan(431, 2, k, **kw) for k in ("fused3", "agk3", "ref3"))
    assert shipped["policy"] == "none" and shipped["kept_per_pass"] == 24                         # the shipped composition at 431: nothing checkpointed
    assert now["kept_per_pass"] < was["kept_per_pass"] <= 24 and now["policy"].startswith("ckpt:") and now["kept_per_pass"] <= 8
    assert now["est_peak_bytes"] <= kw["budget_bytes"] and now["kernels"] == "ref3"
    # what the 'agk3' price under-counted: the same kept blocks cost (ref3 - agk3) KiB x 431^2 more per kept block-pass than planned — tens of GiB
    assert (bc.KEPT_BYTES_PER_POS["ref3"] - bc.KEPT_BYTES_PER_POS["agk3"]) * 431 ** 2 * was["kept_per_pass"] * 2 > 30 * GIB


def test_the_floor_rule_is_planned_all_checkpoint_with_its_reason(bc):
    """plan(floor=True) (big's composition word `ckpt: floor`, fastkit.ckpt_floor): policy 'block', kept 0, rule 'floor', reason 'memory_floor'
    whatever would fit — the budget and the kept-0 estimate still computed; the budget rule's record carries rule 'budget' and no reason word;
    a floor plan is a PLANNED record (no `forced` key), which is what evidence F5 accepts on big and refuses on fast / exact."""
    kw = dict(budget_bytes=71.26 * GIB, base_bytes=20.4 * GIB, total_bytes=79.6 * GIB)
    for toks in (195, 431, 800, 1200):
        fill, floor = bc.plan(toks, 2, "fused3", **kw), bc.plan(toks, 2, "fused3", floor=True, **kw)
        assert (floor["policy"], floor["kept_per_pass"], floor["rule"], floor["floor"], floor["reason"]) == ("block", 0, "floor", True, "memory_floor") and "forced" not in floor, toks
        assert floor["budget_bytes"] == fill["budget_bytes"] == kw["budget_bytes"] and floor["est_peak_bytes"] == bc.estimate_peak_bytes(toks, 0, 2, "fused3", 24, kw["base_bytes"], 1) <= fill["est_peak_bytes"], toks
        assert (fill["rule"], fill["floor"]) == ("budget", False) and "reason" not in fill and bc.FLOOR_REASON == "memory_floor", toks
    assert bc.plan(195, 2, "fused3", **kw)["policy"] == "none" and bc.plan(195, 2, "fused3", floor=True, **kw)["policy"] == "block"   # at 195 tokens everything fits: the floor still checkpoints every block
