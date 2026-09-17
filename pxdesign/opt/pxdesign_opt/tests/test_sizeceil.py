"""The package's size levers (sizeceil.py): the padmask index algebra equals the stock dense construction element for element (the three
stock functions loaded verbatim from the pinned source under stock/src, real torch on CPU); featdiet drops exactly DIET_KEYS; apply() names
an unknown lever and a lever whose patch points are missing as fallbacks, never silently."""
import os
import re
import sys
import types

import pytest

from pxdesign_opt import registry, sizeceil

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PROTENIX_SRC = os.path.join(TREE, "stock", "src", "Protenix")


def _real_torch():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "arange") or not hasattr(torch, "Generator"):
        pytest.skip("torch in this interpreter is a stub")
    return torch


def _stock_primitives(torch):
    """rearrange_qk_to_dense_trunk / optimized_concat_split / rearrange_to_dense_trunk and their utils, from the pinned source text."""
    prim = open(os.path.join(PROTENIX_SRC, "protenix/model/modules/primitives.py"), encoding="utf-8").read()
    util = open(os.path.join(PROTENIX_SRC, "protenix/model/utils.py"), encoding="utf-8").read()
    import math, typing
    ns = {"torch": torch, "nn": torch.nn, "F": torch.nn.functional, "math": math, "Optional": typing.Optional, "Union": typing.Union}

    def grab(src, name):
        m = re.search(rf"^def {name}\(.*?(?=^def |^class |\Z)", src, re.S | re.M)
        assert m, name
        return m.group(0)
    for nm in ("pad_at_dim", "reshape_at_dim", "move_final_dim_to_dim"):
        exec(grab(util, nm), ns)
    for nm in ("rearrange_qk_to_dense_trunk", "optimized_concat_split", "rearrange_to_dense_trunk"):
        exec(grab(prim, nm), ns)
    return types.SimpleNamespace(**{k: ns[k] for k in ("rearrange_qk_to_dense_trunk", "rearrange_to_dense_trunk")})


@pytest.mark.parametrize("n", [1, 31, 32, 33, 100, 128, 129, 257, 500])
def test_padmask_equals_the_stock_dense_construction(n):
    torch = _real_torch()
    PR = _stock_primitives(torch)
    new_qk = sizeceil._wrap_qk(PR.rearrange_qk_to_dense_trunk)
    new_dense = sizeceil._wrap_dense(PR.rearrange_to_dense_trunk, PR.rearrange_qk_to_dense_trunk)
    nq, nk = sizeceil.N_QUERIES, sizeceil.N_KEYS
    g = torch.Generator(); g.manual_seed(n)
    ref_pos, uid = torch.randn(n, 3, generator=g), torch.randint(0, max(1, n // 3), (n,), generator=g)
    _, _, i_s = PR.rearrange_qk_to_dense_trunk([ref_pos, uid], [ref_pos, uid], [-2, -1], [-2, -1], n_queries=nq, n_keys=nk, compute_mask=True)
    _, _, i_n = new_qk([ref_pos, uid], [ref_pos, uid], [-2, -1], [-2, -1], n_queries=nq, n_keys=nk, compute_mask=True)
    assert i_s["mask_trunked"].dtype == i_n["mask_trunked"].dtype and torch.equal(i_s["mask_trunked"], i_n["mask_trunked"])   # AtomAttentionEncoder's list form
    idx = torch.arange(n)
    _, _, j_s = PR.rearrange_qk_to_dense_trunk(idx, idx, -1, -1, n_queries=nq, n_keys=nk, compute_mask=True)
    _, _, j_n = new_qk(idx, idx, -1, -1, n_queries=nq, n_keys=nk, compute_mask=True)
    assert torch.equal(j_s["mask_trunked"], j_n["mask_trunked"])                # broadcast_token_to_local_atom_pair's 1-D form
    for q in (torch.randn(1, n, 8, generator=g), torch.randn(2, 3, n, 8, generator=g).to(torch.bfloat16)):
        a_s = PR.rearrange_to_dense_trunk(q, q, q, n_queries=nq, n_keys=nk, attn_bias=None, inf=1e10)
        a_n = new_dense(q, q, q, n_queries=nq, n_keys=nk, attn_bias=None, inf=1e10)
        assert a_s[3].dtype == a_n[3].dtype and a_s[3].shape == a_n[3].shape and torch.equal(a_s[3], a_n[3])   # the padding bias, fp32 and bf16 lead dims
        assert a_s[4] == a_n[4] and all(torch.equal(x, y) for x, y in zip(a_s[:3], a_n[:3]))
    b = sizeceil.analytic_bias(n, nq, nk, 1e10, torch.float32, torch.device("cpu"), 1)
    assert torch.equal(b, PR.rearrange_to_dense_trunk(torch.zeros(1, n, 1), torch.zeros(1, n, 1), torch.zeros(1, n, 1), nq, nk, None, 1e10)[3])   # the h5 seed


def test_featdiet_drops_exactly_the_diet_keys():
    torch = _real_torch()
    f = {"bond_mask": torch.zeros(7, 7, dtype=torch.int64), "ref_pos": torch.zeros(7, 3)}
    data = {"input_feature_dict": f, "sample_name": "x"}
    freed = sizeceil.featdiet(data)
    assert freed == 7 * 7 * 8 and set(f) == {"ref_pos"} and sizeceil.DIET_KEYS == ("bond_mask",)
    assert sizeceil.featdiet(data) == 0 and sizeceil.featdiet({"no": "features"}) == 0


def test_apply_names_every_lever_not_applied(monkeypatch):
    """An unknown name and a lever whose module is absent are fallbacks with a reason; nothing is applied silently, nothing raises."""
    for m in ("protenix.model.modules.primitives", "protenix.model.modules.transformer", "pxdesign.runner.inference"):
        monkeypatch.setitem(sys.modules, m, None)                                 # import raises ImportError
    rec = sizeceil.apply(["featdiet", "padmask", "nope"])
    assert rec["planned"] == ["featdiet", "padmask", "nope"] and rec["applied"] == [] and rec["fallback"] == ["featdiet", "padmask", "nope"]
    assert all(rec["reasons"].get(k) for k in rec["fallback"]) and "unknown package lever" in rec["reasons"]["nope"]


def test_registry_rows_match_the_module():
    assert tuple(registry.PACKAGE_LEVERS) == tuple(sizeceil.APPLIERS) == tuple(sizeceil.REQUIRED_FIELDS) == ("featdiet", "padmask", "rowpipe", "tf32", "sdedup")
    homes = {"rowpipe": "rowpipe.py", "tf32": "precision.py", "sdedup": "sdedup.py"}
    assert all(homes.get(n, "sizeceil.py") in lv.code for n, lv in registry.PACKAGE_LEVERS.items())
    assert set(sizeceil.RUNTIME_GATES) == set(sizeceil.RUNTIME_CENSUS) == {"sdedup"}
    assert sizeceil.MASK_CACHE_CAP == 64 and (sizeceil.N_QUERIES, sizeceil.N_KEYS) == (32, 128)
