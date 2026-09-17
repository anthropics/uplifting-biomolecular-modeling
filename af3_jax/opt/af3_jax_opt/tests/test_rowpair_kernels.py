"""--n_gpu P > 1: the family's exit line NAMES the per-shard kernels of the two pair ops (inprocess/rowpair.kernel_evidence over the trace-time
census count_call books) — `kernel=triatt:<word>,trimul:<word> kernel_reason=<gate word>` where `kernel=unrecorded` used to print — plus
`<op>_served= <op>_fallback= <op>_stock=`; a per-shard fallback is a partial activation by name (modes.rowpair_evidence). P=1 never loads the
module. CPU only (fake modules, fake mesh)."""
import types

import pytest

from af3_jax_opt import modes
from af3_jax_opt.inprocess import rowpair as RP

P_ = "[af3-jax-opt]"


class Act:
    def __init__(self, *shape): self.shape = shape


def gsa(impl="triton"):
    return types.SimpleNamespace(global_config=types.SimpleNamespace(flash_attention_implementation=impl), config=types.SimpleNamespace(num_head=4))


def tm(glu=True):
    return types.SimpleNamespace(config=types.SimpleNamespace(use_glu_kernel=glu, equation="ikc,jkc->ijc"), global_config=types.SimpleNamespace())


def fresh():
    return {op: {"kernel": None, "reason": None, "served": 0, "fallback": 0, "stock": 0} for op in RP.KERNEL_OPS}


def test_kernel_words_are_the_models_own():
    assert RP.kernel_word("triatt", gsa("triton")) == "af3_attention:triton" and RP.kernel_word("triatt", gsa("xla")) == "af3_attention:xla"
    assert RP.kernel_word("trimul", tm(True)) == "tokamax_glu+xla_einsum" and RP.kernel_word("trimul", tm(False)) == "xla_einsum"
    with pytest.raises(ValueError):
        RP.kernel_word("transition", tm())


def test_census_and_exit_fields():
    c = fresh()
    assert RP.kernel_evidence(c) == {"kernel": "unrecorded", "kernel_reason": "no_trace", "triatt_served": 0, "triatt_fallback": 0, "triatt_stock": 0,
                                     "trimul_served": 0, "trimul_fallback": 0, "trimul_stock": 0}          # nothing traced (executables loaded): the word says so
    for _ in range(3):
        RP.count_call("triatt", gsa(), Act(768, 1536, 128), True, c); RP.count_call("trimul", tm(), Act(768, 1536, 128), True, c)
    RP.count_call("trimul", tm(), Act(1536, 1536, 128), False, c)                                        # a stock-body call outside the sharded region
    assert RP.kernel_evidence(c) == {"kernel": "triatt:af3_attention:triton,trimul:tokamax_glu+xla_einsum", "kernel_reason": "unconstrained",
                                     "triatt_served": 3, "triatt_fallback": 0, "triatt_stock": 0, "trimul_served": 3, "trimul_fallback": 0, "trimul_stock": 1}
    c["trimul"]["fallback"] = 2                                                                           # a block the named kernel did not serve: the reason names it
    assert RP.kernel_evidence(c)["kernel_reason"] == "per_shard_fallback:trimul=2"


def test_install_kernel_census_shims_the_recipes_bodies_once():
    calls = []
    class GridSelfAttention:
        global_config = types.SimpleNamespace(flash_attention_implementation="triton")
        def __call__(self, act, pair_mask_rows, pair_mask_cols=None, num_residues_global=None):
            calls.append(("gsa", act.shape)); return "gsa_out"
    class TriangleMultiplication:
        config = types.SimpleNamespace(use_glu_kernel=True)
        def __call__(self, act, mask):
            calls.append(("tm", act.shape)); return "tm_out"
    modules = types.SimpleNamespace(GridSelfAttention=GridSelfAttention, TriangleMultiplication=TriangleMultiplication)
    saved = {op: dict(v) for op, v in RP.KERNELS.items()}
    try:
        for v in RP.KERNELS.values(): v.update(kernel=None, reason=None, served=0, fallback=0, stock=0)
        region = {"in": True}
        assert RP.install_kernel_census(modules, lambda: region["in"]) == ["GridSelfAttention", "TriangleMultiplication"]
        assert RP.install_kernel_census(modules, lambda: region["in"]) == []                              # idempotent: never a shim over a shim
        assert GridSelfAttention()(Act(768, 1536, 128), None, None, 1536) == "gsa_out" and TriangleMultiplication()(Act(768, 1536, 128), None) == "tm_out"
        region["in"] = False; TriangleMultiplication()(Act(1536, 1536, 128), None)
        assert calls == [("gsa", (768, 1536, 128)), ("tm", (768, 1536, 128)), ("tm", (1536, 1536, 128))]  # the bodies ran unchanged
        assert (RP.KERNELS["triatt"]["served"], RP.KERNELS["trimul"]["served"], RP.KERNELS["trimul"]["stock"], RP.KERNELS["trimul"]["kernel"]) == (1, 1, 1, "tokamax_glu+xla_einsum")
    finally:
        for op, v in saved.items(): RP.KERNELS[op].update(v)


def test_a_per_shard_fallback_is_a_partial_activation_by_name():
    inst = f"{P_} ROWPAIR installed n_gpu=2 sites=9 heads=sharded conf=sharded diffusion=sharded n_sites=9 library=x schedule=gather mem_fraction_ceiling=none xla_pool_fraction=0.75"
    base = (f"{P_} LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@0.5.17.5 origin=core n_gpu=2 sharding=rowpair xla_peak_gb_max=31.5 schedule=gather "
            "kernel=triatt:af3_attention:triton,trimul:tokamax_glu+xla_einsum kernel_reason=unconstrained sites=trunk,model,heads "
            "triatt_served=96 triatt_fallback=0 triatt_stock=0 trimul_served=96 trimul_fallback=0 trimul_stock=0")
    ok = modes.rowpair_evidence([inst, base], 2)
    assert ok["ok"] and ok["kernel"] == "triatt:af3_attention:triton,trimul:tokamax_glu+xla_einsum" and ok["per_shard_fallback"] == {}
    bad = modes.rowpair_evidence([inst, base.replace("trimul_fallback=0", "trimul_fallback=2")], 2)
    assert not bad["ok"] and bad["reason"] == "ROWPAIR per-shard fallback: trimul_fallback=2 (kernel=triatt:af3_attention:triton,trimul:tokamax_glu+xla_einsum)"
    old = f"{P_} LEVER name=rowpair state=on impl=opt_core.mem.rowpair_jax@0.4.1 origin=core n_gpu=2 sharding=rowpair xla_peak_gb_max=31.5 schedule=gather kernel=unrecorded kernel_reason=kernel_fields_unavailable sites=trunk,model,heads"
    assert modes.rowpair_evidence([inst, old], 2)["ok"]                                                    # a line without the census (an older kit) reads as before


def test_render_the_family_line_at_p2(monkeypatch):
    """The exact tokens, rendered through the core's evidence.line with a stand-in mesh: old fields vs new fields (the report quotes these)."""
    pytest.importorskip("jax")
    from opt_core.mem.rowpair_jax import evidence as ev

    class FakeMesh:
        n_gpu = 2
        def describe(self):
            return {"axis": "rows", "visible": 2, "platform": "gpu", "devices": "cuda:0,cuda:1", "device_kind": "NVIDIA_H100_80GB_HBM3", "sm": "9.0",
                    "xla_pool_limit": 63753420800, "device_total_bytes": 85045870592, "xla_pool_fraction": 0.7497}
    for k in ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_ALLOCATOR"):
        monkeypatch.delenv(k, raising=False)
    common = dict(rmesh=FakeMesh(), peaks=None, schedule="gather", sites="TriangleMultiplication.__call__,GridSelfAttention.__call__", heads="sharded", conf="sharded",
                  diffusion="sharded", hazards="none", mem_fraction_ceiling="none", xla_pool_fraction=0.7497)
    old = ev.line(RP.TAG, "on", 2, kernel="unrecorded", kernel_reason="kernel_fields_unavailable", **common)
    c = fresh()
    for _ in range(96):
        RP.count_call("triatt", gsa(), Act(768, 1536, 128), True, c); RP.count_call("trimul", tm(), Act(768, 1536, 128), True, c)
    ke = RP.kernel_evidence(c)
    census = {k: v for k, v in ke.items() if k not in ("kernel", "kernel_reason")}
    new = ev.line(RP.TAG, "on", 2, kernel=ke["kernel"], kernel_reason=ke["kernel_reason"], **common, **census)   # exit_line's order: the census fields last
    head = old.split(" kernel=unrecorded")[0]
    assert new.startswith(head) and old.endswith(" xla_pool_fraction=0.7497")
    assert old[len(head):] == " kernel=unrecorded kernel_reason=kernel_fields_unavailable sites=TriangleMultiplication.__call__,GridSelfAttention.__call__ heads=sharded conf=sharded diffusion=sharded hazards=none mem_fraction_ceiling=none xla_pool_fraction=0.7497"
    assert new[len(head):] == (" kernel=triatt:af3_attention:triton,trimul:tokamax_glu+xla_einsum kernel_reason=unconstrained sites=TriangleMultiplication.__call__,GridSelfAttention.__call__ "
                               "heads=sharded conf=sharded diffusion=sharded hazards=none mem_fraction_ceiling=none xla_pool_fraction=0.7497 "
                               "triatt_served=96 triatt_fallback=0 triatt_stock=0 trimul_served=96 trimul_fallback=0 trimul_stock=0")
    print("OLD", old); print("NEW", new)
