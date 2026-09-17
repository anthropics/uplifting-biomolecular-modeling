"""The generation route's memory rule (opt/serving/pipeline_v0_4/progen2_decode.py): the decode component holds static K/V slots per BATCH
SIZE, allocated for the work in front of the model — when the caller names its batch sizes (an --input job, a call's own: Handle.hold) or when a
unit first names one — each after ONE fit rule (static_kv_fits over the component's kv_bytes_per_sample × B and the device's free memory); a batch
size that cannot fit is answered by the named out-of-memory row, nothing allocated, nothing rerouted. CPU only: the fit arithmetic and the buckets
are pure functions of the kit file; the component's allocate / release / byte count run on a tiny ProGen model on the CPU (plain torch)."""
import importlib.util
import math
import os
import sys

import pytest

from .conftest import OPT, TREE

SERVING = os.path.join(OPT, "serving", "pipeline_v0_4")
COMPONENT = os.path.join(SERVING, "components", "plm_transfer_t1_v0")
STOCK_SRC = os.path.join(TREE, "stock", "src", "progen2")
GiB = 1 << 30
XLARGE_PER_SAMPLE_512 = 32 * 512 * 4096 * (4 + 2)     # n_layer × slots × n_embd × (float32 K + float16 V) = 0.375 GiB: the xlarge model sized for max_length 512


@pytest.fixture
def decode_mod():
    """progen2_decode.py imported by path (torch absent or present: it imports only the standard library at import)."""
    if SERVING not in sys.path:
        sys.path.insert(0, SERVING)
    spec = importlib.util.spec_from_file_location("progen2_decode_memory_under_test", os.path.join(SERVING, "progen2_decode.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_the_fit_rule_is_need_against_free_plus_cache_less_a_margin_of_the_card(decode_mod):
    total = int(79.65 * GiB)                                                        # an H100's usable size
    assert decode_mod.STATIC_KV_FIT_MARGIN == 0.04
    assert decode_mod.static_kv_fits(64 * XLARGE_PER_SAMPLE_512, 66 * GiB, 0, total)            # 24 GiB beside the xlarge weights: fits
    assert decode_mod.static_kv_fits(128 * XLARGE_PER_SAMPLE_512, 66 * GiB, GiB, total)        # 48 GiB <= 67 - 3.19
    assert not decode_mod.static_kv_fits(256 * XLARGE_PER_SAMPLE_512, 66 * GiB, GiB, total)    # 96 GiB: the named row, nothing allocated
    edge = 10 * GiB
    assert decode_mod.static_kv_fits(edge, edge + 0.04 * total, 0, total) and not decode_mod.static_kv_fits(edge + 1, edge + 0.08 * total, 0, total, margin=0.08)
    assert decode_mod.static_kv_fits(edge, edge, 0, total, margin=0.0)


def test_nothing_is_sized_to_the_card_and_no_server_remains(decode_mod):
    src = open(os.path.join(SERVING, "progen2_decode.py"), encoding="utf-8").read()
    for gone in ("--kit-bmax-auto", "--kit-bmax-cap", "0.55 * total", "--kit-batches", "socket", "argparse", "CUDAGraph", ".replay("):   # no card-budgeted cap, no boot list, no transport, no command line of its own, every step eager
        assert gone not in src, gone
    assert sorted(f for f in os.listdir(SERVING) if not f.startswith((".", "__"))) == ["_oom.py", "components", "components.json", "oneread_loader.py", "progen2_decode.py", "sampler_exact.py"]   # the generation kit: no server, no client


def test_buckets_and_levels(decode_mod):
    assert decode_mod.KIT_BUCKETS == (256, 512, 1024, 2048)
    assert [decode_mod.bucket(L, 1024) for L in (1, 64, 256, 257, 512, 513, 1024)] == [256, 256, 256, 512, 512, 1024, 1024]
    assert decode_mod.bucket(2048, 1024) == 1024 and decode_mod.bucket(1500, 2048) == 2048 and decode_mod.bucket(4096, 2048) == 2048   # capped at n_positions
    assert decode_mod.batches_word([1, 64]) == "1,64" and decode_mod.batches_word([]) == "-"
    comp = decode_mod.components()
    assert comp["decode_kit"] == {"dir": "plm_transfer_t1_v0", "module": "kit_t1", "level": "t3s", "levers": ["resident_rotary", "static_kv"]}   # two eager levers, one token each; no other keys
    assert set(comp) == {"decode_kit", "stock"} and decode_mod.AUTO_LEVERS == ("oneread_mmap", "sampler_exact")


class _FakeCuda:
    def __init__(self, free, total, reserved, allocated):
        self._m = (free, total, reserved, allocated)
    def mem_get_info(self): return self._m[0], self._m[1]
    def memory_reserved(self): return self._m[2]
    def memory_allocated(self): return self._m[3]


class _FakeTorch:
    def __init__(self, **kw): self.cuda = _FakeCuda(**kw)


class _FakeKit:
    """kit_t1's three memory reads as the kit file calls them."""
    def __init__(self, per_sample, held):
        self.per, self.held = per_sample, dict(held)
    def kv_bytes_per_sample(self, model): return self.per
    def allocated(self, B): return B in self.held
    def kv_bytes_held(self, B=None): return sum(v for b, v in self.held.items() if B is None or b == B)


def test_kit_fits_prices_a_batch_size_with_the_components_own_arithmetic(decode_mod):
    total = int(79.65 * GiB); weights = int(12.9e9)
    kit = _FakeKit(XLARGE_PER_SAMPLE_512, {1: XLARGE_PER_SAMPLE_512})               # the sanity pass's batch-1 slots held
    torch = _FakeTorch(free=total - weights - XLARGE_PER_SAMPLE_512 - GiB, total=total, reserved=weights + 2 * GiB, allocated=weights + GiB)
    ok, info = decode_mod.kit_fits(kit, None, 1, torch)
    assert ok and info["already_allocated"] is True
    ok, info = decode_mod.kit_fits(kit, None, 64, torch)
    assert ok and info["need_bytes"] == 64 * XLARGE_PER_SAMPLE_512 and info["per_sample_bytes"] == XLARGE_PER_SAMPLE_512 and info["reserved_unused_bytes"] == GiB and info["margin"] == 0.04
    assert decode_mod.kit_fits(kit, None, 128, torch)[0]                            # 48 GiB beside the weights on an 80 GB card: allocated on demand (the boot-budgeted cap stopped at 64)
    ok, info = decode_mod.kit_fits(kit, None, 256, torch)
    assert not ok and info["need_bytes"] == 256 * XLARGE_PER_SAMPLE_512               # 96 GiB: refused before allocating
    kit40 = _FakeKit(XLARGE_PER_SAMPLE_512, {1: XLARGE_PER_SAMPLE_512}); total40 = 40 * GiB
    torch40 = _FakeTorch(free=total40 - weights - XLARGE_PER_SAMPLE_512 - GiB, total=total40, reserved=weights + GiB, allocated=weights + GiB)
    assert decode_mod.kit_fits(kit40, None, 64, torch40)[0] and not decode_mod.kit_fits(kit40, None, 128, torch40)[0]   # a 40 GB card: the 64 x 512 row shape beside the xlarge weights (24 GiB of slots), not 128
    partial = _FakeKit(1000, {5: 3000})                                             # slots a forward already made for the batch size count as held, not needed again
    ok, info = decode_mod.kit_fits(partial, None, 5, _FakeTorch(free=10**6, total=10**7, reserved=0, allocated=0))
    assert info["already_allocated"] is True                                        # held by the fake's own rule; the real component: allocated() = buffers present
    partial.held = {}; partial.kv_bytes_held = lambda B=None: 3000 if B in (None, 5) else 0
    ok, info = decode_mod.kit_fits(partial, None, 5, _FakeTorch(free=10**6, total=10**7, reserved=0, allocated=0))
    assert ok and info["need_bytes"] == 5 * 1000 - 3000


def test_the_named_row_answers_a_batch_size_that_cannot_fit(decode_mod):
    """The refusal is OOM_POLICY's row (Handle.unit raises OutOfMemory carrying it; the caller fails the unit by name): built from a message, not an exception."""
    row = decode_mod.oom_row("sample N=256 L=512", "the static K/V slots for 256 samples at 512 positions need 96.00 GiB and the card has 66.70 GiB free of 79.65 GiB (held: the weights and the slots of batches 1); not allocated", mem_policy={"fit_info": {"need_bytes": 1}})
    assert row["oom"] is True and "result" not in row and row["mem_policy"] == {"fit_info": {"need_bytes": 1}}
    assert row["error"].startswith("OUT OF MEMORY at sample N=256 L=512: the static K/V slots for 256 samples") and row["error"].endswith(decode_mod.OOM_ADVICE)
    e = decode_mod.OutOfMemory(row)
    assert isinstance(e, RuntimeError) and str(e) == row["error"] and e.row is row and decode_mod.is_oom(e)   # the unit's exception IS the row, and reads as out-of-memory by the one test


@pytest.fixture
def tiny_progen():
    """A 2-layer ProGen on the CPU from the in-tree stock module + the decode component imported by path; the stock module's patched globals and the
    component's state are restored after the test."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    for p in (STOCK_SRC, COMPONENT):
        if p not in sys.path:
            sys.path.insert(0, p)
    from models.progen.configuration_progen import ProGenConfig
    from models.progen.modeling_progen import ProGenForCausalLM
    import kit_t1
    torch.manual_seed(0)
    model = ProGenForCausalLM(ProGenConfig(vocab_size=32, n_positions=64, n_ctx=64, n_embd=64, n_layer=2, n_head=8, rotary_dim=4)).eval()
    yield torch, model, kit_t1
    mp = kit_t1.mp
    mp.fixed_pos_embedding = kit_t1._ORIG["fixed_pos_embedding"]; mp.apply_rotary_pos_emb = kit_t1._ORIG["apply_rotary_pos_emb"]
    mp.ProGenAttention.forward = kit_t1._ORIG["attn_forward"]   # the component patches three stock names (tables ×2, the attention forward); the model forward is never patched
    kit_t1.S.__init__()


def test_static_kv_slots_are_allocated_per_batch_size_when_asked_and_released_by_name(tiny_progen):
    torch, model, kit = tiny_progen
    ids = torch.tensor([[3, 5, 7, 9, 11]]); nxt = torch.tensor([[13]])
    with torch.no_grad():                                                           # the stock forward before any patch: prefill + one cached decode step
        ref = model(ids, use_cache=True, return_dict=True); ref_step = model(nxt, past_key_values=ref.past_key_values, use_cache=True, return_dict=True).logits
    info = kit.install(model, level="t3s", max_slots=32, device="cpu")             # the static-KV level: nothing allocated at install
    assert info["batches"] == [] and kit.allocated_batches() == [] and kit.kv_bytes_held() == 0 and info["kv_bytes_per_sample"] == kit.kv_bytes_per_sample(model)
    per = kit.kv_bytes_per_sample(model)
    assert per == 2 * 32 * 64 * (4 + 4)                                             # n_layer × max_slots × n_embd × (float32 K + the model's float32 V on the CPU)
    assert kit.allocate(model, 3) is True and kit.allocated(3) and kit.allocated_batches() == [3]
    assert kit.kv_bytes_held(3) == 3 * per and kit.kv_bytes_held() == 3 * per      # the estimate IS the allocation
    for (_, b), (kc, vc) in kit.S.kv.items():
        assert b == 3 and tuple(kc.shape) == (3, 8, 32, 8) and kc.dtype == kit.KEY_SLOT_DTYPE == torch.float32 and tuple(vc.shape) == (3, 8, 32, 8)
    assert kit.S.batches == {3} and set(kit.S.stats) == {"rot_hits", "rot_builds", "allocations"} and info["level"] == "t3s"   # the component's whole state: tables, slots, the batch sizes held
    assert kit.allocate(model, 3) is False and kit.S.stats["allocations"] == 1      # idempotent
    with torch.no_grad():                                                           # a forward at a batch size nobody allocated (the load's sanity pass at batch 1) makes its slots on the way: held, not yet adopted
        out = model(ids, use_cache=True, return_dict=True); step = model(nxt, past_key_values=out.past_key_values, attention_mask=torch.ones((1, 6), dtype=torch.long), use_cache=True, return_dict=True).logits
    assert torch.equal(out.logits, ref.logits) and torch.equal(step, ref_step)      # the static-slot path is the stock's numbers
    assert kit.allocated_batches() == [1, 3] and not kit.allocated(1) and kit.kv_bytes_held(1) == per
    assert kit.allocate(model, 1) is True and kit.allocated(1) and kit.S.stats["allocations"] == 2
    r = kit.release(keep={1})
    assert r == {"evicted_kv_entries": 2, "released_batches": [3]} and kit.allocated_batches() == [1] and kit.kv_bytes_held() == per and 3 not in kit.S.batches
    assert kit.allocate(model, 3) is True and kit.allocated_batches() == [1, 3]     # a released batch size is allocated again when asked
    assert kit.release(keep=())["released_batches"] == [1, 3] and kit.allocated_batches() == [] and kit.kv_bytes_held() == 0
