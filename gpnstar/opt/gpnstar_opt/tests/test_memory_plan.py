"""The memory plan of the reduced K/V route is plain arithmetic on known sizes: these tests lift the three pure functions (and their
constants) out of accel/patches.py by syntax tree - no torch here - and check the threshold arithmetic and the decision."""
import ast
import os

ACCEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "accel", "patches.py")
NAMES = ("_reduced_route_bytes", "_stock_layer_bytes", "_memory_decision")


def _lift():
    tree = ast.parse(open(ACCEL).read())
    body = [n for n in tree.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id.startswith("MEMORY_") for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name in NAMES)]
    ns = {}
    exec(compile(ast.Module(body=body, type_ignores=[]), ACCEL, "exec"), ns)  # noqa: S102 - our own source, pure arithmetic
    assert all(n in ns for n in NAMES), sorted(ns)
    return ns


GIB = 1 << 30
M942 = 942 * 128 * 45          # stock K/V rows of a 942 x 128 batch with 45 clades
M727 = 727 * 128 * 45


def test_reduced_route_bytes_arithmetic():
    f = _lift()["_reduced_route_bytes"]
    r = f(16384, 512, 512, M942, True, 4)
    assert r["x_bytes"] == 16384 * 512 * 4 and r["projection_bytes"] == 16384 * 512 * 4
    assert r["check_bytes"] == GIB                                   # the self-check slice, capped
    assert r["extra_bytes"] == r["x_bytes"] + r["projection_bytes"] + r["check_bytes"]
    assert f(16384, 512, 512, M942, False, 4)["check_bytes"] == 0   # a validated pair pays no check
    assert f(100, 8, 8, 100 * 45, True, 4)["check_bytes"] == 100 * 45 * 8 * 4   # small shapes: the whole reference, below the cap


def test_stock_layer_bytes_is_five_k_tensors():
    ns = _lift()
    assert ns["_stock_layer_bytes"](M942, 512, 4) == ns["MEMORY_STOCK_LAYER_FACTOR"] * M942 * 512 * 4


def test_decision_threshold_is_exact_and_monotonic():
    ns = _lift(); dec = ns["_memory_decision"]
    extra, layer = 3 * GIB, ns["_stock_layer_bytes"](M942, 512, 4)
    fits, d = dec(extra, layer, 10 ** 12, 1.16)
    assert fits and d["need_bytes"] == int(round((extra + layer) * 1.16)) + ns["MEMORY_MARGIN_BYTES"]
    need = d["need_bytes"]
    assert dec(extra, layer, need, 1.16)[0] is True and dec(extra, layer, need - 1, 1.16)[0] is False   # the threshold itself
    assert dec(extra, ns["_stock_layer_bytes"](M727, 512, 4), need - 1, 1.16)[0] is True               # a smaller batch fits where the larger does not
    assert dec(extra, layer, need - 1, 1.05)[0] is True                                                # the expandable-segments layout needs less reserve


def test_decision_examples_default_allocator_80gb():
    """A 942 x 128 batch does not fit the reduced route beside the stock attention in ~63 GiB obtainable with the default allocator layout
    (-> stock projections by name); a 727 x 128 batch fits in ~66 GiB."""
    ns = _lift(); dec, rb, sl = ns["_memory_decision"], ns["_reduced_route_bytes"], ns["_stock_layer_bytes"]
    e942 = rb(500_000, 512, 512, M942, True, 4)["extra_bytes"]
    assert dec(e942, sl(M942, 512, 4), 63 * GIB, ns["MEMORY_ALLOC_OVERHEAD"]["default"])[0] is False
    e727 = rb(400_000, 512, 512, M727, True, 4)["extra_bytes"]
    assert dec(e727, sl(M727, 512, 4), 66 * GIB, ns["MEMORY_ALLOC_OVERHEAD"]["default"])[0] is True


def test_releasable_bytes_count_only_where_released_blocks_are_reusable():
    ns = _lift(); dec = ns["_memory_decision"]; layer = ns["_stock_layer_bytes"](M942, 512, 4)
    extra, budget, stash = int(1.3 * GIB), 50 * GIB, 15 * GIB
    assert dec(extra, layer, budget, 1.05)[0] is False                                              # measured budget alone: step aside
    fits, d = dec(extra, layer, budget, 1.05, releasable_bytes=stash, count_releasable=True)       # expandable segments: the stash returns whole
    assert fits and d["budget_bytes"] == budget + stash and d["measured_budget_bytes"] == budget and d["releasable_bytes"] == stash
    fits, d = dec(extra, layer, budget, 1.16, releasable_bytes=stash, count_releasable=False)      # default layout: not counted
    assert fits is False and d["budget_bytes"] == budget

def test_layer_factor_of_the_route_decides_near_the_ceiling():
    """The plan budgets the per-layer K-sized tensors of the ROUTE in use: stock's SDPA-math layout needs five (K, V, the scaled key, its
    transposed copy, V's copy); the colattn layout route holds one at a time (colattn.COLATTN_LAYER_FACTOR = 1.25 with its activations).
    At 992 x 128 on an 80 GB card ~48 GiB is obtainable once the source stage is done: the five-tensor estimate steps aside (65.8 GiB
    needed), the layout route's estimate fits (18.6 GiB) -- as measured on an H100 80 GB card with the default allocator."""
    ns = _lift(); dec, rb, sl = ns["_memory_decision"], ns["_reduced_route_bytes"], ns["_stock_layer_bytes"]
    M992 = 992 * 128 * 45
    extra = rb(61440, 1024, 512, M992, True, 4)["extra_bytes"]
    five = sl(M992, 512, 4)
    one = sl(M992, 512, 4, 1.25)
    assert five == ns["MEMORY_STOCK_LAYER_FACTOR"] * M992 * 512 * 4 and one == int(round(1.25 * M992 * 512 * 4))
    ovh = ns["MEMORY_ALLOC_OVERHEAD"]["default"]
    assert dec(extra, five, int(48.75 * GIB), ovh)[0] is False
    fits, d = dec(extra, one, int(48.75 * GIB), ovh)
    assert fits and d["need_bytes"] < 20 * GIB
