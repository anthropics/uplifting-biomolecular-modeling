"""The cells' fallback accounting (no torch, no model): a refusal of the TRUNK pair width that is not a route by design is a degradation —
counted under `degraded`, returned by `fallbacks()` for the kit's EXIT census (`fallbacks=<lever>:<reason>=<n>,...`), and the same predicate the
strict switch raises on; the template stack's width, the token floor, chunked and training calls are routes by design (counted `fallback:<reason>`
on the census, never in `fallbacks()`)."""
import pytest

from openfold3_ob0_opt.cells import pairfused, rollout


@pytest.fixture(autouse=True)
def fresh_counts():
    saved = {k: pairfused.STATE[k] for k in ("counts", "reasons", "degraded", "levers", "strict")}
    pairfused.STATE["counts"] = {n: {"served": 0, "fallback": 0} for n in pairfused.LEVER_NAMES}
    pairfused.STATE["reasons"] = {n: {} for n in pairfused.LEVER_NAMES}
    pairfused.STATE["degraded"] = {n: {} for n in pairfused.LEVER_NAMES}
    pairfused.STATE["levers"] = list(pairfused.LEVER_NAMES)
    pairfused.STATE["strict"] = False
    yield
    pairfused.STATE.update(saved)


def test_routes_by_design_are_not_degradations():
    C = pairfused.TRUNK_C
    assert not pairfused._is_degradation("c:64x64", 64)                 # the template pair stack's width: the line's route by design
    assert not pairfused._is_degradation("n<256", C)                    # below the TriMul cell's token floor
    assert not pairfused._is_degradation("chunk", C) and not pairfused._is_degradation("training", C)
    assert not pairfused._is_degradation("stream:float32", C)           # an fp32 pair stream (confidence head under pairformer_dtype=float32; the fp32 line): by design
    assert not pairfused._is_degradation("cell:cueq", C)                # the core's measured TriMul table names the stock op for the class (N<=100 bucket): the small-N route's provider-era word
    assert not pairfused._is_degradation("cell:torch_math", C)
    assert not pairfused._is_degradation("cell:e01:stock_block(256<keys<512)", C)   # a tri-attention engage cell: the engine's own block by cell
    assert pairfused._is_degradation("dtype:float16", C)                # the trunk shape refused for any other reason is a degradation
    assert pairfused._is_degradation("mask-shape", C)


def test_fallbacks_counts_degradations_only():
    C = pairfused.TRUNK_C
    pairfused._count("trimul_v4", False, "c:64x64", C=64)
    pairfused._count("trimul_v4", False, "n<256", C=C)
    pairfused._count("triatt_block", True)
    assert pairfused.fallbacks() == {}
    pairfused._count("triatt_block", False, "mask-shape", C=C)
    pairfused._count("triatt_block", False, "mask-shape", C=C)
    pairfused._count("pair_transition", False, "dtype:float16", C=C)
    assert pairfused.fallbacks() == {"triatt_block:mask-shape": 2, "pair_transition:dtype:float16": 1}
    cen = "\n".join(pairfused.census_lines())
    assert "name=triatt_block state=on served=1 fallback=2 degraded=2 fallback:mask-shape=2" in cen
    assert "name=trimul_v4 state=on served=0 fallback=2 degraded=0" in cen


def test_strict_raises_on_exactly_the_degradations():
    C = pairfused.TRUNK_C
    pairfused.STATE["strict"] = True
    pairfused._strict_refusal("trimul_v4", "n<256", C, 100)             # a route by design: no raise
    pairfused._strict_refusal("trimul_v4", "c:64x64", 64, 400)
    with pytest.raises(RuntimeError):
        pairfused._strict_refusal("trimul_v4", "mask-shape", C, 400)


def test_rollout_has_no_fallback_path():
    assert rollout.fallbacks() == {}
