"""K.T23-auto (0.5.18.15): the tri-mul A-block schedule is PLANNED from the agreed free device bytes — `ablock_auto_cap` / `ablock_rows(..., N, C,
elt_bytes, free_bytes)`. Pure arithmetic (no CUDA needed): the B300 card at 70,320 / 66,720 / 52,000 tokens x8, the H100 K/M shapes, the sources
and their precedence (fixed > env > auto/whole; ROWPAIR_TRIMUL_ABLOCK=0 = off), no CUDA = whole."""
import os
import pytest

from opt_core.mem.rowpair import trimul as T

GiB = 2 ** 30
C, ELT, P = 128, 2, 8
B300_TOTAL = 275_040 * 2 ** 20          # the B300 card (MiB as nvidia-smi reports it)
H100_TOTAL = 81_559 * 2 ** 20


def u_bytes(N, P_):                       # one rank's z shard = the whole A plane of its update
    return -(-N // P_) * N * C * ELT


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("ROWPAIR_TRIMUL_ROWS_A", "ROWPAIR_TRIMUL_ABLOCK"):
        monkeypatch.delenv(k, raising=False)


# The free device bytes recorded at the trunk pair stack's FIRST update (after the template stage; device_free_bytes = driver free + the allocator's reserved-allocated slack),
# from the TPCENSUS `after_template` marks of the two row-blocked forwards on the 287.429 GB B300 card:
#   the 70,320-token x8 forward (first recorded run) (70,320 x8): used 258.551, reserved 253.166, alloc 163.965 GB -> 118.08 GB free; ran 6 x 1,472 (env cap 1,536), device peak 247.2 GB
#   the 66,720-token x8 forward (second recorded run) (66,720 x8): used 234.386, reserved 229.049, alloc 147.846 GB -> 134.25 GB free; ran 6 x 1,392 (env cap 1,536), device peak 226.8 GB, EXIT ok
CENSUS_FREE = {70320: int((287.429 - 258.551 + (253.166 - 163.965)) * 1e9), 66720: int((287.429 - 234.386 + (229.049 - 147.846)) * 1e9)}
TOTAL_B = int(287.429e9)


@pytest.mark.parametrize("N,ran_ra,ran_passes", [(70320, 1472, 6), (66720, 1392, 6)])
def test_b300_x8_at_66k_70k_plans_at_least_the_validated_passes_from_the_census_free_bytes(N, ran_ra, ran_passes):
    Rmax = -(-N // P); free = CENSUS_FREE[N]
    cap, src = T.ablock_auto_cap(Rmax, N, C, ELT, free)
    assert src == "auto" and cap % 128 == 0 and 128 <= cap <= 1536 < Rmax
    assert 2 * cap * N * C * ELT <= T.ABLOCK_BLOCK_FRAC * free and 2 * (cap + 128) * N * C * ELT > T.ABLOCK_BLOCK_FRAC * free    # the LARGEST multiple of 128 within the share
    ra, passes, src2 = T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=free)
    assert src2 == "auto" and passes >= ran_passes and ra <= ran_ra and ra % 16 == 0 and ra * passes >= Rmax and ra * (passes - 1) < Rmax   # never a larger block than the one that ran
    print(f"N={N} x{P}: free of record {free / 1e9:.1f} GB -> cap {cap} -> schedule {passes} x {ra} rows (ran {ran_passes} x {ran_ra}); working set {2 * ra * N * C * ELT / 1e9:.1f} GB")


def test_the_calibration_reproduces_66k_and_errs_toward_more_passes_at_70k():
    assert T.ablock_rows(8340, None, N=66720, C=C, elt_bytes=ELT, free_bytes=CENSUS_FREE[66720]) == (1392, 6, "auto")   # = the schedule 9I8E-3 completed with
    assert T.ablock_rows(8790, None, N=70320, C=C, elt_bytes=ELT, free_bytes=CENSUS_FREE[70320]) == (1264, 7, "auto")   # one pass more than 9BC8-3's 6 x 1,472
    # under MORE memory pressure the plan only gets finer, never coarser
    assert T.ablock_rows(8790, None, N=70320, C=C, elt_bytes=ELT, free_bytes=int(0.7 * CENSUS_FREE[70320]))[1] >= 9


def test_60k_x8_stays_whole_as_it_ran():
    # 60,120 x8 (9EQW, one pass on 0.5.18.8, EXIT ok): u = 115.7 GB; free after its template stage by the same structure as the two censuses (used ~1.64u, slack ~0.57u) ~ 163.6 GB
    N = 60120; Rmax = -(-N // P); u = u_bytes(N, P); free = int(TOTAL_B - 1.07 * u)
    ra, passes, src = T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=free)
    print(f"N={N} x{P}: u {u / 1e9:.1f} GB, free ~{free / 1e9:.1f} GB -> {src} {passes} x {ra}")
    assert (ra, passes, src) == (Rmax, 1, "whole") and u <= T.ABLOCK_WHOLE_FRAC * free


@pytest.mark.parametrize("N,P_,total,other_gib", [
    (52000, 8, B300_TOTAL, 45), (48000, 8, B300_TOTAL, 45),                          # <= 52K x8 on B300: whole (the 52K shot ran one pass)
    (5120, 2, H100_TOTAL, 8), (10240, 2, H100_TOTAL, 8), (10240, 4, H100_TOTAL, 8), (10240, 8, H100_TOTAL, 8),
    (15360, 4, H100_TOTAL, 8), (15360, 8, H100_TOTAL, 8), (20480, 4, H100_TOTAL, 8), (23040, 8, H100_TOTAL, 8),   # H100 shapes modelled (residents: z shard + 8 GiB) at u/free <= 0.75: whole; 23,040 x4 is NOT in this list — see the next test
])
def test_shapes_whose_plane_fits_plan_whole_identical_to_today(N, P_, total, other_gib):
    Rmax = -(-N // P_); u = u_bytes(N, P_)
    free = total - u - other_gib * GiB                                                  # the z shard itself + the model's other residents are already on the card
    cap, src = T.ablock_auto_cap(Rmax, N, C, ELT, free)
    assert (cap, src) == (Rmax, "whole"), (N, P_, u / GiB, free / GiB)
    assert T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=free) == (Rmax, 1, "whole")


def test_sources_and_precedence(monkeypatch):
    Rmax, N = 8790, 70320
    assert T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=None) == (Rmax, 1, "whole")       # no CUDA on any rank: whole
    assert T.ablock_rows(Rmax, None) == (Rmax, 1, "whole")                                                # no shape given (0.5.18.13 callers): whole
    assert T.ablock_rows(Rmax, 1536, N=N, C=C, elt_bytes=ELT, free_bytes=60 * GiB) == (1472, 6, "fixed")  # RA= beats the planner
    monkeypatch.setenv("ROWPAIR_TRIMUL_ROWS_A", "1536")
    assert T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=200 * GiB) == (1472, 6, "env")  # the diagnostic override beats the planner (which would say whole)
    monkeypatch.setenv("ROWPAIR_TRIMUL_ABLOCK", "0")
    assert T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=60 * GiB) == (Rmax, 1, "off")   # bisecting by name: whole whatever the rest say


def test_planner_constants_are_documented():
    api = open(os.path.join(os.path.dirname(T.__file__), "API.md")).read()
    assert "ABLOCK_WHOLE_FRAC" in api and "ABLOCK_BLOCK_FRAC" in api and "ablock_src=auto" in api and 0 < T.ABLOCK_BLOCK_FRAC <= T.ABLOCK_WHOLE_FRAC < 1


def test_whole_threshold_boundary_keeps_rows_of_record_whole_and_blocks_above_080():
    """0.5.18.16: ABLOCK_WHOLE_FRAC = 0.80 — the modelled 23,040 x4 H100 shape (u/free 0.79, the 0.5.18.15 note's example) plans WHOLE again; a shape at ~0.85 of free is
    row-blocked; the 60,120 x8 recorded row (0.72) keeps >= 0.08 of margin."""
    N, P_ = 23040, 4; Rmax = -(-N // P_); u = u_bytes(N, P_); free = H100_TOTAL - u - 8 * GiB
    assert 0.75 < u / free < 0.80 and T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=free) == (Rmax, 1, "whole")
    free85 = int(u / 0.85); ra, passes, src = T.ablock_rows(Rmax, None, N=N, C=C, elt_bytes=ELT, free_bytes=free85)
    assert src == "auto" and passes >= 3 and 2 * ra * N * C * ELT < u                       # blocked above the threshold; the pass working set is below the whole plane
    N6 = 60120; u6 = u_bytes(N6, P); free6 = int(TOTAL_B - 1.07 * u6); assert u6 / free6 <= T.ABLOCK_WHOLE_FRAC - 0.08
