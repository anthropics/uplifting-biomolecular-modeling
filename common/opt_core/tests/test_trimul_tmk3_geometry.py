"""opt_core.trimul.tmk3 has no proof-range size cap: above n_min the only size condition is the TM-K3 kernels' launch geometry (stages A/C tile
the N*N token rows on CUDA grid axis 1, limit 65535 programs).  The package schedules the stage tile height by N (fpf_trimul.trimul._select_cfg:
the cell's own tiles through their limit, then 128 / 256 / 512-row tiles of the same arithmetic) so every served cell launches through N = 5632;
above that the geometry is refused by name before any launch.  CPU-only (the package's cfg tables + arithmetic)."""
import os
import re

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
HERE = os.path.dirname(os.path.abspath(__file__)); CORE = os.path.dirname(HERE)


@pytest.fixture(scope="module")
def mods():
    import sys
    os.environ.setdefault("FPF_TRIMUL_MODE", "exact")
    from opt_core import kernels as CK, trimul as T
    core_copy = os.path.join(os.path.dirname(CK.__file__), "fpf_trimul")
    top = sys.modules.get("fpf_trimul")
    if top is not None and not os.path.abspath(getattr(top, "__file__", "") or "").startswith(os.path.abspath(core_copy)):
        for m in [m for m in sys.modules if m == "fpf_trimul" or m.startswith("fpf_trimul.")]:      # a copy planted by another test in this process: this file reads the core's
            del sys.modules[m]
    CK.route("fpf_trimul")
    import fpf_trimul.kernels as K, fpf_trimul.trimul as Tm
    assert os.path.abspath(K.__file__).startswith(os.path.abspath(core_copy)), K.__file__
    return T, K, Tm


CELLS = ((torch.bfloat16, 128), (torch.bfloat16, 256), (torch.float32, 384))
LAST_SERVED = 5632                                   # 512-row stage-A tiles: N * ceil(N / 512) <= 65535 through N = 5632 (11 * 512), every cell


def _table_cfg(Tm, cdt, N, C):
    """The cell's cfg WITHOUT the schedule: the CONFIG_TABLE bucket + the shared-memory guard, as _select_cfg served it before the schedule existed."""
    cls = "bf16" if cdt in (torch.bfloat16, torch.float16) else "fp32"
    tab = Tm.CONFIG_TABLE[cls]; sub = tab.get("C%d" % C, tab["*"])
    cfg = None
    for k, v in sub.items():
        if k == "*":
            cfg = cfg or v
        else:
            lo, hi = [int(x) for x in k.split("-")]
            if lo <= N <= hi: cfg = v
    cfg = dict(cfg); elem = 2 if cls == "bf16" else 4
    if Tm.smem_bytes(cfg["A"], elem, False) > Tm.SMEM_CAP: cfg["A"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    if Tm.smem_bytes(cfg["C"], elem, True) > Tm.SMEM_CAP: cfg["C"] = dict(v=9, BM=64, BN=64, BK=32, num_warps=4, num_stages=2)
    return cfg


def test_stage_grid_y_is_what_trimul_forward_launches(mods):
    T, K, Tm = mods
    assert K.CUDA_GRID_Y_MAX == 65535
    a9, c9, c2, a5 = dict(v=9, BM=64), dict(v=9, BM=128), dict(v=2, BM=64), dict(v=5, BM=64, SPLIT=96)
    assert K.stage_grid_y("A", 2047, a9) == 2047 * 32 and K.stage_grid_y("A", 2048, a9) == 2048 * 32 == 65536        # grid_a: N * ceil(N / BM)
    assert K.stage_grid_y("C", 2896, c9) == -(-2896 * 2896 // 128) == 65522 and K.stage_grid_y("C", 2897, c9) > 65535  # _k_out9: ceil(N*N / BM)
    assert K.stage_grid_y("C", 2000, c2) == 2000 * 32 and K.stage_grid_y("A", 9999, a5) == 96 == K.stage_grid_y("C", 9999, a5)   # grid_c epilogues; split variants
    cfg = Tm._select_cfg(torch.bfloat16, 705, 128)
    assert K.launch_grid_y(705, 128, cfg) == max(K.stage_grid_y("A", 705, cfg["A"]), K.stage_grid_y("C", 705, cfg["C"]))


def test_below_each_cells_own_limit_the_cfg_is_the_tables_unchanged(mods):
    """The census: every N a cell's own tiles launch at is served EXACTLY the table's cfg (no schedule) — nothing measured changes."""
    T, K, Tm = mods
    own_limit = {}
    for cdt, C in CELLS:
        limit = max(N for N in range(101, 4000) if K.launch_grid_y(N, C, _table_cfg(Tm, cdt, N, C)) <= K.CUDA_GRID_Y_MAX)
        own_limit[(str(cdt), C)] = limit
        for N in range(101, limit + 1):
            assert Tm._select_cfg(cdt, N, C) == _table_cfg(Tm, cdt, N, C), (str(cdt), C, N)
        assert Tm._select_cfg(cdt, limit + 1, C) != _table_cfg(Tm, cdt, limit + 1, C)                       # the first N the schedule acts on
    assert own_limit == {("torch.bfloat16", 128): 2047, ("torch.bfloat16", 256): 2849, ("torch.float32", 384): 2849}, own_limit


def test_the_schedule_serves_every_cell_through_n_5632_with_the_tables_arithmetic(mods):
    T, K, Tm = mods
    heights = [bm for bm, _s in Tm.GRID_SCHEDULE]
    assert heights == [64, 128, 256, 512] and Tm.GRID_SCHEDULE[2][1] == {"num_warps": 8} == Tm.GRID_SCHEDULE[3][1] and Tm.BN_FLOOR == 32
    for cdt, C in CELLS:
        elem = 2 if cdt == torch.bfloat16 else 4
        served = [N for N in range(101, 6200) if K.launch_grid_y(N, C, Tm._select_cfg(cdt, N, C)) <= K.CUDA_GRID_Y_MAX]
        assert served == list(range(101, LAST_SERVED + 1)), (str(cdt), C, [n for n in range(101, 6200) if (n in served) != (n <= LAST_SERVED)][:5])
        steps = {}
        for N in range(101, LAST_SERVED + 1):
            cfg, tab = Tm._select_cfg(cdt, N, C), _table_cfg(Tm, cdt, N, C)
            assert {k: v for k, v in cfg.items() if k not in ("A", "C")} == {k: v for k, v in tab.items() if k not in ("A", "C")}   # B, LN, pad, contract ...: untouched
            for stage, dual_x in (("A", False), ("C", True)):
                st, tt = cfg[stage], tab[stage]
                assert {k: v for k, v in st.items() if k not in ("BM", "BN", "num_warps", "num_stages")} == {k: v for k, v in tt.items() if k not in ("BM", "BN", "num_warps", "num_stages")}, (N, stage)  # v, BK: the arithmetic
                assert st["BM"] >= tt["BM"] and st["BM"] in heights + [tt["BM"]] and st["num_stages"] <= tt["num_stages"] and Tm.smem_bytes(st, elem, dual_x) <= Tm.SMEM_CAP
                assert st["num_warps"] == (max(tt["num_warps"], 8) if st["BM"] >= 256 and st["BM"] != tt["BM"] else tt["num_warps"])
                floor = Tm.BN_FLOOR * (2 if tt.get("v", 2) == 14 else 1)                                    # the accumulator rule: BN_table * BM_table / BM, floor 32 columns (64 for v=14)
                assert st["BN"] == (tt["BN"] if st["BM"] == tt["BM"] else min(tt["BN"], max(tt["BN"] * tt["BM"] // st["BM"], floor))) and tt["BN"] % st["BN"] == 0
                if st["BM"] != tt["BM"]:                                                                       # scheduled: the smallest height that fits, and only when the table's does not
                    assert K.stage_grid_y(stage, N, tt) > K.CUDA_GRID_Y_MAX >= K.stage_grid_y(stage, N, st)
                    assert all(K.stage_grid_y(stage, N, dict(tt, BM=bm)) > K.CUDA_GRID_Y_MAX for bm in heights if tt["BM"] < bm < st["BM"])
                    steps.setdefault(stage, {}).setdefault(st["BM"], N)
        first = {stage: sorted(v.items()) for stage, v in steps.items()}
        expect_a = [(128, 2048), (256, 2850), (512, 4096)] if C == 128 else [(256, 2850), (512, 4096)]
        expect_c = [(128, 2048), (256, 2897), (512, 4096)] if C == 128 else [(256, 2897), (512, 4096)]
        assert first == {"A": expect_a, "C": expect_c}, (str(cdt), C, first)


class _Z:
    """A CUDA-resident [N, N, C] stand-in for eligibility (no data)."""
    is_cuda = True

    def __init__(self, N, C, dtype):
        self.shape, self.dtype, self.device = (N, N, C), dtype, torch.device("cuda", 0)

    def dim(self):
        return 3


def _weights(C, dtype):
    t = lambda *s: torch.zeros(*s, dtype=dtype)
    return {"ln_in_w": t(C), "ln_in_b": t(C), "w_ap": t(C, C), "w_bp": t(C, C), "w_ag": t(C, C), "w_bg": t(C, C), "ln_out_w": t(C), "ln_out_b": t(C), "w_o": t(C, C), "w_og": t(C, C)}


def test_tmk3_refuses_only_the_unlaunchable_geometry_by_name(mods):
    T, K, Tm = mods
    for C, last_ok in ((128, LAST_SERVED), (256, LAST_SERVED)):
        prov = T.tmk3(lambda m, C=C: _weights(C, torch.bfloat16), mode="exact")
        for N, served in ((512, True), (2048, True), (3072, True), (4096, True), (last_ok, True), (last_ok + 1, False), (8192, False)):
            call = T.Call(module=object(), z=_Z(N, C, torch.bfloat16), mask=None, direction="outgoing", residual=False, orig=lambda: None)
            if served:
                prov.eligible(call)
            else:
                with pytest.raises(T.Refused) as ei:
                    prov.eligible(call)
                assert ei.value.reason == T.GRID_REFUSED == "launch_grid_y>65535", ei.value.reason
    call = T.Call(module=object(), z=_Z(100, 128, torch.bfloat16), mask=None, direction="outgoing", residual=False, orig=lambda: None)
    with pytest.raises(T.Refused) as ei:
        T.tmk3(lambda m: _weights(128, torch.bfloat16), mode="exact").eligible(call)
    assert ei.value.reason == "n<101"                                                              # the small-N floor stays (the vendor's algorithm switch)


def test_no_proof_range_size_word_in_the_tmk3_path():
    src = open(os.path.join(CORE, "opt_core", "trimul.py")).read()
    assert not re.search(r'\bn_max\b|"n>|n>%d|N_MAX|NMAX', src)
    i = src.index("def tmk3("); body = src[i: src.index("\ndef custom(", i)]
    assert not re.search(r"\b(2047|2048|2849|2850|4096|5632|5633)\b", body.split('"""', 2)[2])          # numbers live in the docstring's evidence sentence only, never in code
