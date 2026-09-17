"""cc 8.0 (A100) rows of the core's per-architecture kernel tables: schema, the sm_80 shared-memory budget, selection keys.  Pure-python (no torch):
the tables are data files read here directly; the arithmetic mirrors the documented per-stage estimate of each package."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS = os.path.join(os.path.dirname(HERE), "opt_core", "kernels")
SM80_SMEM = 163 * 1024          # A100: opt-in dynamic shared memory per block = 166,912 B


def _trimul_v3_smem(stage, elem, dual_x):
    """fpf_trimul.trimul.smem_bytes: pipelined-loop estimate for A' (x + 2 weight tiles) / C' (2 x + 2 weight tiles)."""
    xt = stage["BM"] * stage["BK"] * (2 if dual_x else 1); wt = 2 * stage["BK"] * stage["BN"]
    return (xt + wt) * elem * stage["num_stages"]


def test_fpf_trimul_arch_tables_sm80_entry():
    at = json.load(open(os.path.join(KERNELS, "fpf_trimul", "arch_tables.json")))
    tabs = at["tables"]
    assert {"sm_90", "sm_100", "sm_103", "sm_80"} <= set(tabs)
    assert tabs["sm_90"]["fp32_384"] is None                                   # H100: shipped tiles (no override)
    sm80 = tabs["sm_80"]
    assert {"fp32_384", "bf16_256", "bf16_128", "source"} <= set(sm80)
    for key, elem in (("fp32_384", 4), ("bf16_256", 2), ("bf16_128", 2)):
        cell = sm80[key]
        for stage, dual in (("A", False), ("C", True)):
            s = cell[stage]
            assert set(s) == {"v", "BM", "BN", "BK", "num_warps", "num_stages"} and s["v"] == 9, (key, stage, s)
            assert _trimul_v3_smem(s, elem, dual) <= SM80_SMEM, (key, stage, _trimul_v3_smem(s, elem, dual))
    # every other arch entry keeps the single fp32_384 cell form
    for arch in ("sm_100", "sm_103"):
        assert set(tabs[arch]) == {"fp32_384", "source"} and set(tabs[arch]["fp32_384"]) == {"A", "C"}


def test_fpf_trimul_loader_applies_every_cls_C_key():
    """trimul.py's arch-table loader keys on "<cls>_<C>" for every shipped TILES cell (not only fp32_384): a source check (the module imports torch)."""
    src = open(os.path.join(KERNELS, "fpf_trimul", "trimul.py")).read()
    assert '.get("%s_%d" % (_cls, _C))' in src and 'get("fp32_384")' not in src


def test_flash_triattn_cc_triton_rows():
    """flash_triattn.py: _CONFIG_TABLE_BY_CC is keyed "cc|triton major.minor" (a named exception) / "cc|*" (the capability's default); cc 9.0 has NO
    row (H100 keeps _CONFIG_TABLE / _CONFIG_TABLE_F32); "8.0|*" = the tuned multi-stage rows, "8.0|2.3" = _SAFE_SINGLE_STAGE (every cell one
    stage: triton 2.3.x aborts sm_80 lowering at num_stages >= 2); read by AST."""
    import ast
    src = open(os.path.join(KERNELS, "flash_triattn.py")).read()
    mod = ast.parse(src)
    def _table(name):
        node = next(n for n in mod.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == name)
        return node.value
    safe = _table("_SAFE_SINGLE_STAGE"); bycc = _table("_CONFIG_TABLE_BY_CC")
    keys = [k.value for k in bycc.keys]
    assert keys == ["8.0|*", "8.0|2.3", "10.0|*", "10.3|*"], keys
    assert not any(k.startswith("9.0") for k in keys)
    for row in bycc.values[2:]:                                                               # cc 10.0 / 10.3: D=32 and D=16 cells, bf16 and fp32 (D=64/128 = the default tables)
        assert [k.value for k in row.keys] == ["16bit", "fp32"] and all([k.value for k in sub.keys] == [32, 16] for sub in row.values)
        cfg = {kw.arg: kw.value.value for kw in row.values[0].values[0].elts[0].elts[1].keywords}    # bf16 D=32: one cell on both cards
        assert cfg == {"BLOCK_M": 128, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "num_stages": 3, "ORDER": 0}, cfg
    assert getattr(bycc.values[1], "id", None) == "_SAFE_SINGLE_STAGE"                       # the exception row = the safe single-stage set
    for cls_node, dmap in zip(safe.keys, safe.values):
        assert cls_node.value in ("16bit", "fp32")
        assert sorted(k.value for k in dmap.keys) == [16, 32, 64, 128]
        for lst in dmap.values:
            (tup,) = lst.elts
            cfg = {kw.arg: kw.value.value for kw in tup.elts[1].keywords}
            assert cfg["num_stages"] == 1 and set(cfg) == {"BLOCK_M", "BLOCK_N", "ROWS", "num_warps", "num_stages", "ORDER"}, cfg
    wild = bycc.values[0]
    assert [k.value for k in wild.keys] == ["16bit"] and sorted(k.value for k in wild.values[0].keys) == [16, 32]
    stages = {k.value: {kw.arg: kw.value.value for kw in lst.elts[0].elts[1].keywords}["num_stages"] for k, lst in zip(wild.values[0].keys, wild.values[0].values)}
    assert stages == {32: 3, 16: 2}                                                           # the tuned (multi-stage) default
    assert "pick_config(D, SQ, SK, H, q.dtype, q.device)" in src and "def _launch_or_safe(" in src and "class BuildFailed(" in src and src.count("if is_oom(e):") == 2


def test_fpf_transition_sm80_pinned_cells():
    """transition.py PINNED_CONFIGS: the sm80 exact cell(s) beside the sm90 ones (AST read; the module imports torch)."""
    import ast
    src = open(os.path.join(KERNELS, "fpf_transition", "transition.py")).read()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "PINNED_CONFIGS")
    table = {tuple(ast.literal_eval(k)): ast.literal_eval(v) for k, v in zip(node.value.keys, node.value.values)}
    assert {(256, 1024, "sm90"), (128, 512, "sm90"), (384, 1536, "sm90")} <= set(table)
    assert table[(128, 512, "sm80")] == {"BM": 64, "BH": 32, "num_warps": 4, "num_stages": 2, "IL": 1}
    assert (384, 1536, "sm80") not in table and (256, 1024, "sm80") not in table          # no exact sm80 cell for those (see CANDIDATE_CONFIGS_UNVERIFIED)
    c = table[(128, 512, "sm80")]
    assert c["BM"] * 128 * 2 + (2 * 128 * c["BH"] * 2 + c["BH"] * 128 * 2) * c["num_stages"] + c["BM"] * c["BH"] * 2 <= SM80_SMEM


def test_k2b_cells_sm80_entry_is_the_package_default_cell():
    tab = json.load(open(os.path.join(KERNELS, "fpf_triatt_k2b", "K2B_CELLS.json")))
    assert tab["by_cc"]["8.0"] == "sm80" and tab["by_cc"]["9.0"] == "sm90"
    assert tab.get("by_name", {}).get("A100") == "sm80"
    e80, e90 = tab["entries"]["sm80"], tab["entries"]["sm90"]
    assert e80["status"].startswith("verified:") and "A100" in e80["status"]
    assert e80["bf16"]["32"] == e90["bf16"]["32"]                      # the A100 cell = the H100 cell = the package default's own values (bitwise to it)
    (max_seq, cell), = e80["bf16"]["32"]
    assert max_seq == 1 << 30 and cell == {"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "num_stages": 3, "ORDER": 0, "MAXNREG": 128}


def test_lnl_fused_tiles_by_arch_8_0_row_covers_the_9_0_keys():
    """lnl_fused.tiles_by_arch.json (read by lnl_fused._tile_table; PF_LNL_TILES overrides the path): the '8.0' row pins a config for every autotune
    key the '9.0' row pins (fused transition + LN-linear kernels), so an A100 warm-up searches no config list."""
    tab = json.load(open(os.path.join(KERNELS, "lnl_fused.tiles_by_arch.json")))
    assert {"8.0", "9.0"} <= set(tab)
    for kern, keys90 in tab["9.0"].items():
        keys80 = tab["8.0"][kern]
        assert set(keys90) <= set(keys80), (kern, sorted(set(keys90) - set(keys80))[:3])
        for cfg in keys80.values():
            assert {"num_warps", "num_stages"} <= set(cfg) and set(cfg) <= {"BM", "BH", "num_warps", "num_stages"}, cfg
    src = open(os.path.join(KERNELS, "lnl_fused.py")).read()
    assert "PF_LNL_TILES" in src and "lnl_fused.tiles_by_arch.json" in src


# ---------------------------------------------------------------------------------------------- in-module cc rows (AST reads; the modules import torch)
def _module_tables(fname, names):
    """{name: literal value} of the top-level assignments `name = <literal>` / `name: T = <literal>` in kernels/<fname> (ast.literal_eval; `1 << 30`-style
    shifts folded first)."""
    import ast
    src = open(os.path.join(KERNELS, fname), encoding="utf-8").read()

    class _Fold(ast.NodeTransformer):
        def visit_BinOp(self, node):
            self.generic_visit(node)
            if isinstance(node.op, ast.LShift) and isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                return ast.copy_location(ast.Constant(node.left.value << node.right.value), node)
            return node

    out = {}
    for node in ast.parse(src).body:
        target, value = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        if target in names and value is not None:
            out[target] = ast.literal_eval(_Fold().visit(value))
    assert set(out) == set(names), (fname, sorted(set(names) - set(out)))
    return out, src


def _cc_keys_ok(table):
    """Every key is "8.0|*" or a named exception "8.0|<major>.<minor>"; cc 9.0 has NO row (its values are the capability-free defaults, unchanged)."""
    assert "8.0|*" in table, sorted(table)
    for key in table:
        cc, _, mm = key.partition("|")
        assert cc == "8.0" and (mm == "*" or all(part.isdigit() for part in mm.split("."))), key


def _warps_pairs_ok(pairs, widest):
    """((max block, num_warps), ...): ascending max block, the last one covering the kernel's widest tile, num_warps a power of two <= 16."""
    assert isinstance(pairs, tuple) and pairs and all(isinstance(p, tuple) and len(p) == 2 for p in pairs), pairs
    blocks = [b for b, _ in pairs]
    assert blocks == sorted(blocks) and blocks[-1] >= widest, pairs
    assert all(w in (1, 2, 4, 8, 16) for _, w in pairs), pairs


def test_dtk_row_kernels_cc_rows():
    """dtk_kernels.py: _ROW_WARPS is the capability-free rule (ln_modulate / gate_residual: 4 warps up to BC 1024, 8 above; swiglu: 4); _ROW_WARPS_BY_CC
    carries the cc-8.0 row(s) only, per kernel (max block, num_warps) pairs; the three row-kernel launches read row_warps(<kernel>, <tile>, <device>)."""
    t, src = _module_tables("dtk_kernels.py", ("_ROW_WARPS", "_ROW_WARPS_BY_CC"))
    assert t["_ROW_WARPS"] == {"ln_modulate": ((1024, 4), (1 << 30, 8)), "swiglu": ((1 << 30, 4),), "gate_residual": ((1024, 4), (1 << 30, 8))}
    _cc_keys_ok(t["_ROW_WARPS_BY_CC"])
    for key, row in t["_ROW_WARPS_BY_CC"].items():
        assert set(row) == {"ln_modulate", "swiglu", "gate_residual"}, (key, sorted(row))
        for pairs in row.values():
            _warps_pairs_ok(pairs, 1 << 20)                                           # row tiles are next_pow2(C): open-ended
    for call in ('row_warps("ln_modulate", BC, x2.device)', 'row_warps("swiglu", BH, ab.device)', 'row_warps("gate_residual", BC, x.device)'):
        assert call in src, call
    import re
    assert 'nw = row_warps("ln_modulate", BC, x2.device)' in src
    for kern, arg in (("_ln_mod_kernel", "num_warps=nw)"), ("_swiglu_kernel", 'num_warps=row_warps("swiglu", BH, ab.device))'),
                      ("_gate_res_kernel", 'num_warps=row_warps("gate_residual", BC, x.device))')):
        launch = src[src.index(kern + "[("):]
        launch = launch[:launch.index(")\n") + 2]
        assert re.search(r"num_warps=\d", launch) is None and launch.rstrip().endswith(arg), (kern, launch[-120:])   # no inline literal: the row's value


def test_rfd_layernorm_cc_rows():
    """rfd_layernorm.py: _SETTINGS = the capability-free {min_numel 2^18, warps by BLOCK 1/2/4/8}; _SETTINGS_BY_CC carries the cc-8.0 row(s) with the same
    two fields; the size gate is a per-capability constant (no environment knob), _MIN_NUMEL is only a process pin a caller's serve layer may set."""
    t, src = _module_tables("rfd_layernorm.py", ("_SETTINGS", "_SETTINGS_BY_CC", "_MIN_NUMEL"))
    assert t["_SETTINGS"] == {"min_numel": 1 << 18, "warps": ((64, 1), (128, 2), (512, 4), (1024, 8))}
    assert t["_MIN_NUMEL"] is None
    _cc_keys_ok(t["_SETTINGS_BY_CC"])
    for key, row in t["_SETTINGS_BY_CC"].items():
        assert set(row) == {"min_numel", "warps"}, (key, sorted(row))
        assert isinstance(row["min_numel"], int) and 0 <= row["min_numel"] <= 1 << 24, (key, row["min_numel"])
        _warps_pairs_ok(row["warps"], 1024)                                           # the kernel's widest row is d = 1024
    assert "RFD_TRITON_LN_MIN_NUMEL" not in src and "os.environ.get('RFD_TRITON_LN'" in src   # the lever's on/off switch stays; the size knob is gone
    assert "input.numel() < min_numel(input.device)" in src and "nw = num_warps(BLOCK, input.device)" in src


def test_gather_attn_cc_rows():
    """gather_attn.py: CELLS / CONFIGS are the capability-free (H100-measured) tables; ROWS_BY_CC carries the cc-8.0 row(s): "cells" inside the kernel's
    envelope with a status word and "configs" = launch geometry (KC, BQ, num_warps) per head-dim class; supported() and the launch read the device's row
    (cell_status(..., q.device) / launch_config(dh, q.device))."""
    t, src = _module_tables("gather_attn.py", ("CELLS", "CONFIGS", "ROWS_BY_CC", "KERNEL_VERSION"))
    assert t["CELLS"] == {(32, 128, "bfloat16"): "certified", (48, 32, "bfloat16"): "certified"}
    assert t["CONFIGS"] == {32: (8, 16, 4), 64: (4, 16, 4), 128: (4, 16, 4)}
    assert t["KERNEL_VERSION"] == "gather_attn/1.0"
    _cc_keys_ok(t["ROWS_BY_CC"])
    for key, row in t["ROWS_BY_CC"].items():
        assert set(row) == {"cells", "configs"} and row["cells"] and row["configs"], (key, sorted(row))
        for (dh, k, dtype), status in row["cells"].items():
            assert 1 <= dh <= 128 and k >= 1 and dtype in ("bfloat16", "float16") and status in ("certified", "candidate"), (key, dh, k, dtype, status)
        assert set(row["configs"]) <= {32, 64, 128}, (key, sorted(row["configs"]))
        for cls, (kc, bq, nw) in row["configs"].items():
            assert kc >= 1 and kc & (kc - 1) == 0 and bq >= 16 and bq & (bq - 1) == 0 and nw in (1, 2, 4, 8), (key, cls, (kc, bq, nw))
        for (dh, k, _dtype) in row["cells"]:                                             # a certified cell's head-dim class has its geometry in the row
            assert (32 if dh <= 32 else 64 if dh <= 64 else 128) in row["configs"], (key, dh)
    assert t["ROWS_BY_CC"]["8.0|*"]["configs"] == t["CONFIGS"]                            # cc 8.0: the geometry whose output equals CONFIGS' there
    assert 'cell_status(dh, idx.shape[2], v.dtype, q.device)' in src and "KC, BQ, NW = launch_config(dh, q.device)" in src
    assert "KC, BQ, NW = CONFIGS[" not in src
