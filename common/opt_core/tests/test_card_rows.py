"""Card rows: the compute-capability-keyed tables of the pair-stack levers carry rows for cc 8.0 (A100), 10.0 (B200) and 10.3 (B300)
beside the H100 rows, and EVERYTHING cc 9.0 (H100, H200) resolves to is pinned: the literal values below are the cc 9.0 content of the
tables; the 8.0 / 10.0 / 10.3 rows sit beside them. Pure python (no torch): the tables are read as data.

  attn/pair_fused_cells.json     rows keyed (impl, piece, key, cc, triton): 9.0 rows pinned field by field; 8.0 / 10.0 / 10.3 carry certified
                                 rows for the c_z=128 trunk shapes (prologue/epilogue (128,4,32), transition (128,512)).
  kernels/flash_triattn.py       _CONFIG_TABLE_BY_CC: cc 9.0 has NO row on any triton (the capability-free tables serve it); 8.0 / 10.0 / 10.3
                                 have a `<cc>|*` row.
  kernels/safe_settings.py       SAFE_ROWS: the 9.0 safe settings pinned; every pair_fused lever has an 8.0, 10.0 and 10.3 entry.
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, os.pardir, "opt_core")

PAIR_FUSED_CC90 = {'r01': ['prologue',
         [256, 8, 32],
         '3.7',
         'mkpf_f1',
         'certified',
         {'BI': 8,
          'BJ': 16,
          'num_warps': 8,
          'num_stages': 2,
          'BN': 64,
          'NUM_STAGES_W': 3,
          'WS': 0,
          'GROUP_I': 0,
          'ln_arith': 'welford',
          'fma_flags': [1, 1, 1]}],
 'r02': ['prologue',
         [256, 8, 32],
         '3.3',
         'mkpf_f1',
         'candidate',
         {'BI': 8,
          'BJ': 16,
          'num_warps': 8,
          'num_stages': 2,
          'BN': 64,
          'NUM_STAGES_W': 3,
          'WS': 0,
          'GROUP_I': 0,
          'ln_arith': 'fused',
          'fma_flags': [1, 1, 1]}],
 'r03': ['prologue',
         [256, 8, 32],
         '3.7',
         'prologue_v4',
         'certified',
         {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2, 'BN': 64, 'NUM_STAGES_W': 3, 'WS': 1, 'GLOOP': 1}],
 'r04': ['prologue',
         [256, 8, 32],
         '3.3',
         'prologue_v4',
         'certified',
         {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2, 'BN': 64, 'NUM_STAGES_W': 3, 'WS': 0, 'GLOOP': 1}],
 'r05': ['prologue', [256, 8, 32], '*', 'v3', 'certified', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r06': ['prologue', [64, 2, 32], '*', 'v3', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r07': ['prologue', [128, 4, 32], '*', 'v3', 'certified', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r08': ['prologue', [256, 4, 64], '*', 'v3', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r09': ['prologue', [64, 4, 16], '*', 'v3', 'certified', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r10': ['prologue', [64, 4, 64], '*', 'v3', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r11': ['prologue', [64, 4, 32], '*', 'v3', 'certified', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r12': ['prologue', [128, 8, 32], '*', 'v3', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r13': ['epilogue',
         [256, 8, 32],
         '3.7',
         'epilogue_v3',
         'certified',
         {'BI': 16, 'BJ': 8, 'num_warps': 8, 'num_stages': 1, 'KC': 64, 'NUM_STAGES_K': 3, 'WS': 0}],
 'r14': ['epilogue',
         [256, 8, 32],
         '3.3',
         'epilogue_v3',
         'certified',
         {'BI': 16, 'BJ': 8, 'num_warps': 8, 'num_stages': 1, 'KC': 64, 'NUM_STAGES_K': 3, 'WS': 0}],
 'r15': ['epilogue', [256, 8, 32], '*', 'v2', 'certified', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 8, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r16': ['epilogue', [128, 4, 32], '*', 'v2', 'certified', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r17': ['epilogue', [64, 2, 32], '*', 'v2', 'candidate', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r18': ['epilogue', [64, 4, 16], '*', 'v2', 'certified', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r19': ['epilogue', [256, 4, 64], '*', 'v2', 'candidate', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 8, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r20': ['epilogue', [64, 4, 64], '*', 'v2', 'candidate', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r21': ['epilogue', [64, 4, 32], '*', 'v2', 'certified', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r22': ['epilogue', [128, 8, 32], '*', 'v2', 'candidate', {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 8, 'num_stages': 1, 'EXP': 'libdevice'}],
 'r23': ['transition', [256, 1024], '*', 'v1', 'certified', {'BM': 128, 'BH': 32, 'num_warps': 8, 'num_stages': 3, 'IL': 0}],
 'r24': ['transition', [128, 512], '*', 'v1', 'certified', {'BM': 128, 'BH': 64, 'num_warps': 8, 'num_stages': 2, 'IL': 0}],
 'r25': ['transition', [384, 1536], '*', 'v1', 'certified', {'BM': 16, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 0}],
 'r26': ['transition', [64, 128], '*', 'v1', 'certified', {'BM': 128, 'BH': 128, 'num_warps': 8, 'num_stages': 1, 'IL': 0}],
 'r27': ['transition', [64, 256], '*', 'v1', 'candidate', {'BM': 128, 'BH': 128, 'num_warps': 8, 'num_stages': 1, 'IL': 0}],
 'r28': ['transition', [128, 256], '*', 'v1', 'certified', {'BM': 128, 'BH': 64, 'num_warps': 8, 'num_stages': 2, 'IL': 0}],
 'r29': ['transition', [256, 512], '*', 'v1', 'certified', {'BM': 128, 'BH': 32, 'num_warps': 8, 'num_stages': 3, 'IL': 0}],
 'r30': ['ln_linear', [64], '*', 'ln_linear', 'certified', {}],
 'r31': ['fused_transition', [64, 256], '*', 'fused_transition', 'certified', {}],
 'r32': ['ln_linear', [128], '*', 'ln_linear', 'certified', {}],
 'r33': ['fused_transition', [128, 512], '*', 'fused_transition', 'certified', {}],
 'r34': ['fused_transition', [64, 128], '*', 'fused_transition', 'certified', {}],
 'r35': ['gate_transpose', [128], '*', 'gate_transpose', 'certified', {}],
 'r36': ['gate_transpose', [64], '*', 'gate_transpose', 'candidate', {}],
 'r37': ['gate_transpose', [256], '*', 'gate_transpose', 'candidate', {}],
 'r38': ['transition', [384, 768], '*', 'v1', 'candidate', {'BM': 16, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 0}],
 'r39': ['prologue', [256, 8, 32], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r40': ['prologue', [64, 2, 32], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r41': ['prologue', [128, 4, 32], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r42': ['prologue', [256, 4, 64], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r43': ['prologue', [64, 4, 16], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r44': ['prologue', [64, 4, 64], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r45': ['prologue', [64, 4, 32], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 4, 'num_stages': 2}],
 'r46': ['prologue', [128, 8, 32], '*', 'v3_fp32z', 'candidate', {'BI': 8, 'BJ': 16, 'num_warps': 8, 'num_stages': 2}],
 'r47': ['transition', [256, 1024], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 32, 'num_warps': 8, 'num_stages': 3, 'IL': 0}],
 'r48': ['transition', [128, 512], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 64, 'num_warps': 8, 'num_stages': 2, 'IL': 0}],
 'r49': ['transition', [384, 1536], '*', 'v1_fp32x', 'candidate', {'BM': 16, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 0}],
 'r50': ['transition', [64, 128], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 128, 'num_warps': 8, 'num_stages': 1, 'IL': 0}],
 'r51': ['transition', [64, 256], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 128, 'num_warps': 8, 'num_stages': 1, 'IL': 0}],
 'r52': ['transition', [128, 256], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 64, 'num_warps': 8, 'num_stages': 2, 'IL': 0}],
 'r53': ['transition', [256, 512], '*', 'v1_fp32x', 'candidate', {'BM': 128, 'BH': 32, 'num_warps': 8, 'num_stages': 3, 'IL': 0}],
 'r54': ['transition', [384, 768], '*', 'v1_fp32x', 'candidate', {'BM': 16, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 0}],
 'r88': ['transition', [128, 512], '*', 'v2_fold', 'certified', {'BM': 64, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 1}],
 'r89': ['transition', [64, 128], '*', 'v2_fold', 'certified', {'BM': 64, 'BH': 32, 'num_warps': 4, 'num_stages': 3, 'IL': 0}],
 'r90': ['transition', [64, 256], '*', 'v2_fold', 'certified', {'BM': 64, 'BH': 64, 'num_warps': 4, 'num_stages': 2, 'IL': 0}],
 'r101': ['prologue', [256, 8, 32], '3.7', 'v3', 'candidate', {'BI': 16, 'BJ': 16, 'num_warps': 8, 'num_stages': 1}]}   # a donor kit's former triton-3.7 sm90 v3-prologue overlay cell, lifted with provenance; re-measured later

SAFE_ROWS_CC90 = {'pair_fused:transition': {'BM': 16, 'BH': 32, 'num_warps': 4, 'num_stages': 1, 'IL': 0},
 'pair_fused:trimul': {'k1': {'BM': 128, 'BN': 128, 'num_warps': 8, 'num_stages': 1}, 'k3': {'BM': 64, 'BN': 64, 'num_warps': 4, 'num_stages': 1}},
 'pair_fused:prologue': {'BI': 8, 'BJ': 8, 'num_warps': 4, 'num_stages': 2},
 'pair_fused:epilogue': {'KVER': 2, 'BI': 16, 'BJ': 8, 'num_warps': 4, 'num_stages': 1, 'EXP': 'libdevice'}}

TRUNK_KEYS = {"prologue": [128, 4, 32], "epilogue": [128, 4, 32], "transition": [128, 512]}


def _pair_fused_rows():
    with open(os.path.join(CORE, "attn", "pair_fused_cells.json"), encoding="utf-8") as f:
        return json.load(f)["rows"]


def test_pair_fused_cc90_rows_are_unchanged():
    rows = {r["id"]: r for r in _pair_fused_rows() if r["cc"] == "9.0"}
    assert sorted(rows) == sorted(PAIR_FUSED_CC90), sorted(set(rows) ^ set(PAIR_FUSED_CC90))
    for rid, (piece, key, triton, variant, status, cfg) in PAIR_FUSED_CC90.items():
        r = rows[rid]
        assert [r["piece"], r["key"], r["triton"], r["variant"], r["status"], r["cfg"]] == [piece, key, triton, variant, status, cfg], rid


def test_pair_fused_resolution_on_cc90_is_unchanged():
    """Through the core's one resolver: every (piece, key, variant) certified at 9.0 resolves on (9,0) x every triton to the pinned cfg."""
    from opt_core.kernels import safe_settings as S
    rows = _pair_fused_rows()
    for rid, (piece, key, triton, variant, status, cfg) in PAIR_FUSED_CC90.items():
        table = {"%s|%s" % (r["cc"], r["triton"]): r["cfg"] for r in rows
                 if r["piece"] == piece and r["key"] == key and r["variant"] == variant and r["status"] == status}
        want_mm = triton if triton != "*" else "3.6"
        k, got = S.resolve_key(table, (9, 0), want_mm), S.resolve_row(table, (9, 0), want_mm)
        assert got == cfg and k == "9.0|%s" % triton, (rid, k, got)


def test_pair_fused_has_certified_trunk_rows_on_every_card():
    rows = _pair_fused_rows()
    for cc in ("8.0", "9.0", "10.0", "10.3"):
        for piece, key in TRUNK_KEYS.items():
            hit = [r for r in rows if r["impl"] == "fpf" and r["piece"] == piece and r["key"] == key and r["cc"] == cc and r["status"] == "certified"]
            if (cc, piece) == ("8.0", "epilogue"):                                                   # the fused epilogue measured slower than the lnl statements on 8.0:
                assert not hit and [r for r in rows if r["impl"] == "lnl" and r["piece"] == "gate_transpose" and r["cc"] == cc and r["status"] == "certified"], (cc, piece, key)
                continue                                                                              # named off, the lnl gate_transpose row serves the piece
            assert hit, (cc, piece, key)
    ids = [r["id"] for r in rows]
    assert len(ids) == len(set(ids)) and all(i == "r%02d" % (n + 1) for n, i in enumerate(ids)), ids      # r01.. consecutive, unique


def test_flash_triattn_cc_rows_by_ast():
    src = open(os.path.join(CORE, "kernels", "flash_triattn.py"), encoding="utf-8").read()
    mod = ast.parse(src)
    node = next(n for n in mod.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "_CONFIG_TABLE_BY_CC")
    keys = [k.value for k in node.value.keys]
    assert not any(k.startswith("9.0") for k in keys), keys                                  # H100 / H200: the capability-free tables, unchanged
    for cc in ("8.0", "10.0", "10.3"):
        assert "%s|*" % cc in keys, (cc, keys)


def test_safe_rows_cc90_unchanged_and_every_pair_fused_lever_has_a_row_per_card():
    from opt_core.kernels import safe_settings as S
    for lever, settings in SAFE_ROWS_CC90.items():
        assert S.SAFE_ROWS[lever]["9.0"]["settings"] == settings, lever
    assert {lever for lever, rows in S.SAFE_ROWS.items() if "9.0" in rows} == set(SAFE_ROWS_CC90)
    for lever in ("pair_fused:transition", "pair_fused:trimul", "pair_fused:prologue", "pair_fused:epilogue"):
        for cc in ("8.0", "9.0", "10.0", "10.3"):
            row = S.SAFE_ROWS[lever].get(cc)
            assert row and row["status"].startswith("SAFE") and row["settings"], (lever, cc)
            assert S.safe_settings_for(lever, cc, None)[0] is not None or S.safe_settings_for(lever, cc, None)[1], (lever, cc)


# ----------------------------------------------------------------------------------------------------------------- fpf_triatt_k2b K2B_CELLS.json
K2B_CELLS = os.path.join(CORE, "kernels", "fpf_triatt_k2b", "K2B_CELLS.json")
K2B_BY_CC = {"8.0": "sm80", "9.0": "sm90", "10.0": "sm100", "10.3": "sm103", "12.0": "sm120"}
K2B_PINNED_BF16 = {"sm100": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "MAXNREG": 128, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}, "sm120": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}, "sm80": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "MAXNREG": 128, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}, "sm90": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "MAXNREG": 128, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}}      # the sm80 / sm90 / sm100 / sm120 cells as they stand (A100 / H100+H200 / B200 / RTX PRO 6000)
K2B_PINNED_STATUS_PREFIX = {"sm100": "checked: NVIDIA B200 (cc 10.0), triton 3.6.0 and", "sm120": "checked: NVIDIA RTX PRO 6000 Blackwell Server Ed", "sm80": "verified: NVIDIA A100 80GB PCIe (cc 8.0), torch ", "sm90": "checked: NVIDIA H100 80GB HBM3 (cc 9.0), torch 2"}
K2B_V1_KEYS = {"B200": {"bf16": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "MAXNREG": 128, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}}, "H100": {"bf16": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "MAXNREG": 128, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}}, "RTX PRO 6000": {"bf16": {"32": [[1073741824, {"BLOCK_M": 64, "BLOCK_N": 32, "ORDER": 0, "ROWS": 2, "num_stages": 3, "num_warps": 4}]]}}}           # the v1 name-keyed cells an unpatched loader reads
K2B_SM103_CELL = {"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "num_stages": 2, "ORDER": 0, "MAXNREG": 128}


def test_k2b_cells_every_capability_resolves_to_a_checked_entry_and_the_other_cards_are_unchanged():
    """K2B_CELLS.json (k2b_cells/v2, read by fpf_triatt_k2b.triatt_k2b._pf_resolve_cells: by_cc first, by_name second, v1 name keys for an unpatched loader):
    cc 10.3 (B300) resolves to the `sm103` entry — a checked cell, no UNTESTED word anywhere a device can reach — and every other capability's entry,
    status and cell is byte-for-byte what it was (H100 / H200 / A100 / B200 / RTX PRO 6000 resolve to unchanged values)."""
    tab = json.load(open(K2B_CELLS))
    assert dict(tab["by_cc"]) == K2B_BY_CC
    assert tab["by_name"] == {"B200": "sm100", "B300": "sm103", "RTX PRO 6000": "sm120", "H100": "sm90", "H200": "sm90", "A100": "sm80"}
    for key in set(tab["by_cc"].values()) | set(tab["by_name"].values()):
        ent = tab["entries"][key]
        assert "UNTESTED" not in key and not ent["status"].upper().startswith("UNTESTED"), (key, ent["status"][:60])
        assert ent["device_names_seen"], key                                             # a checked entry names the device it was checked on
    for key, cells in K2B_PINNED_BF16.items():
        assert tab["entries"][key]["bf16"] == cells, key
        assert tab["entries"][key]["status"].startswith(K2B_PINNED_STATUS_PREFIX[key]), key
    for name, cells in K2B_V1_KEYS.items():
        assert tab[name] == cells, name
    e103 = tab["entries"]["sm103"]
    assert e103["status"].startswith("checked: NVIDIA B300") and "torch.equal" in e103["status"] and e103["device_names_seen"] == ["NVIDIA B300 SXM6 AC"]
    (max_seq, cell), = e103["bf16"]["32"]
    assert max_seq == 1 << 30 and cell == K2B_SM103_CELL
    assert tab["B300"] == {"bf16": {"32": [[1 << 30, K2B_SM103_CELL]]}}                 # the v1 name key carries the same cell
    assert {k: v for k, v in cell.items() if k != "num_stages"} == {k: v for k, v in tab["entries"]["sm100"]["bf16"]["32"][0][1].items() if k != "num_stages"}   # B300 = the B200 tile at 2 stages
