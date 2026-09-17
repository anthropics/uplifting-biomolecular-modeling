"""0.5.212.4: under the EXACT word kernels.trimul.select() never serves ANOTHER column's stock row.
The two C256/H256 cells without a column for the protenix_v2 image stack (N<=1400 out / in) read the library-less reference column whose
exact value is `torch_math`; on a library stack the engine's stock op is the cuEquivariance op (`cueq`) -> byte identity was lost at
1201..1400 tokens.  CPU only: pure table resolution."""
import re
from opt_core.kernels import trimul as T

PTX2 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"          # protenix_v2 / rosettafold3 / boltzgen image stack word
NOCUEQ = "H100:2.13.0+cu130/3.7.1/nocueq"            # the N<=1400 cells' reference column (library-less)


def _sel(N, direction="outgoing", **kw):
    return T.select(9.0, "bf16", 256, 256, N, direction, word="exact", **kw)


def test_unlisted_library_column_serves_this_process_stock_op():
    for d in ("outgoing", "incoming"):
        for N in (1201, 1300, 1340, 1400):
            s = _sel(N, d, stack=PTX2)
            assert s.row == "cueq" and s.cls == "stock", (d, N, s.row, s.cls, s.reason)
            assert "exact_stock_row_of_this_stack(torch_math->cueq)" in s.reason, s.reason
            assert s.cell.endswith("N<=1400|%s|fwd" % ("out" if d == "outgoing" else "in")), s.cell


def test_neighbour_buckets_unchanged_bitwise_rows():
    for d in ("outgoing", "incoming"):
        for N in (101, 448, 995, 1200, 1401, 1536, 2048):
            s = _sel(N, d, stack=PTX2)
            assert s.row == "native_exact" and str(s.cls).startswith("bitwise"), (d, N, s.row, s.cls)
            assert "exact_stock_row_of_this_stack" not in s.reason


def test_library_less_process_keeps_torch_math():
    assert _sel(1340, stack=NOCUEQ).row == "torch_math"                      # a nocueq stack word, has_cueq unstated: torch_math IS that process's stock op
    assert _sel(1340, stack=PTX2, has_cueq=False).row == "torch_math"        # the kit states no library
    assert _sel(1340, "incoming", stack=NOCUEQ, has_cueq=False).row == "torch_math"


def test_c_hidden_ne_c_z_keeps_torch_math():
    s = T.select(9.0, "bf16", 64, 128, 300, "outgoing", word="exact", stack=PTX2)
    assert s.row == "torch_math", (s.row, s.reason)                          # the library op does not take c_hidden != c_z: the module math is the stock op


def test_planning_call_without_stack_keeps_the_column_word():
    assert _sel(1340).row == "torch_math"                                    # no stack word, has_cueq unstated: the reference column's word, as before
    assert _sel(1340, has_cueq=True).row == "cueq"                           # has_cueq stated: this process's library op


def test_no_library_stack_word_answers_torch_math_under_exact_for_square_classes():
    """Table-wide: for every 9.0 bf16 cell with c_hidden == c_z, every LIBRARY stack word (table columns + the kits' library images) answers a
    bitwise-vouched exact-class row or `cueq` under the exact word -- never `torch_math` (another column's stock row)."""
    tab = T.table()
    key = re.compile(r"^9\.0\|bf16\|C(\d+)\|H(\d+)\|N<=(\d+)\|(out|in)\|fwd$")
    words = set()
    for cell in tab["cells"].values():
        for fld in ("ms", "fast_per_stack", "exact_per_stack", "stock_row"):
            v = cell.get(fld) or {}
            if isinstance(v, dict):
                for k, vv in v.items():
                    words.update(vv.keys() if isinstance(vv, dict) else [k])
    words = {w for w in words if ":" in w and "/cueq" in w} | {PTX2, "H100:2.12.0+cu130/3.7.0/cueq0.10.0", "H100:2.10.0+cu128/3.6.0/cueq0.10.0",
                                                              "H100:2.7.1+cu126/3.3.1/cueq0.8.0", "H100:2.7.1+cu126/3.3.1/cueq0.10.0", "H100:2.7.1+cu128/3.3.1/cueq0.10.0", "A100:2.7.1+cu126/3.3.1/cueq0.10.0"}
    n = 0
    for k in tab["cells"]:
        m = key.match(k)
        if not m or m.group(1) != m.group(2):
            continue
        C, N, d = int(m.group(1)), int(m.group(3)), m.group(4)
        for st in sorted(words):
            for n_tok in (N, max(1, N - 1)):
                s = T.select(9.0, "bf16", C, C, n_tok, "outgoing" if d == "out" else "incoming", word="exact", stack=st)
                n += 1
                assert s.row != "torch_math", (k, st, n_tok, s.row, s.reason)
                assert s.row == "cueq" or str(s.cls).startswith("bitwise"), (k, st, n_tok, s.row, s.cls)
    assert n > 1000, n


# ---- fix7 (protenix_v1): at or under the library's torch-path threshold the exact word on a library stack answers the library op itself
PTX1 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"


def test_exact_at_or_under_the_library_threshold_is_the_library_op_for_every_precision():
    for prec in ("tf32", "f32z_bf16", "fp32", "bf16"):
        for n in (2, 12, 20, 61, 100):
            for d in ("outgoing", "incoming"):
                s = T.select(9.0, prec, 128, 128, n, d, word="exact", stack=PTX1, has_cueq=True)
                assert s.row == "cueq", (prec, n, d, s.row, s.reason)          # protenix_v1's confidence head: tf32 N<=100 was tmk3_exact (vouched vs the KERNEL, not the torch path stock runs there)


def test_above_the_threshold_the_exact_winner_is_unchanged():
    assert T.select(9.0, "tf32", 128, 128, 101, "outgoing", word="exact", stack=PTX1, has_cueq=True).row == "tmk3_exact"      # byte-equal to the library kernel at 101..256 (identity rows bitwise there)
    assert T.select(9.0, "tf32", 128, 128, 250, "incoming", word="exact", stack=PTX1, has_cueq=True).row == "tmk3_exact"
    assert T.select(9.0, "bf16", 128, 128, 103, "outgoing", word="exact", stack=PTX1, has_cueq=True).row == "native_exact"
    assert T.select(9.0, "bf16", 256, 256, 1340, "outgoing", word="exact", stack=PTX1, has_cueq=True).row == "cueq"          # fix7 (protenix_v2) still holds


def test_threshold_follows_the_library_environment_word(monkeypatch):
    monkeypatch.setenv("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "0")                 # a process that disables the library's torch path keeps the table's exact winner
    assert T.select(9.0, "tf32", 128, 128, 20, "outgoing", word="exact", stack=PTX1, has_cueq=True).row == "tmk3_exact"
    monkeypatch.setenv("CUEQ_TRIMUL_FALLBACK_THRESHOLD", "256")
    assert T.select(9.0, "tf32", 128, 128, 250, "outgoing", word="exact", stack=PTX1, has_cueq=True).row == "cueq"


def test_library_less_stacks_and_widths_the_library_does_not_take_are_untouched():
    nocq = "H100:2.13.0+cu130/3.7.1/nocueq"
    assert T.select(9.0, "bf16", 128, 128, 20, "outgoing", word="exact", stack=nocq, has_cueq=False).row != "cueq"
    assert T.select(9.0, "bf16", 64, 128, 20, "outgoing", word="exact", stack=PTX1, has_cueq=True).row != "cueq"       # c_hidden != c_z: no library op; the module math is stock there
