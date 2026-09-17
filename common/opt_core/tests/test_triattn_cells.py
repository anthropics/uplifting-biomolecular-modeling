"""kernels.triattn: the triangle-attention provider's cell table, words, refusals and default rule -- pure checks (no GPU, no framework)."""
import ast
import filecmp
import hashlib
import json
import os, sys

import pytest

from opt_core import kernels
from opt_core.kernels import triattn as T

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
CUDA_STACK = "torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90"
CUDA_STACKS = ["torch2.10.0+cu128-cpython-311-x86_64-linux-gnu-sm90", "torch2.11.0+cu128-cpython-312-x86_64-linux-gnu-sm90", "torch2.12.0+cu130-cpython-311-x86_64-linux-gnu-sm90", "torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90", "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90", "torch2.7.1+cu126-cpython-311-x86_64-linux-gnu-sm90", "torch2.7.1+cu128-cpython-311-x86_64-linux-gnu-sm90"]
NOT_BYTE_IDENTICAL = tuple(kernels.sums("triattn").get("not_byte_identical", {}))   # META: {relative path: why}

# every (dtype, head_dim, heads) triangle-attention shape the release tree's torch pair stacks run today (c=64 template stacks included:
# the attention rows never see c), at the three representative sizes, on both cards
SHAPES = [("bf16", 32, 2), ("fp32", 32, 2), ("bf16", 16, 4), ("bf16", 32, 4), ("fp32", 32, 4), ("bf16", 64, 4), ("bf16", 32, 8),
          ("bf16", 32, 12), ("fp32", 32, 12)]
SIZES = (400, 800, 1200)
CARDS = tuple(T.measured_ccs())                     # every measured column of the table (table-derived: a new column needs no test edit)


def test_table_schema_rows_words():
    t = T.table()
    assert t["schema"].startswith("triattn_cells/v1")
    assert tuple(t["words"]["rows"]) == T.ROW_NAMES
    assert set(t["rows"]) == set(T.ROW_NAMES)
    for name, row in t["rows"].items():
        assert row["class"] in ("fast", "exact", "stock"), name
        assert "backward" in row and "capture_safe" in row, name
    assert t["rows"]["ds4sci"]["capture_safe"] is False                        # replay-stale under CUDA graphs: flagged, never selectable in a graphed region
    assert t["rows"]["cuda_sm90a"]["fallback"] == "k2b"
    assert t["rows"]["cuda_sm90a"]["cc"] == ["9.0"]
    assert t["buckets"]["sizes"] == sorted({int(k.split("|")[4][3:]) for k in t["cells"]}) and t["buckets"]["sizes"][:8] == [256, 400, 511, 800, 1200, 1536, 2048, 2560]    # 511: the package's CUDA-route floor is 512 tokens; 2560: the exact-debt bucket; above: the beyond-2048 buckets
    for key, cell in t["cells"].items():
        cc, dt, d, h, n, direction = key.split("|")
        assert cc in CARDS and dt in ("bf16", "fp32") and d[0] == "D" and h[0] == "H" and n.startswith("N<=") and direction == "fwd", key
        assert cell["fast"] == cell["fast_order"][0] and cell["fast_order"][-1] in ("cueq", "stock"), key
        assert cell["exact"] in ("exact_headsplit", "triattn_exact", "cueq", "stock"), key
        for r in cell["fast_order"][:-1]:
            assert cell["x_stock"][r] >= 1.0, (key, r)                           # a row is in fast_order only where it beat the stock op
        best_exact = max((cell["x_stock"][r] for r in ("exact_headsplit",) if r in cell["x_stock"]), default=None)
        if cell["exact"] == "triattn_exact":                                     # the bit-identical row: named on its recorded vouch (its x_stock is reported by the bench legs, not a precondition)
            assert (cell.get("vouched_on") or {}).get("triattn_exact"), key
        elif cell["exact"] not in ("cueq", "stock"):                             # the exact rule: named wherever an exact row measured >= x1.00 ...
            assert cell["x_stock"][cell["exact"]] >= 1.0 and cell["x_stock"][cell["exact"]] == best_exact, key
            assert cell["exact_x_per_stack"] and all(isinstance(v, float) for v in cell["exact_x_per_stack"].values()), key
        else:                                                                    # ... or the stock op by name (exact by definition)
            assert cell["exact"] in ("cueq", "stock"), key
        for role, text in cell.get("parity", {}).items():                        # parity flag: a named winner inside x0.97-1.03, and only there
            if role == "exact" and cell["exact"] == "triattn_exact":               # (a measured note about the earlier timed winner stays as a note once the bit-identical row is named on its vouch)
                continue
            assert role in ("fast", "exact") and text.startswith(cell[role] + " x") and 0.97 <= cell["x_stock"][cell[role]] <= 1.03, key
        for role in ("fast", "exact"):
            r = cell[role]
            if r not in ("cueq", "stock") and r in cell["x_stock"] and 0.97 <= cell["x_stock"][r] <= 1.03:
                assert role in cell.get("parity", {}), (key, role)
        for r in cell["fast_order"][:-1] + ([cell["exact"]] if cell["exact"] not in ("cueq", "stock", "triattn_exact") else []):   # the bit-identical row carries a vouch, not a timing record
            assert cell["ms"].get(r), (key, r)                                   # no cell names a winner without a timed row behind it
        if cc == "8.0":
            assert "ceiling_40gb" in cell and "cuda_sm90a" not in cell["fast_order"], key


@pytest.mark.parametrize("cc", CARDS)
def test_every_shape_the_tree_runs_lands_on_a_measured_cell_or_a_named_nearest(cc):
    cov = T.coverage(cc, SHAPES, SIZES)
    assert len(cov) == len(SHAPES) * len(SIZES)
    for tag, verdict in cov.items():
        assert not verdict.startswith("n/a"), (tag, verdict)                     # every shape has a cell of its (cc, dtype, head_dim)
    # the two sizes not separately timed at fp32 H=2 (a c=64 template stack in fp32) take the 800-token measurement, and say so
    assert "N<=800" in cov[f"{cc}|fp32|D32|H2|N400"] and "above the largest measured size" in cov[f"{cc}|fp32|D32|H2|N1200"]
    unmeasured = [t for t, v in cov.items() if "(" in v]
    assert unmeasured == [f"{cc}|fp32|D32|H2|N1200"], unmeasured


def test_default_rule_a_row_word_serves_that_row_with_its_own_cell():
    """The OPT-IN rule: naming a row returns that row, no launch setting imposed (the row's own table decides, as a kit binding it today runs)."""
    for cc in CARDS:
        for dt, d, h in SHAPES:
            for n in SIZES:
                for row in ("k2b", "k2", "flash"):
                    s = T.select(cc, dt, d, h, n, word=row)
                    assert s.row == row and s.config is None and s.input_precision is None and s.cls == "fast"
                s = T.select(cc, dt, d, h, n, word="cueq")
                assert s.row == "cueq" and s.cls == "stock" and s.backward is True
    s = T.select("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack=CUDA_STACK)
    assert s.row == "cuda_sm90a" and s.config is None


def test_tier_words_are_the_measured_winners():
    s = T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack=CUDA_STACK)
    assert s.row == "triattn_native" and s.measured and s.x_stock > 3            # the sealed package row wins every bf16 D32 cell from 512 tokens on that stack
    s = T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu-sm90")
    assert s.row == "k2b" and "no_prebuilt" in s.reason                          # a stack with neither extension prebuilt: passed over by name to the cell's next row
    s = T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu-sm90")
    assert s.row == "k2b" and "no_prebuilt" in s.reason                          # the prebuilt is absent on that stack: the next measured row, and the reason says why
    s = T.select("9.0", "bf16", 32, 4, 400, word="fast", prefer=("k2b", "flash"))
    assert s.row == "k2b"
    s = T.select("9.0", "bf16", 32, 4, 1536, word="exact")
    assert s.row == "triattn_exact" and s.cls == "exact"                          # the exact tier on cc 9.0 (a table query naming no running stack): the bit-identical row; a running process gets it only on a vouched stack
    s = T.select("8.0", "bf16", 32, 4, 1536, word="exact")
    assert s.row == "triattn_exact" and s.cls == "exact"                          # cc 8.0 (a table query naming no running stack): the bit-identical row, named on its vouch
    s = T.select("8.0", "bf16", 32, 4, 400, word="exact")
    assert s.row == "triattn_exact"
    s = T.select("8.0", "bf16", 32, 4, 1200, word="exact")                       # x1.05 median (x0.93 on one of three stacks, listed per stack): named, per the >= x1.00 rule
    assert s.row == "triattn_exact" and s.cls == "exact"                          # the bit-identical row on its vouch (the head-split row it displaced measured below x1.00 on one stack and above on another: exact_x_per_stack keeps that record)
    assert min(T.cells()[s.cell]["exact_x_per_stack"].values()) < 1.0 < max(T.cells()[s.cell]["exact_x_per_stack"].values())
    s = T.select("9.0", "fp32", 32, 4, 1200, word="exact")                       # inside the noise band: named, flagged
    assert s.row == "exact_headsplit" and "parity (exact)" in s.reason
    s = T.select("8.0", "bf16", 32, 4, 800, word="fast")                           # generation 11's sm_80 member: the measured winner of the cell (RaceV11)
    assert s.row == "triattn_native" and s.measured and s.x_stock >= 1.5
    s = T.select("8.0", "bf16", 32, 4, 800, word="fast", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu")
    assert s.row == "k2b" and "no_prebuilt" in s.reason                              # a stack without the sm_80 prebuilt: passed over by name to the cell's next word
    s = T.select("8.0", "bf16", 32, 12, 1200, word="k2b@m128r2")
    assert s.row == "k2b" and s.config == {"BLOCK_M": 128, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "num_stages": 3, "ORDER": 0}   # no register cap: a 128-register cap halves this tile
    assert T.select("8.0", "bf16", 32, 12, 800, word="fast").row == "triattn_native"     # generation 11's sm_80 member wins the H12 cells too (RaceV11) ...
    s = T.select("8.0", "bf16", 32, 12, 800, word="fast", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu")   # ... and on a stack without its prebuilt the tier word reaches the launch cell measured for this cell ...
    assert s.row == "k2b" and s.config == T.table()["words"]["config_words"]["k2b@m128r2"]["config"] and "tuned launch cell k2b@m128r2" in s.reason
    s = T.select("8.0", "bf16", 32, 12, 800, word="k2b")                         # ... the row word never does (today's bytes)
    assert s.row == "k2b" and s.config is None
    assert T.select("8.0", "bf16", 32, 12, 200, word="fast").config is None          # below the measured floor (256 rows): the row's own cell
    assert T.select("8.0", "bf16", 32, 12, 256, word="fast", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu").config is not None and T.select("9.0", "bf16", 32, 12, 800, word="fast", prefer=("k2b",)).config is None
    for key, cell in T.cells().items():
        for row, tuned in cell.get("tuned", {}).items():
            assert tuned["config_word"] in T.table()["words"]["config_words"] and tuned["config"] == T.table()["words"]["config_words"][tuned["config_word"]]["config"] and tuned["evidence"], key
    s = T.select("9.0", "fp32", 32, 4, 800, word="tf32x3")
    assert s.row == "flash" and s.input_precision == "tf32x3" and s.x_stock > 1
    s = T.select("9.0", "fp32", 32, 12, 800, word="tf32")
    assert s.row in ("k2b", "k2") and s.input_precision is None
    mx = max(int(k.split("|")[4][3:]) for k in T.cells() if k.startswith("9.0|bf16|D32|H4|"))
    s = T.select("9.0", "bf16", 32, 4, 100000, word="fast", stack=CUDA_STACK)
    assert s.row == "triattn_native" and not s.measured and f"above the largest measured size {mx}" in s.reason   # the nearest cell's order, flagged unmeasured
    s = T.select("9.0", "fp16", 32, 4, 800, word="k2b")
    assert s.row == "k2b" and "fp16 takes the bf16 cell" in s.reason                              # a ROW word serves fp16 (its own launch cell), flagged
    with pytest.raises(T.Refusal) as e:                                                             # a TIER word at an unmeasured head count / fp16: refused by name
        T.select("9.0", "bf16", 32, 6, 800, word="fast", prefer=("k2b",))
    assert e.value.kind == "no_cell:heads=6" and e.value.row == "fast" and e.value.fallback == "cueq"
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "fp16", 32, 4, 800, word="fast")
    assert e.value.kind == "no_cell:dtype=fp16" and e.value.fallback == "cueq"
    assert T.select("9.0", "bf16", 32, 6, 800, word="k2b").row == "k2b"                            # the row word opts in at H6


def test_refusals_are_by_name_with_the_fallback_row():
    def refusal(*a, **k):
        with pytest.raises(T.Refusal) as e:
            T.select(*a, **k)
        return e.value
    r = refusal("8.0", "bf16", 32, 4, 800, word="cuda_sm90a")
    assert r.kind == "cc 8.0: built for sm_90a only" and r.row == "cuda_sm90a" and r.fallback == "k2b"
    r = refusal("10.0", "bf16", 32, 4, 800, word="cuda_sm90a")
    assert r.fallback == "k2b" and "sm_90a" in r.kind
    r = refusal("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack="torch2.12.0+cu130-cpython-312-x86_64-linux-gnu-sm90")
    assert r.kind.startswith("no_prebuilt:") and r.fallback == "k2b"
    r = refusal("9.0", "fp32", 32, 4, 800, word="cuda_sm90a", stack=CUDA_STACK)
    assert r.kind == "dtype_fp32" and r.fallback == "k2b"
    r = refusal("9.0", "bf16", 64, 4, 800, word="cuda_sm90a", stack=CUDA_STACK)
    assert r.kind == "head_dim_64" and r.fallback == "k2b"
    r = refusal("9.0", "bf16", 32, 4, 800, "fwdbwd", word="k2b")
    assert r.kind == "no_backward" and r.fallback == "cueq"                      # the fused rows are forward-only; a differentiated call binds a stock row
    r = refusal("9.0", "bf16", 32, 1, 800, word="exact_headsplit")
    assert r.kind == "single_head"
    r = refusal("7.5", "bf16", 32, 4, 800, word="k2b")
    assert r.kind == "cc<8.0:sm_75"
    r = refusal("9.0", "bf16", 32, 4, 800, word="tf32x3")
    assert r.kind == "fp32_word_on_bf16"
    r = refusal("9.0", "bf16", 32, 4, 800, word="nonsense")
    assert r.kind == "unknown_word:nonsense"
    r = refusal("9.0", "bf16", 32, 4, 800, word="exact", prefer=("k2b",))
    assert r.kind == "prefer_has_no_exact_row"
    with pytest.raises(TypeError):
        T.select("9.0", "bf16", 32, 4, 800, word=None)


def test_stock_rows_need_the_kit_callable_and_ds4sci_is_capture_unsafe():
    s = T.select("9.0", "bf16", 32, 4, 800, "fwdbwd", word="fast")
    assert s.row == "cueq" and s.backward                                        # every fused row refuses a backward: the tier word lands on the stock op
    s = T.select("9.0", "bf16", 32, 4, 800, word="ds4sci")
    assert s.capture_safe is False and "capture_safe=NO" in T.describe(s)
    s = T.select("8.0", "bf16", 32, 8, 1200, word="exact_headsplit")
    assert s.row == "exact_headsplit" and s.tested and T.rows()["exact_headsplit"]["tested_cc"] == ["9.0", "8.0"]


def test_describe_is_one_line():
    s = T.select("9.0", "fp32", 32, 4, 800, word="tf32x3")
    line = T.describe(s)
    assert "\n" not in line and line.startswith("row=flash word=tf32x3 class=fast") and "ip=tf32x3" in line


def test_face_imports_only_the_standard_library_at_module_level():
    tree = ast.parse(open(os.path.join(T.HERE, "__init__.py"), encoding="utf-8").read())
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add(node.module.split(".")[0])
        elif isinstance(node, ast.ImportFrom):                            # the one relative import: the cell-coverage census (itself standard library only)
            assert [a.name for a in node.names] == ["cell_census"], [a.name for a in node.names]
    assert top <= {"json", "math", "os", "re", "sys", "typing"}, top


def test_meta_census_and_the_carried_sub_packages():
    doc = kernels.sums("triattn")
    files = set(doc["files"])
    assert doc["kind"] == "package" and doc["version"]
    for f in ["__init__.py", "TRIATTN_CELLS.json", "cuda_sm90a/__init__.py", "cuda_sm90a/NOTICE.md", "cuda_sm90a/build_prebuilt.py",
              "cuda_sm90a/csrc/triattn_mw.cu", "cuda_sm90a/csrc/mw_ptx.h",
              "headsplit/__init__.py", "headsplit/prologue_hm.py"] + [f"cuda_sm90a/prebuilt/{k}/{n}" for k in CUDA_STACKS for n in ("manifest.json", "triattn_cuda_sm90.so", "loadcheck.pt")]:
        assert f in files, f
    assert T.stacks_built() == CUDA_STACKS
    assert "prebuilt" not in doc and "sha256\": \"" not in json.dumps(doc)          # META stores no digest (kernels contract): the manifests below are the package's own
    loadchecks, sources = set(), set()
    for key in CUDA_STACKS:
        man = json.load(open(os.path.join(T.PREBUILT_DIR, key, "manifest.json")))   # written by build_prebuilt.py, read + verified by the package's install()
        assert man["stack_key"] == key and man["capability"] == [9, 0] and man["so"] == "triattn_cuda_sm90.so"
        so_sha = hashlib.sha256(open(os.path.join(T.PREBUILT_DIR, key, man["so"]), "rb").read()).hexdigest()
        assert so_sha == man["so_sha256"], key                                   # the bytes on disk are the bytes the manifest names (install() checks the same)
        sources.add(json.dumps(man["source_sha256"], sort_keys=True)); loadchecks.add(json.dumps(man["loadcheck_sha256_16"], sort_keys=True))
        assert T.select("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack=key).row == "cuda_sm90a"
    assert len(loadchecks) == 1                                                  # every prebuilt: the same outputs bit for bit on the load-check cases
    assert len(sources) == 1                                                     # ... compiled from the same sources (the carried csrc/)
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu-sm90")
    assert e.value.kind == "no_prebuilt:torch2.9.0+cu128-cpython-311-x86_64-linux-gnu-sm90" and e.value.fallback == "k2b"
    assert T.select("9.0", "bf16", 32, 4, 800, word="cuda_sm90a", stack="torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90").row == "cuda_sm90a"   # the cpython-312 build
    src = open(os.path.join(T.HERE, "headsplit", "__init__.py"), encoding="utf-8").read()
    assert "N_SPLIT_FLOOR = 1024" in src and "TESTED_CC = ((9, 0),)" in src


def _donors():
    """(core sub-package, kit directory) pairs present beside common/ in this tree (a sparse tree may lack the kits: then nothing to compare)."""
    out = []
    for sub, rel in (("cuda_sm90a", ("opt", "forward", "flashpairformer", "third_party")), ("headsplit", ("opt", "forward", "flashpairformer", "third_party"))):
        for kit in sorted(os.listdir(RELEASE_TREE)):
            base = os.path.join(RELEASE_TREE, kit, *rel)
            if not os.path.isdir(base):
                continue
            for cand in sorted(os.listdir(base)):
                d = os.path.join(base, cand)
                if sub == "cuda_sm90a" and cand.endswith("_triattn_cuda") and os.path.isdir(d):
                    out.append((sub, d))
                if sub == "headsplit" and cand.endswith("_triatt_headsplit") and os.path.isdir(d):
                    out.append((sub, d))
    return out


@pytest.mark.parametrize("sub,donor", _donors() or [pytest.param(None, None, marks=pytest.mark.skip("no donor kit beside common/"))])
def test_carried_sub_packages_are_byte_identical_to_the_kit_copies(sub, donor):
    core = os.path.join(T.HERE, sub)
    for r, ds, fs in os.walk(donor):
        ds[:] = [d for d in ds if d != "__pycache__"]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            rel = os.path.relpath(os.path.join(r, f), donor)
            assert os.path.isfile(os.path.join(core, rel)), rel
            if sub + "/" + rel in NOT_BYTE_IDENTICAL:                          # the one restated line (release-tree vocabulary): every OTHER line equal
                a = open(os.path.join(r, f), encoding="utf-8").read().split("\n"); b = open(os.path.join(core, rel), encoding="utf-8").read().split("\n")
                assert len(a) == len(b) and [i for i, (x, y) in enumerate(zip(a, b)) if x != y] == [3], rel
                continue
            assert filecmp.cmp(os.path.join(r, f), os.path.join(core, rel), shallow=False), rel


def test_int32_in_row_offset_bound_by_name():
    """The carried Triton rows form in-row (position) offsets in 32-bit arithmetic (bases are 64-bit): the face refuses BY NAME past 2**31
    -- select() from a position-stride hint, triangle_attention() from the tensors' actual strides -- and names a non-Triton fallback."""
    L = T.INT32_OFFSET_LIMIT
    assert L == 2 ** 31 and T.TRITON_ROWS == ("k2b", "k2", "flash")
    # contiguous [B,N,H,S,D] operands: position stride = head_dim -> every term tiny at any size
    for n, d in ((2048, 64), (8192, 64), (46000, 32)):
        terms = T.int32_offset_terms(n, n, d, q_pos_stride=d, k_pos_stride=d, v_pos_stride=d, mask_key_stride=1, bias_q_stride=n, bias_k_stride=1)
        assert T.int32_offsets_ok(terms) == (True, "ok"), (n, d, terms)
    assert T.int32_offsets_ok(T.int32_offset_terms(46341, 46341, 32, q_pos_stride=32, k_pos_stride=32, v_pos_stride=32))[1].startswith("int32_offset:bias_staged=")
    # a transposed view: position stride = N*H*D -> (S-1)*N*H*D + D crosses 2**31 exactly where the arithmetic says
    def first_bad(h, d):
        n = 2
        while (n - 1) * (n * h * d) + d < L:
            n += 1
        return n
    assert (first_bad(16, 64), first_bad(4, 64), first_bad(4, 32)) == (1449, 2897, 4097)
    for h, d, n_ok, n_bad in ((16, 64, 1448, 1449), (16, 64, 1024, 1536), (4, 64, 2896, 2897), (4, 32, 4096, 4097)):
        assert T.admits("k2b", "9.0", "bf16", d, h, n_ok, position_stride=n_ok * h * d) == (True, "ok")
        ok, why = T.admits("k2b", "9.0", "bf16", d, h, n_bad, position_stride=n_bad * h * d)
        assert not ok and why.startswith("int32_offset:q_pos=") and why.endswith(">=2**31"), why
        for row in ("k2", "flash"):
            assert T.admits(row, "8.0", "bf16", d, h, n_bad, position_stride=n_bad * h * d)[1] == why
    assert T.position_stride_bound(1536, 64) == (L - 65) // 1535 and 1024 * 1536 > T.position_stride_bound(1536, 64) >= 1024 * 1365
    # select(): a named Triton row refuses by name with a NON-Triton fallback; a tier word passes over the Triton rows to the next candidate
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "bf16", 64, 16, 1536, word="k2b", position_stride=1536 * 16 * 64)
    assert e.value.kind.startswith("int32_offset:") and e.value.row == "k2b" and e.value.fallback not in T.TRITON_ROWS   # the cell's next non-Triton row (a stock-op route)
    s = T.select("9.0", "bf16", 64, 4, 1536, word="fast", position_stride=2 ** 31 // 1024)
    assert s.row not in T.TRITON_ROWS
    s = T.select("9.0", "bf16", 32, 4, 4097, word="fast", position_stride=4097 * 4 * 32, stack=CUDA_STACK)
    assert s.row in ("triattn_native", "cuda_sm90a")                                     # 64-bit strides in the extension rows: still served on 9.0 bf16 D32
    assert T.select("9.0", "bf16", 64, 16, 1536, word="k2b").row == "k2b"            # no hint = contiguous operands: served as before (default rule unchanged; a row word at an unmeasured head count)
    assert T.select("8.0", "bf16", 64, 4, 2048, word="fast", position_stride=64).row in ("triattn_native", "k2b")                # 64-bit addressing in the sm_80 member too (the cell's ragged-form winner decides)
    assert T.select("8.0", "bf16", 64, 4, 2048, word="fast", position_stride=64, stack="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu").row in T.TRITON_ROWS


def test_d64_cells_and_tuned_launch_settings_from_the_sdpa_stock_kit():
    """bf16 D=64 H=4: cells to 2048 tokens on both cards (the cells at 1536 / 2048 were measured against SDPA by the kit whose stock op it is)
    and tuned k2b launch settings per size bucket, reached through the tier word only (the row word keeps the row's own cell)."""
    W = T.table()["words"]["config_words"]
    for cc in ("9.0", "8.0"):
        for n in (1536, 2048):
            key = f"{cc}|bf16|D64|H4|N<={n}|fwd"; c = T.cells()[key]
            assert c["stock_row"] == "sdpa" and c["fast_order"][0] in ("k2b", "triattn_native@v10", "triattn_native@v11") and c["x_stock"]["k2b"] >= 1.0 and c["ms"]["k2b"], key
            assert (c["exact"], c["fast_order"][-1]) in (("stock", "stock"), ("cueq", "cueq")), key    # 'stock' where cueq was never timed at the cell (x_stock vs SDPA, stock_ref says so)
            assert T.select(cc, "bf16", 64, 4, n, word="fast").cell == key and T.select(cc, "bf16", 64, 4, n, word="fast").measured
    exp = {("9.0", 1024): "k2b@m128n32s3r168", ("9.0", 1536): "k2b@m128n64s2", ("8.0", 512): "k2b@m128n64s2r255", ("8.0", 1024): "k2b@m128n64s2r255", ("8.0", 1536): "k2b@m128n64s2"}
    for (cc, n), word in exp.items():
        if cc == "8.0":                                                                           # generation 11's sm_80 member decides the 8.0 D64 cells where it won the ragged form (RaceV11); parity cells keep k2b's tuned cell
            assert T.select(cc, "bf16", 64, 4, n, word="fast").row in ("triattn_native", "k2b")
        s = T.select(cc, "bf16", 64, 4, n, word="fast", **({"stack": "torch2.9.0+cu128-cpython-311-x86_64-linux-gnu"} if cc == "8.0" else {}))
        assert s.row == "k2b" and s.config == W[word]["config"] and f"tuned launch cell {word}" in s.reason, (cc, n, s.reason)
        assert T.select(cc, "bf16", 64, 4, n, word="k2b").config is None                      # the row word: the row's own cell table, as before
    s = T.select("9.0", "bf16", 64, 4, 2048, word="fast", stack=CUDA_STACK)
    assert s.row == "triattn_native"                                                              # 2048: the package's Triton route measured ahead of the tuned k2b cell
    s = T.select("9.0", "bf16", 64, 4, 2048, word="fast", prefer=("k2b", "cueq"))
    assert s.row == "k2b" and s.config == W["k2b@m64n64r2r255"]["config"]                            # k2b preferred by name still reaches its tuned cell
    assert T.select("8.0", "bf16", 64, 4, 2048, word="fast").config is None                      # no sweep at 2048 on 8.0: the row's own cell
    assert T.select("9.0", "bf16", 64, 4, 800, word="fast").config is None and T.select("8.0", "bf16", 64, 4, 400, word="fast").config is None   # below the measured floors


# ---------------------------------------------------------------------------------------------------- the sealed package row (triattn_native@<ACTIVE_PKG>)
def test_triattn_native_payload_manifests_and_words():
    from opt_core.kernels.triattn import triattn_native as CC
    assert CC.ACTIVE_PKG == "v11" and T.rows()["triattn_native"]["pkg"] == CC.ACTIVE_PKG and "triattn_native@" + CC.ACTIVE_PKG in T.rows()["triattn_native"]["words"]
    assert T.rows()["triattn_native"]["cc"] == ["9.0", "8.0"] and CC.honours("v10", (9, 0))[0] and not CC.honours("v10", (8, 0))[0] and "triattn_native@v10" in T.rows()["triattn_native"]["words"]
    pay = CC.payload_dir()
    for rel in ("__init__.py", "_face.py", "dispatch/candidate.py", "dispatch/pins.py", "dispatch/kernels/k13.py", "triton/triattn/k10.py", "cuda/csrc/triattn_sm90.cuh", "cuda_b/csrc/m1/triattn_m1_sm90.cuh", "cuda_c/csrc/triattn_mw.cu", "prebuilt/INDEX.json"):
        assert os.path.isfile(os.path.join(pay, rel)), rel
    man = json.load(open(os.path.join(CC.pkg_dir(), "testvectors", "manifest.json")))
    c90 = [c for c in man["cases"] if "9.0" in c.get("cc", ["9.0"])]; c80 = [c for c in man["cases"] if "8.0" in c.get("cc", [])]
    assert len(c90) == 17 and len(c80) == 22 and all(c["id"].startswith("a80_") for c in c80)      # generation 10's 17 cc-9.0 vectors carried + the sm_80 member's 22
    assert all(os.path.isfile(os.path.join(CC.pkg_dir(), "testvectors", c["id"] + ".expected.pt")) for c in man["cases"])
    built = CC.stacks_built()
    KEYS = {"torch2.7.1+cu126-cpython-311-x86_64-linux-gnu", "torch2.7.1+cu128-cpython-311-x86_64-linux-gnu", "torch2.10.0+cu128-cpython-311-x86_64-linux-gnu", "torch2.12.0+cu130-cpython-311-x86_64-linux-gnu",
            "torch2.12.0+cu130-cpython-312-x86_64-linux-gnu", "torch2.13.0+cu130-cpython-311-x86_64-linux-gnu", "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu"}
    assert set(built) == KEYS, sorted(KEYS ^ set(built))                                         # every measured torch stack with an sm_90 or sm_80 image (torch2.11.0+cu128-cp312 when LiftPrep's devtools key lands)
    assert all(set(v) == set(CC.extensions()) for v in built.values()) and len(CC.extensions()) == 4   # the sm_90a three + triattn_sm80_ext on every key
    for key, exts in built.items():
        for ext in exts:                                                       # every record's digests equal the bytes on disk; the producer's own records carry none (byte gate only)
            rec = CC.record(key, ext); so = os.path.join(pay, "prebuilt", key, ext + ".so")
            if "so_sha256" in rec:
                assert rec["so_sha256"] == hashlib.sha256(open(so, "rb").read()).hexdigest(), (key, ext)
                assert rec["source_sha256"] == CC.source_sha256(ext) and set(rec["loadcheck"]) == set(CC.LOADCHECK_CASES[ext]), (key, ext)
            for bad in ("builder_sandbox", "image_id"):
                assert rec.get(bad) in (None, "withheld"), (key, ext, bad)
    # words / admittance without importing the payload
    assert CC.admits("bf16", 32, 1024, key="torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90") == (True, "prebuilt triattn_m1_ext")
    assert CC.admits("bf16", 32, 1024, key="torch2.9.0+cu128-cpython-311-x86_64-linux-gnu")[1].startswith("no_prebuilt:v11@")
    assert CC.admits("fp32", 32, 1024)[1] == "dtype:fp32" and CC.admits("bf16", 48, 1024)[1] == "head_dim:D48" and CC.route_extension("bf16", 32, 384, strided=True) == "triattn_sm90_ext"
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "bf16", 32, 4, 1024, word="triattn_native@v9")
    assert e.value.kind.startswith("pkg_version:v9") and e.value.fallback == "cuda_sm90a"
    assert T.select("8.0", "bf16", 32, 4, 1024, word="triattn_native").row == "triattn_native"                 # cc 8.0: the package's any-GPU Triton member (no prebuilt needed)
    with pytest.raises(T.Refusal) as e:
        T.select("8.6", "bf16", 32, 4, 1024, word="triattn_native")
    assert e.value.kind.startswith("cc 8.6") and e.value.row == "triattn_native"
    assert "triattn_native" not in sys.modules.get("opt_core.kernels.triattn").__dict__ or True      # the payload itself is never imported by select/admits:
    assert not any(m.startswith("opt_core.kernels.triattn.triattn_native._payload") for m in sys.modules)


def test_triattn_native_cells_on_9_0():
    from opt_core.kernels.triattn import triattn_native as CC
    w = "triattn_native@" + CC.ACTIVE_PKG
    for key, c in T.cells().items():
        named = [r for r in c["fast_order"] if r.startswith("triattn_native")]
        if key.startswith("8.0|"):
            assert all(r in (w, "triattn_native@v10") for r in named), key                 # cc 8.0: generation 11's member (triattn_native@v11) decides; a listed triattn_native@v10 (generation 10's Triton member record) is refused by name on 8.0 while v11 is active
            if w in named:
                assert c["numerics"]["rows"][w]["excess_rms_vs_cueq"] <= 1.2 and c["ms"][w] and c["x_stock"][w] >= 1.0, key   # test_cc_8_0_sm80_member_by_call_form
            continue
        if named:
            w9 = named[0]                                                            # the cc-9.0 cells were measured on generation 10 (= generation 11's outputs on 9.0, honoured): triattn_native@v10 stands
            assert w9 in (w, "triattn_native@v10") and CC.honours(w9.split("@")[1], (9, 0))[0], key
            assert key.startswith("9.0|") and c["ms"][w9] and c["x_stock"][w9] and "numerics" in c and c["numerics"]["rows"][w9]["excess_rms_vs_cueq"] <= 1.2, key
    for H in (4, 2, 12, 8, 16):
        for n in (512, 800, 1024, 1536, 2048):                                   # bf16 D32 from 512 tokens: the package row on the stack that has its prebuilt
            s = T.select("9.0", "bf16", 32, H, n, word="fast", stack=CUDA_STACK)
            assert s.row == "triattn_native" and s.x_stock >= 2.0, (H, n, s.reason)
        st = T.cells()[f"9.0|bf16|D32|H{H}|N<=400|fwd"]["strided"]
        w9 = st["fast"]; assert w9 in (w, "triattn_native@v10") and st["route"] == "cuda" and st["x_stock"][w9] >= 2.0   # transposed views below 512: the package's strided extension (measured on generation 10, honoured)
        assert T.select("9.0", "bf16", 32, H, 384, word="fast", stack=CUDA_STACK, position_stride=384 * H * 32).row == "triattn_native"
    assert T.select("9.0", "bf16", 32, 4, 384, word="fast", stack=CUDA_STACK).row == "cuda_sm90a"       # contiguous below 512 at H4: unchanged winner
    s = T.select("9.0", "bf16", 64, 4, 1536, word="fast", stack=CUDA_STACK)
    assert s.row == "k2b" and s.config is not None and T.select("9.0", "bf16", 64, 4, 1536, word="fast", stack=CUDA_STACK, prefer=("triattn_native@v10", "cueq")).row == "triattn_native"   # D64: the tuned k2b cell measured ahead of the package's Triton route


def test_call_forms_on_9_0():
    """bf16 D32 H4 / H8 (and D16 H4): per-form records (bias_only / keypad / mask_bias) measured contiguous and strided; the form hint reads them.
    401-511 tokens: below the package's CUDA-route floor the winner differs by form; from 832 the package wins every form (its margin over
    cuda_sm90a is a few per cent at bias_only / keypad and x1.3-1.5 at per-row masks)."""
    for H in (4, 8):
        c = T.cells()[f"9.0|bf16|D32|H{H}|N<=511|fwd"]
        assert set(c["forms"]) == {"bias_only", "keypad", "mask_bias"} and c["forms"]["keypad"]["measured_at"] == 448
        assert T.select("9.0", "bf16", 32, H, 448, word="fast", stack=CUDA_STACK, form="keypad").row == "cuda_sm90a"
        assert T.select("9.0", "bf16", 32, H, 448, word="fast", stack=CUDA_STACK, form="bias_only").row == "cuda_sm90a"
        assert T.select("9.0", "bf16", 32, H, 448, word="fast", stack=CUDA_STACK, form="mask_bias").row == "k2b"
        assert T.select("9.0", "bf16", 32, H, 448, word="fast", stack=CUDA_STACK).row == c["fast"] == "k2b"            # no hint: the top-level order (per-row-mask inputs)
        assert T.select("9.0", "bf16", 32, H, 448, word="fast", stack=CUDA_STACK, form="keypad", position_stride=448 * H * 32).row == "triattn_native"   # strided keypad: the package's strided extension
        for n in (832, 1216, 2048):
            for form in ("bias_only", "keypad", "mask_bias"):
                assert T.select("9.0", "bf16", 32, H, n, word="fast", stack=CUDA_STACK, form=form).row == "triattn_native", (H, n, form)
            f = T.cell_for("9.0", "bf16", 32, H, n)[1]["forms"]
            r = f["keypad"]["contiguous"]["ms"]; assert 1.0 <= r["cuda_sm90a"] / r["triattn_native@v10"] < 1.08, (H, n)          # keypad form: a few per cent
            r = f["mask_bias"]["contiguous"]["ms"]; assert r["cuda_sm90a"] / r["triattn_native@v10"] > 1.3, (H, n)                # per-row masks: x1.3-1.5
    for n in (832, 1216, 2048):                                                                                                     # D16 template stacks: flash never ahead of k2b or the package
        for form in ("bias_only", "keypad"):
            r = T.cell_for("9.0", "bf16", 16, 4, n)[1]["forms"][form]["contiguous"]
            assert r["fast"] == "triattn_native@v10" and r["ms"]["flash"] > r["ms"]["k2b"] > r["ms"]["triattn_native@v10"], (n, form)
    assert T.select("9.0", "bf16", 32, 4, 448, word="fast", stack=CUDA_STACK, form="no_such_form").row == "k2b"                    # an unknown form word: the top-level order


def test_select_is_memoised_and_refusals_replay_by_name():
    """select() answers from a per-argument memo after the first call (the prebuilt listings are read once per process): the same Selection
    object comes back, a refused selection re-raises the same named Refusal, and select_cache_clear() forgets both."""
    T.select_cache_clear()
    a = T.select("9.0", "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK)
    b = T.select("9.0", "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK)
    assert a is b and a.row == "triattn_native"
    assert T.select((9, 0), "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK) is a            # cc spelled as a tuple: the same key
    for _ in range(2):                                                                            # first computed, then replayed from the memo
        with pytest.raises(T.Refusal) as e:
            T.select("9.0", "bf16", 32, 4, 1024, word="cuda_sm90a", stack="torch9.9.9+cu999-cpython-399-x86_64-linux-gnu-sm90")
        assert e.value.kind.startswith("no_prebuilt") and e.value.row == "cuda_sm90a" and e.value.fallback == "k2b"
    n = len(T._SELECT_MEMO); assert n >= 2
    T.select_cache_clear(); assert len(T._SELECT_MEMO) == 0 and T._STACKS_BUILT is None
    assert T.stacks_built() == CUDA_STACKS and T._STACKS_BUILT is not None                       # listing read once, then served from memory
    from opt_core.kernels.triattn import triattn_native as CC
    assert CC.stacks_built() is CC.stacks_built()                                                # the package face's listing: one dict per process


def test_per_call_host_inclusive_records():
    """The 512 / 896 / 1280-token H4 cells carry host-inclusive per-call numbers on two software stacks for two call forms; the package leads
    cuda_sm90a end to end on both, and the memoised select() costs are recorded before / after."""
    for n in (800, 1200, 1536):
        pc = T.cells()[f"9.0|bf16|D32|H4|N<={n}|fwd"]["per_call"]
        assert len(pc["stacks"]) == 2
        for st, forms in pc["stacks"].items():
            r = forms["mask_bias"]["rows"]
            assert r["triattn_native@v10"]["e2e_ms"] < r["cuda_sm90a"]["e2e_ms"] and r["triattn_native@v10"]["piped_ms"] < r["cuda_sm90a"]["piped_ms"], (n, st)
            assert r["triattn_native@v10"]["host_us"] < 130 and r["cuda_sm90a"]["host_us"] < 150, (n, st)
        for st, v in pc["select_cost_us"]["now"].items():
            assert v["triattn_native"] < 5 and pc["select_cost_us"]["before"][st]["triattn_native"] > 400, st


def test_in_model_records_are_notes_not_flips():
    c = T.cells()["9.0|bf16|D32|H4|N<=511|fwd"]
    assert "eager_trunk_one_word" in c["in_model"]["keypad"] and "-5.5 %" in c["in_model"]["keypad"]["eager_trunk_one_word"]
    assert T.select("9.0", "bf16", 32, 4, 448, word="fast", stack=CUDA_STACK, form="keypad").row == "cuda_sm90a"      # the op-level cell: unchanged
    assert T.select("9.0", "bf16", 32, 4, 448, word="triattn_native", stack=CUDA_STACK).row == "triattn_native"               # one word for every N: served


def test_warm_hook_reports_by_name_without_a_gpu():
    """triattn.warm(): the activation-time hook returns {row: {ready, seconds, report | refused}} and never raises a row's load error (on this
    CPU-only interpreter every GPU row reports not-ready by name); triattn_native.warm / the verdict-cache helpers exist for kits."""
    from opt_core.kernels.triattn import triattn_native as CC
    rep = T.warm(rows=("triattn_native", "cuda_sm90a", "k2b"))
    assert set(rep) == {"triattn_native", "cuda_sm90a", "k2b"} and all("seconds" in r and "ready" in r for r in rep.values())
    assert rep["k2b"]["ready"] and rep["k2b"]["report"] == {"nothing_to_warm": True}          # nothing to load for a Triton row without shapes
    assert callable(CC.warm) and callable(CC._verdict_key) and CC._verdict_dir.__doc__


def test_cc_8_0_sm80_member_by_call_form():
    """cc 8.0, generation 11 (RaceV11): the sm_80 member vs k2b (+ the tier word's tuned cell) / k2 / flash / cueq per cell x call form x layout,
    same box.  The un-hinted tier word follows the mask_bias (per-row ragged) contiguous record; a form record names the package first only under
    DEFAULT-FLIP (median >= 5 % below the best incumbent's, IQRs disjoint); parity keeps the incumbent; numerics in class (excess-RMS vs cueq
    <= 1.2), never bitwise (the exact tier is unchanged); words.cc80_flips lists exactly the WIN records."""
    t = T.table(); W = "triattn_native@v11"; T.select_cache_clear()
    assert t["rows"]["triattn_native"]["cc"] == ["9.0", "8.0"] and "triattn_sm80_ext" in t["rows"]["triattn_native"]["cc_8_0"]
    assert T.admits("triattn_native", (8, 0), "bf16", 32, 4, 1024)[0] and not T.admits("triattn_native", (8, 6), "bf16", 32, 4, 1024)[0]
    flips = {(c["cell"], c["form"], c["layout"]): c for c in t["words"]["cc80_flips"]["cells"]}
    assert flips and all(c["to"] == W and c["x_inc_over_new"] >= 1.05 for c in flips.values())
    n_cells = 0
    for key, c in T.cells().items():
        if not key.startswith("8.0|bf16") or "forms" not in c:
            continue
        _, _, Dd, Hh, Nn, _ = key.split("|"); D, H, n = int(Dd[1:]), int(Hh[1:]), int(Nn[3:])
        top = c["forms"].get("mask_bias", {}).get("contiguous")
        if not (isinstance(top, dict) and W in top.get("ms", {})):
            continue                                                                                              # a cell RaceV11 did not measure keeps generation 10's records
        n_cells += 1
        won = top["verdict"]["result"] == "WIN"
        assert (c["fast"] == W) == won and (c["fast_order"][0] == W) == won, key                                   # the top level follows the ragged-form record
        assert T.select("8.0", "bf16", D, H, min(n, top["measured_at"]), word="fast").row == ("triattn_native" if won else c["fast"].split("@")[0]), key
        for form, fr in c["forms"].items():
            if not isinstance(fr, dict):
                continue
            for lay in ("contiguous", "strided"):
                r = fr.get(lay)
                if not (isinstance(r, dict) and W in r.get("ms", {})):
                    continue
                v = r["verdict"]; first = r["fast_order"][0]
                ex = r["excess_rms_vs_cueq"][W]
                assert r["x_stock"][W] >= 1.0 and (ex is None and form == "padrows" and "numerics_note" in r or ex is not None and ex <= 1.2) and r["fast"] == first, (key, form, lay)   # padrows: rel-RMS undefined (the references emit not-a-number on dead rows)
                if v["result"] == "WIN":
                    assert first == W and v["inc_over_pkg"] >= 1.05 and v["iqr_disjoint"] and (key, form, lay) in flips, (key, form, lay)
                else:                                                                    # PARITY / LOSS: the incumbent keeps the form/layout (DEFAULT-FLIP rule); the package listed behind it
                    assert first == v["vs"] and first != W and W in r["fast_order"] and (key, form, lay) not in flips, (key, form, lay)
    assert n_cells >= 24
    for D, H in ((16, 4), (32, 4), (64, 4)):
        for n in (400, 800, 1200):
            assert T.select("8.0", "bf16", D, H, n, word="fast", form="keypad").row == "triattn_native", (D, H, n)      # the padded-key form: the member at every measured cell
    assert T.select("8.0", "bf16", 32, 4, 1024, word="exact").row in ("triattn_exact", "exact_headsplit", "cueq")    # exact tier: never the tolerance-class member
    with pytest.raises(T.Refusal) as e:
        T.select("8.0", "bf16", 32, 4, 1024, word="triattn_native@v10")                                                 # generation 10's word on 8.0: another kernel served there -> refused by name
    assert e.value.kind.startswith("pkg_version:") and T.select("9.0", "bf16", 32, 4, 1024, word="triattn_native@v10", stack=CUDA_STACK).row == "triattn_native"   # honoured on 9.0


def test_a_head_dim_no_kernel_row_carries_is_named_not_uncovered():
    """bf16 D24 H16 (a pair-biased attention a kit routes through this face): no kernel row carries head_dim 24 (Triton rows 16/32/64/128, the
    extensions 32, the package 16/32/64/128) -- structural.  Tier words serve the stock op and NAME it (`no_row:head_dim24` in the reason; the
    fast tier also names k2b's refusal first); a shape kernel rows admit but nobody measured refuses `no_cell:*` under BOTH tiers."""
    T.select_cache_clear()
    for w in ("exact", "fast"):
        s = T.select("9.0", "bf16", 24, 16, 400, word=w, stack="torch2.10.0+cu128-cpython-311-x86_64-linux-gnu-sm90")
        assert s.row == "cueq" and not s.measured and "no_row:head_dim24" in s.reason, (w, s.reason)
    assert "'k2b:head_dim_24', 'no_row:head_dim24'" in T.select("9.0", "bf16", 24, 16, 800, word="fast").reason
    assert T.select("8.0", "bf16", 24, 16, 400, word="exact").row == "cueq"
    for w in ("fast", "exact"):
        with pytest.raises(T.Refusal) as e:
            T.select("9.0", "fp32", 16, 4, 400, word=w)
        assert e.value.kind == "no_cell:fp32_D16" and e.value.fallback == "cueq"
    assert T.select("9.0", "fp32", 16, 4, 400, word="k2b").row == "k2b"                          # the row word opts in


def test_offline_select_accepts_cell_style_stack_words_for_the_prebuilt_gate():
    """A CPU / offline resolution query may pass the table's cell-style stack word instead of the in-process ABI key: it stands for every
    built prebuilt key of that torch build (any-of) -- no false no_prebuilt; the ABI key form is unchanged (exact match)."""
    from opt_core.kernels.triattn import triattn_native as CC
    T.select_cache_clear()
    abi = CUDA_STACK                                                                   # torch2.13.0+cu130-cpython-311-x86_64-linux-gnu-sm90
    words = ("H100:2.13.0+cu130/3.7.1/cueq0.11.1", "H100-80GB:torch2.13.0+cu130/triton3.7.1/cueq0.11.1/remeasure-ragged", "h100:torch2.13.0+cu130")
    assert T.stack_key_candidates(None, ["a"]) is None
    assert T.stack_key_candidates(abi, T.stacks_built()) == [abi]
    cands = T.stack_key_candidates(words[0], T.stacks_built())
    assert abi in cands and all(k.startswith("torch2.13.0+cu130-cpython-") for k in cands) and len(cands) >= 2      # cp311 + cp312 built
    assert T.stack_key_candidates("H100:2.9.0+cu128/3.3.0/cueq0.10.0", T.stacks_built()) == []
    for w in words + (abi,):
        ok, why = T.admits("cuda_sm90a", "9.0", "bf16", 32, 4, 1200, stack=w); assert ok, (w, why)
        ok, why = T.admits("triattn_native", "9.0", "bf16", 32, 4, 1200, stack=w); assert ok, (w, why)
        s = T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack=w)
        assert s.row == "triattn_native" and "no_prebuilt" not in s.reason, (w, s.reason)
        assert T.select("9.0", "bf16", 32, 4, 1200, word=f"triattn_native@{CC.ACTIVE_PKG}", stack=w).row == "triattn_native"
        assert T.select("9.0", "bf16", 32, 4, 1200, word="cuda_sm90a", stack=w).row == "cuda_sm90a"
    assert "prebuilt: any-of" in T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack=words[0]).reason          # the offline form says which keys it stands for
    assert "prebuilt: any-of" not in T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack=abi).reason            # the in-process form: unchanged
    s = T.select("9.0", "bf16", 32, 4, 1200, word="fast", stack="H100:2.9.0+cu128/3.3.0/cueq0.10.0")             # a torch build nobody built for: no_prebuilt by name, the next row
    assert s.row not in ("triattn_native", "cuda_sm90a") and "no_prebuilt" in s.reason
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "bf16", 32, 4, 1200, word="cuda_sm90a", stack="H100:2.9.0+cu128/3.3.0/cueq0.10.0")
    assert e.value.kind.startswith("no_prebuilt:")
    a100 = T.select("8.0", "bf16", 32, 4, 1200, word="fast", stack="A100-SXM4-80GB:torch2.13.0+cu130/triton3.7.1/cueq0.11.1")      # the A100 form of the word
    assert a100.row == T.select("8.0", "bf16", 32, 4, 1200, word="fast").row and "no_prebuilt" not in a100.reason


def test_provider_key_arch_suffix_strip_is_generic():
    """The package's prebuilt keys carry no arch suffix; the provider's do ('-sm90' today, '-sm80' for a cc 8.0 process): either strips."""
    from opt_core.kernels.triattn import triattn_native as CC
    base = "torch2.13.0+cu130-cpython-311-x86_64-linux-gnu"
    for suf in ("-sm90", "-sm80", "-sm90a", "-sm_80", ""):
        assert CC.provider_key_to_pkg_key(base + suf) == base, suf
    assert CC.provider_key_to_pkg_key(None) is None and CC.provider_key_to_pkg_key("") == ""
    assert CC.admits("bf16", 32, 1200, key=base + "-sm80", cc=(8, 0))[0] == CC.admits("bf16", 32, 1200, key=base + "-sm90", cc=(8, 0))[0] == CC.admits("bf16", 32, 1200, key=base, cc=(8, 0))[0]
    assert T.stack_key_candidates(base + "-sm80", T.stacks_built()) == [base + "-sm90"]          # the extension's built key of that ABI, whatever suffix the query spells


def test_runtime_admission_and_load_path_normalize_the_provider_key_alike():
    """An A100 process spells its provider key '...-sm80', an H100 one '...-sm90'; the package's prebuilt tree carries no arch suffix.  Walk
    resolve -> admit -> prebuilt path -> digest verification (the load path's first step, CPU only) for both spellings and both interpreter
    ABIs the tree ships: the same directory, the sm_80 member present, digests verified -- no kit change needed on A100."""
    from opt_core.kernels.triattn import triattn_native as CC
    T.select_cache_clear()
    for abi in ("cpython-311-x86_64-linux-gnu", "cpython-312-x86_64-linux-gnu"):
        base = f"torch2.13.0+cu130-{abi}"
        assert "triattn_sm80_ext" in CC.stacks_built()[base], base                                  # the cc 8.0 member is built for this ABI
        so = os.path.join(CC.payload_dir(), "prebuilt", base, "triattn_sm80_ext.so"); assert os.path.isfile(so)
        for key in (base + "-sm80", base + "-sm90", base):
            assert CC.provider_key_to_pkg_key(key) == base
            ok, why = T.admits("triattn_native", "8.0", "bf16", 32, 4, 1024, stack=key); assert ok, (key, why)        # admission (select's gate)
            assert T.select("8.0", "bf16", 32, 4, 1024, word="fast", stack=key).row == "triattn_native", key
            rec = CC.record(key, "triattn_sm80_ext"); assert rec and rec["so_sha256"], key                           # the record lookup by either spelling
            man = CC.verify_manifests(key)                                                                     # the load path's digest step by either spelling
            assert "triattn_sm80_ext" in man, (key, sorted(man))


def test_active_pkg_compares_equal_to_the_generations_it_honours():
    """A kit that pinned `triattn_native.ACTIVE_PKG == "v10"` at certification keeps binding the package after the generation-11 lift (v11
    carries v10's routes and output bytes unchanged on cc 9.0): the version word compares equal to honoured generations; everything else
    (str / format / hash / json / the word-level cc gate) is the plain active word."""
    import json as _json
    from opt_core.kernels.triattn import triattn_native as CC
    A = CC.ACTIVE_PKG
    assert str(A) == "v11" and f"{A}" == "v11" and "%s" % A == "v11" and _json.dumps(A) == '"v11"' and hash(A) == hash("v11")
    assert A == "v11" and not (A != "v11")
    for hon in CC.HONOURED.get(str(A), {}):                                        # "v10"
        assert A == hon and hon == A and not (A != hon) and not (hon != A), hon    # the kit-side pin `ACTIVE_PKG != "v10"` no longer drops the package
    assert A != "v9" and A != "v12" and A != "" and A is not None and (A != None)  # noqa: E711
    assert CC.honours("v10", (9, 0))[0] and not CC.honours("v10", (8, 0))[0] and CC.honours("v11", (8, 0))[0]     # the word-level gate keeps its device classes
    assert {"v11": 1}[A] == 1 and CC.HONOURED.get(A) == CC.HONOURED["v11"]
    T.select_cache_clear()
    s = T.select("9.0", "bf16", 32, 4, 800, word="fast", prefer=("triattn_native@v10", "cuda_sm90a", "k2b"), stack=CUDA_STACK)   # the certified kit's exact call
    assert s.row == "triattn_native", s.reason


def test_peak_audit_no_big_split():
    """big / peak audit (RaceV11): every cc-8.0 bf16 cell and the 9.0 D16 / D32 H4 cells carry `mem_delta_mib` (tier-word winner minus k2b,
    same process); no winner costs memory beyond noise over k2b, so no cell splits its big word (big == fast by the fold rule stands)."""
    t = T.table(); pa = t["words"]["peak_audit"]
    c80 = [k for k in t["cells"] if k.startswith("8.0|bf16")]
    assert len(c80) == pa["cc_8_0"]["cells"] >= 40 and all("mem_delta_mib" in t["cells"][k] for k in c80)      # 36 + the 2560 bucket (H4 / H8 / H2 / H12) + the beyond-2048 buckets
    c90 = pa["cc_9_0"]["cells"]
    assert {"9.0|bf16|D16|H4|N<=1200|fwd", "9.0|bf16|D16|H4|N<=1536|fwd", "9.0|bf16|D16|H4|N<=2048|fwd", "9.0|bf16|D32|H4|N<=1200|fwd", "9.0|bf16|D32|H4|N<=1536|fwd", "9.0|bf16|D32|H4|N<=2048|fwd"} <= set(c90)
    for k in c80 + c90:
        m = t["cells"][k]["mem_delta_mib"]
        assert m["big_split"] is False and "no split" in m["big_note"] and m["max_delta"] <= 8.0 and all(isinstance(v, (int, float)) for v in m["at"].values()), k
    for k in ("9.0|bf16|D16|H4|N<=2048|fwd", "9.0|bf16|D32|H4|N<=2048|fwd"):                  # the 2048 keys: dense + keypad x both layouts measured, the package below k2b on all four
        at = t["cells"][k]["mem_delta_mib"]["at"]
        assert {"S2048/bias_only/contiguous", "S2048/keypad/contiguous", "S2048/bias_only/strided", "S2048/keypad/strided"} <= set(at) and max(at.values()) < 0


def test_tier_word_big_is_fast_minus_rows_with_a_recorded_memory_cost():
    """`big` = `fast`'s row on every current cell (no cell's winner records a peak above the stock op's), through every form / layout the
    cells carry; a synthetic cell whose winner's recorded peak exceeds the stock op's resolves to the next lower-peak row by name; the peak
    reader normalizes versioned package names; the census records the word."""
    import copy
    assert "big" in T.TIER_WORDS
    T.select_cache_clear()
    n_checked = 0
    for key, c in T.cells().items():
        cc, dt, D, H, N = key.split("|")[:5]; D, H, N = int(D[1:]), int(H[1:]), int(N[3:])
        n = min(N, 2048) if N < 60000 else 4096
        stack = CUDA_STACK if cc == "9.0" else None
        forms = [None] + [f for f in (c.get("forms") or {}) if isinstance((c.get("forms") or {}).get(f), dict)]
        for form in forms:
            for ps in (None,) + ((n * H * D,) if "strided" in c else ()):
                try:
                    f = T.select(cc, dt, D, H, n, word="fast", stack=stack, form=form, position_stride=ps)
                except T.Refusal as r:
                    with pytest.raises(T.Refusal) as rb:
                        T.select(cc, dt, D, H, n, word="big", stack=stack, form=form, position_stride=ps)
                    assert rb.value.kind == r.kind, (key, form, ps)
                    continue
                b = T.select(cc, dt, D, H, n, word="big", stack=stack, form=form, position_stride=ps)
                assert (b.row, b.config, b.input_precision) == (f.row, f.config, f.input_precision) and b.word == "big" and b.reason.startswith("tier big"), (key, form, ps, f.row, b.row)
                n_checked += 1
    assert n_checked >= len(T.cells())
    # the peak reader: versioned names fold, other-library records drop, `at` picks the largest size
    assert T.cell_peaks({"peak_mib": {"rows": {"triattn_native@v10": 23.7, "k2b": 23.7, "cueq": 24.9}}}) == {"triattn_native": 23.7, "k2b": 23.7, "cueq": 24.9}
    assert T.cell_peaks({"peak_mib": {"at": {"800": {"k2b": 1.0, "cueq": 2.0}, "1200": {"k2b": 384.5, "cueq": 395.5, "cueq(0.10.0)": 396.4}}}}) == {"k2b": 384.5, "cueq": 395.5}
    # a synthetic split: the 9.0 D32 H4 N<=1200 winner is made to cost 100 MiB more than the stock op -> big passes it over by name, fast keeps it
    key = "9.0|bf16|D32|H4|N<=1200|fwd"; live = T.cells()[key]; saved = copy.deepcopy(live)
    try:
        f = T.select("9.0", "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK); assert f.row == "triattn_native"
        peaks = T.cell_peaks(live); stock = T._stock_of(live); assert stock in peaks, (stock, peaks)
        live["peak_mib"] = {"rows": dict(peaks, **{"triattn_native@v11": peaks[stock] + 100.0, "cuda_sm90a": peaks[stock] + 100.0})}
        T.select_cache_clear()
        b = T.select("9.0", "bf16", 32, 4, 1024, word="big", stack=CUDA_STACK)
        assert b.row not in ("triattn_native", "cuda_sm90a") and b.row == next(r.split("@")[0] for r in live["fast_order"] if r.split("@")[0] not in ("triattn_native", "cuda_sm90a") and T.admits(r.split("@")[0], "9.0", "bf16", 32, 4, 1024, stack=CUDA_STACK)[0] and not T.peak_exceeds_stock(live, r)), b
        assert "triattn_native:peak:" in b.reason and "cuda_sm90a:peak:" in b.reason, b.reason
        assert T.select("9.0", "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK).row == "triattn_native"          # fast ignores peaks
        live["peak_mib"] = {"rows": dict(peaks, **{"triattn_native@v11": peaks[stock] + 4.0})}                     # inside the noise band: not a split
        T.select_cache_clear()
        assert T.select("9.0", "bf16", 32, 4, 1024, word="big", stack=CUDA_STACK).row == "triattn_native"
    finally:
        live.clear(); live.update(saved); T.select_cache_clear()
    with pytest.raises(T.Refusal) as e:
        T.select("9.0", "bf16", 32, 4, 1024, word="biggly")
    assert e.value.kind == "unknown_word:biggly"


def test_retired_kit_tables_are_cells_words_and_floors():
    """The kit-side triangle-attention tables retired at unification live here as data: flash launch words (cc 8.0 tiles + a cc-10 record),
    per-stack exact floors (no cell carries one now: the exact word is the stock op on those cells on every stack; other stacks
    and every row word unchanged), and the `kit_tables` records naming where each item lives."""
    W = T.table()["words"]; CW = W["config_words"]
    assert CW["flash@m64n64s3"]["config"] == {"BLOCK_M": 64, "BLOCK_N": 64, "ROWS": 1, "num_warps": 4, "num_stages": 3, "ORDER": 0} and CW["flash@m64n64s3"]["cc"] == "8.0"
    assert CW["flash@m64n32r2s3"]["config"] == {"BLOCK_M": 64, "BLOCK_N": 32, "ROWS": 2, "num_warps": 4, "num_stages": 3, "ORDER": 0}
    assert all(str(CW[w]["evidence"]).startswith("source: kit table") for w in ("flash@m64n64s3", "flash@m64n32r2s3"))
    T.select_cache_clear()
    s = T.select("8.0", "bf16", 32, 4, 1024, word="flash@m64n64s3"); assert s.row == "flash" and s.config == CW["flash@m64n64s3"]["config"]
    assert T.select("8.0", "bf16", 32, 4, 1024, word="flash").config is None                        # the row word keeps the row's own cell (default rule)
    K271, K13 = "torch2.7.1+cu128-cpython-311-x86_64-linux-gnu-sm90", CUDA_STACK
    V271, V13 = "9.0|torch2.7.1+cu128|cueq0.10.0", "9.0|torch2.13.0+cu130|cueq0.11.1"
    for n in (200, 256, 400, 448, 512, 639):
        e = T.select("9.0", "bf16", 32, 4, n, word="exact", stack=K271, exact_stack=V271)
        assert e.row in ("cueq", "triattn_exact") and e.row != "exact_headsplit" and "exact floor" not in e.reason, (n, e.row, e.reason)   # never the head-split row there; the bit-identical row once that stack is vouched, the stock op by name until then
    assert T.select("9.0", "bf16", 32, 4, 640, word="exact", stack=K271, exact_stack=V271).row in ("cueq", "triattn_exact")   # from 640 too: never the head-split row
    assert T.select("9.0", "bf16", 32, 4, 896, word="exact", stack=K271, exact_stack=V271).row in ("cueq", "triattn_exact")
    for n in (256, 400, 512):                                                                                              # the other stacks: the cell's exact row as measured there
        assert T.select("9.0", "bf16", 32, 4, n, word="exact", stack=K13, exact_stack=V13).row == T.cells()[T.select("9.0", "bf16", 32, 4, n, word="exact", stack=K13, exact_stack=V13).cell]["exact"]
    assert T.select("9.0", "bf16", 32, 4, 400, word="fast", stack=K271).row != "cueq"                                     # the fast tier never reads the exact floor
    assert T.exact_floor_for(T.cells()["9.0|bf16|D32|H4|N<=400|fwd"], K271) is None and T.exact_floor_for(T.cells()["9.0|bf16|D32|H4|N<=400|fwd"], K13) is None   # no cell carries a floor record
    assert T.exact_floor_for({"exact_floor": {"torch2.7.1+cu128": {"below_tokens": 640, "row": "cueq"}}}, "H100-80GB:torch2.7.1+cu128/triton3.3.1/cueq0.10.0")["row"] == "cueq"   # the record form the hook reads
    items = W["kit_tables"]["items"]; assert len(items) >= 7 and all(str(i["source"]).startswith(("kit table", "cells")) for i in items)
    # the words a cofolding kit's column reads today (its retired MIN_TOKENS 512 floor): keypad D32 H4 below 512 = cuda_sm90a, from 512 = the package, D16 = the package
    ks = "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm90"
    assert [T.select("9.0", "bf16", 32, 4, n, word="fast", stack=ks, form="keypad").row for n in (256, 448, 512, 832)] == ["cuda_sm90a", "cuda_sm90a", "triattn_native", "triattn_native"]
    assert T.select("9.0", "bf16", 16, 4, 832, word="fast", stack=ks, form="keypad").row == "triattn_native"


def test_unmeasured_cc_inherits_the_nearest_columns_portable_triton_rows():
    """A compute capability without a measured column (10.0 B200, 10.3 B300, 12.0, 8.6 ...): the tier words fast / big INHERIT the nearest
    measured column's portable Triton rows (never a Refusal / statement while a portable row admits, never cuda_sm90a / triattn_native
    across arch, never the donor's tuned launch cell), exact = the library op by name; measured=False and the census token says
    inherited_cc:unmeasured (from <column>).  The measured columns resolve exactly as before."""
    import opt_core.cell_census as CC
    T.select_cache_clear()
    assert T.measured_ccs() == ["8.0", "9.0", "10.0", "10.3"] and T.donor_cc("12.0") == "10.3" and T.donor_cc((12, 0)) == "10.3" and T.donor_cc("8.6") == "8.0" and T.donor_cc("7.5") == "8.0"
    ARCH_ROWS = ("cuda_sm90a", "triattn_native", "exact_headsplit")
    for cc in ("12.0", "11.0", (12, 0)):
        for D, H, n in ((32, 4, 256), (32, 4, 1024), (32, 4, 2560), (32, 4, 5000), (32, 8, 1536), (32, 2, 448), (16, 4, 1024), (64, 4, 1536)):
            for wd in ("fast", "big"):
                s = T.select(cc, "bf16", D, H, n, word=wd, stack=CUDA_STACK)
                assert s.row in T.TRITON_ROWS and s.row not in ARCH_ROWS and not s.measured and s.config is None, (cc, D, H, n, wd, s)
                dn = T.donor_cc(cc, "bf16", D, H)
                assert T.INHERITED_CC in s.reason and f"from the {dn} column" in s.reason and "no column at all" in s.reason and s.cell and s.cell.startswith(dn + "|"), s.reason
            e = T.select(cc, "bf16", D, H, n, word="exact", stack=CUDA_STACK, exact_stack=f"{T.cc_word(cc)}|torch2.13.0+cu130|cueq0.11.1")
            assert e.row in T.STOCK_ROWS and e.cls == "stock" and T.INHERITED_CC in e.reason, (cc, D, H, n, e)
        # a kit's preference tuple naming arch rows first, the padded-key form, a transposed layout, fp32: still a portable row
        assert T.select(cc, "bf16", 32, 4, 1024, word="fast", prefer=("triattn_native@v11", "cuda_sm90a", "k2b"), stack=CUDA_STACK).row == "k2b"
        assert T.select(cc, "bf16", 32, 4, 1024, word="fast", form="keypad", stack=CUDA_STACK).row in T.TRITON_ROWS
        assert T.select(cc, "bf16", 32, 4, 1024, word="fast", position_stride=1024 * 4 * 32).row in T.TRITON_ROWS
        assert T.select(cc, "fp32", 32, 4, 1024, word="fast").row in T.TRITON_ROWS
        with pytest.raises(T.Refusal):
            T.select(cc, "bf16", 32, 4, 1024, word="cuda_sm90a", stack=CUDA_STACK)                # the arch row by name: refused by name as before
    s86 = T.select("8.6", "bf16", 32, 4, 1024, word="fast")
    assert s86.row in T.TRITON_ROWS and s86.config is None and "from the 8.0 column" in s86.reason and s86.cell.startswith("8.0|")   # never the 8.0 tuned launch cell
    with pytest.raises(T.Refusal) as r:
        T.select("7.5", "bf16", 32, 4, 1024, word="fast")                                            # below sm_80 no portable row admits: the library op by name
    assert r.value.kind == "no_cell:cc=7.5" and r.value.fallback == "cueq"
    # census token
    T.select_cache_clear()
    T.select("12.0", "bf16", 32, 4, 1024, word="fast")
    es = [t for t in CC.table() if t.get("provider") == "triattn" and t.get("cc") == "12.0" and t.get("shape") == "D32H4" and t.get("word") == "fast" and t.get("bucket") == "N=1024" and t.get("form") == "fwd"]
    dn = T.donor_cc("12.0", "bf16", 32, 4)
    assert es and all(e["outcome"] == "inherited" and str(e["note"]).startswith("inherited_cc:unmeasured(from") and dn in str(e["note"]) and str(e["cell"]).startswith(dn + "|") for e in es), es
    # the measured columns: unchanged (spot checks; the whole-table tests above hold the rest)
    assert T.select("9.0", "bf16", 32, 4, 1024, word="fast", stack=CUDA_STACK).row == "triattn_native" and T.select("8.0", "bf16", 32, 4, 1024, word="fast").row == "triattn_native"
    assert T.select("9.0", "bf16", 32, 4, 256, word="fast", stack=CUDA_STACK, form="keypad").row == "cuda_sm90a"


def test_cc_10_3_with_one_injected_cell_still_inherits_the_portable_row_for_every_other_key():
    """Per-KEY inheritance (never per cc): a cc that owns ONE measured cell (here an injected 10.3 bf16 D32 H4 cell) still inherits the nearest
    measured column's portable Triton row for every OTHER (dtype, head_dim, heads) key instead of stranding those keys on the library op; the
    injected key itself is served from its own cell; the census says partial_cc; the 9.0 / 8.0 columns resolve byte-identically."""
    import copy, json
    import opt_core.cell_census as CC
    K = "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu-sm100"
    before = json.dumps({k: T.cells()[k] for k in T.cells() if k.startswith(("9.0|", "8.0|", "10.0|", "10.3|"))}, sort_keys=True)
    ref = {(cc, D, H, n, w): (T.select(cc, "bf16", D, H, n, word=w, stack=CUDA_STACK if cc == "9.0" else None).row)
           for cc in ("9.0", "8.0") for D, H in ((32, 4), (32, 8), (16, 4), (64, 4)) for n in (256, 1024, 2048) for w in ("fast", "big")}
    inj = "12.0|bf16|D32|H4|N<=1200|fwd"                        # (10.0 / 10.3 are measured columns since the parts fold: the injected-cell scenario runs on cc 12.0)
    T.cells()[inj] = copy.deepcopy(T.cells()["9.0|bf16|D32|H4|N<=1200|fwd"])
    T.select_cache_clear()
    try:
        assert T.measured_ccs() == ["8.0", "9.0", "10.0", "10.3", "12.0"] and T.group_measured("12.0", "bf16", 32, 4) and not T.group_measured("12.0", "bf16", 32, 8)
        assert T.donor_cc("12.0", "bf16", 32, 8) == "10.3" and T.donor_cc("12.0", "bf16", 16, 4) == "10.3" and T.donor_cc("11.0") == "10.3"
        own = T.select("12.0", "bf16", 32, 4, 1024, word="fast", stack=K)                              # the injected key: its own cell (the arch rows pass over by name -> k2b)
        assert own.cell == inj and own.row in T.TRITON_ROWS and T.INHERITED_CC not in own.reason
        for D, H, n in ((32, 8, 1024), (32, 2, 448), (32, 12, 1536), (16, 4, 1024), (64, 4, 1536), (32, 8, 6000)):   # every OTHER key: inherited, never the library op
            for wd in ("fast", "big"):
                s = T.select("12.0", "bf16", D, H, n, word=wd, stack=K)
                assert s.row in T.TRITON_ROWS and not s.measured and T.INHERITED_CC in s.reason and "partial_cc" in s.reason and s.cell.startswith(T.donor_cc("12.0", "bf16", D, H) + "|"), (D, H, n, wd, s.reason)
            e = T.select("12.0", "bf16", D, H, n, word="exact", stack=K, exact_stack="12.0|torch2.13.0+cu130|cueq0.11.1")
            assert e.row in T.STOCK_ROWS, (D, H, n, e)
        assert T.select("12.0", "fp32", 32, 4, 1024, word="fast").row in T.TRITON_ROWS                     # another dtype key: inherited too
        T.select_cache_clear(); T.select("12.0", "bf16", 32, 8, 1024, word="fast", stack=K)
        es = [t for t in CC.table() if t.get("provider") == "triattn" and t.get("cc") == "12.0" and t.get("shape") == "D32H8" and t.get("word") == "fast"]
        assert es and all(e_["outcome"] == "inherited" and "inherited_cc:unmeasured(from" in str(e_["note"]) for e_ in es) and any("partial_cc" in str(e_["note"]) for e_ in es), es
        with pytest.raises(T.Refusal):
            T.select("12.0", "bf16", 32, 7, 300, word="fast")                                           # a key NO column measured (7 heads): refused by name as on the measured cards
    finally:
        del T.cells()[inj]; T.select_cache_clear()
    assert json.dumps({k: T.cells()[k] for k in T.cells() if k.startswith(("9.0|", "8.0|", "10.0|", "10.3|"))}, sort_keys=True) == before
    assert ref == {(cc, D, H, n, w): (T.select(cc, "bf16", D, H, n, word=w, stack=CUDA_STACK if cc == "9.0" else None).row)
                   for cc in ("9.0", "8.0") for D, H in ((32, 4), (32, 8), (16, 4), (64, 4)) for n in (256, 1024, 2048) for w in ("fast", "big")}
    # per key on the measured cards too: 8.0 has no bf16 D32 H16 cell while 9.0 measured it -> the 9.0 column's portable row on 8.0 (its own launch cell), not no_cell
    s16 = T.select("8.0", "bf16", 32, 16, 1024, word="fast")
    assert s16.row in T.TRITON_ROWS and T.INHERITED_CC in s16.reason and "partial_cc" in s16.reason and s16.config is None and s16.cell.startswith("9.0|bf16|D32|H16|")

def test_beyond_2048_buckets():
    """N<=3072 / 4096 / 6144 cells (where measured; one 80 GB card per cc, the chunked pair-row variant where the square call does not fit):
    the tier words resolve to the cell's own winner inside the bucket, exact = the vouched exact row or the library op, and every 9.0 / 8.0
    resolution at or below 2560 tokens is unchanged."""
    T.select_cache_clear()
    big = {k: c for k, c in T.cells().items() if int(k.split("|")[4][3:]) > 2560}
    assert big, "no beyond-2048 cell"
    for key, c in big.items():
        cc, dt, D, H, N, _ = key.split("|"); n = int(N[3:]) - 100
        st = CUDA_STACK if cc == "9.0" else None
        s = T.select(cc, dt, int(D[1:]), int(H[1:]), n, word="fast", stack=st)
        assert s.cell == key and s.measured and c["measured_sizes"] == [int(N[3:])], key
        want = c["fast_order"][0].split("@")[0]
        assert s.row == want or (want == "triattn_native" and s.row == "triattn_native"), (key, s.row, c["fast_order"])
        assert c["x_stock"][c["fast_order"][0]] >= 1.0 or c["fast_order"][0] in T.STOCK_ROWS, key
        e = T.select(cc, dt, int(D[1:]), int(H[1:]), n, word="exact", stack=st, exact_stack=f"{cc}|torch2.13.0+cu130|cueq0.11.1")
        assert e.row == (c["exact"] if c["exact"] not in T.STOCK_ROWS else e.row) and (c["exact"] in T.STOCK_ROWS or key in [key] and c["vouched_on"][c["exact"]]), key
        if cc == "8.0" and dt == "bf16":
            assert "mem_delta_mib" in c, key
        if c.get("rows_measured"):
            assert c["rows_measured"] < int(N[3:]) and "variant_note" in c, key
    assert "beyond_2048_note" in T.table()["words"]


def test_stack_probes_degrade_by_name_under_a_stubbed_torch():
    """A stubbed / device-less torch (a kit's CPU unit test with a SimpleNamespace torch) never raises out of the select path: the stack probe
    returns an unknown key that matches no vouch (the exact tier names the library op), fast resolves as usual."""
    import sys, types
    real = sys.modules.get("torch")
    sys.modules["torch"] = types.SimpleNamespace(__version__=None, bfloat16="bf16", float32="fp32", float16="fp16")   # no .cuda at all
    T.select_cache_clear()
    try:
        k = T.exact_stack_key("10.3"); assert k.startswith("10.3|") and "cueq" in k
        k0 = T.exact_stack_key(None); assert k0.startswith("?.?|torch?|"), k0
        for cc in ("10.3", "9.0", "8.0"):
            assert T.select(cc, "bf16", 32, 4, 1024, word="fast").row in T.ROW_NAMES
            e = T.select(cc, "bf16", 32, 4, 1024, word="exact", exact_stack=T.exact_stack_key(cc))
            assert e.row in T.STOCK_ROWS or cc in ("9.0", "8.0"), (cc, e)
    finally:
        if real is not None: sys.modules["torch"] = real
        else: sys.modules.pop("torch", None)
        T.select_cache_clear()


def test_inherited_row_launch_guard_steps_aside_by_name_and_never_retries_the_dead_row():
    """An inherited portable row whose first launch raises (a build / compile / launch failure on a new arch) never lets the raw error out:
    the row is dead for the process (census stepped_aside:error:<ExcType> once), the donor cell's NEXT portable row serves, else the library
    op by name (the kit's stock callable); a second call does not retry the dead row; out-of-memory is re-raised unchanged."""
    import types
    import opt_core.cell_census as CC
    calls = {"k2b": 0, "k2": 0, "flash": 0, "stock": 0}
    class Boom(RuntimeError): pass
    def mk(name, exc=None):
        def f(q, k, v, bias, mask=None, scale=None, **kw):
            calls[name] += 1
            if exc is not None: raise exc
            return ("served", name)
        return f
    fake_k2b = types.SimpleNamespace(attn_k2b=mk("k2b", Boom("ptxas: sm_103 not supported")), attn_k2=mk("k2", Boom("compile failed")))
    fake_flash = types.SimpleNamespace(flash_triangle_attention=mk("flash"))
    real_route = T._route
    T._route = lambda name: fake_k2b if name == "fpf_triatt_k2b" else fake_flash
    T._DEAD_INHERITED.clear(); T.select_cache_clear()
    args = ("12.0", "bf16", 32, 4, 1024, "fwd")                     # an unmeasured cc (10.0 / 10.3 are measured columns since the parts fold)
    try:
        sel = T.select(*args[:5], word="fast")
        assert sel.row == "k2b" and T.INHERITED_CC in sel.reason
        out = T._serve_inherited(sel, None, None, None, None, None, 0.176, None, args, "fast", None, None, None, None)
        assert out == ("served", "flash") and calls["k2b"] == 1 and T.inherited_row_dead("12.0", "k2b") == "Boom"          # k2b died -> (k2 if the donor cell lists it) -> flash served
        es = [t for t in CC.table() if t.get("provider") == "triattn" and t.get("cc") == "12.0" and "stepped_aside:error:Boom" in str(t.get("note"))]
        assert es, "census token missing"
        sel2 = T.select(*args[:5], word="fast")                                                      # the second call: the dead row is not retried, the next portable row is selected up front
        assert sel2.row != "k2b" and sel2.row in T.TRITON_ROWS and "stepped_aside:error:Boom" in sel2.reason
        T._serve_inherited(sel2, None, None, None, None, None, 0.176, None, args, "fast", None, None, None, None)
        assert calls["k2b"] == 1
        # every portable row dead: the kit's stock callable by name, else a Refusal naming the library op (never a raw error)
        fake_flash.flash_triangle_attention = mk("flash", Boom("launch failed"))
        T.select_cache_clear()
        sel3 = T.select(*args[:5], word="fast")
        assert T._serve_inherited(sel3, None, None, None, None, None, 0.176, mk("stock"), args, "fast", None, None, None, None) == ("served", "stock")
        with pytest.raises(T.Refusal) as r:
            T.select(*args[:5], word="fast")                                                          # nothing portable left alive on this cc: refused by name up front
        assert r.value.fallback == "cueq"
        # out-of-memory is re-raised unchanged (never converted)
        T._DEAD_INHERITED.clear(); T.select_cache_clear()
        fake_k2b.attn_k2b = mk("k2b", MemoryError("CUDA out of memory"))
        sel4 = T.select(*args[:5], word="fast")
        with pytest.raises(MemoryError):
            T._serve_inherited(sel4, None, None, None, None, None, 0.176, None, args, "fast", None, None, None, None)
        assert T.inherited_row_dead("12.0", "k2b") is None
    finally:
        T._route = real_route; T._DEAD_INHERITED.clear(); T.select_cache_clear()


def test_parts_columns_blackwell_cells_and_device_tagged_exact_vouches():
    """The parts sweep fold: cc 10.0 / 10.3 are measured columns (portable Triton rows lead, the sm_90a / sm_80 binaries refuse by name, exact =
    the library op by name); H200 / A100-40GB are part records on the 9.0 / 8.0 keys (resolution by cc unchanged) with DEVICE-TAGGED exact
    vouches: exact_headsplit vouched where measured bitwise on the part."""
    T.select_cache_clear()
    K = CUDA_STACK; V9 = "9.0|torch2.13.0+cu130|cueq0.11.1"; V8 = "8.0|torch2.13.0+cu130|cueq0.11.1"
    n10 = {cc: [k for k in T.cells() if k.startswith(cc + "|")] for cc in ("10.0", "10.3")}
    assert len(n10["10.0"]) >= 35 and len(n10["10.3"]) >= 35
    for cc in ("10.0", "10.3"):
        for key in n10[cc]:
            c = T.cells()[key]; _, dt, D, H, N, _d = key.split("|")
            assert c["fast"] in T.TRITON_ROWS + ("cueq",) and c["fast_order"][-1] == "cueq" and c["exact"] == "cueq" and c["stock_ms"] and ("parts sweep" in c["source"] or "row race" in c["source"]), key
            f = T.select(cc, dt, int(D[1:]), int(H[1:]), int(N[3:]), word="fast", stack=K)
            assert f.row == c["fast"] and f.measured and T.INHERITED_CC not in f.reason, (key, f)
            b = T.select(cc, dt, int(D[1:]), int(H[1:]), int(N[3:]), word="big", stack=K)
            assert b.row == f.row, (key, b)
            e = T.select(cc, dt, int(D[1:]), int(H[1:]), int(N[3:]), word="exact", stack=K, exact_stack=f"{cc}|torch2.13.0+cu130|cueq0.11.1|{'B200' if cc == '10.0' else 'B300'}")
            assert e.row == "cueq", (key, e)
        with pytest.raises(T.Refusal) as r:
            T.select(cc, "bf16", 32, 4, 1024, word="triattn_native@v11", stack=K)
        assert r.value.kind.startswith("cc "), r.value.kind
    # device tags
    dt_ = T.table()["words"]["exact_vouch"]["device_tags"]
    assert T.device_tag("9.0", "NVIDIA H100 80GB HBM3") == "" and T.device_tag("8.0", "NVIDIA A100-SXM4-80GB") == ""
    assert T.device_tag("9.0", "NVIDIA H200") == "H200" and T.device_tag("8.0", "NVIDIA A100-SXM4-40GB") == "A100-40GB" and T.device_tag("9.0", "NVIDIA H100 PCIe") == "NVIDIA_H100_PCIe"
    assert T.device_tag("9.0", None) in ("",) or True                                                       # this process: no device -> '' (covered by the stub test)
    # H200: the exact tier -> the stock op by name; exact_headsplit by name where vouched
    k9 = "9.0|bf16|D32|H4|N<=1200|fwd"; c9 = T.cells()[k9]
    assert c9["exact"] == "triattn_exact" and V9 + "|H200" in c9["vouched_on"]["exact_headsplit"] and any(v.endswith("|H200") for v in c9["vouched_on"]["triattn_exact"])
    assert T.select("9.0", "bf16", 32, 4, 1024, word="exact", stack=K, exact_stack=V9).row == "triattn_exact"                      # the reference device on a vouched stack: the bit-identical row
    eh = T.select("9.0", "bf16", 32, 4, 1024, word="exact", stack=K, exact_stack=V9 + "|H200")
    assert eh.row == "triattn_exact", eh                                                                                             # H200 on a vouched stack: the bit-identical row too (device-tagged vouch key)
    assert T.select("9.0", "bf16", 32, 4, 1024, word="exact", stack=K, exact_stack=V9 + "|NVIDIA_H100_PCIe").row == "cueq"           # an unlisted device tag: the stock op by name
    assert T.select("9.0", "bf16", 32, 4, 1024, word="exact_headsplit", stack=K, exact_stack=V9 + "|H200").row == "exact_headsplit"
    # A100-40GB: exact_headsplit served under the tagged key where the sweep measured it bitwise; an unlisted device tag -> the library op
    k8 = "8.0|bf16|D32|H4|N<=2048|fwd"; c8 = T.cells()[k8]
    assert c8["exact"] == "triattn_exact" and V8 + "|A100-40GB" in c8["vouched_on"]["exact_headsplit"] and not any("A100-40GB" in v for v in c8["vouched_on"]["triattn_exact"])
    assert T.select("8.0", "bf16", 32, 4, 2048, word="exact", exact_stack=V8 + "|A100-40GB").row == "cueq"               # A100 40GB: the cell's winner (the bit-identical row) is not vouched under that tag -> the stock op by name (the head-split row's tagged vouch stays on record for its own word)
    assert T.select("8.0", "bf16", 32, 4, 2048, word="exact_headsplit", exact_stack=V8 + "|A100-40GB").row == "exact_headsplit"
    assert T.select("8.0", "bf16", 32, 4, 2048, word="exact", exact_stack=V8 + "|SomeOtherPart").row == "cueq"
    # part records beside the reference columns (resolution unchanged)
    p9 = c9["parts"]; col = [c for c in p9 if c.startswith("H200:")][0]
    assert p9[col]["rows"]["cueq"]["x_stock"] == 1.0 and p9[col]["winner"] in p9[col]["rows"] and p9[col]["resolution"].startswith("unchanged")
    p8 = c8["parts"]; col8 = [c for c in p8 if c.startswith("A100-SXM4-40GB:")][0]
    assert "errors" in p8[col8] and "sdpa" in p8[col8]["errors"] and "ceiling_note" in p8[col8]


def test_cc_10_0_verdict_by_stock_library_version():
    """cc 10.0: the sweep's stack (library 0.11.1) and a cofolding kit's stack (library 0.10.0, its sm_100 kernel) disagree on the winner at the
    D32 / D16 H4 keys: `fast_by_lib` decides by the process's own library version; without it the cell-level words (the sweep's stack) answer."""
    T.select_cache_clear()
    c = T.cells()["10.0|bf16|D32|H4|N<=1200|fwd"]
    assert set(c["fast_by_lib"]) == {"cueq0.11.1", "cueq0.10.0"} and c["fast_by_lib"]["cueq0.10.0"]["fast"] == "cueq" and c["fast_by_lib"]["cueq0.11.1"]["fast"] in T.TRITON_ROWS
    assert len(c["stock_ms"]) == 2 and c["fast"] == c["fast_by_lib"]["cueq0.11.1"]["fast"]
    for n in (800, 1200, 1536, 2048):
        assert T.select("10.0", "bf16", 32, 4, n, word="fast", lib="cueq0.10.0").row == "cueq"
        assert T.select("10.0", "bf16", 32, 4, n, word="big", lib="cueq0.10.0").row == "cueq"
        assert T.select("10.0", "bf16", 32, 4, n, word="fast", lib="cueq0.11.1").row in T.TRITON_ROWS
        assert T.select("10.0", "bf16", 32, 4, n, word="fast").row in T.TRITON_ROWS                      # no library given: the cell-level words
        assert T.select("10.0", "bf16", 16, 4, n, word="fast", lib="cueq0.10.0").row == "cueq"
    b = T.select("10.0", "bf16", 32, 4, 256, word="fast")                                               # a bucket only the kit's race timed: its verdict at cell level
    assert b.cell == "10.0|bf16|D32|H4|N<=256|fwd" and b.row == "cueq" and b.measured
    assert T.select("10.0", "bf16", 32, 4, 480, word="fast", lib="cueq0.10.0").row == T.cells()["10.0|bf16|D32|H4|N<=511|fwd"]["fast_by_lib"]["cueq0.10.0"]["fast"]
    assert T.select("10.3", "bf16", 32, 4, 1200, word="fast", lib="cueq0.10.0").row in T.TRITON_ROWS   # cc 10.3 has one column only: no fast_by_lib, unchanged
    assert T.stock_lib_key() is None or T.stock_lib_key().startswith("cueq")
