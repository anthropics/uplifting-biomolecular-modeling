"""kernels/ln: the LayerNorm-family provider — its cell table, selection rules (default rule / opt-in tiers / EXACT FLOOR x1.00 / refusals by
name / capture safety), the forms (fp32 | bf16 | bf16w) and passes (fwd | bwd | fwdbwd), the module census and the carried sub-packages' byte
identity with the kit copies.  CPU only."""
import ast
import filecmp
import json
import os

import pytest

from opt_core import kernels
from opt_core.kernels import ln as L

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
CARDS = ("9.0", "8.0")
SIZES = (400, 800, 1200)
# every LayerNorm the release tree's torch engines run today: (dtype form, cell, widen, pass)
SHAPES = [("fp32", "pair_c128", False, "fwd"), ("bf16", "pair_c128", False, "fwd"), ("bf16", "pair_c128", True, "fwd"), ("fp32", "pair_c64", False, "fwd"),
          ("fp32", "pair_c256", False, "fwd"), ("bf16", "pair_c256", False, "fwd"), ("fp32", "pair_c384", False, "fwd"), ("bf16", "pair_c384", False, "fwd"),
          ("fp32", "single_c384", False, "fwd"), ("bf16", "single_c384", False, "fwd"), ("fp32", "atom_c128", False, "fwd"), ("bf16", "atom_c128", False, "fwd"),
          ("fp32", "dit_c768", False, "fwd"), ("bf16", "dit_c768", False, "fwd"), ("bf16", "lnlinear_c384", False, "fwd"),
          ("bf16", "pairbwd_c256_rowmajor", False, "bwd"), ("bf16", "pairbwd_c256_rowmajor_res", False, "bwd"), ("bf16", "pairbwd_c256_cmajor", False, "bwd"),
          ("bf16", "pair_c256", False, "fwdbwd"), ("fp32", "pair_c256", False, "fwdbwd")]


def test_table_schema_rows_words():
    t = L.table()
    assert t["schema"] == "ln_cells/v1"
    for k in ("op", "key_grammar", "evidence", "stacks", "rows", "tiers", "default_rule", "exact_rule", "capture_rule", "cells", "coverage", "capture_checks"):
        assert k in t, k
    assert set(L.ROW_NAMES) == set(t["rows"]), sorted(set(L.ROW_NAMES) ^ set(t["rows"]))
    for name, row in t["rows"].items():
        for k in ("class", "capture_safe", "fallback"):
            assert k in row, (name, k)
        assert row["class"] in ("fast", "exact", "stock"), name
    assert {n for n, r in t["rows"].items() if r["class"] == "stock"} == set(L.STOCK_ROWS)
    assert {n for n, r in t["rows"].items() if r["class"] == "exact"} == set(L.EXACT_ROWS) == {"exactln", "ef2_ln_bwd_dx"}
    assert {n for n, r in t["rows"].items() if not r["capture_safe"]} == set(L.CAPTURE_UNSAFE_ROWS) == {"fast_layernorm"}
    assert len(t["cells"]) >= 400
    assert {"9.0", "8.0"} <= {k.split("|")[0] for k in t["cells"]} <= {"9.0", "8.0", "10.0", "10.3"}
    for key, cell in t["cells"].items():
        cc, dt, cw, n, timing, pas = key.split("|")
        assert dt in L.FORM_WORDS and cw in L.CELL_WORDS and n.startswith("N<=") and timing in ("eager", "graph") and pas in L.PASS_WORDS, key
        assert cell["ref_stack"] in cell["stacks"] and cell["ref_stack"] in t["stacks"], key
        for w in ("fast", "exact"):
            arm = cell.get(w)
            assert arm is None or L.split_word(arm)[0] in L.ROW_NAMES, (key, w, arm)
        if cell.get("exact") and L.split_word(cell["exact"])[0] in L.EXACT_ROWS:
            assert cell["exact_x"] >= 1.0 and cell["exact_parity_flag"].startswith("bitwise_vs:"), key   # EXACT FLOOR x1.00
        if timing == "graph":
            assert L.split_word(cell["fast"])[0] not in L.CAPTURE_UNSAFE_ROWS, key


@pytest.mark.parametrize("cc", CARDS)
def test_every_shape_the_tree_runs_lands_on_a_measured_cell_or_a_named_stock_row(cc):
    for dt, cw, widen, pas in SHAPES:
        for n in SIZES:
            for word in ("fast", "exact"):
                sel = L.select(cc, dt, cw, n, word=word, widen=widen, pass_=pas, has_esm=True)
                assert sel.row in L.ROW_NAMES
                if sel.cell is None:
                    assert sel.row in L.STOCK_ROWS, (cc, dt, cw, n, word, sel)
                else:
                    assert sel.cell in L.table()["cells"]


def test_default_rule_a_row_word_serves_that_row():
    for word, row, variant in (("exactln", "exactln", None), ("exactln:triton", "exactln", "triton"), ("fastln", "fastln", None), ("ln_rows", "ln_rows", None),
                               ("rfd", "rfd", None), ("dtk_ln", "dtk_ln", None), ("aten", "aten", None), ("fast_layernorm", "fast_layernorm", None)):
        sel = L.select("9.0", "fp32", "pair_c128", 800, word=word)
        assert (sel.row, sel.variant, sel.word) == (row, variant, word), (word, sel)
    w = L.select("9.0", "bf16", "pair_c128", 800, word="exactln", widen=True)          # the autocast form resolves the row word to its widen variant
    assert (w.row, w.variant) == ("exactln", "widen")
    assert L.select("9.0", "bf16", "pairbwd_c256_rowmajor_res", 800, word="ef2_ln_bwd_dx", pass_="bwd").row == "ef2_ln_bwd_dx"
    assert L.select("9.0", "bf16", "lnlinear_c384", 800, word="ln_proj_ln_linear").row == "ln_proj_ln_linear"


def test_tier_words_are_the_measured_winners_and_the_exact_floor():
    t = L.table()["cells"]
    # N^2 pair rows, fp32, H100: the exact replica beats ATen -> exact = exactln (bitwise_vs aten, x >= 1); fast = a Triton row
    H213, A213 = "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1", "A100:torch2.13.0+cu130/3.7.1/cueq0.11.1"   # EXACT VOUCH IS STACK-SPECIFIC
    x = L.select("9.0", "fp32", "pair_c128", 800, word="exact", stack=H213)
    assert x.row == "exactln" and x.cls == "bitwise" and x.exact_vs == "aten" and x.x_stock >= 1.0 and x.stack_measured
    assert H213 in t[x.cell]["vouched_on"]["exactln"]
    u = L.select("9.0", "fp32", "pair_c128", 800, word="exact")                                  # stack not named -> ATen (the floor), by name
    assert u.row == "aten" and "exact vouch not recorded" in u.reason
    u2 = L.select("9.0", "fp32", "pair_c128", 800, word="exact", stack="H100:torch2.99.0+cu130/9.9.9/cueq0.0.0")   # a stack word no column / vouch names
    assert u2.row == "aten" and "exact vouch not recorded on H100:torch2.99.0" in u2.reason
    v2 = L.select("9.0", "fp32", "pair_c128", 800, word="exact", stack="H100:torch2.10.0+cu128/3.6.0/cueq0.10.0")   # a measured + vouched column (the an AF3-family kit kits' stack)
    assert v2.row == "exactln" and v2.cls == "bitwise" and v2.stack_measured
    assert L.select("9.0", "fp32", "pair_c128", 800, word="exactln").row == "exactln"            # a row word is served as asked
    f = L.select("9.0", "fp32", "pair_c128", 800, word="fast")
    assert f.row in ("fastln", "ln_rows", "rfd", "dtk_ln", "exactln") and f.x_stock >= x.x_stock - 1e-9
    assert L.arm_word(f.row, f.variant) == t[f.cell]["fast"]
    # N-row single rows in eager: ATen wins -> exact = aten by name (EXACT FLOOR); under capture the graph cell's fused winner serves
    s = L.select("9.0", "fp32", "single_c384", 800, word="exact")
    assert s.row == "aten"
    g = L.select("9.0", "fp32", "single_c384", 800, word="fast", capture=True)
    assert g.cell.split("|")[4] == "graph" and g.row not in L.CAPTURE_UNSAFE_ROWS
    # the autocast form: exact = the widen variant where it is at or above the statement
    w = L.select("9.0", "bf16", "pair_c128", 800, word="exact", widen=True, stack=H213)
    assert (w.row, w.variant) in (("exactln", "widen"), ("aten_autocast", None))
    # dx-only backward, residual folded: the carried row is the exact winner vs the vendored backward
    b = L.select("9.0", "bf16", "pairbwd_c256_rowmajor_res", 800, word="exact", pass_="bwd", has_esm=True, stack="H100:torch2.11.0+cu128/3.6.0/cueq0.10.0")
    assert b.row == "ef2_ln_bwd_dx" and b.exact_vs == "esm_vendored_bwd"
    assert L.select("9.0", "bf16", "pairbwd_c256_rowmajor_res", 800, word="exact", pass_="bwd", has_esm=True, stack=H213).row == "esm_vendored_bwd"   # not vouched there
    # the c=384 pair cells on cc 8.0 (measured): fp32 exact = exactln (bitwise ATen, above its speed); bf16 exact = the statement
    e8 = L.select("8.0", "fp32", "pair_c384", 800, word="exact", stack=A213)                   # measured on cc 8.0 (the c=384 pair cells): exactln bitwise ATen and above its speed
    assert (e8.row, e8.size_measured) == ("exactln", True) and e8.x_stock >= 1.0
    assert L.select("8.0", "bf16", "pair_c384", 800, word="exact").row == "aten"        # bf16: no exact-class row at or above ATen there -> the statement
    # prefer narrows the fast tier
    p = L.select("9.0", "fp32", "dit_c768", 800, word="fast", prefer=("fastln", "rfd"))
    assert p.row in ("fastln", "rfd")
    # cell_for_rows: the family from a live row count
    assert L.cell_for_rows(800 * 800, 800, 128) == "pair_c128" and L.cell_for_rows(800, 800, 384) == "single_c384"
    assert L.cell_for_rows(8 * 800, 800, 128) == "atom_c128" and L.cell_for_rows(5 * 800, 800, 768) == "dit_c768"


def test_refusals_are_by_name_with_the_fallback_row():
    with pytest.raises(L.Refusal) as ei:
        L.select("9.0", "fp32", "pair_c128", 800, word="fast_layernorm", capture=True)     # capture-UNSAFE by name
    assert "capture" in ei.value.kind and ei.value.fallback
    assert L.capture_unsafe("fast_layernorm") and not L.capture_unsafe("exactln")
    with pytest.raises(L.Refusal):
        L.select("9.0", "fp32", "pair_c100", 800, word="exactln:triton", C=100)           # the triton variant serves C 64 / 128 only
    with pytest.raises(L.Refusal):
        L.select("9.0", "fp32", "pair_c128", 800, word="exactln", has_nvrtc=False)         # no NVRTC bindings: refused by name (fallback aten)
    with pytest.raises(L.Refusal):
        L.select("9.0", "fp32", "pair_c128", 800, word="ef2_ln_bwd_dx")                   # a backward row asked on the forward pass
    with pytest.raises((L.Refusal, ValueError)):
        L.select("9.0", "fp32", "pair_c128", 800, word="no_such_row")
    for cw in L.CELL_WORDS:                                                            # a tier word never raises for a listed cell
        pas = "bwd" if cw.startswith("pairbwd") else "fwd"
        assert L.select("8.0", "bf16", cw, 1200, word="fast", pass_=pas, has_esm=True).row in L.ROW_NAMES


def test_int64_addressing_of_every_served_kernel():
    """Index audit: every kernel this face launches forms its row offset (row x width, up to N^2 x 768) in int64 — exactln's CUDA replica
    (long long row index), its Triton form, fastln, ln_rows / ln_linear (ln_proj), rfd, dtk ln_modulate, the dx-only backward — and ATen is
    int64-indexed; so the face has no int32 size refusal.  Pin the int64 casts in the carried sources so an update cannot silently drop them."""
    K = kernels.KERNELS_DIR
    for rel, needle in (("ln/exactln/exactln_fwd.cu", "(long long)blockIdx.x"), ("ln/exactln/triton_ln.py", ".to(tl.int64) * stride_row"),
                        ("ln/fastln/fastln.py", "rows.to(tl.int64)"), ("ln/ef2/ef2_fused_ln.py", "tl.program_id(0).to(tl.int64)"),
                        ("rfd_layernorm.py", "tl.program_id(0).to(tl.int64)"), ("ln_proj.py", "r64 = rows.to(tl.int64)"), ("dtk_kernels.py", "row = tl.program_id(0).to(tl.int64)")):
        assert needle in open(os.path.join(K, rel), encoding="utf-8").read(), rel


def test_describe_coverage_rows_for_kit_are_plain_data():
    sel = L.select("9.0", "fp32", "pair_c128", 800, word="exact", stack="H100:torch2.13.0+cu130/3.7.1/cueq0.11.1")
    assert "pair_c128" in L.describe(sel) and "exactln" in L.describe(sel)
    assert set(L.rows_for_kit(["exactln", "fastln"])) >= {"exactln", "fastln"}
    json.dumps(L.coverage("8.0", [("fp32", "pair_c128", 800), ("bf16", "pair_c128", 800, "fwd", True)], word="exact"))


def test_face_imports_only_the_standard_library_at_module_level():
    tree = ast.parse(open(os.path.join(CORE_DIR, "opt_core", "kernels", "ln", "__init__.py"), encoding="utf-8").read())
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:                                           # the one relative import: the cell-coverage census (itself standard library only)
                assert [a.name for a in node.names] == ["cell_census"], [a.name for a in node.names]
                continue
            top.append((node.module or "").split(".")[0])
    assert set(top) <= {"json", "os", "re", "collections"}, top


def test_meta_census_and_the_carried_sub_packages():
    assert "ln" in kernels.names()
    meta = kernels.sums("ln")
    assert meta["kind"] == "package" and meta["version"] == "1.0.0"
    assert set(meta["runtime_imports"]) == {"ln_proj", "rfd_layernorm", "dtk_kernels"}
    files = set(meta["files"])
    for f in ("__init__.py", "LN_CELLS.json", "exactln/__init__.py", "exactln/exactln_fwd.cu", "exactln/triton_ln.py", "fastln/__init__.py", "fastln/fastln.py",
              "ef2/__init__.py", "ef2/ef2_fused_ln.py"):
        assert f in files, f


def _kit_copies_by_name(name):
    """Every file called `name` in the release tree outside common/ (no engine is named here)."""
    core = os.path.join(RELEASE_TREE, "common")
    hits = []
    for root, dirs, files in os.walk(RELEASE_TREE):
        dirs[:] = [x for x in dirs if x not in ("__pycache__", ".git")]
        if root.startswith(core):
            continue
        if name in files:
            hits.append(os.path.join(root, name))
    return hits


def _sha256(path):
    import hashlib
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


@pytest.mark.parametrize("carried", sorted(json.load(open(os.path.join(kernels.KERNELS_DIR, "META", "ln.json"), encoding="utf-8"))["carried_digests"]))
def test_carried_files_have_their_recorded_digest_and_match_kit_copies_by_digest(carried):
    """The carried file has the digest META records for it (the core's own consistency: always checked), and a kit copy found by NAME anywhere in the
    release tree with the SAME digest witnesses the byte identity.  Kit-side moves and edits never fail this gate: no copy in the checkout, or only
    copies with another digest (the kit edited its file after the carry, or an unrelated file of that name), SKIP with the reason named -- the provider
    serves the carried bytes either way."""
    meta = json.load(open(os.path.join(kernels.KERNELS_DIR, "META", "ln.json"), encoding="utf-8"))
    mine_path = os.path.join(kernels.KERNELS_DIR, "ln", carried)
    assert os.path.isfile(mine_path), mine_path
    mine = _sha256(mine_path)
    assert meta["carried_digests"][carried] == mine, (carried, meta["carried_digests"][carried], mine)   # the carried copy changed without its META digest: re-carry = copy + record
    if not os.path.isdir(RELEASE_TREE):
        pytest.skip("kits not beside common/")
    copies = _kit_copies_by_name(os.path.basename(carried))
    if not copies:
        pytest.skip("no kit copy of %s in this checkout (byte identity as recorded in META carried_digests)" % carried)
    same = [b for b in copies if _sha256(b) == mine]
    if not same:
        shown = ", ".join(os.path.relpath(b, RELEASE_TREE) for b in copies[:5]) + (" ..." if len(copies) > 5 else "")
        pytest.skip("kit copy ahead of common (or unrelated files of that name): %s; carried digest %s.. -- the provider serves the carried bytes until re-carried" % (shown, mine[:16]))
    for b in same:
        assert filecmp.cmp(mine_path, b, shallow=False), (carried, os.path.relpath(b, RELEASE_TREE))


def test_bf16o_form_is_opt_in_and_routes_to_the_low_precision_output_rows():
    """The compiled-extension form (bf16 x, fp32 gamma/beta, fp32 statistics, bf16 OUT) is its own form word `bf16o`, reached only through
    ``out='bf16'`` (select) / ``out_dtype=torch.bfloat16`` (layer_norm): every existing word on the fp32 / bf16 / bf16w forms answers exactly as
    before (hazard 11), and on bf16o the rows are: fastln -> its lp kernel (fastln:lp), exactln -> exactln:widen with the cast fused (bitwise the
    statement aten_autocast + .to(bfloat16)), dtk_ln, the statement aten_autocast; ln_rows / rfd / plain aten refuse BY NAME naming aten_autocast."""
    assert "bf16o" in L.FORM_WORDS and L.VARIANTS["fastln"] == ("lp",)
    assert L.dtype_word("bf16", widen=True, out="bf16") == "bf16o" and L.dtype_word("bf16", widen=True) == "bf16w" and L.dtype_word("bf16") == "bf16"
    assert L.dtype_word("torch.bfloat16", True, "torch.bfloat16") == "bf16o" and L.dtype_word("fp32", True, "bf16") == "fp32"
    for cc in ("9.0", "8.0"):
        f = L.select(cc, "bf16", "pair_c384", 800, word="fastln", widen=True, out="bf16")
        assert (f.row, f.variant, f.fallback) == ("fastln", "lp", "aten_autocast")
        e = L.select(cc, "bf16", "pair_c384", 800, word="exactln", widen=True, out="bf16")
        assert (e.row, e.variant, e.exact_vs) == ("exactln", "widen", "aten")
        assert L.select(cc, "bf16", "pair_c384", 800, word="dtk_ln", widen=True, out="bf16").row == "dtk_ln"
        assert L.select(cc, "bf16", "pair_c384", 800, word="aten_autocast", widen=True, out="bf16").row == "aten_autocast"
        for w in ("ln_rows", "rfd", "aten"):
            with pytest.raises(L.Refusal) as ei:
                L.select(cc, "bf16", "pair_c384", 800, word=w, widen=True, out="bf16")
            assert ei.value.fallback == "aten_autocast" and ei.value.kind.startswith("dtype:bf16o")
        for tier in ("fast", "exact"):                                             # tier words: the measured cell's winner where measured, else the statement by name
            t = L.select(cc, "bf16", "pair_c384", 800, word=tier, widen=True, out="bf16")
            assert t.row in L.ROW_NAMES and (t.size_measured or t.row == "aten_autocast")
    # hazard 11: the other forms' words are unchanged by the new form
    assert L.select("9.0", "bf16", "pair_c128", 800, word="fast", widen=True).row in ("exactln", "fastln", "ln_rows", "rfd", "dtk_ln", "aten_autocast")
    x = L.select("9.0", "bf16", "pair_c128", 800, word="exactln", widen=True); assert (x.row, x.variant) == ("exactln", "widen")
    with pytest.raises(L.Refusal) as ei:
        L.select("9.0", "bf16", "pair_c128", 800, word="fastln", widen=True)       # bf16w: the carried kernel stores x's dtype -> refused by name, the lp variant named
    assert ei.value.fallback == "fastln:lp"
    p = L.select("9.0", "fp32", "pair_c128", 800, word="fastln"); assert (p.row, p.variant) == ("fastln", None) and p.x_stock and p.x_stock > 1
    assert L.select("9.0", "fp32", "pair_c128", 800, word="fastln:lp").variant == "lp"


def test_exactln_binds_nvrtc_through_ctypes_when_cuda_python_is_absent():
    """exactln's four cuda.bindings imports resolve through _bindings(): cuda.bindings when importable, else cubind (ctypes over libcuda +
    libnvrtc) -- so an image without the cuda-python wheel is served.  cubind exposes every driver / NVRTC entry point the carried module calls,
    with cuda.bindings' (err, *values) convention, and imports on a machine with no GPU (nothing loads until first use)."""
    import importlib.util, re
    root = os.path.join(kernels.KERNELS_DIR, "ln", "exactln")
    src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
    assert "from cuda.bindings import" not in src and src.count("_bindings(\"nvrtc\")") >= 2 and "_bindings(\"driver\")" in src and "OPT_CORE_EXACTLN_BINDINGS" in src
    spec = importlib.util.spec_from_file_location("cubind_t", os.path.join(root, "cubind.py")); cb = importlib.util.module_from_spec(spec); spec.loader.exec_module(cb)
    text = open(os.path.join(root, "cubind.py"), encoding="utf-8").read()
    assert "import torch" not in text.split("def _nvrtc_candidates")[0]           # standard library at import; torch only consulted for the CUDA version at load time
    for call in sorted(set(re.findall(r"\bcu\.(cu[A-Za-z]+)\(", src))):
        assert hasattr(cb.driver, call), call
    for call in sorted(set(re.findall(r"\bnvrtc\.(nvrtc[A-Za-z]+)\(", src))):
        assert hasattr(cb.nvrtc, call), call
    assert isinstance(cb.available(), bool) and set(cb.describe()) == {"driver", "nvrtc", "nvrtc_version"}
    buf = b" " * 6; cb._Nvrtc._fill(buf, b"cubin!"); assert buf == b"cubin!"      # output buffers are filled in place, as cuda.bindings does
    ba = bytearray(3); cb._Nvrtc._fill(ba, b"ptx"); assert bytes(ba) == b"ptx"
    err = cb._Err(500, "CUDA_ERROR_NOT_FOUND"); assert int(err) == 500 and "NOT_FOUND" in str(err) and int(cb._OK) == 0


def test_fastln_lp_is_the_carried_kernel_with_fp32_arithmetic_and_a_named_output_dtype():
    """lpout.py (house-written) differs from the carried fastln.py kernel only where intended: loads widened to fp32, the store cast to the output
    dtype, the output allocated in out_dtype; the tiling rule (_cfg) is the carried one verbatim; int64 row offsets kept."""
    root = os.path.join(kernels.KERNELS_DIR, "ln", "fastln")
    lp = open(os.path.join(root, "lpout.py"), encoding="utf-8").read(); fl = open(os.path.join(root, "fastln.py"), encoding="utf-8").read()
    cfg = fl[fl.index("def _cfg(N):"):fl.index("def fast_layer_norm_triton")]
    assert cfg.strip() in lp
    assert ".to(tl.float32)" in lp and "y.to(Y.dtype.element_ty)" in lp and "rows.to(tl.int64)" in lp and "torch.empty(xc.shape, dtype=out_dtype" in lp
    assert "tl.div_rn(1.0, tl.sqrt_rn(var + eps))" in lp and "tl.div_rn(1.0, tl.sqrt_rn(var + eps))" in fl
    meta = kernels.sums("ln")
    assert {"ln/exactln/__init__.py", "ln/exactln/cubind.py", "ln/fastln/lpout.py"} <= set(meta["not_byte_identical"])


def test_exactln_ships_its_cubins_and_serving_never_compiles_the_served_set():
    """QoL rule (no runtime building): exactln/prebuilt/ carries ONE CUBIN module per arch (sm_90, sm_80) with every (form, C, affine)
    instantiation the provider's cells serve -- forms fp32 / bf16 / bf16w / bf16o x C 64/128/256/384/768 x affine both|neither -- built
    from THIS exactln_fwd.cu with the module's own compile options (index.json: source sha256, options, bundle sha256, lowered names);
    _compile() consults the bundle first (facts prebuilt_hits) and compiles only what it lacks; OPT_CORE_EXACTLN_PREBUILT=0 forces compiling."""
    import hashlib, re
    root = os.path.join(kernels.KERNELS_DIR, "ln", "exactln")
    src = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
    assert "PREBUILT_ENV = \"OPT_CORE_EXACTLN_PREBUILT\"" in src and "pf = _prebuilt_function(expr, cc)" in src and '"prebuilt_hits"' in src
    idx = json.load(open(os.path.join(root, "prebuilt", "index.json")))
    assert idx["schema"] == "exactln_prebuilt/v1"
    assert idx["source_sha256"] == hashlib.sha256(open(os.path.join(root, "exactln_fwd.cu"), "rb").read()).hexdigest()      # built from the shipped source
    opts = re.search(r'tail = \[(.*?)\]\n', src).group(1)
    assert idx["opts"] == [o.strip().strip('b').strip('"') for o in opts.split(",")]                                        # ... with the module's options
    assert set(idx["bundles"]) == {"sm90", "sm80"}
    T = {"float32": "float", "bfloat16": "exactln::bf16_t"}
    forms = {"fp32": ("float32", "float32", "float32"), "bf16": ("bfloat16", "bfloat16", "bfloat16"), "bf16w": ("bfloat16", "float32", "float32"), "bf16o": ("bfloat16", "float32", "bfloat16")}
    m = re.search(r"def _rows_per_warp\(C: int\) -> int:\n(.*?)\n\n", src, re.S)                                    # the module's rows-per-warp rule, evaluated here
    ns = {}; exec("def _rows_per_warp(C):\n" + m.group(1), ns)
    for b in idx["bundles"].values():
        data = open(os.path.join(root, "prebuilt", b["file"]), "rb").read()
        assert hashlib.sha256(data).hexdigest() == b["sha256"] and len(data) == b["bytes"] and data[:4] == b"\x7fELF"
        for form, (tin, tpar, tout) in forms.items():
            for C in (64, 128, 256, 384, 768):
                for a in (3, 0):
                    expr = "exactln::exactln_fwd<%s, %s, %s, %d, %d, %d>" % (T[tin], T[tpar], T[tout], C, a, ns["_rows_per_warp"](C))
                    assert expr in b["kernels"], (b["file"], expr)
        assert b["n_kernels"] == len(b["kernels"]) == 40
    meta = kernels.sums("ln")
    assert {"ln/exactln/prebuilt/", "ln/exactln/build_prebuilt.py"} <= set(meta["not_byte_identical"])


LADDER_KEYS_SEEN = [   # (form, the binder's cell label, its n = x.shape[-2], timing): every kernels.ln call class an engine's clean-install smoke produced at
    ("bf16w", "dit_c768", 2000, "graph"), ("bf16w", "dit_c768", 4000, "eager"), ("bf16w", "dit_c768", 4000, "graph"),      # the 400 / 800-token rungs (H100)
    ("bf16w", "pair_c128", 3256, "eager"), ("bf16w", "pair_c128", 3256, "graph"), ("bf16w", "pair_c128", 6472, "eager"), ("bf16w", "pair_c128", 6472, "graph"),
    ("bf16w", "pair_c128", 16280, "eager"), ("bf16w", "pair_c128", 16280, "graph"), ("bf16w", "pair_c128", 32360, "eager"), ("bf16w", "pair_c128", 32360, "graph"),
    ("fp32", "atom_c128", 3256, "eager"), ("fp32", "atom_c128", 3256, "graph"), ("fp32", "atom_c128", 6472, "eager"), ("fp32", "atom_c128", 6472, "graph"),
    ("fp32", "pair_c128", 3256, "eager"), ("fp32", "pair_c128", 3256, "graph"), ("fp32", "pair_c128", 6472, "eager"), ("fp32", "pair_c128", 6472, "graph"),
]


def test_ladder_row_counts_are_cell_hits_by_rows_and_unknown_widths_are_named_fallbacks():
    """A generic LayerNorm binder passes n = x.shape[-2] (an ATOM count for atom tensors, samples x tokens for DiT rows) and rows = numel / C;
    cell_for_rows(rows, n, C) returns a CellWord carrying the rows and select() buckets BY ROWS: every call class above lands on a cell
    MEASURED at >= its rows on both cards (census cell_hit, or the stock row where the statement is the measured winner) -- never a
    beyond_measured neighbour; a width no family lists (16, 833) is the statement BY NAME (census named_fallback, not an uncovered cell)."""
    from opt_core import cell_census as CC
    for cc, stack in (("9.0", "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1"), ("8.0", "A100:torch2.13.0+cu130/3.7.1/cueq0.11.1")):
        for form, label, n, timing in LADDER_KEYS_SEEN:
            C = int(label.split("_c")[1]); rows = n                                 # B = 1: rows = n (a larger leading batch only moves up inside the same measured ranges)
            cell = L.cell_for_rows(rows, n, C)
            assert isinstance(cell, L.CellWord) and cell.rows == rows and str(cell) in L.CELL_WORDS
            CC.reset()
            sel = L.select(cc, "fp32" if form == "fp32" else "bf16", cell, n, word="exact", widen=(form == "bf16w"), stack=stack, capture=(timing == "graph"), C=C)
            e = CC.table()[-1]
            assert e["outcome"] in ("cell_hit", "stock"), (cc, form, label, n, timing, str(cell), e)
            assert sel.size_measured and L.cell_rows(sel.cell) >= rows, (cc, form, label, n, timing, sel)
        for form, C, n in (("bf16w", 16, 128), ("fp32", 833, 400), ("fp32", 833, 800)):
            assert L.cell_for_rows(n, n, C) is None
            CC.reset()
            sel = L.select(cc, "fp32" if form == "fp32" else "bf16", None, n, word="exact", widen=(form == "bf16w"), stack=stack, C=C)
            e = CC.table()[-1]
            assert sel.row in L.STOCK_ROWS and e["outcome"] == "named_fallback" and e["refused"].startswith("cells:no_family"), e
    # true token callers are unchanged: n_tokens buckets when no rows are given
    k = L.select("9.0", "fp32", "pair_c128", 800, word="exact", stack="H100:torch2.13.0+cu130/3.7.1/cueq0.11.1")
    assert k.cell == "9.0|fp32|pair_c128|N<=800|eager|fwd"
    k2 = L.select("9.0", "fp32", L.CellWord("pair_c128", 640000), 123456, word="exact", stack="H100:torch2.13.0+cu130/3.7.1/cueq0.11.1")
    assert k2.cell == k.cell                                                         # 640000 rows = the N 800 pair cell whatever n says


def test_plain_token_replay_keys_and_the_per_card_family_fallback_by_rows():
    """A census replay that passes the binder's n (an atom count) as TOKENS with the plain word lands inside on cc 9.0 (atom / dit / single cells
    reach 8 192 tokens there); bucketed by ROWS a family that stops below the rows on one card hands over to another family of the same width
    measured at >= the rows on that card (family_by_rows), never a beyond_measured neighbour."""
    from opt_core import cell_census as CC
    H = "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1"; A = "A100:torch2.13.0+cu130/3.7.1/cueq0.11.1"
    for n in (3256, 6472):
        CC.reset(); s = L.select("9.0", "fp32", "atom_c128", n, word="exact", stack=H)
        assert s.size_measured and CC.table()[-1]["outcome"] in ("cell_hit", "stock"), (n, s)
    CC.reset(); s = L.select("8.0", "bf16", L.CellWord("atom_c128", 32360), 32360, word="exact", widen=True, stack=A)
    assert s.size_measured and L.cell_rows(s.cell) >= 32360 and CC.table()[-1]["outcome"] in ("cell_hit", "stock")            # an atom cell that large, or the pair family by rows
    assert "family_by_rows:pair_c128" in s.reason or s.cell.split("|")[2] == "atom_c128"



def test_big_word_q35_rule_every_cell():
    """The literal tier word `big` is accepted for every cell and resolves by the table's big rule: the fast winner unless its recorded peak
    exceeds the stock arm's (2% + 16 MiB noise bound), then the fastest admissible arm within the bound (the stock arm included); cells
    without peak records: big == fast."""
    from opt_core.kernels import ln as L
    tab = L.table(); assert "big_rule" in tab and "16 MiB" in tab["big_rule"]
    n = 0
    for key, cell in tab["cells"].items():
        cc, dt, cw, nb, timing, pas = key.split("|"); N = int(nb[3:])
        ref = cell["ref_stack"]; sc = cell["stacks"][ref]
        assert "big_per_stack" in cell and cell["big"] == sc.get("big")
        fast, stock, pk = sc.get("fast"), sc.get("stock_arm"), sc.get("peak_mib", {})
        if fast and fast in pk and stock in pk and pk[fast] <= pk[stock] * 1.02 + 16:
            assert sc.get("big") == fast, key
        elif fast and (fast not in pk or stock not in pk):
            assert sc.get("big") == fast, key
        else:
            assert sc.get("big") is None or pk[sc["big"]] <= pk[stock] * 1.02 + 16, key
        n += 1
    assert n == len(tab["cells"])
    s = L.select("9.0", "bf16", "pair_c128", 800, word="big", widen=True); f = L.select("9.0", "bf16", "pair_c128", 800, word="fast", widen=True)
    assert s.word == "big" and s.row == f.row



def test_tier_floors_no_affine_eager_and_the_recorded_exact_floor():
    """LN_CELLS.json tier_floors: f01 = no-affine LayerNorm in eager mode below 327680 rows -> the statement row for tier words (row words never
    floored); f02 = the fp32 exact floor at 65536 rows, recorded only (the cells enforce it: exactln measures below ATen there)."""
    from opt_core.kernels import ln as L
    rows = {r["id"]: r for r in L.table()["tier_floors"]["rows"]}
    assert rows["f01"]["min_rows"] == 327680 and rows["f02"]["min_rows"] == 65536 and rows["f02"]["scope"].get("row") == "exactln"
    s = L.select("9.0", "fp32", L.CellWord("dit_c768", 4000), 800, word="fast", affine=False, rows=4000)
    assert s.row == "aten" and s.reason.startswith("tier_floor:f01")
    s = L.select("9.0", "fp32", L.CellWord("dit_c768", 400000), 800, word="fast", affine=False, rows=400000)
    assert not s.reason.startswith("tier_floor")
    s = L.select("9.0", "fp32", L.CellWord("dit_c768", 4000), 800, word="fast", rows=4000)                     # affine: no floor
    assert not s.reason.startswith("tier_floor")
    s = L.select("9.0", "fp32", L.CellWord("dit_c768", 4000), 800, word="fast", affine=False, rows=4000, timing="graph")
    assert not s.reason.startswith("tier_floor")
    s = L.select("9.0", "fp32", L.CellWord("dit_c768", 4000), 800, word="fastln", affine=False, rows=4000)   # row words never floored
    assert s.row == "fastln"
    for cc in ("9.0", "8.0"):                                                                                # f02 is what the cells say already
        for cw, N in (("single_c384", 800), ("dit_c768", 800), ("atom_c128", 800)):
            e = L.select(cc, "fp32", cw, N, word="exact")
            assert e.row == "aten", (cc, cw, e.row)


def test_fast_layernorm_ext_row_wiring():
    """Row fast_layernorm_ext (the trunks' stream-corrected fused LayerNorm extension, carried under kernels/ln/rows/fast_layernorm_ext/ by
    its producing kit's move): the word is registered, aliases the measured arm fast_layernorm, capture-safe, admits fp32 / bf16 / bf16o (not
    bf16w); until the bytes land every selection refuses `not_landed` BY NAME with the bf16o fallback fastln:lp / fp32 aten; once landed:
    digests == manifests, no_prebuilt:<key> on other stacks."""
    import hashlib, os
    from opt_core.kernels import ln as L
    from opt_core.kernels.ln import ext_loader as S
    assert "fast_layernorm_ext" in L.ROW_NAMES and L.ARM_ALIASES["fast_layernorm_ext"] == "fast_layernorm" and "fast_layernorm_ext" not in L.CAPTURE_UNSAFE_ROWS
    row = L.table()["rows"]["fast_layernorm_ext"]; assert row["class"] == "fast" and row["capture_safe"] is True and row["exact_vs"].startswith("fast_layernorm")
    assert L.carried_module("fast_layernorm_ext") is S and S.ROWS_DIR.endswith(os.path.join("kernels", "ln", "rows", "fast_layernorm_ext"))
    st213, st210 = "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1", "H100:torch2.10.0+cu128/3.6.0/cueq0.10.0"
    try:
        L.select("9.0", "bf16", "pair_c128", 800, word="fast_layernorm_ext", widen=True, stack=st213); raise AssertionError("bf16w must refuse")
    except L.Refusal as r:
        assert r.kind.startswith("dtype:bf16w") and r.fallback == "aten_autocast"
    if not S.landed():
        assert S.keys() == [] and S.available() == (False, "not_landed")
        for kw, fb in ((dict(widen=True, out="bf16", stack=st213), "fastln:lp"), (dict(stack=st213), "aten")):
            try:
                L.select("9.0", "bf16" if kw.get("widen") else "fp32", "pair_c128", 800, word="fast_layernorm_ext", **kw); raise AssertionError("expected not_landed")
            except L.Refusal as r:
                assert r.kind.startswith(("not_landed", "no_prebuilt")), r.kind
                assert r.fallback == fb, (r.kind, r.fallback)
        return
    for k in S.keys():
        m = S.manifest(k); assert os.path.isfile(m["_so"]) and S.available(k) == (True, k)
        if m.get("so_sha256"):
            assert hashlib.sha256(open(m["_so"], "rb").read()).hexdigest() == m["so_sha256"]
    s = L.select("9.0", "bf16", "pair_c128", 800, word="fast_layernorm_ext", widen=True, out="bf16", stack=st213); assert s.row == "fast_layernorm_ext" and s.capture_safe
    try:
        L.select("9.0", "bf16", "pair_c128", 800, word="fast_layernorm_ext", widen=True, out="bf16", stack=st210); raise AssertionError("expected no_prebuilt")
    except L.Refusal as r:
        assert r.kind.startswith("no_prebuilt:") and r.fallback == "fastln:lp"



def test_tier_words_serve_fast_layernorm_ext_where_the_extension_arm_wins(monkeypatch):
    """Cells whose winner is the trunk-extension arm (fast_layernorm): a tier word serves row fast_layernorm_ext when its prebuilt loads on the
    caller's stack (simulated landed here), else the arm by name as before; vouched_on lists the four measured stacks."""
    from opt_core.kernels import ln as L
    from opt_core.kernels.ln import ext_loader as S
    st271_a, st271_h, st210 = "A100:torch2.7.1+cu126/3.3.1/cueq0.10.0", "H100:torch2.7.1+cu126/3.3.1/cueq0.10.0", "H100:torch2.8.0+cu128/3.4.0/cueq0.10.0"   # the third: a stack with NO cells of its own (the sibling-column path); torch 2.10.0+cu128/3.6.0 has its own measured cells since 0.5.128.1
    cell = L.table()["cells"]["8.0|bf16o|pair_c128|N<=400|eager|fwd"]
    assert cell["stacks"][st271_a]["fast"] == "fast_layernorm" and st271_a in cell["vouched_on"]["fast_layernorm_ext"]
    L._SELECT_CACHE.clear() if hasattr(L, "_SELECT_CACHE") else None
    if not S.landed():                                                         # before the bytes land: the cell's NEXT measured row (never the arm's name / the statement first)
        s = L.select("8.0", "bf16", "pair_c128", 400, word="fast", widen=True, out="bf16", stack=st271_a)
        assert s.row not in ("fast_layernorm", "aten_autocast") and "skip fast_layernorm>fast_layernorm_ext(not_landed" in s.reason, (s.row, s.reason)
    monkeypatch.setattr(S, "landed", lambda: True)
    monkeypatch.setattr(S, "keys", lambda: ["torch2.13.0+cu130-cp311", "torch2.7.1+cu126-cp311"])
    monkeypatch.setattr(S, "available", lambda key=None: (True, key or "torch2.7.1+cu126-cp311"))
    L._SELECT_CACHE.clear() if hasattr(L, "_SELECT_CACHE") else None
    s = L.select("8.0", "bf16", "pair_c128", 400, word="fast", widen=True, out="bf16", stack=st271_a)
    assert s.row == "fast_layernorm_ext", (s.row, s.reason)
    s = L.select("9.0", "bf16", "pair_c128", 800, word="exact", widen=True, out="bf16", stack=st271_h)
    assert s.row == "fast_layernorm_ext", (s.row, s.reason)                     # the exact floor (the trunk's extension) served by its carried row on a vouched stack
    monkeypatch.setattr(S, "available", lambda key=None: (False, "no_prebuilt:%s" % key))
    L._SELECT_CACHE.clear() if hasattr(L, "_SELECT_CACHE") else None
    s = L.select("9.0", "bf16", "pair_c128", 800, word="fast", widen=True, out="bf16", stack=st210)    # a stack without a prebuilt: the next measured row, the skip named
    assert s.row not in ("fast_layernorm_ext", "fast_layernorm", "aten_autocast") and "skip fast_layernorm>fast_layernorm_ext(no_prebuilt" in s.reason, (s.row, s.reason)
    s = L.select("9.0", "bf16", "pair_c128", 800, word="fast_layernorm", widen=True, out="bf16", stack=st271_h)   # the row word itself is untouched (an engine naming its own object)
    assert s.row == "fast_layernorm"



def test_tier_word_falls_to_the_next_measured_row_not_the_statement(monkeypatch):
    """A cell whose fast winner cannot serve on the caller's stack (the trunk-extension arm with the carried row dormant / without a prebuilt)
    -> the NEXT measured admissible row in speed order (9.0 bf16o pair_c128 N<=800: fastln:lp), the skip named in the reason; a cell whose
    measured rows all refuse -> the statement by name; the exact word whose floor is that arm -> the form's ATen statement by name."""
    from opt_core.kernels import ln as L
    from opt_core.kernels.ln import ext_loader as S
    st271_h = "H100:torch2.7.1+cu126/3.3.1/cueq0.10.0"
    monkeypatch.setattr(S, "landed", lambda: False)
    L._SELECT_CACHE.clear() if hasattr(L, "_SELECT_CACHE") else None
    cell = L.table()["cells"]["9.0|bf16o|pair_c128|N<=800|eager|fwd"]; sc = cell["stacks"][st271_h]
    assert sc["fast"] == "fast_layernorm"
    nxt = [a for a in sorted(sc["ms"], key=sc["ms"].get) if a != "fast_layernorm" and not a.startswith("x:")][0]
    s = L.select("9.0", "bf16", "pair_c128", 800, word="fast", widen=True, out="bf16", stack=st271_h)
    assert (s.row + (":" + s.variant if s.variant else "")) == {"fastln:lp": "fastln:lp"}.get(nxt, nxt) and s.row == "fastln" and s.variant == "lp", (s.row, s.variant, nxt, s.reason)
    assert "skip fast_layernorm>fast_layernorm_ext(not_landed" in s.reason
    b = L.select("9.0", "bf16", "pair_c128", 800, word="big", widen=True, out="bf16", stack=st271_h); assert b.row != "fast_layernorm" and b.row != "aten_autocast"
    e = L.select("9.0", "bf16", "pair_c128", 800, word="exact", widen=True, out="bf16", stack=st271_h)
    assert e.row == "aten_autocast" and "exact floor fast_layernorm unservable" in e.reason, (e.row, e.reason)
    real_admits = L.admits
    def refuse_all(row, *a, **k):
        if row in L.STOCK_ROWS and row not in L.KIT_SIDE_ARMS:
            return real_admits(row, *a, **k)
        raise L.Refusal("test_refusal", row, "aten")
    monkeypatch.setattr(L, "admits", refuse_all)
    L._SELECT_CACHE.clear() if hasattr(L, "_SELECT_CACHE") else None
    s = L.select("9.0", "bf16", "pair_c128", 800, word="fast", widen=True, out="bf16", stack=st271_h)
    assert s.row == "aten_autocast" and "skip" in s.reason, (s.row, s.reason)      # every measured row refused: the statement, by name, skips listed



def test_noreg_unmeasured_capability_inherits_the_nearest_measured_columns_portable_rows():
    """NO-REGRESSION rule for a capability the table never measured FOR A KEY (12.0 today; 11.0 hypothetically): fast | big inherit the
    nearest arch-compatible measured column FOR THAT KEY (10.3 / 10.0 where the parts sweep measured the key, else 9.0) -- its source-compiled
    rows first, never a prebuilt built for another arch, the statement only after every portable arm refused by name; exact = the statement;
    token names both capabilities; measured columns resolve byte-identically."""
    from opt_core.kernels import ln as L
    cols = L.measured_columns()
    assert {"9.0", "8.0"} <= set(cols) and "12.0" not in cols and "11.0" not in cols
    for cc in ("12.0", "11.0"):
        fam = ("bf16", "pair_c128", "eager", "fwd")
        col = L.inherit_column(cc, fam); assert col in ("10.3", "10.0", "9.0"), col
        for kw in (dict(widen=True), dict(widen=True, out="bf16"), dict()):
            s = L.select(cc, "bf16", "pair_c128", 800, word="fast", **kw)
            colk = L.inherit_column(cc, (L.dtype_word("bf16", kw.get("widen", False), kw.get("out")), "pair_c128", "eager", "fwd"))
            assert s.row not in ("fast_layernorm_ext", "fast_layernorm"), (cc, kw, s.row)
            assert s.row in L.PORTABLE_ROWS or (s.row in L.STOCK_ROWS and "skip " in s.reason), (cc, kw, s.row, s.reason)
            assert s.reason.startswith("inherited_cc:unmeasured(%s->%s)" % (cc, colk)) and s.cell.startswith(colk + "|"), s.reason
        assert L.select(cc, "bf16", "pair_c128", 800, word="fast").row in L.PORTABLE_ROWS
        e = L.select(cc, "bf16", "pair_c128", 800, word="exact")
        assert e.row == "aten" and e.reason.startswith("inherited_cc:unmeasured(%s->%s)" % (cc, col)), (e.row, e.reason)
        r = L.select(cc, "bf16", "pair_c128", 800, word="fastln")
        assert r.row == "fastln" and not r.reason.startswith(L.INHERIT_TOKEN)
    assert L.inherit_column("8.6", ("bf16", "pair_c128", "eager", "fwd")) == "8.0" and L.inherit_column("9.0") is None and L.inherit_column("7.5") is None
    cells = L.table()["cells"]
    for key in list(cells)[:400]:
        cc, dt, cw, nb, tm, pas = key.split("|")
        widen = dt in ("bf16w", "bf16o"); out = "bf16" if dt == "bf16o" else None
        dtype = "bf16" if dt.startswith("bf16") else dt
        for word in ("fast", "exact", "big"):
            a = L._select(cc, dtype, cw, int(nb[3:]), word=word, widen=widen, out=out, timing=tm, pass_=pas)
            b = L._select_measured(cc, dtype, cw, int(nb[3:]), word=word, widen=widen, out=out, timing=tm, pass_=pas)
            assert a == b, (key, word, a, b)
    L._CENSUS.reset()
    L.select("12.0", "bf16", "pair_c128", 800, word="fast")
    ent = L._CENSUS.table()[-1]
    assert ent["outcome"] == "inherited" and ent["note"].startswith("inherited_cc:unmeasured(12.0->"), ent
def test_noreg_partial_cc_one_injected_cell_still_inherits_every_other_key(monkeypatch):
    """cc 10.3 with ONE injected cell (pair_c256 bf16w only): inheritance is decided PER FAMILY KEY -- pair_c128 (and every other key the
    part did not measure) still inherits 9.0's portable row with the token (+partial_cc), the injected key decides for itself (no token),
    and the measured columns stay byte-identical."""
    import copy
    from opt_core.kernels import ln as L
    tab = dict(L.table()); tab["cells"] = dict(tab["cells"]); cells = tab["cells"]
    monkeypatch.setattr(L, "table", lambda: tab)                                          # the injected table object is the one the face reads (cache reloads elsewhere cannot drop it)
    if hasattr(L, "_FAMILIES"):
        monkeypatch.setattr(L, "_FAMILIES", {})                                            # the family memo starts clean (an earlier query of this key may have cached 'no family')
    src = "9.0|bf16w|pair_c256|N<=800|eager|fwd"; inj = "12.0|bf16w|pair_c256|N<=800|eager|fwd"
    assert src in cells and inj not in cells and "12.0" not in L.measured_columns()
    cells[inj] = copy.deepcopy(cells[src])
    try:
        assert L.inherit_column("12.0") is None                                              # per-cc view: the part now 'has cells' ...
        colk = L.inherit_column("12.0", ("bf16", "pair_c128", "eager", "fwd")); assert colk in ("10.3", "10.0", "9.0")   # ... per key it still inherits
        assert L.inherit_column("12.0", ("bf16w", "pair_c256", "eager", "fwd")) is None
        for w in ("fast", "big"):
            s = L.select("12.0", "bf16", "pair_c128", 800, word=w)
            assert s.row in L.PORTABLE_ROWS and s.reason.startswith("inherited_cc:unmeasured(12.0->%s);partial_cc" % colk) and s.cell.startswith(colk + "|"), (w, s.row, s.reason)
        s = L.select("12.0", "fp32", "pair_c384", 1200, word="fast")
        assert s.row in L.PORTABLE_ROWS and s.reason.startswith(L.INHERIT_TOKEN), (s.row, s.reason)
        e = L.select("12.0", "bf16", "pair_c128", 800, word="exact")
        assert e.row == "aten" and e.reason.startswith(L.INHERIT_TOKEN)
        own = L.select("12.0", "bf16", "pair_c256", 800, word="fast", widen=True)               # the injected key: its own cell, no token
        assert own.cell == inj and not own.reason.startswith(L.INHERIT_TOKEN)
        for key in [k for k in cells if k.startswith(("9.0|", "8.0|"))][:200]:                  # measured columns byte-identical
            cc, dt, cw, nb, tm, pas = key.split("|")
            widen, out = (dt in ("bf16w", "bf16o")), ("bf16" if dt == "bf16o" else None)
            dtype = "bf16" if dt.startswith("bf16") else dt
            for word in ("fast", "exact", "big"):
                assert L._select(cc, dtype, cw, int(nb[3:]), word=word, widen=widen, out=out, timing=tm, pass_=pas) == \
                       L._select_measured(cc, dtype, cw, int(nb[3:]), word=word, widen=widen, out=out, timing=tm, pass_=pas), (key, word)
    finally:
        del cells[inj]



def test_noreg_inherited_row_launch_guard_serves_the_next_row_then_the_statement(monkeypatch):
    """An INHERITED portable row that fails to build / compile / launch on the new part (RuntimeError at first launch): the call does not raise --
    the arm is retired for that part (census stepped_aside:error:RuntimeError ONCE), the next portable row of the donor cell is tried, the ATen
    statement serves last; a second call does not retry the dead rows; out-of-memory is re-raised as before."""
    import torch
    from opt_core.kernels import ln as L
    calls = {"n": 0}

    class _Boom(object):
        def __getattr__(self, name):
            def f(*a, **k):
                calls["n"] += 1
                raise RuntimeError("ptxas fatal: unsupported gpu architecture (simulated)")
            return f
    orig = L.inherit_column
    monkeypatch.setattr(L, "inherit_column", lambda cc, family=None: "9.0" if L.cc_word(cc) == "0.0" else orig(cc, family))   # a CPU call stands in for a new part
    monkeypatch.setattr(L, "carried_module", lambda *a, **k: _Boom())
    monkeypatch.setattr(L, "_DEAD_ARMS", {})
    L._CENSUS.reset()
    x = torch.randn(64, 128); w = torch.ones(128); b = torch.zeros(128)
    y, sel = L.layer_norm(x, (128,), w, b, 1e-5, word="fast", n_tokens=8, cell="pair_c128")
    assert sel.row in L.STOCK_ROWS and sel.reason.startswith(L.INHERIT_TOKEN), (sel.row, sel.reason)
    assert torch.allclose(y, torch.nn.functional.layer_norm(x, (128,), w, b, 1e-5))
    dead = L.dead_arms("0.0"); assert dead and "RuntimeError" in dead.values(), dead                # (the simulated module also trips TypeErrors in rows that probe it first: any non-OOM error retires the arm)
    toks = [e for e in L._CENSUS.table() if "stepped_aside:error:" in str(e.get("refused"))]
    assert len(toks) == len(dead), (toks, dead)                                                # once per retired arm
    n1 = calls["n"]
    y2, sel2 = L.layer_norm(x, (128,), w, b, 1e-5, word="fast", n_tokens=8, cell="pair_c128")
    assert sel2.row == sel.row and calls["n"] == n1 and L.dead_arms("0.0") == dead                 # the dead rows are not retried
    assert len([e for e in L._CENSUS.table() if "stepped_aside:error" in str(e.get("refused"))]) == len(dead)

    class _OOM(object):
        def __getattr__(self, name):
            def f(*a, **k):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)") if hasattr(torch.cuda, "OutOfMemoryError") else MemoryError("out of memory")
            return f
    monkeypatch.setattr(L, "_DEAD_ARMS", {})
    monkeypatch.setattr(L, "carried_module", lambda *a, **k: _OOM())
    import pytest as _pt
    with _pt.raises(Exception) as ei:
        L.layer_norm(x, (128,), w, b, 1e-5, word="fast", n_tokens=8, cell="pair_c128")
    from opt_core.oom import is_oom
    assert is_oom(ei.value) and L.dead_arms("0.0") == {}
