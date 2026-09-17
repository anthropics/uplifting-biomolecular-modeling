"""kernels/apb: the pair-bias attention provider — its cell table, selection rules (default rule / opt-in tiers / refusals by name / capture
safety), the fp32 exact row's ABI gate, the module census and the carried sub-packages' byte identity with the kit copies.  CPU only."""
import ast
import filecmp
import json
import os

import pytest

from opt_core import kernels
from opt_core.kernels import apb as A

CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RELEASE_TREE = os.path.dirname(os.path.dirname(CORE_DIR))
CARDS = ("9.0", "8.0")
SIZES = (400, 800, 1200)
# every pair-bias attention geometry the release tree's torch engines run today (dtype, cell, samples):
CORE_SHAPES = [("bf16", "dit_h16d48", 5), ("fp32", "dit_h16d48", 5), ("bf16", "dit_h16d48", 1), ("fp32", "dit_h16d48", 1), ("bf16", "pf_h16d24", 1), ("fp32", "pf_h16d24", 1),
               ("bf16", "msarow_h8d32", 128), ("fp32", "msarow_h8d32", 128), ("bf16", "atom_h4d32w32x128", 5), ("fp32", "atom_h4d32w32x128", 5)]
PRODUCER_SHAPES = [("bf16", "bias_c128h16", 1), ("fp32", "bias_c128h16", 1), ("bf16", "bias_c128h4", 1), ("fp32", "bias_c64h4", 1), ("bf16", "bias_c256h16", 1), ("bf16", "bias_c384h16", 1)]
MODULE_SHAPES = [("bf16", "mod_pf_c384cz128", 1), ("bf16", "mod_dit_c768cz128", 1)]


def test_table_schema_rows_words():
    t = A.table()
    assert t["schema"] == "apb_cells/v1"
    for k in ("op", "key_grammar", "evidence", "stacks", "rows", "tiers", "default_rule", "exact_rule", "capture_rule", "cells", "coverage", "retune_fpf_apb", "capture_checks"):
        assert k in t, k
    assert set(A.ROW_NAMES) == set(t["rows"]), sorted(set(A.ROW_NAMES) ^ set(t["rows"]))
    for name, row in t["rows"].items():
        for k in ("class", "boundary", "capture_safe", "fallback"):
            assert k in row, (name, k)
        assert row["class"] in ("fast", "exact", "stock"), name
        assert row["boundary"].split(":")[0] in ("core", "windowed", "module", "producer", "blockrows"), name
        assert A.BOUNDARY[name] == row["boundary"].split(":")[0], name
    assert {n for n, r in t["rows"].items() if r["class"] == "stock"} == set(A.STOCK_ROWS)
    assert {n for n, r in t["rows"].items() if r["class"] == "exact"} == set(A.EXACT_ROWS)
    assert {n for n, r in t["rows"].items() if not r["capture_safe"]} == set(A.CAPTURE_UNSAFE_ROWS) == {"ds4sci"}
    assert set(t["tiers"]) == set(A.TIER_WORDS)
    assert len(t["cells"]) >= 300
    ccs = {k.split("|")[0] for k in t["cells"]}
    assert {"9.0", "8.0"} <= ccs <= {"9.0", "8.0", "10.0", "10.3"}
    for key, cell in t["cells"].items():
        cc, dt, cw, s, n, timing, pas = key.split("|")
        assert dt in ("bf16", "fp32") and cw in A.CELL_WORDS and s.startswith("S") and n.startswith("N<=") and timing in ("eager", "graph") and pas == "fwd", key
        assert cell["ref_stack"] in cell["stacks"] and cell["ref_stack"] in t["stacks"], key
        for w in ("fast", "exact"):
            arm = cell.get(w)
            assert arm is None or A.split_word(arm)[0] in A.ROW_NAMES, (key, w, arm)
        if cell.get("exact") and A.split_word(cell["exact"])[0] in A.EXACT_ROWS:
            assert cell["exact_x"] >= 1.0, key                                    # EXACT FLOOR x1.00
            assert cell["exact_parity_flag"].startswith("bitwise_vs:"), key
        st = cell["stacks"][cell["ref_stack"]]
        for arm in st["ms"]:
            assert A.split_word(arm)[0] in A.ROW_NAMES or arm.startswith("x:"), (key, arm)
        if timing == "graph":
            assert A.split_word(cell["fast"])[0] not in A.CAPTURE_UNSAFE_ROWS, key   # a capture-unsafe row is never a graph cell's winner


@pytest.mark.parametrize("cc", CARDS)
def test_every_shape_the_tree_runs_lands_on_a_measured_cell_or_a_named_stock_row(cc):
    for dt, cw, s in CORE_SHAPES + PRODUCER_SHAPES + MODULE_SHAPES:
        for n in SIZES:
            for word in ("fast", "exact"):
                sel = A.select(cc, dt, cw, n, word=word, samples=s, abi=A.DIT_EXACT_ABIS[0])
                assert sel.row in A.ROW_NAMES
                if sel.cell is None:
                    assert sel.row in A.STOCK_ROWS and not sel.size_measured, (cc, dt, cw, n, word, sel)
                else:
                    assert sel.cell in A.table()["cells"]
    cov = A.coverage(cc, [(dt, cw, 800, s) for dt, cw, s in CORE_SHAPES])
    assert len(cov) == len(CORE_SHAPES) and all("row" in c for c in cov)


def test_default_rule_a_row_word_serves_that_row():
    """HAZARD 11: a row word is exactly that row (no substitution); the existing binders' kernels are rows of this table by name."""
    for word, row, variant in (("apb_attn", "apb_attn", None), ("fpf_apb", "fpf_apb", None), ("fpf_apb:fp16", "fpf_apb", "fp16"), ("l3a", "l3a", None), ("l3a:c", "l3a", "c"),
                               ("dtk_loop", "dtk_loop", None), ("sba", "sba", "tf32"), ("sba:ieee", "sba", "ieee"), ("sdpa", "sdpa", "auto"), ("sdpa:cudnn", "sdpa", "cudnn"),
                               ("sdpa_upcast", "sdpa_upcast", None), ("naive", "naive", None), ("ds4sci", "ds4sci", None)):
        dt = "fp32" if word in ("fpf_apb:fp16", "sdpa_upcast") else "bf16"
        sel = A.select("9.0", dt, "dit_h16d48", 800, word=word, samples=5)
        assert (sel.row, sel.variant, sel.word) == (row, variant, word), (word, sel)            # a plain word with a default variant reports that variant (its x_stock / class are that arm's)
        assert "row word" in sel.reason
    sel = A.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi=A.DIT_EXACT_ABIS[0])
    assert sel.row == "dit_exact" and sel.cls in ("exact", "bitwise") and sel.exact_vs == "sdpa_upcast"
    for word in ("ln_proj", "fpf_pf_bias", "lnl_ln_linear", "torch_ln_linear"):
        assert A.select("9.0", "bf16", "bias_c128h16", 800, word=word).row == word
    for word in ("fpf_atom", "dtk_window", "sdpa_gather"):
        assert A.select("8.0", "bf16", "atom_h4d32w32x128", 800, word=word, samples=5).row == word


def test_tier_words_are_the_measured_winners():
    t = A.table()["cells"]
    # DiT bf16 x5, H100: eager -> the SDPA statement family wins (the table says so); one captured graph -> a fused kernel wins
    e = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5)
    g = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, capture=True)
    assert e.cell == "9.0|bf16|dit_h16d48|S5|N<=800|eager|fwd" and g.cell == "9.0|bf16|dit_h16d48|S5|N<=800|graph|fwd"
    assert A.arm_word(e.row, e.variant) == t[e.cell]["fast"] and A.arm_word(g.row, g.variant) == t[g.cell]["fast"]
    assert e.word == "fast" and e.x_stock is not None and e.stack_measured
    # fp32 DiT exact on H100 with the prebuilt's ABI -> the exact replica; on another ABI or on A100 -> the named statement (exact by definition), skip recorded
    H213 = "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1"                                # EXACT VOUCH IS STACK-SPECIFIC: the exact word lands on a non-stock arm only on a vouched stack
    x = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, abi=A.DIT_EXACT_ABIS[0], stack=H213)
    assert x.row == "dit_exact" and x.cls == "bitwise" and x.exact_vs == "sdpa_upcast" and x.x_stock >= 1.0
    assert H213 in A.table()["cells"][x.cell]["vouched_on"]["dit_exact"]
    u = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, abi=A.DIT_EXACT_ABIS[0])          # stack not named -> the floor, by name
    assert u.row in A.STOCK_ROWS and "exact vouch not recorded" in u.reason
    u2 = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, abi=A.DIT_EXACT_ABIS[0], stack="H100:torch2.10.0+cu128/3.6.0/cueq0.10.0")
    assert u2.row in A.STOCK_ROWS and "exact vouch not recorded on H100:torch2.10.0" in u2.reason
    assert A.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi=A.DIT_EXACT_ABIS[0]).row == "dit_exact"   # a row word is served as asked
    y = A.select("9.0", "fp32", "dit_h16d48", 800, word="exact", samples=5, abi="torch2.10.0-cu128-sm90", stack=H213)
    assert y.row in A.STOCK_ROWS and "skip dit_exact(no_prebuilt_for_stack" in y.reason
    z = A.select("8.0", "fp32", "dit_h16d48", 800, word="exact", samples=5)
    assert z.row in A.STOCK_ROWS
    # bf16 has no exact-class kernel: exact = the stock statement (or a stock backend bitwise to it)
    b = A.select("9.0", "bf16", "pf_h16d24", 800, word="exact")
    assert b.row in A.STOCK_ROWS
    # a pinned SDPA backend is an exact-tier arm only where it was MEASURED bitwise to the auto backend on the caller's stack
    g = A.select("9.0", "bf16", "pf_h16d24", 1200, word="exact", capture=True, stack=H213)
    assert (g.row, g.variant) in (("sdpa", "cudnn"), ("sdpa", "auto"), ("sdpa", None))
    S210 = "H100:torch2.10.0+cu128/3.6.0/cueq0.10.0"
    g2 = A.select("9.0", "bf16", "pf_h16d24", 1200, word="exact", capture=True, stack=S210)
    arm2 = A.arm_word(g2.row, g2.variant)
    assert g2.row == "sdpa" and g2.variant != "cudnn"                                  # cudnn != auto bitwise on that stack: never served there under exact
    assert arm2 == A.table()["cells"][g2.cell]["stacks"][g2.stack]["stock_arm"] or S210 in A.table()["cells"][g2.cell]["vouched_on"].get(arm2, [])
    # prefer narrows a tier to the caller's rows (plain variant first)
    p = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, prefer=("l3a", "dtk_loop"))
    assert (p.row, p.variant) == ("l3a", None)
    p = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, prefer=("l3a:c",))
    assert (p.row, p.variant) == ("l3a", "c")
    # per-stack winners: a stack the table measured answers with its own winner, another stack with the reference stack's (flagged)
    st = t[e.cell]["ref_stack"]
    s1 = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, stack=st)
    assert s1.stack_measured and s1.stack == st
    s2 = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, stack="H100:torch9.9.9+cu999/9.9.9/none")
    assert not s2.stack_measured
    # sizes: the smallest measured size >= n, else the largest flagged beyond_measured; samples: nearest measured S
    assert A.cell_key("9.0", "bf16", "dit_h16d48", 401, 5)[:2] == ("9.0|bf16|dit_h16d48|S5|N<=800|eager|fwd", True)
    big = A.select("9.0", "bf16", "dit_h16d48", 10000, word="fast", samples=5)
    assert not big.size_measured and "beyond_measured" in big.reason
    assert A.cell_key("9.0", "bf16", "dit_h16d48", 800, 2)[0].split("|")[3] in ("S1", "S5")
    # module boundary: the composed form (fused bias producer + one-launch core) is the measured winner at the trunk widths on H100
    m = A.select("9.0", "bf16", "mod_pf_c384cz128", 800, word="fast")
    assert m.row == "composed" and m.x_stock > 2.0
    # producers: c 384 is outside the fused producers' widths -> the statement by name
    assert A.select("9.0", "bf16", "bias_c384h16", 800, word="fast").row == "torch_ln_linear"
    assert A.select("9.0", "bf16", "bias_c128h16", 800, word="fast").row in ("ln_proj", "fpf_pf_bias", "lnl_ln_linear")


def test_refusals_are_by_name_with_the_fallback_row():
    with pytest.raises(A.Refusal) as ei:
        A.select("9.0", "fp32", "pf_h16d24", 800, word="apb_attn")               # fp32 activations are not served by the core's kernel
    assert ei.value.kind.startswith("dtype:float32") and ei.value.row == "apb_attn" and ei.value.fallback == "sdpa"
    with pytest.raises(A.Refusal) as ei:
        A.select("9.0", "bf16", "pf_h16d24", 800, word="ds4sci", capture=True)   # capture-UNSAFE by name
    assert "capture" in ei.value.kind and ei.value.fallback
    assert A.capture_unsafe("ds4sci") and not A.capture_unsafe("fpf_apb")
    with pytest.raises(A.Refusal) as ei:
        A.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi="torch2.10.0-cu128-sm90")
    assert "no_prebuilt_for_stack" in ei.value.kind and ei.value.fallback == "sdpa_upcast"
    with pytest.raises(A.Refusal) as ei:
        A.select("8.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi=A.DIT_EXACT_ABIS[0])
    assert "cc" in ei.value.kind
    with pytest.raises(A.Refusal):
        A.select("9.0", "bf16", "dit_h16d48", 800, word="dit_exact", samples=5, abi=A.DIT_EXACT_ABIS[0])   # the fp32 statement's replica: bf16 refused by name
    with pytest.raises((A.Refusal, ValueError)):
        A.select("9.0", "bf16", "dit_h16d48", 800, word="no_such_row")
    with pytest.raises(A.Refusal):
        A.select("7.5", "bf16", "dit_h16d48", 800, word="apb_attn")               # below the kernels' floor
    # a tier word never raises for a listed cell word: it lands on a row (a stock row at worst)
    for cw in A.CELL_WORDS:
        assert A.select("8.0", "fp32", cw, 1200, word="fast", samples=5).row in A.ROW_NAMES


def test_describe_and_coverage_are_plain_data():
    sel = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5)
    line = A.describe(sel)
    assert isinstance(line, str) and "dit_h16d48" in line and sel.row in line
    kit = A.rows_for_kit(["fpf_apb", "dit_exact", "apb_attn"])
    assert set(kit) >= {"fpf_apb", "dit_exact", "apb_attn"} and set(kit) <= set(A.ROW_NAMES)   # the kit's rows + the named stock rows they fall back to
    json.dumps(A.coverage("9.0", [("bf16", "dit_h16d48", 800, 5)]))


def test_face_imports_only_the_standard_library_at_module_level():
    tree = ast.parse(open(os.path.join(CORE_DIR, "opt_core", "kernels", "apb", "__init__.py"), encoding="utf-8").read())
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
    assert "apb" in kernels.names()
    meta = kernels.sums("apb")
    assert meta["kind"] == "package" and meta["version"] == "1.0.0"
    assert set(meta["runtime_imports"]) == {"apb_attn", "dtk_kernels", "ln_proj", "lnl_fused", "flash_triattn"}
    files = set(meta["files"])
    for f in ("__init__.py", "APB_CELLS.json", "fpf_apb/__init__.py", "fpf_apb/apb_triton.py", "fpf_apb/atom_triton.py", "fpf_apb/pf_triton.py", "fpf_apb/NOTICE.md",
              "dit_exact/__init__.py", "dit_exact/csrc/dit_attn_exact.cu", "dit_exact/prebuilt/torch2.13.0-cu130-sm90/manifest.json", "dit_exact/prebuilt/torch2.13.0-cu130-sm90/dit_attn_exact.so", "dit_exact/LICENSE_NOTE.md",
              "l3a/__init__.py", "l3a/fab_batched.py", "l3a/bias_layout.py", "l3a/NOTICE.md", "ef2/ef2_pairbias_attn.py", "ef2/__init__.py"):
        assert f in files, f
    import hashlib
    pre = os.path.join(kernels.KERNELS_DIR, "apb", "dit_exact", "prebuilt")
    abis = sorted(d for d in os.listdir(pre) if os.path.isdir(os.path.join(pre, d)))
    assert tuple(abis) == tuple(sorted(A.DIT_EXACT_ABIS))                          # the face's ABI list is the prebuilt directory's
    for abi in abis:
        man = json.load(open(os.path.join(pre, abi, "manifest.json")))
        so = os.path.join(pre, abi, man.get("so", "dit_attn_exact.so"))
        assert os.path.isfile(so), so
        digest = hashlib.sha256(open(so, "rb").read()).hexdigest()
        assert digest in json.dumps(man), abi                                      # the manifest pins the binary it names


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


@pytest.mark.parametrize("carried", sorted(json.load(open(os.path.join(kernels.KERNELS_DIR, "META", "apb.json"), encoding="utf-8"))["carried_digests"]))
def test_carried_files_have_their_recorded_digest_and_match_kit_copies_by_digest(carried):
    """The carried file has the digest META records for it (the core's own consistency: always checked), and a kit copy found by NAME anywhere in the
    release tree with the SAME digest witnesses the byte identity.  Kit-side moves and edits never fail this gate: no copy in the checkout, or only
    copies with another digest (the kit edited its file after the carry, or an unrelated file of that name), SKIP with the reason named -- the provider
    serves the carried bytes either way."""
    meta = json.load(open(os.path.join(kernels.KERNELS_DIR, "META", "apb.json"), encoding="utf-8"))
    mine_path = os.path.join(kernels.KERNELS_DIR, "apb", carried)
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


def test_dit_exact_prebuilts_are_kit_builds_byte_for_byte():
    """Every prebuilt/<key>/ directory of dit_exact equals, file for file, the build shipped by a kit package of the SAME source (a directory in the
    release tree with csrc/dit_attn_exact.cu identical to core's and prebuilt/<key>/manifest.json) -- for every kit that ships that key."""
    root = os.path.join(kernels.KERNELS_DIR, "apb", "dit_exact")
    if not os.path.isdir(RELEASE_TREE):
        pytest.skip("kits not beside common/")
    builds = {}                                                                   # key -> [kit prebuilt dirs]
    for dirpath, dirs, files in os.walk(RELEASE_TREE):
        if os.sep + "common" + os.sep in dirpath + os.sep or "__pycache__" in dirpath:
            continue
        if "dit_attn_exact.cu" in files and os.path.basename(dirpath) == "csrc":
            pkg = os.path.dirname(dirpath)
            if not filecmp.cmp(os.path.join(dirpath, "dit_attn_exact.cu"), os.path.join(root, "csrc", "dit_attn_exact.cu"), shallow=False):
                continue
            pre = os.path.join(pkg, "prebuilt")
            for key in (sorted(os.listdir(pre)) if os.path.isdir(pre) else []):
                if os.path.isfile(os.path.join(pre, key, "manifest.json")):
                    builds.setdefault(key, []).append(os.path.join(pre, key))
    if not builds:
        pytest.skip("no kit build beside common/")
    for key in A.DIT_EXACT_ABIS:
        if key not in builds:                                                     # no kit in this checkout ships that key's build: nothing to witness (the manifest digests test covers the payload)
            continue
        mine = os.path.join(root, "prebuilt", key)
        for kit_dir in builds[key]:
            for f in sorted(os.listdir(mine)):
                assert filecmp.cmp(os.path.join(mine, f), os.path.join(kit_dir, f), shallow=False), (key, kit_dir, f)


def test_l3a_is_standard_library_plus_guarded_triton_at_import():
    src = open(os.path.join(kernels.KERNELS_DIR, "apb", "l3a", "fab_batched.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    names = {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert {"flash_bias_attn_batched", "KoptAttnUnavailable", "KoptAttnRefused"} <= names
    notice = open(os.path.join(kernels.KERNELS_DIR, "apb", "l3a", "NOTICE.md"), encoding="utf-8").read()
    assert "a275f728eeca" in notice                                              # the source commit of the carried bytes


def test_int32_offset_bounds_are_refused_by_name():
    """Index audit: the carried fpf_apb package forms per-(sample, head) base offsets in int32 (s*stride_s, h*stride_h, h*N*ld); the face refuses
    it BY NAME above the proven-safe bound (2^31 - 2^24) and names the int64-addressed fallback; every other served kernel forms its offset
    products in int64 (apb_attn: int64 bases + its own size:int32_* refusals; dtk / l3a / shared-bias / ln_proj / lnl: int64 program indices)."""
    assert set(A.INT32_OFFSET_ROWS) == {"fpf_apb", "fpf_atom"} and A.INT32_SAFE == 2 ** 31 - 2 ** 24
    # the DiT geometry (16 heads x 48): the shared bias base h*N*ld passes 2^31 first, near N = 11,500; production sizes (N <= 5,120, S <= 5) are far inside
    assert A.int32_products("fpf_apb", samples=5, n_tokens=5120, heads=16, head_dim=48)["H*N*ld"] < A.INT32_SAFE
    A.admits("fpf_apb", "9.0", "bf16", samples=5, n_tokens=5120, heads=16, head_dim=48)
    A.admits("fpf_apb", "9.0", "bf16", samples=5, n_tokens=11000, heads=16, head_dim=48)
    with pytest.raises(A.Refusal) as ei:
        A.admits("fpf_apb", "9.0", "bf16", samples=5, n_tokens=11600, heads=16, head_dim=48)
    assert ei.value.kind.startswith("int32_offsets:H*N*ld") and ei.value.fallback == "apb_attn"
    with pytest.raises(A.Refusal) as ei:                                        # the sample base s*N*H*D: S 200 x N 16384 x 16 x 48
        A.admits("fpf_apb", "9.0", "bf16", samples=200, n_tokens=16384, heads=16, head_dim=48)
    assert ei.value.kind.startswith("int32_offsets:")
    # select() applies the same bound: a row word refuses, a tier word skips the row by name and lands elsewhere
    with pytest.raises(A.Refusal):
        A.select("9.0", "bf16", "dit_h16d48", 12000, word="fpf_apb", samples=5, heads=16, head_dim=48)
    sel = A.select("9.0", "bf16", "dit_h16d48", 12000, word="fast", samples=5, heads=16, head_dim=48, capture=True)
    assert sel.row != "fpf_apb"
    # atom windows: S*NA*H*D and H*blocks*32*128; 5 samples x 4 heads x 32: NA up to ~3.3M atoms
    A.admits("fpf_atom", "9.0", "bf16", samples=5, n_atoms=8 * 5120, heads=4, head_dim=32)
    with pytest.raises(A.Refusal) as ei:
        A.admits("fpf_atom", "9.0", "bf16", samples=5, n_atoms=3_400_000, heads=4, head_dim=32)
    assert ei.value.fallback == "dtk_window"
    assert A.int32_products("apb_attn", samples=5, n_tokens=10 ** 6) == {}       # int64-addressed rows form no int32 products here


def _apb_triton_pure(*names):
    """apb_config / resolve_cfg (+ their module constants) executed from the carried source WITHOUT importing triton / CUDA."""
    import ast
    src = open(os.path.join(kernels.KERNELS_DIR, "apb", "fpf_apb", "apb_triton.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in names)
            or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("_CFG", "LAUNCH_OPTION_KEYS") for t in n.targets))
            or (isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id in ("_CFG", "LAUNCH_OPTION_KEYS"))]
    ns = {"__name__": "apb_triton_pure"}
    exec(compile(ast.Module(body=keep, type_ignores=[]), "apb_triton_pure", "exec"), ns)
    return ns


def test_fpf_apb_cfg_is_key_merged_over_the_cell():
    """apb_views' launch cell = apb_config(D, opd, in_fp32) with the caller's cfg merged over it KEY BY KEY (resolve_cfg): a partial cfg such as
    {'BIAS_TMA': False} -- what a caller on a Triton without host-side TMA descriptors passes -- keeps D1 / D2 / BLOCK_* from the cell (before:
    KeyError 'D1'); None / {} = the cell; a full cfg = exactly that cfg (bytes and launch unchanged for those callers); unknown keys raise by name."""
    ns = _apb_triton_pure("apb_config", "resolve_cfg")
    apb_config, resolve_cfg = ns["apb_config"], ns["resolve_cfg"]
    src = open(os.path.join(kernels.KERNELS_DIR, "apb", "fpf_apb", "apb_triton.py"), encoding="utf-8").read()
    assert "c = resolve_cfg(cfg, D, opd_name, q.dtype == torch.float32)" in src and "cfg or apb_config(" not in src
    for D, opd, f32 in ((48, "bf16", False), (48, "bf16", True), (24, "bf16", False), (32, "fp16", False)):
        cell = dict(apb_config(D, opd, f32))
        assert {"D1", "D2", "BLOCK_M", "BLOCK_N", "num_warps", "num_stages", "SG", "BIAS_TMA"} <= set(cell)
        assert resolve_cfg(None, D, opd, f32) == cell and resolve_cfg({}, D, opd, f32) == cell            # no cfg: the cell
        part = resolve_cfg({"BIAS_TMA": False}, D, opd, f32)                                               # the no-host-TMA partial cfg
        assert part["BIAS_TMA"] is False and {k: v for k, v in part.items() if k != "BIAS_TMA"} == {k: v for k, v in cell.items() if k != "BIAS_TMA"}
        assert resolve_cfg({"BLOCK_N": 32, "num_stages": 1}, D, opd, f32) == dict(cell, BLOCK_N=32, num_stages=1)
        full = dict(cell, BLOCK_M=128, num_warps=8)
        assert resolve_cfg(full, D, opd, f32) == full                                                       # a full cfg: exactly it
        assert resolve_cfg(dict(cell, maxnreg=128), D, opd, f32)["maxnreg"] == 128                          # launch options pass
        r = resolve_cfg({"BIAS_TMA": False}, D, opd, f32); r.pop("D1")
        assert "D1" in apb_config(D, opd, f32)                                                              # a fresh dict: the cell is never mutated by the caller's pops
        with pytest.raises(ValueError) as ei:
            resolve_cfg({"BIAS_TMA": False, "BLOCK_K": 64}, D, opd, f32)
        assert "BLOCK_K" in str(ei.value)


def test_dit_exact_prebuilts_and_the_full_abi_word():
    """dit_exact ships prebuilts for two stacks (torch 2.13.0+cu130 and torch 2.7.1+cu126, sm_90a, cpython 3.11), each directory self-describing
    (manifest: so / source sha256, the load-check digests -- identical across the two builds: same bytes out on the same inputs); the face lists
    exactly those directories, refuses any other stack BY NAME (no_prebuilt_for_stack:<key>), and appends -cp<XY> to the key when the running
    interpreter is not the build's (a cpython extension module)."""
    import hashlib
    root = os.path.join(kernels.KERNELS_DIR, "apb", "dit_exact")
    dirs = sorted(d for d in os.listdir(os.path.join(root, "prebuilt")) if os.path.isfile(os.path.join(root, "prebuilt", d, "manifest.json")))
    assert dirs == sorted(A.DIT_EXACT_ABIS) == ["torch2.13.0-cu130-sm90", "torch2.7.1-cu126-sm90"] and set(A.DIT_EXACT_PYTHON) == set(dirs)
    src_sha = hashlib.sha256(open(os.path.join(root, "csrc", "dit_attn_exact.cu"), "rb").read()).hexdigest()
    digests = set()
    for d in dirs:
        man = json.load(open(os.path.join(root, "prebuilt", d, "manifest.json")))
        so = os.path.join(root, "prebuilt", d, man["so"])
        assert hashlib.sha256(open(so, "rb").read()).hexdigest() == man["so_sha256"], d
        assert man["source_sha256"] == {"dit_attn_exact.cu": src_sha} and man["arch"] == ["sm90"] and man["kernel_version"] == "7", d
        assert "torch%s-cu%s-sm90" % (man["torch"].split("+")[0], man["cuda"].replace(".", "")) == d
        assert tuple(int(x) for x in man["python"].split(".")[:2]) == A.DIT_EXACT_PYTHON[d]
        assert man["loadcheck_bit_equal_vs_sdpa_at_build"] is True and len(man["loadcheck_digests"]) == man["loadcheck_cases"] == 3
        digests.add(tuple(man["loadcheck_digests"]))
    assert len(digests) == 1                                                                                # both builds: the same output bytes on the load-check cases
    for abi in A.DIT_EXACT_ABIS:
        assert A.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi=abi).row == "dit_exact"
    for abi in ("torch2.7.1-cu126-sm90-cp312", "torch2.8.0-cu128-sm90", "torch2.7.1-cu126-sm80"):
        with pytest.raises(A.Refusal) as ei:
            A.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", samples=5, abi=abi)
        assert ei.value.kind == "no_prebuilt_for_stack:%s" % abi and ei.value.fallback == "sdpa_upcast"


def test_dit_fast_row_words_aliases_admission_and_statements():
    """Row dit_fast = the fused DiT block's carried Triton row kernels (token stream kernels.py; variant atom = atom_kernels.py), boundary
    'blockrows' with the statements row torch_dit_rows; the producing kits' lever words are aliases; fp32 | bf16 activations only (fp16 refused
    by name); a token-stream cell refuses the atom variant and v.v.; tier words resolve per op cell (the statements when a cell is unmeasured);
    the statements class computes in fp32 (fp64 with WIDE) and matches a direct evaluation."""
    import torch
    assert "dit_fast" in A.ROW_NAMES and "torch_dit_rows" in A.STOCK_ROWS and A.VARIANTS["dit_fast"] == ("atom",)
    assert A.BOUNDARY["dit_fast"] == A.BOUNDARY["torch_dit_rows"] == "blockrows" and A.STOCK_OF_BOUNDARY["blockrows"] == "torch_dit_rows"
    assert A.split_word("ditfast") == ("dit_fast", None) and A.split_word("dit_fused") == ("dit_fast", None)
    assert A.split_word("atom_fused") == A.split_word("atomfast") == A.split_word("dit_fast:atom") == ("dit_fast", "atom")
    assert A.cell_word("ditrows", op="adaln") == "ditrows_adaln_c768" and A.cell_word("atomrows", op="gate2d") == "atomrows_gate2d_c128" and A.cell_word("ditrows", op="nope") is None
    for op in A.DIT_ROW_OPS[None]:
        assert "ditrows_%s_c768" % op in A.CELL_WORDS
    for op in A.DIT_ROW_OPS["atom"]:
        assert "atomrows_%s_c128" % op in A.CELL_WORDS
    rows = A.table()["rows"]
    assert rows["dit_fast"]["class"] == "fast" and rows["dit_fast"]["fallback"] == "torch_dit_rows" and rows["torch_dit_rows"]["class"] == "stock"
    for cc in ("9.0", "8.0"):
        s = A.select(cc, "bf16", None, 800, word="ditfast", samples=5)
        assert (s.row, s.variant, s.fallback) == ("dit_fast", None, "torch_dit_rows")
        s = A.select(cc, "fp32", "atomrows_adaln2_c128", 800, word="atom_fused", samples=5)
        assert (s.row, s.variant) == ("dit_fast", "atom")
        with pytest.raises(A.Refusal) as ei:
            A.select(cc, "fp16", None, 800, word="dit_fast")
        assert ei.value.kind.startswith("dtype:fp16") and ei.value.fallback == "torch_dit_rows"
        with pytest.raises(A.Refusal) as ei:
            A.select(cc, "bf16", "ditrows_adaln_c768", 800, word="dit_fast:atom")
        assert ei.value.kind.startswith("variant:atom")
        with pytest.raises(A.Refusal):
            A.select(cc, "bf16", "atomrows_gate2d_c128", 800, word="dit_fast")
        for op in ("adaln", "gate2d"):                                            # tier words per op cell: the measured winner, or the statements by name when unmeasured
            kind = "atomrows" if op in A.DIT_ROW_OPS["atom"] else "ditrows"
            f = A.select(cc, "bf16", A.cell_word(kind, op=op), 800, word="fast", samples=5)
            assert f.row in ("dit_fast", "torch_dit_rows")
            e = A.select(cc, "bf16", A.cell_word(kind, op=op), 800, word="exact", samples=5)
            assert e.row == "torch_dit_rows"                                      # tolerance-class kernels never serve the exact word
    with pytest.raises(ValueError):
        A.dit_block_rows(word="fast", cc="9.0")                                   # a tier word needs op=
    mod, sel = A.dit_block_rows(word="torch_dit_rows", cc="9.0", dtype="fp32")
    assert mod is A.torch_dit_rows and sel.row == "torch_dit_rows"
    # the statements: fp32 arithmetic, row-modulo conditioning, named out dtype; WIDE subclass = fp64
    T = A.torch_dit_rows; T64 = type("T64", (T,), {"WIDE": True})
    g = torch.Generator().manual_seed(3)
    a = torch.randn(10, 8, generator=g); x1 = torch.randn(5, 8, generator=g); x2 = torch.randn(5, 8, generator=g)
    y = T.adaln(a, x1, x2, torch.float32)
    ref = torch.sigmoid(x1.repeat(2, 1)) * torch.nn.functional.layer_norm(a, (8,), None, None, 1e-5) + x2.repeat(2, 1)
    assert y.dtype == torch.float32 and torch.allclose(y, ref, atol=1e-6) and T64.adaln(a, x1, x2, torch.float64).dtype == torch.float64
    x12 = torch.randn(6, 8, generator=g)
    assert torch.allclose(T.swiglu(x12, torch.float32), torch.nn.functional.silu(x12[:, :4]) * x12[:, 4:], atol=1e-6)
    o = torch.randn(2, 4, 3, 8, generator=g); gg = torch.randn(6, 32, generator=g)
    assert torch.allclose(T.gate(o, gg, 3, torch.float32), o.permute(0, 2, 1, 3).reshape(6, 32) * torch.sigmoid(gg), atol=1e-6)
    res = torch.randn(10, 8, generator=g); x = torch.randn(10, 8, generator=g); gl = torch.randn(5, 8, generator=g); res0 = res.clone()
    an = T.resgate_adaln(gl, x, res, x1, x2, torch.bfloat16)
    assert torch.allclose(res, torch.sigmoid(gl.repeat(2, 1)) * x + res0, atol=1e-6) and an.dtype == torch.bfloat16
    A_ = torch.randn(12, 8, generator=g); sca = torch.rand(4, 8, generator=g); sha = torch.randn(4, 8, generator=g)
    an2, kvn = T.adaln2(A_, sca, sha, sca, sha, torch.float32)
    assert torch.allclose(an2, sca.repeat(3, 1) * torch.nn.functional.layer_norm(A_, (8,), None, None, 1e-5) + sha.repeat(3, 1), atol=1e-6) and kvn.shape == an2.shape


def test_dit_fast_carried_kernels_are_int64_addressed_plain_jit():
    """Index audit + QoL: both carried row-kernel files form row offsets in int64 and carry no autotuner (one launch configuration per call)."""
    root = os.path.join(kernels.KERNELS_DIR, "apb", "ditfast")
    for f in ("kernels.py", "atom_kernels.py"):
        src = open(os.path.join(root, f), encoding="utf-8").read()
        assert "tl.int64" in src and "autotune" not in src and "@triton.jit" in src
    init = open(os.path.join(root, "__init__.py"), encoding="utf-8").read()
    assert "import" not in [l.split()[0] for l in init.splitlines() if l.strip() and not l.strip().startswith(('"', "#"))]   # the marker imports nothing


def test_nearest_is_size_only_other_dimensions_are_named_fallbacks():
    """A sample count (or timing form) no cell measured never borrows a neighbour's winner for a tier word: the boundary's stock row serves BY
    NAME with the measured sample counts listed; row words are unaffected (their cell is informational)."""
    s3 = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=3, capture=True)
    assert s3.row in A.STOCK_ROWS and "skip cells(samples_unmeasured:S3,measured:S1+S5)" in s3.reason
    s5 = A.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=5, capture=True)
    assert s5.row not in A.STOCK_ROWS                                                    # the measured S keeps its winner
    assert A.select("9.0", "bf16", "dit_h16d48", 800, word="fpf_apb", samples=3).row == "fpf_apb"


def test_weight_pack_caches_key_on_identity_and_version_never_shape_alone():
    """Multi-weight audit (the opt-in `cache=` dicts of pair_bias_planes / ln_linear): same shapes, weight sets A, B, A, C interleaved ->
    each call packs / serves the entry of THAT weight set; an in-place update (version bump) re-packs; entries pin their source tensors so a
    freed tensor's id cannot be recycled into a false hit; cache None = no caching; the dict is bounded."""
    import torch
    calls = []

    def builder(tag):
        def _b():
            calls.append(tag)
            return "packed-%s-%d" % (tag, len(calls))
        return _b
    g = torch.Generator().manual_seed(0)
    A, B, C_ = (torch.randn(16, 128, generator=g) for _ in range(3))          # same shape, three weight sets
    lnw, lnb = torch.ones(128), torch.zeros(128)
    cache = {}
    got = []
    for tag, W in (("A", A), ("B", B), ("A", A), ("C", C_)):
        got.append(A_pc(cache, tag, W, lnw, lnb, builder))
    assert calls == ["A", "B", "C"]                                            # A served from the cache the second time; B and C packed
    assert got[0] == got[2] and got[0] != got[1] and got[3].startswith("packed-C")
    assert all(ent[1][0] is W for ent, W in zip((cache[k] for k in cache), (A, B, C_)))   # entries pin their sources (ids cannot recycle)
    A.mul_(2.0)                                                               # in-place update: version bump -> re-pack
    assert A_pc(cache, "A", A, lnw, lnb, builder).startswith("packed-A") and calls == ["A", "B", "C", "A"]
    lnw2 = torch.full((128,), 3.0)                                            # another folded tensor (gamma) with the same weight: a different entry
    A_pc(cache, "A", A, lnw2, lnb, builder)
    assert calls[-1] == "A" and len(calls) == 5
    A_pc(cache, "A", A, lnw, lnb, builder, eps=1e-6)                          # eps is part of the key
    assert len(calls) == 6
    assert A_pc(None, "A", A, lnw, lnb, builder).startswith("packed-A") and len(calls) == 7   # cache None: build every call
    small = {}
    for i in range(A_mod.PACK_CACHE_MAX + 5):
        A_mod.pack_cached(small, "t", (torch.empty(1),), i, lambda: i)
    assert len(small) <= A_mod.PACK_CACHE_MAX


def A_pc(cache, tag, W, lnw, lnb, builder, eps=1e-5):
    return A_mod.pack_cached(cache, "ln_proj_packed", (W, lnw, lnb), (float(eps), "cpu"), builder(tag))


A_mod = A



def test_big_word_q35_rule_every_cell():
    """`big` accepted for every pair-bias cell; == fast where the winner's peak fits the stock arm's (2% + 16 MiB), else the fastest admissible
    arm within the bound (e.g. cc 8.0 bias producers under graph replay: the fused LN->linear row's recorded peak exceeds the statement's ->
    big = the statement row while fast stays the fused row)."""
    from opt_core.kernels import apb as A
    tab = A.table(); assert "big_rule" in tab
    diff = 0
    for key, cell in tab["cells"].items():
        ref = cell["ref_stack"]; sc = cell["stacks"][ref]
        fast, stock, pk, ms = sc.get("fast"), sc.get("stock_arm"), sc.get("peak_mib", {}), sc.get("ms", {})
        b = sc.get("big")
        assert cell["big"] == b and "big_per_stack" in cell
        if fast and fast in pk and stock in pk and pk[fast] > pk[stock] * 1.02 + 16:
            assert b in pk and pk[b] <= pk[stock] * 1.02 + 16 and (b == stock or ms[b] <= ms[stock]), key
            diff += int(b != fast)
        elif fast:
            assert b == fast, key
    assert diff >= 1
    f = A.select("8.0", "bf16", "bias_c64h4", 800, word="fast", samples=1); b = A.select("8.0", "bf16", "bias_c64h4", 800, word="big", samples=1)
    assert f.row == "lnl_ln_linear" and b.row == "torch_ln_linear" and b.word == "big"
    s5 = A.select("9.0", "bf16", "dit_h16d48", 800, word="big", samples=5); assert s5.row



def test_excluded_arms_never_answer_a_tier_word_on_8_0_dit_cells():
    """cc 8.0 DiT cells carry excluded_arms (l3a, l3a:c: an engine's in-model measurement); no tier word resolves them there; the row word does."""
    from opt_core.kernels import apb as A
    tab = A.table(); assert "excluded_rule" in tab
    cells = [k for k in tab["cells"] if k.startswith("8.0|bf16|dit_h16d48|")]
    assert cells and all({"l3a", "l3a:c"} <= set(tab["cells"][k]["excluded_arms"]) <= {"l3a", "l3a:c", "sdpa:cudnn"} for k in cells)
    for N in (256, 800, 1200, 2048):
        for w in ("fast", "big", "faithful"):
            for S in (1, 5):
                s = A.select("8.0", "bf16", "dit_h16d48", N, word=w, samples=S)
                assert s.row != "l3a", (N, w, S, s.row)
    s = A.select("8.0", "bf16", "dit_h16d48", 1200, word="l3a", samples=5); assert s.row == "l3a"          # by name it serves
    assert all(set(tab["cells"][k].get("excluded_arms", [])) <= {"sdpa:cudnn"} for k in tab["cells"] if k.startswith("9.0|bf16|dit_h16d48|"))   # 9.0: no l3a exclusion



def test_atom_exact_row_registration_and_carried_bytes():
    """Row atom_exact: carried kernel.py / CELLS.json / vectors.json / NOTICE / README.md digests == META; exact class, windowed boundary,
    fp32 / H%4 / D32 / 32x128 windows admitted, bf16 and other geometries refused BY NAME -> sdpa_gather; the exact word resolves it only where a
    cell vouches it on the caller's stack (none yet: exact -> the statement / today's exact row)."""
    import hashlib, json, os
    from opt_core.kernels import apb as A
    meta = json.load(open(os.path.join(os.path.dirname(A.__file__), "..", "META", "apb.json")))
    root = os.path.join(os.path.dirname(A.__file__), "atom_exact")
    for f in ("kernel.py", "CELLS.json", "vectors.json", "NOTICE", "README.md"):
        assert hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest() == meta["carried_digests"]["atom_exact/" + f], f
    assert "atom_exact" in A.ROW_NAMES and "atom_exact" in A.EXACT_ROWS and A.BOUNDARY["atom_exact"] == "windowed"
    row = A.table()["rows"]["atom_exact"]; assert row["class"] == "exact" and row["fallback"] == "sdpa_gather" and row["capture_safe"] is True
    s = A.select("9.0", "fp32", "atom_h4d32w32x128", 800, word="atom_exact", samples=5, heads=4, head_dim=32); assert s.row == "atom_exact"
    for kw, kind in ((dict(dtype="bf16"), "dtype:"), (dict(heads=6), "geometry:"), (dict(head_dim=64), "geometry:")):
        dt = kw.pop("dtype", "fp32")
        try:
            A.select("9.0", dt, "atom_h4d32w32x128", 800, word="atom_exact", samples=5, **{"heads": 4, "head_dim": 32, **kw}); raise AssertionError("expected refusal %s" % kind)
        except A.Refusal as r:
            assert r.kind.startswith(kind) and r.fallback == "sdpa_gather", (r.kind, r.fallback)
    e = A.select("9.0", "fp32", "atom_h4d32w32x128", 800, word="exact", samples=5, heads=4, head_dim=32, stack="H100:torch2.13.0+cu130/3.7.1/cueq0.11.1")
    assert e.row != "atom_exact" or "atom_exact" in (A.table()["cells"].get(e.cell, {}).get("vouched_on", {}))      # exact word: only where vouched
    src = open(os.path.join(root, "kernel.py")).read(); assert "def fused_local_attention" in src and "def determine_routes" in src and "def make_local_attention" in src



def test_fpf_apb_refuses_per_sample_bias_and_key_mask_by_name_and_samples_select_the_s_class_cell():
    """S > 1 with one bias plane set PER SAMPLE ([S,H,N,N]) or a key mask row per sample: the fused shared-bias row refuses BY NAME before any
    kernel import (fallback dtk_loop, which loops the samples); select(samples=5) resolves the S5 cell (never the S1 cell's row)."""
    import pytest, torch
    from opt_core.kernels import apb as P
    sel5 = P.select("9.0", "bf16", "dit_h16d48", 2000, word="fast", samples=5)
    assert "|S5|" in sel5.cell and sel5.row == "fpf_apb", (sel5.cell, sel5.row)
    sel1 = P.select("9.0", "bf16", "dit_h16d48", 2000, word="fast", samples=1)
    assert "|S1|" in sel1.cell
    S, N, H, D = 5, 16, 16, 48
    q = torch.zeros(S, N, H, D, dtype=torch.bfloat16); k = torch.zeros_like(q); v = torch.zeros_like(q)
    fused = P.Selection(**dict(sel5._asdict())) if hasattr(sel5, "_asdict") else sel5
    with pytest.raises(P.Refusal) as r:
        P.pair_bias_attention(q, k, v, torch.zeros(S, H, N, N), word="fpf_apb", selection=fused)          # per-sample planes
    assert r.value.kind.startswith("bias_per_sample") and r.value.row == "fpf_apb" and r.value.fallback == "dtk_loop", (r.value.kind, r.value.fallback)
    with pytest.raises(P.Refusal) as r2:
        P.pair_bias_attention(q, k, v, torch.zeros(1, H, N, N), key_mask=torch.ones(S, N), word="fpf_apb", selection=fused)   # shared planes, per-sample mask
    assert r2.value.kind.startswith("key_mask_per_sample") and r2.value.fallback == "dtk_loop", r2.value.kind



def test_noreg_unmeasured_capability_inherits_the_nearest_measured_columns_portable_rows():
    """NO-REGRESSION rule for a capability the table never measured FOR A KEY (12.0; 11.0 hypothetically): fast | big inherit the nearest
    arch-compatible measured column for that key (10.3 / 10.0 where the parts sweep measured it, else 9.0) -- Triton source rows first, never
    a prebuilt of another arch (dit_exact, atom_exact), never the SDPA statement while a portable row exists; exact = the statement by name;
    measured columns byte-identical."""
    from opt_core.kernels import apb as A
    cols = A.measured_columns(); assert {"9.0", "8.0"} <= set(cols) and "12.0" not in cols
    for cc in ("12.0", "11.0"):
        for cell, dt, S, ok_rows in (("dit_h16d48", "bf16", 5, ("fpf_apb", "l3a", "dtk_loop", "apb_attn")), ("dit_h16d48", "fp32", 5, ("fpf_apb", "l3a", "dtk_loop", "sba", "apb_attn")),
                                    ("pf_h16d24", "bf16", 1, ("fpf_apb", "l3a", "dtk_loop", "apb_attn")), ("atom_h4d32w32x128", "fp32", 1, ("fpf_atom", "dtk_window"))):
            colk = A.inherit_column(cc, (dt, cell, "eager")); assert colk in ("10.3", "10.0", "9.0"), (cc, cell, colk)
            for w in ("fast", "big"):
                s = A.select(cc, dt, cell, 800, word=w, samples=S)
                assert s.row in ok_rows, (cc, cell, dt, w, s.row, s.reason)
                assert s.reason.startswith("inherited_cc:unmeasured(%s->%s)" % (cc, colk)) and s.cell.startswith(colk + "|"), s.reason
        e = A.select(cc, "fp32", "dit_h16d48", 800, word="exact", samples=5)
        assert e.row == "sdpa" and e.reason.startswith(A.INHERIT_TOKEN), (e.row, e.reason)
        r = A.select(cc, "bf16", "dit_h16d48", 800, word="apb_attn", samples=5)
        assert r.row == "apb_attn" and not r.reason.startswith(A.INHERIT_TOKEN)
    assert A.inherit_column("8.9", ("bf16", "dit_h16d48", "eager")) == "8.0" and A.inherit_column("9.0") is None
    cells = A.table()["cells"]
    for key in list(cells)[:400]:
        cc, dt, cw, sw, nb, tm, pas = key.split("|")
        for word in ("fast", "exact", "big"):
            try:
                a = A._select(cc, dt, cw, int(nb[3:]), word=word, samples=int(sw[1:]), timing=tm)
            except A.Refusal as x:
                a = ("refusal", x.kind)
            try:
                b_ = A._select_measured(cc, dt, cw, int(nb[3:]), word=word, samples=int(sw[1:]), timing=tm)
            except A.Refusal as x:
                b_ = ("refusal", x.kind)
            assert a == b_, (key, word, a, b_)
def test_noreg_partial_cc_one_injected_cell_still_inherits_every_other_key(monkeypatch):
    """cc 10.3 with ONE injected cell (pairformer bf16 S1 only): the DiT / atom / block-row keys still inherit 9.0's portable rows (token
    +partial_cc), the injected key decides for itself, measured columns byte-identical."""
    import copy
    from opt_core.kernels import apb as A
    tab = dict(A.table()); tab["cells"] = dict(tab["cells"]); cells = tab["cells"]
    monkeypatch.setattr(A, "table", lambda: tab)                                          # the injected table object is the one the face reads (cache reloads elsewhere cannot drop it)
    if hasattr(A, "_FAMILIES"):
        monkeypatch.setattr(A, "_FAMILIES", {})                                            # the family memo starts clean (an earlier query of this key may have cached 'no family')
    src = "9.0|bf16|pf_h16d24|S1|N<=800|eager|fwd"; inj = "12.0|bf16|pf_h16d24|S1|N<=800|eager|fwd"
    assert src in cells and inj not in cells and "12.0" not in A.measured_columns()
    cells[inj] = copy.deepcopy(cells[src])
    try:
        assert A.inherit_column("12.0") is None and A.inherit_column("12.0", ("bf16", "dit_h16d48", "eager")) in ("10.3", "10.0", "9.0") and A.inherit_column("12.0", ("bf16", "pf_h16d24", "eager")) is None
        for cellw, dt, S, want in (("dit_h16d48", "bf16", 5, ("fpf_apb", "l3a", "apb_attn", "dtk_loop")), ("dit_h16d48", "fp32", 5, ("fpf_apb", "sba", "l3a", "apb_attn")), ("atom_h4d32w32x128", "fp32", 1, ("fpf_atom", "dtk_window")), ("ditrows_adaln_c768", "bf16", 5, ("dit_fast",))):
            colk = A.inherit_column("12.0", (dt, cellw, "eager"))
            for w in ("fast", "big"):
                s = A.select("12.0", dt, cellw, 800, word=w, samples=S)
                assert s.row in want and s.reason.startswith("inherited_cc:unmeasured(12.0->%s);partial_cc" % colk) and s.cell.startswith(colk + "|"), (cellw, dt, w, s.row, s.reason)
        e = A.select("12.0", "fp32", "dit_h16d48", 800, word="exact", samples=5)
        assert e.row == "sdpa" and e.reason.startswith(A.INHERIT_TOKEN)
        own = A.select("12.0", "bf16", "pf_h16d24", 800, word="fast", samples=1)
        assert own.cell == inj and not own.reason.startswith(A.INHERIT_TOKEN)
        for key in [k for k in cells if k.startswith(("9.0|", "8.0|"))][:200]:
            cc, dt, cw, sw, nb, tm, pas = key.split("|")
            for word in ("fast", "exact", "big"):
                try:
                    a = A._select(cc, dt, cw, int(nb[3:]), word=word, samples=int(sw[1:]), timing=tm)
                except A.Refusal as x:
                    a = ("refusal", x.kind)
                try:
                    b_ = A._select_measured(cc, dt, cw, int(nb[3:]), word=word, samples=int(sw[1:]), timing=tm)
                except A.Refusal as x:
                    b_ = ("refusal", x.kind)
                assert a == b_, (key, word)
    finally:
        del cells[inj]



def test_noreg_inherited_row_launch_guard_serves_the_next_row_then_the_statement(monkeypatch):
    """kernels.apb: an inherited portable row raising at launch (RuntimeError) -> retired for the part (census once), next portable row, SDPA
    statement last; no raw exception; second call skips the dead rows."""
    import torch
    from opt_core.kernels import apb as A

    class _Boom(object):
        Unsupported = RuntimeError
        def __getattr__(self, name):
            def f(*a, **k):
                raise RuntimeError("no kernel image is available for execution on the device (simulated)")
            return f
    orig = A.inherit_column
    monkeypatch.setattr(A, "inherit_column", lambda cc, family=None: "9.0" if A.cc_word(cc) == "0.0" else orig(cc, family))
    monkeypatch.setattr(A, "carried_module", lambda *a, **k: _Boom())
    monkeypatch.setattr(A, "_imp", lambda name, row, fb: _Boom() if name != "opt_core.attn.sdpa_bias" else __import__("importlib").import_module(name))
    monkeypatch.setattr(A, "_DEAD_ARMS", {})
    A._CENSUS.reset()
    S, N, H, D = 5, 16, 16, 48
    q = torch.randn(S, N, H, D); bias = torch.zeros(1, H, N, N)
    o, sel = A.pair_bias_attention(q, q, q, bias, word="fast", cell="dit_h16d48")
    assert sel.row in A.STOCK_ROWS and sel.reason.startswith(A.INHERIT_TOKEN) and tuple(o.shape) == (S, N, H, D), (sel.row, sel.reason)
    dead = A.dead_arms("0.0"); assert dead
    toks = [e for e in A._CENSUS.table() if "stepped_aside:error:" in str(e.get("refused"))]
    assert len(toks) == len(dead) and "RuntimeError" in dead.values(), dead
    o2, sel2 = A.pair_bias_attention(q, q, q, bias, word="fast", cell="dit_h16d48")
    assert sel2.row == sel.row and A.dead_arms("0.0") == dead and len([e for e in A._CENSUS.table() if "stepped_aside:error" in str(e.get("refused"))]) == len(dead)



def test_no_tier_word_answers_the_cudnn_sdpa_backend_anywhere():
    """The cuDNN SDPA backend refuses the pair-bias head geometries in-model on the measured stacks (harness-only timings): no cell names it as a
    fast / big / faithful / exact winner on any stack, cells that measured it carry it in excluded_arms, and no tier word resolves it; the
    row word sdpa:cudnn still serves by name."""
    from opt_core.kernels import apb as A
    tab = A.table()["cells"]
    for k, c in tab.items():
        for w in ("fast", "big", "faithful", "exact"):
            assert c.get(w) != "sdpa:cudnn", (k, w)
            assert "sdpa:cudnn" not in (c.get(w + "_per_stack") or {}).values(), (k, w)
            assert all(sc.get(w) != "sdpa:cudnn" for sc in c.get("stacks", {}).values()), (k, w)
    for k in [k for k in tab if k.startswith(("8.0|bf16|pf_h16d24|S1|", "9.0|bf16|pf_h16d24|S1|", "8.0|bf16|dit_h16d48|S1|")) and "|eager|" in k]:
        cc, dt, cw, sw, nb = k.split("|")[:5]
        for w in ("fast", "big", "exact"):
            for st in list(tab[k]["stacks"]) + [None]:
                s = A.select(cc, dt, cw, int(nb[3:]), word=w, samples=int(sw[1:]), stack=st)
                assert (s.row, s.variant) != ("sdpa", "cudnn"), (k, w, st)
    s = A.select("8.0", "bf16", "pf_h16d24", 800, word="sdpa:cudnn", samples=1)
    assert (s.row, s.variant) == ("sdpa", "cudnn")



def test_strided_caller_views_are_relaid_or_refused_by_name_never_asserted(monkeypatch):
    """A pair bias handed as a permuted VIEW of the pair activation (stride(-1) != 1, the S=1 sampler form) is re-laid contiguous before the
    row kernel's launch (token relaid:bias_key_stride on the Selection; output equals the contiguous-bias call), and an assertion raised under
    the dispatch (the carried modules assert their operand contracts) surfaces as a named Refusal with the row's fallback -- CPU, stand-in
    modules; no assert on a caller's layout leaves the face."""
    import math, torch
    from opt_core.kernels import apb as A
    S, N, H, D = 1, 16, 4, 32
    q, k, v = torch.randn(S, N, H, D), torch.randn(S, N, H, D), torch.randn(S, N, H, D)
    z = torch.randn(N, N, H)
    bias_view = z.permute(2, 0, 1)[None]
    assert bias_view.stride(-1) != 1
    seen = {}

    class FakePA(object):                                                     # asserts unit key stride like the carried module, computes the statement
        @staticmethod
        def apb_views(q_, k_, v_, b3, g=None, scale=None, out_dtype=None, opd=None, cfg=None, **kw):
            assert b3.stride(2) == 1, b3.stride()
            seen["stride"] = b3.stride()
            att = torch.softmax(torch.einsum("snhd,smhd->shnm", q_ * (scale or 1 / math.sqrt(q_.shape[-1])), k_) + b3[None], -1)
            return torch.einsum("shnm,smhd->snhd", att, v_).to(out_dtype or q_.dtype)
    orig = A.carried_module
    monkeypatch.setattr(A, "carried_module", lambda row: FakePA if row == "fpf_apb" else orig(row))
    monkeypatch.setattr(A, "_RELAID", {})
    sel = A.select("9.0", "fp32", "dit_h16d48", 800, word="fpf_apb", samples=1)
    o_view, s1 = A.pair_bias_attention(q, k, v, bias_view, word="fpf_apb", selection=sel)
    o_cont, s2 = A.pair_bias_attention(q, k, v, bias_view.contiguous(), word="fpf_apb", selection=sel)
    assert torch.equal(o_view, o_cont) and seen["stride"][-1] == 1
    assert "relaid:bias_key_stride" in s1.reason and "relaid" not in (s2.reason or "") and A.relaid() == {("fpf_apb", "bias_key_stride"): 1}
    qv = q.transpose(1, 3).contiguous().transpose(1, 3)                       # q with a non-unit last stride: re-laid too
    assert qv.stride(-1) != 1
    o_q, s3 = A.pair_bias_attention(qv, k, v, bias_view.contiguous(), word="fpf_apb", selection=sel)
    assert torch.allclose(o_q, o_cont) and "relaid:qkv_last_stride" in s3.reason

    class BoomPA(object):
        @staticmethod
        def apb_views(*a, **kw):
            assert False, "bias shape (H, N, >=N) expected"
    monkeypatch.setattr(A, "carried_module", lambda row: BoomPA if row == "fpf_apb" else orig(row))
    try:
        A.pair_bias_attention(q, k, v, bias_view.contiguous(), word="fpf_apb", selection=sel)
        assert False, "an assertion under the dispatch must surface as a Refusal"
    except A.Refusal as r:
        assert r.kind.startswith("layout:bias_shape") and r.row == "fpf_apb" and r.fallback
    for fn in (A.pair_bias_attention, A.atom_attention, A.pair_bias_planes):   # the three serving doors carry the guard
        assert getattr(fn, "__wrapped__", None) is not None, fn.__name__
    src = open(A.__file__).read()
    assert "\n    assert " not in src.split("def pair_bias_attention(")[1].split("\ndef reference(")[0]      # no assert statements in the serving paths of the face
