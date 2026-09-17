"""The exact-class row ``triattn_exact`` (kernels/triattn/exact_member.py over the carried package kernels/triattn_exact/), on the CPU:
the carried directory is held to its checkpoint record; the row is additive (every other row's served answers are the golden digest's business,
tests/test_h200_exact_vouch_rows.py); its select-time refusal words come from metadata alone; the EXACT word never serves it on a stack without
a recorded vouch; the serve path hands a typed refusal to the kit's stock callable for that call and counts it; nothing of the package or torch
is imported by the face or the glue at module import."""
import hashlib
import re
import json
import os
import subprocess
import sys
import textwrap

import pytest

from opt_core import kernels as KS
from opt_core.kernels import triattn as T
from opt_core.kernels.triattn import exact_member as EM

OPT_CORE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(KS.__file__))))
VOUCHED = ["9.0|torch2.10.0+cu128|cueq0.10.0+cu12", "9.0|torch2.12.0+cu130|cueq0.10.0+cu13", "9.0|torch2.13.0+cu130|cueq0.10.0+cu13", "9.0|torch2.13.0+cu130|cueq0.11.1+cu12", "9.0|torch2.13.0+cu130|cueq0.11.1+cu13", "9.0|torch2.7.1+cu126|cueq0.10.0+cu12", "9.0|torch2.7.1+cu128|cueq0.10.0+cu12", "9.0|torch2.10.0+cu128|cueq0.10.0+cu12|H200", "9.0|torch2.12.0+cu130|cueq0.10.0+cu13|H200", "9.0|torch2.13.0+cu130|cueq0.10.0+cu13|H200", "9.0|torch2.13.0+cu130|cueq0.11.1+cu12|H200", "9.0|torch2.13.0+cu130|cueq0.11.1+cu13|H200", "9.0|torch2.7.1+cu126|cueq0.10.0+cu12|H200", "9.0|torch2.7.1+cu128|cueq0.10.0+cu12|H200", "8.0|torch2.10.0+cu128|cueq0.10.0+cu12", "8.0|torch2.12.0+cu130|cueq0.10.0+cu13", "8.0|torch2.13.0+cu130|cueq0.10.0+cu13", "8.0|torch2.13.0+cu130|cueq0.11.1+cu12", "8.0|torch2.13.0+cu130|cueq0.11.1+cu13", "8.0|torch2.7.1+cu126|cueq0.10.0+cu12", "8.0|torch2.7.1+cu128|cueq0.10.0+cu12"]   # the row's recorded vouch keys (7 stacks x H100 / H200-tagged / A100 80GB)
VOUCHED_BASE = sorted({re.sub(r"\+cu1[23](?=\||$)", "", k) for k in VOUCHED})                     # as a running process names its stack (kernels.triattn.exact_stack_key): no build suffix, device tag kept                                       # as a running process names its stack (kernels.triattn.exact_stack_key)


def test_row_is_carried_and_listed():
    assert "triattn_exact" in T.ROW_NAMES and "triattn_exact" in T.NEEDS_STOCK and "triattn_exact" in T.KERNEL_ROWS
    assert "triattn_exact" not in T.STOCK_ROWS and "triattn_exact" not in T.TRITON_ROWS
    R = T.rows()["triattn_exact"]
    assert R["class"] == "exact" and R["fallback"] == "cueq" and R["backward"] is False and R["capture_safe"] is True
    assert R["exact_vs"].split(" ")[0] == "cueq"
    assert R["cc"] == ["9.0", "8.0"] and R["dtypes"] == ["bf16"] and R["head_dims"] == [32]
    assert R["lib_versions"] == ["0.10.0", "0.11.1"] and R["ops_builds"] == ["cu13", "cu12"]
    assert R["vouched_on"] == VOUCHED == T.table()["words"]["exact_vouch"]["keys"]["triattn_exact"]
    assert all(k.split("|")[0] in ("9.0", "8.0") and k.count("|") in (2, 3) and re.search(r"\+cu1[23](\|H200)?$", k) for k in VOUCHED)
    assert KS.sums("triattn_exact")["kind"] == "package" and os.path.isdir(EM.PKG_DIR) and EM.PKG_DIR == KS.carried_path("triattn_exact")


def test_carried_directory_matches_its_checkpoint_record():
    up = EM.upstream()
    assert set(up) >= {"checkpoint", "files", "routes", "cells", "prebuilt", "route_fingerprints"} and len(up["checkpoint"]) == 40
    assert up["routes"] == ["cuda_mma", "_prebuilt"]
    for rel, digest in up["files"].items():
        p = os.path.join(EM.PKG_DIR, rel)
        assert os.path.isfile(p), rel
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == digest, f"{rel}: bytes differ from UPSTREAM.json"
    on_disk = {os.path.relpath(os.path.join(r, f), EM.PKG_DIR) for r, ds, fs in os.walk(EM.PKG_DIR) for f in fs if "__pycache__" not in r}
    assert on_disk == set(up["files"]) | {"UPSTREAM.json", "README.md"}, sorted(on_disk ^ (set(up["files"]) | {"UPSTREAM.json", "README.md"}))
    for route in up["routes"]:
        assert os.path.isfile(os.path.join(EM.PKG_DIR, route, "__init__.py")), route
    assert os.path.isfile(os.path.join(EM.PKG_DIR, "csrc", "cuda_mma", "triattn_v3.cu")) and os.path.isfile(os.path.join(EM.PKG_DIR, "_prebuilt", "cuda_mma.py"))
    man = json.load(open(os.path.join(EM.PKG_DIR, "_prebuilt", "manifest.json"), encoding="utf-8"))
    names = lambda xs: {os.path.basename(x) for x in xs}                        # noqa: E731
    assert names(b["blob"] for b in man["builds"]) == names(b["blob"] for b in up["prebuilt"]) == names(f for f in up["files"] if f.startswith("_prebuilt/blobs/")) != set()
    for b in man["builds"]:                                                      # build records carry consumed keys only
        assert set(b) <= {"route", "kernel", "abi", "arch", "sm", "arch_specific", "toolkit", "nvcc_flags", "blob", "blob_sha256", "cubin_sha256",
                          "cubin_bytes", "kernels", "cfg_table", "source_fingerprint", "tag"}, sorted(b)
    for route, rec in up["route_fingerprints"].items():                        # the table certifies THIS directory's bytes
        for c in EM.proven_cells():
            if c["route"] == route and c.get("route_fingerprint"):
                assert c["route_fingerprint"] == {"sha256": rec["sha256"]}, c["id"]


def test_cell_table_carries_consumed_keys_only_and_counts_match_the_record():
    doc = json.load(open(EM.CELLS_FILE, encoding="utf-8"))
    assert set(doc) == {"cells"}
    for c in doc["cells"]:
        assert set(c) <= {"id", "status", "route", "when", "binary_fingerprint", "route_fingerprint"} and c["status"] == "proven", c
        assert c["route"] in EM.upstream()["routes"], c
    up = EM.upstream()["cells"]
    assert up == {"proven": sum(1 for c in doc["cells"] if c["status"] == "proven"), "total": len(doc["cells"])}
    assert EM.proven_cells() == [c for c in doc["cells"] if c["status"] == "proven"]


@pytest.mark.parametrize("args, word", [
    (((10, 0), "bf16", 32, 4, 512), "cc 10.0"),
    (((12, 0), "bf16", 32, 4, 512), "cc 12.0"),
    (((9, 0), "fp32", 32, 4, 512), "dtype_fp32"),
    (((9, 0), "fp16", 32, 4, 512), "dtype_fp16"),
    (((9, 0), "bf16", 64, 4, 512), "head_dim_64"),
    (((8, 0), "bf16", 16, 4, 512), "head_dim_16"),
    (((9, 0), "bf16", 32, 4, 64), "below_min_tokens_101"),
    (((8, 0), "bf16", 32, 4, 100), "below_min_tokens_101"),
])
def test_admits_refuses_by_metadata(args, word):
    ok, why = T.admits("triattn_exact", *args)
    assert not ok and why.startswith(word), (args, why)


def test_admits_no_backward_and_structural_ok_offline():
    ok, why = T.admits("triattn_exact", (9, 0), "bf16", 32, 4, 512, "fwdbwd")
    assert not ok and why == "no_backward"
    ok, why = T.admits("triattn_exact", (9, 0), "bf16", 32, 4, 512)
    assert ok and (why == "library: unverified offline" or why.startswith("library ")), why       # the rig may or may not carry the library


def test_library_gate_words(monkeypatch):
    R = T.rows()["triattn_exact"]
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.11.1", "cu11"))
    assert EM.library_gate(R) == (False, "ops_build:cu11")
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.9.0", "cu13"))
    assert EM.library_gate(R) == (False, "lib_version:0.9.0")
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.10.0", "cu13"))
    assert EM.library_gate(R) == (True, "library 0.10.0 cu13")
    monkeypatch.setattr(EM, "ops_distribution", lambda: None)
    assert EM.library_gate(R) == (True, "library: unverified offline")
    T.select_cache_clear()
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.11.1", "cu11"))
    with pytest.raises(T.Refusal) as e:
        T.select((9, 0), "bf16", 32, 4, 1024, word="triattn_exact", exact_stack="9.0|torch2.13.0+cu130|cueq0.11.1")
    assert e.value.kind == "ops_build:cu11" and e.value.fallback == "cueq"
    T.select_cache_clear()


@pytest.mark.parametrize("cc", [(9, 0), (8, 0)])
@pytest.mark.parametrize("stack", ["9.0|torch2.13.0+cu130|cueq0.11.1", "8.0|torch2.12.0+cu130|cueq0.10.0", "9.0|torch9.9.9+cu999|cueq0.11.1", None])
def test_named_row_serves_only_on_a_vouched_stack_and_names_the_stock_op_elsewhere(cc, stack):
    T.select_cache_clear()
    if stack is None:                                        # an offline table query (no running stack named): the row answers structurally, class exact vs cueq
        sel = T.select(cc, "bf16", 32, 4, 1024, word="triattn_exact", exact_stack=None)
        assert sel.row == "triattn_exact" and sel.cls == "exact" and sel.exact_vs.split(" ")[0] == "cueq" and sel.backward is False
        return
    if "%d.%d" % cc != stack.split("|")[0]:
        pytest.skip("a process names its own card in its stack key")
    if stack in VOUCHED_BASE:                                                  # a vouched (cc, torch, library) stack: the named word selects the row
        sel = T.select(cc, "bf16", 32, 4, 1024, word="triattn_exact", exact_stack=stack)
        assert sel.row == "triattn_exact" and sel.cls == "exact", (cc, stack, sel)
        return
    with pytest.raises(T.Refusal) as e:
        T.select(cc, "bf16", 32, 4, 1024, word="triattn_exact", exact_stack=stack)
    assert e.value.kind.startswith("exact_vouch_not_recorded") and e.value.row == "triattn_exact" and e.value.fallback == "cueq", (cc, stack, e.value.kind)


def test_exact_word_lands_on_the_row_only_in_its_vouched_cells_on_vouched_stacks():
    """The EXACT word selects the row exactly where a cell names it the exact winner AND the running stack is vouched; everywhere else it answers
    what the cells' other recorded winners say (cueq / exact_headsplit / stock) or passes the row over by name."""
    T.select_cache_clear()
    stacks = sorted({k for ks in T.table()["words"]["exact_vouch"]["keys"].values() for k in ks} | set(VOUCHED_BASE)) + [None, "9.0|torch2.13.0+cu130|cueq0.11.1|H200"]
    landed = expected = 0
    for key, cell in T.cells().items():
        cc, dt, D, H, n = key.split("|")[:5]
        ccT = tuple(int(x) for x in cc.split("."))
        n_tokens = int(n.replace("N<=", ""))
        for st in stacks:
            may = cell.get("exact") == "triattn_exact" and (st is None or (st in VOUCHED_BASE and st.startswith(cc + "|")))   # None: a table query naming no running stack answers the cell's winner
            if st is not None and not st.startswith(cc + "|"):
                continue                                                     # a process names its own card in its stack key
            try:
                sel = T.select(ccT, dt, int(D[1:]), int(H[1:]), n_tokens, word="exact", exact_stack=st)
                assert (sel.row == "triattn_exact") == may, (key, st, sel.row)
                landed += sel.row == "triattn_exact"; expected += bool(may)
            except T.Refusal as r:
                assert r.row != "triattn_exact" or r.kind.startswith("exact_vouch_not_recorded"), (key, st, r.kind)
    flipped = [k for k, c in T.cells().items() if c.get("exact") == "triattn_exact"]            # cells naming the row their exact winner 
    carrying = [k for k, c in T.cells().items() if "triattn_exact" in (c.get("vouched_on") or {})]
    assert len(carrying) == 66 and set(flipped) == set(carrying) and all(re.match(r"(9|8)\.0\|bf16\|D32\|H(2|4|8|12)\|N<=\d+\|fwd$", k) for k in carrying), carrying
    assert all(int(k.split("|")[3][1:]) in (2, 4, 8, 12) and int(k.split("|")[4][3:]) <= 4096 for k in carrying)
    assert landed == expected > 0, (landed, expected)


def test_serve_hands_a_typed_refusal_to_the_stock_callable_and_counts_it(monkeypatch):
    torch = pytest.importorskip("torch")

    class Refused(RuntimeError):
        def __init__(self, reason, cell=None):
            self.reason = reason; self.cell = cell or {}
            super().__init__(f"triattn_exact refused: {reason}")

    class Face:
        calls = 0

        @staticmethod
        def triangle_attention(q, k, v, bias, mask=None, scale=None, kv_lengths=None):
            Face.calls += 1
            if q.shape[3] <= 100:
                raise Refused(f"S={q.shape[3]} <= CUEQ_TRIATTN_FALLBACK_THRESHOLD=100: the library takes its torch path here")
            if q.shape[2] == 3:
                raise Refused("no proven cell covers this call")
            return v.clone()

    monkeypatch.setitem(EM._STATE, "face", Face)
    monkeypatch.setattr(EM, "refused_class", lambda: Refused)
    EM.reset_counts()
    seen = []

    def stock(q, k, v, bias, mask=None, scale=None):
        seen.append((q.shape, scale)); return q.clone()

    q = torch.zeros(1, 4, 4, 128, 32); k = torch.ones_like(q); v = torch.full_like(q, 2.0); b = torch.zeros(1, 1, 4, 128, 128)
    out = EM.serve(q, k, v, b, None, 0.25, stock)
    assert torch.equal(out, v) and seen == [] and EM.counts()["served"] == 1
    qs = torch.zeros(1, 4, 4, 64, 32); bs = torch.zeros(1, 1, 4, 64, 64)
    out = EM.serve(qs, qs, qs, bs, None, 0.25, stock)
    assert torch.equal(out, qs) and seen == [(qs.shape, 0.25)]
    q3 = torch.zeros(1, 4, 3, 128, 32); b3 = torch.zeros(1, 1, 3, 128, 128)
    EM.serve(q3, q3, q3, b3, None, None, stock); EM.serve(q3, q3, q3, b3, None, None, stock)
    c = EM.counts()
    assert c["served"] == 1 and c["refused"] == {"small_s": 1, "no_proven_cell": 2} and c["calls"] == 4
    assert c["cells"] == "carried" and c["selfcheck"] == "on" and EM.primed()
    assert EM.evidence_line() == "triattn_exact: served 1/4 calls (refused: {no_proven_cell: 2, small_s: 1})"
    monkeypatch.setenv("TRIATTN_EXACT_SKIP_SELFCHECK", "1"); monkeypatch.setenv("TRIATTN_EXACT_CELLS", "/elsewhere/my_cells.json")
    assert EM.evidence_line() == "triattn_exact: served 1/4 calls (refused: {no_proven_cell: 2, small_s: 1}) cells=external:my_cells.json selfcheck=SKIPPED"
    EM.reset_counts()
    assert EM.counts()["calls"] == 0


def test_capture_before_first_launch_refuses_by_name(monkeypatch):
    torch = pytest.importorskip("torch")

    class Face:
        @staticmethod
        def triangle_attention(q, k, v, bias, mask=None, scale=None, kv_lengths=None):
            return v.clone()

    monkeypatch.setitem(EM._STATE, "face", Face)
    monkeypatch.setitem(EM._STATE, "primed", False)
    monkeypatch.setattr(EM, "refused_class", lambda: RuntimeError)
    monkeypatch.setattr(EM, "_capturing", lambda t: True)             # as if a CUDA graph were being captured on this stream
    EM.reset_counts()
    served_by_stock = []
    stock = lambda q, k, v, bias, mask=None, scale=None: served_by_stock.append(1) or q.clone()   # noqa: E731
    q = torch.zeros(1, 4, 4, 128, 32); v = torch.ones_like(q); b = torch.zeros(1, 1, 4, 128, 128)
    assert torch.equal(EM.serve(q, q, v, b, None, None, stock), q) and served_by_stock == [1]
    assert EM.counts()["refused"] == {"capture_first_use": 1} and not EM.primed()
    monkeypatch.setattr(EM, "_capturing", lambda t: False)            # outside capture the member launches and primes the process
    assert torch.equal(EM.serve(q, q, v, b, None, None, stock), v) and EM.primed()
    monkeypatch.setattr(EM, "_capturing", lambda t: True)             # a later capture replays a launched member: served
    assert torch.equal(EM.serve(q, q, v, b, None, None, stock), v) and EM.counts()["served"] == 2
    EM.reset_counts()


def test_vouch_keys_carry_the_ops_build_for_this_row_only(monkeypatch):
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.11.1", "cu13"))
    assert T.vouch_keys("triattn_exact", "9.0|torch2.13.0+cu130|cueq0.11.1") == ["9.0|torch2.13.0+cu130|cueq0.11.1+cu13"]
    assert T.vouch_keys("triattn_exact", "9.0|torch2.13.0+cu130|cueq0.11.1|H200") == ["9.0|torch2.13.0+cu130|cueq0.11.1+cu13|H200"]
    assert T.vouch_keys("exact_headsplit", "9.0|torch2.13.0+cu130|cueq0.11.1|H200") == ["9.0|torch2.13.0+cu130|cueq0.11.1|H200"]
    assert T.vouch_keys("triattn_exact", None) == []
    monkeypatch.setattr(EM, "ops_distribution", lambda: None)          # offline: any declared build's spelling
    assert T.vouch_keys("triattn_exact", "8.0|torch2.12.0+cu130|cueq0.10.0") == ["8.0|torch2.12.0+cu130|cueq0.10.0+cu13", "8.0|torch2.12.0+cu130|cueq0.10.0+cu12"]
    cell = {"vouched_on": {"triattn_exact": ["9.0|torch2.13.0+cu130|cueq0.11.1+cu13"], "exact_headsplit": ["9.0|torch2.13.0+cu130|cueq0.11.1"]}}
    assert T.vouched("triattn_exact", cell, "9.0|torch2.13.0+cu130|cueq0.11.1") and not T.vouched("triattn_exact", cell, "9.0|torch2.13.0+cu130|cueq0.11.1|H200")
    assert T.vouched("exact_headsplit", cell, "9.0|torch2.13.0+cu130|cueq0.11.1") and not T.vouched("exact_headsplit", {}, "1.0|torchX|cueqY")
    monkeypatch.setattr(EM, "ops_distribution", lambda: ("0.11.1", "cu12"))   # a cu12 install never matches a cu13 vouch
    assert not T.vouched("triattn_exact", cell, "9.0|torch2.13.0+cu130|cueq0.11.1")


def test_meta_digests_are_the_checkpoint_record():
    doc = KS.sums("triattn_exact")
    assert doc["carried_digests"] == EM.upstream()["files"]


def test_provider_serve_branch_reaches_the_glue(monkeypatch):
    torch = pytest.importorskip("torch")
    got = {}
    monkeypatch.setitem(T._READY, "triattn_exact", {"probe": "test"})
    monkeypatch.setattr(EM, "serve", lambda q, k, v, bias, mask, sc, stock: got.setdefault("args", (q.shape, sc, stock)) and v)
    sel = T.Selection("triattn_exact", "triattn_exact", "9.0|bf16|D32|H4|N<=1200|fwd", None, None, "exact", "cueq", True, False, True, None, True, "test")
    q = torch.zeros(1, 2, 4, 256, 32, dtype=torch.bfloat16); b = torch.zeros(1, 1, 4, 256, 256)
    stock = lambda *a, **k: None                                                                    # noqa: E731
    out = T.triangle_attention(q, q, q, b, None, word="triattn_exact", stock=stock, selection=sel)
    assert out is q and got["args"][0] == q.shape and abs(got["args"][1] - 32 ** -0.5) < 1e-12 and got["args"][2] is stock
    with pytest.raises(T.Refusal) as e:                                                             # a NEEDS_STOCK row without stock= refuses by name
        T.triangle_attention(q, q, q, b, None, word="triattn_exact", selection=sel)
    assert e.value.kind == "needs_stock"


def test_reason_words():
    assert EM.reason_word("no proven cell covers this call") == "no_proven_cell"
    assert EM.reason_word("S=64 <= CUEQ_TRIATTN_FALLBACK_THRESHOLD=100: the library takes its torch path here") == "small_s"
    assert EM.reason_word("input requires grad (forward-only implementation)") == "autograd"
    assert EM.reason_word("route 'cuda_mma' refuses: q/k/v rows must be contiguous in D") == "route_refuses"
    assert EM.reason_word("q base address not 16-byte aligned (the library itself faults on such views)") == "layout"
    assert EM.reason_word("self-check FAILED earlier in this process: max|diff|=1") == "selfcheck_failed"
    assert EM.reason_word("Something Else: odd") == "something_else_odd"


def test_nothing_heavy_at_import_and_the_route_serves_the_core_copy():
    """Fresh interpreter: importing the face + glue imports neither torch nor the package; load() routes `triattn_exact` to the core copy,
    repoints the member's sources under the package and the cell table at the carried file (torch is needed for load itself)."""
    code = textwrap.dedent(f"""
        import os, sys, json
        sys.path.insert(0, {OPT_CORE_ROOT!r})
        from opt_core.kernels import triattn as T
        from opt_core.kernels.triattn import exact_member as EM
        out = {{"torch_at_import": "torch" in sys.modules, "pkg_at_import": "triattn_exact" in sys.modules}}
        try:
            import torch  # noqa
        except Exception:
            print(json.dumps(dict(out, skipped="no torch"))); raise SystemExit(0)
        face = EM.load()
        import triattn_exact, triattn_exact.cuda_mma as CM
        from opt_core import kernels as KS
        out.update(face=face.__name__, where=KS.resolve("triattn_exact"), core=KS.carried_path("triattn_exact"),
                   cells=face._CELLS_PATH, src=CM._SRC_V3, refused=EM.refused_class().__name__,
                   verify=KS.verify_carry("triattn_exact"))
        print(json.dumps(out))
    """)
    env = {k_: v_ for k_, v_ in os.environ.items() if k_ not in ("TRIATTN_EXACT_CELLS", "TRIATTN_EXACT_CACHE")}
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["torch_at_import"] is False and out["pkg_at_import"] is False
    if out.get("skipped"):
        pytest.skip(out["skipped"])
    assert out["face"] == "triattn_exact.face" and out["refused"] == "Refused"
    assert os.path.samefile(out["where"], out["core"]) and out["verify"] == []
    assert os.path.samefile(out["cells"], EM.CELLS_FILE)
    assert out["src"] == os.path.join(EM.PKG_DIR, "csrc", "cuda_mma", "triattn_v3.cu") and os.path.isfile(out["src"])


def test_package_sources_hold_the_tree_vocabulary():
    """The carried package speaks release-tree vocabulary: no path climbing out of the package, no sys.path edits, no network."""
    for rel in EM.upstream()["files"]:
        if not rel.endswith(".py"):
            continue
        text = open(os.path.join(EM.PKG_DIR, rel), encoding="utf-8").read()
        assert not any(tok in text for tok in ("sys.path.insert", "sys.path.append", "urllib", "requests.", "socket.")), rel


def test_route_fingerprints_recompute_to_the_record_and_cover_the_carried_source():
    """The face's own route_fingerprint (after load() adds the carried source dir to its table) equals UPSTREAM.json's stamped value for every
    fingerprinted route, every carried proven cell certified on source carries exactly that value, and the cuda_mma fingerprint covers the .cu."""
    pytest.importorskip("torch")
    face = EM.load()
    up = EM.upstream()
    face._fp_cache.clear()
    for route, rec in up["route_fingerprints"].items():
        assert face.route_fingerprint(route) == rec["sha256"], route
        assert rec["files"], route
    assert EM.SOURCE_FP_LOGICAL in up["route_fingerprints"]["cuda_mma"]["files"]
    stamped = {(c["route"], c["route_fingerprint"]["sha256"]) for c in EM.proven_cells() if c.get("route_fingerprint")}
    assert stamped and stamped <= {(r, rec["sha256"]) for r, rec in up["route_fingerprints"].items()}, stamped
    br = up["bridge"]
    assert set(br) == set(up["files"]) and all(b["carried_sha256"] == up["files"][rel] for rel, b in br.items())
    assert all(b["transform"] == "verbatim" for rel, b in br.items() if rel.startswith("_prebuilt/blobs/") or rel == "_prebuilt/driver.py")
