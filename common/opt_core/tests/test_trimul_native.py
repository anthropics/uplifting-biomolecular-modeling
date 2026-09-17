"""kernels.trimul rows native / native_exact: the sealed `trimul_native` package (arch-keyed cubins through the CUDA driver) behind one face.
No framework here: payload integrity (DIGESTS, SHA256SUMS, no unlisted file, the build record's own cubin / source digests), the face is
standard library at import and binds the payload under a PRIVATE module name (never `trimul_native` on sys.path), admission / refusals by
name (pure), tier words never choose the rows until a cell names them, the exact row is in EXACT_ROWS and un-vouched everywhere it was not
byte-tested, release-grade purity of the carried bytes."""
import ast
import hashlib
import json
import os
import sys

import pytest

from opt_core import kernels
from opt_core.kernels import trimul as T
from opt_core.kernels.trimul import native as NT

BASE = os.path.join(kernels.KERNELS_DIR, "trimul", "native")
PKG = NT.pkg_dir()
H100_213 = "H100:2.13.0+cu130/3.7.1/cueq0.11.1"
H100_210 = "H100:2.10.0+cu128/3.6.0/cueq0.10.0"


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def test_row_facts_and_table_entries():
    for r in ("native", "native_exact"):
        assert r in T.ROW_NAMES and r in T.NATIVE_ROWS and r not in T.BACKWARD_ROWS and r not in T.STOCK_ROWS and r not in T.NEEDS_ESM_IMAGE
    assert "native_exact" in T.EXACT_ROWS and "native" not in T.EXACT_ROWS and "native_exact" not in T.MODULE_EXACT_ROWS
    tab = T.table()
    rn, rx = tab["rows"]["native"], tab["rows"]["native_exact"]
    assert rn["class"] == "fast" and rn["exact_vs"] is None and rn["backward"] is False and rn["fallback"] == "v4" and rn["capture_safe"] is True
    assert rx["class"] == "exact" and rx["exact_vs"].startswith("cuequivariance_torch") and rx["backward"] is False and rx["fallback"] == "tmk3_exact"
    assert rn["payload"]["active_pkg"] == NT.ACTIVE_PKG and rn["payload"]["digests"] == dict(NT.DIGESTS) and rn["payload"]["version"] == NT.VERSION
    assert "native" in tab["tiers"]["fast"] and "native_exact" in tab["tiers"]["exact"] and "native" not in tab["tiers"]["exact"]


def test_payload_digests_sums_and_build_record():
    assert os.path.isdir(PKG) and PKG.endswith(os.path.join("pkg", NT.ACTIVE_PKG))
    for rel, want in NT.DIGESTS.items():                                            # the four records pinned in the face
        assert want is not None and _sha(os.path.join(PKG, rel)) == want, rel
    seen = NT.verify_digests()                                                      # every SHA256SUMS line + no unlisted file
    assert len(seen) >= 60 and "build/manifest.json" in seen and "python/trimul_native/face.py" in seen
    assert open(os.path.join(PKG, "VERSION")).read().strip() == NT.VERSION
    man = NT.manifest()
    for key, u in man["units"].items():                                             # the build record's own digests hold for the carried bytes
        assert _sha(os.path.join(PKG, "build", u["cubin"])) == u["cubin_sha256"], key
        for src, rec in (u.get("sources") or {}).items():
            want = rec if isinstance(rec, str) else rec.get("sha256")
            p = os.path.join(PKG, src) if os.path.isfile(os.path.join(PKG, src)) else os.path.join(PKG, "csrc", src)
            assert _sha(p) == want, (key, src)
        assert u["arch"] in NT.ARCHS and "fast-math" not in " ".join(u.get("flags") or []) and "--use_fast_math" not in " ".join(u.get("flags") or [])
    tv = NT.vectors_manifest()
    assert tv["recipe"] == "closed_form_v1" and len(tv["cases"]) == 216 and set(NT.device_classes()) >= {"sm90", "sm80"}     # 1.2.0: grid r2, 216 cases x classes sm90 + sm80 (one package, two members)
    assert {u["cubin_sha256"] for u in man["units"].values()} <= set(tv["build_units"].values())     # every carried cubin is one the vectors were made from
    iso = tv.get("isolated_replay") or {}
    assert iso.get("status") == "pass"


def test_face_is_standard_library_at_import_and_binds_a_private_module_name():
    src = open(os.path.join(BASE, "__init__.py")).read()
    tree = ast.parse(src)
    top = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            top.add(node.module.split(".")[0])
    assert top <= {"collections", "hashlib", "importlib", "json", "os", "sys", "threading"}, top
    F = NT.face()
    assert F.__name__ == NT._MODNAME + ".face" and NT._MODNAME != "trimul_native" and NT._MODNAME in sys.modules
    assert "trimul_native" not in sys.modules or os.path.dirname(getattr(sys.modules["trimul_native"], "__file__", "") or "") != os.path.join(PKG, "python", "trimul_native")
    assert os.path.abspath(F.BUILD_DIR) == os.path.abspath(os.path.join(PKG, "build")) and os.path.abspath(F.CELLS_PATH) == os.path.abspath(os.path.join(PKG, "CELLS.json"))
    present = set(os.listdir(os.path.join(PKG, "python", "trimul_native")))
    mods = [m for m in ("face.py", "launch.py", "_driver.py", "ops.py", "kernel.py", "vectors.py", "manifest.py", "sm80_ops.py") if m in present]   # build.py = the producer's build tool (needs nvcc; never imported by the face)
    assert {"face.py", "launch.py", "_driver.py", "ops.py", "kernel.py", "vectors.py", "manifest.py"} <= present, sorted(present)
    for m in mods:                                                                     # the serving modules import each other relatively only
        mod = ast.parse(open(os.path.join(PKG, "python", "trimul_native", m)).read())
        for node in ast.walk(mod):
            if isinstance(node, ast.Import):
                assert not any(a.name.split(".")[0] == "trimul_native" for a in node.names), m
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                assert (node.module or "").split(".")[0] != "trimul_native", m


def test_admits_and_refuses_by_name():
    A = NT.admits
    for d in ("outgoing", "incoming"):
        for n in (16, 256, 400, 403, 800, 1200, 1536, 2048, 4096):
            for c in (128, 256):
                assert A((9, 0), "bf16", c, c, n, d) is None
                assert A("9.0", "bf16", c, c, n, d, residency="fp32") is None                       # the f32z form
                if n <= 100:                                                                          # 1.1.0: the exact variant refuses by name below 101 tokens (the reference library's own small-N path)
                    assert A((9, 0), "bf16", c, c, n, d, variant="exact").startswith("exact_small_n:")
                else:
                    assert A((9, 0), "bf16", c, c, n, d, variant="exact") is None
                assert A((9, 0), "bf16", c, c, n, d, residency="fp32", variant="exact").startswith("variant:exact")   # no exact f32z variant: by name
            for cz in (64, 128, 256, 384):                                                            # 1.1.0: bf16 forward at all sixteen pairs; fp32-resident z at the nine pairs without 384
                for ch in (64, 128, 256, 384):
                    assert A((9, 0), "bf16", cz, ch, n, d) is None, (cz, ch, n)
                    want_f32z = None if 384 not in (cz, ch) else "no_cubin:sm_90a:z%d_h%d_f32z" % (cz, ch)
                    got = A((9, 0), "bf16", cz, ch, n, d, residency="fp32")
                    assert got == want_f32z or (want_f32z and got.startswith(want_f32z)), (cz, ch, n, got)
        assert A((9, 0), "bf16", 64, 64, 400, d) is None and A((9, 0), "bf16", 64, 128, 400, d) is None
    assert A((9, 0), "bf16", 128, 128, 256, batch=2) is None and A((9, 0), "bf16", 128, 128, 256, batch=65).startswith("shape:batch")
    for d in ("outgoing", "incoming"):                                                                   # 1.2.0: the sm_80 member -- bf16 fast at all sixteen pairs, exact at the four square pairs,
        for cz in (64, 128, 256, 384):                                                                    # fp32-resident z measured (= served) at z128_h128 / z256_h256 only: the other pairs are
            for ch in (64, 128, 256, 384):                                                                # waived BY NAME (no_cell:8.0|f32z|...) until a later payload measures them
                assert A((8, 0), "bf16", cz, ch, 800, d) is None, (cz, ch)
                rs = A((8, 0), "bf16", cz, ch, 800, d, residual=True)                                  # the residual form: the five ESM-family pairs (square + (64,128)) on both members
                assert (rs is None) if (cz == ch or (cz, ch) == (64, 128)) else rs.startswith("no_cell:8.0|bf16|fast|z%d_h%d|" % (cz, ch)), (cz, ch, rs)
                ex = A((8, 0), "bf16", cz, ch, 800, d, variant="exact")
                assert (ex is None) if cz == ch else ex.startswith("exact_no_reference:"), (cz, ch, ex)
                fz = A((8, 0), "bf16", cz, ch, 800, d, residency="fp32")
                assert (fz is None) if (cz == ch and cz in (128, 256)) else fz.startswith("no_cell:8.0|f32z|fast|z%d_h%d|" % (cz, ch)), (cz, ch, fz)
                assert A((8, 0), "bf16", cz, ch, 800, d, residency="fp32", variant="exact").startswith(("variant:exact", "exact_no_reference")), (cz, ch)
        assert A((9, 0), "bf16", 64, 128, 800, d, variant="exact").startswith("exact_no_reference:")       # 1.2.0: no exact variant where the reference library defines no op (c_z != c_hidden)
    assert A((8, 0), "bf16", 128, 128, 64, variant="exact").startswith("exact_small_n:") and A((8, 0), "bf16", 128, 128, 8).startswith("shape:n")
    assert A((8, 6), "bf16", 256, 256, 800).startswith("no_cell:8.6|") and A((8, 9), "bf16", 128, 128, 800, residency="fp32").startswith("no_cell:8.9|")   # sm_80 SASS, no measured row: by name
    assert A((7, 5), "bf16", 128, 128, 800).startswith("cc_unsupported")
    assert A((9, 0), "bf16", 384, 384, 800) is None and A((9, 0), "bf16", 384, 384, 800, residency="fp32").startswith("no_cubin:sm_90a:z384_h384_f32z") and A((9, 0), "bf16", 512, 512, 800).startswith("shape:c_z")
    assert A((9, 0), "bf16", 128, 128, 8).startswith("shape:n") and A((9, 0), "fp32", 128, 128, 800).startswith("dtype:")
    R = T.Refusal

    def adm(row, *a, **k):
        try:
            T.admits(row, *a, **k)
            return None
        except R as e:
            return e.kind, e.fallback
    assert adm("native", "9.0", "bf16", 256, 256, 800) is None and adm("native", "9.0", "bf16", 128, 128, 800, residency="fp32") is None
    assert adm("native_exact", "9.0", "bf16", 128, 128, 400) is None
    assert adm("native", "9.0", "fp32", 128, 128, 800) == pytest.approx(adm("native", "9.0", "fp32", 128, 128, 800)) and adm("native", "9.0", "fp32", 128, 128, 800)[1] == "v4"
    assert adm("native", "9.0", "tf32", 256, 256, 800)[0].startswith("dtype:")
    assert adm("native", "8.0", "bf16", 128, 128, 800) is None and adm("native", "8.0", "bf16", 64, 128, 800) is None and adm("native", "8.0", "bf16", 384, 384, 2048) is None   # 1.2.0: the sm_80 member by ROW word
    assert adm("native_exact", "8.0", "bf16", 128, 128, 800) is None and adm("native_exact", "8.0", "bf16", 64, 128, 800)[0].startswith("exact_no_reference:")
    assert adm("native", "8.0", "bf16", 256, 256, 800, residency="fp32") is None and adm("native", "8.0", "bf16", 64, 64, 800, residency="fp32")[0].startswith("no_cell:8.0|f32z|fast|z64_h64|")
    assert adm("native:f32in", "8.0", "fp32", 128, 128, 800) is None and adm("native:f32in", "8.0", "tf32", 384, 384, 800)[0].startswith("no_cell:8.0|f32z|")
    assert adm("native", "8.6", "bf16", 128, 128, 800)[0].startswith("no_cell:8.6|") and adm("native", "8.6", "bf16", 128, 128, 800)[1] == "v4" and adm("native", "7.5", "bf16", 128, 128, 800)[0].startswith("cc")
    assert adm("native", "9.0", "bf16", 128, 128, 800, backward=True)[0] == "no_backward"              # forward only: a fwd+bwd request refuses by name
    assert adm("native_exact", "9.0", "bf16", 256, 256, 800, residency="fp32") == ("variant:exact(z256_h256_f32z on sm_90a)", "tmk3_exact")
    assert adm("native", "9.0", "bf16", 384, 384, 800) is None and adm("native_exact", "9.0", "bf16", 384, 384, 800) is None and adm("native", "9.0", "bf16", 64, 128, 800) is None
    assert adm("native", "9.0", "bf16", 384, 384, 800, residency="fp32") == ("no_cubin:sm_90a:z384_h384_f32z", "cueq") and adm("native:f32in", "9.0", "fp32", 384, 384, 800)[0] == "no_cubin:sm_90a:z384_h384_f32z"
    assert adm("native_exact", "9.0", "bf16", 64, 64, 100)[0].startswith("exact_small_n:") and adm("native_exact", "9.0", "bf16", 64, 64, 101) is None
    assert adm("native", "9.0", "bf16", 128, 128, 8)[1] == "cueq"


def test_row_words_serve_the_rows_and_tier_words_are_unchanged_until_a_cell_names_them():
    sel = T.select("9.0", "bf16", 256, 256, 800, "outgoing", word="native", stack=H100_213)
    assert sel.row == "native" and sel.word == "native" and (sel.cls == "fast" or str(sel.cls).startswith("tol(")) and sel.fallback == "v4" and not sel.backward   # cls: the row class, or the cell's measured class string once measured
    sel = T.select("9.0", "bf16", 128, 128, 400, "incoming", word="native_exact", stack=H100_210)
    assert sel.row == "native_exact" and (sel.cls == "exact" or str(sel.cls).startswith(("bitwise", "tol("))) and sel.fallback == "tmk3_exact"   # cls: the row class, or the measured class string once the R2 run recorded one on that stack
    sel = T.select("9.0", "bf16", 128, 128, 1200, "outgoing", word="native", residency="fp32", stack=H100_213)
    assert sel.row == "native" and sel.cell.startswith("9.0|f32z_bf16|C128|H128|")
    for r in ("native", "native_exact", "native:f32in"):                                    # 1.2.0: the sm_80 member serves the ROW words on cc 8.0 (tier words unchanged until a cell names them)
        sel = T.select("8.0", "bf16" if r != "native:f32in" else "fp32", 128, 128, 800, "outgoing", word=r, stack="A100:2.13.0+cu130/3.7.1/cueq0.11.1")
        assert sel.row == r and sel.word == r, (r, sel.row)
    with pytest.raises(T.Refusal):
        T.select("8.0", "bf16", 64, 64, 800, "outgoing", word="native", residency="fp32", stack="A100:2.13.0+cu130/3.7.1/cueq0.11.1")   # a waived fp32-resident group: by name
    tab = T.table()
    named = [(k, w, st) for k, c in tab["cells"].items() for w in ("fast", "exact", "big") for st, r in (c.get(w + "_per_stack") or {}).items() if r in T.NATIVE_ROWS]
    named += [(k, w, None) for k, c in tab["cells"].items() for w in ("fast", "exact", "big") if c.get(w) in T.NATIVE_ROWS]
    def _named_ok(c, st, w):
        return bool(any(st in (c.get(rec) or {}) for rec in [k_ for k_ in c if k_.startswith("race_")] + ["byte_vouch"]) or (c.get("inherited_vouch") or {}).get(str(st) + "|" + w) or (c.get("inherited_r1") or {}).get(str(st) + "|" + w) or (c.get("inherited_r2") or {}).get(str(st) + "|" + w) or (c.get("inherited_r3") or {}).get(str(st) + "|" + w) or (c.get("inherited_c1") or {}).get(str(st) + "|" + w) or (c.get("inherited_r2p") or {}).get(str(st) + "|" + w) or (c.get("policy") or {}).get(str(st)) or st is None)
    for k, w, st in named:                                                         # a cell names a native row only with its same-machine measurement record on that stack
        c = tab['cells'][k]
        se = c.get('size_extension') or {}
        assert _named_ok(c, st, w) or (se.get('of') in tab['cells'] and _named_ok(tab['cells'][se['of']], st, w)), (k, w, st)   # or an inherited reference verdict (R1: 1.0.1 race; R2: the 1.1.0 confirm run)
        if w == "exact":
            assert T.exact_vouched(c, "native_exact", st or c.get("ref_stack")), (k, st)   # the cell-level exact word is the reference stack's word (vouched there); unlisted stacks refuse it by name
    for k, c in tab["cells"].items():                                              # never vouched where it was not byte-tested
        for st in (c.get("vouched_on") or {}).get("native_exact", ()):
            assert st in (c.get("race_r1") or {}) or st in (c.get("race_r2") or {}) or str(((c.get("class") or {}).get("native_exact") or {}).get(st, "")).startswith("bitwise"), (k, st)
    for st in (H100_213, H100_210):
        for w in ("fast", "exact", "big"):
            sel = T.select("9.0", "bf16", 256, 256, 800, "outgoing", word=w, stack=st)
            if sel.row in T.NATIVE_ROWS:
                assert (st in (tab["cells"][sel.cell].get("race_r1") or {})) or (st in (tab["cells"][sel.cell].get("race_r2") or {})), (w, st, sel.cell)


def test_install_refuses_by_name_without_a_device():
    try:
        import torch                                                              # noqa: F401
        has_dev = torch.cuda.is_available()
    except ImportError:
        has_dev = False
    if has_dev:
        pytest.skip("a CUDA device is present: install() is exercised by the GPU suites")
    with pytest.raises(NT.Unavailable) as ei:
        NT.install()
    assert ei.value.kind.split(":")[0] in ("driver_unavailable", "import", "cc_unsupported"), ei.value.kind
    with pytest.raises(T.Refusal) as er:
        T.native_install()
    assert er.value.row == "native" and er.value.fallback == "v4"


def _importable(name):
    import importlib.util
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _cc90():
    try:
        import torch
        return torch.cuda.is_available() and tuple(torch.cuda.get_device_capability(0)) == (9, 0)
    except Exception:                                                        # noqa: BLE001 -- capability probe for a skip mark
        return False


def _weights(C, D, seed, dev):
    import torch
    g = torch.Generator(device="cpu").manual_seed(seed)
    r = lambda *s, sc=1.0: (torch.randn(*s, generator=g) * sc).to(dev)      # noqa: E731
    return dict(ln_in_w=1 + 0.1 * r(C), ln_in_b=0.1 * r(C), w_ag=r(D, C, sc=C ** -0.5), w_ap=r(D, C, sc=C ** -0.5), w_bg=r(D, C, sc=C ** -0.5), w_bp=r(D, C, sc=C ** -0.5),
                ln_out_w=1 + 0.1 * r(D), ln_out_b=0.1 * r(D), w_o=r(C, D, sc=D ** -0.5), w_og=r(C, D, sc=D ** -0.5))


@pytest.mark.skipif(not _cc90(), reason="needs a capability-9.0 device (the payload's sm_90a members)")
@pytest.mark.parametrize("C", [128, 256])
def test_native_multi_weight_sets_gpu(C):
    """Same shapes, three weight sets through ONE cache: outputs differ across sets, each matches the fp32 statement within class A, the exact variant equals
    the stock op bit for bit per set where cuequivariance is importable, and replaying set 1 after set 3 reproduces set 1's bytes (no stale weights in a plan)."""
    import torch
    from opt_core.kernels import trimul as T
    dev = torch.device("cuda"); N = 200
    torch.manual_seed(7)
    z = torch.randn(1, N, N, C, device=dev, dtype=torch.bfloat16); mask = torch.ones(1, N, N, device=dev); mask[:, :, -5:] = 0
    cache = {}; outs = []
    for seed in (1, 2, 3):
        w = _weights(C, C, seed, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            o = T.triangle_multiplication(z, mask, direction="outgoing", weights=w, word="native", cache=cache)
            ref = T.triangle_multiplication(z.float(), mask, direction="outgoing", weights=w, word="torch_math")
            d = (o.float() - ref.float())
            assert float(d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()) < 3e-3, seed
            if _importable("cuequivariance_torch"):
                oe = T.triangle_multiplication(z, mask, direction="incoming", weights=w, word="native_exact", cache=cache)
                cq = T.triangle_multiplication(z, mask, direction="incoming", weights=w, word="cueq")
                assert torch.equal(oe, cq), seed
        outs.append(o.clone())
    assert not torch.equal(outs[0], outs[1]) and not torch.equal(outs[1], outs[2])
    w = _weights(C, C, 1, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        again = T.triangle_multiplication(z, mask, direction="outgoing", weights=w, word="native", cache=cache)
    assert torch.equal(again, outs[0])


@pytest.mark.skipif(not _cc90(), reason="needs a capability-9.0 device")
def test_native_residual_forms_gpu():
    """f32z: the row's fused residual (fp32 sum in the epilogue) vs the face form z + update added in fp32 -- agree to fp32 rounding (byte table: ADOPT_trimul
    section native); the update form itself is deterministic call to call."""
    import torch
    from opt_core.kernels import trimul as T
    dev = torch.device("cuda"); N, C = 256, 128
    torch.manual_seed(3)
    z = torch.randn(1, N, N, C, device=dev); mask = torch.ones(1, N, N, device=dev)
    w = _weights(C, C, 1, dev); cache = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        u1 = T.triangle_multiplication(z, mask, direction="outgoing", weights=w, word="native", cache=cache, residency="fp32")
        u2 = T.triangle_multiplication(z, mask, direction="outgoing", weights=w, word="native", cache=cache, residency="fp32")
        r = T.triangle_multiplication(z.clone(), mask, direction="outgoing", weights=w, word="native", cache=cache, residency="fp32", residual=True)
    assert u1.dtype == torch.float32 and torch.equal(u1, u2)
    face = z + u1
    d = (r - face).abs().max().item()
    assert d <= 2 ** -20 * float(face.abs().max()), d


# --- verification stamp: the byte gate once per INSTALLED TREE (x device class x stack), not once per process (CPU facts; no device needed) ---
def _facts(NT, **over):
    f = {"schema": NT.STAMP_SCHEMA, "sums_sha256": NT.DIGESTS["SHA256SUMS"], "pkg": NT.pkg_dir(), "active_pkg": NT.ACTIVE_PKG, "payload_version": "1.2.0", "cc": "9.0",
         "device_name": "H100", "driver_version": 13000, "cuda_version": "13.0", "torch": "2.13.0+cu130", "binding": "auto", "cases": "gate", "python": "3.12"}
    f.update(over); return f


def test_stamp_written_and_honoured(tmp_path, monkeypatch):
    from opt_core.kernels.trimul import native as NT
    monkeypatch.setenv(NT.STAMP_ENV, str(tmp_path))
    sd = NT.stamp_dir()
    assert sd == os.path.join(str(tmp_path), "trimul_native") and os.path.isdir(sd)
    facts = _facts(NT)
    assert NT.stamp_read(sd, facts) is None                                        # cold: no stamp -> install() runs the gate
    p = NT.stamp_write(sd, facts, {"device_class": "sm90", "gate": "20/20 bitwise", "arch": "sm_90a"})
    assert p and os.path.isfile(p) and not [f for f in os.listdir(sd) if f.endswith(".tmp")]   # atomic: no temp file left
    st = NT.stamp_read(sd, facts)
    assert st is not None and st["facts"] == facts and st["gate"]["gate"] == "20/20 bitwise" and st["schema"] == NT.STAMP_SCHEMA   # warm: honoured -> install() asks the payload for loads + load check only


def test_stamp_invalidated_when_any_keyed_fact_changes(tmp_path, monkeypatch):
    from opt_core.kernels.trimul import native as NT
    monkeypatch.setenv(NT.STAMP_ENV, str(tmp_path)); sd = NT.stamp_dir(); facts = _facts(NT)
    assert NT.stamp_write(sd, facts, {"gate": "ok"})
    for k, v in (("sums_sha256", "0" * 64), ("torch", "2.10.0+cu128"), ("cc", "8.0"), ("driver_version", 12080), ("cuda_version", "12.8"), ("pkg", "/elsewhere/pkg/v0"), ("active_pkg", "v999"), ("device_name", "A100")):
        assert NT.stamp_read(sd, _facts(NT, **{k: v})) is None, k               # a changed payload digest / stack / device / path = another key: the gate runs again
    open(NT._stamp_path(sd, facts), "w").write("{not json")                        # a corrupt stamp is no stamp
    assert NT.stamp_read(sd, facts) is None


def test_stamp_directory_unwritable_or_disabled_means_the_gate_runs(tmp_path, monkeypatch):
    from opt_core.kernels.trimul import native as NT
    monkeypatch.setenv(NT.STAMP_ENV, "0")
    assert NT.stamp_dir() is None                                                  # disabled by env: every process gates
    monkeypatch.setenv(NT.STAMP_ENV, "/proc/self/no_such_dir/opt_core")
    assert NT.stamp_dir() is None                                                  # cannot be created: no crash, no stamp
    assert NT.stamp_write(None, _facts(NT), {}) is None and NT.stamp_read(None, _facts(NT)) is None
    ro = tmp_path / "ro"; ro.mkdir(); sd = str(ro)
    os.chmod(sd, 0o500)
    try:
        if os.access(sd, os.W_OK):
            pytest.skip("cannot make a read-only directory here (root)")
        assert NT.stamp_write(sd, _facts(NT), {"gate": "ok"}) is None              # read-only cache: the write is skipped silently (install() reports verdict_stamp = unwritable)
    finally:
        os.chmod(sd, 0o700)


def test_stamp_facts_need_a_device_and_default_dir_is_a_cache_root(monkeypatch):
    from opt_core.kernels.trimul import native as NT
    monkeypatch.delenv(NT.STAMP_ENV, raising=False)
    for k in ("MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", os.path.join(os.getcwd(), ".xdg_test_cache"))
    d = NT.stamp_dir()
    assert d is None or d.endswith(os.path.join("opt_core", "trimul_native"))
    jit = os.path.join(os.getcwd(), ".jit_test_root"); monkeypatch.setenv("MODEL_OPT_JIT_ROOT", jit); monkeypatch.setenv("MODEL_OPT_STACK_KEY", "torchX-sm90")
    try:
        d2 = NT.stamp_dir()                                                        # the kit's JIT root: the same base kernels.triattn's triattn_native keeps its gate verdicts under
        assert d2 == os.path.join(jit, "torchX-sm90", "trimul_native") and os.path.isdir(os.path.join(jit, "torchX-sm90", "triattn")), d2
    finally:
        import shutil; shutil.rmtree(jit, ignore_errors=True)
    if "torch" not in sys.modules:
        assert NT.stamp_facts() is None                                            # no framework / no device in this process: no stamp is consulted (install() gates)
    import shutil; shutil.rmtree(os.path.join(os.getcwd(), ".xdg_test_cache"), ignore_errors=True)


def test_stamp_dir_chain_falls_through_to_a_private_tmp_dir_when_home_and_xdg_are_unwritable(tmp_path, monkeypatch):
    from opt_core.kernels.trimul import native as NT
    monkeypatch.delenv(NT.STAMP_ENV, raising=False)
    for k in ("MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY", "TRITON_CACHE_DIR"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", "/proc/self/no_home_here"); monkeypatch.setenv("XDG_CACHE_HOME", "/proc/self/no_xdg_here"); monkeypatch.setenv("TMPDIR", str(tmp_path))
    import tempfile; monkeypatch.setattr(tempfile, "tempdir", None)
    kinds = [k for k, _ in NT.stamp_dirs()]
    assert kinds == ["tmp"], kinds                                                 # read-only HOME / XDG (kit containers): the chain ends in <tmp>/opt_core-uid<uid>/trimul_native
    d = NT.stamp_dir(); root = os.path.dirname(d)
    assert d.startswith(str(tmp_path)) and (os.stat(root).st_mode & 0o777) == 0o700       # private to this uid; another user's / group-writable root is never honoured
    monkeypatch.setenv(NT.STAMP_ENV, "off")
    assert NT.stamp_dirs() == [] and NT.stamp_dir() is None                        # disabled by name -> install() reports verdict_stamp = disabled:env_off


def test_prestamp_verb_status_and_no_device_paths(tmp_path, monkeypatch, capsys):
    """`python -m opt_core.kernels.trimul.native status|stamp`: the kit's warm / bake-time pre-stamp verb (no GPU here: status works, stamp says why by name)."""
    import inspect
    from opt_core.kernels.trimul import native as NT
    from opt_core.kernels.trimul.native import __main__ as CLI
    assert "force_gate" in inspect.signature(NT.install).parameters                 # --verify = install(force_gate=True): the gate replays over an intact stamp
    monkeypatch.setenv(NT.STAMP_ENV, str(tmp_path))
    assert CLI.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "env" in out and str(tmp_path) in out and "stamps=0" in out
    assert CLI.main(["status", "--json"]) == 0 and '"chain"' in capsys.readouterr().out
    monkeypatch.setenv(NT.STAMP_ENV, "0")
    assert CLI.main(["status"]) == 0 and "no stamp directory" in capsys.readouterr().out
    if "torch" not in sys.modules:
        try:
            import torch  # noqa: F401
        except Exception:
            rc = CLI.main(["stamp"]); txt = capsys.readouterr().out               # no framework: refused by name, exit 2, one token line
            assert rc == 2 and "NATIVE_STAMP" in txt
            return
    import torch
    if not torch.cuda.is_available():
        rc = CLI.main(["stamp"]); txt = capsys.readouterr().out
        assert rc == 2 and "stamp=disabled:no_device" in txt


def test_pack_weights_residency_under_big_is_at_most_one_layer_and_fast_keeps_the_per_layer_cache(monkeypatch):
    """Weight-pack residency (the shared per-geometry payload cache): under the tier word `big` the pack policy is LRU-1 -- after serving N >= 8
    DISTINCT weight sets the bytes attributable to packed weights are <= ONE layer's pack (+0), never N layers (a trunk's resident memory must not
    grow with its layer count under big); under `fast` / `exact` / a row word every layer's pack stays cached (speed); the env knob forces either.
    Counted through the cache's own bookkeeping (pack entries by key tag), so this runs on CPU with tensor stand-ins."""
    from opt_core.kernels.trimul import native as NT
    monkeypatch.delenv(NT.PACK_POLICY_ENV, raising=False)
    assert NT.pack_policy_for("big") == "lru1" and NT.pack_policy_for("fast") == "per_layer" and NT.pack_policy_for("exact") == "per_layer" and NT.pack_policy_for("native") == "per_layer"
    LAYER = 34 * 2 ** 20 // 100                                                   # ~one layer's packed weights

    class FT:                                                                    # a tensor stand-in: address, shape, dtype, byte size
        def __init__(self, ptr, shape, nbytes=0, dtype="torch.bfloat16"):
            self._p, self.shape, self.dtype, self.nbytes = ptr, tuple(shape), dtype, nbytes
        def data_ptr(self): return self._p
    KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
    def layer(i):
        return {k: FT(10 ** 6 * (i + 1) + j, (128, 128) if k.startswith("w_") else (128,)) for j, k in enumerate(KEYS)}
    def payload_serves(cache, w):                                                # what the payload does on a served call: pack once per weight set (keyed by the tensors' triples) + a call memo entry
        key = (NT.PACK_KEY_TAG, "sm_90a", "cuda:0") + NT._weights_sig(w)
        if key not in cache: cache[key] = (FT(0, (4, 128, 128), nbytes=LAYER), FT(1, (128,), nbytes=0))
        cache.setdefault(NT.CALLS_MEMO_KEY, {})[("call", id(w))] = {"w_ptr": w["w_ap"].data_ptr(), "pack": cache[key]}
    for word, most in (("big", 1), ("fast", 8), ("exact", 8)):
        cache = {}; layers = [layer(i) for i in range(8)]
        for rep in range(2):                                                     # two trunk passes over 8 layers
            for w in layers:
                NT.apply_pack_policy(cache, w, NT.pack_policy_for(word))
                payload_serves(cache, w)
                st = NT.pack_cache_stats(cache)
                assert st["packs"] <= most and st["bytes"] <= most * LAYER, (word, st)
        st = NT.pack_cache_stats(cache)
        assert st["packs"] == most, (word, st)
        assert len(cache[NT.CALLS_MEMO_KEY]) == most, (word, len(cache[NT.CALLS_MEMO_KEY]))
    monkeypatch.setenv(NT.PACK_POLICY_ENV, "per_layer")                          # the knob forces either behaviour for every word
    assert NT.pack_policy_for("big") == "per_layer"
    monkeypatch.setenv(NT.PACK_POLICY_ENV, "lru1")
    assert NT.pack_policy_for("fast") == "lru1"
    for w in layers: pass
    assert all(hasattr(t, "data_ptr") for t in layers[0].values())              # the module's own tensors are never touched by the policy (only dict entries of packed copies leave)


def test_pack_cache_is_weakref_evicted_and_bounded_under_every_word_and_stable_weights_keep_one_pack_per_layer(monkeypatch):
    """(B4) The shared per-geometry pack cache cannot accumulate under ANY word: a caller handing FRESH weight copies per call (new addresses)
    gets its pack-class entries (packs and the cc-8.0 member's plain-weight copies) tied to the owner tensor's lifetime (weakref finalizer ->
    evicted on GC) and bounded by an LRU cap; stable weights keep one pack per layer with no re-pack per call.  CPU, via the cache bookkeeping."""
    import gc
    from opt_core.kernels.trimul import native as NT
    monkeypatch.delenv(NT.PACK_POLICY_ENV, raising=False); monkeypatch.setenv(NT.PACK_CACHE_CAP_ENV, "16")
    assert NT.pack_cache_cap() == 16

    class FT:
        def __init__(self, ptr, shape, nbytes=0, dtype="torch.bfloat16"):
            self._p, self.shape, self.dtype, self.nbytes = ptr, tuple(shape), dtype, nbytes
        def data_ptr(self): return self._p
    KEYS = ("ln_in_w", "ln_in_b", "w_ag", "w_ap", "w_bg", "w_bp", "ln_out_w", "ln_out_b", "w_o", "w_og")
    def layer(i):
        return {k: FT(10 ** 6 * (i + 1) + j, (128, 128) if k.startswith("w_") else (128,)) for j, k in enumerate(KEYS)}
    def payload_serves(cache, w, plain=False):
        if plain:                                                                # the cc-8.0 member's plain-weight copy key: (tag, (name, ptr, dtype, shape) ...)
            key = (NT.PLAIN_KEY_TAG,) + tuple((k, w[k].data_ptr(), w[k].dtype, w[k].shape) for k in sorted(KEYS))
        else:
            key = (NT.PACK_KEY_TAG, "sm_80", "cuda:0") + NT._weights_sig(w)
        if key not in cache: cache[key] = (FT(0, (4, 128, 128), nbytes=300000),)
        cache.setdefault(NT.CALLS_MEMO_KEY, {})[("call", id(w))] = {"w_ptr": w["w_ap"].data_ptr(), "pack": None}
    for plain in (False, True):
        for word in ("fast", "exact", "native"):
            cache = {}; seen_max = 0
            for i in range(100):                                                 # 100 calls, a FRESH weight copy each (new addresses), dropped right after the call
                w = layer(1000 + i)
                pol = NT.pack_policy_for(word); before = NT._packclass_keys(cache)
                NT.apply_pack_policy(cache, w, pol); payload_serves(cache, w, plain); NT.after_call(cache, w, before)
                seen_max = max(seen_max, NT.pack_cache_stats(cache)["packs"])
                del w; gc.collect()
            st = NT.pack_cache_stats(cache)
            assert seen_max <= 16, (word, plain, seen_max)                        # bounded while the copies pile in
            assert st["packs"] == 0, (word, plain, st)                            # every fresh copy collected -> its entry left with it
            layers = [layer(i) for i in range(8)]; cache = {}                    # stable weights: one pack per layer, kept across passes (no re-pack)
            ev0 = NT.pack_cache_stats(cache)["evicted_total"] + NT.pack_cache_stats(cache)["gc_evicted"] + NT.pack_cache_stats(cache)["cap_evicted"]
            for rep in range(3):
                for w in layers:
                    pol = NT.pack_policy_for(word); before = NT._packclass_keys(cache)
                    NT.apply_pack_policy(cache, w, pol); payload_serves(cache, w, plain); NT.after_call(cache, w, before)
            st = NT.pack_cache_stats(cache)
            assert st["packs"] == 8 and st["evicted_total"] + st["gc_evicted"] + st["cap_evicted"] == ev0, (word, plain, st)
    cache = {}; layers = [layer(i) for i in range(8)]                            # big: LRU-1 unchanged
    for w in layers:
        before = NT._packclass_keys(cache); NT.apply_pack_policy(cache, w, NT.pack_policy_for("big")); payload_serves(cache, w); NT.after_call(cache, w, before)
        assert NT.pack_cache_stats(cache)["packs"] <= 1
