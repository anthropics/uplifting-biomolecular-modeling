"""0.1.4 (delta against release-branch 454a76ff): digest vs build metadata, pae_stream monomer-only line, check-verb kernel probe,
AFO_STOCK_DET removed."""
import importlib.util, os, shutil, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

def _check_pins():
    spec = importlib.util.spec_from_file_location("check_pins", os.path.join(KIT, "stock", "check_pins.py")); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def test_tree_digest_ignores_build_metadata_but_not_source_edits():
    cp = _check_pins()
    src = os.path.join(KIT, "stock", "src")
    d0 = cp.tree_digest(src)
    tmp = tempfile.mkdtemp(prefix="afo_src_"); root = os.path.join(tmp, "src"); shutil.copytree(src, root, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.egg-info", "build", "dist", ".eggs"))   # a real `run.sh install` may have left src/atlasfold.egg-info in the tree: excluded from the copy as from the digest
    try:
        assert cp.tree_digest(root) == d0
        # what `pip install -e stock/src` leaves behind on a networked machine + bytecode + a build dir
        os.makedirs(os.path.join(root, "src", "atlasfold.egg-info"), exist_ok=True); open(os.path.join(root, "src", "atlasfold.egg-info", "PKG-INFO"), "w").write("Metadata-Version: 2.1\n")
        os.makedirs(os.path.join(root, "build", "lib"), exist_ok=True); open(os.path.join(root, "build", "lib", "x.py"), "w").write("x = 1\n")
        os.makedirs(os.path.join(root, "src", "atlasfold", "__pycache__"), exist_ok=True); open(os.path.join(root, "src", "atlasfold", "__pycache__", "a.cpython-311.pyc"), "wb").write(b"\0")
        assert cp.tree_digest(root) == d0, "build metadata / bytecode must not move the digest"
        # a source edit must move it
        py = next(os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.join(root, "src", "atlasfold")) for f in fn if f.endswith(".py"))
        open(py, "a").write("\n# edit\n")
        assert cp.tree_digest(root) != d0, "a source edit must move the digest"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def test_pae_stream_is_monomer_only_and_says_na_on_multimer():
    from atlasfold.model.network import confidence_head as CH
    from atlasfold_opt.hooks import pae as P
    ins = P.install("exact", "atlasfold-opt", {})
    try:
        assert ins.applied and ins.facts.get("scope") == "monomer-only"
        assert hasattr(CH.ConfidenceHead_Multimer.forward, "__wrapped_stock__")      # passthrough wired (counts, changes nothing)
        assert "reason=" not in ins.lines[0]() or "no_calls" in ins.lines[0]()         # no calls yet: the core's own wording
        ins.facts["multimer_calls"][0] += 3                                            # as if AtlasFold-M's head ran 3 times
        line = ins.lines[0]()
        assert "state=skipped" in line and "reason=n/a:monomer_only" in line and "x3" in line, line
    finally:
        CH.ConfidenceHead_Monomer.forward = CH.ConfidenceHead_Monomer.forward.__wrapped_stock__
        CH.ConfidenceHead_Multimer.forward = CH.ConfidenceHead_Multimer.forward.__wrapped_stock__
        from atlasfold.model.utils import confidence_metrics as CM; import importlib; importlib.reload(CM)

def test_check_verb_kernel_probe_routes_before_route_check():
    from atlasfold_opt.cli import kernel_probe
    rows = kernel_probe()
    assert rows and all(r["ok"] is True for r in rows if r["ok"] is not None), rows
    assert not any(r["kernel"] == "fpf_trimul_v4" for r in rows), rows                                          # tri-mul binds opt_core.kernels.trimul by tier word: the carried v4 route is not probed
    assert any(r["kernel"] == "fpf_trimul" and r.get("routed") for r in rows), rows
    assert "FPF_TRIMUL_V4_CELLS" not in os.environ or "atlasfold_opt" not in os.environ["FPF_TRIMUL_V4_CELLS"]   # no kit cells table is exported: the provider's own table serves

def test_stock_subprocess_recipe_has_no_AFO_STOCK_DET():
    from atlasfold_opt import det as D
    assert D.env_for_subprocess(1) == {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"} and D.env_for_subprocess(0) == {}
    src = open(os.path.join(KIT, "opt", "atlasfold_opt", "cli.py")).read()
    assert "AFO_STOCK_DET" not in src
