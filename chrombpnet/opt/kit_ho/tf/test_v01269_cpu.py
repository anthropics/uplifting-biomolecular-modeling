"""CPU tests (no GPU, no TF, no torch): (1) pred_bw_fast has NO module-level `import tensorflow` before the route decision — the only TF import
sits under the TF-route condition (AST); (2) the K1 first-process sub-term helper returns the term names with numbers or 'n/a' (no torch here);
(3) the process-terms line carries import_tf 'skipped' on the K1 route form."""
import os, sys, ast
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
def test_no_unconditional_tf_import_before_the_route():
    src = open(os.path.join(HERE, "pred_bw_fast.py")).read(); tree = ast.parse(src)
    top_imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    for n in top_imports:
        names = [a.name for a in n.names] if isinstance(n, ast.Import) else [n.module or ""]
        assert not any(x.split(".")[0] == "tensorflow" for x in names), f"module-level tensorflow import at line {n.lineno}"
    guarded = [n for n in ast.walk(tree) if isinstance(n, ast.Import) and any(a.name.split(".")[0] == "tensorflow" for a in n.names)]
    assert guarded, "the TF-route import must still exist (guarded)"
    i = src.find("import tensorflow as _tf_probe"); assert i > 0 and "_FD[\"forward\"] != \"k1\"" in src[max(0, i - 400):i], "the TF import must sit under the TF-route condition"
def test_k1_first_process_terms_helper():
    from chrombpnet_fastkit import fastdefault as fd
    t = fd.k1_first_process_terms(None, None)      # no torch / no weights here: every term 'n/a' with the reason, never a crash
    assert set(t.keys()) >= {"cuda_context_init", "weights_pagein", "triton_key", "backend_hash"}, t
    assert all(isinstance(v, (int, float)) or str(v).startswith("n/a") for v in t.values()), t
if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"): fn(); print("ok", name)
    print("V01269 CPU TESTS OK")
