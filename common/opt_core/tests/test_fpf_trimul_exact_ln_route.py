"""fpf_trimul: the exact row's LayerNorm stages are the installed library's own op, called at run time (cfg ``ln_mode = 'stock'``); the fast row
uses the package's LN kernels (``_k_ln_rows`` / ``_k_ln_cols_T``).  The package carries ONE LayerNorm implementation of its own and no alternate
exact-LayerNorm path: no ``ln_exact`` cfg key, no second kernel restating the library op's arithmetic, no restatement of that op's launch
heuristic.  Held on the source text (CPU; nothing of the package is imported, so no torch / triton is needed)."""
import ast
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "opt_core", "kernels", "fpf_trimul")
ABSENT_NAMES = ("_k_ln_exact", "cueq_ln_heuristic", "ln_exact_rows", "ln_exact_cols")      # no such symbol anywhere in the package
ABSENT_FILES = ("exact_ln.py",)
LIBRARY_LN = "from cuequivariance_ops_torch.fused_layer_norm_torch import layer_norm_transpose"


def _py_files():
    return sorted(os.path.join(PKG, f) for f in os.listdir(PKG) if f.endswith(".py"))


def _src(name):
    with open(os.path.join(PKG, name), encoding="utf-8") as fh:
        return fh.read()


def test_package_files():
    present = set(os.listdir(PKG))
    assert {"kernels.py", "trimul.py", "engines.py"} <= present, sorted(present)
    assert not present & set(ABSENT_FILES), sorted(present & set(ABSENT_FILES))


def test_no_alternate_exact_layernorm_path():
    word = re.compile(r"\b(?:%s)\b" % "|".join(map(re.escape, ABSENT_NAMES + ("ln_exact",))))
    hits = []
    for path in _py_files():
        with open(path, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if word.search(line):
                    hits.append("%s:%d: %s" % (os.path.basename(path), n, line.strip()[:100]))
    assert not hits, hits
    tree = ast.parse(_src("kernels.py"))
    defined = {node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert not defined & set(ABSENT_NAMES), sorted(defined & set(ABSENT_NAMES))
    ln_kernels = sorted(n for n in defined if n.startswith("_k_ln"))
    assert ln_kernels == ["_k_ln_cols_T", "_k_ln_rows"], ln_kernels                        # the package's own two LN kernels, nothing else


def test_exact_mode_layernorm_is_the_library_call():
    # trimul.MODES["exact"] carries ln_mode="stock" (read from the source: the module imports torch at top level)
    tree = ast.parse(_src("trimul.py"))
    modes = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MODES" for t in n.targets)]
    assert len(modes) == 1
    table = modes[0].value
    assert isinstance(table, ast.Dict)
    entry = dict(zip([k.value for k in table.keys], table.values))
    assert set(entry) == {"exact", "fast"}, sorted(entry)
    exact_kw = {kw.arg: kw.value.value for kw in entry["exact"].keywords if isinstance(kw.value, ast.Constant)}
    assert exact_kw.get("ln_mode") == "stock" and exact_kw.get("pad") == 1, exact_kw
    fast_kw = {kw.arg for kw in entry["fast"].keywords}
    assert "ln_mode" not in fast_kw, fast_kw                                                  # fast: the package's LN kernels (the default ln_mode)
    # kernels.trimul_forward: the ln_mode == "stock" branches call the library op; both LayerNorm sites do
    ksrc = _src("kernels.py")
    assert ksrc.count(LIBRARY_LN) == 2, ksrc.count(LIBRARY_LN)
    assert ksrc.count('cfg.get("ln_mode", "fpf") == "stock"') == 2
