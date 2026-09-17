"""The LayerNorm binding's dialect needles (of3_exactln.DIALECTS) identify THIS tree's stock LayerNorm.forward: the matcher run on the
statement's source text under stock/ picks the expected dialect with nothing missing — a needle that drifts from the stock statement would make the
binding step aside by name at model import on every line (CPU-checkable here, without torch or a GPU)."""
import os
import re

from openfold3_opt import of3_exactln as X

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK = os.path.normpath(os.path.join(HERE, "..", "..", "..", "stock", "src", "openfold3", "core", "model", "primitives", "normalization.py"))
EXPECTED = "cast16"


def _forward_source() -> str:
    t = open(STOCK, encoding="utf-8").read()
    i = t.index("class LayerNorm")
    j = t.find("\nclass ", i + 10)
    blk = t[i:(j if j > 0 else len(t))]
    return blk[blk.index("def forward"):]


def test_the_needles_identify_the_stock_statement():
    src = _forward_source()
    for name, want in X.DIALECTS.items():
        missing = [w for w in want if w not in src]
        if name == EXPECTED:
            assert missing == [], (name, missing)
        else:
            assert missing, (name, "a second dialect must not also match this tree's statement")


def test_the_matcher_picks_the_expected_dialect_on_the_stock_source():
    class _F:
        pass
    fn = _F()
    import inspect
    orig = inspect.getsource
    try:
        inspect.getsource = lambda o: _forward_source()
        X._innermost = getattr(X, "_innermost", lambda f: f)
        got, why = X._dialect_of(fn)
    finally:
        inspect.getsource = orig
    assert (got, why) == (EXPECTED, ""), (got, why)


def test_every_operand_tuple_the_binding_unpacks_has_the_arity_it_names():
    """`w_, b_, od = (...) if affine else (...)`: every literal tuple assigned to a tuple of names in the binding (both branches of a conditional
    expression included) carries exactly as many elements as there are names — the cast16 dialect's bf16 operand triple included, so a bf16 call
    that reaches a provider row binds (bf16 weight, bf16 bias, no out dtype) instead of raising inside LayerNorm.forward."""
    import ast
    tree = ast.parse(open(X.__file__, encoding="utf-8").read())
    checked = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Tuple):
            names = len(node.targets[0].elts)
            for v in ([node.value.body, node.value.orelse] if isinstance(node.value, ast.IfExp) else [node.value]):
                if isinstance(v, ast.Tuple):
                    assert len(v.elts) == names, f"{os.path.basename(X.__file__)}:{node.lineno}: {names} names <- a {len(v.elts)}-tuple"
                    checked.append(node.lineno)
    assert len(checked) >= 6, checked                     # the three dialect/form operand lines, both branches each
