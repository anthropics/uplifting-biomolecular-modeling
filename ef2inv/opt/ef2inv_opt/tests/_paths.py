"""Paths of the tree for the tests (the package location is the tree: opt/ef2inv_opt -> ef2inv/)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))          # ef2inv/
OPT = os.path.join(ROOT, "opt")
FWD = os.path.join(OPT, "forward")
DK = os.path.join(FWD, "ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1")
KDIR = os.path.join(DK, "k")
SDK_STANDIN = os.path.join(OPT, "ef2inv_opt", "_absent_sdk_stub.py")              # the kit's stand-in for the cloud SDK the cookbook imports (modes.SDK_STANDIN)
STOCK_FILE = os.path.join(ROOT, "stock", "src", "cookbook", "tutorials", "binder_design.py")
PINS = json.load(open(os.path.join(ROOT, "stock", "PINS.json")))


def stock_literals(names):
    """The stock file's top-level ``NAME = <literal>`` assignments for ``names`` (parsed with ast, nothing imported)."""
    import ast
    tree = ast.parse(open(STOCK_FILE).read())
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in names}
