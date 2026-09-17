"""The shipped entry v17/borzoi_sad.py is exactly build.py's render(): the pinned stock script (from stock/'s archive, sha256 asserted) plus
the kit's anchored insertions — no stock line edited. An edit to the entry that build.py does not reproduce fails here."""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE = os.path.dirname(HERE)


def _build():
    spec = importlib.util.spec_from_file_location("pipeline_tf_build", os.path.join(PIPELINE, "build.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_entry_is_stock_plus_insertions():
    b = _build()
    shipped = open(os.path.join(PIPELINE, b.FROZEN, b.ENTRY), encoding="utf-8").read()
    assert b.render() == shipped


def test_insertions_only_add_lines():
    b = _build()
    stock = b.stock_source()
    rendered = b.render()
    added = sum(block.count("\n") for _, _, block in b.INSERTIONS)
    assert rendered.count("\n") == stock.count("\n") + added
    for where, anchor, block in b.INSERTIONS:
        assert stock.count(anchor) == 1 and anchor in rendered
        assert block.lstrip().startswith("# ---- ") and "kit" in block.split("\n", 1)[0]
