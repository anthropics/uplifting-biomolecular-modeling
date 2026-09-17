"""The template census (templ.py): a declared template selection is one TEMPLATE line with its atom counts and the stock result is
returned unchanged (a selection that selects no atom folds untemplated, as upstream does); no selection = untouched."""
import sys
import types

import numpy as np
import pytest

from .. import templ


class FakeAtomArray:
    def __init__(self, n):
        self._ann = {}
        self.n = n

    def __len__(self):
        return self.n

    def get_annotation_categories(self):
        return list(self._ann)

    def get_annotation(self, k):
        return self._ann[k]

    def set_annotation(self, k, v):
        self._ann[k] = np.asarray(v)


def _fake_module(select_n):
    """apply_template_selection as upstream shapes it: no selection -> zeros annotation; a selection -> a mask with select_n atoms True."""
    mod = types.ModuleType("rf3.utils.inference")

    def apply_template_selection(atom_array, template_selection):
        sel = [template_selection] if isinstance(template_selection, str) else list(template_selection or [])
        mask = np.zeros(len(atom_array), dtype=bool)
        if sel:
            mask[:select_n] = True
        atom_array.set_annotation(templ.ANNOTATION, mask)
        return atom_array
    mod.apply_template_selection = apply_template_selection
    return mod


@pytest.fixture(autouse=True)
def _reset():
    templ.STATE.update({"installed": False, "declared": 0, "items": []})
    yield


def test_declared_selection_selecting_no_atom_is_a_census_line_and_folds_as_upstream_does(capsys):
    mod = _fake_module(select_n=0)
    assert templ.wrap(mod) and not templ.wrap(mod)                     # wrapped once
    arr = FakeAtomArray(10)
    assert mod.apply_template_selection(arr, ["A1-50"]) is arr         # the stock result, unchanged: the fold proceeds untemplated as upstream's does
    err = capsys.readouterr().err
    assert "[rosettafold3-opt] TEMPLATE selection=1 atoms_selected=0/10" in err and "NOT ACTIVE" not in err and "refused" not in err
    assert templ.state()["declared"] == 1 and "refused" not in templ.state()

def test_declared_selection_with_atoms_is_a_named_census(capsys):
    mod = _fake_module(select_n=7)
    templ.wrap(mod)
    arr = mod.apply_template_selection(FakeAtomArray(10), "A1-50")
    assert int(arr.get_annotation(templ.ANNOTATION).sum()) == 7
    assert capsys.readouterr().err.strip() == "[rosettafold3-opt] TEMPLATE selection=1 atoms_selected=7/10"
    assert templ.state() == {"installed": True, "declared": 1, "items": [{"syntaxes": 1, "atoms_selected": 7, "atoms": 10}]}


def test_no_selection_is_untouched(capsys):
    mod = _fake_module(select_n=0)
    templ.wrap(mod)
    arr = mod.apply_template_selection(FakeAtomArray(4), None)
    assert not arr.get_annotation(templ.ANNOTATION).any() and capsys.readouterr().err == "" and templ.state()["declared"] == 0
