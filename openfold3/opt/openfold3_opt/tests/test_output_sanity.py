"""A model file whose coordinates are not all finite is not a structure: manifest.count_structures leaves it out of the count the exit rule
compares with the expected number and names it once on stderr (`OUTPUT REJECTED file=… reason=nan_coordinates`) — on every route, the stock
`off` route included (an upstream build has been seen to write an all-nan model at sizes where its kernels overflow)."""
import os

from openfold3_opt import manifest, report

CIF_OK = """data_x
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.label_atom_id
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.B_iso_or_equiv
ATOM 1 N 10.123 -4.500 0.001 55.2
ATOM 2 CA 11.000 -3.250 1.250 57.9
#
"""


def _write(root, name, text):
    d = os.path.join(root, "q", "seed_101"); os.makedirs(d, exist_ok=True)
    p = os.path.join(d, name); open(p, "w").write(text); return p


def test_a_nan_model_file_is_not_counted_and_is_named(tmp_path, capsys):
    manifest.rejected_structures.clear()
    _write(str(tmp_path), "q_seed_101_sample_1_model.cif", CIF_OK)
    bad = _write(str(tmp_path), "q_seed_101_sample_2_model.cif", CIF_OK.replace("11.000 -3.250 1.250", "nan nan nan"))
    assert manifest.coordinate_defect(bad) == "nan_coordinates"
    assert manifest.count_structures(str(tmp_path)) == 1
    err = capsys.readouterr().err
    assert f"[{report.TAG}] OUTPUT REJECTED file={bad} reason=nan_coordinates" in err
    assert manifest.count_structures(str(tmp_path)) == 1 and capsys.readouterr().err == ""      # named once per process
    assert manifest.rejected_structures == {bad: "nan_coordinates"}


def test_inf_and_garbage_coordinates_are_defects_finite_ones_are_not(tmp_path):
    ok = _write(str(tmp_path), "a_model.cif", CIF_OK)
    assert manifest.coordinate_defect(ok) is None
    assert manifest.coordinate_defect(_write(str(tmp_path), "b_model.cif", CIF_OK.replace("10.123", "inf"))) == "nan_coordinates"
    assert manifest.coordinate_defect(_write(str(tmp_path), "c_model.cif", CIF_OK.replace("10.123", "1O.5"))) == "nan_coordinates"
    assert manifest.coordinate_defect(_write(str(tmp_path), "d_model.cif", "data_x\n#\n")) is None                       # no atom_site loop: nothing to judge, counted as written
    assert manifest.count_structures(None) is None and manifest.count_structures(str(tmp_path / "absent")) is None
