"""The evidence lines of the stock driver: one grammar, regex-locked; the JSON copy; the tree digest."""
import json
import re

import pytest

from esm_if1_opt import lines


@pytest.mark.parametrize("route,unit", [("stock", "batch")])
def test_clock_sequence_prints_locked_lines(route, unit, capsys, tmp_path, monkeypatch):
    sink = tmp_path / "timing.jsonl"
    monkeypatch.setenv(lines.TIMING_ENV, str(sink))
    c = lines.Clocks(route)
    c.prepass_begin()
    c.prepass_done(3)
    c.load_begin()
    s = c.load_done("cpu")
    assert re.fullmatch(lines.STARTUP_RE, s), s
    c.begin_unit("b 1", 100)
    c.model_begin(); c.model_end()
    c.model_begin(); c.model_end()
    it = c.item_done(unit, 1, 8)
    c.outputs_written(1, 2)
    kind, f = lines.parse(it)
    assert kind == "ITEM" and f["route"] == route and f["unit"] == unit and f["name"] == "b_1" and f["n_backbones"] == 1 and f["n_seq"] == 8 and f["length"] == 100
    assert f["t_end"] >= f["t_start"] and f["wall_s"] >= 0 and f["model_s"] >= 0 and f["alloc_gib"] == 0.0
    assert len(f["call_s"]) == 2 and abs(sum(f["call_s"]) - f["model_s"]) < 2e-3
    k = c.finish(batch=4, inference_mode=True)
    kind, kf = lines.parse(k)
    assert kind == "KERNELS" and kf["batch"] == 4 and kf["esm"] == lines.esm_facts()["esm"] and (kf["esm_tree"] == "none") == (kf["esm"] == "none") and kf["upstream"] == lines.UPSTREAM_COMMIT[:8]
    assert kf["torch_scatter"] == "absent" or kf["torch_scatter"].startswith("importable@")
    assert kf["inference_mode"] == "True"
    err = capsys.readouterr().err.splitlines()
    kinds = [lines.parse(l)[0] for l in err]
    want = ["PREPASS", "STARTUP", "CALL", "CALL", "ITEM", "OUTPUTS_WRITTEN", "PEAK", "KERNELS"]
    assert kinds == want
    _, pp = lines.parse(err[want.index("PREPASS")]); assert pp["n"] == 3 and pp["t_end"] >= pp["t_start"]
    _, ow = lines.parse(err[want.index("OUTPUTS_WRITTEN")]); assert (ow["n_files"], ow["n_records"]) == (1, 2) and ow["t"] >= f["t_end"]
    _, pk = lines.parse(err[want.index("PEAK")])
    assert pk["item"] == "pass" and pk["device"] == "none"
    assert all(" " not in tok.split("=", 1)[1] for l in err for tok in l.split("] ", 1)[1].split(" ")[1:] if "=" in tok)   # every value is one token
    _, c1 = lines.parse(err[want.index("CALL")]); _, c2 = lines.parse(err[want.index("CALL") + 1])
    assert (c1["i"], c2["i"], c1["name"]) == (1, 2, "b_1") and c1["t_start"] <= c1["t_end"] <= c2["t_start"] <= c2["t_end"] <= f["t_end"] and c1["t_start"] == f["t_start"]
    recs = [json.loads(l) for l in sink.read_text().splitlines()]
    assert [r["kind"] for r in recs] == want
    assert recs[want.index("ITEM")]["name"] == "b_1" and (recs[want.index("KERNELS")]["esm_tree_sha256"] == "none") == (lines.esm_facts()["esm"] == "none")


def test_bad_route_and_unit_refused():
    with pytest.raises(ValueError):
        lines.Clocks("fast")
    with pytest.raises(ValueError):
        lines.Clocks("stock").item_done("row", 1, 1)


def test_tree_digest_is_stable_and_content_sensitive(tmp_path):
    d = tmp_path / "pkg"
    (d / "sub").mkdir(parents=True)
    (d / "__init__.py").write_text("a = 1\n")
    (d / "sub" / "m.py").write_text("b = 2\n")
    (d / "sub" / "data.txt").write_text("ignored\n")
    h1, n1 = lines.tree_digest(str(d))
    assert n1 == 2 and len(h1) == 64
    (d / "__pycache__").mkdir()
    (d / "__pycache__" / "x.py").write_text("c\n")
    assert lines.tree_digest(str(d)) == (h1, n1)
    (d / "sub" / "m.py").write_text("b = 3\n")
    assert lines.tree_digest(str(d))[0] != h1
