"""`warm --tiling` takes one tiling or a comma-separated list (1to1 .. 6to6 = 199 .. 1194 tokens of PDB 1BRS barnase:barstar): the public input
builder writes ONE query JSON with one query per tiling, so the one warm process folds every size in turn (the size-dependent kernel variants
and captures of the 400 / 800 / 1200-token classes are met at warm time, not on the first real input of each size); the default is unchanged
(1to1, one query, the bytes the builder always wrote); an unknown or duplicate tiling is a usage error naming it, before anything is built."""
import json
import os
import subprocess
import sys

import pytest

from openfold3_ob0_opt import cli, modes, warm
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
SCRIPT = os.path.join(HOME, modes.KITS["fast_inference"], warm.PUBLIC_INPUTS)


def test_default_is_one_small_query_and_the_flag_parses():
    assert warm.DEFAULT_TILING == "1to1" and warm.tilings_of(None) == ["1to1"] and warm.tilings_of("") == ["1to1"]
    assert warm.TILINGS == ("1to1", "2to2", "3to3", "4to4", "5to5", "6to6") and [warm.tiling_tokens(t) for t in warm.TILINGS] == [199, 398, 597, 796, 995, 1194]
    a = cli.build_parser().parse_args(["warm", "--mode", "fast", "--out", "o", "--tiling", "2to2,4to4,6to6"])
    assert a.tiling == "2to2,4to4,6to6" and warm.tilings_of(a.tiling) == ["2to2", "4to4", "6to6"]
    assert cli.build_parser().parse_args(["warm", "--mode", "fast", "--out", "o"]).tiling is None


@pytest.mark.parametrize("bad,word", [("7to7", "unknown tiling 7to7"), ("2to3", "unknown tiling 2to3"), ("2to2,2to2", "duplicate tiling"), ("big", "unknown tiling big")])
def test_unknown_or_duplicate_tilings_are_usage_errors_by_name(bad, word):
    with pytest.raises(ValueError) as e:
        warm.tilings_of(bad)
    assert word in str(e.value) and "--tiling" in str(e.value)


def test_several_tilings_are_several_queries_of_one_query_json(tmp_path):
    q = warm.public_query(HOME, str(tmp_path / "multi"), "2to2,4to4,6to6")
    queries = json.load(open(q))["queries"]
    assert list(queries) == ["brs_2to2", "brs_4to4", "brs_6to6"] and [len(v["chains"]) for v in queries.values()] == [4, 8, 12]
    six = queries["brs_6to6"]["chains"]
    assert "".join(c["chain_ids"][0] for c in six) == "ABCDEFGHIJKL" and sum(len(c["sequence"]) for c in six) == 1194
    assert all(os.path.isfile(os.path.join(c["main_msa_file_paths"][0], "mmseqs_colabfold.a3m")) for v in queries.values() for c in v["chains"])
    meta = json.load(open(os.path.join(os.path.dirname(q), "meta.json")))
    assert meta["n_tokens"] == [398, 796, 1194] and [m["query"] for m in meta["queries"]] == list(queries)
    assert cli.expected_structures(q, None, 1, 1) == 3                                 # warm's exit rule counts one structure per query at one seed, one sample
    one = warm.public_query(HOME, str(tmp_path / "one"), "1to1")                      # the default form: one query, the builder's original record
    assert list(json.load(open(one))["queries"]) == ["brs_1to1"] and json.load(open(os.path.join(os.path.dirname(one), "meta.json")))["n_tokens"] == 199
    r = subprocess.run([sys.executable, SCRIPT, "1to1,3to3", str(tmp_path / "cli")], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.count("[public_inputs] brs_") == 2 and "brs_3to3 n_tokens=597" in r.stdout, r.stderr[-300:]
