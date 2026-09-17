"""``mem.rowpair.rankdata``, launcher side (standard library): ONE hash seed for every rank interpreter of a launch. Proven on real child
interpreters started through ``launch.rank_env`` (the environment ``run_rank_processes`` starts every rank with): with no seed in the parent both
children report ``PYTHONHASHSEED=0`` and the same ``hash()`` of a probe string; with ``7`` in the parent both report ``7``."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from opt_core.mem.rowpair import rankdata as RD  # noqa: E402

PROBE = [sys.executable, "-c", "import json, os; print(json.dumps({'seed': os.environ.get('PYTHONHASHSEED'), 'h': hash('rowpair-probe')}))"]


def _child(env):
    out = subprocess.run(PROBE, env=env, check=True, capture_output=True, text=True, timeout=60).stdout
    return json.loads(out.strip().splitlines()[-1])


def test_hash_seed_default_when_parent_names_none():
    for parent in ({}, {"PYTHONHASHSEED": ""}, {"PYTHONHASHSEED": "random"}, {"PYTHONHASHSEED": "-3"}, {"PYTHONHASHSEED": "4294967296"}, {"PYTHONHASHSEED": "0x10"}):
        assert RD.hash_seed(parent) == ("0", "default"), parent


def test_seed_rule_four_cases():
    """unset / '7' / 'random' / '': inherited only for a decimal integer in [0, 4294967295]; everything else exports 0 as `default`, and a
    present-but-invalid parent value is shown on the word (`parent=`), never refused."""
    assert RD.hashseed_word({}, 4) == "hashseed=0 source=default ranks=4" and RD.invalid_parent({}) is None
    assert RD.hashseed_word({"PYTHONHASHSEED": "7"}, 4) == "hashseed=7 source=inherited ranks=4" and RD.invalid_parent({"PYTHONHASHSEED": "7"}) is None
    assert RD.hashseed_word({"PYTHONHASHSEED": "random"}, 4) == "hashseed=0 source=default parent='random' ranks=4"
    assert RD.hashseed_word({"PYTHONHASHSEED": ""}, 4) == "hashseed=0 source=default ranks=4"                      # empty counts as unset (CPython treats it so)
    assert RD.hashseed_word({"PYTHONHASHSEED": "12345678901234567890xyz"}, 2) == "hashseed=0 source=default parent='1234567890123456' ranks=2"
    assert RD.hashseed_fields({"PYTHONHASHSEED": "random"}) == {"hashseed": "0", "hashseed_source": "default", "hashseed_parent": "random"}
    env, word = RD.ranks_env({"PYTHONHASHSEED": "random", "X": "1"}, 2)
    assert env == {"PYTHONHASHSEED": "0", "X": "1"} and word == "hashseed=0 source=default parent='random' ranks=2"


def test_hash_seed_inherited_and_canonical():
    assert RD.hash_seed({"PYTHONHASHSEED": "7"}) == ("7", "inherited")
    assert RD.hash_seed({"PYTHONHASHSEED": " 007 "}) == ("7", "inherited")
    assert RD.hash_seed({"PYTHONHASHSEED": "0"}) == ("0", "inherited")
    assert RD.hash_seed({"PYTHONHASHSEED": "4294967295"}) == ("4294967295", "inherited")


def test_ranks_env_exports_one_seed_and_the_census_word():
    parent = {"PATH": "/bin", "X": "1"}
    env, word = RD.ranks_env(parent, 4)
    assert env == {"PATH": "/bin", "X": "1", "PYTHONHASHSEED": "0"} and parent == {"PATH": "/bin", "X": "1"}      # a copy; the parent mapping is untouched
    assert word == "hashseed=0 source=default ranks=4" == RD.hashseed_word(parent, 4)
    assert RD.hashseed_fields(parent) == {"hashseed": "0", "hashseed_source": "default"}
    env7, word7 = RD.ranks_env({**parent, "PYTHONHASHSEED": "7"}, 2)
    assert env7["PYTHONHASHSEED"] == "7" and word7 == "hashseed=7 source=inherited ranks=2"
    assert RD.hashseed_fields({"PYTHONHASHSEED": "7"}) == {"hashseed": "7", "hashseed_source": "inherited"}


def test_ranks_env_default_parent_is_os_environ(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "11")
    env, word = RD.ranks_env(None, 3)
    assert env["PYTHONHASHSEED"] == "11" and word == "hashseed=11 source=inherited ranks=3" and env["PATH"] == os.environ["PATH"]


def _parent_without_seed():
    return {k: v for k, v in os.environ.items() if k != "PYTHONHASHSEED"}


def test_two_rank_interpreters_share_one_seed_when_the_parent_has_none():
    """The launcher's env-building function (``launch.rank_env``, layered on ``rankdata.ranks_env``) with PYTHONHASHSEED ABSENT from the parent:
    both rank interpreters report PYTHONHASHSEED=0 and the same hash of a probe string. The control proves the probe is sensitive: two
    interpreters started with different explicit seeds hash the probe differently."""
    from opt_core.mem.rowpair import launch
    parent = _parent_without_seed()
    kids = [_child(launch.rank_env(r, 2, 29999, base=parent)) for r in range(2)]
    assert [k["seed"] for k in kids] == ["0", "0"], kids
    assert kids[0]["h"] == kids[1]["h"], kids
    env0 = launch.rank_env(0, 2, 29999, base=parent)
    assert env0[launch.ENV_RANK] == "0" and env0[launch.ENV_WORLD] == "2" and env0["PYTHONHASHSEED"] == "0"
    control = [_child({**parent, "PYTHONHASHSEED": s}) for s in ("1", "2")]
    assert control[0]["h"] != control[1]["h"], "probe insensitive: str hashing did not change with the seed"


def test_two_rank_interpreters_inherit_the_parents_seed():
    from opt_core.mem.rowpair import launch
    parent = {**_parent_without_seed(), "PYTHONHASHSEED": "7"}
    kids = [_child(launch.rank_env(r, 2, 29999, base=parent)) for r in range(2)]
    assert [k["seed"] for k in kids] == ["7", "7"] and kids[0]["h"] == kids[1]["h"], kids
    assert kids[0]["h"] == _child({**parent})["h"]                          # the same value a seed-7 interpreter computes: inherited, not replaced


