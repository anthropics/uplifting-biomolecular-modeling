"""``report.tally_fields``: the EXIT-tally grammar never drops a key silently — equality keys (installed / armed / active / mode / state /
n_gpu / rank / group, those present) print first, the rest alphabetically, every key by default; a cap names the overflow as ``…+N``."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from opt_core import report                                              # noqa: E402


def _stats(n_extra: int) -> dict:
    d = {"installed": True, "armed": True, "n_gpu": 4, "rank": 0, "group": True}
    d.update({f"zz_counter_{i:02d}": i for i in range(n_extra)})           # alphabetically AFTER the equality keys' names on purpose
    d.update({"aa_first": 1, "block_rewraps": 3, "weights_guarded": True})
    return {"rowpair_tp": d, "plain": 7}


def test_every_key_prints_by_default_identity_first_29_keys():
    st = _stats(21)                                                          # 5 equality + 21 + 3 = 29 scalar keys
    (grp, plain) = report.tally_fields(st)
    assert plain == "plain=7"
    assert grp.startswith("rowpair_tp={installed=True,armed=True,n_gpu=4,rank=0,group=True,aa_first=1,block_rewraps=3,"), grp
    body = grp[len("rowpair_tp={"):-1].split(",")
    assert len(body) == 29 and "…" not in grp, grp


def test_35_keys_all_present_and_ordered():
    st = _stats(27)                                                          # 35 scalar keys
    grp = report.tally_fields(st)[0]
    assert len(grp[len("rowpair_tp={"):-1].split(",")) == 35 and grp.index("installed=") < grp.index("aa_first="), grp


def test_a_cap_names_the_overflow_never_silent():
    st = _stats(27)
    grp = report.tally_fields(st, max_fields=10)[0]
    parts = grp[len("rowpair_tp={"):-1].split(",")
    assert parts[:5] == ["installed=True", "armed=True", "n_gpu=4", "rank=0", "group=True"], parts
    assert parts[-1] == "…+25" and len(parts) == 11, parts


def test_nested_values_are_counted_not_dumped():
    grp = report.tally_fields({"m": {"installed": True, "per_block": {"a": 1, "b": 2}, "walls": [1.0, 2.0, 3.0]}})[0]
    assert grp == "m={installed=True,per_block=<2keys>,walls=<3items>}", grp


def test_long_scalar_is_shortened_with_a_marker():
    grp = report.tally_fields({"m": {"path": "x" * 100}})[0]
    assert grp.startswith("m={path=" + "x" * 39 + "…") and len(grp) < 60, grp
