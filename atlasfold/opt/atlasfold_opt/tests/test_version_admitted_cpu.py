"""The kit's version word is admitted by the activation tables that grade its ACTIVE line.

Every `run.sh pred` prints `ACTIVE … kit=<atlasfold_opt.__version__> …`; the repository's activation tables for this engine
(`<repo>/*/models/atlasfold/definitions.json`, a sibling tree of this kit) carry, per card and mode, the regex that line must match.  A version
word the tables do not admit fails every graded run on the version token alone — this test fails first, on a CPU host, when that happens.
Skips by reason when the tables are not beside this tree (an installed kit, a partial checkout)."""
import glob
import json
import os
import re

import pytest

import atlasfold_opt

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", "..", "..", ".."))          # tests -> atlasfold_opt -> opt -> atlasfold -> model-opt-release -> <repo>


def _tables():
    hits = sorted(glob.glob(os.path.join(REPO, "*", "models", "atlasfold", "definitions.json")))
    if not hits:
        pytest.skip(f"activation tables not beside this tree ({REPO}/*/models/atlasfold/definitions.json)")
    return json.load(open(hits[0]))


def _arms(d):
    for key, arms in d["activation"].items():
        arms = d["activation"][arms] if isinstance(arms, str) else arms
        for mode in ("exact", "fast", "big"):
            if mode in arms:
                yield key, mode, arms[mode]


def _kit_token_rx(expect_rx: str) -> str:
    m = re.search(r"kit=(\S+) det=", expect_rx)
    assert m, expect_rx[:160]
    return m.group(1)


def test_kit_version_is_admitted_by_every_active_expectation():
    v = atlasfold_opt.__version__
    d = _tables()
    seen = 0
    for key, mode, R in _arms(d):
        acts = [x for x in R["expect"] if "ACTIVE mode=" + mode + " n_gpu" in x.replace("\\", "")]
        assert acts, (key, mode)
        for x in acts:
            assert re.fullmatch(_kit_token_rx(x), v), f"kit {v} is not admitted by the {key} / {mode} ACTIVE expectation (kit token {_kit_token_rx(x)!r})"
            seen += 1
    assert seen >= 3


def test_an_empty_kit_word_is_refused():
    d = _tables()
    for key, mode, R in _arms(d):
        x = [x for x in R["expect"] if "ACTIVE mode=" + mode + " n_gpu" in x.replace("\\", "")][0]
        rx = _kit_token_rx(x)
        assert not re.fullmatch(rx, "") and not re.fullmatch(rx, "0.2 41")            # a version word is one non-empty token
        break
