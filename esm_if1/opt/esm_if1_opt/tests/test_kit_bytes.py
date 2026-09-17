"""The engine directory's own bytes: the carried upstream archive and its extracted example script are on disk, and the two core template
copies (`_build_backend.py`, `_core_gate.py`) are the core's `kit_template` byte for byte (when the tree's `common/` is beside the engine
directory). A file tracked in this repo is pinned by the git commit that carries it — this file checks presence and shape."""
import hashlib
import os

import pytest

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))     # esm_if1/ (this file: esm_if1/opt/esm_if1_opt/tests/)
CORE = os.path.join(os.path.dirname(HERE), "common", "opt_core")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def test_carried_upstream_files_are_on_disk():
    assert os.path.isfile(os.path.join(HERE, "stock", "fair-esm-2b369911.tar.gz"))
    assert os.path.isfile(os.path.join(HERE, "stock", "src", "examples", "inverse_folding", "sample_sequences.py"))
    import json
    pins = json.load(open(os.path.join(HERE, "stock", "PINS.json"), encoding="utf-8"))
    up = pins["upstream"]["fair-esm"]
    assert (up["repo"], up["commit"], up["version"], up["archive"]) == ("https://github.com/facebookresearch/esm", "2b369911bb5b4b0dda914521b9475cad1656b2ac", "2.0.1", "stock/fair-esm-2b369911.tar.gz")
    assert os.path.isfile(os.path.join(HERE, up["archive"])) and pins["weights"]["file"] == "esm_if1_gvp4_t16_142M_UR50.pt"
    assert sorted(os.listdir(os.path.join(HERE, "configs"))) == ["a100.env", "h100.env", "h200.env"]


@pytest.mark.skipif(not os.path.isdir(CORE), reason="common/opt_core is not beside this engine directory")
def test_template_copies_are_the_cores():
    assert sha(os.path.join(HERE, "opt", "_build_backend.py")) == sha(os.path.join(CORE, "kit_template", "_build_backend.py"))
    assert sha(os.path.join(HERE, "opt", "esm_if1_opt", "_core_gate.py")) == sha(os.path.join(CORE, "kit_template", "_core_gate.py"))
