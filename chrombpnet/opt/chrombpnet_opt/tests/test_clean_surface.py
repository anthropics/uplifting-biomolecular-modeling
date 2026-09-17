"""The clean surface, locked by grep: the package reads no removed variable (the removed one-item switch and its h5 sub-switch, the
two multi-item variables, the alternate-kit override), no tree file and no file of opt/kit_ho names them, the package's own variables are the documented set, and
the mode line carries `items=<N>` under --items."""
import os
import re

from chrombpnet_opt import report, stack
from . import _stubs

REMOVED = tuple("CHROMBPNET_OPT_" + s for s in ("HOST" + "OVERLAP", "HOST" + "OVERLAP_H5", "MUL" + "TI", "MUL" + "TI_ITEMS", "K" + "IT", "K" + "IT_REAL"))
REMOVED_WORDS = ("host" + "overlap",)
PACKAGE_VARIABLES = {"CHROMBPNET_OPT", "CHROMBPNET_OPT_DET", "CHROMBPNET_OPT_HOME", "CHROMBPNET_OPT_CACHE_TAR", "CHROMBPNET_OPT_DATA",
                     "CHROMBPNET_OPT_MODEL", "CHROMBPNET_OPT_WEIGHTS", "CHROMBPNET_OPT_GENOME", "CHROMBPNET_OPT_CHROM_SIZES", "CHROMBPNET_OPT_REGIONS", "CHROMBPNET_OPT_BIGWIG"}
TREE_FILES = ("README.md", "run.sh", os.path.join("opt", "pyproject.toml"),
              os.path.join("opt", "chrombpnet_opt_autoload.pth"))


def _tree_texts():
    out = {}
    for dp, dns, fns in os.walk(_stubs.PKG_DIR):
        dns[:] = [d for d in dns if d != "__pycache__"]
        for fn in fns:
            if fn.endswith(".py"):
                out[os.path.join(dp, fn)] = open(os.path.join(dp, fn), encoding="utf-8").read()
    ho = os.path.join(_stubs.TREE_DIR, "opt", "kit_ho")                               # the fast mode's kit directory (house code): no removed name in it either
    if os.path.isdir(ho):
        for dp, dns, fns in os.walk(ho):
            dns[:] = [d for d in dns if d != "__pycache__"]
            for fn in fns:
                out[os.path.join(dp, fn)] = open(os.path.join(dp, fn), encoding="utf-8", errors="replace").read()
    for rel in TREE_FILES:
        p = os.path.join(_stubs.TREE_DIR, rel)
        if os.path.isfile(p):
            out[p] = open(p, encoding="utf-8").read()
    for fn in sorted(os.listdir(os.path.join(_stubs.TREE_DIR, "configs"))):
        p = os.path.join(_stubs.TREE_DIR, "configs", fn)
        out[p] = open(p, encoding="utf-8").read()
    return out


def test_removed_names_are_read_nowhere():
    texts = _tree_texts()
    assert len([p for p in texts if p.endswith(".py")]) >= 10
    for path, text in texts.items():
        for name in REMOVED:
            assert not re.search(r"\b" + name + r"\b", text), (path, name)
        for word in REMOVED_WORDS:
            assert word not in text.lower(), (path, word)


def test_package_variables_are_the_documented_set():
    """Every CHROMBPNET_OPT* name the package's code reads or the configs export is in the documented set."""
    pat = re.compile(r"\bCHROMBPNET_OPT[A-Z0-9_]*\b")
    found = set()
    for path, text in _tree_texts().items():
        if path.endswith(".py") and os.sep + "tests" + os.sep in path:
            continue
        found |= set(pat.findall(text))
    assert found - PACKAGE_VARIABLES == set(), sorted(found - PACKAGE_VARIABLES)
    assert stack.PACKAGE_ENV_PREFIX == "CHROMBPNET_OPT"


def test_mode_line_items_word():
    rep = {"active": True, "mode": "fast", "route": "k1", "kit_version": "x", "gpu": {"class": "H100"}, "det": False, "multi": 3}
    assert report.mode_line(rep) == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=x gpu=H100 det=0 precision=tf32 multi=3"      # `pred_bw --items`: N items in one process
    rep["multi"] = None
    assert report.mode_line(rep) == "[chrombpnet-opt] ACTIVE mode=fast route=k1 kit=x gpu=H100 det=0 precision=tf32"
    assert report.mode_line({"active": False, "mode": "off"}) == report.OFF_LINE


def test_configs_and_package_env_names_are_the_documented_surface():
    assert sorted(os.listdir(os.path.join(_stubs.TREE_DIR, "configs"))) == ["a100.env", "h100.env", "h200.env"]   # the supported classes' configs, nothing else
    assert sorted(n for n in dir(stack) if n.startswith("ENV")) == ["ENV", "ENV_CACHE_TAR", "ENV_HOME", "ENV_TARGET_GPU"]   # the package's whole environment surface in stack
    assert sorted(n for n in dir(stack) if n.endswith("_on")) == []   # no switch predicates: the mode is the whole composition
