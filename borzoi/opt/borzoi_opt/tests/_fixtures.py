"""Test fixtures: a scratch model root = the REAL kit bytes (copied from this tree's opt/datapath/pipeline_tf) + a FAKE stock
``borzoi_sad.py`` (a stand-in script with the stock's option grammar; it runs without TensorFlow) pinned in a scratch ``stock/PINS.json``.
Nothing here touches the tree; the kit is never executed (its entry imports TensorFlow) — only located, checked and, for the swap,
its exec argv computed."""
from __future__ import annotations

import hashlib
import json
import os
import shutil

from .. import kit

HERE = os.path.dirname(os.path.abspath(__file__))
TREE_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))                  # borzoi/
REAL_KIT = os.path.join(TREE_ROOT, kit.KIT_RELPATH)
TREE_OPT = os.path.join(TREE_ROOT, "opt")                                           # the package directory under test (borzoi_opt's parent): a CHILD interpreter binds THIS tree through PYTHONPATH=TREE_OPT whether or not some borzoi_opt is pip-installed on the interpreter


def installed_here() -> str | None:
    """The borzoi_opt a CLEAN child interpreter imports with no PYTHONPATH (i.e. the pip-installed package and its site .pth), when that
    package is THIS tree's; else None. Tests of the INSTALLED state (the autoload .pth firing at interpreter start) skip by name without it —
    `pip install -e opt` is the documented install; a PYTHONPATH-only interpreter (a CPU gate importing the tree) has no .pth to fire."""
    import subprocess, sys
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "BORZOI_OPT") and not k.startswith("KIT_")}
    try:
        p = subprocess.run([sys.executable, "-s", "-c", "import borzoi_opt._autoload, borzoi_opt; print(borzoi_opt.__file__)"], env=env, cwd=os.sep,
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    where = (p.stdout or "").strip().splitlines()[-1:] if p.returncode == 0 else []
    if not where: return None
    return where[0] if os.path.realpath(where[0]).startswith(os.path.realpath(TREE_OPT) + os.sep) else None

FAKE_STOCK = '''#!/usr/bin/env python
"""A stand-in for the stock borzoi_sad.py: the stock's option grammar (borzoi_sad.py main(), add_option calls), no model. Writes
<out_dir>/fake_stock.json with what it saw and exits 0."""
from __future__ import print_function
from optparse import OptionParser
import json, os, sys

def main():
    usage = 'usage: %prog [options] <params_file> <model_file> <vcf_file>'
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta', default='%s/assembly/ucsc/hg38.fa' % os.environ.get('BORZOI_HG38', 'hg38'), help='Genome FASTA for sequences [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-p', dest='processes', default=None, type='int', help='Number of processes, passed by multi script')
    parser.add_option('--rc', dest='rc', default=False, action='store_true', help='Average forward and reverse complement predictions [Default: %default]')
    parser.add_option('--shifts', dest='shifts', default='0', type='str', help='Ensemble prediction shifts [Default: %default]')
    parser.add_option('--stats', dest='sad_stats', default='SAD', help='Comma-separated list of stats to save. [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, type='str', help='File specifying target indexes and labels in table format')
    parser.add_option('-u', dest='untransform_old', default=False, action='store_true')
    parser.add_option('--no_untransform', dest='no_untransform', default=False, action='store_true')
    (options, args) = parser.parse_args()
    if len(args) != 3:
        parser.error('Must provide parameters and model files and QTL VCF file')
    os.makedirs(options.out_dir, exist_ok=True)
    rec = {"argv": sys.argv, "options": vars(options), "args": args, "env_KIT": sorted(k for k in os.environ if k.startswith("KIT_")),
           "env_BORZOI_OPT": os.environ.get("BORZOI_OPT"), "modules_kitlib": sorted(m for m in sys.modules if m.split(".")[0] == "kitlib")}
    json.dump(rec, open(os.path.join(options.out_dir, "fake_stock.json"), "w"), indent=1)
    print("fake stock ran", json.dumps(rec["args"]))

if __name__ == '__main__':
    main()
'''


def sha256(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_root(tmp_path, *, stock_script: str = FAKE_STOCK, pin_sha: str | None = None) -> dict:
    """A scratch model root: <root>/opt/datapath/pipeline_tf (the real kit bytes), <root>/stock/PINS.json, and a stock checkout at
    <root>/stock_checkout/src/scripts/borzoi_sad.py (+ borzoi_sed.py). Returns paths + the env to set (MODEL_OPT, BORZOI_DIR)."""
    root = os.path.join(str(tmp_path), "borzoi")
    shutil.copytree(REAL_KIT, os.path.join(root, kit.KIT_RELPATH), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    scripts = os.path.join(root, "stock_checkout", "src", "scripts")
    os.makedirs(scripts)
    sad = os.path.join(scripts, kit.ENTRY)
    open(sad, "w").write(stock_script)
    os.chmod(sad, 0o755)
    sed = os.path.join(scripts, kit.SED_ENTRY)
    open(sed, "w").write(stock_script.replace("fake_stock.json", "fake_sed.json"))
    os.chmod(sed, 0o755)
    os.makedirs(os.path.join(root, "stock"))
    pins = {"schema": "borzoi_opt.pins/1", "stock_scripts": {
        kit.ENTRY: {"path": "src/scripts/" + kit.ENTRY, "sha256": pin_sha or sha256(sad), "bytes": os.path.getsize(sad)},
        kit.SED_ENTRY: {"path": "src/scripts/" + kit.SED_ENTRY, "sha256": sha256(sed), "bytes": os.path.getsize(sed)}}}
    json.dump(pins, open(os.path.join(root, "stock", "PINS.json"), "w"), indent=1)
    return {"root": root, "kit_dir": os.path.join(root, kit.KIT_RELPATH), "stock_sad": sad, "stock_sed": sed,
            "env": {"MODEL_OPT": root, "BORZOI_DIR": os.path.join(root, "stock_checkout")}}


def clean_env(monkeypatch, extra: dict | None = None) -> None:
    """No KIT_* / BORZOI_OPT / MODEL_OPT / BORZOI_DIR / PYTHONPATH from the outside; then `extra`."""
    for k in list(os.environ):
        if k.startswith("KIT_") or k in ("BORZOI_OPT", "MODEL_OPT", "BORZOI_DIR", "PYTHONPATH", "MODEL_OPT_TARGET_GPU"):
            monkeypatch.delenv(k, raising=False)
    for k, v in (extra or {}).items():
        monkeypatch.setenv(k, v)
