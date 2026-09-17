"""The autoload whitelist (`_autoload.DECLARED` == `stack.PACKAGE_ENV`) names EVERY environment variable the package reads under its
`PROTENIX_V1_OPT` prefix, and nothing `configs/*.env` exports is refused as undeclared: a process started from the kit's own configuration,
with or without a mode, with the big evidence line or the weights-memo directory set, is never refused at interpreter start."""
import os
import re
import shutil
import subprocess
import sys

import pytest

from protenix_v1_opt import _autoload as A
from protenix_v1_opt import big, kit as K, stack

PKG = os.path.dirname(os.path.abspath(A.__file__))
TREE = os.path.dirname(os.path.dirname(PKG))                                                     # .../protenix_v1
CONFIGS = sorted(os.path.join(TREE, "configs", f) for f in os.listdir(os.path.join(TREE, "configs")) if f.endswith(".env"))


def _package_sources():
    return {f: open(os.path.join(PKG, f), encoding="utf-8").read() for f in os.listdir(PKG) if f.endswith(".py")}


def _exports(path):
    with open(path, encoding="utf-8") as fh:
        return re.findall(r"^export ([A-Z0-9_]+)=", fh.read(), re.M)


def _install(env):
    refused = []
    f = A.install(dict(env), exit=refused.append)
    if f is not None and f in sys.meta_path:
        sys.meta_path.remove(f)
    return refused, f


def test_every_prefixed_name_the_package_reads_is_declared():
    read = sorted({m for src in _package_sources().values() for m in re.findall(r"[\"\'](%s[A-Z_]*)[\"\']" % A.ENV, src)})
    assert read == sorted(A.DECLARED) == sorted(stack.PACKAGE_ENV), (read, A.DECLARED)
    assert K.WEIGHTS_MEMO_ENV in A.DECLARED and "PROTENIX_V1_OPT_EVIDENCE" not in A.DECLARED


def test_configs_exist_and_export_nothing_undeclared_under_the_prefix():
    assert CONFIGS and all(_exports(c) for c in CONFIGS), CONFIGS
    for cfg in CONFIGS:
        names = _exports(cfg)
        assert [n for n in names if n.startswith(A.ENV) and n not in A.DECLARED] == [], (cfg, names)
        refused, f = _install({n: "x" for n in names})                                          # no mode: nothing happens, nothing refused
        assert refused == [] and f is None, (cfg, refused)
        refused, f = _install({**{n: "x" for n in names}, A.ENV: "fast"})                        # a mode: the finder installs, nothing refused
        assert refused == [] and f is not None, (cfg, refused)


def test_the_weights_memo_dir_is_a_declared_name():
    refused, f = _install({A.ENV: "big", K.WEIGHTS_MEMO_ENV: "/tmp/x"})
    assert refused == [] and f is not None


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash sources the configuration")
def test_a_clean_shell_that_sources_the_configs_exports_imports_the_autoload():
    """`configs/*.env`'s export lines sourced in a clean shell (the weights probe line is not an export and is not run here), then the
    interpreter imports the autoload module as the .pth does: exit 0, no NOT ACTIVE, with and without a mode."""
    for cfg in CONFIGS:
        for mode in ("", "fast"):
            script = (f'set -a; eval "$(grep -E "^export " {cfg})"; ' + (f"export {A.ENV}={mode}; " if mode else "")
                      + f"exec {sys.executable} -c 'import protenix_v1_opt._autoload; print(20260903)'")
            r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env={"PATH": os.environ.get("PATH", ""), "HOME": os.environ.get("HOME", "/tmp"),
                                                                                          "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)})
            assert r.returncode == 0 and "20260903" in r.stdout and "NOT ACTIVE" not in r.stderr, (cfg, mode, r.returncode, r.stderr[-600:])
