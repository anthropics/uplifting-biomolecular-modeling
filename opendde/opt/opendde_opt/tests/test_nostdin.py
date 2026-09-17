"""The verbs run off the caller's stdin (nostdin): detach() re-points a non-tty fd 0 at /dev/null so in-process upstream and its forks read
EOF, and every child the package launches gets stdin=/dev/null — kalign 3.3.5 under --use_template otherwise blocks on an open pipe."""
import ast
import os
import subprocess
import sys
import time

import pytest

from opendde_opt import cli, nostdin
from opendde_opt.tests._stubs import TREE

PKG = os.path.join(TREE, "opt", "opendde_opt")
PROBE = "import sys; d = sys.stdin.read(); print('EOF', len(d), flush=True)"        # stands in for kalign: reads stdin to EOF before doing anything
ENV = {**os.environ, "PYTHONPATH": os.pathsep.join(p for p in [os.path.dirname(PKG)] + os.environ.get("PYTHONPATH", "").split(os.pathsep) if p)}


def _under_open_pipe(code, timeout):
    """Run `python -c code` with an open pipe that never reaches EOF as its stdin (the container exec channel's shape); (rc, stdout, seconds)."""
    r, w = os.pipe()
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-c", code], stdin=r, capture_output=True, text=True, timeout=timeout, env=ENV)
        return p.returncode, p.stdout, time.time() - t0
    finally:
        os.close(r); os.close(w)


def test_the_probe_blocks_on_an_open_pipe_without_the_helper():
    """The driver is real: the stand-in inherits the open pipe and never returns (control for the two tests below)."""
    with pytest.raises(subprocess.TimeoutExpired):
        _under_open_pipe(f"import subprocess, sys; subprocess.run([sys.executable, '-c', {PROBE!r}], check=True)", timeout=1.5)


def test_detach_takes_the_process_tree_off_an_open_pipe():
    code = ("from opendde_opt import nostdin; import subprocess, sys\n"
            "assert nostdin.detach() is True\n"                                      # a pipe is not a terminal: fd 0 re-pointed
            "assert sys.stdin is not None and sys.stdin.read() == ''\n"              # this process reads EOF …
            f"subprocess.run([sys.executable, '-c', {PROBE!r}], check=True, timeout=20)\n")   # … and so does a child that INHERITS fd 0 (upstream's kalign call inherits)
    rc, out, dt = _under_open_pipe(code, timeout=30)
    assert rc == 0 and "EOF 0" in out and dt < 20, (rc, out, dt)


def test_detach_leaves_a_terminal_alone(monkeypatch):
    monkeypatch.setattr(nostdin.os, "isatty", lambda fd: True)
    dup2 = []
    monkeypatch.setattr(nostdin.os, "dup2", lambda *a: dup2.append(a))
    assert nostdin.detach() is False and dup2 == []


def test_the_stock_launcher_passes_devnull():
    """cli._call_relay (the stock arm's child under templates) hands the child /dev/null, whatever this process's own stdin is."""
    code = ("from opendde_opt import cli; import os, sys\n"
            f"rc, transcript = cli._call_relay([sys.executable, '-c', {PROBE!r}], dict(os.environ))\n"
            "assert rc == 0 and 'EOF 0' in transcript, (rc, transcript)\n")
    rc, out, dt = _under_open_pipe(code, timeout=30)                               # NO detach() in that process: the launch keyword alone must suffice
    assert rc == 0 and "EOF 0" in out, (rc, out, dt)


def test_every_launch_in_the_verbs_names_its_stdin():
    """Names only: each subprocess.Popen / subprocess.call in cli and warm passes stdin=…CHILD_STDIN (nostdin), and cli.main
    detaches before dispatching a verb."""
    for mod in ("cli.py", "warm.py"):
        tree = ast.parse(open(os.path.join(PKG, mod)).read())
        launches = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name) and n.func.value.id == "subprocess" and n.func.attr in ("Popen", "call")]
        assert launches, mod
        for c in launches:
            kw = {k.arg: k.value for k in c.keywords}
            assert "stdin" in kw and isinstance(kw["stdin"], ast.Attribute) and kw["stdin"].attr == "CHILD_STDIN", (mod, c.lineno)
    main_src = ast.get_source_segment(open(os.path.join(PKG, "cli.py")).read(),
                                      next(n for n in ast.parse(open(os.path.join(PKG, "cli.py")).read()).body if isinstance(n, ast.FunctionDef) and n.name == "main"))
    assert main_src.index("_nostdin.detach()") < main_src.index("COMMANDS.get(cmd)")
    assert nostdin.CHILD_STDIN is subprocess.DEVNULL


def test_the_autoload_activation_detaches():
    """opendde_opt.enable — what the OPENDDE_OPT=<mode> autoload finder and a library caller call — takes the process off an open pipe before
    stack.activate installs anything (stack.activate is stood in: no model, no GPU)."""
    code = ("import subprocess, sys\n"
            "import opendde_opt\n"
            "from opendde_opt import stack\n"
            "seen = {}\n"
            "def fake_activate(mode, **kw):\n"
            "    import os; seen['fd0_devnull_at_activate'] = (not os.isatty(0)) and sys.stdin is not None and sys.stdin.read() == ''\n"
            "    return {'active': True, 'mode': mode}\n"
            "stack.activate = fake_activate\n"
            "rep = opendde_opt.enable('exact', trigger='runner')\n"
            "assert rep == {'active': True, 'mode': 'exact'} and seen['fd0_devnull_at_activate'] is True, (rep, seen)\n"
            f"subprocess.run([sys.executable, '-c', {PROBE!r}], check=True, timeout=20)\n")   # a child inheriting fd 0 (upstream's kalign) reads EOF
    rc, out, dt = _under_open_pipe(code, timeout=30)
    assert rc == 0 and "EOF 0" in out, (rc, out, dt)
    src = open(os.path.join(PKG, "__init__.py")).read()
    enable_src = ast.get_source_segment(src, next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "enable"))
    assert enable_src.index("nostdin.detach()") < enable_src.index("stack.activate(")            # before any lever installs
    from opendde_opt import _autoload
    assert _autoload.spec().package == "opendde_opt"                                          # the core finder calls <spec.package>.enable(...) (opt_core/autoload.py): the one activation entry
