"""The environment route without the shared core: a process under PROTENIX_V1_OPT=<mode> whose interpreter cannot import opt_core (or a
module of the package) ends with the kit's NOT ACTIVE line naming the missing module and exit 3 — never a traceback, never a silent
stock run — and the declared names are the package's environment (a set PROTENIX_V1_OPT_N_GPU does not refuse the process as mistyped)."""
import os
import subprocess
import sys
import textwrap

from protenix_v1_opt import _autoload as A, kit as K, stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))       # .../opt (the package's parent)


def test_core_missing_is_exit_3_with_the_line(tmp_path):
    fake = tmp_path / "site"; (fake / "protenix").mkdir(parents=True); (fake / "runner").mkdir()
    (fake / "protenix" / "__init__.py").write_text(""); (fake / "runner" / "__init__.py").write_text("")
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROTENIX_V1_OPT") and k != "PYTHONPATH"}
    env.update(PYTHONPATH=os.pathsep.join([OPT, str(fake)]), PROTENIX_V1_OPT="fast", PYTHONDONTWRITEBYTECODE="1")   # the package, the fake stock — and NO opt_core
    code = textwrap.dedent("""
        import protenix_v1_opt._autoload      # what the .pth line does at interpreter start
        import runner                         # the trigger: the stock CLI's package
        print("STOCK RAN")                    # must never print
    """)
    p = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert p.returncode == A.EXIT_NOT_ACTIVE, (p.returncode, p.stderr[-600:])
    assert "STOCK RAN" not in p.stdout
    lines = [l for l in p.stderr.splitlines() if l.startswith("[protenix-v1-opt] NOT ACTIVE: reason=core_missing:opt_core")]
    assert len(lines) == 1 and "nothing importable as opt_core on sys.path" in lines[0], p.stderr[-600:]     # the entry gate's words (_core_gate.gate)
    assert "Traceback" not in p.stderr


def test_the_declared_names_are_the_package_env():
    assert tuple(A.DECLARED) == tuple(stack.PACKAGE_ENV)
