"""The lazy autoload: no finder without ESMC_OPT; with ESMC_OPT=exact the finder fires on the first `import esm` (a fake package here),
calls enable(), and — the mode being inactive on this box — the process exits 3 with the NOT ACTIVE line: stock never runs silently."""
import os
import subprocess
import sys
import textwrap

from esmc_opt import _autoload

from ._paths import OPT


def test_no_finder_without_env():
    assert _autoload.install({}) is None
    assert _autoload.install({"ESMC_OPT": "off"}) is None


def test_unknown_mode_named(capsys):
    assert _autoload.install({"ESMC_OPT": "turbo"}) is None
    assert "NOT ACTIVE: unknown ESMC_OPT" in capsys.readouterr().err


def test_finder_fires_and_exits_3(tmp_path):
    fake = tmp_path / "esm"
    fake.mkdir()
    (fake / "__init__.py").write_text("LOADED = True\n")
    code = textwrap.dedent("""
        import sys
        import esmc_opt._autoload as a
        f = a.install()
        assert f is not None and f in sys.meta_path
        import esm
        print("should not reach", esm.LOADED)
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ESMC_")}
    env.update({"PYTHONPATH": f"{OPT}{os.pathsep}{tmp_path}", "ESMC_OPT": "exact", "PYTHONDONTWRITEBYTECODE": "1"})
    out = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 3, (out.stdout, out.stderr[-500:])
    assert "[esmc-opt] NOT ACTIVE" in out.stdout
    assert "should not reach" not in out.stdout


def test_pth_path_refuses_with_exit_3(tmp_path):
    """The .pth route itself (what `pip install -e opt` ships: esmc_opt_autoload.pth at the site root), processed by site.addsitedir on a
    CPU box: the finder is installed at site time, `import esm` (a fake package) fires it, the mode cannot activate here, and the process
    ends with rc 3 and the NOT ACTIVE line — never a fatal interpreter error (a SystemExit inside site processing would be one)."""
    site_dir = tmp_path / "site"
    site_dir.mkdir()
    (site_dir / "esmc_opt.pth").write_text(f"{OPT}\n")                       # the package on the path (what the editable install does)
    (site_dir / "esmc_opt_autoload.pth").write_text(open(os.path.join(OPT, "esmc_opt_autoload.pth")).read())
    fake = tmp_path / "fake"
    (fake / "esm").mkdir(parents=True)
    (fake / "esm" / "__init__.py").write_text("LOADED = True\n")
    code = textwrap.dedent(f"""
        import site, sys
        site.addsitedir({str(site_dir)!r})
        import esmc_opt._autoload as a
        assert a.FINDER is not None and a.FINDER in sys.meta_path, "the .pth did not arm the finder"
        import esm
        print("should not reach", esm.LOADED)
    """)
    env = {k: v for k, v in os.environ.items() if not k.startswith("ESMC_")}
    env.update({"PYTHONPATH": str(fake), "ESMC_OPT": "exact", "PYTHONDONTWRITEBYTECODE": "1"})
    out = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, env=env)
    assert out.returncode == 3, (out.returncode, out.stdout[-400:], out.stderr[-600:])
    assert "[esmc-opt] NOT ACTIVE" in out.stdout and "Fatal Python error" not in out.stderr and "should not reach" not in out.stdout
