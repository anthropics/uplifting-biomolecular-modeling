"""PEP 517 build backend of an optimization kit: setuptools.build_meta plus the kit's autoload .pth at the wheel root when the kit ships one.

A .pth at the wheel root lands in site-packages, where the interpreter executes its import line at start-up; that is the whole autoload
mechanism. The .pth text is GENERATED, never hand-written: `pth_text(package, env, tag)` below (python _build_backend.py <package> <ENV> <tag>
prints it) — a header comment naming the three, then one guarded import line. site.py swallows an ImportError raised by a .pth line and
carries on, which would run STOCK silently under a set kit mode; the guard instead prints `[<tag>] NOT ACTIVE: ...` and ends the process with
os._exit(<exit>) when `<package>._autoload` cannot be imported while `<ENV>` holds a mode (unset / off: nothing happens, stock is untouched),
and turns a SystemExit raised by the kit's own refusal into that exit code. The build refuses a .pth whose text is not pth_text's output. setuptools has no declarative way to place a file there, so this backend adds it (with its RECORD entry,
so pip uninstall removes it) to both the regular and the editable wheel. The .pth is the one `*_autoload.pth` file beside this backend; a kit that ships none (an argv-only kit) gets plain setuptools wheels.
pip requires a build backend to live inside the project directory, so the kit carries this file byte-for-byte: the copy is
`common/opt_core/kit_template/_build_backend.py`, and the core's tests hold the kit's copy to it. An sdist carries no .pth step and is
not a supported route.
"""
import base64
import glob
import hashlib
import os
import zipfile

from setuptools import build_meta as _setuptools

HERE = os.path.dirname(os.path.abspath(__file__))
PTH_HEADER = "# opt_core autoload guard: package={package} env={env} tag={tag} exit={exit_code}"
PTH_GUARD = ("import os as _o, sys as _s; exec(\"try:\\n import {package}._autoload\\nexcept SystemExit as _x:\\n _s.stderr.flush(); "
             "_o._exit(_x.code if isinstance(_x.code, int) else {exit_code})\\nexcept BaseException as _e:\\n"
             " if (_o.environ.get('{env}') or '').strip().lower() not in ('', 'off'):\\n"
             "  _s.stderr.write('[{tag}] NOT ACTIVE: {package}._autoload is not importable (%s: %s) under {env}=%s (exit {exit_code})\\\\n' % "
             "(type(_e).__name__, _e, _o.environ.get('{env}'))); _s.stderr.flush(); _o._exit({exit_code})\\n\")")
_PTH_HEADER_RX = r"^# opt_core autoload guard: package=(?P<package>[A-Za-z_][A-Za-z0-9_]*) env=(?P<env>[A-Za-z_][A-Za-z0-9_]*) tag=(?P<tag>[A-Za-z0-9_.-]+) exit=(?P<exit_code>[0-9]+)$"


def pth_text(package, env, tag, exit_code=3):
    """The whole text of `<package>_autoload.pth`: the header naming package / env / tag / exit code, and the one guarded import line."""
    for what, value in (("package", package), ("env", env)):
        if not (value and (value[0].isalpha() or value[0] == "_") and all(c.isalnum() or c == "_" for c in value)):
            raise ValueError(f"pth_text: {what}={value!r} is not an identifier")
    if not (tag and all(c.isalnum() or c in "_.-" for c in tag)):
        raise ValueError(f"pth_text: tag={tag!r} must be [A-Za-z0-9_.-]+")
    kw = dict(package=package, env=env, tag=tag, exit_code=int(exit_code))
    return PTH_HEADER.format(**kw) + "\n" + PTH_GUARD.format(**kw) + "\n"


def pth_fields(text):
    """(package, env, tag, exit_code) of a generated .pth text; ValueError naming the defect when the text is not pth_text's output."""
    import re
    first = text.split("\n", 1)[0]
    m = re.match(_PTH_HEADER_RX, first)
    if not m:
        raise ValueError("the .pth does not start with the opt_core autoload guard header (regenerate it: python _build_backend.py <package> <ENV> <tag>)")
    f = m.groupdict()
    if text != pth_text(f["package"], f["env"], f["tag"], int(f["exit_code"])):
        raise ValueError(f"the .pth text differs from pth_text({f['package']!r}, {f['env']!r}, {f['tag']!r}, {f['exit_code']}) (regenerate it)")
    return f["package"], f["env"], f["tag"], int(f["exit_code"])


_pths = sorted(glob.glob(os.path.join(HERE, "*_autoload.pth")))
if len(_pths) > 1:
    raise RuntimeError(f"expected at most one *_autoload.pth beside {__file__}, found {_pths}")
PTH = os.path.basename(_pths[0]) if _pths else None          # None: a kit without an autoload .pth (argv-only) builds plain setuptools wheels


def _add_pth(wheel_directory, wheel_name):
    if PTH is None:
        return wheel_name
    path = os.path.join(wheel_directory, wheel_name)
    data = open(os.path.join(HERE, PTH), "rb").read()
    try:
        pth_fields(data.decode("utf-8"))
    except ValueError as e:
        raise RuntimeError(f"{PTH}: {e}") from None
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    tmp = path + ".tmp"
    with zipfile.ZipFile(path) as zin, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            buf = zin.read(item.filename)
            if item.filename.endswith(".dist-info/RECORD"):
                lines = buf.decode().splitlines()
                own = [ln for ln in lines if ln.startswith(item.filename)]
                rest = [ln for ln in lines if ln and not ln.startswith(item.filename)]
                buf = ("\n".join(rest + [f"{PTH},sha256={digest},{len(data)}"] + own) + "\n").encode()
            zout.writestr(item, buf)
        zout.writestr(PTH, data)
    os.replace(tmp, path)
    return wheel_name


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    name = _setuptools.build_wheel(wheel_directory, config_settings, metadata_directory)
    return _add_pth(wheel_directory, name)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    name = _setuptools.build_editable(wheel_directory, config_settings, metadata_directory)
    return _add_pth(wheel_directory, name)


build_sdist = _setuptools.build_sdist
get_requires_for_build_wheel = _setuptools.get_requires_for_build_wheel
get_requires_for_build_editable = getattr(_setuptools, "get_requires_for_build_editable", None)
get_requires_for_build_sdist = _setuptools.get_requires_for_build_sdist
prepare_metadata_for_build_wheel = _setuptools.prepare_metadata_for_build_wheel
prepare_metadata_for_build_editable = getattr(_setuptools, "prepare_metadata_for_build_editable", None)


if __name__ == "__main__":
    import sys
    if len(sys.argv) not in (4, 5):
        raise SystemExit("usage: python _build_backend.py <package> <ENV> <tag> [<exit code>]  > <package>_autoload.pth")
    sys.stdout.write(pth_text(*sys.argv[1:4], **({"exit_code": int(sys.argv[4])} if len(sys.argv) == 5 else {})))
