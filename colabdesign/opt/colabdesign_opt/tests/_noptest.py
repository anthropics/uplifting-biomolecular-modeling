"""A pytest stand-in, so a kit test file runs as a plain script on an interpreter that has no pytest (the stack image ships none):

    try:
        import pytest
    except ImportError:                                   # the image: unittest only
        from colabdesign_opt.tests import _noptest as pytest
    ...
    if __name__ == "__main__":
        raise SystemExit(pytest.main([__file__]))

Covers what kit tests use: `raises`, `skip`, `importorskip`, `mark.skipif`, `mark.parametrize`, `fail`, `MonkeyPatch`, and `main([file])` = run every
`test_*` function and every `unittest.TestCase` of the calling module, print one summary line, return 0 | 1. Under real pytest none of this runs.
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect
import re
import sys
import traceback
import unittest


class _Skip(unittest.SkipTest):
    pass


def skip(reason: str = ""):
    raise _Skip(reason)


def fail(msg: str = ""):
    raise AssertionError(msg)


def importorskip(name: str, reason: str = ""):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise _Skip(reason or f"could not import {name!r}: {e}")


@contextlib.contextmanager
def raises(exc, match: str = None):
    box = {"value": None}
    try:
        yield box
    except exc as e:                                                    # noqa
        if match is not None and not re.search(match, str(e)):
            raise AssertionError(f"{type(e).__name__} raised but {str(e)!r} does not match {match!r}") from None
        box["value"] = e
        return
    raise AssertionError(f"DID NOT RAISE {exc}")


class _Mark:
    @staticmethod
    def skipif(cond, reason: str = ""):
        def deco(f):
            if not cond:
                return f
            @functools.wraps(f)
            def skipped(*a, **k):
                raise _Skip(reason)
            return skipped
        return deco

    @staticmethod
    def skip(reason: str = ""):
        return _Mark.skipif(True, reason=reason)

    @staticmethod
    def parametrize(names, values, ids=None):
        keys = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
        def deco(f):
            cases = [v if isinstance(v, (tuple, list)) or len(keys) > 1 else (v,) for v in values]
            @functools.wraps(f)
            def run_all(*a, **k):
                for case in cases:
                    f(*a, **dict(k, **dict(zip(keys, case if isinstance(case, (tuple, list)) else (case,)))))
            return run_all
        return deco


mark = _Mark()
_NOTSET = object()


class MonkeyPatch:
    """setattr / setitem / setenv / delenv with undo(), as the kit tests use pytest's."""
    def __init__(self):
        self._undo = []

    def setattr(self, target, name, value=_NOTSET, raising: bool = True):
        if value is _NOTSET:                                             # the "module.attr" string form
            modname, name = target.rsplit(".", 1); value = name and None
            raise TypeError("MonkeyPatch.setattr: pass (obj, name, value)")
        old = getattr(target, name, _NOTSET)
        if old is _NOTSET and raising:
            raise AttributeError(f"{target!r} has no attribute {name!r}")
        self._undo.append(("attr", target, name, old)); setattr(target, name, value)

    def delattr(self, target, name, raising: bool = True):
        old = getattr(target, name, _NOTSET)
        if old is _NOTSET:
            if raising:
                raise AttributeError(name)
            return
        self._undo.append(("attr", target, name, old)); delattr(target, name)

    def setitem(self, mapping, key, value):
        self._undo.append(("item", mapping, key, mapping.get(key, _NOTSET))); mapping[key] = value

    def delitem(self, mapping, key, raising: bool = True):
        if key not in mapping:
            if raising:
                raise KeyError(key)
            return
        self._undo.append(("item", mapping, key, mapping[key])); del mapping[key]

    def setenv(self, name, value, prepend=None):
        import os
        self.setitem(os.environ, name, str(value) if not prepend or name not in os.environ else f"{value}{prepend}{os.environ[name]}")

    def delenv(self, name, raising: bool = True):
        import os
        self.delitem(os.environ, name, raising=raising)

    def undo(self):
        while self._undo:
            kind, obj, key, old = self._undo.pop()
            if kind == "attr":
                if old is _NOTSET:
                    delattr(obj, key)
                else:
                    setattr(obj, key, old)
            elif old is _NOTSET:
                obj.pop(key, None)
            else:
                obj[key] = old


def main(argv=None) -> int:
    """Run the calling module's tests: every module-level `test_*` function (no fixtures: plain calls) and every unittest.TestCase."""
    mod = sys.modules["__main__"]
    passed = failed = skipped = 0
    for name, f in sorted(vars(mod).items()):
        if name.startswith("test_") and inspect.isfunction(f):
            try:
                f(); passed += 1
            except unittest.SkipTest as s:
                skipped += 1; print(f"SKIP {name}: {s}")
            except Exception:                                            # noqa: BLE001
                failed += 1; print(f"FAIL {name}"); traceback.print_exc()
    suite = unittest.defaultTestLoader.loadTestsFromModule(mod)
    if suite.countTestCases():
        r = unittest.TextTestRunner(verbosity=1).run(suite)
        failed += len(r.failures) + len(r.errors); skipped += len(r.skipped); passed += r.testsRun - len(r.failures) - len(r.errors) - len(r.skipped)
    print(f"{getattr(mod, '__file__', '?')}: {passed} passed, {failed} failed, {skipped} skipped (pytest stand-in)")
    return 1 if failed else 0
