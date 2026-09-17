"""Upstream's IGSO(3) schedule cache directory, when the checkout's own cannot or must not be used.

Upstream keeps that cache as pickles in ``<checkout>/schedules`` (rfdiffusion/inference/model_runners.py picks and makes the directory;
rfdiffusion/diffusion.py reads a file it finds there with ``pickle.load`` and writes the one it computes) unless
``inference.schedule_directory_path`` names another directory. Unpickling runs what the file says, so the directory has to be one that
only this user (or root) can write:

* the image's checkout is root's and its ``schedules`` directory is closed to other users (environment/Dockerfile: mode 0755). Root writes
  there as upstream always did; a process under another uid cannot, and a read-only image takes no write at all;
* a ``schedules`` directory that every user can write (an ``o+w`` mode, sticky or not) is refused by name: a file there could be anyone's.

In both cases ``design`` and ``warm`` pass ``inference.schedule_directory_path=${TMPDIR:-/tmp}/rfdiffusion1_schedules-uid<uid>`` themselves
— never when the key is typed (the caller's own directory, verbatim). That directory is made here with mode 0700 and used only when it then
is a real directory (no symbolic link) of this user (root: of any user) with no group or other write bit; anything else is refused by name
on one stderr line, nothing in it is read, and the run computes its schedule into a private temporary directory removed at exit (the cache
behaves as absent). A checkout directory this user can write and others cannot — route C, the image as root, the README's route-B bind —
stays upstream's own affair: no token, no line.
"""
import atexit
import os
import shutil
import stat
import tempfile
from typing import List, Optional

from . import report as _report
from . import stack

KEY = "inference.schedule_directory_path"
NAME = "rfdiffusion1_schedules-uid%d"


def checkout_dir() -> Optional[str]:
    """``<checkout>/schedules`` of the checkout upstream runs from (RFD_ROOT, else pip's editable location); None when there is no checkout."""
    root = stack.rfd_root()
    return os.path.join(root, "schedules") if root and os.path.isdir(root) else None


def open_to_all(d: str) -> bool:
    """``d`` is a directory every user can write (sticky or not: a file pre-planted under the cache's name is read all the same)."""
    try:
        st = os.stat(d)
    except OSError:
        return False
    return stat.S_ISDIR(st.st_mode) and bool(st.st_mode & stat.S_IWOTH)


def writable(d: str) -> bool:
    """Upstream's ``os.mkdir`` and pickle writes would succeed: ``d`` is a writable directory, or absent under a writable one."""
    if os.path.isdir(d):
        return os.access(d, os.W_OK | os.X_OK)
    return not os.path.lexists(d) and os.access(os.path.dirname(d) or ".", os.W_OK | os.X_OK)


def per_user_dir(environ=None) -> str:
    environ = os.environ if environ is None else environ
    return os.path.join(os.path.abspath(environ.get("TMPDIR") or "/tmp"), NAME % os.getuid())   # absolute: upstream's process does not share this one's working directory


def refusal(d: str, make: bool = True) -> Optional[str]:
    """Make ``d`` with mode 0700 when absent (``make``); None when it then is a real directory of this user (root: of any user) with no group
    or other write bit — else the reason it is refused (nothing in it is touched). ``make=False`` (a dry run): an absent ``d`` is not refused."""
    try:
        if make:
            os.mkdir(d, 0o700)
    except OSError:
        pass                                                             # present already (checked below), or not creatable (named below)
    try:
        st = os.lstat(d)
    except OSError as e:
        return f"cannot be made ({e.strerror})" if make else None
    if stat.S_ISLNK(st.st_mode):
        return "is a symbolic link"
    if not stat.S_ISDIR(st.st_mode):
        return "is not a directory"
    if st.st_uid != os.getuid() and os.getuid() != 0:
        return f"belongs to uid {st.st_uid}, not to uid {os.getuid()}"
    if st.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        return "is writable by group or others (mode %04o)" % stat.S_IMODE(st.st_mode)
    return None


def override(overrides, dry_run: bool = False) -> List[str]:
    """The token ``design`` / ``warm`` append to the typed overrides: ``[]`` when the key is typed, there is no checkout, or the checkout's
    schedules directory is this user's to write and closed to others; else ``[KEY=<per-user directory>]`` with one stderr line naming why —
    or, the per-user directory refused, that refusal on one line and ``[KEY=<a private temporary directory, removed at exit>]``."""
    from .upstream_args import parse
    if KEY in parse([str(o) for o in (overrides or [])]).typed():
        return []
    d = checkout_dir()
    if d is None:
        return []
    if open_to_all(d):
        open_d = f"REFUSED {d}: every user can write it (mode {stat.S_IMODE(os.stat(d).st_mode):04o}) and upstream unpickles what it finds there (`chmod o-w {d}` to use it again)"
    elif writable(d):
        return []
    else:
        open_d = None
    t = per_user_dir()
    bad = refusal(t, make=not dry_run)
    if bad is None:
        _report.emit(f"{_report.PREFIX} schedule cache: {open_d} — {KEY}={t} is used instead" if open_d else
                     f"{_report.PREFIX} schedule cache: {KEY}={t} — {d} is not writable by uid {os.getuid()}")
        return [f"{KEY}={t}"]
    also = f"; also {open_d}" if open_d else f"; {d} is not writable by uid {os.getuid()}"
    fix = f"remove or repair {t} (expected: a directory of uid {os.getuid()}, mode 0700) or type {KEY}=<a directory of yours>"
    if dry_run:                                                          # nothing runs: nothing is made
        _report.emit(f"{_report.PREFIX} schedule cache: REFUSED {t}: it {bad}; nothing in it is read and a run would compute its schedule anew every time — {fix}{also}")
        return []
    tmp = tempfile.mkdtemp(prefix="rfdiffusion1_schedules-", dir=os.path.dirname(t))          # mode 0700, a name nobody could make first
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    _report.emit(f"{_report.PREFIX} schedule cache: REFUSED {t}: it {bad}; nothing in it is read and this run computes its schedule anew into {tmp} (removed at exit) — {fix}{also}")
    return [f"{KEY}={tmp}"]
