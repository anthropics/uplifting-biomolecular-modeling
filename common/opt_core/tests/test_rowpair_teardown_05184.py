"""``mem.rowpair.dist`` — a rank's leave of its process group and the exit guard. A HEALTHY process leaves through the collective library's
own ``destroy_process_group`` in the calling thread, unbounded and silent (it may wait for its peers' leave — that wait is correct), and the
exit guard never touches it. A FAILED process (a non-zero registered ``exit_code``, ``failure()``, an exception propagating through the caller,
an uncaught one already reported) leaves within ``ROWPAIR_ABORT_TIMEOUT_S``: silent in time; on overrun one ``RANKLEAVE`` line, the pending
EXIT tallies and ``os._exit`` with the registered status there and then; and a failed process still in its exit hooks / threads
``ROWPAIR_EXIT_GRACE_S`` after its main thread ended is ended by the guard (``RANKSTUCK``, stacks, status). The collective library is a
stand-in whose calls sleep or raise (no torch group, CPU, seconds); the exits are proven in child interpreters.

Run: ``python -m pytest tests/test_rowpair_teardown_05184.py -q -rfE``.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from opt_core.mem.rowpair import dist as D  # noqa: E402


def _fake_dist(destroy_s: float = 0.0, destroy_raises=None):
    """A stand-in for ``torch.distributed`` as :func:`D.destroy` uses it: ``is_available/is_initialized`` say a group is live and
    ``destroy_process_group`` sleeps ``destroy_s`` in slices (then raises ``destroy_raises`` if given); ``calls`` counts it and records the thread."""
    calls = {"destroy": 0, "threads": []}

    def destroy_process_group():
        calls["destroy"] += 1
        calls["threads"].append(__import__("threading").current_thread().name)
        t0 = time.monotonic()
        while time.monotonic() - t0 < destroy_s:
            time.sleep(0.01)
        if destroy_raises is not None:
            raise destroy_raises
    fake = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, destroy_process_group=destroy_process_group,
                                 distributed_c10d=types.SimpleNamespace())
    return fake, calls


@pytest.fixture
def lib(monkeypatch):
    """``D.destroy`` on the stand-in library; the hard exit is RECORDED (its line), never taken (it would end the test interpreter); no failure
    is registered and none is inferable unless a test says so."""
    ended = []
    monkeypatch.setattr(D, "_hard_exit", lambda line: ended.append(line))
    monkeypatch.setitem(D._EXIT, "code", None)
    monkeypatch.setitem(D._EXIT, "failure", False)
    monkeypatch.delenv(D.ENV_ABORT_TIMEOUT, raising=False)
    monkeypatch.setenv("ROWPAIR_TAG", "kit")
    monkeypatch.setenv("ROWPAIR_RANK", "1")
    for name in ("last_exc", "last_value", "last_type", "last_traceback"):     # a previous test's failure report must not read as this process's
        monkeypatch.delattr(sys, name, raising=False)

    def install(**kw):
        fake, calls = _fake_dist(**kw)
        monkeypatch.setattr(D, "dist", fake)
        return calls
    return install, ended


# ================================================================================================ the healthy leave: the library's own, unbounded
def test_a_healthy_leave_is_the_librarys_own_call_in_the_calling_thread_unbounded_and_silent(lib, capsys):
    install, ended = lib
    calls = install(destroy_s=1.5)                                                 # longer than the bound below: a healthy leave is NOT bounded
    t0 = time.monotonic()
    assert D.destroy(timeout_s=0.2) == "destroyed"
    assert time.monotonic() - t0 >= 1.4                                            # it waited for the library (its peers), as before the bound existed
    assert calls["destroy"] == 1 and calls["threads"] == [__import__("threading").current_thread().name]   # inline, no helper thread
    assert capsys.readouterr().err == "" and ended == []                           # nothing printed, nothing ended: byte-identical to a run without the module's help


def test_no_group_is_none(monkeypatch, capsys):
    monkeypatch.setattr(D, "dist", types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: False))
    assert D.destroy() == "none" and D.destroy(failure=True) == "none" and capsys.readouterr().err == ""


def test_what_counts_as_a_failure_path(lib, monkeypatch):
    install, ended = lib
    assert D._failure_known() is False                                             # a clean process
    D.exit_code(0)
    assert D._failure_known() is False                                             # a registered success
    D.exit_code(3)
    assert D._failure_known() is True                                              # a registered non-zero verdict
    D.exit_code(None)
    D.failure(True)
    assert D._failure_known() is True                                              # the explicit word
    D.failure(False)
    try:
        raise RuntimeError("simulated OOM")
    except RuntimeError:
        assert D._failure_known() is True                                          # an exception in flight (a leave inside except/finally)
        assert D._failure_known(in_caller=False) is False                          # ... which only the calling thread can see
    try:
        raise SystemExit(0)
    except SystemExit:
        assert D._failure_known() is False                                         # sys.exit(0) propagating is not a failure
    try:
        raise SystemExit(2)
    except SystemExit:
        assert D._failure_known() is True
    assert D._failure_known() is False
    monkeypatch.setattr(sys, "last_value", RuntimeError("uncaught, already reported"), raising=False)
    assert D._failure_known(in_caller=False) is True                               # an exit hook after the command raised
    D.exit_code(0)
    assert D._failure_known(in_caller=False) is False                              # ... unless the kit registered success
    try:
        raise OSError("handled above the leave")
    except OSError:
        assert D._failure_known() is False                                         # a registered success under an active outer handler stays healthy
    D.exit_code(None)


# ================================================================================================ the failure-path leave: bounded, then the process ends
def test_a_failed_ranks_leave_that_returns_in_time_is_silent(lib, capsys):
    install, ended = lib
    calls = install(destroy_s=0.05)
    D.exit_code(1)
    t0 = time.monotonic()
    assert D.destroy(timeout_s=2.0) == "destroyed"
    assert time.monotonic() - t0 < 1.0 and calls["destroy"] == 1 and calls["threads"] == ["rowpair-destroy"]
    assert capsys.readouterr().err == "" and ended == []


@pytest.mark.parametrize("how", ["registered", "explicit", "propagating"])
def test_a_failed_ranks_leave_that_overruns_ends_the_process_at_the_bound(lib, capsys, how):
    install, ended = lib
    calls = install(destroy_s=30.0)                                                # the library would wait 30 s (a wedged communicator: for ever)
    t0 = time.monotonic()
    if how == "registered":
        D.exit_code(5)
        verdict = D.destroy(timeout_s=1.0)
    elif how == "explicit":
        verdict = D.destroy(timeout_s=1.0, failure=True)
    else:
        try:
            raise RuntimeError("simulated OOM")
        except RuntimeError:
            verdict = D.destroy(timeout_s=1.0)                                     # e.g. a kit's `finally: destroy()` while its failure propagates
    wall = time.monotonic() - t0
    assert verdict == "abandoned" and 1.0 <= wall < 3.0, (verdict, wall)           # the bound (+ thread start slack), never the library's 30 s
    assert calls["destroy"] == 1
    assert ended == [f"[kit] RANKLEAVE rank=1 destroy=timeout:1s verdict=abandoned exit={5 if how == 'registered' else 70}"], ended
    assert capsys.readouterr().err == ""                                           # the line belongs to the hard exit (printed with the tallies, then os._exit)


def test_the_bound_comes_from_the_environment_first(lib, monkeypatch):
    install, ended = lib
    install(destroy_s=30.0)
    monkeypatch.setenv(D.ENV_ABORT_TIMEOUT, "0.4")
    t0 = time.monotonic()
    assert D.destroy(timeout_s=60.0, failure=True) == "abandoned" and time.monotonic() - t0 < 2.0
    assert "destroy=timeout:0.4s" in ended[0]


@pytest.mark.parametrize("failed", [False, True])
def test_an_error_raised_by_the_library_propagates_as_before(lib, failed):
    install, ended = lib
    install(destroy_s=0.0, destroy_raises=ValueError("no group to destroy"))
    with pytest.raises(ValueError, match="no group to destroy"):
        D.destroy(timeout_s=2.0, failure=failed)
    assert ended == []


@pytest.mark.parametrize("code,status", [(0, 0), (1, 1), (3, 3), (70, 70), (256, 1), (257, 1), (-1, 255), (None, 70)])
def test_the_registered_status_is_a_process_status_and_never_turns_a_failure_into_success(monkeypatch, code, status):
    monkeypatch.setitem(D._EXIT, "code", None)
    D.exit_code(code)
    assert D._exit_status() == status


def test_line_formats():
    assert D.rankleave_line("kit", 3, "timeout:60s", "abandoned", 1) == "[kit] RANKLEAVE rank=3 destroy=timeout:60s verdict=abandoned exit=1"
    assert D.rankstuck_line("kit", 1, 90.04, 70, "teardown_stuck") == "[kit] RANKSTUCK rank=1 reason=teardown_stuck alive_s=90.04 exit=70"


def test_the_guard_arms_once_and_its_default_grace_outlasts_the_leave_bound(monkeypatch):
    monkeypatch.setitem(D._EXIT, "armed", False)
    started = []
    monkeypatch.setattr(D.threading, "Thread", lambda target, name, daemon: types.SimpleNamespace(start=lambda: started.append(name)))
    monkeypatch.delenv(D.ENV_EXIT_GRACE, raising=False)
    monkeypatch.setenv(D.ENV_ABORT_TIMEOUT, "200")
    seen = {}
    real = D._env_seconds
    monkeypatch.setattr(D, "_env_seconds", lambda name, default: seen.setdefault(name, real(name, default)))
    assert D.arm_exit_guard() is True and started == ["rowpair-exit-guard"]
    assert D.arm_exit_guard() is False and started == ["rowpair-exit-guard"]
    assert seen[D.ENV_ABORT_TIMEOUT] == 200.0                                       # the leave's bound was read: grace = max(90, 200 + 15)


# ================================================================================================ the exits, in child interpreters
def _child(body: str, timeout: float = 60.0, env=None):
    code = "import os, sys, time, threading, types, atexit\nsys.path.insert(0, %r)\nfrom opt_core.mem.rowpair import dist as D\nfrom opt_core import report\n" % ROOT + textwrap.dedent(body)
    e = {k: v for k, v in os.environ.items() if not k.startswith("ROWPAIR_")}
    e.update({"ROWPAIR_TAG": "kit", "ROWPAIR_RANK": "1"}, **(env or {}))
    t0 = time.monotonic()
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=timeout, env=e)
    return p.returncode, p.stderr, time.monotonic() - t0


_FAKE_IN_CHILD = """
    def hang(*a, **k):
        time.sleep(120)
    def slow(*a, **k):
        time.sleep(2.0); print("library teardown ran", file=sys.stderr, flush=True)
    def healthy(*a, **k):
        print("library teardown ran", file=sys.stderr, flush=True)
    def install(fn):
        D.dist = types.SimpleNamespace(is_available=lambda: True, is_initialized=lambda: True, destroy_process_group=fn, distributed_c10d=types.SimpleNamespace())
"""


@pytest.mark.parametrize("registered,expect_rc", [(None, 70), (5, 5)])
def test_the_guard_ends_a_FAILED_process_whose_teardown_never_finishes(registered, expect_rc):
    rc, err, wall = _child(f"""
        report.register_exit_tally("kit", lambda: "[kit] EXIT pid=0 levers=ok")   # registered first: atexit would print it last — it never gets there
        D.arm_exit_guard(grace_s=1.0)
        if {registered!r} is not None:
            D.exit_code({registered!r})
        else:
            D.failure()                                                # e.g. inferred: the command raised
        threading.Thread(target=time.sleep, args=(120,), name="never-ends", daemon=False).start()   # the interpreter would wait for it for 120 s
        print("main done", file=sys.stderr, flush=True)
    """)
    assert rc == expect_rc, (rc, err[-600:])
    assert wall < 15.0, wall                                                       # ~1 s grace + interpreter start, not 120 s
    assert "[kit] RANKSTUCK rank=1 reason=teardown_stuck alive_s=" in err and f" exit={expect_rc}" in err, err[-600:]
    assert "never-ends" in err or "Thread 0x" in err, err[-600:]                   # the thread stacks follow the line
    assert err.count("[kit] EXIT pid=0 levers=ok") == 1                            # the kit's EXIT tally is printed by the forced exit


def test_the_guard_after_an_uncaught_exception_needs_no_registration():
    rc, err, wall = _child("""
        D.arm_exit_guard(grace_s=1.0)
        threading.Thread(target=time.sleep, args=(120,), name="never-ends", daemon=False).start()
        raise RuntimeError("simulated OOM in the command")         # reported by the interpreter (sys.last_exc): the process is a failed one
    """)
    assert rc == 70 and wall < 15.0, (rc, wall, err[-600:])
    assert "RuntimeError: simulated OOM in the command" in err and "[kit] RANKSTUCK rank=1 reason=teardown_stuck" in err


def test_the_guard_never_touches_a_HEALTHY_process_however_long_its_exit_takes():
    """Rank 0 writing its outputs for longer than the grace after its command returned (a non-daemon writer thread; an exit hook) exits on its
    own, with its own status, nothing printed by the core."""
    rc, err, wall = _child("""
        D.arm_exit_guard(grace_s=0.5)
        D.exit_code(0)                                             # a registered success (a kit that registers nothing is the same: see below)
        threading.Thread(target=time.sleep, args=(3.0,), name="writer", daemon=False).start()
        atexit.register(lambda: (time.sleep(1.0), print("late hook ran", file=sys.stderr, flush=True)))
        print("main done", file=sys.stderr, flush=True)
    """)
    assert rc == 0 and wall >= 3.5, (rc, wall, err[-400:])
    assert [ln for ln in err.splitlines() if ln.strip()] == ["main done", "late hook ran"], err
    rc, err, wall = _child("""
        D.arm_exit_guard(grace_s=0.5)                              # nothing registered at all
        threading.Thread(target=time.sleep, args=(3.0,), name="writer", daemon=False).start()
        sys.exit(0)
    """)
    assert rc == 0 and wall >= 2.5 and err.strip() == "", (rc, wall, err[-400:])


def test_a_FAILED_ranks_leave_that_overruns_ends_the_process_there_and_then():
    """A kit's `finally: destroy()` while its failure propagates, on a wedged communicator: the traceback that would otherwise be lost, the
    RANKLEAVE line, the kit's EXIT tally, os._exit(registered) — nothing after the call runs (not the kit's re-raise, not the exit hooks,
    not the library's destructors)."""
    rc, err, wall = _child(_FAKE_IN_CHILD + """
    report.register_exit_tally("kit", lambda: "[kit] EXIT pid=0 levers=ok")
    atexit.register(lambda: print("hook: never runs", file=sys.stderr, flush=True))
    install(hang)
    D.exit_code(1)
    t0 = time.monotonic()
    try:
        try:
            raise RuntimeError("simulated OOM")
        finally:
            D.destroy(timeout_s=1.0)
    except RuntimeError:
        print("never reached", file=sys.stderr, flush=True)
    """)
    assert rc == 1, (rc, err[-800:])
    assert wall < 15.0, wall
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert "RuntimeError: simulated OOM" in err                                    # the propagating failure's traceback, printed by the hard exit
    tail = [ln for ln in lines if ln.startswith("[kit] ") or "never" in ln or "hook:" in ln]
    assert tail == ["[kit] RANKLEAVE rank=1 destroy=timeout:1s verdict=abandoned exit=1", "[kit] EXIT pid=0 levers=ok"], tail


def test_the_leave_registered_at_join_bounds_a_FAILED_rank_that_never_left():
    """The command raised and nothing in the kit leaves the group: the interpreter reports the exception, then the join-time hook sees a
    failed process and leaves within the bound instead of blocking in the library's destructors."""
    rc, err, wall = _child(_FAKE_IN_CHILD + """
    report.register_exit_tally("kit", lambda: "[kit] EXIT pid=0 levers=ok")
    D._register_leave_at_exit()                                    # what init_from_env does when it joins a group
    install(hang)
    raise RuntimeError("simulated OOM")
    """, env={"ROWPAIR_ABORT_TIMEOUT_S": "1"})
    assert rc == 70, (rc, err[-800:])                                             # nothing registered a status: the stuck-teardown word
    assert wall < 15.0, wall
    assert "RuntimeError: simulated OOM" in err
    tail = [ln for ln in err.splitlines() if ln.startswith("[kit] ")]
    assert tail == ["[kit] RANKLEAVE rank=1 destroy=timeout:1s verdict=abandoned exit=70", "[kit] EXIT pid=0 levers=ok"], tail


def test_the_leave_registered_at_join_does_nothing_on_a_HEALTHY_process():
    """A healthy rank whose kit never calls destroy(): the hook leaves the group alone (no library call at exit, as before it existed) —
    even when the library's teardown would take longer than any bound."""
    rc, err, wall = _child(_FAKE_IN_CHILD + """
    report.register_exit_tally("kit", lambda: "[kit] EXIT pid=0 levers=ok")
    D._register_leave_at_exit()
    D.arm_exit_guard(grace_s=0.5)
    install(hang)                                                  # would block for 120 s if anything called it
    print("main done", file=sys.stderr, flush=True)
    sys.exit(0)
    """, env={"ROWPAIR_ABORT_TIMEOUT_S": "0.5"})
    assert rc == 0 and wall < 10.0, (rc, wall, err[-400:])
    assert [ln for ln in err.splitlines() if ln.strip()] == ["main done", "[kit] EXIT pid=0 levers=ok"], err


def test_a_HEALTHY_kits_own_leave_waits_for_the_library_however_long():
    """The regression 0.5.18.6 exists for: a rank that finished early leaves while rank 0 still writes; the library's teardown waits for the
    peer (seconds here, minutes in life) and the rank simply waits — no bound, no line, its own status."""
    rc, err, wall = _child(_FAKE_IN_CHILD + """
    report.register_exit_tally("kit", lambda: "[kit] EXIT pid=0 levers=ok")
    D._register_leave_at_exit()
    D.arm_exit_guard(grace_s=0.5)
    install(slow)                                                  # the peer arrives 2 s later
    D.exit_code(0)
    assert D.destroy(timeout_s=0.2) == "destroyed"                 # the kit's own leave at the end of its command
    sys.exit(0)
    """, env={"ROWPAIR_ABORT_TIMEOUT_S": "0.2"})
    assert rc == 0 and wall >= 2.0, (rc, wall, err[-400:])
    assert [ln for ln in err.splitlines() if ln.strip()] == ["library teardown ran", "[kit] EXIT pid=0 levers=ok"], err


def test_the_registered_leave_words_a_raising_teardown_in_one_line_and_keeps_the_status():
    rc, err, wall = _child(_FAKE_IN_CHILD + """
    D._register_leave_at_exit()
    def raising(*a, **k): raise ValueError("no such group")
    install(raising)
    D.exit_code(3)                                                 # a failed rank (else the hook does nothing)
    sys.exit(3)
    """)
    assert rc == 3, (rc, err[-400:])
    assert [ln for ln in err.splitlines() if ln.strip()] == ["[kit] RANKLEAVE rank=1 destroy=raised:ValueError verdict=none exit=-"], err
