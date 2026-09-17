"""install: the add-on's own install.sh applied into a second stub interpreter (its `python` on PATH), states asserted before and after;
the stock interpreter is never written; one interpreter for both is refused."""
import os
import shutil

import pytest

from .. import install, stack, tree
from . import _stubs


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash")
def test_install_applies_the_kit_into_the_patched_interpreter(tmp_path, monkeypatch, capsys):
    stock = _stubs.make_tree(str(tmp_path / "stock_sp"), "stock")
    opt = _stubs.make_tree(str(tmp_path / "opt_sp"), "stock")          # a second, still pristine, tree
    spy = _stubs.make_interpreter(str(tmp_path / "sbin"), stock)
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), opt)
    lists = stack.tree_digests()
    before = {f: tree.sha256_file(os.path.join(stock, f)) for f in tree.RF3_FILES[1:]}
    res = install.run(stock_python=spy, opt_python=opy, log_path=str(tmp_path / "install.log"))
    assert res["status"] == "PASS" and res["stock"]["state"] == "stock" and res["opt_before"]["state"] == "stock" and res["opt"]["state"] == "patched"
    assert res["kit"]["already"] is False and res["kit"]["apply"]["rc"] == 0
    assert res["kit"]["kit_line"] and "RF3_HOIST" in res["kit"]["kit_line"]              # the kit's own install-state line (install.sh:39)
    assert tree.state_of(opy, lists).state == "patched"
    assert {f: tree.sha256_file(os.path.join(stock, f)) for f in tree.RF3_FILES[1:]} == before   # the stock tree is untouched
    assert tree.state_of(spy, lists).state == "stock"
    assert any(f.endswith(".hoist_orig") for f in os.listdir(os.path.join(opt, "rf3", "loss")))     # the kit's own backups
    # second run: already installed (install.sh --check passes)
    res2 = install.run(stock_python=spy, opt_python=opy)
    assert res2["kit"]["already"] is True and res2["opt"]["state"] == "patched"
    err = capsys.readouterr().err
    assert "[rosettafold3-opt] install stock interpreter" in err and "patched interpreter" in err


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash")
def test_no_addon_does_every_step_but_the_add_on(tmp_path, capsys):
    stock = _stubs.make_tree(str(tmp_path / "stock_sp"), "stock")
    opt = _stubs.make_tree(str(tmp_path / "opt_sp"), "stock")
    spy = _stubs.make_interpreter(str(tmp_path / "sbin"), stock)
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), opt)
    before = {f: tree.sha256_file(os.path.join(opt, f)) for f in tree.RF3_FILES}
    res = install.run(stock_python=spy, opt_python=opy, addon=False)
    assert res["status"] == "PASS" and res["opt_before"]["state"] == "stock" and res["opt"]["state"] == "stock" and res["stock"]["state"] == "stock"
    assert res["kit"]["skipped"] is True and res["kit"]["apply"] is None and res["kit"]["check"] is None and res["kit"]["kit_line"] is None
    assert {f: tree.sha256_file(os.path.join(opt, f)) for f in tree.RF3_FILES} == before          # nothing is written into the patched tree
    assert not [f for _, _, fs in os.walk(opt) for f in fs if f.endswith((".hoist_orig", ".hoist_absent"))]   # install.sh never ran: no backups, no markers
    assert "add-on NOT applied (--no-addon)" in capsys.readouterr().err                        # said so, by name
    assert install.summary_line(res).endswith("opt=stock addon=not_applied(--no-addon)")
    res2 = install.run(stock_python=spy, opt_python=opy)                                       # the later install applies it
    assert res2["kit"]["already"] is False and res2["opt"]["state"] == "patched" and tree.state_of(opy, stack.tree_digests()).state == "patched"
    assert install.summary_line(res2).endswith("opt=patched kit_already_installed=False")


def test_no_addon_on_an_already_patched_interpreter_keeps_it(tmp_path):
    stock = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    opt = _stubs.make_tree(str(tmp_path / "op"), "patched")
    spy = _stubs.make_interpreter(str(tmp_path / "bin"), stock)
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), opt)
    res = install.run(stock_python=spy, opt_python=opy, addon=False)
    assert res["opt_before"]["state"] == "patched" and res["opt"]["state"] == "patched" and res["kit"]["skipped"] is True


def test_install_refuses_one_interpreter(tmp_path):
    stock = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    py = _stubs.make_interpreter(str(tmp_path / "bin"), stock)
    with pytest.raises(RuntimeError, match="two interpreters"):
        install.run(stock_python=py, opt_python=py)
    assert tree.state_of(py, stack.tree_digests()).state == "stock"


def test_install_refuses_a_patched_stock_interpreter(tmp_path):
    p = _stubs.make_tree(str(tmp_path / "sp"), "patched")
    o = _stubs.make_tree(str(tmp_path / "op"), "stock")
    spy = _stubs.make_interpreter(str(tmp_path / "bin"), p)
    opy = _stubs.make_interpreter(str(tmp_path / "obin"), o)
    with pytest.raises(RuntimeError, match="tree state is 'patched'"):
        install.run(stock_python=spy, opt_python=opy)


def test_install_missing_interpreters(tmp_path):
    with pytest.raises(RuntimeError, match="stock interpreter not found"):
        install.run(stock_python=str(tmp_path / "nope"))
    stock = _stubs.make_tree(str(tmp_path / "sp"), "stock")
    spy = _stubs.make_interpreter(str(tmp_path / "bin"), stock)
    with pytest.raises(RuntimeError, match="patched interpreter not found"):
        install.run(stock_python=spy, opt_python=str(tmp_path / "nope2"))
