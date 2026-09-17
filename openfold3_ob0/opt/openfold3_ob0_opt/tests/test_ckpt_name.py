"""--inference-ckpt-name is honoured as upstream's own flag (cli.resolve_checkpoint): a NAME is passed through instead of a path and the environment's
default checkpoint path is NOT injected; --ckpt PATH together with a NAME is refused by name; with neither, the path comes from --ckpt or
$OPENFOLD3_OB0_CKPT as before."""

import pytest

from openfold3_ob0_opt import cli
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()


def _pred_args(*extra):
    return cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "/tmp/o"] + list(extra))


def test_a_name_is_passed_through_and_the_env_default_path_is_not_injected(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_OB0_CKPT", "/weights/w.pt")
    a = _pred_args("--inference-ckpt-name", "openbind-2025-06-30-174k")
    assert cli.resolve_checkpoint(a) == (None, "openbind-2025-06-30-174k")
    argv = cli.stock_argv(a, HOME, None, "q.json")
    assert "--inference-ckpt-name" in argv and argv[argv.index("--inference-ckpt-name") + 1] == "openbind-2025-06-30-174k"
    assert "--inference-ckpt-path" not in argv                                                  # the environment default is not injected under an explicit name
    info = cli.weights_by_name("openbind-2025-06-30-174k", "pred --mode off --det 0")
    assert info == {"ckpt": None, "ckpt_name": "openbind-2025-06-30-174k", "weights_pinned": None, "resolved_by": "upstream"}


def test_a_path_and_a_name_together_are_refused_by_name(monkeypatch, capsys):
    monkeypatch.delenv("OPENFOLD3_OB0_CKPT", raising=False)
    a = _pred_args("--ckpt", "/w.pt", "--inference-ckpt-name", "openbind-2025-06-30-174k")
    with pytest.raises(ValueError) as e:
        cli.resolve_checkpoint(a)
    assert "ambiguous checkpoint" in str(e.value) and "--inference-ckpt-name openbind-2025-06-30-174k" in str(e.value)
    assert cli.main(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "/tmp/o", "--ckpt", "/w.pt", "--inference-ckpt-name", "x"]) == cli.EXIT_USAGE
    assert "ambiguous checkpoint" in capsys.readouterr().err


def test_without_a_name_the_path_is_the_flag_else_the_environment(monkeypatch):
    monkeypatch.setenv("OPENFOLD3_OB0_CKPT", "/weights/w.pt")
    assert cli.resolve_checkpoint(_pred_args()) == ("/weights/w.pt", None)
    assert cli.resolve_checkpoint(_pred_args("--ckpt", "/other.pt")) == ("/other.pt", None)
    argv = cli.stock_argv(_pred_args(), HOME, "/weights/w.pt", "q.json")
    assert argv[argv.index("--inference-ckpt-path") + 1] == "/weights/w.pt" and "--inference-ckpt-name" not in argv
    monkeypatch.delenv("OPENFOLD3_OB0_CKPT")
    assert cli.resolve_checkpoint(_pred_args()) == (None, None)                               # cmd_pred then refuses: no checkpoint
