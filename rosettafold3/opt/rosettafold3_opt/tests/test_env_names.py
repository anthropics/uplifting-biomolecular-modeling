"""A mistyped ROSETTAFOLD3_OPT_* name is refused by name (never ignored): the CLI (rc 2) before any command; activation refuses it too
(test_activation.py); the declared names are one table (stack.ENV_NAMES) and every name the tree reads is in it."""
import io
import os
import re
import sys

from .. import cli, stack


def test_declared_names_are_every_name_the_tree_reads():
    root = stack.tree_root()
    seen = set()
    for rel in ("run.sh", "configs/h100.env"):
        p = os.path.join(root, rel)
        if os.path.exists(p):
            seen |= set(re.findall(r"ROSETTAFOLD3_OPT_[A-Z0-9_]+", open(p, encoding="utf-8").read()))
    for f in os.listdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))):
        if f.endswith(".py"):
            seen |= set(re.findall(r"ROSETTAFOLD3_OPT_[A-Z0-9_]+", open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f), encoding="utf-8").read()))
    seen.discard("ROSETTAFOLD3_OPT_PATH_PROBE")                       # stock_fold's probe sentinel: a stdout marker, not an environment name
    assert seen == set(stack.ENV_NAMES), seen ^ set(stack.ENV_NAMES)
    assert stack.undeclared_env({"ROSETTAFOLD3_OPT": "exact", "ROSETTAFOLD3_OPT_CKPT": "/w", "RF3_HOIST": "1"}) == {}
    assert stack.undeclared_env({"ROSETTAFOLD3_OPT_MODE": "exact", "ROSETTAFOLD3_OPTS": "x"}) == {"ROSETTAFOLD3_OPT_MODE": "exact"}


def test_cli_refuses_an_undeclared_name(monkeypatch):
    monkeypatch.setenv("ROSETTAFOLD3_OPT_MODE", "exact")
    err = io.StringIO()
    monkeypatch.setattr(sys, "stderr", err)
    assert cli.main(["check", "--mode", "exact"]) == cli.EXIT_USAGE
    assert err.getvalue().startswith("[rosettafold3-opt] undeclared ROSETTAFOLD3_OPT_* / ROSETTAFOLD3_BIG_* names in the environment (this tree reads ROSETTAFOLD3_OPT_STOCK_PYTHON, ")
    assert "ROSETTAFOLD3_OPT_MODE='exact'" in err.getvalue()
