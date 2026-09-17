"""The accelerator proof (kernels.py): the reader's words off a built model's bound objects, the expected words per stack from the modes
table and stock/PINS.json, the KERNELS line grammar, the guard naming a shortfall (KERNELS SHORT, the run proceeding
path), and the tables that must agree (modes / _autoload / PINS.json)."""
import json
import os
import re
import subprocess
import sys
import textwrap

import pytest

from esmc_opt import _autoload, cli, kernels, modes, report

from . import _fake_upstream as fake
from ._paths import OPT, STOCK, TREE

LINE = re.compile(r"^\[esmc-opt (off|exact)\] KERNELS route=(stock|exact) via=(stock_child|kit) regime=(b1|batched) stack=(\S+) "
                  r"flash_attn=(\S+) rotary=(\S+) te=(\S+) xformers=(\S+)$")


def _pins():
    return json.load(open(os.path.join(STOCK, "PINS.json")))


@pytest.fixture(autouse=True)
def _clean():
    kernels._reset_for_tests()
    yield
    fake.uninstall()
    kernels._reset_for_tests()


@pytest.fixture
def bare_host(monkeypatch):
    """The host the fallback tests describe: no accelerator distribution installed and none importable, so a fallback the fake binds reads
    `fallback:<impl>:absent` whatever the interpreter running the tests carries (on the pinned stack itself the same fake would read
    `import-error(…)` / `importable-unbound`, the words of a broken or unused install — named by test_cli_and_env's KERNELS SHORT case)."""
    monkeypatch.setattr(kernels, "_dist_version", lambda name: None)
    monkeypatch.setattr(kernels, "_importable", lambda modname: False)


# ------------------------------------------------------------------------------------------------------------ the tables agree
def test_mode_tables_agree():
    assert set(modes.ROUTE_WORDS) == set(modes.MODES)
    assert set(modes.STOCK_MODES) < set(modes.MODES) and set(modes.STOCK_USE_FLASH_ATTN) == set(modes.REGIMES) == set(modes.STOCK_RULE_NAMES)
    assert all(modes.KIT_MODES[m] == () for m in modes.STOCK_MODES)             # a stock-class mode applies no lever
    for rg in modes.REGIMES:
        assert modes.load_call("exact", rg) == modes.KIT_CALL == {"device": "cuda", "use_flash_attn": True} and modes.call_rule("exact", rg) is None
    for rg in modes.REGIMES:                                                                             # stock: the shipped varlen class in both regimes (compatibility.py:157-158 defaults, named)
        assert modes.load_call("off", rg) == {"device": "cuda", "use_flash_attn": True} and modes.call_rule("off", rg) is None and modes.STOCK_USE_FLASH_ATTN[rg] is True
    assert modes.load_call("off", None) == modes.load_call("off", modes.DEFAULT_REGIME)
    assert set(_autoload.MODES) == set(modes.MODES) and tuple(_autoload.STOCK_MODES) == tuple(modes.STOCK_MODES)
    assert tuple(modes.ACCELERATORS) == ("flash_attn", "rotary", "te", "xformers")
    assert kernels.VIAS == ("stock_child", "kit") and not hasattr(cli, "EXIT_KERNELS") and not hasattr(kernels, "KernelsRefused")   # a shortfall is named, never an exit


def test_accelerator_pins_exist_in_every_stack():
    p = _pins()
    assert modes.pinned_stack(p) == "accel"
    for name, a in modes.ACCELERATORS.items():
        assert a["site"] and a["engaged"], name
        if a["pin"] is None:
            assert p["pins"][a["dist"]] is None, name                            # pinned absent on every stack
        else:
            for st in p["stacks"].values():
                assert a["pin"] in st, (name, a["pin"])
    assert fake.FLASH_VERSION == p["stacks"]["accel"]["flash_attn"] and fake.TE_VERSION == p["stacks"]["accel"]["transformer_engine"]


def test_expected_words_per_stack():
    p = _pins()
    acc = kernels.expected(p, "accel")
    assert acc == {"flash_attn": f"engaged:flash_attn@{p['stacks']['accel']['flash_attn']}", "rotary": f"engaged:flash_attn.ops.triton.rotary@{p['stacks']['accel']['flash_attn']}",
                   "te": f"engaged:transformer_engine@{p['stacks']['accel']['transformer_engine']}", "xformers": "absent"}
    with pytest.raises(modes.ResolveError):
        kernels.expected(p, "nostack")


# --------------------------------------------------------------------------------------------------------- the reader's words
def test_census_engaged_words_and_line_grammar(capsys):
    client = fake.install("engaged")
    rec = kernels.prove_once(client, mode="exact", via="kit", p=_pins())
    assert rec["verdict"] == "ok" and rec["differences"] == []
    w = rec["words"]
    assert w["flash_attn"] == f"engaged:flash_attn@{fake.FLASH_VERSION}"
    assert w["rotary"] == f"engaged:flash_attn.ops.triton.rotary@{fake.FLASH_VERSION}"
    assert w["te"] == f"engaged:transformer_engine@{fake.TE_VERSION}(LayerNormLinear,LayerNormMLP,Linear)"
    assert w["xformers"] == "absent"
    assert rec["bound"]["attn"] == ["esm.models.esmc.layers.EsmcFlashMultiHeadAttention"] and rec["use_flash_attn_resolved"] is True
    line = capsys.readouterr().out.strip().splitlines()[-1]
    m = LINE.match(line)
    assert m, line
    assert m.group(1, 2, 3, 4, 5) == ("exact", "exact", "kit", "b1", "accel")
    assert line == rec["line"] == report.kernels_line(rec)
    assert kernels.prove_once(client, mode="exact", via="kit", p=_pins()) is rec      # once per model object
    assert kernels.record() is rec


@pytest.mark.parametrize("mode,route", [("off", "stock"), ("exact", "exact")])
def test_route_word_per_mode(mode, route, capsys):
    rec = kernels.prove_once(fake.install("engaged"), mode=mode, via="stock_child" if mode in modes.STOCK_MODES else "kit", p=_pins())
    assert rec["route"] == route and rec["mode"] == mode
    assert LINE.match(capsys.readouterr().out.strip().splitlines()[-1]).group(1, 2) == (mode, route)


def test_guard_names_te_fallback(capsys, bare_host):
    client = fake.install("te_off")
    rec = kernels.prove_once(client, mode="off", via="stock_child", p=_pins())
    assert rec["verdict"] == "short"                                                         # named, returned — never raised
    assert [d["accelerator"] for d in rec["differences"]] == ["te"]
    assert rec["words"]["te"] == "fallback:torch:absent" and rec["words"]["flash_attn"].startswith("engaged:")
    out = capsys.readouterr().out.strip().splitlines()
    assert out[0].startswith("[esmc-opt off] KERNELS route=stock via=stock_child regime=b1 stack=accel")   # the KERNELS line first
    assert out[1].startswith("[esmc-opt] KERNELS SHORT route=stock via=stock_child regime=b1 stack=accel: te=fallback:torch:absent (expected engaged:transformer_engine@")
    assert out[1].endswith("(named; the run proceeds on what the model engages)") and out[1] == rec["short_line"] and len(out) == 2
    assert "REFUSED" not in " ".join(out) and "exit" not in out[1]


def test_guard_names_every_fallback(capsys, bare_host):
    p = _pins()
    rec = kernels.prove_once(fake.install("fallback"), mode="off", via="stock_child", p=p)
    assert rec["verdict"] == "short" and [d["accelerator"] for d in rec["differences"]] == ["flash_attn", "rotary", "te"]
    assert rec["words"] == {"flash_attn": "fallback:sdpa:absent", "rotary": "fallback:torch:absent", "te": "fallback:torch:absent", "xformers": "absent"}
    out = capsys.readouterr().out
    assert rec["stack_token"] == "accel" and "stack=accel " in out and "KERNELS SHORT" in out      # one stack: the line always names the pinned stack


def test_off_by_route_and_xformers_present_words():
    rec = kernels.census(fake.install("engaged", attn_impl="sdpa"))                           # use_flash_attn=False: the sdpa class built while flash_attn imports
    assert rec["words"]["flash_attn"] == "off-by-route:sdpa-class:attn_implementation=sdpa"
    assert rec["words"]["rotary"].startswith("engaged:")                                       # the sdpa class takes the Triton rotary on CUDA too (layers.py:186)
    rec = kernels.census(fake.install("engaged", xformers=True))
    assert rec["words"]["xformers"] == "present:0.0.35"
    assert kernels.differences(rec["words"], kernels.expected(_pins(), "accel")) == [("xformers", "present:0.0.35", "absent")]
    rec = kernels.census(fake.install("engaged", device="cpu"))                                # built off CUDA: upstream binds nothing accelerated (model.py:286, :298)
    assert rec["words"]["flash_attn"] == "fallback:sdpa-class:not-cuda" and rec["words"]["te"] == "fallback:torch:not-cuda" and rec["words"]["rotary"] == "fallback:torch:not-cuda"


def test_rule_words_when_a_regime_names_the_other_class(capsys, bare_host):
    """The rule mechanism of the attention-class table (modes.STOCK_USE_FLASH_ATTN / STOCK_RULE_NAMES; the table itself names the shipped class in
    both regimes, asserted in test_mode_tables_agree): a call that says use_flash_attn=False BY A NAMED RULE must show the non-varlen class bound
    with flash-attention still importable — the word `off-by-rule:<rule>(...)`, expected and observed alike; a call for the shipped class expects
    the varlen class engaged; a caller's own use_flash_attn=False with no rule earns no expectation and is refused."""
    p = _pins()
    fa = p["stacks"]["accel"]["flash_attn"]
    call_b1, rule_b1 = {"device": "cuda", "use_flash_attn": False}, modes.STOCK_RULE_NAMES["b1"]
    exp = kernels.expected(p, "accel", call_b1, rule_b1)
    assert exp["flash_attn"] == f"off-by-rule:stock-b1(use_flash_attn=False;sdpa-class;flash_attn@{fa})" and exp["rotary"].startswith("engaged:") and exp["te"].startswith("engaged:")
    for rg in modes.REGIMES:
        assert kernels.expected(p, "accel", modes.load_call("off", rg), modes.call_rule("off", rg))["flash_attn"] == f"engaged:flash_attn@{fa}"   # the table as it stands: shipped class, both regimes
    assert kernels.expected(p, "accel", {"use_flash_attn": False}, None)["flash_attn"] == f"engaged:flash_attn@{fa}"        # no rule names it: a caller's own switch earns no expectation
    rec = kernels.prove_once(fake.install("engaged", attn_impl="sdpa"), mode="off", via="stock_child", regime="b1", call=call_b1, rule=rule_b1, p=p)   # the b1 stock model: sdpa class, flash importable
    assert rec["verdict"] == "ok" and rec["words"]["flash_attn"] == exp["flash_attn"] and rec["bound"]["attn"] == ["esm.models.esmc.layers.EsmcMultiHeadAttention"]
    m = LINE.match(capsys.readouterr().out.strip().splitlines()[-1])
    assert m and m.group(1, 2, 3, 4, 5, 6) == ("off", "stock", "stock_child", "b1", "accel", exp["flash_attn"])
    kernels._reset_for_tests()
    rec = kernels.prove_once(fake.install("engaged"), mode="off", via="stock_child", regime="b1", call=call_b1, rule=rule_b1, p=p)   # the class binding is wrong for the rule: named
    assert rec["verdict"] == "short" and [(d["accelerator"], d["observed"]) for d in rec["differences"]] == [("flash_attn", f"engaged:flash_attn@{fa}")]
    kernels._reset_for_tests()
    rec = kernels.prove_once(fake.install("fallback"), mode="off", via="stock_child", regime="b1", call=call_b1, rule=rule_b1, p=p)  # an image without flash-attention under the b1 stock call: named
    assert [d["accelerator"] for d in rec["differences"]] == ["flash_attn", "rotary", "te"] and rec["words"]["flash_attn"] == "fallback:sdpa:absent"
    kernels._reset_for_tests()
    rec = kernels.prove_once(fake.install("engaged"), mode="off", via="stock_child", regime="batched", call=modes.load_call("off", "batched"), rule=None, p=p)
    assert rec["verdict"] == "ok" and rec["words"]["flash_attn"] == f"engaged:flash_attn@{fa}" and rec["regime"] == "batched"
    kernels._reset_for_tests()
    rec = kernels.prove_once(fake.install("engaged", attn_impl="sdpa"), mode="exact", via="kit", regime="b1", call={"use_flash_attn": False}, rule=modes.call_rule("exact", "b1"), p=p)   # a kit mode whose caller passed use_flash_attn=False: named
    assert rec["verdict"] == "short" and rec["words"]["flash_attn"] == "off-by-route:sdpa-class:attn_implementation=sdpa"


def test_word_matching_rules():
    assert kernels.word_matches("off-by-rule:stock-b1(use_flash_attn=False;sdpa-class;flash_attn@2.7.4.post1+cu130)", "off-by-rule:stock-b1(use_flash_attn=False;sdpa-class;flash_attn@2.7.4.post1)")
    assert not kernels.word_matches("off-by-rule:stock-b1(use_flash_attn=False;sdpa-class;flash_attn@2.8.3)", "off-by-rule:stock-b1(use_flash_attn=False;sdpa-class;flash_attn@2.7.4.post1)")
    assert kernels.word_matches("engaged:transformer_engine@2.15.0+c47f8a3(LayerNormLinear,Linear)", "engaged:transformer_engine@2.15.0")
    assert not kernels.word_matches("engaged:transformer_engine@2.16.0", "engaged:transformer_engine@2.15.0")
    assert not kernels.word_matches("engaged:xformers@2.15.0", "engaged:transformer_engine@2.15.0")
    assert kernels.word_matches("fallback:sdpa:absent", "fallback:sdpa:absent") and not kernels.word_matches("fallback:sdpa:import-error(x)", "fallback:sdpa:absent")
    assert not kernels.word_matches("unread", "absent")


def test_upstream_warnings_are_captured():
    import logging
    kernels.listen(); kernels.listen()
    logging.getLogger(kernels.UPSTREAM_KERNELS).warning("ESMC: Transformer Engine is not installed; falling back")
    assert any("Transformer Engine" in w for w in kernels.warnings_seen())
    rec = kernels.census(fake.install("engaged"))
    assert rec["upstream_warnings"] and "Transformer Engine" in rec["upstream_warnings"][0]


def test_cli_check_no_stack_flag_and_sdk_model_name():
    env = {k: v for k, v in os.environ.items() if not k.startswith("ESMC_")}
    env.update({"PYTHONPATH": OPT, "PYTHONDONTWRITEBYTECODE": "1"})
    run = lambda args: subprocess.run([sys.executable, "-s", "-m", "esmc_opt"] + args, capture_output=True, text=True, env=env)
    r = run(["check", "--variant", "300m", "--mode", "exact", "--stack", "sdpa"])          # the stack-shift flag is gone: a usage error (rc 2), the pinned stack is the only stack
    assert r.returncode == 2 and "unrecognized arguments: --stack" in r.stderr, (r.stdout, r.stderr[-400:])
    r = run(["check", "--variant", "esmc_300m", "--mode", "off"])                              # the SDK's model name names the size
    assert r.returncode == 0 and "NOT ACTIVE: mode off: stock" in r.stdout and "variant=300m" in r.stdout, (r.stdout, r.stderr[-400:])
