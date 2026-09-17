"""Package lever `autotune_cache`: the wrapper puts TRITON_CACHE_AUTOTUNING=1 into the model process's environment when the mode carries the
lever and not when it is named in MODEL_OPT_LEVERS_OFF; the LEVER line carries the model process's census (knob / hits / benched); the
in-process switch returns a named step-aside instead of raising when Triton (or its autotune cache) is absent; the lever is in every mode but
off and composes with --n_gpu > 1 (ROWPAIR_GATES)."""
import os

from af3_torch_opt import cli, modes, registry
from af3_torch_opt import forward as F

from .test_cli_manifest import _inputs


def _lever_lines(err):
    return {ln.split("name=")[1].split()[0]: ln for ln in err.splitlines() if " LEVER name=" in ln}


def test_modes_carry_the_lever():
    for m in ("exact", "fast", "big"):
        assert "autotune_cache" in modes.resolve(m)["package_levers"], m
    assert "autotune_cache" not in modes.resolve("off")["package_levers"]
    assert registry.EVIDENCE["autotune_cache"] == "package" and registry.STRATEGY["autotune_cache"] == "F3.jit_cache_keyed"
    assert dict(F.ROWPAIR_GATES)["autotune_cache"].startswith("composes:") and "autotune_cache" not in F.PACKAGE_LEVERS_SKIPPED_SHARDED


def test_env_reaches_the_model_process_and_ablates_by_name(box, capsys, monkeypatch):
    out = box["tmp"] / "out"
    assert cli.main(["pred", "--mode", "exact", "--json_path", _inputs(box["tmp"])[0], "--output_dir", str(out)]) == 0
    M = cli.last_run(); err = capsys.readouterr().err
    assert M["reports"]["forward"]["autotune_cache"]["env"] == "1"                      # the stub's model process saw TRITON_CACHE_AUTOTUNING=1
    ln = _lever_lines(err)["autotune_cache"]
    assert " state=on " in ln and " knob=1 " in ln and " hits=0 " in ln and " benched=0 " in ln, ln
    monkeypatch.setenv("MODEL_OPT_LEVERS_OFF", "autotune_cache", "feat_par")
    out2 = box["tmp"] / "out2"
    assert cli.main(["pred", "--mode", "exact", "--json_path", _inputs(box["tmp"])[0], "--output_dir", str(out2)]) == 0
    M = cli.last_run(); err = capsys.readouterr().err
    assert "autotune_cache" not in M["activation"]["package_levers"] and M["reports"]["forward"]["autotune_cache"]["env"] == "unset"
    ln = _lever_lines(err)["autotune_cache"]
    assert " state=off " in ln and "reason=" in ln, ln


def test_in_process_switch_names_its_step_aside_or_counts():
    keep = os.environ.get("TRITON_CACHE_AUTOTUNING")
    try:
        r = F._autotune_cache_on()
    finally:
        if keep is None: os.environ.pop("TRITON_CACHE_AUTOTUNING", None)
        else: os.environ["TRITON_CACHE_AUTOTUNING"] = keep
    assert set(r) >= {"knob"}
    if r["knob"]:                                             # a venv with Triton: the census dict the LEVER line prints
        assert r["census"]["hits"] == 0 and r["census"]["benched"] == 0 and r["census"]["knob"] == 1
    else:                                                     # no Triton here (the package venv): named, never raised
        assert r["reason"].startswith("triton_knob_absent:")


def test_release_sources_carry_the_lever():
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    assert "`autotune_cache`" in open(os.path.join(here, "README.md"), encoding="utf-8").read() and "`autotune_cache`" in open(os.path.join(here, "CHANGES.md"), encoding="utf-8").read()
