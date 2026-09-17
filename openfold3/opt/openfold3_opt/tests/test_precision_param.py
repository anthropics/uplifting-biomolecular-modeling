"""The fast mode is ONE composition: bf16 (modes.FAST_PRECISION) on every route — the CLI's runner yaml carries upstream's
pl_trainer_args.precision: bf16-mixed, and the fast activation patches upstream's ExperimentRunner so the Trainer reads bf16-mixed whatever
the config said (precision.py: silent when it already did, ONE named line when it did not); no precision flag or variable."""
import os
import sys
import types

import pytest

from openfold3_opt import _autoload, cli, modes, precision
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
CLEAN = {k: v for k, v in os.environ.items() if not k.startswith(modes.SWITCH_PREFIXES + ("OPENFOLD3_OPT",))}


def test_fast_is_bf16():
    assert modes.FAST_PRECISION == "bf16" and ("fast", None) not in modes.LINE_PARAMS
    r = modes.resolve("fast", HOME, environ=dict(CLEAN))
    assert r.precision == "bf16" and not r.conflicts
    assert modes.resolve("exact", HOME, environ=dict(CLEAN)).precision is None
    ns = cli.build_parser().parse_args(["pred", "--mode", "fast", "--query-json", "q.json", "--output-dir", "o"])
    assert cli.call_precision(ns, "fast") == "bf16" and cli.call_precision(ns, "exact") is None


def test_no_precision_flag_or_variable():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["pred", "--mode", "fast", "--precision", "bf16", "--query-json", "q.json", "--output-dir", "o"])
    assert "OPENFOLD3_OPT_PRECISION" not in _autoload.DECLARED
    with pytest.raises(SystemExit) as e:                                                # an undeclared variable under the prefix: refused by name, exit 3
        _autoload.install({"OPENFOLD3_OPT": "exact", "OPENFOLD3_OPT_PRECISION": "fp32"})
    assert e.value.code == 3


def test_env_route_fast_arms_the_finder():
    """OPENFOLD3_OPT=fast on the env/.pth route arms the core's finder like every mode (no refusal): the composition's precision is applied at
    activation (precision.install), not through a yaml this route cannot pass."""
    from opt_core.autoload import Finder
    f = _autoload.install({"OPENFOLD3_OPT": "fast"})
    try:
        assert isinstance(f, Finder) and f.mode == "fast" and f.armed
    finally:
        sys.meta_path.remove(f)


def _stub_runner_module(monkeypatch, default="32-true"):
    """A stand-in for openfold3.entry_points.experiment_runner: ExperimentRunner.__init__ keeps experiment_config.pl_trainer_args (upstream 0.4.1's shape)."""
    mod = types.ModuleType(precision.TARGET)

    class PlTrainerArgs:
        def __init__(self, precision=default):
            self.precision = precision

    class ExperimentRunner:
        def __init__(self, experiment_config):
            self.experiment_config = experiment_config
            self.pl_trainer_args = experiment_config.pl_trainer_args

    class InferenceExperimentRunner(ExperimentRunner):
        def __init__(self, experiment_config, extra=None):
            super().__init__(experiment_config)
            self.extra = extra

    mod.PlTrainerArgs, mod.ExperimentRunner, mod.InferenceExperimentRunner = PlTrainerArgs, ExperimentRunner, InferenceExperimentRunner
    monkeypatch.setitem(sys.modules, precision.TARGET, mod)
    return mod


def _fresh_patch():
    from opt_core import autoload as core_autoload
    p = core_autoload._PATCHES.pop((precision.TARGET, precision.ATTR), None)
    if p is not None:
        p.restore() if p.state == "installed" else p.disarm()


def test_fast_precision_wins_where_the_trainer_reads_it(monkeypatch, capsys):
    """A config that says fp32 (upstream's default on the env route, or a caller's yaml): after upstream builds the runner the fast mode's
    precision is in pl_trainer_args, named ONCE on stderr; a subclass's __init__ (upstream's InferenceExperimentRunner) is covered."""
    _fresh_patch()
    mod = _stub_runner_module(monkeypatch)
    try:
        p = precision.install()
        assert p.state == "installed" and precision.PATCH is p
        cfg = types.SimpleNamespace(pl_trainer_args=mod.PlTrainerArgs("32-true"))
        r = mod.InferenceExperimentRunner(cfg, extra=1)
        assert r.pl_trainer_args.precision == "bf16-mixed" == precision.BF16_MIXED and r.extra == 1
        err = capsys.readouterr().err
        assert err.count("\n") == 1 and err == "[openfold3-opt] fast: pl_trainer_args.precision '32-true' -> 'bf16-mixed' (the fast mode's precision)\n", err
    finally:
        _fresh_patch()


def test_fast_precision_is_silent_under_the_composed_yaml(monkeypatch, capsys):
    """The CLI route: the composed runner yaml already carries bf16-mixed — nothing changes, nothing is printed (the run's lines are the yaml route's)."""
    _fresh_patch()
    mod = _stub_runner_module(monkeypatch)
    try:
        precision.install()
        r = mod.ExperimentRunner(types.SimpleNamespace(pl_trainer_args=mod.PlTrainerArgs("bf16-mixed")))
        assert r.pl_trainer_args.precision == "bf16-mixed" and capsys.readouterr().err == ""
    finally:
        _fresh_patch()


def test_fast_precision_names_a_changed_upstream_shape(monkeypatch):
    """An ExperimentRunner without pl_trainer_args.precision after __init__ (upstream changed shape): a named error, never a silent fp32 fast run."""
    _fresh_patch()
    mod = _stub_runner_module(monkeypatch)

    class Bare:
        def __init__(self, experiment_config):
            self.experiment_config = experiment_config
    mod.ExperimentRunner = Bare
    try:
        precision.install()
        with pytest.raises(RuntimeError, match="precision_bf16: openfold3.entry_points.experiment_runner.ExperimentRunner carries no pl_trainer_args.precision"):
            mod.ExperimentRunner(types.SimpleNamespace())
    finally:
        _fresh_patch()


def test_fast_activation_installs_the_precision_patch_and_other_modes_do_not():
    src = open(os.path.join(os.path.dirname(os.path.dirname(__file__)), "stack.py"), encoding="utf-8").read()
    head = src.split("_precision.install()")[0].splitlines()
    assert src.count("_precision.install()") == 1 and any(l.strip().startswith('if mode == "fast":') for l in head[-3:]), head[-3:]
