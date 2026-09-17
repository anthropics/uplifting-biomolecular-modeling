"""The shipped configuration (modes.SHIPPED_YAML): upstream's own default — a caller's explicit --runner-yaml replaces the row's own, and a
row run without one falls back to the stock configuration (modes.STOCK_YAML), which is the shipped configuration plus the kernel lines."""
import os

import yaml

from openfold3_opt import cli, modes
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
CUEQ = "use_cueq_triangle_kernels"
FLAG = "use_deepspeed_evo_attention"


def _load(rel):
    with open(os.path.join(HOME, rel), encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_pred_off_takes_the_shipped_yaml_as_the_runner_yaml():
    argv = cli.predict_argv("/w.pt", "q.json", "out", os.path.join(HOME, modes.SHIPPED_YAML))
    assert argv[argv.index("--runner-yaml") + 1] == os.path.join(HOME, modes.SHIPPED_YAML)   # the caller's yaml is the run's (cli.predict_argv)
    assert cli.row_yaml(HOME, mode="off") == os.path.join(HOME, modes.STOCK_YAML)   # without it the stock arm runs the stock configuration


def test_the_stock_yamls_are_the_shipped_configuration_plus_the_kernel_lines():
    """stock = the predict preset as shipped + the cuEquivariance triangle kernels; under the det recipe the DS4Sci attention off as well; the
    kernels-off configuration turns the DS4Sci attention off and leaves cuEquivariance off."""
    shipped = _load(modes.SHIPPED_YAML)
    for yml, ev_want in ((modes.STOCK_YAML, {CUEQ: True}), (modes.STOCK_DET_YAML, {FLAG: False, CUEQ: True}), (modes.KERNELS_OFF_YAML, {FLAG: False})):
        doc = _load(yml)
        assert doc["model_update"]["custom"]["settings"]["memory"]["eval"] == ev_want, yml
        assert {"model_update": {k: v for k, v in doc["model_update"].items() if k != "custom"}} == shipped, yml   # every other line is the shipped configuration's
