"""The shipped configuration (modes.SHIPPED_YAML): upstream's own default — a caller's explicit --runner-yaml replaces the row's own, and a
row run without one falls back to the stock configuration (modes.STOCK_YAML), which is the shipped configuration plus the kernel lines."""
import os

import yaml

from openfold3_ob0_opt import cli, modes
from openfold3_ob0_opt.tests import _stubs

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


def test_the_route_yamls_are_the_shipped_configuration_plus_their_kernel_lines():
    """Every configuration a route names = the predict preset as shipped + its explicit eval-kernel triple and trainer precision (+ the pinned chunking where
    the name says so): stock (cuEquivariance, bf16-mixed, pinned; = the det file), kernels-off (32-true, tuner as shipped);
    the shipped yaml is the preset and nothing else (the `default` arm)."""
    assert _load(modes.SHIPPED_YAML) == {"model_update": {"presets": ["predict"]}}
    assert modes.STOCK_DET_YAML == modes.STOCK_YAML
    expect = {modes.STOCK_YAML: ((True, True, False), "bf16-mixed", True), modes.KERNELS_OFF_YAML: ((False, False, False), "32-true", False)}
    for rel, (triple, precision, pinned) in expect.items():
        doc = _load(rel)
        assert doc["model_update"]["presets"] == ["predict"], rel
        ev = doc["model_update"]["custom"]["settings"]["memory"]["eval"]
        flags = {k: v for k, v in ev.items() if k.startswith("use_")}
        assert set(flags) == set(cli.KERNEL_FLAG_DEFAULTS) and all(isinstance(v, bool) for v in flags.values()), rel      # the triple stated explicitly, never left to a default
        assert (ev["use_triton_triangle_kernels"], ev["use_cueq_triangle_kernels"], ev["use_deepspeed_evo_attention"]) == triple, rel
        assert doc["pl_trainer_args"]["precision"] == precision, rel
        if pinned:
            assert ev["tune_chunk_size"] is False and ev["chunk_size"] == modes.CHUNK_SIZE_PINNED, rel
        else:
            assert "tune_chunk_size" not in ev and "chunk_size" not in ev, rel                                              # the tuner as shipped
    assert cli.yaml_kernels_on(os.path.join(HOME, modes.KERNELS_OFF_YAML)) == ()
