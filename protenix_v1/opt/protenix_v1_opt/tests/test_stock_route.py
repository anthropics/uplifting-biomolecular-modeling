"""The stock route proves its environment: the forbidden prefixes are the pin's (locked equal), kit PYTHONPATH entries and kit modules
are detected, and cli._stock_env strips exactly the package's and the kit's names."""
import json
import os

from .conftest import KIT, PINS
from protenix_v1_opt import cli, stock_pred


def test_forbidden_prefixes_are_the_pins():
    pins = json.load(open(PINS, encoding="utf-8"))
    assert list(stock_pred.FORBIDDEN_PREFIXES) == pins["stock_environment"]["must_be_absent_prefixes"]
    from protenix_v1_opt import stack
    from protenix_v1_opt import _autoload as A
    assert set(stack.PACKAGE_ENV) == set(A.DECLARED) == {"PROTENIX_V1_OPT", "PROTENIX_V1_OPT_DET", "PROTENIX_V1_OPT_ALLOW_PARTIAL", "PROTENIX_V1_OPT_KIT", "PROTENIX_V1_OPT_N_GPU",
                                                        "PROTENIX_V1_OPT_WEIGHTS_MEMO"}
    for k in stack.PACKAGE_ENV:
        assert k.startswith(stock_pred.FORBIDDEN_PREFIXES)


def test_the_child_command_is_the_cores_contract_plus_det(tmp_path):
    from opt_core import stock_proof as SP
    cmd = stock_pred.command("py", proof_json=str(tmp_path / "p.json"), kit_dirs=[KIT], stock_args=["--input", "x.json"], det=True)
    assert cmd[:4] == ["py", "-s", "-m", "protenix_v1_opt.stock_pred"] and cmd[4] == "--det" and cmd[-3:] == ["--", "--input", "x.json"]
    own, stock = SP.split_argv(cmd[4:]); own.remove("--det")
    opts, stock2 = SP.parse_stock_argv(own + ["--"] + stock)
    assert opts.env_absent == list(stock_pred.FORBIDDEN_PREFIXES) and opts.kit_dirs == [KIT] and opts.module_prefixes == list(stock_pred.KIT_MODULES) and stock2 == ["--input", "x.json"]
    assert stock_pred.command("py", proof_json="p", kit_dirs=[KIT], stock_args=[], det=False).count("--det") == 0


def test_the_proof_is_the_cores_and_flags_kit_paths_names_and_modules(monkeypatch, tmp_path):
    from opt_core import stock_proof as SP
    clean = SP.env_proof(env_absent=stock_pred.FORBIDDEN_PREFIXES, kit_dirs=[KIT], module_prefixes=stock_pred.KIT_MODULES,
                         environ={"PATH": ""}, modules={}, path=["/usr/lib/python3"], meta_path=[])
    assert clean["ok"] is True and clean["forbidden_present"] == [] and clean["kit_dirs_on_path"] == [] and clean["kit_modules_loaded"] == []
    import types
    lev = types.ModuleType("levers_ptx1"); lev.__file__ = os.path.join(KIT, "ptxfpf", "levers_ptx1.py")
    bad = SP.env_proof(env_absent=stock_pred.FORBIDDEN_PREFIXES, kit_dirs=[KIT], module_prefixes=stock_pred.KIT_MODULES,
                       environ={"PROTENIX_V1_OPT": "exact", "PTX_TG_MAX": "4"}, modules={"levers_ptx1": lev}, path=[os.path.join(KIT, "ptxfpf")], meta_path=[])
    assert bad["ok"] is False and bad["forbidden_present"] == ["PROTENIX_V1_OPT", "PTX_TG_MAX"]
    assert bad["kit_dirs_on_path"] == [os.path.join(KIT, "ptxfpf")] and "levers_ptx1" in bad["kit_modules_loaded"]


def test_kit_modules_list_names_every_carried_top_level_module():
    """KIT_MODULES travels to the child as --module-prefixes: every importable top-level name under the kit's sys.path roots is listed."""
    from protenix_v1_opt import kit as K
    names = set()
    for rel in K.KIT_SYS_PATHS:
        d = os.path.join(KIT, rel)
        for e in os.listdir(d):
            if e.endswith(".py") and e != "__init__.py":
                names.add(e[:-3])
            elif os.path.isfile(os.path.join(d, e, "__init__.py")):
                names.add(e)
    missing = sorted(n for n in names if not n.startswith(stock_pred.KIT_MODULES) and n not in ("sitecustomize",))
    assert missing == [], missing


def test_stock_env_strips_package_and_kit_names(monkeypatch):
    monkeypatch.setenv("PROTENIX_V1_OPT", "off")
    monkeypatch.setenv("PTX_TG_POOL", "shared")
    monkeypatch.setenv("FPF_TRIMUL_MODE", "x")
    monkeypatch.setenv("PROTENIX_ROOT_DIR", "/srv/weights/protenix_v1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([os.path.join(KIT, "lib"), "/opt/other"]))
    env, stripped, removed = cli._stock_env()
    assert stripped == ["FPF_TRIMUL_MODE", "PROTENIX_V1_OPT", "PTX_TG_POOL"]
    assert env["PROTENIX_ROOT_DIR"] == "/srv/weights/protenix_v1" and env["PYTHONPATH"] == "/opt/other" and removed == [os.path.join(KIT, "lib")]
