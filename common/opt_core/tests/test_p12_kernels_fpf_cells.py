"""The carried FPF registry tables (fpf_triatt_epi.epilogue, fpf_triatt_pro.prologue, fpf_transition.transition) under P12 / R4: PINNED / VERIFIED keys select
exactly their recorded configuration; the keys certified engines meet with the kernel OFF (candidate cells, pinned-without-record keys, the named sm100/sm103
transition cells) are the stock path BY NAME (`…+off(not-measured)`); any OTHER key is UNKNOWN and is served the capability's SAFE settings
(opt_core.kernels.safe_settings pair_fused rows), engaged and named ONCE; a key no safe row admits keeps its plain miss word.  CPU only: gpu class and cc are named."""
import os
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from opt_core.kernels import safe_settings as SS  # noqa: E402


def _routed(name, sub):
    from opt_core import kernels as CK
    core_copy = os.path.join(os.path.dirname(CK.__file__), name)
    top = sys.modules.get(name)
    if top is not None and not os.path.abspath(getattr(top, "__file__", "") or "").startswith(os.path.abspath(core_copy)):
        for m in [m for m in sys.modules if m == name or m.startswith(name + ".")]:
            del sys.modules[m]
    CK.route(name)
    import importlib
    m = importlib.import_module(name + "." + sub)
    assert os.path.abspath(m.__file__).startswith(os.path.abspath(core_copy)), m.__file__
    return m


@pytest.fixture(scope="module")
def E():
    return _routed("fpf_triatt_epi", "epilogue")


@pytest.fixture(scope="module")
def P():
    return _routed("fpf_triatt_pro", "prologue")


@pytest.fixture(scope="module")
def TR():
    return _routed("fpf_transition", "transition")


@pytest.fixture
def nets(E, P, TR):
    for m in (E, P, TR): m._NET.reset()
    yield
    for m in (E, P, TR): m._NET.reset()
    os.environ.pop("FPF_TRIATT_EPI_ALLOW_UNVERIFIED", None)


# ------------------------------------------------------------------------------------------------------------------------------------------ epilogue
def test_epi_pinned_cell_selects_its_recorded_config(E, nets, capfd):
    assert E.PINNED_CONFIG[(256, 8, 32, "sm90")] == dict(KVER=2, BI=16, BJ=8, num_warps=8, num_stages=1, EXP="libdevice")
    for key, cfg in E.PINNED_CONFIG.items():
        c, H, D, gpu = key
        assert E.cell_verdict(c, H, D, gpu=gpu, cc="9.0") == ("pinned", dict(cfg), "")
        assert E.kernel_eligible(c, H, D, None, gpu=gpu, cc="9.0") and E.pick_config(c, H, D, None, gpu=gpu, cc="9.0") == dict(cfg) and E.stock_word(c, H, D, None, gpu=gpu, cc="9.0") == ""
    assert capfd.readouterr().err == "" and E.settings_word() is None


@pytest.mark.parametrize("key", [(128, 4, 32, "sm90"), (64, 2, 32, "sm90"), (64, 4, 16, "sm90"), (256, 8, 32, "sm80")])
def test_epi_candidate_cells_are_measured_off_by_name(E, nets, key, capfd):
    c, H, D, gpu = key
    cc = {"sm90": "9.0", "sm80": "8.0"}[gpu]
    assert key in E.CANDIDATE_CONFIG_UNVERIFIED and key in E.MEASURED_OFF
    kind, cfg, word = E.cell_verdict(c, H, D, gpu=gpu, cc=cc)
    assert (kind, cfg, word) == ("off", None, f"cell:c{c}_h{H}_d{D}_{gpu}+off(not-measured)")
    assert not E.kernel_eligible(c, H, D, None, gpu=gpu, cc=cc) and E.stock_word(c, H, D, None, gpu=gpu, cc=cc) == word
    with pytest.raises(NotImplementedError) as e:
        E.pick_config(c, H, D, None, gpu=gpu, cc=cc)
    assert word in str(e.value) and capfd.readouterr().err == ""
    os.environ["FPF_TRIATT_EPI_ALLOW_UNVERIFIED"] = "1"                           # the qualification switch still serves candidates
    assert E.cell_verdict(c, H, D, gpu=gpu, cc=cc)[:2] == ("candidate", dict(E.CANDIDATE_CONFIG_UNVERIFIED[key]))


def test_epi_runtime_merged_cell_serves_pick_config_and_stays_off_for_the_registry_gate(E, nets, monkeypatch):
    """A kit merges (256, 8, 32, 'sm100') into PINNED_CONFIG at run time: pick_config serves it (as it always did); kernel_eligible (VERIFIED record) stays False, by name."""
    monkeypatch.setitem(E.PINNED_CONFIG, (256, 8, 32, "sm100"), dict(KVER=2, BI=16, BJ=8, num_warps=8, num_stages=1, EXP="libdevice"))
    assert E.pick_config(256, 8, 32, None, gpu="sm100", cc="10.0") == dict(KVER=2, BI=16, BJ=8, num_warps=8, num_stages=1, EXP="libdevice")
    assert not E.kernel_eligible(256, 8, 32, None, gpu="sm100", cc="10.0") and E.stock_word(256, 8, 32, None, gpu="sm100", cc="10.0") == "cell:c256_h8_d32_sm100+off(not-measured)"


def test_epi_unknown_cell_is_served_the_safe_settings_named_once(E, nets, capfd):
    kind, cfg, word = E.cell_verdict(32, 2, 16, gpu="sm90", cc="9.0")
    assert (kind, word) == ("safe", "no_cell:c32_h2_d16_sm90") and cfg == dict(SS.safe_row("pair_fused:epilogue", "9.0")["settings"])
    assert E.kernel_eligible(32, 2, 16, None, gpu="sm90", cc="9.0") and capfd.readouterr().err == ""
    assert E.pick_config(32, 2, 16, None, gpu="sm90", cc="9.0") == cfg
    err = capfd.readouterr().err
    assert err.startswith("[opt_core/fpf_triatt_epi] safe settings served (no_cell:c32_h2_d16_sm90, cc 9.0, triton ") and err.count("\n") == 1
    E.pick_config(32, 2, 16, None, gpu="sm90", cc="9.0")
    assert capfd.readouterr().err == "" and E.settings_word() == "safe:no_cell:c32_h2_d16_sm90"
    k12 = E.cell_verdict(256, 8, 32, gpu="sm120", cc="12.0")                            # a capability without safe rows (unless the core carries an any-capability row)
    assert k12[0] == ("safe" if SS.safe_row("pair_fused:epilogue", "12.0") else "none")
    assert E.cell_verdict(256, 8, 32, gpu="sm86", cc="8.6")[0] in ("safe", "none")
    n = E.cell_verdict(512, 8, 64, gpu="sm80", cc="8.0")                                 # cc 8.0's safe epilogue rows exclude c_z > 128 by name
    assert n == ("none", None, "cell:c512_h8_d64_sm80+no_safe(c_z_above_128_slower_than_stock_on_cc8.0)")


# ------------------------------------------------------------------------------------------------------------------------------------------ prologue
def test_pro_verified_key_selects_its_recorded_config(P, nets, capfd):
    assert (256, 256, "H100") in P.VERIFIED
    assert P.cell_verdict(256, 256, gpu="H100", cc="9.0") == ("pinned", dict(P.PINNED_CONFIG[(256, 256, "H100")]), "")
    assert P.eligible(256, 256, None, gpu="H100", cc="9.0") and P.pick_config(256, 256, None, gpu="H100", cc="9.0") == dict(BI=8, BJ=16, num_warps=8, num_stages=2)
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize("key,cc", [((64, 64, "H100"), "9.0"), ((128, 128, "H100"), "9.0"), ((256, 256, "A100"), "8.0")])
def test_pro_pinned_keys_without_a_record_are_measured_off_for_the_registry_op(P, nets, key, cc, capfd):
    C, HD, gpu = key
    assert key in P.PINNED_CONFIG and key not in P.VERIFIED
    assert not P.eligible(C, HD, None, gpu=gpu, cc=cc) and P.stock_word(C, HD, None, gpu=gpu, cc=cc) == f"cell:c{C}_hd{HD}_{gpu}+off(not-measured)"
    assert P.pick_config(C, HD, None, gpu=gpu, cc=cc) == dict(P.PINNED_CONFIG[key])     # an explicit caller (cfg by key) is served the pinned settings, as always
    assert capfd.readouterr().err == ""


def test_pro_unknown_key_is_served_the_safe_settings_named_once(P, nets, capfd):
    assert P.cell_verdict(512, 512, gpu="H100", cc="9.0") == ("safe", dict(SS.safe_row("pair_fused:prologue", "9.0")["settings"]), "no_cell:c512_hd512_H100")
    assert P.eligible(512, 512, None, gpu="H100", cc="9.0")
    assert P.pick_config(512, 512, None, gpu="H100", cc="9.0") == {"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}
    assert capfd.readouterr().err.startswith("[opt_core/fpf_triatt_pro] safe settings served (no_cell:c512_hd512_H100, cc 9.0, triton ")
    kind = P.cell_verdict(256, 256, gpu="sm120", cc="12.0")[0]
    assert kind == ("safe" if SS.safe_row("pair_fused:prologue", "12.0") else "none")
    if kind == "none":
        with pytest.raises(KeyError):
            P.pick_config(256, 256, None, gpu="sm120", cc="12.0")


# ------------------------------------------------------------------------------------------------------------------------------------------ transition
def test_transition_pinned_cells_select_their_recorded_config(TR, nets, capfd):
    for key, cfg in TR.PINNED_CONFIGS.items():
        c, nh, gpu = key
        cc = {"sm90": "9.0", "sm80": "8.0"}[gpu]
        assert TR.cell_verdict(c, nh, gpu=gpu, cc=cc) == ("pinned", dict(cfg), "") and TR.pick_config(c, nh, None, gpu=gpu, cc=cc) == dict(cfg)
    assert TR.PINNED_CONFIGS[(256, 1024, "sm90")] == {"BM": 128, "BH": 32, "num_warps": 8, "num_stages": 3, "IL": 0} and capfd.readouterr().err == ""


@pytest.mark.parametrize("key,cc", [((64, 128, "sm90"), "9.0"), ((64, 256, "sm90"), "9.0"), ((128, 256, "sm90"), "9.0"), ((256, 1024, "sm80"), "8.0"), ((384, 1536, "sm80"), "8.0"),
                                    ((128, 512, "sm100"), "10.0"), ((384, 1536, "sm103"), "10.3")])
def test_transition_measured_off_cells_run_the_stock_body_by_name(TR, nets, key, cc, capfd):
    c, nh, gpu = key
    assert key in TR.MEASURED_OFF
    assert TR.cell_verdict(c, nh, gpu=gpu, cc=cc) == ("off", None, f"no-config-{c}-{nh}+off(not-measured)")
    assert TR.pick_config(c, nh, None, gpu=gpu, cc=cc) is None and capfd.readouterr().err == ""


def test_transition_runtime_merged_cell_is_pinned(TR, nets, monkeypatch):
    monkeypatch.setitem(TR.PINNED_CONFIGS, (256, 1024, "sm80"), {"BM": 64, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 1})   # what a kit's cells table does
    assert TR.cell_verdict(256, 1024, gpu="sm80", cc="8.0")[0] == "pinned"


def test_transition_unknown_cell_is_fused_with_the_safe_settings_named_once(TR, nets, capfd):
    assert TR.cell_verdict(96, 384, gpu="sm90", cc="9.0") == ("safe", dict(SS.safe_row("pair_fused:transition", "9.0")["settings"]), "no_cell:96-384-sm90")
    assert TR.pick_config(96, 384, None, gpu="sm90", cc="9.0") == {"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 0}
    assert capfd.readouterr().err.startswith("[opt_core/fpf_transition] safe settings served (no_cell:96-384-sm90, cc 9.0, triton ")
    assert TR.describe()["settings"] == "safe:no_cell:96-384-sm90"
    assert TR.cell_verdict(256, 1024, gpu="sm100", cc="10.0")[0] == "safe"                # B200 pair transition without a merged cell: SAFE settings of 10.0
