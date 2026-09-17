"""The RMSNorm autotune pin on a (GPU class, size, first-call shape) with NO censused cell: the same lever at the size's DEFAULT pin of the
card's compute-capability class (pins.default_w_row: cc 9.0 → the H100 row, cc 8.0 → the A100 row, else pins.DEFAULT_W_ROW) with ONE named
line, exit 0 — never a refusal; every censused path unchanged (the cell's own pin, no line)."""
import sys, types
import pytest
from forward.engines.e1.kits import pins


class _Cfg:
    def __init__(self, kwargs, num_warps): self.kwargs, self.num_warps = kwargs, num_warps


class _Autotuner:
    def __init__(self): self.configs = [_Cfg({}, w) for w in (1, 2, 4, 8, 16, 32)]; self.cache = {}


@pytest.fixture
def fake_triton(monkeypatch):
    """A host without triton/torch: a stand-in `triton.Config` and the hub kernel's autotuner object (configs list + empty cache)."""
    mod = types.ModuleType("triton"); mod.Config = _Cfg
    monkeypatch.setitem(sys.modules, "triton", mod)
    k = _Autotuner(); monkeypatch.setattr(pins, "rmsnorm_autotuner", lambda: k)
    return k


def test_uncensused_cell_is_served_the_default_pin_with_one_line(fake_triton, capsys):
    with pytest.raises(pins.PinUncensused):                                  # w_cell itself still names the missing cell (a PinDrift subclass)
        pins.w_cell("600m", (1, 202), "NVIDIA A10G")
    assert issubclass(pins.PinUncensused, pins.PinDrift)
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA A10G")            # an uncensused class on a host with no device (cc unknown): NOT refused, the DEFAULT_W_ROW pin
    W_default = int(pins.TRITON_AUTOTUNE_PIN[pins.DEFAULT_W_ROW]["600m"])
    assert rec["num_warps"] == W_default and rec["pinned"] is True and rec["uncensused"] is True and rec["cell"] is None and rec["outcome_set"] == [] and rec["default_row"] == pins.DEFAULT_W_ROW
    assert [c.num_warps for c in fake_triton.configs] == [W_default]            # the hub kernel IS pinned (one config): the kit's kernels stay W-consistent with this stock
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].startswith("[e1/pins] autotune pin: no censused cell for 'NVIDIA A10G' / 600m at first-call shape [1, 202] — pinned to W=%d (the default row's pin, row %r)" % (W_default, pins.DEFAULT_W_ROW)), out
    rec2 = pins.apply_autotune_pin("600m", gpu_name="NVIDIA A10G", num_warps=4)   # an explicit W on an uncensused class: taken as given, the line says so
    assert rec2["num_warps"] == 4 and "the caller's num_warps=" in capsys.readouterr().out


def test_censused_cell_path_is_unchanged_and_silent(fake_triton, capsys):
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA H100 80GB HBM3")   # the H100 (1, 202) cell: its own pin, no line
    assert rec["num_warps"] == 8 and rec["uncensused"] is False and rec["cell"]["gpu_key"] == "NVIDIA H100 80GB HBM3" and rec["cell"]["first_call_shape"] == [1, 202]
    assert capsys.readouterr().out == ""
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA A100-SXM4-80GB")   # the A100 cell (num_warps 4), PCIe spelling via the class slug alike
    assert (rec["num_warps"], rec["uncensused"]) == (4, False)
    assert pins.apply_autotune_pin("600m", gpu_name="NVIDIA A100 80GB PCIe")["num_warps"] == 4
    with pytest.raises(pins.PinDrift):                                        # a W outside a censused cell's outcome set: still refused by name
        pins.apply_autotune_pin("600m", gpu_name="NVIDIA H100 80GB HBM3", num_warps=3)
    assert capsys.readouterr().out == ""


def test_the_default_row_is_the_compute_capability_classs(fake_triton, capsys, monkeypatch):
    """An uncensused card is served its COMPUTE-CAPABILITY CLASS's default row (never a device-name key): cc 8.0 → the A100 row's pins
    (600m: W=4), cc 9.0 → the H100 row's (W=8), any other cc / no device → DEFAULT_W_ROW; the one named line names the row served."""
    assert pins.default_w_row("8.0") == "NVIDIA A100-SXM4-80GB" and pins.default_w_row("9.0") == "NVIDIA H100 80GB HBM3"
    assert pins.default_w_row("8.6") == pins.DEFAULT_W_ROW == pins.default_w_row("12.0")
    monkeypatch.setattr(pins, "_cc_now", lambda: None); assert pins.default_w_row() == pins.DEFAULT_W_ROW          # a host that cannot say
    monkeypatch.setattr(pins, "_cc_now", lambda: "8.0"); assert pins.default_w_row() == "NVIDIA A100-SXM4-80GB"   # device 0's capability read when the caller passes none
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA A30", cc="8.0")     # an uncensused cc-8.0 card: the A100 row's 600m pin
    assert (rec["num_warps"], rec["uncensused"], rec["default_row"]) == (4, True, "NVIDIA A100-SXM4-80GB")
    assert [c.num_warps for c in fake_triton.configs] == [4]
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "no censused cell for 'NVIDIA A30' / 600m" in out[0] and "pinned to W=4 (the default row's pin, row 'NVIDIA A100-SXM4-80GB')" in out[0], out
    fake_triton.configs = [_Cfg({}, w) for w in (1, 2, 4, 8, 16, 32)]
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA H100 PCIe-ish", cc="9.0")   # an uncensused cc-9.0 name: the H100 row's pin (8) — the name keys only measured cells
    assert (rec["num_warps"], rec["default_row"]) == (8, "NVIDIA H100 80GB HBM3"); capsys.readouterr()
    fake_triton.configs = [_Cfg({}, w) for w in (1, 2, 4, 8, 16, 32)]
    rec = pins.apply_autotune_pin("600m", gpu_name="NVIDIA A10G")               # device 0 reads cc 8.0 (patched above) and the caller passed none: the A100 row
    assert (rec["num_warps"], rec["default_row"]) == (4, "NVIDIA A100-SXM4-80GB"); capsys.readouterr()
    for name in ("NVIDIA H100 80GB HBM3", "NVIDIA A100-SXM4-80GB", "NVIDIA A100 80GB PCIe"):        # censused cells: unchanged and silent whatever cc says
        fake_triton.configs = [_Cfg({}, w) for w in (1, 2, 4, 8, 16, 32)]
        r = pins.apply_autotune_pin("600m", gpu_name=name, cc="8.0"); assert r["uncensused"] is False and r["default_row"] is None
    assert capsys.readouterr().out == ""
