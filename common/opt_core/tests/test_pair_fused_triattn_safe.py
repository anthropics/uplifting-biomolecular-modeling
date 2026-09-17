"""attn.pair_fused tri-attention prologue / epilogue on the core's safe-settings mechanism: on a capability the table does not know (no row and no
named_off entry) the SAFE rows serve every UNKNOWN key (engaged, ONE line at the first call) within their measured bounds — the cc 8.0 epilogue only up to c_z 128 (H<=4,
or the H<=8 alternative); above that NO safe row: the no-cell word names why and those calls take the kit's per-call stock route (counted,
exit 0) — and run_cell puts each launch under the net (case 4 = Refused only when the safe settings themselves cannot build).  CPU only."""
import copy

import pytest

from opt_core.attn import pair_fused as PF
from opt_core.kernels import safe_settings as SS

A100, H100 = ("8.0", "3.6"), ("9.0", "3.3")
PRO_SAFE = {"BI": 8, "BJ": 8, "num_warps": 4, "num_stages": 2}
EPI_SAFE = {"KVER": 2, "BI": 8, "BJ": 4, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}
EPI_SAFE_H8 = {"KVER": 2, "BI": 8, "BJ": 8, "num_warps": 8, "num_stages": 1, "EXP": "libdevice"}
EPI_NO_SAFE = "c_z_above_128_slower_than_stock_on_cc8.0"


@pytest.fixture
def clean():
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()
    yield
    for n in PF._NETS.values(): n.reset()
    PF.SERVED_KEYS.clear()


@pytest.fixture
def no_80_rows(monkeypatch):
    tab = copy.deepcopy(PF.cells()); tab["rows"] = [r for r in tab["rows"] if r["cc"] != "8.0"]
    tab["named_off"] = [e for e in tab.get("named_off", []) if e.get("cc") != "8.0"]          # the capability's measured-off entries go with its rows: an unknown card
    monkeypatch.setattr(PF, "cells", lambda path=None: tab)


def test_safe_rows_data():
    assert SS.safe_row("pair_fused:prologue", "8.0")["settings"] == PRO_SAFE == SS.safe_row("pair_fused:prologue", "9.0")["settings"]
    assert SS.safe_row("pair_fused:epilogue", "9.0")["settings"] == {"KVER": 2, "BI": 16, "BJ": 8, "num_warps": 4, "num_stages": 1, "EXP": "libdevice"}
    row, why = SS.safe_settings_for("pair_fused:epilogue", "8.0", {"c_z": 128, "H": 4, "D": 32}); assert row["settings"] == EPI_SAFE and why is None
    row, why = SS.safe_settings_for("pair_fused:epilogue", "8.0", {"c_z": 64, "H": 4, "D": 16}); assert row["settings"] == EPI_SAFE
    row, why = SS.safe_settings_for("pair_fused:epilogue", "8.0", {"c_z": 128, "H": 8, "D": 32}); assert row["settings"] == EPI_SAFE_H8 and why is None   # keyed by H: the alternative
    row, why = SS.safe_settings_for("pair_fused:epilogue", "8.0", {"c_z": 256, "H": 8, "D": 32}); assert row is None and why == EPI_NO_SAFE            # bounded out: no safe row
    row, why = SS.safe_settings_for("pair_fused:epilogue", "8.0", None); assert row is None and why == EPI_NO_SAFE                                        # a bound with no shape given does not apply
    row, why = SS.safe_settings_for("pair_fused:epilogue", "9.0", {"c_z": 256, "H": 8, "D": 32}); assert row is not None and why is None              # no bound on 9.0
    row, why = SS.safe_settings_for("pair_fused:epilogue", "7.0", {"c_z": 256, "H": 8, "D": 32}); assert row is SS.SAFE_ROWS["pair_fused:epilogue"]["*"] and why is None   # any other capability: the '*' row, no shape bound
    assert SS.safe_settings_for("pair_fused:transition", "8.0", {"c_z": 384, "n_hidden": 1536})[0]["settings"]["BM"] == 16                            # unbounded rows serve every shape


def test_no_rows_capability_serves_prologue_and_bounded_epilogue(no_80_rows, capfd, clean):
    dp = PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=A100)
    assert dp.safe and dp.row["cfg"] == PRO_SAFE and dp.row["variant"] == "v3" and dp.reason == "no_cell:128x4x32"
    assert PF.lookup_cell("fpf", "prologue", (256, 8, 32), stack=A100).row["cfg"] == PRO_SAFE                 # prologue: every swept shape
    assert not PF.lookup_cell("fpf", "prologue", (128, 4, 32), stack=A100, variant="prologue_v4").served       # another kernel variant: no safe cell
    de = PF.lookup_cell("fpf", "epilogue", (128, 4, 32), stack=A100)
    assert de.safe and de.row["cfg"] == EPI_SAFE and de.row["variant"] == "v2"
    assert PF.lookup_cell("fpf", "epilogue", (128, 8, 32), stack=A100).row["cfg"] == EPI_SAFE_H8
    dr = PF.lookup_cell("fpf", "epilogue", (256, 8, 32), stack=A100)                                        # c_z 256 on cc 8.0: no safe row — the no-cell word names why; those calls go to stock per call
    assert not dr.served and dr.reason == f"no-cell:fpf:epilogue:256x8x32:8.0|3.6+no_safe({EPI_NO_SAFE})" and dr.word() == "refused:" + dr.reason
    assert capfd.readouterr().err == ""
    with pytest.raises(PF.Unsupported) as e:                                                                  # the kit's per-call stock route catches exactly this
        PF.find_cell("fpf", "epilogue", (256, 8, 32), None, stack=A100)
    assert e.value.reason == dr.reason
    PF.find_cell("fpf", "prologue", (128, 4, 32), None, stack=A100)
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:prologue] safe settings served (no_cell:128x4x32, cc 8.0, triton 3.6)"
    PF.find_cell("fpf", "epilogue", (128, 4, 32), None, stack=A100)
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:epilogue] safe settings served (no_cell:128x4x32, cc 8.0, triton 3.6)"
    assert PF.evidence_tail()["settings"] == "safe:no_cell:128x4x32;safe:no_cell:128x4x32"                 # one word per engaged net (epilogue, prologue)


def test_pinned_capability_is_unchanged(capfd, clean):
    for piece, key in (("prologue", (128, 4, 32)), ("epilogue", (128, 4, 32))):
        d = PF.lookup_cell("fpf", piece, key, stack=H100)
        assert d.served and not d.safe and d.served_by.startswith("9.0|")
    d = PF.lookup_cell("fpf", "prologue", (64, 4, 16), stack=H100)                                             # the template pair stack's key: a certified 9.0 row (never the safe settings)
    assert d.served and not d.safe and d.served_by.startswith("9.0|") and d.row["id"] == "r09"
    d = PF.lookup_cell("fpf", "prologue", (64, 4, 32), stack=H100)                                             # the template pairformer's c=64, 4 x 32 key: certified on 9.0 (r11)
    assert d.served and not d.safe and d.row["id"] == "r11"
    assert not PF.lookup_cell("fpf", "prologue", (64, 4, 64), stack=H100).served                              # candidate rows only on 9.0: the refusal word (SAFE_SCOPE capability), as today
    assert capfd.readouterr().err == ""


class _Build(RuntimeError):
    pass


def test_run_cell_nets_for_the_triattn_pieces(capfd, clean):
    seen = []
    def build(cfg):
        seen.append(dict(cfg))
        if cfg.get("BJ") == 16:
            raise _Build("PassManager::run failed")
        return cfg
    tuned = {"BI": 8, "BJ": 16, "num_warps": 8, "num_stages": 2}
    assert PF.run_cell("pair_fused:prologue", "8.0", "3.6", build, tuned, dims={"c_z": 256, "H": 8, "D": 32}) == PRO_SAFE
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:prologue] safe settings served (build_failed:_Build, cc 8.0, triton 3.6)"
    etuned = {"KVER": 2, "BI": 16, "BJ": 16, "num_warps": 8, "num_stages": 1, "EXP": "libdevice"}
    assert PF.run_cell("pair_fused:epilogue", "8.0", "3.6", build, etuned, dims={"c_z": 128, "H": 8, "D": 32}) == EPI_SAFE_H8   # the H-keyed alternative
    capfd.readouterr()
    PF.safety_net("pair_fused:epilogue").reset()
    with pytest.raises(PF.Unsupported) as e:                                                                  # c_z 256 on 8.0: a tuned cell failing to build has no safe row — a shape miss: the per-call stock route, by name
        PF.run_cell("pair_fused:epilogue", "8.0", "3.6", build, etuned, dims={"c_z": 256, "H": 8, "D": 32})
    assert e.value.reason == f"no-cell:pair_fused:epilogue:build_failed:_Build+no_safe({EPI_NO_SAFE})" and capfd.readouterr().err == ""
    def never(cfg):
        raise _Build("ptxas fatal")
    PF.safety_net("pair_fused:prologue").reset()
    with pytest.raises(PF.Refused) as e4:                                                                     # case 4: the safe cell cannot build either — the lever cannot run at all; the message names --mode off
        PF.run_cell("pair_fused:prologue", "8.0", "3.6", never, tuned, dims={"c_z": 128, "H": 4, "D": 32})
    assert "pair_fused:prologue" in str(e4.value) and "--mode off" in str(e4.value)


def test_safe_settings_are_scoped_to_the_unknown_cell_pinned_cells_keep_theirs(capfd, clean):
    """Two cells of ONE lever in ONE process (the request behind this test: a DiT-width transition (768, 3072) met on cc 9.0 used to switch the
    whole `pair_fused:transition` lever to its safe tiles, demoting the pinned (128, 512) pair transition for the rest of the process).  Now:
    an unknown key engages the safe settings FOR ITSELF (ONE line naming it); the pinned key's launches keep receiving exactly the table's cfg
    (so their bytes and speed are unchanged); a second unknown key engages for itself (its own line); only a BUILD failure is process-wide.
    ((768, 3072) itself is now listed measured-off on cc 9.0 -- the statements are x2-3 faster there -- so two other unlisted widths stand in.)"""
    H = ("9.0", "3.7")
    pinned = PF.lookup_cell("fpf", "transition", (128, 512), stack=H)
    assert pinned.served and not pinned.safe and pinned.served_by.startswith("9.0|")
    pinned_cfg = dict(pinned.row["cfg"])
    unknown = PF.lookup_cell("fpf", "transition", (768, 3072), stack=H)                         # no row, no named_off entry on 9.0: the capability's safe row admits it
    safe_cfg = dict(SS.safe_row("pair_fused:transition", "9.0")["settings"])
    assert unknown.safe and unknown.served_by == "safe" and unknown.row["cfg"] == safe_cfg and unknown.reason == "no_cell:768x3072"
    PF.find_cell("fpf", "transition", (768, 3072), None, stack=H)                                # the serving path's engagement of the unknown key
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:transition] safe settings served (no_cell:768x3072, cc 9.0, triton 3.7)"
    net = PF.safety_net("pair_fused:transition")
    assert net.on and not net.wide and net.cells == {"768x3072": "no_cell:768x3072"} and net.word() == "safe:no_cell:768x3072"
    seen = []
    def build(cfg):
        seen.append(dict(cfg)); return dict(cfg)
    dims128 = {"c_z": 128, "n_hidden": 512}; dims768 = {"c_z": 768, "n_hidden": 3072}
    # the pinned cell launches with ITS cfg after the unknown cell engaged (before: the safe tiles for the rest of the process)
    assert PF.run_cell("pair_fused:transition", "9.0", "3.7", build, pinned_cfg, dims=dims128, key=(128, 512)) == pinned_cfg
    assert PF.run_cell("pair_fused:transition", "9.0", "3.7", build, pinned_cfg, dims=dims128) == pinned_cfg                       # also for a caller that names no key
    # the unknown cell launches the safe settings (its decision row IS the safe row; naming the key skips straight to them)
    assert PF.run_cell("pair_fused:transition", "9.0", "3.7", build, dict(unknown.row["cfg"]), dims=dims768, key=(768, 3072)) == safe_cfg
    assert seen == [pinned_cfg, pinned_cfg, safe_cfg] and capfd.readouterr().err == ""                                             # no second line
    # a second unknown key: its own engagement and line; the pinned cell still keeps its cfg
    PF.find_cell("fpf", "transition", (1024, 2048), None, stack=H)
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:transition] safe settings served (no_cell:1024x2048, cc 9.0, triton 3.7)"
    assert set(net.cells) == {"768x3072", "1024x2048"} and not net.wide
    assert PF.run_cell("pair_fused:transition", "9.0", "3.7", build, pinned_cfg, dims=dims128, key=(128, 512)) == pinned_cfg
    # only a BUILD failure of a tuned cell switches the whole process (unchanged): then every cell of the lever runs the safe settings
    def flaky(cfg):
        if cfg == pinned_cfg:
            raise _Build("PassManager::run failed")
        return dict(cfg)
    assert PF.run_cell("pair_fused:transition", "9.0", "3.7", flaky, pinned_cfg, dims=dims128, key=(128, 512)) == safe_cfg
    assert capfd.readouterr().err.strip() == "[opt_core/pair_fused:transition] safe settings served (build_failed:_Build, cc 9.0, triton 3.7)"
    assert net.wide and PF.run_cell("pair_fused:transition", "9.0", "3.7", build, pinned_cfg, dims=dims128, key=(128, 512)) == safe_cfg
