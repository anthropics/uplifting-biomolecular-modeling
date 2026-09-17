"""The tp line's triangle-kernel levers (``tp_rowpair.pairstack``: tp_triatt = the core's row-block dispatch with the word ``flash_triattn``, tp_trimul = the
core's fused provider ``fpf_v4`` around OpenFold3's ``TriMulFns``, the statements kept as its fallback): on CPU tensors the dispatch refuses the flash kernel
BY NAME unless the core's opt-out ``ROWPAIR_TRIATT_CORE=torch`` is set, under which it runs OpenFold3's torch statement (the output equals the stock call byte
for byte, the fallback counted by reason); the rank census lines and the launcher's parse + exit rule. Needs torch + openfold3 + opt_core with the row-block
kernels (skipped by name otherwise)."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("openfold3.core.model.layers.triangular_attention")
RA = pytest.importorskip("opt_core.mem.rowpair.triatt")
if not hasattr(RA, "attention_core"):
    pytest.skip("opt_core.mem.rowpair.triatt.attention_core absent in this opt_core", allow_module_level=True)

from openfold3_ob0_opt import modes, tp as TP  # noqa: E402
from openfold3_ob0_opt.registry import LEVERS  # noqa: E402
from openfold3_ob0_opt.tp_rowpair import env, pairstack as PS  # noqa: E402

ROWS, N, C, H, D = 3, 20, 16, 4, 8


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    for k in ("ROWPAIR_TRIATT_QBLOCK", "ROWPAIR_TRIATT_CORE", "ROWPAIR_TRIMUL_KERNELS"):
        monkeypatch.delenv(k, raising=False)
    saved_a, saved_m = dict(PS.TRIATT), dict(PS.TRIMUL)
    PS.TRIATT.update({"kernel": None, "bound": 0, "calls": 0}); PS.TRIMUL.update({"kernels": None, "bound": 0})
    yield
    PS.TRIATT.update(saved_a); PS.TRIMUL.update(saved_m)


def _ta(seed=1):
    from openfold3.core.model.layers.triangular_attention import TriangleAttention
    ta = TriangleAttention(c_in=C, c_hidden=D, no_heads=H, starting=True, inf=1e9)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in ta.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return ta.eval()


def _tmu(seed=2, c=C):
    from openfold3.core.model.layers.triangular_multiplicative_update import TriangleMultiplicationOutgoing
    tm = TriangleMultiplicationOutgoing(c_z=c, c_hidden=c)
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in tm.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    return tm.eval()


def _operands(seed=7):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(ROWS, N, C, generator=g)
    mask = (torch.rand(ROWS, N, generator=g) > 0.15).float()
    tb_full = torch.randn(N, N, H, generator=g) * 0.5                    # channel-last, as the core hands it (triangle_bias_rows)
    return x, mask, tb_full


def _attend(core_word, monkeypatch, ta, x, mask, tb_full):
    """Bind and run the tri-att fns under the core's opt-out word (``torch``) or without it (None: the lever's flash kernel as the line runs it)."""
    if core_word is None:
        monkeypatch.delenv("ROWPAIR_TRIATT_CORE", raising=False)
    else:
        monkeypatch.setenv("ROWPAIR_TRIATT_CORE", core_word)
    fns = PS.triatt_fns(ta)
    with torch.no_grad():
        return fns.attend(x.clone(), mask, tb_full, None)


def test_kernel_words_line_profile_and_lever_rows():
    assert PS.triatt_kernel() == PS.TRIATT_KERNEL == env.KERNEL_WORDS["triatt"] == "flash_triattn"      # the line's kernels are constants of the binding
    assert PS.trimul_kernels() == PS.TRIMUL_KERNELS == env.KERNEL_WORDS["trimul"] == "fpf_v4"
    assert PS.TRIATT_KERNEL in RA.CORE_KERNELS                                                            # ONE vocabulary with the core
    from opt_core.mem.rowpair import trimul_fused as RF
    assert PS.TRIMUL_KERNELS in RF.KERNEL_WORDS
    assert not env.known("OF3TP_TRIATT_KERNEL") and not env.known("OF3TP_TRIMUL_KERNELS")                 # no kernel switch on the kit surface
    ln = modes.LINES[("big", "tp")]
    assert not any(k.endswith(("_KERNEL", "_KERNELS")) for k in ln.env), ln.env
    assert ln.levers[-2:] == ("tp_triatt", "tp_trimul")
    assert "tp_triatt" not in modes.LINES[("big", "resident")].levers
    a, m = LEVERS["tp_triatt"], LEVERS["tp_trimul"]
    assert (a.kit, a.tier, a.env_keys, a.modes) == ("tp_rowpair", "tolerance", (), ("big/tp",))
    assert (m.kit, m.tier, m.env_keys, m.modes) == ("tp_rowpair", "tolerance", (), ("big/tp",))


def test_the_opt_out_runs_the_stock_statement_through_the_dispatch(monkeypatch):
    ta = _ta(); x, mask, tb_full = _operands()
    out = _attend("torch", monkeypatch, ta, x, mask, tb_full)                              # ROWPAIR_TRIATT_CORE=torch: the dispatch's torch statement, counted
    mask_bias = (ta.inf * (mask - 1))[..., :, None, None, :]; tb = tb_full.movedim(-1, 0).unsqueeze(0)
    with torch.no_grad():
        ref = ta.mha(q_x=x, kv_x=x, biases=[mask_bias, tb])                                   # the stock call on the row batch
    assert torch.equal(out, ref) and PS.TRIATT == {"kernel": "flash_triattn", "bound": 1, "calls": 1}
    c = PS.triatt_census()
    assert c["served"] == 0 and c["fallback"] == c["calls"] >= 1 and set(c["fallback_by"]) == {"kernel_torch"}, c


def test_flash_word_on_cpu_tensors_is_refused_by_name(monkeypatch):
    """CPU tensors: the carried kernel cannot run in this process -> the core refuses by name (``RowpairRefused``: the lever, the reason, the opt-out
    ``ROWPAIR_TRIATT_CORE=torch`` / ``--mode off``) at bind or at the first call — never a silent torch statement (opt_core's one outcome rule)."""
    from opt_core.mem.rowpair import RowpairRefused
    ta = _ta(); x, mask, tb_full = _operands()
    _attend("torch", monkeypatch, ta, x, mask, tb_full)                                      # under the opt-out it binds and runs on CPU
    with pytest.raises((RowpairRefused, PS.PairstackRefused)) as ei:
        _attend(None, monkeypatch, ta, x, mask, tb_full)                                     # without it: refused by name, never a silent torch statement
    assert "torch" in str(ei.value) or "--mode off" in str(ei.value), str(ei.value)         # the sentence names the opt-out


def test_the_provider_wraps_the_statements(monkeypatch, tmp_path):
    from opt_core.mem.rowpair import trimul_fused as RF
    from openfold3_ob0_opt.cells import pairfused
    monkeypatch.delenv("FPF_TRIMUL_V4_CELLS", raising=False)
    tm = _tmu()
    fns = PS.trimul_fns(tm)
    assert isinstance(fns, RF.FusedTriMulFns) and PS.TRIMUL == {"kernels": "fpf_v4", "bound": 1}
    assert "FPF_TRIMUL_V4_CELLS" not in os.environ                                          # no kit cell table: the kernels' loader reads the core's own
    w = pairfused.trimul_weights(tm)
    assert w["w_ap"] is tm.linear_a_p.weight and w["w_bg"] is tm.linear_b_g.weight and w["w_o"] is tm.linear_z.weight and w["w_og"] is tm.linear_g.weight
    assert w["ln_in_w"] is tm.layer_norm_in.weight and w["b_o"] is tm.linear_z.bias
    z = torch.randn(4, N, C); m = torch.ones(4, N, 1)
    with torch.no_grad():                                                                       # the provider's statements ARE the module's (the fallback path)
        assert torch.equal(fns.proj(z, m, True), (torch.sigmoid(tm.linear_a_g(tm.layer_norm_in(z))) * tm.linear_a_p(tm.layer_norm_in(z)) * m).to(z.dtype))
    line = PS.trimul_census_line()
    assert line.startswith("TRIMUL kernels=fpf_v4 bound=1 served=") and "fallback_by=" in line and "cells=" in line
    rec = _parse(line)
    assert rec["kernels"] == "fpf_v4" and rec["bound"] == 1


def _parse(line):
    kv = dict(tok.split("=", 1) for tok in line.split(" ", 1)[1].split() if "=" in tok)
    return {k: (int(v) if v.isdigit() else v) for k, v in kv.items()}


def test_launcher_parses_the_rank_lines_and_fails_loud(tmp_path):
    log = tmp_path / "rank0.log"
    log.write_text("[tp_rowpair hook r0] armed: world=2 rank=0\n"
                   "[tp_rowpair] TRIATT kernel=flash_triattn bound=52 calls=400 served=400 fallback=0 fallback_by=none\n"
                   "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=0 fallback=96 fallback_by=below_gate:96 cells=None k1_impl=None\n")
    rec = TP.parse_rank_log(str(log))
    assert rec["triatt"] == {"kernel": "flash_triattn", "bound": 52, "calls": 400, "served": 400, "fallback": 0, "fallback_by": "none"}
    assert rec["trimul"] == {"kernels": "fpf_v4", "bound": 104, "served": 0, "fallback": 96, "fallback_by": "below_gate:96", "cells": "None", "k1_impl": "None"}
    kern = {"triatt": "flash_triattn", "trimul": "fpf_v4"}

    def census(lines, kernels=kern):
        out = tmp_path / ("out%d" % abs(hash(lines)))
        (out / "_tp").mkdir(parents=True)
        common = "[rowpair rR] process group ready: backend=gloo world=1 mode=S nccl=n/a\n[rowpair rR] [model] trunk done N=10 P=1 rows 0:10 in 0.1s\n[openfold3_ob0-opt tp rank R] allocator peak: max_allocated_mib=1 max_reserved_mib=2\n"
        (out / "_tp" / "rank0.log").write_text("[tp_rowpair hook r0] armed: world=1 rank=0 mode=S\n" + common.replace("R", "0") + lines)
        procs = [{"rank": 0, "gpu": "0", "rc": 0, "started": 0.0, "ended": 1.0, "log": str(out / "_tp" / "rank0.log"), "dir": str(out)}]
        return TP.census(str(out), procs, 1, "S", "gloo", 64, "test", 1, {}, {"period_s": 1.0, "samples": 0, "peak_mib": {}}, kernels=kernels)

    ok = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=52 calls=400 served=400 fallback=0 fallback_by=none\n"
                "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=0 fallback=96 fallback_by=below_gate:96 cells=None k1_impl=None\n")
    assert ok["reasons"] == [] and ok["triatt_kernel"] == "flash_triattn" and ok["trimul_kernels"] == "fpf_v4"          # below the size gate: served nothing, as declared
    assert ok["triatt"]["0"]["served"] == 400 and ok["trimul"]["0"]["fallback_by"] == "below_gate:96"
    missing = census("[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=3072 fallback=0 fallback_by=none cells=9.0|* k1_impl=pointer\n")
    assert len(missing["reasons"]) == 1 and "lever tp_triatt (flash_triattn) but no `[tp_rowpair] TRIATT` line" in missing["reasons"][0]
    wrong = census("[tp_rowpair] TRIATT kernel=cueq bound=52 calls=400 served=400 fallback=0 fallback_by=none\n"
                   "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=3072 fallback=0 fallback_by=none cells=9.0|* k1_impl=pointer\n")
    assert wrong["reasons"] == ["rank 0: lever tp_triatt (flash_triattn) but the rank bound cueq"]
    err = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=52 calls=400 served=400 fallback=0 fallback_by=none\n"
                 "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=12 fallback=90 fallback_by=error:RuntimeError:1,disabled:89 cells=9.0|* k1_impl=pointer\n")
    assert len(err["reasons"]) == 1 and "lever tp_trimul (fpf_v4) declined units for an undocumented reason: disabled:89,error:RuntimeError:1" in err["reasons"][0] \
        and "ROWPAIR_TRIMUL_KERNELS=torch" in err["reasons"][0] and "--mode off" in err["reasons"][0]
    dead = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=52 calls=400 served=0 fallback=400 fallback_by=unsupported:dtype:400\n"
                  "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=3072 fallback=0 fallback_by=none cells=9.0|* k1_impl=pointer\n")
    assert len(dead["reasons"]) == 1 and "lever tp_triatt (flash_triattn) declined units for an undocumented reason: unsupported:dtype:400" in dead["reasons"][0]
    partial = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=52 calls=400 served=400 fallback=0 fallback_by=none\n"
                     "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=104 served=3000 fallback=96 fallback_by=no_cells:8.0:96 cells=None k1_impl=None\n")
    assert len(partial["reasons"]) == 1 and "undocumented reason: no_cells:8.0:96" in partial["reasons"][0]      # a mode is a contract: a partial decline for an undocumented reason fails the run too
    gates = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=448 calls=9408 served=9408 fallback=0 fallback_by=none\n"
                   "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=536 served=47520 fallback=1760 fallback_by=c=64/64:1760 cells=9.0|* k1_impl=pointer\n")
    assert gates["reasons"] == []                                                              # the template embedder's c=64 pair stack is the documented width gate: counted, not failed
    optout = census("[tp_rowpair] TRIATT kernel=flash_triattn bound=448 calls=1344 served=0 fallback=1344 fallback_by=kernel_torch:1344\n"
                    "[tp_rowpair] TRIMUL kernels=fpf_v4 bound=536 served=0 fallback=480 fallback_by=env_torch:480 cells=None k1_impl=None\n")
    assert optout["reasons"] == []                                                             # the explicit opt-outs (ROWPAIR_*=torch): every unit on the torch statements by request
    quiet = census("", kernels={})                                                             # no kernels declared (a launcher stand-in): no kernel census expected, no kernel reasons
    p = tmp_path / "refused_rank.log"
    p.write_text("Traceback (most recent call last):\n  ...\nopt_core.mem.rowpair.RowpairRefused: F2.trimul_rows (fpf_trimul_v4): cannot run in this process — z is not on a CUDA device; "
                 "run the kit with --mode off, or opt this lever out with ROWPAIR_TRIMUL_KERNELS=torch\n")
    assert TP.parse_rank_log(str(p))["refusal"].startswith("RowpairRefused: F2.trimul_rows (fpf_trimul_v4): cannot run in this process")
    assert quiet["reasons"] == [] and quiet["triatt_kernel"] == "torch" and quiet["trimul_kernels"] == "torch"
