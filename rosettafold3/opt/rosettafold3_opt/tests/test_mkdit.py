"""The mkdit lever's declarations and its adapter on the CPU (the megakernel itself runs on a GPU only): the forward loops the
single-sample kernel over every diffusion-batch sample (no self-disable at batch > 1), leaves the atom transformers and below-gate calls to
the class's own forward COUNTED, and refuses by name — at enable when a prerequisite is absent (pristine tree, RF3_HOIST off, dtk / dattn
installed, a carried kernel file absent), inside a served call when the input is not on a CUDA device (no silent stock branch)."""
import hashlib
import os
import inspect
import sys
import types

import pytest
import torch

from .. import big, dtk, fold, mkdit, modes, registry, report, stack


def test_mkdit_declarations():
    assert mkdit.MIN_I == 400 == mkdit.make_gate().min_tokens and mkdit.MIN_CAPABILITY == (8, 0) == min(mkdit.CONFIGS)
    assert mkdit.ADDON == registry.MKDIT == "forward/rf3_mk_dit_addon" and mkdit.KERNEL_FILES == ("mkdit/mk2.py", "mkdit/mkrf3.py") and mkdit.KERNEL == "mk2"
    assert mkdit.CONFIG == {"BMR": 64, "BN": 128, "BK": 64, "BNK": 64, "num_warps": 8, "num_stages": 3, "bms": 16}     # one tile configuration per card (the kernel's numerics are config-specific)
    assert set(mkdit.CONFIGS) == {(8, 0), (9, 0)} and mkdit.CONFIGS[(9, 0)] == mkdit.CONFIG                                 # keyed by compute capability: a card takes the row of the highest key at or below its own
    assert mkdit.CONFIGS[(8, 0)] == {"BMR": 64, "BN": 128, "BK": 64, "BNK": 64, "num_warps": 8, "num_stages": 3, "bms": 16}  # sm_80 (A100): the same tiles (fit 163 KB; the fastest sm_80 candidate measured)
    assert all(set(c) == set(mkdit.CONFIG) for c in mkdit.CONFIGS.values())
    assert mkdit.config_for((9, 0)) == mkdit.config_for((10, 0)) == mkdit.config_for((12, 0)) == mkdit.CONFIGS[(9, 0)]      # H100, B200 and up: the sm_90 row
    assert mkdit.config_for((8, 0)) == mkdit.config_for((8, 6)) == mkdit.config_for((8, 9)) == mkdit.CONFIGS[(8, 0)]        # A100 (and sm_86 / sm_89 cards): the sm_80 row
    assert mkdit.config_for((7, 5)) is None and mkdit.config_for(None) is None                                              # below 8.0: no row (enable refuses by name)
    lv = registry.LEVERS["mkdit"]
    assert lv.kit == registry.PACKAGE and lv.kit_files == ("mkdit.py",) and lv.probe == ("mkdit", "served") and lv.switch is None and lv.component is None
    assert lv.strategy == "LOCAL.dit_fused_kernels" and registry.IMPL_OF["mkdit"] == ("forward/rf3_mk_dit_addon/mkdit/mk2.py", "kit")   # a canonical id of the core's STRATEGIES.json (the LEVER line refuses any other)
    assert "mkdit" in modes.KIT_LEVERS and "mkdit" not in modes.KIT_LEVERS_ON_FPF_SEAM and "mkdit" not in stack.KIT_LEVER_KERNELS   # its kernels are carried in the kit, not routed from the core
    assert modes.KIT_MODES["fast"].kit_levers[0] == "mkdit" and "mkdit" not in modes.KIT_MODES["exact"].kit_levers and "mkdit" not in modes.kit_mode("big").kit_levers   # fast's package levers open with mkdit (test_modes holds the whole tuple); exact carries none of it; big replaces it by dtk
    assert big.REPLACED_KIT["mkdit"] == "dtk" and big.DISENGAGED_KIT["mkdit"].startswith("memory_holder:")


def test_carried_kernel_files_are_present():
    root = stack.opt_root()
    got = mkdit.check_files(root)
    assert sorted(got) == sorted(mkdit.KERNEL_FILES)
    for rel, h in got.items():
        assert h == hashlib.sha256(open(os.path.join(root, mkdit.ADDON, rel), "rb").read()).hexdigest()
    assert os.path.isfile(os.path.join(root, mkdit.ADDON, "README.md"))                          # the add-on's own card, carried beside its kernels


def test_check_files_refuses_when_a_kernel_file_is_absent(tmp_path):
    root = tmp_path / "opt"
    (root / mkdit.ADDON / "mkdit").mkdir(parents=True)
    for rel in mkdit.KERNEL_FILES:
        (root / mkdit.ADDON / rel).write_text("x = 1\n")
    good = hashlib.sha256((root / mkdit.ADDON / "mkdit" / "mk2.py").read_bytes()).hexdigest()   # both stand-in files carry the same bytes
    assert mkdit.check_files(str(root)) == {rel: good for rel in mkdit.KERNEL_FILES}            # present: served (the hashes are reporting evidence)
    os.remove(root / mkdit.ADDON / "mkdit" / "mkrf3.py")
    with pytest.raises(mkdit.MkditRefused, match="is absent"):
        mkdit.check_files(str(root))


# ----------------------------------------------------------------------------------------------------- the adapter (forward) on the CPU
class _FakeMK:
    """Stands in for MK2TokenTransformer: hoist(Z) -> [NB, H, I, I]; forward(A [I, C], S [I, Cs]) -> A + 1 + sample-dependent S mean (records every call)."""
    def __init__(self, tok, **cfg):
        self.tok, self.cfg, self.Bt, self.I, self.calls, self.hoisted = tok, cfg, None, None, [], []

    def hoist(self, Z_II):
        Z = Z_II.reshape(-1, Z_II.shape[-3], Z_II.shape[-2], Z_II.shape[-1])
        assert Z.shape[0] == 1, "diffusion batch > 1 not supported by the fused path"                 # the carried kernel's own assert on a batched pair bias
        self.hoisted.append(tuple(Z_II.shape))
        return torch.zeros(2, 3, Z.shape[1], Z.shape[2])

    def forward(self, A, S, Z_II=None, Beta_II=None):
        assert A.dim() == 2 and S.dim() == 2 and A.shape[0] == S.shape[0]
        self.calls.append((tuple(A.shape), tuple(S.shape), float(S.mean())))
        return A + 1.0 + S.mean()


class _GF:
    """rf3.graph_flags' hoist cache: one value per key per roll-out."""
    HOIST = True

    def __init__(self):
        self.cache, self.gets = {}, 0

    def hoist_get(self, key, fn):
        self.gets += 1
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]


@pytest.fixture
def adapter(monkeypatch):
    """mkdit installed by hand on the CPU: the gate at its default, the kernel and the hoist cache stood in, the device check a no-op (its own test below)."""
    orig_calls = []
    fakes = []

    def orig_forward(self, A_I, S_I, Z_II, Beta_II):
        orig_calls.append((tuple(A_I.shape), Beta_II is not None))
        return A_I * 0.5

    def new_mk(tok):
        fakes.append(_FakeMK(tok, **mkdit.CONFIG))
        return fakes[-1]
    gf = _GF()
    monkeypatch.setattr(mkdit, "GATE", mkdit.make_gate())
    monkeypatch.setattr(mkdit, "STATE", dict(mkdit.STATE, on=True, impl="fake", home="/nonexistent", shapes={}, samples=0, hoists=0, atom_calls=0, objs=0, error=None))
    monkeypatch.setattr(mkdit, "_ORIG", {"forward": orig_forward})
    monkeypatch.setattr(mkdit, "_OBJS", {})
    monkeypatch.setattr(mkdit, "_new_mk", new_mk)
    monkeypatch.setattr(mkdit, "_check_device", lambda t: None)
    monkeypatch.setitem(sys.modules, "rf3.graph_flags", gf)
    return types.SimpleNamespace(orig_calls=orig_calls, fakes=fakes, gf=gf, tok=object())


def test_forward_loops_the_single_sample_kernel_over_every_diffusion_batch_sample(adapter):
    D, I, C, CS = 5, 448, 8, 6                                    # upstream diffusion_batch_size=5; I above the 400-token gate
    A = torch.randn(D, I, C); S = torch.arange(D, dtype=torch.float32).view(D, 1, 1).expand(D, I, CS).contiguous(); Z = torch.randn(1, I, I, 4)
    out = mkdit.forward(adapter.tok, A, S, Z, None)
    mk = adapter.fakes[0]
    assert len(adapter.fakes) == 1 and len(mk.calls) == D and adapter.orig_calls == []          # served: five launches, one per sample; no stock branch
    assert [c[0] for c in mk.calls] == [(I, C)] * D and [c[1] for c in mk.calls] == [(I, CS)] * D
    assert [c[2] for c in mk.calls] == [0.0, 1.0, 2.0, 3.0, 4.0]                                 # each sample's own S_I row, in order
    assert out.shape == (D, I, C) and torch.equal(out, torch.stack([A[d] + 1.0 + d for d in range(D)], 0))
    assert mk.hoisted == [(1, I, I, 4)] and adapter.gf.gets == 1                                 # the pair bias laid out once (per roll-out), shared by the samples
    out2 = mkdit.forward(adapter.tok, A, S, Z, None)                                             # the next denoiser call of the roll-out: no new layout, the same kernel object
    assert torch.equal(out2, out) and mk.hoisted == [(1, I, I, 4)] and adapter.gf.gets == 2 and len(adapter.fakes) == 1 and len(mk.calls) == 2 * D
    c = mkdit.census()
    assert (c["calls"], c["served"], c["gated"], c["samples"], c["hoists"], c["objs"], c["atom_calls"]) == (2, 2, 0, 2 * D, 1, 1, 0) and c["shapes"] == {f"D{D}_I{I}": 2}
    assert mkdit.problems() == [] and mkdit.describe()["ok"] is True


def test_forward_serves_rank2_and_broadcast_conditioning(adapter):
    I, C, CS = 512, 8, 6
    A2 = torch.randn(I, C); S2 = torch.zeros(I, CS); Z = torch.randn(I, I, 4)
    out = mkdit.forward(adapter.tok, A2, S2, Z, None)                                            # [I, C]: one sample, rank kept
    assert out.shape == (I, C) and torch.equal(out, A2 + 1.0) and len(adapter.fakes[0].calls) == 1
    A3 = torch.randn(3, I, C); S1 = torch.zeros(1, I, CS)
    out3 = mkdit.forward(adapter.tok, A3, S1, Z, None)                                           # S_I [1, I, Cs] broadcast over the batch
    assert out3.shape == (3, I, C) and len(adapter.fakes[0].calls) == 4 and mkdit.census()["samples"] == 4


def test_atom_transformers_and_below_gate_calls_run_the_class_forward_counted(adapter):
    A = torch.randn(5, 448, 8); S = torch.zeros(5, 448, 6); Z = torch.randn(1, 448, 448, 4)
    out = mkdit.forward(adapter.tok, A, S, Z, torch.zeros(1))                                    # Beta_II given: an atom transformer — not the lever's site
    assert torch.equal(out, A * 0.5) and adapter.orig_calls == [((5, 448, 8), True)] and adapter.fakes == [] and mkdit.STATE["atom_calls"] == 1
    small = torch.randn(5, 200, 8)
    out = mkdit.forward(adapter.tok, small, torch.zeros(5, 200, 6), torch.randn(1, 200, 200, 4), None)   # 200 tokens < mkdit.MIN_I=400: the stock blocks, counted
    assert torch.equal(out, small * 0.5) and adapter.fakes == []
    c = mkdit.census()
    assert (c["calls"], c["served"], c["gated"]) == (1, 0, 1) and c["atom_calls"] == 1
    assert mkdit.problems() == []                                                                # a gated call is a named policy, not a failure; the seam was reached


def test_gate_lower_bound_is_inclusive():
    g = mkdit.make_gate(1000)
    assert g.min_tokens == 1000 and not g.decide(999).served and g.decide(1000).served
    assert mkdit.make_gate().min_tokens == 400 and not mkdit.make_gate().decide(399).served and mkdit.make_gate().decide(400).served


def test_a_served_call_off_cuda_refuses_by_name_no_stock_branch(adapter, monkeypatch):
    """The shipped device check on a CPU tensor: MkditRefused out of the forward, the error recorded for the exit tally, the class's own
    forward NOT called (no silent stock branch at any batch size)."""
    monkeypatch.setattr(mkdit, "_check_device", _SHIPPED_CHECK_DEVICE)
    A = torch.randn(5, 448, 8); S = torch.zeros(5, 448, 6); Z = torch.randn(1, 448, 448, 4)
    with pytest.raises(mkdit.MkditRefused, match="CUDA device only"):
        mkdit.forward(adapter.tok, A, S, Z, None)
    assert adapter.orig_calls == [] and adapter.fakes == []
    assert "MkditRefused" in mkdit.STATE["error"] and mkdit.problems() and mkdit.describe()["ok"] is False


_SHIPPED_CHECK_DEVICE = mkdit._check_device                                                   # the fixture stands the device check out for the loop tests; this is the real one


def test_a_batched_pair_bias_raises_out_of_the_forward(adapter):
    A = torch.randn(5, 448, 8); S = torch.zeros(5, 448, 6); Zb = torch.randn(5, 448, 448, 4)   # a pair bias with its own batch: the kernel's assert propagates (recorded), never a silent branch
    with pytest.raises(AssertionError, match="diffusion batch > 1"):
        mkdit.forward(adapter.tok, A, S, Zb, None)
    assert adapter.orig_calls == [] and "AssertionError" in mkdit.STATE["error"]
    assert any("mkdit error" in p for p in mkdit.problems())


def test_dead_seam_is_a_problem(adapter):
    assert mkdit.census()["calls"] == 0
    assert any("no DiffusionTransformer token call reached the mkdit seam" in p for p in mkdit.problems()) and mkdit.describe()["ok"] is False


# ----------------------------------------------------------------------------------------------------- enable(): prerequisites refused by name
@pytest.fixture
def pristine(monkeypatch):
    monkeypatch.setattr(mkdit, "STATE", dict(mkdit.STATE, on=False, impl=None, home=None, files_sha256=None, error=None, reason=None, gpu=None))
    monkeypatch.setattr(mkdit, "GATE", None)
    monkeypatch.setattr(dtk, "STATE", dict(dtk.STATE, on=False))
    for m in ("rf3", "rf3.graph_flags", "rf3.model", "rf3.model.layers", "rf3.model.layers.af3_diffusion_transformer", mkdit.ADAPTER):
        monkeypatch.delitem(sys.modules, m, raising=False)
    return monkeypatch


def _patched_tree(monkeypatch, hoist=True):
    rf3 = types.ModuleType("rf3"); gf = types.ModuleType("rf3.graph_flags"); gf.HOIST = hoist; gf.hoist_get = lambda key, fn: fn()
    monkeypatch.setitem(sys.modules, "rf3", rf3); monkeypatch.setitem(sys.modules, "rf3.graph_flags", gf); rf3.graph_flags = gf
    return gf


def test_enable_refuses_on_the_pristine_tree(pristine):
    monkeypatch = pristine
    monkeypatch.setattr(sys, "path", [p for p in sys.path])                                     # rf3 absent from this interpreter: the import fails by name
    monkeypatch.setitem(sys.modules, "rf3", None)                                                # `import rf3.graph_flags` -> ImportError
    with pytest.raises(mkdit.MkditRefused, match="not the patched tree"):
        mkdit.enable(opt_root=stack.opt_root(), gpu={"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"})
    assert mkdit.STATE["on"] is False and mkdit.describe()["reason"] == "not installed"


def test_enable_refuses_when_the_hoist_is_off(pristine):
    _patched_tree(pristine, hoist=False)
    with pytest.raises(mkdit.MkditRefused, match="RF3_HOIST is off"):
        mkdit.enable(opt_root=stack.opt_root(), gpu={"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"})
    assert mkdit.STATE["on"] is False


def test_enable_refuses_beside_dtk_or_dattn(pristine):
    monkeypatch = pristine
    _patched_tree(monkeypatch)
    monkeypatch.setattr(dtk, "STATE", dict(dtk.STATE, on=True))
    with pytest.raises(mkdit.MkditRefused, match="dtk is installed"):
        mkdit.enable(opt_root=stack.opt_root(), gpu={"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"})
    monkeypatch.setattr(dtk, "STATE", dict(dtk.STATE, on=False))
    adp = types.ModuleType(mkdit.ADAPTER); adp.DATTN = {"on": True}
    monkeypatch.setitem(sys.modules, mkdit.ADAPTER, adp)
    with pytest.raises(mkdit.MkditRefused, match="dattn is on"):
        mkdit.enable(opt_root=stack.opt_root(), gpu={"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"})
    assert mkdit.STATE["on"] is False


def test_enable_refuses_when_a_carried_kernel_file_is_absent(pristine, tmp_path):
    _patched_tree(pristine)
    root = tmp_path / "opt"
    (root / mkdit.ADDON / "mkdit").mkdir(parents=True)
    (root / mkdit.ADDON / mkdit.KERNEL_FILES[0]).write_text("x = 1\n")          # mkrf3.py never written: absent
    with pytest.raises(mkdit.MkditRefused, match="is absent"):
        mkdit.enable(opt_root=str(root), gpu={"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"})
    assert mkdit.STATE["on"] is False


def test_stack_turns_a_refusal_into_not_active():
    """stack.fpf_apply enables mkdit after the arm for a row naming it and converts MkditRefused into the NOT ACTIVE line + ActivationError
    (exit 3 under the hook) — never a fold without the lever; report/fold read its verdict (tally block, LEVER line, lever_failures)."""
    import inspect
    from .. import fold, report
    src = inspect.getsource(stack.fpf_apply)
    assert 'if "mkdit" in (rep.get("levers") or [])' in src and "except _mkdit.MkditRefused" in src and "raise ActivationError" in src and "_report.print_mkdit(rep)" in src
    assert 'out["mkdit"] = _mkdit.describe()' in inspect.getsource(report.tally)
    assert '"mkdit"' in inspect.getsource(fold).split("# the package levers with a verdict of their own")[0].rsplit("for name in", 1)[1]   # fold judges mkdit with the package levers (lever_failures / lever_rank_failures)
    line = report.mkdit_line({"mkdit": {"on": True, "impl": "/opt/forward/rf3_mk_dit_addon/mkdit/mk2.py", "files_sha256": {"mkdit/mk2.py": "ab" * 32, "mkdit/mkrf3.py": "cd" * 32},
                                        "min_tokens": 400, "config": dict(mkdit.CONFIG), "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0"}}})
    assert line == ("[rosettafold3-opt] MKDIT on=True executes_from=/opt/forward/rf3_mk_dit_addon/mkdit/mk2.py files=mk2.py:abababababab,mkrf3.py:cdcdcdcdcdcd "
                    "min_tokens=400 config=BMR=64,BN=128,BK=64,BNK=64,num_warps=8,num_stages=3,bms=16 gpu=NVIDIA_H100_80GB_HBM3/cc9.0")


def test_below_sm80_fast_refuses_by_name(pristine):
    """The megakernel's tile configurations start at compute capability 8.0 (``CONFIGS``). On a card below that enable() raises
    ``MkditRefused`` naming the card and the capabilities that have a configuration — the mode refuses by name (a mode is all of its levers,
    never a subset); nothing is installed or imported. No visible GPU at all is likewise a refusal by name."""
    monkeypatch = pristine
    _patched_tree(monkeypatch)
    for name, cc in (("Tesla T4", "7.5"), ("Tesla V100-SXM2-32GB", "7.0")):
        monkeypatch.setattr(mkdit, "STATE", dict(mkdit.STATE, on=False, impl=None, reason=None, gpu=None, config=None))
        with pytest.raises(mkdit.MkditRefused, match="cannot run here") as ei:
            mkdit.enable(opt_root=stack.opt_root(), gpu={"name": name, "cc": cc})
        assert f"compute capability {cc!r}" in str(ei.value) and "8.0, 9.0 and up" in str(ei.value) and name in str(ei.value)
        assert mkdit.STATE["on"] is False and mkdit.STATE["config"] is None and "rf3.model.layers.af3_diffusion_transformer" not in sys.modules
        assert " inactive=" not in report.mkdit_line({"mkdit": mkdit.describe()})
    # at or above 8.0 the card takes its CONFIGS row and enable() proceeds to the next precondition (here: the carried kernel files under a bare root)
    for name, cc, want in (("NVIDIA A100-SXM4-80GB", "8.0", (8, 0)), ("NVIDIA L40S", "8.9", (8, 0)), ("NVIDIA H100 80GB HBM3", "9.0", (9, 0)), ("NVIDIA B200", "10.0", (9, 0))):
        monkeypatch.setattr(mkdit, "STATE", dict(mkdit.STATE, on=False, impl=None, reason=None, gpu=None, config=None))
        with pytest.raises(mkdit.MkditRefused, match="is absent"):
            mkdit.enable(opt_root=str(stack.opt_root()) + "/nonexistent", gpu={"name": name, "cc": cc})
        assert mkdit.STATE["config"] == mkdit.CONFIGS[want] and mkdit.describe()["config"] == mkdit.CONFIGS[want]
    src = inspect.getsource(stack.fpf_apply)
    assert "MkditRefused" in src and "raise ActivationError" in src and "inactive" not in src                   # a refusal of the lever is the mode's NOT ACTIVE line, never a subset
    monkeypatch.setattr(stack, "gpu_info", lambda: None)
    monkeypatch.setattr(mkdit, "STATE", dict(mkdit.STATE, on=False, impl=None, reason=None, gpu=None))
    with pytest.raises(mkdit.MkditRefused, match="no CUDA GPU visible"):
        mkdit.enable(opt_root=stack.opt_root())
    assert mkdit.STATE["on"] is False
    assert mkdit._cc_tuple("9.0") == (9, 0) and mkdit._cc_tuple("10.0") == (10, 0) and mkdit._cc_tuple((12, 0)) == (12, 0) and mkdit._cc_tuple(None) is None and mkdit._cc_tuple("n/a") is None


def test_a_missing_mkdit_tally_or_a_false_verdict_still_fails_by_name():
    assert fold.lever_failures("mkdit", [{"rc": 0, "seed": 7}], {"7": {}}) == ["seed 7: no mkdit tally (the lever's counters were not written at exit)"]
    assert fold.lever_failures("mkdit", [{"rc": 0, "seed": 7}], {"7": {"mkdit": {"on": True, "ok": False, "reason": "dead seam"}}}) == ["seed 7: mkdit dead seam"]