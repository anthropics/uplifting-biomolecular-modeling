"""The dummy-template elision lever: switch grammar, the all-dummy decision, the wiring (registry / modes rows / attach /
evidence tuple / the row-sharded line's by-name drop)."""
import numpy as np
from .. import templskip as TS, modes, registry, worker_launch


def test_switch_grammar():
    assert TS.SWITCH == "BOLTZ_TEMPL_SKIP" and TS.VARIANTS == ("1",) and TS.LEVERS == ("templ_skip",)
    assert TS.variant({"BOLTZ_TEMPL_SKIP": "1"}) == "1" and TS.variant({"BOLTZ_TEMPL_SKIP": "0"}) is None and TS.variant({}) is None


def test_dummy_pass_reads_the_template_mask_only():
    assert TS.dummy_pass({"template_mask": np.zeros((1, 2, 7))}) is True             # every slot all zero: the module's update is the zero tensor
    m = np.zeros((1, 2, 7)); m[0, 1, 3] = 1
    assert TS.dummy_pass({"template_mask": m}) is False                             # one token of one slot set: the module runs
    assert TS.dummy_pass({"template_mask_cb": np.zeros((1, 2, 7))}) is None          # no template_mask: the module runs (counted no_mask)
    assert TS.dummy_pass(None) is None


def test_not_applied_reports_nothing():
    assert TS.apply("0") == [] and TS.report()["applied"] == [] and TS.line() is None


def test_wiring():
    L = registry.LEVERS["templ_skip"]
    assert L["switch"] == "BOLTZ_TEMPL_SKIP" and L["value"] == "1" and L["tier"] == 1
    assert registry.LEVER_IDS["templ_skip"]["strategy"].startswith("LOCAL.")
    assert modes.LEVER_ATTACH["templ_skip"] == "templskip" and worker_launch.ATTACH["templskip"]["module"] == "boltz2_opt.templskip"
    for m in ("exact", "fast", "big"):
        r = modes.resolve(m)
        assert "templ_skip" in r["levers"] and r["env"].get("BOLTZ_TEMPL_SKIP") == "1" and "templskip" in r["attach"], m
        if "graph" in r["attach"]:                                                              # attached right after `graph`: its templ unit pins the stock source and restates the body this lever wraps
            assert r["attach"].index("templskip") == r["attach"].index("graph") + 1
        else:                                                                                   # big names the trunk graphs off by rule (no CUDA graphs in the memory mode): the wrapper sits on the stock body; the TABLED order keeps the rule
            assert m == "big" and modes.MODES[m]["attach"].index("templskip") == modes.MODES[m]["attach"].index("graph") + 1
    modes.set_n_gpu(2)
    try:
        r = modes.resolve("big")
        assert "templ_skip" not in r["levers"] and r["tp_off"]["templ_skip"].startswith("replaced_by_rowpair") and "BOLTZ_TEMPL_SKIP" not in r["env"]
    finally:
        modes.set_n_gpu(1)


def test_the_wrapper_elides_all_dummy_passes_with_the_modules_tail_and_runs_the_rest():
    """_make_forward returns a forward that serves u_proj(relu(v_norm(0))) for an all-dummy pass and calls the wrapped body otherwise (CPU)."""
    import torch
    calls = []
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.v_norm = torch.nn.LayerNorm(6); self.relu = torch.nn.ReLU(); self.u_proj = torch.nn.Linear(6, 8, bias=True)
            torch.nn.init.normal_(self.u_proj.bias); torch.nn.init.normal_(self.u_proj.weight)
    m = M()
    def body(self, z, feats, pair_mask, use_kernels=False):
        calls.append(1); return z + 1
    fwd = TS._make_forward(body)
    assert callable(fwd)
    z = torch.randn(1, 5, 5, 8)
    with torch.no_grad():
        u = fwd(m, z, {"template_mask": torch.zeros(1, 2, 5)}, None)
        v = m.v_norm(torch.randn(1, 2, 5, 5, 6)); mask = torch.zeros(1, 2)[:, :, None, None, None]
        stock_tail = m.u_proj(m.relu((v * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)))   # stock's statements after the Pairformer under an all-zero mask
    assert torch.equal(u, stock_tail) and u.shape == z.shape and calls == [] and TS.census()["skipped"] >= 1
    assert not torch.equal(u, torch.zeros_like(z)), "the update is the projection's bias, not zero"
    u = fwd(m, z, {"template_mask": torch.ones(1, 2, 5)}, None)
    assert torch.equal(u, z + 1) and calls == [1] and TS.census()["live"] >= 1
    u = fwd(m, z, {}, None)
    assert calls == [1, 1] and TS.census()["no_mask"] >= 1


def test_the_elision_replays_the_pairformers_random_draws():
    """Skipping the template Pairformer must leave the generator where stock leaves it: get_dropout_mask draws torch.rand(float32) four
    times per PairformerNoSeqLayer ([B*T, N, 1, 1] x3, [B*T, 1, N, 1] x1) even in eval mode (CPU generator here; CUDA Philox in the worker)."""
    import torch
    class L(torch.nn.Module):
        pass
    class M(torch.nn.Module):
        def __init__(self, n_layers):
            super().__init__(); self.relu = torch.nn.ReLU(); self.u_proj = torch.nn.Linear(6, 8)
            self.pairformer = torch.nn.Module(); self.pairformer.layers = torch.nn.ModuleList([L() for _ in range(n_layers)])
    B, T, N = 1, 2, 7
    m = M(2)                                                                        # built before seeding: nn.Linear's init draws from the CPU generator
    torch.manual_seed(11)
    z = torch.zeros(B * T, N, N, 6)
    for _ in range(2):                                                              # stock: 2 layers x 4 get_dropout_mask draws
        for col in (False, False, False, True):
            v = z[:, 0:1, :, 0:1] if col else z[:, :, 0:1, 0:1]
            torch.rand(v.shape, dtype=torch.float32, device=v.device)
    want = torch.rand(5)                                                            # the next consumer's draw after stock's template pass
    torch.manual_seed(11)
    assert TS._replay_dropout_draws(m, B * T, N, z.device) == 8
    assert torch.equal(torch.rand(5), want), "the replay advances the generator exactly as the skipped Pairformer does"
    torch.manual_seed(11)
    with torch.no_grad():
        TS._elided_update(m, torch.zeros(B, N, N, 8), torch.zeros(B, T, N))         # the whole elided path: same draws, then the projection
    assert torch.equal(torch.rand(5), want)


def test_the_row_chunked_transition_is_idle_by_name_when_every_template_pass_was_elided():
    """stack._templ_all_elided: templ_skip served every call (no templated / no-mask pass, no error) — with the layer driver serving the C=128
    stacks, xl_trans has no caller left and its idle census is named (xl_trans_idle_by), not a NOT ACTIVE."""
    from .. import stack
    ok = {"templskip_report": {"applied": ["templ_skip"], "census": {"calls": 20, "skipped": 20, "live": 0, "capturing": 0, "no_mask": 0, "errors": 0}}}
    assert stack._templ_all_elided(ok) is True
    for bad in ({"live": 4, "skipped": 0}, {"skipped": 16, "live": 4}, {"skipped": 20, "no_mask": 1}, {"skipped": 20, "errors": 1}):
        r = {"templskip_report": {"applied": ["templ_skip"], "census": {**ok["templskip_report"]["census"], **bad}}}
        assert stack._templ_all_elided(r) is False, bad
    assert stack._templ_all_elided({}) is False and stack._templ_all_elided({"templskip_report": {"applied": [], "census": {"skipped": 20}}}) is False
