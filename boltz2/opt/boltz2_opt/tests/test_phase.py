"""boltz2_opt.phase: the per-item PHASE timing line on a stand-in model — no boltz, no torch, no GPU.
The stand-in mimics torch's Module call protocol (forward pre/post hooks around forward) and Boltz2's statement order
(trunk → diffusion_conditioning → structure_module.sample → confidence_module inside forward; predict_step calls the instance)."""
import re
import time
import types

from .. import phase, worker_launch


class _Handle:
    def __init__(self, lst, fn):
        self._lst, self._fn = lst, fn

    def remove(self):
        if self._fn in self._lst:
            self._lst.remove(self._fn)


class FakeModule:
    def __init__(self):
        self._pre, self._post = [], []

    def register_forward_pre_hook(self, fn):
        self._pre.append(fn); return _Handle(self._pre, fn)

    def register_forward_hook(self, fn):
        self._post.append(fn); return _Handle(self._post, fn)

    def __call__(self, *a, **k):
        for h in list(self._pre):
            h(self, a)
        out = self.forward(*a, **k)
        for h in list(self._post):
            h(self, a, out)
        return out


class Sleeper(FakeModule):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, *a, **k):
        time.sleep(self.s); return 1


class FakeSampler(FakeModule):
    CALLS = []

    def sample(self, *a, **k):
        FakeSampler.CALLS.append("stock"); time.sleep(0.04); return {"coords": 0}


class FakeBoltz2(FakeModule):
    def __init__(self, confidence=True):
        super().__init__()
        self.diffusion_conditioning = Sleeper(0.02)
        self.structure_module = FakeSampler()
        self.confidence_module = Sleeper(0.03) if confidence else None

    def forward(self, feats, **kw):
        time.sleep(0.05)                                   # the trunk
        self.diffusion_conditioning(feats)
        out = dict(self.structure_module.sample(feats))
        if self.confidence_module is not None:
            self.confidence_module(feats)
        return out

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)


LINE = re.compile(r"^PHASE item=(\S+) lm_s=(\S+) trunk_s=(\S+) sampler_s=(\S+) conf_s=(\S+) total_s=(\S+) cond_s=(\S+) fwd_s=(\S+)$")


def _fresh_cls():
    return type("FakeBoltz2X", (FakeBoltz2,), {"predict_step": FakeBoltz2.predict_step})


def test_line_fields_boundaries_and_sum_guard(capsys):
    cls = _fresh_cls()
    assert phase.install(cls) is True and phase.install(cls) is False          # idempotent
    m = cls(); rec = types.SimpleNamespace(id="7EBY_crop")
    out = m.predict_step({"record": [rec]}, 0)
    assert out == {"coords": 0}
    text = capsys.readouterr().out
    assert "PHASE-NOTE boundaries" in text and "separable=yes" in text
    rows = [LINE.match(l) for l in text.splitlines() if l.startswith("PHASE item=")]
    assert len(rows) == 1 and rows[0], text
    item, lm, trunk, samp, conf, total, cond, fwd = rows[0].groups()
    assert item == "7EBY_crop" and lm == "-"
    trunk, samp, conf, total, cond, fwd = map(float, (trunk, samp, conf, total, cond, fwd))
    assert 0.045 <= trunk < 0.05 + 0.03, trunk                      # forward entry → conditioner entry
    assert 0.035 <= samp < 0.07 and 0.025 <= conf < 0.06 and 0.015 <= cond < 0.05
    assert trunk + cond + samp + conf <= fwd * 1.001 + 1e-3 <= total * 1.01 + 2e-3
    assert "PHASE-WARN" not in text
    assert "sample" not in m.structure_module.__dict__ and not m._pre and not m.diffusion_conditioning._pre   # every per-call hook removed
    assert phase.uninstall(cls) is True and not getattr(cls.predict_step, "_phase_timing", False)


def test_late_class_replacement_of_sample_runs_inside_the_marks_and_missing_confidence_prints_dash(capsys):
    cls = _fresh_cls(); phase.install(cls)
    ran = []

    def graphed_sample(self, *a, **k):                     # a kit's class-level replacement installed AFTER the instrument (graph sampler / hoist)
        ran.append("graphed"); time.sleep(0.01); return {"coords": 1}
    FakeSampler.sample, orig = graphed_sample, FakeSampler.sample
    try:
        m = cls(confidence=False)
        assert m.predict_step({"record": [types.SimpleNamespace(id="x")]}, 0) == {"coords": 1}
    finally:
        FakeSampler.sample = orig; phase.uninstall(cls)
    text = capsys.readouterr().out
    g = [LINE.match(l) for l in text.splitlines() if l.startswith("PHASE item=")][0].groups()
    assert ran == ["graphed"] and g[4] == "-" and float(g[3]) >= 0.009          # conf_s '-' ; sampler timed the replacement


def test_oom_style_return_inside_predict_step_still_prints_one_line(capsys):
    cls = _fresh_cls()

    def ps(self, batch, batch_idx, dataloader_idx=0):        # boltz's predict_step swallows an out-of-memory RuntimeError and returns {"exception": True}
        try:
            raise RuntimeError("CUDA out of memory")
        except RuntimeError:
            return {"exception": True}
    cls.predict_step = ps; phase.install(cls)
    try:
        assert cls().predict_step({"record": []}, 0) == {"exception": True}
    finally:
        phase.uninstall(cls)
    rows = [l for l in capsys.readouterr().out.splitlines() if l.startswith("PHASE item=")]
    assert len(rows) == 1 and "item=? " in rows[0] and "trunk_s=- sampler_s=- conf_s=-" in rows[0]


def test_both_arms_place_the_same_instrument():
    """The worker's xl-attach hook and the stock process's phase instrumentation watch for the same model class -- a drift between them
    would make one arm instrument the wrong import (or none)."""
    assert phase.MODEL_MODULE == "boltz.model.models.boltz2" == worker_launch.ATTACH["xl"]["trigger"]


class _FakeAlloc:
    """opt_core.mem.allocator's surface (reset_peak / counters), recorded: the PEAK line's reads and the per-item reset."""
    def __init__(self, log, gib=(3.0, 4.5)): self.log, self.gib = log, gib
    def reset_peak(self, device=None): self.log.append("reset"); return True
    def counters(self, device=None):
        self.log.append("read"); a, r = self.gib
        return {"max_allocated_gib": a, "max_reserved_gib": r, "allocated_gib": a, "reserved_gib": r, "device": "0"}


PEAK = re.compile(r"PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$")     # the log reader's grammar, verbatim


def test_peak_line_grammar_reset_at_item_start_and_rank_note(capsys, monkeypatch):
    """The per-item allocator PEAK line (both arms, the PHASE instrument's site, through opt_core.mem.allocator): reset_peak at item start —
    before the model's forward — then exactly `[boltz2-opt] PEAK item=<id> alloc_gib=<f> reserved_gib=<f>` after the PHASE line, nothing else
    on it; a row-sharded rank (ROWPAIR_RANK) adds a separate PEAK-NOTE rank line; without CUDA a PEAK-NOTE names the absence, never a zero."""
    log = []
    monkeypatch.setattr(phase, "_alloc", lambda: _FakeAlloc(log))
    cls = _fresh_cls(); fwd = cls.forward
    cls.forward = lambda self, feats, **kw: (log.append("forward"), fwd(self, feats, **kw))[1]
    phase.install(cls)
    monkeypatch.delenv("ROWPAIR_RANK", raising=False)
    cls().predict_step({"record": [types.SimpleNamespace(id="7EBY_crop")]}, 0)
    out = capsys.readouterr().out
    lines = [l for l in out.splitlines() if l.startswith("[boltz2-opt] PEAK ")]
    assert lines == ["[boltz2-opt] PEAK item=7EBY_crop alloc_gib=3.00 reserved_gib=4.50"] and PEAK.search(lines[0]).groups() == ("7EBY_crop", "3.00", "4.50"), lines
    assert log.index("reset") < log.index("forward") < log.index("read"), log      # reset at item start, read at item end
    assert out.index("PHASE item=7EBY_crop") < out.index("[boltz2-opt] PEAK item=7EBY_crop"), "the PEAK line follows the item's PHASE line"
    monkeypatch.setenv("ROWPAIR_RANK", "1")
    cls().predict_step({"record": [types.SimpleNamespace(id="x2")]}, 0)
    out = capsys.readouterr().out.splitlines()
    assert "[boltz2-opt] PEAK item=x2 alloc_gib=3.00 reserved_gib=4.50" in out and "[boltz2-opt] PEAK-NOTE item=x2 rank=1" in out, out
    monkeypatch.delenv("ROWPAIR_RANK")
    monkeypatch.setattr(phase, "_alloc", lambda: _FakeAlloc(log, gib=(None, None)))     # no torch / no device: the core's counters read None
    cls().predict_step({"record": [types.SimpleNamespace(id="cpu")]}, 0)
    out = capsys.readouterr().out
    assert "[boltz2-opt] PEAK-NOTE item=cpu cuda=absent" in out and "PEAK item=cpu" not in out
    phase.uninstall(cls)


def test_peak_summary_reduces_rank0_and_rank_max(tmp_path):
    from .. import worker
    text = "\n".join(["[boltz2-opt] PEAK item=a alloc_gib=10.50 reserved_gib=12.00", "[boltz2-opt] PEAK item=b alloc_gib=20.25 reserved_gib=22.00", "noise"])
    assert worker.peak_summary(text, [], 1) == "[boltz2-opt] PEAK-SUMMARY items=2 rank0_max_alloc_gib=20.25 rankmax_alloc_gib=20.25 rankmax_reserved_gib=22.00 ranks=1"
    r0 = tmp_path / "rank0.log"; r0.write_text(text + "\n"); r1 = tmp_path / "rank1.log"; r1.write_text("[boltz2-opt] PEAK item=b alloc_gib=31.00 reserved_gib=33.50\n[boltz2-opt] PEAK-NOTE item=b rank=1\n")
    assert worker.peak_summary("relayed lines ignored at P>1", [str(r0), str(r1)], 2) == "[boltz2-opt] PEAK-SUMMARY items=2 rank0_max_alloc_gib=20.25 rankmax_alloc_gib=31.00 rankmax_reserved_gib=33.50 ranks=2"
    assert worker.peak_summary("nothing here", [], 1) == "[boltz2-opt] PEAK-SUMMARY items=0 rank0_max_alloc_gib=- rankmax_alloc_gib=- rankmax_reserved_gib=- ranks=1"
