"""phase.py: the per-item PHASE line — the runner's predict wrapped once, the three Protenix stage methods wrapped per item over whatever
is on the class at item start and restored at item end, NA + PHASE-NOTE for a boundary with no call, no line for an item that raised."""
import re
import sys
import time
import types

import pytest


@pytest.fixture()
def fake(monkeypatch):
    from protenix_v1_opt import phase as PH
    monkeypatch.setitem(PH._STATE, "installed", None)

    class Protenix:
        def get_pairformer_output(self, **kw):
            time.sleep(0.01)
            return "trunk"

        def sample_diffusion(self, **kw):
            time.sleep(0.02)
            return "xyz"

        def run_confidence_head(self, *a, **kw):
            time.sleep(0.005)
            return "conf"

    class InferenceRunner:
        model = Protenix()

        def predict(self, data):
            if data.get("boom"):
                raise RuntimeError("boom")
            m = self.model
            out = [m.get_pairformer_output(), m.sample_diffusion()]
            if not data.get("skip_conf"):
                out.append(m.run_confidence_head())
            time.sleep(0.005)
            return out

    mm = types.ModuleType(PH.MODEL_MODULE); mm.Protenix = Protenix
    rm = types.ModuleType(PH.RUNNER_MODULE); rm.InferenceRunner = InferenceRunner
    monkeypatch.setitem(sys.modules, PH.MODEL_MODULE, mm)
    monkeypatch.setitem(sys.modules, PH.RUNNER_MODULE, rm)
    lines = []
    q = PH.install("[pfx]", log=lines.append)
    return PH, Protenix, InferenceRunner, lines, q


LINE = re.compile(r"^\[pfx\] PHASE item=(\S+) lm_s=- trunk_s=([\d.]+|NA) sampler_s=([\d.]+|NA) conf_s=([\d.]+|NA) total_s=([\d.]+)$")


def test_line_per_item_and_sum_le_total(fake):
    PH, Protenix, Runner, lines, q = fake
    stock = {n: vars(Protenix)[n] for _, n in PH.PHASES}
    assert Runner().predict({"sample_name": "a b"}) == ["trunk", "xyz", "conf"]
    assert len(lines) == 1, lines
    m = LINE.match(lines[0]); assert m, lines[0]
    assert m.group(1) == "a_b"
    tr, sa, co, tot = (float(m.group(i)) for i in (2, 3, 4, 5))
    assert tr > 0 and sa > 0 and co > 0 and tr + sa + co <= tot + 1e-6
    assert {n: vars(Protenix)[n] for _, n in PH.PHASES} == stock          # the wrappers come off at item end
    assert q["trunk"].endswith("Protenix.get_pairformer_output") and q["sampler"].endswith("Protenix.sample_diffusion") and q["conf"].endswith("Protenix.run_confidence_head")
    assert getattr(Runner.predict, "__phase__") == "total" and Runner.predict.__wrapped__ is not None
    assert PH.install("[pfx]") == q                                        # idempotent


def test_kit_patch_at_build_is_the_boundary(fake):
    PH, Protenix, Runner, lines, _ = fake
    calls = []

    def kit_trunk(self, **kw):
        calls.append(1); time.sleep(0.01); return "kit-trunk"
    Protenix.get_pairformer_output = kit_trunk                              # a mode's replacement installed at runner build
    assert Runner().predict({"sample_name": "x"})[0] == "kit-trunk" and calls == [1]
    assert vars(Protenix)["get_pairformer_output"] is kit_trunk             # restored to the kit's object, not the stock's
    assert float(LINE.match(lines[0]).group(2)) > 0


def test_no_call_is_NA_with_note_and_failed_item_prints_nothing(fake):
    PH, Protenix, Runner, lines, _ = fake
    Runner().predict({"sample_name": "s", "skip_conf": True})
    assert LINE.match(lines[0]).group(4) == "NA"
    assert any("PHASE-NOTE conf no call at protenix.model.protenix.Protenix.run_confidence_head in item=s" in l for l in lines[1:]), lines
    lines.clear()
    with pytest.raises(RuntimeError):
        Runner().predict({"sample_name": "f", "boom": True})
    assert lines == []


def test_no_model_module_loaded(fake, monkeypatch):
    PH, Protenix, Runner, lines, _ = fake
    monkeypatch.delitem(sys.modules, PH.MODEL_MODULE)
    Runner().predict({"sample_name": "m"})
    m = LINE.match(lines[0]); assert m and m.group(2) == m.group(3) == m.group(4) == "NA"
    assert any("no model class" in l for l in lines[1:])


def test_stock_route_and_stack_install_the_same_module():
    import inspect
    from protenix_v1_opt import stack, stock_pred
    assert "PHASE.install(PREFIX, log=R.log)" in inspect.getsource(stack._wrap_runner)
    assert 'proof["phase"] = PH.install(PREFIX)' in inspect.getsource(stock_pred.main)
