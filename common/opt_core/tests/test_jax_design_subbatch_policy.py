"""jax_design.subbatch_policy: the request parser, every named source of the decision, the census fragment, the quadratic fit,
and the import hygiene (nothing under opt_core.jax_design imports jax or torch)."""
import subprocess
import sys

import pytest

from opt_core.jax_design import subbatch_policy as sp


# ----------------------------------------------------------------------------------------------------- parse_request


@pytest.mark.parametrize("text,expected", [
    (None, "auto"), ("auto", "auto"), ("AUTO", "auto"), ("stock", "stock"),
    ("none", None), ("null", None), ("0", None), ("", None), ("unchunked", None), (" 4 ", 4), ("128", 128),
])
def test_parse_request_words(text, expected):
    assert sp.parse_request(text) == expected


def test_parse_request_default_and_errors():
    assert sp.parse_request(None, default=None) is None
    assert sp.parse_request(None, default=4) == 4
    with pytest.raises(sp.PolicyError):
        sp.parse_request("fast")
    with pytest.raises(sp.PolicyError):
        sp.parse_request("-2")


# ----------------------------------------------------------------------------------------------------- choose: every source





GB = 10 ** 9
PEAK = sp.QuadraticPeak(a=7.2e-5, b=1.0e-3, c=0.5, unit=1e9, margin=0.10)   # a fixture polynomial; the values are the test's own


def test_stock_value_is_required_and_validated():
    with pytest.raises(TypeError):
        sp.choose(tokens=500, requested=None)                      # the engine's chunk is the kit's to supply; the core holds no value
    assert sp.choose(tokens=500, stock_value=None, requested="auto", device_bytes=1, peak_estimator=PEAK).value is None   # an unchunked stock
    for bad in (0, -4, True, 2.5):
        with pytest.raises(sp.PolicyError):
            sp.choose(tokens=500, stock_value=bad, requested=None)



def test_requested_values_do_not_consult_the_estimate():
    d = sp.choose(stock_value=4, tokens=500, requested=None, device_bytes=None)
    assert (d.value, d.source, d.estimated_bytes) == (None, "requested", None)
    d = sp.choose(stock_value=4, tokens=500, requested=8)
    assert (d.value, d.source) == (8, "requested")
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=500, requested=0)
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=500, requested=True)          # a bool is not a chunk


def test_auto_fits_and_exceeds():
    fits = sp.choose(tokens=571, requested="auto", device_bytes=80 * GB, peak_estimator=PEAK, stock_value=4)
    assert fits.source == "auto:fits" and fits.value is None
    assert fits.estimated_bytes == PEAK(571) and fits.device_bytes == 80 * GB
    exceeds = sp.choose(tokens=1200, requested="auto", device_bytes=80 * GB, peak_estimator=PEAK, stock_value=4)
    assert exceeds.source == "auto:exceeds" and exceeds.value == 4
    assert PEAK(1200) > 80 * GB > PEAK(571)


def test_auto_without_a_device_size_is_named_either_way():
    d = sp.choose(tokens=571, requested="auto", device_bytes=None, peak_estimator=PEAK, stock_value=4)
    assert (d.value, d.source, d.details["when_device_unknown"]) == (4, "auto:no_device", "stock")
    d = sp.choose(tokens=571, requested="auto", device_bytes=0, peak_estimator=PEAK, stock_value=4, when_device_unknown="unchunked")
    assert (d.value, d.source) == (None, "auto:no_device")
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=571, requested="auto", peak_estimator=PEAK, when_device_unknown="guess")
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=571, requested="auto", device_bytes=80 * GB)        # auto needs the adapter's estimator


def test_stock_rule_source():
    rule = sp.threshold_rule(384, 4)
    assert rule(384) is None and rule(385) == 4
    d = sp.choose(stock_value=4, tokens=215, requested="stock", stock_rule=rule)
    assert (d.value, d.source) == (None, "stock")
    d = sp.choose(stock_value=4, tokens=705, requested="stock", stock_rule=rule)
    assert (d.value, d.source) == (4, "stock")
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=705, requested="stock")
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=0, requested="auto", peak_estimator=PEAK)
    with pytest.raises(sp.PolicyError):
        sp.choose(stock_value=4, tokens=10, requested="sometimes", peak_estimator=PEAK)


def test_every_source_is_declared():
    seen = {
        sp.choose(stock_value=4, tokens=500, requested=None).source,
        sp.choose(stock_value=4, tokens=571, requested="auto", device_bytes=80 * GB, peak_estimator=PEAK).source,
        sp.choose(stock_value=4, tokens=1200, requested="auto", device_bytes=80 * GB, peak_estimator=PEAK).source,
        sp.choose(stock_value=4, tokens=571, requested="auto", device_bytes=None, peak_estimator=PEAK).source,
        sp.choose(stock_value=4, tokens=571, requested="stock", stock_rule=sp.threshold_rule(384, 4)).source,
        sp.fixed(stock_value=4, tokens=571, value=None).source,
    }
    assert seen == set(sp.SOURCES)


def test_fixed_is_the_kits_shipped_constant_with_source_kit():
    d = sp.fixed(stock_value=4, tokens=900, value=None)
    assert (d.value, d.source, d.tokens, d.estimated_bytes, d.device_bytes) == (None, "kit", 900, None, None)
    assert d.fields("grad_subbatch") == {"grad_subbatch": "none", "grad_subbatch_source": "kit"} and "shipped" in d.reason()
    assert sp.fixed(stock_value=4, tokens=900, value=4).fields() == {"subbatch": 4, "subbatch_source": "kit"}
    for bad in (0, -1, True, "4"):
        with pytest.raises(sp.PolicyError):
            sp.fixed(stock_value=4, tokens=900, value=bad)


# ----------------------------------------------------------------------------------------------------- census fragment + record


def test_fields_and_record_shape():
    d = sp.choose(tokens=571, requested="auto", device_bytes=80 * GB, peak_estimator=PEAK, stock_value=4)
    assert d.fields() == {"subbatch": "none", "subbatch_source": "auto:fits", "subbatch_est_gb": round(PEAK(571) / 1e9, 1), "subbatch_dev_gb": 80}
    from opt_core import report
    assert report.kv(**d.fields()) == f"subbatch=none subbatch_source=auto:fits subbatch_est_gb={PEAK(571) / 1e9:.1f} subbatch_dev_gb=80"
    assert list(d.fields(prefix="grad_subbatch"))[:2] == ["grad_subbatch", "grad_subbatch_source"]
    rec = d.as_dict()
    assert set(rec) == {"value", "source", "tokens", "stock_value", "estimated_bytes", "device_bytes", "requested", "details"}
    assert rec["requested"] == "auto" and rec["tokens"] == 571
    assert sp.choose(stock_value=4, tokens=500, requested=8).fields() == {"subbatch": 8, "subbatch_source": "requested"}
    for src_decision in (d, sp.choose(stock_value=4, tokens=500, requested=8), sp.choose(stock_value=4, tokens=571, requested="auto", peak_estimator=PEAK)):
        assert isinstance(src_decision.reason(), str) and src_decision.reason()


# ----------------------------------------------------------------------------------------------------- the quadratic fit


def test_fit_quadratic_is_exact_through_three_points_and_least_squares_beyond():
    q = sp.QuadraticPeak(a=2.0, b=-3.0, c=5.0, unit=1.0, margin=0.0)
    pts = [(10, q(10)), (20, q(20)), (40, q(40))]
    f = sp.fit_quadratic(pts, unit=1.0, margin=0.0)
    assert (round(f.a, 9), round(f.b, 9), round(f.c, 9)) == (2.0, -3.0, 5.0)
    f4 = sp.fit_quadratic(pts + [(80, q(80))], unit=1.0, margin=0.0)
    assert abs(f4(60) - q(60)) <= 1
    assert sp.fit_quadratic(pts, unit=1.0, margin=0.5)(40) == int(q(40) * 1.5)
    with pytest.raises(sp.PolicyError):
        sp.fit_quadratic(pts[:2])
    with pytest.raises(sp.PolicyError):
        sp.fit_quadratic([(10, 1.0), (10, 2.0), (10, 3.0)])
    assert "t^2" in f.describe()


def test_quadratic_peak_is_monotone_over_design_sizes():
    vals = [PEAK(t) for t in (128, 215, 384, 571, 705, 850, 1400)]
    assert vals == sorted(vals)


# ----------------------------------------------------------------------------------------------------- import hygiene


def test_importing_the_package_pulls_no_framework():
    code = ("import sys; import opt_core.jax_design; import opt_core.jax_design.subbatch_policy as sp; "
            "sp.choose(stock_value=4, tokens=500, requested='auto', device_bytes=1, peak_estimator=sp.QuadraticPeak(1,1,1)); "
            "bad = [m for m in ('jax', 'jaxlib', 'torch', 'triton', 'numpy') if m in sys.modules]; "
            "print(','.join(bad)); sys.exit(1 if bad else 0)")
    import os
    import opt_core
    core_dir = os.path.dirname(os.path.dirname(os.path.abspath(opt_core.__file__)))      # common/opt_core (tree or install)
    env = dict(os.environ, PYTHONPATH=core_dir + os.pathsep + os.environ.get("PYTHONPATH", ""))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"framework modules imported: {r.stdout.strip()} {r.stderr[-300:]}"
