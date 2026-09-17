"""Exit rule: `xtr` / `ttr` served 0 because the shared core's transition provider ANSWERED every eligible call with a STOCK row
by name (`stock:<lever>:row=<row>` — `exact` at cell 9.0|bf16|pair_c128_n4|N<=256|eager names torch_swiglu: every pair-transition call of an
input of <= 256 tokens on cc 9.0) and nothing fell back = by design: accounted `state=skipped reason=aside
aside=<word>`, exit 0, never partial — the same rule as the TriMul word's refusal by name. A provider error after engagement
(`fallback:*`) stays partial; a lever that met NO eligible call (only `stock:C=<C>` dtype / width gates) stays partial; a run that served
calls is unchanged. Class contract of report.kit_evidence / partial_of / lever_lines only: no torch, no GPU, no assertion on the row the
core serves today (the census words are given)."""
from protenix_v1_opt import report as R

def _levers(counts):
    return {"cfg": {"trimul": "exact", "xtr": True}, "counts": {"trimul": {"exact": 40}, "transition": dict(counts)},
            "transition": {"lever": "xtr", "word": "exact", "selections": {}, "refusals": {}}}

def test_stock_row_tier_for_every_call_is_aside():
    ev = R.kit_evidence(_levers({"stock:xtr:row=torch_swiglu": 9241, "stock:C=384": 7728, "stock:C=128": 308}), "exact", ("xtr",), [{"N_token": 200}])
    assert ev["xtr"]["served"] == 0 and not ev["xtr"]["fallback"]
    assert ev["xtr"]["aside"] == {"state": "aside", "word": "stock:xtr:row=torch_swiglu"}
    assert R.partial_of(ev) == ([], None)
    line = [l for l in R.lever_lines(ev, {"mode": "exact"}) if l.rstrip().endswith("lever=xtr")]
    assert line and " state=skipped reason=aside " in line[0] and " served=0 " in line[0], line

def test_provider_error_stays_partial():
    ev = R.kit_evidence(_levers({"stock:xtr:row=torch_swiglu": 9240, "fallback:xtr:error:RuntimeError": 1, "stock:C=384": 7728}), "exact", ("xtr",), [{"N_token": 200}])
    assert "aside" not in ev["xtr"] and R.partial_of(ev)[0] == ["xtr"]

def test_no_eligible_call_stays_partial():
    ev = R.kit_evidence(_levers({"stock:C=384": 7728, "stock:C=128": 308}), "exact", ("xtr",), [{"N_token": 200}])
    assert "aside" not in ev["xtr"] and R.partial_of(ev)[0] == ["xtr"]

def test_served_run_unchanged():
    ev = R.kit_evidence(_levers({"xtr:C=128": 9000, "stock:xtr:row=torch_swiglu": 241, "stock:C=384": 7728}), "exact", ("xtr",), [{"N_token": 400}])
    assert ev["xtr"]["served"] == 9000 and "aside" not in ev["xtr"] and R.partial_of(ev) == ([], None)
