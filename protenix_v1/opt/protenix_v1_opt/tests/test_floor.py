"""The kit has NO flash triangle-attention size floor (modes.TRIATTN_FLOOR is None: the fused block serves every item upstream hands the kernel
whole and the shared core's provider names the attention core per key); nothing is exported as PTX_TRIATTN_MIN_TOKENS and a run prints no FLOOR
line. The report still reads a floor a CALLER's report record carries (floor_state / floor_line / the `floor-off` state: every item below that
floor = the stock triangle attention by design, not partial) — exercised here on synthetic records."""
import pytest

import protenix_v1_opt
from protenix_v1_opt.tests._ditfast_account import DF_OK as _DF_OK
from protenix_v1_opt import kit as K, modes, report as R, stack

FAST = modes.resolve("fast"); EXACT = modes.resolve("exact"); OFF = modes.resolve("off")


def test_no_mode_carries_a_tri_attention_floor():
    assert modes.TRIATTN_FLOOR is None                                                    # the kit has no flash triangle-attention size floor
    for m, km in modes.KIT_MODES.items():
        res = modes.resolve(m)
        assert res.triattn_floor is None and km.triattn_floor is None, m


def test_a_floor_on_an_arm_without_gflash_is_refused(monkeypatch):
    monkeypatch.setitem(modes.KIT_MODES, "fast", modes.KitMode("fast", "fast+gflash+ttr+sg+hoist", "T2+hoist", 2, "x"))
    assert modes.resolve("fast").triattn_floor is None                                    # gflash without a floor: the kit's state
    monkeypatch.setitem(modes.KIT_MODES, "exact", modes.KitMode("exact", "exact+sg+hoist", "T1+hoist", 1, "x", triattn_floor=5))
    with pytest.raises(RuntimeError, match="has no gflash but triattn_floor=5"):
        modes.resolve("exact")


def test_nothing_is_exported_as_the_floor_variable():
    for res in (FAST, EXACT, OFF, modes.resolve("big")):
        env = {}
        assert stack.export_triattn_floor(res, env) is None and env == {}                 # no PTX_TRIATTN_MIN_TOKENS from the kit
    assert K.TRIATTN_FLOOR_ENV == "PTX_TRIATTN_MIN_TOKENS"
    rep = {"mode": "fast", "arm": FAST.arm, "cfg": {"trimul": "fast", "gflash": True}, "levers": {}, "n_gpu": 1, "triattn_floor": stack.export_triattn_floor(FAST, {})}
    assert "smalln=" not in R.activation_line(rep)                                        # the ACTIVE line of a real run carries no small-N token


def _fast_account(fused, gated):
    return {"cfg": {"trimul": "fast", "gflash": True, "ttr": True, "sg": True, "hoist": True},
            "counts": {"dit": {"apb:fp16": 48}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "pf": {"t2:pf@9.0": 96}, "opm": {"t2:opm@9.0": 8}, "pwa": {"t2:pwa@9.0": 6}, "atom": {"apb:tf32rn": 12}, "trimul": {"fast": 40}, "triattn": {"gflash": fused, "stock:gate": gated}, "transition": {"ttr:C=128": 40}},
            "keep_pool": {"installed": True, "skipped_total": 3, "passed_total": 1, "errors": 0}, "summary_hostidx": {"installed": True, "samples": 5, "delegated": 0, "errors": 0}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 3.9, "patched": 12}, "ditfast": _DF_OK, "templ": {"template_dedupe": {"installed": True, "calls": 1, "evaluated": 2, "reused": 2, "stock": {}}, "tmpl_triatt": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_trimul_exact": {"on": True, "routed_total": 8, "stock": {}}, "tmpl_xtr": {"on": True, "routed_total": 4, "stock": {}, "fallback": {}}, "tmpl_pairfused": {"on": True, "calls": 8}}, "dit_attn_exact": {"installed": True, "calls": 72, "routes": {"kernel": 72}}, "apb": {"dit": {"engaged": True, "fp16": True, "opd": "fp16", "cell_key": "9.0", "installed_on": 24, "calls": 48}, "atom": {"engaged": True, "opd": "tf32rn", "cell_key": "9.0", "installed_on": 6, "calls": 12}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "trunk2": {"pf": {"engaged": True, "cell_key": "9.0", "installed_on": 52, "calls": 96}, "opm": {"engaged": True, "cell_key": "9.0", "installed_on": 4, "calls": 8}, "pwa": {"engaged": True, "cell_key": "9.0", "installed_on": 3, "calls": 6}}, "sampler": {"graphs": True, "prep": {"on": True, "parts": "rot_async+keycheck+warmup1+poison_once+pool_chain+stepvec", "poison": "ok", "stats": {}, "aside": None}, "hoist_installed": True, "sampler": {"replays": 100}, "hoist": {"hits": 10, "records": 1}}}


@pytest.mark.parametrize("fused,gated,floor,items,state", [
    (96, 0, 300, [{"N_token": 356}], "served"),
    (0, 96, 300, [{"N_token": 185}, {"N_token": 239}], "floor-off"),            # every item below the floor: the stock triangle attention by design
    (0, 96, 300, [{"N_token": 185}, {"N_token": 300}], "partial"),              # an item at the floor got no fused call
    (0, 96, 300, [{"N_token": 185}, {}], "partial"),                            # an unsized item cannot be excused
    (0, 96, 128, [{"N_token": 185}], "partial"),                                # a caller's lower floor: 185 is at/above it
])
def test_floor_state(fused, gated, floor, items, state):
    rep = {"levers": _fast_account(fused, gated), "items": items, "triattn_floor": {"mode": 300, "exported": floor, "source": "mode"}}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    f = v["evidence"]["gflash"]["floor"]
    assert f["state"] == state and f["min_tokens"] == floor
    assert ("gflash" in v["partial"]) == (state == "partial")
    assert v["exit_code"] == (R.EXIT_NOT_ACTIVE if state == "partial" else R.EXIT_OK)


def test_floor_off_line(capsys):
    rep = {"levers": _fast_account(0, 96), "items": [{"N_token": 185}, {"N_token": 239}], "triattn_floor": {"mode": 300, "exported": 300, "source": "mode"}}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK
    assert R.floor_line(v["evidence"]) == "[protenix-v1-opt] FLOOR gflash min_tokens=300 items_atleast=0 items_below=2 served=0 state=floor-off"
    R.log_exit_lines(v)
    err = capsys.readouterr().err.strip().splitlines()
    assert err[:2] == ["[protenix-v1-opt] FLOOR gflash min_tokens=300 items_atleast=0 items_below=2 served=0 state=floor-off",
                       "[protenix-v1-opt] NOTE gflash inactive by size: every item is below the flash triangle-attention floor (N_token < 300, items_below=2); the fused block served 0 calls around the stock cuEquivariance attention core (gflash:cueq<floor) — a served route, not a partial activation"]
    assert all(" LEVER " in l for l in err[3:-2]) and any("state=on impl=opt_core/attn/pair_fused.py origin=core strategy=F1.flash_triatt served=0 gated=96 gated_by=stock:gate:96 min_tokens=300 floor=floor-off lever=gflash" in l for l in err)
    assert err[-2].startswith(f"{R.PREFIX} TEMPLATE event=exit templates=") and err[-1].startswith(f"{R.PREFIX} ITEMS total=")   # the template record, then the item census close the exit lines


def test_an_unknown_floor_has_no_floor_block_and_no_excuse():
    rep = {"levers": _fast_account(0, 96), "items": [{"N_token": 185}]}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    assert v["evidence"]["gflash"]["floor"] is None and v["partial"] == ["gflash"] and R.floor_line(v["evidence"]) is None


# ------------------------------------------------------------------------------------------------ below the floor: a served route named up front, exit 0
def _below_floor_account():
    """The kit's account of a run whose every item is below the floor: the fused block served every triangle-attention call around the
    stock cuEquivariance core (`gflash:cueq<floor`), no fused-core call."""
    acc = _fast_account(0, 0); acc["counts"]["triattn"] = {"gflash:cueq<floor": 1040, "stock:c=64": 160}
    return acc


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_below_the_floor_is_a_named_served_route_exit_0(mode, capsys):
    res = modes.resolve(mode)
    rep = {"mode": mode, "arm": res.arm, "cfg": {"trimul": "fast", "gflash": True, "ttr": True, "sg": True, "hoist": True},
           "levers": _below_floor_account(), "items": [{"N_token": 200}], "triattn_floor": {"mode": 300, "exported": 300, "source": "mode"}, "n_gpu": 1}
    assert R.activation_line(rep).endswith(" smalln=gflash:cueq_core(N<300)")             # named up front on the ACTIVE line (one whitespace-free token at the end)
    v = R.verdict(rep, "fast", res.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK                              # not a partial activation: exit 0
    g = v["evidence"]["gflash"]
    assert g["served"] == 1040 and g["fallback"] == {} and g["floor"]["state"] == "floor-off" and g["floor"]["items_below"] == 1
    R.log_exit_lines(v, rep)
    err = capsys.readouterr().err
    assert "[protenix-v1-opt] FLOOR gflash min_tokens=300 items_atleast=0 items_below=1 served=1040 state=floor-off" in err   # served = the block's calls; the state reads the fused core's (0)
    assert ("[protenix-v1-opt] NOTE gflash inactive by size: every item is below the flash triangle-attention floor (N_token < 300, items_below=1); "
            "the fused block served 1040 calls around the stock cuEquivariance attention core (gflash:cueq<floor) — a served route, not a partial activation") in err
    assert "PARTIAL" not in err and "NOT ACTIVE" not in err


def test_at_or_above_the_floor_nothing_changes_but_the_static_token(capsys):
    """>= the floor: the fused core serves, state `served`, no NOTE; the evidence record is the served form; exact (no gflash) carries no token."""
    rep = {"mode": "fast", "arm": FAST.arm, "cfg": {"trimul": "fast", "gflash": True}, "levers": _fast_account(1040, 0), "items": [{"N_token": 400}],
           "triattn_floor": {"mode": 300, "exported": 300, "source": "mode"}, "n_gpu": 1}
    v = R.verdict(rep, "fast", FAST.levers, allow_partial=False)
    assert v["partial"] == [] and v["exit_code"] == R.EXIT_OK and v["evidence"]["gflash"]["floor"]["state"] == "served"
    assert R.floor_note(v["evidence"]) is None
    R.log_exit_lines(v, rep)
    assert " NOTE " not in capsys.readouterr().err
    assert R.activation_line(rep).endswith(" smalln=gflash:cueq_core(N<300)")             # the token states the rule, not the item: present at every size for a gflash mode
    exact = {"mode": "exact", "arm": modes.resolve("exact").arm, "cfg": {"trimul": "exact", "gblock": True, "xtr": True}, "levers": {}, "n_gpu": 1, "triattn_floor": None}
    assert "smalln=" not in R.activation_line(exact)                                        # exact carries no flash core: its ACTIVE line is unchanged
