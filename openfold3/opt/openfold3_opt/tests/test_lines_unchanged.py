"""The modes' resolved lines are byte-identical to the recorded spellings (tests/lines_record.json: the `line=...: K=V ... @ hooks` spelling of the
ACTIVE line, the lever list, hooks, exports, unsets and gate words per case) — the line selection by GPU count changes no resolved byte of
exact (above and within its graph cap), fast, big (resident, above and below the confidence gate, token count unknown) or big at --n_gpu 2 (tp, a rank of a 2-wide group)."""
import json
import os

from openfold3_opt import modes
from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
CASES = {"big_resident_2565": dict(mode="big", environ={}, n_tokens=2565),
         "big_resident_1012": dict(mode="big", environ={}, n_tokens=1012),                  # below the offload port's item gate (OF3O_MIN_TOKENS, modes.OF3O_GATE_BY_CARD): the fast line as composed, graphs included
         "big_resident_1500": dict(mode="big", environ={}, n_tokens=1500),                  # the port engaged, below the confidence gate (2048): no confhead pair
         "big_resident_unknown": dict(mode="big", environ={}, n_tokens=None),
         "big_tp_rank0_w2_2565": dict(mode="big", environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2"}, n_tokens=2565, n_gpu="2"),
         "exact": dict(mode="exact", environ={}, n_tokens=612),                                # above the exact line's own 512-token graph cap: eager
         "exact_400": dict(mode="exact", environ={}, n_tokens=400),                            # within it: the graphed exact line
         "fast_612": dict(mode="fast", environ={}, n_tokens=612)}


def _render(res):
    norm = lambda v: v.replace(HOME, "<HOME>") if isinstance(v, str) else v
    return {"line": res.line, "tier": res.tier, "spelling": norm(modes.describe_line(res)), "levers": list(res.levers), "hooks": list(res.hooks),
            "exports": {k: norm(v) for k, v in res.exports.items()}, "unsets": list(res.unsets), "conf_gate": res.conf_gate, "size_gate": res.size_gate,
            "conflicts": list(res.conflicts)}


def test_every_lines_spelling_levers_and_switches_equal_the_record():
    record = json.load(open(os.path.join(os.path.dirname(__file__), "lines_record.json"), encoding="utf-8"))
    assert set(record) == set(CASES)
    for name, c in CASES.items():
        res = modes.resolve(c["mode"], HOME, environ=c["environ"], n_tokens=c["n_tokens"], n_gpu=c.get("n_gpu"))
        assert _render(res) == record[name], (name, _render(res), record[name])
