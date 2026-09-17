"""The modes' resolved lines are byte-identical to the recorded spellings (tests/lines_record.json: the `line=...: K=V ... @ hooks` spelling of the
ACTIVE line, the lever list, hooks, exports, unsets and gate words per case) — exact (cueq), fast, big resident above and below the confidence
gate, below the offload port's item gate (two sizes), and with the token count unknown, and big at --n_gpu 2 (tp, rank 0 of a 2-wide group)."""
import json
import os

from openfold3_ob0_opt import modes
from openfold3_ob0_opt.tests import _stubs

HOME = _stubs.tree_home()
CASES = {"big_resident_2565": dict(mode="big", environ={}, n_tokens=2565),
         "big_resident_1012": dict(mode="big", environ={}, n_tokens=1012),                  # below the offload port's item gate (OF3O_MIN_TOKENS, modes.OF3O_GATE_BY_CARD): the fast line as composed, graphs included
         "big_resident_612": dict(mode="big", environ={}, n_tokens=612),                    #  (two sizes below it: one spelling)
         "big_resident_unknown": dict(mode="big", environ={}, n_tokens=None),
         "big_tp_rank0_w2_2565": dict(mode="big", environ={"OF3TP_RANK": "0", "OF3TP_WORLD": "2", "OPENFOLD3_OB0_OPT_N_GPU": "2"}, n_tokens=2565, n_gpu=2),
         "exact_cueq_612": dict(mode="exact", environ={}, n_tokens=612),
         "fast_612": dict(mode="fast", environ={}, n_tokens=612)}


def _render(res):
    norm = lambda v: v.replace(HOME, "<HOME>") if isinstance(v, str) else v          # noqa: E731
    return {"line": res.line, "tier": res.tier, "spelling": norm(modes.describe_line(res)), "levers": list(res.levers), "hooks": list(res.hooks),
            "exports": {k: norm(v) for k, v in res.exports.items()}, "unsets": list(res.unsets), "conf_gate": res.conf_gate, "size_gate": res.size_gate,
            "conflicts": list(res.conflicts)}


def _resolve(c):
    return modes.resolve(c["mode"], HOME, environ=dict(c["environ"]), n_tokens=c["n_tokens"], n_gpu=c.get("n_gpu"))


def test_every_lines_spelling_levers_and_switches_equal_the_record():
    record = json.load(open(os.path.join(os.path.dirname(__file__), "lines_record.json"), encoding="utf-8"))
    assert set(record) == set(CASES)
    for name, c in CASES.items():
        got = _render(_resolve(c))
        assert got == record[name], (name, got, record[name])


if __name__ == "__main__":                                                  # regenerate: python -m openfold3_ob0_opt.tests.test_lines_unchanged > lines_record.json
    print(json.dumps({name: _render(_resolve(c)) for name, c in CASES.items()}, indent=1, sort_keys=True))
