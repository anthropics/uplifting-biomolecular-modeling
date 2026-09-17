"""The environment's uncertainty is NAMED on the activation line and never switches a lever off (report.card_support_field over
opt_core.arch.supports / registry.declare_arch): a GPU class a requested lever has no test record on prints
`card_support=uncertified:<sm>(<levers>)` after the n_gpu fields; a class every requested lever is tested on (sm90; the A100-tested levers
on sm80) prints nothing, so those ACTIVE / DRY-RUN lines are byte-unchanged."""
import os

from openfold3_opt import modes, report

HOME = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
H100 = {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "sm": "sm90", "memory_mib": 81559}
A100 = {"name": "NVIDIA A100-SXM4-80GB", "cc": "8.0", "sm": "sm80", "memory_mib": 81920}
L40S = {"name": "NVIDIA L40S", "cc": "8.9", "sm": "sm89", "memory_mib": 46068}


def _rep(mode, gpu, n_tokens=612, n_gpu=None, notes=None):
    env = {"PATH": "/bin"}
    if n_gpu:
        env.update({"OPENFOLD3_OPT_N_GPU": str(n_gpu), "OF3TP_RANK": "0", "OF3TP_WORLD": str(n_gpu), "CUDA_VISIBLE_DEVICES": "0"})
    res = modes.resolve(mode, HOME, environ=env, n_tokens=n_tokens, n_gpu=n_gpu)
    rep = {"active": True, "mode": mode, "line": res.line, "line_spelling": modes.describe_line(res), "openfold3_version": "0.4.1", "gpu": gpu,
           "levers_requested": list(res.levers), "hooks_spelling": ">".join(modes.hook_spellings(res)), "n_gpu": n_gpu or 1, "n_tokens": n_tokens}
    if notes:
        rep["notes"] = list(notes)
    return rep


def test_a_tested_class_prints_no_card_support_field():
    for mode, n_tokens, n_gpu in (("exact", 612, None), ("fast", 612, None), ("big", 1012, None), ("big", 2565, None), ("big", 2565, 2)):
        rep = _rep(mode, H100, n_tokens, n_gpu)
        assert report.card_support_field(rep) == "" and "card_support=" not in report.activation_line(rep), (mode, n_tokens, n_gpu)
    for mode, untested in (("exact", ("writer_overlap", "hostfeat", "ckpt_mmap")), ("fast", ("trimul_provider", "writer_overlap", "hostfeat", "ckpt_mmap"))):   # on sm80 these lines name exactly their levers without an A100 record (registry.A100_TESTED)
        field = report.card_support_field(_rep(mode, A100))
        assert field.startswith(" card_support=uncertified:sm80(") and sorted(field.split("(", 1)[1].rstrip(")").split(",")) == sorted(untested), (mode, field)
    assert report.card_support_field(_rep("fast", None)) == "" and report.card_support_field(_rep("fast", {"name": None})) == ""   # no GPU readable: nothing to name


def test_an_uncertified_class_is_named_after_the_ngpu_fields_and_switches_nothing_off():
    rep = _rep("big", A100, 2565, notes=["a note"])
    field = report.card_support_field(rep)
    assert field.startswith(" card_support=uncertified:sm80(") and "confhead" in field and "fast_init" not in field    # fast_init is A100-tested; the resident levers are not
    line = report.activation_line(rep)
    assert " n_gpu=1 sharding=none compile=none card_support=uncertified:sm80(" in line and line.index("card_support=") < line.index(" notes=a note")
    assert rep["levers_requested"] == list(modes.resolve("big", HOME, environ={"PATH": "/bin"}, n_tokens=2565).levers)   # the requested lever set is the line's, card or no card
    rep = _rep("fast", L40S)
    names = report.card_support_field(rep)
    assert names.startswith(" card_support=uncertified:sm89(") and all(l in names for l in rep["levers_requested"])           # no record on sm89 for any lever: all named, all requested
    dry = dict(rep, active=False, dry_run=True, env={"A": "1"})
    assert " n_gpu=1 sharding=none compile=none card_support=uncertified:sm89(" in report.activation_line(dry)                            # the DRY-RUN line (run.sh check) names it before any run
