"""The template guard (templ_census + cli.exit_rule): a query that DECLARED templates and reached the model with an empty featurised template
stack is a NAMED event — a TEMPLATES DROPPED line, the exit census word fallbacks=templates_dropped:<reason>=<k>, exit 5 — unless
--allow-template-drop (exit 0, the word kept); a query with real template slots, or one that declared none, is untouched."""
import importlib
import os
import sys

import pytest

Q_TEMPL = {"queries": {"q1": {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AC", "template_alignment_file_path": "/t/a.m8"}]},
                       "q2": {"chains": [{"molecule_type": "protein", "chain_ids": ["A"], "sequence": "AC"}]}}}


def _live():
    """The package modules as sys.modules holds them NOW (other tests re-import the package: a module object imported at collection time may be
    stale, and the guard's census reads sys.modules)."""
    return tuple(importlib.import_module(f"openfold3_opt.{n}") for n in ("cli", "report", "templ_census"))


@pytest.fixture(autouse=True)
def _fresh():
    global cli, report, templ_census
    cli, report, templ_census = _live()
    templ_census.reset()
    yield
    templ_census.reset()


cli = report = templ_census = None


class _Mask:
    """A stand-in for a [batch, slots, tokens] template mask tensor: `real` slots carry ones, the rest zeros (numpy when present)."""
    def __new__(cls, slots, tokens, real):
        np = pytest.importorskip("numpy")
        m = np.zeros((1, slots, tokens), dtype=float)
        m[:, :real, :] = 1.0
        return m


def test_declared_templated_and_dummy_only_is_dropped_by_name():                       # (i)
    templ_census.record(Q_TEMPL, stream=sys.stderr)
    assert templ_census.declared() == {"q1": 1}
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})
    templ_census.observe("q2", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})
    drops = templ_census.judge(templ_census.declared(), templ_census.featurised())
    assert [(d["query"], d["chains_templated"], d["slots"], d["real_slots"], d["reason"]) for d in drops] == [("q1", 1, 4, 0, "dummy_only")]
    line = templ_census.drop_lines(drops)[0]
    assert line.startswith("[openfold3-opt] TEMPLATES DROPPED: query=q1 declared chains_templated=1 real_slots=0 slots=4 (")
    assert templ_census.fallbacks() == {"dummy_only": 1} and "templates_dropped:dummy_only=1" in report.fallback_word()
    code, gate = cli.exit_rule("pred --mode off", 0, 5, 5, False, None, dropped=len(drops), allow_template_drop=False)
    assert code == cli.EXIT_TEMPLATES_DROPPED == templ_census.EXIT_TEMPLATES_DROPPED == 5
    assert gate["templates_dropped"] == 1 and gate["allow_template_drop"] is False and "TEMPLATES DROPPED" in gate["reason"]


def test_real_slots_are_untouched():                                                   # (ii)
    templ_census.record(Q_TEMPL)
    templ_census.observe("q1", {"template_backbone_frame_mask": _Mask(4, 7, 1)})      # one real slot (the second mask key serves too)
    assert templ_census.judge(templ_census.declared(), templ_census.featurised()) == []
    assert templ_census.fallbacks() == {} and "templates_dropped" not in report.fallback_word()
    assert cli.exit_rule("pred --mode exact", 0, 5, 5, False, True, dropped=0)[0] == cli.EXIT_OK


def test_an_untemplated_query_is_untouched_even_with_an_empty_stack():                 # (iii)
    templ_census.record({"queries": {"q2": Q_TEMPL["queries"]["q2"]}})
    assert templ_census.declared() == {}
    templ_census.observe("q2", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})        # upstream always allocates the slots; none real, none declared
    assert templ_census.judge(templ_census.declared(), templ_census.featurised()) == [] and templ_census.fallbacks() == {}


def test_the_opt_out_exits_zero_and_keeps_the_word():                                  # (iv)
    templ_census.record(Q_TEMPL)
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})
    drops = templ_census.judge(templ_census.declared(), templ_census.featurised())
    code, gate = cli.exit_rule("pred --mode off", 0, 5, 5, False, None, dropped=len(drops), allow_template_drop=True)
    assert code == cli.EXIT_OK and gate["templates_dropped"] == 1 and gate["allow_template_drop"] is True and "--allow-template-drop" in gate["reason"]
    assert "templates_dropped:dummy_only=1" in report.fallback_word()                 # the census word stays under the opt-out
    a = cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "/tmp/o", "--allow-template-drop"])
    assert a.allow_template_drop is True and cli.build_parser().parse_args(["pred", "--mode", "off", "--query-json", "q.json", "--output-dir", "/tmp/o"]).allow_template_drop is False


def test_a_declared_query_the_wrap_never_saw_is_dropped_as_unobserved():
    templ_census.record(Q_TEMPL)
    drops = templ_census.judge(templ_census.declared(), {})
    assert [(d["query"], d["reason"]) for d in drops] == [("q1", "unobserved")] and templ_census.fallbacks() == {"unobserved": 1}
    assert "slots=unknown" in templ_census.drop_lines(drops)[0]


def test_precedence_runner_rc_and_incomplete_come_first():
    assert cli.exit_rule("pred", 1, 5, 5, False, None, dropped=1)[0] == 1                      # the runner's own rc first
    assert cli.exit_rule("pred", 0, 3, 5, False, None, dropped=1)[0] == cli.EXIT_FAIL          # incomplete before the guard
    assert cli.exit_rule("pred", 0, 5, 5, True, None, dropped=1)[0] == cli.EXIT_TEMPLATES_DROPPED   # the guard before the partial rule


def test_the_forward_wrap_observes_and_the_stock_proof_hands_over(monkeypatch):
    """report.wrap_forward_timer records the batch's template slots (templ_census.observe) in whichever process runs the model; the primary's
    cli.templates_guard judges this process's record on a kit route and the stock child's proof record on the stock route."""
    class M:
        def forward(self, batch):
            return 0
    report.wrap_forward_timer(M, "kit")
    templ_census.record(Q_TEMPL)
    M().forward({"query_id": ["q1"], "seed": 42, "template_pseudo_beta_mask": _Mask(4, 5, 0)})
    assert templ_census.featurised()["q1"] == {"slots": 4, "real_slots": 0, "key": "template_pseudo_beta_mask"}
    g = cli.templates_guard(0)                                                                       # a kit route: this process's record
    assert [d["query"] for d in g["dropped"]] == ["q1"] and g["judged"] is True
    g = cli.templates_guard(0, {"templates_featurised": {"q1": {"slots": 4, "real_slots": 2, "key": "template_pseudo_beta_mask"}}})   # the stock route: the child's proof
    assert g["dropped"] == [] and g["featurised"]["q1"]["real_slots"] == 2
    assert cli.templates_guard(1)["judged"] is False and cli.templates_guard(1)["dropped"] == []     # a failed run is not judged (the exit rule's first clause names it)


def test_a_declared_query_whose_preprocessor_parse_hook_never_ran_is_still_judged_from_the_query():
    """The declared side is the QUERY JSON (templ_census.record), not the preprocessor: when upstream abandons the alignment before its parse
    hook runs (templ_guard's `TEMPLATES PARSED` census then never counts the chain), the featurised stack is empty and the guard still names it."""
    templ_guard = importlib.import_module("openfold3_opt.templ_guard")
    before = dict(templ_guard.CENSUS)
    templ_census.record(Q_TEMPL)                                                          # declared: read from the query set
    assert templ_guard.CENSUS == before                                                   # the parse hook never ran: its census is untouched
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})            # the featuriser filled nothing real
    drops = templ_census.judge(templ_census.declared(), templ_census.featurised())
    assert [(d["query"], d["reason"]) for d in drops] == [("q1", "dummy_only")]
    code, gate = cli.exit_rule("pred --mode exact", 0, 5, 5, False, True, dropped=len(drops), allow_template_drop=False)
    assert code == 5 and templ_census.drop_lines(drops)[0].startswith("[openfold3-opt] TEMPLATES DROPPED: query=q1 declared chains_templated=1 real_slots=0 slots=4 (")


def test_the_tp_ranks_carry_the_opt_out(tmp_path):
    """`--allow-template-drop` reaches every rank's argv (tp.launch composes it; each rank judges its own guard)."""
    src = open(importlib.import_module("openfold3_opt.tp").__file__, encoding="utf-8").read()
    assert 'sub.append("--allow-template-drop")' in src


def test_a_templated_query_that_kept_real_slots_prints_the_featurised_line(capsys):
    """Success is loud too: one `TEMPLATES FEATURISED query=<q> real_slots=<r>/<T>` per declared-templated query with a real slot; nothing for an
    untemplated query; the exit rule then has nothing to gate (exit 0, no census word)."""
    templ_census.record(Q_TEMPL)
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 2)})            # 2 of 4 slots real
    templ_census.observe("q2", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})            # untemplated query: never a line
    lines = templ_census.featurised_lines(templ_census.declared(), templ_census.featurised())
    assert lines == ["[openfold3-opt] TEMPLATES FEATURISED query=q1 real_slots=2/4"]
    assert templ_census.judge(templ_census.declared(), templ_census.featurised()) == [] and templ_census.fallbacks() == {}
    rec = cli.templates_guard(0)                                                          # the in-process route's guard: prints the line through _err
    err = capsys.readouterr().err
    assert err.count("TEMPLATES FEATURISED query=q1 real_slots=2/4") == 1 and "TEMPLATES DROPPED" not in err and rec["dropped"] == [] and rec["judged"]
    code, gate = cli.exit_rule("pred --mode exact", 0, 5, 5, False, True, dropped=0, allow_template_drop=False)
    assert code == 0 and gate["templates_dropped"] == 0


def test_templates_guard_on_the_stock_route_judges_the_childs_proof(capsys):
    """`--mode off`: the parent declared (record), the stock child featurised (its proof's templates_featurised); the parent judges and prints."""
    templ_census.record(Q_TEMPL)
    rec = cli.templates_guard(0, {"templates_featurised": {"q1": {"slots": 4, "real_slots": 0, "key": "template_pseudo_beta_mask"}}})
    err = capsys.readouterr().err
    assert [d["reason"] for d in rec["dropped"]] == ["dummy_only"] and err.count("[openfold3-opt] TEMPLATES DROPPED: query=q1 declared chains_templated=1 real_slots=0 slots=4 (") == 1
    assert templ_census.fallbacks() == {"dummy_only": 1}
    templ_census.reset(); templ_census.record(Q_TEMPL)
    rec = cli.templates_guard(0, {"templates_featurised": {"q1": {"slots": 4, "real_slots": 3, "key": "template_pseudo_beta_mask"}}})
    assert rec["dropped"] == [] and capsys.readouterr().err.count("TEMPLATES FEATURISED query=q1 real_slots=3/4") == 1
    rec = cli.templates_guard(1, {"templates_featurised": {}})                            # a failed run is not judged (the exit rule's first clause owns it)
    assert rec["judged"] is False and rec["dropped"] == []


def test_templates_disabled_makes_template_paths_a_non_workload(capsys):
    """`--use-templates false` (record enabled=False): ONE `TEMPLATES DISABLED (--use-templates false): queries_with_template_paths=<n>` info line,
    no declared-templated query — nothing judged, no FEATURISED / DROPPED line, no census fallback word, the exit rule unaffected (rc 0) —
    whatever the featurised stack holds."""
    templ_census.record(Q_TEMPL, enabled=False)
    err = capsys.readouterr().err
    assert err.strip() == "[openfold3-opt] TEMPLATES DISABLED (--use-templates false): queries_with_template_paths=1"
    assert templ_census.declared() == {} and templ_census.counters() == {"templ_declared": 0, "templ_disabled_paths": 1}
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})            # an empty stack under templates-off: not a drop
    rec = cli.templates_guard(0)
    err = capsys.readouterr().err
    assert rec["dropped"] == [] and "TEMPLATES DROPPED" not in err and "TEMPLATES FEATURISED" not in err
    assert templ_census.fallbacks() == {} and "templates_dropped" not in report.fallback_word()
    code, gate = cli.exit_rule("pred --mode fast", 0, 5, 5, False, True, dropped=len(rec["dropped"]), allow_template_drop=False)
    assert code == 0 and gate["templates_dropped"] == 0
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 4)})            # even a full stack prints no FEATURISED line under templates-off
    assert templ_census.featurised_lines(templ_census.declared(), templ_census.featurised()) == []


def test_templates_enabled_keeps_the_v2_behaviour(capsys):
    """record(enabled=True) (the default): DECLARED line; an empty stack is DROPPED + rc 5, a real slot is FEATURISED."""
    templ_census.record(Q_TEMPL, enabled=True)
    assert capsys.readouterr().err.startswith("[openfold3-opt] TEMPLATES DECLARED queries=2 chains_templated=1 chains_untemplated=1 files_resolved=0/1")
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 0)})
    rec = cli.templates_guard(0)
    assert [d["reason"] for d in rec["dropped"]] == ["dummy_only"] and "TEMPLATES DROPPED: query=q1" in capsys.readouterr().err
    code, _ = cli.exit_rule("pred --mode fast", 0, 5, 5, False, True, dropped=len(rec["dropped"]), allow_template_drop=False)
    assert code == 5


def test_cmd_pred_hands_the_use_templates_switch_to_the_census():
    src = open(cli.__file__, encoding="utf-8").read()
    assert 'templ_census.record_file(a.query_json, enabled=effective_use_templates(a) == "true")' in src


def test_effective_use_templates_follows_upstreams_default():
    """No `--use-templates` = upstream's default (true: templates enabled, the guard judges); `false` disables; case-insensitive."""
    class A: pass
    a = A(); a.use_templates = None
    assert cli.effective_use_templates(a) == "true"
    a.use_templates = "False"
    assert cli.effective_use_templates(a) == "false"
    a.use_templates = "true"
    assert cli.effective_use_templates(a) == "true"


def test_featurised_line_names_the_row_born_form_on_the_tp_line():
    """On the tp line (`big --n_gpu P`) an item whose real template slots are computed per rank from per-token precursors carries the flag
    `_lazy_template_real` non-zero (tp_rowpair.data.REAL_FLAG): its FEATURISED line ends ` form=row_born`; a zero flag (an untemplated item under
    the consume policy) or no flag (every one-GPU route) prints the line exactly as before."""
    from openfold3_opt import templ_census
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tp_rowpair", "data.py"), encoding="utf-8").read()
    assert f'REAL_FLAG = "{templ_census.ROW_BORN_FLAG}"' in src                          # the flag the tp line's featurizer writes (read as text: no torch import here)
    templ_census.reset(); templ_census.record({"queries": {"q1": Q_TEMPL["queries"]["q1"]}})
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 2), templ_census.ROW_BORN_FLAG: _Mask(1, 1, 1)})
    assert templ_census.featurised_lines(templ_census.declared(), templ_census.featurised()) == ["[openfold3-opt] TEMPLATES FEATURISED query=q1 real_slots=2/4 form=row_born"]
    templ_census.reset(); templ_census.record({"queries": {"q1": Q_TEMPL["queries"]["q1"]}})
    templ_census.observe("q1", {"template_pseudo_beta_mask": _Mask(4, 7, 2), templ_census.ROW_BORN_FLAG: _Mask(1, 1, 0)})
    assert templ_census.featurised_lines(templ_census.declared(), templ_census.featurised()) == ["[openfold3-opt] TEMPLATES FEATURISED query=q1 real_slots=2/4"]


def _asym(tokens_per_chain):
    np = pytest.importorskip("numpy")
    return np.concatenate([np.full((n,), i + 1) for i, n in enumerate(tokens_per_chain)])[None, :]     # [1, tokens]: upstream's asym_id, one id per chain instance


def test_a_declared_chain_left_untemplated_is_dropped_by_name_per_chain(capsys):
    """Two templated instances declared (one entry, chain_ids A,B); the featurised stack keeps a real slot whose entries cover chain 1's tokens
    only: the query kept real slots AND a declared chain reached the model untemplated — dropped by name (chain_untemplated), FEATURISED still
    printed, exit 5 under the same opt-out. With both instances covered nothing is dropped; without asym_id in the batch nothing is judged per chain."""
    np = pytest.importorskip("numpy")
    q = {"queries": {"q1": {"chains": [{"molecule_type": "protein", "sequence": "AC", "chain_ids": ["A", "B"], "template_alignment_file_path": "/x/a.m8"},
                                       {"molecule_type": "ligand", "smiles": "C", "chain_ids": ["L"]}]}}}
    templ_census.record(q, stream=open(os.devnull, "w"))
    assert templ_census.declared() == {"q1": 1} and templ_census.declared_instances() == {"q1": 2}
    m = np.zeros((1, 4, 9)); m[:, 0, :4] = 1.0                                                              # slot 0 real on chain 1's four tokens only (chain 2: tokens 4-7, ligand: token 8)
    rec = templ_census.observe("q1", {"template_pseudo_beta_mask": m, "asym_id": _asym([4, 4, 1])})
    assert (rec["real_slots"], rec["chains_real"], rec["chains"]) == (1, 1, 3)
    drops = templ_census.judge(templ_census.declared(), templ_census.featurised())
    assert [(d["reason"], d["chain_instances"], d["chains_real"], d["real_slots"]) for d in drops] == [("chain_untemplated", 2, 1, 1)]
    line = templ_census.drop_lines(drops)[0]
    assert line.startswith(f"[{report.TAG}] TEMPLATES DROPPED: query=q1 declared chains_templated=1 chain_instances=2 chains_real=1 real_slots=1 slots=4 ("), line
    assert templ_census.featurised_lines(templ_census.declared(), templ_census.featurised()) == [f"[{report.TAG}] TEMPLATES FEATURISED query=q1 real_slots=1/4"]
    assert templ_census.fallbacks() == {"chain_untemplated": 1} and "templates_dropped:chain_untemplated=1" in report.fallback_word()
    code, gate = cli.exit_rule("pred --mode off", 0, 5, 5, False, None, dropped=len(drops), allow_template_drop=False)
    assert code == cli.EXIT_TEMPLATES_DROPPED
    m2 = np.zeros((1, 4, 9)); m2[:, 0, :8] = 1.0                                                            # both instances carry real entries: nothing dropped
    templ_census.observe("q1", {"template_pseudo_beta_mask": m2, "asym_id": _asym([4, 4, 1])})
    assert templ_census.judge(templ_census.declared(), templ_census.featurised()) == []
    templ_census.reset(); templ_census.record(q, stream=open(os.devnull, "w"))
    assert "chains_real" not in templ_census.observe("q1", {"template_pseudo_beta_mask": m})               # no asym_id in the batch: no per-chain record, the per-query rule alone
    assert templ_census.judge(templ_census.declared(), templ_census.featurised()) == []
