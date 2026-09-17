"""opt_core.cell_census: the process-wide cell-coverage census of the provider faces.  CPU only.

Pinned here: the registry semantics (once per key, revision on a different fact, never raises), the token grammar (regex + literal bytes),
the environment dump and the exit lines (a subprocess), set_context, the NEAREST rule's hard guards (cross-card, exact-vs-tolerance,
unserved shape, form, dtype: no cell; size neighbour: served), every face's hook (a hit, an inherited bucket, a named fallback, a stock
resolution, an opt-in), and SELECTION IDENTITY: every face decides the same Selection / Refusal whether the census records, raises inside,
or is absent."""
import json
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
CORE_DIR = os.path.dirname(HERE)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

from opt_core import cell_census as C                                  # noqa: E402
from opt_core.kernels import apb, ln, pallas, transition, triattn, triattn_xla, trimul   # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(C.ENV_FILE, raising=False)
    monkeypatch.delenv(C.ENV_PRINT, raising=False)
    C.reset()
    triattn.select_cache_clear()
    yield
    C.reset()
    triattn.select_cache_clear()


KEY = dict(cc="9.0", stack="H100:2.13.0+cu130/3.7.1/cueq0.11.1", dtype="bf16", shape="C128H128", bucket="N=3000", form="out.fwd", word="fast")


# ------------------------------------------------------------------------------------------------------------------------------- registry
def test_record_once_per_key_counts_repeats_and_revises_a_different_fact():
    e1 = C.record("trimul", KEY, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")
    e2 = C.record("trimul", KEY, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")
    assert e1["n"] == 1 and e2["n"] == 2 and e2["revised"] == 0
    assert len(C.table()) == 1
    e3 = C.record("trimul", KEY, "named_fallback", "cueq", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", refused="v4:launch_failed")
    assert len(C.table()) == 1 and e3["revised"] == 1 and e3["outcome"] == "named_fallback" and e3["n"] == 3
    assert C.tokens() == [e3["token"]] and C.alerts() == [e3["token"]]
    C.record("apb", dict(KEY, shape="dit_h16d48"), "stock", "sdpa:auto", cell_id="9.0|bf16|dit_h16d48|S1|N<=800|eager|fwd")
    assert [e["provider"] for e in C.table()] == ["trimul", "apb"]                        # decision order
    assert C.alerts() == [e3["token"]]                                                     # stock cells are not alerts
    C.reset()
    assert C.table() == [] and C.context() == {}


def test_record_never_raises_and_counts_what_it_could_not_record():
    assert C.record("trimul", KEY, "not_an_outcome", "v4") is None
    assert C.record("trimul", None, "cell_hit", None) is not None                          # a None key records with '-' words
    assert C.record(None, object(), "cell_hit", object(), cell_id=3, note=5, refused=7) is not None
    d = json.loads(C.dump())
    assert d["errors"] >= 1 and d["schema"] == C.SCHEMA


def test_set_context_is_carried_into_entries_and_the_dump():
    C.set_context(engine="kitword", mode="fast", card="H100", size=800)
    C.record("ln", dict(KEY, shape="pair_c128"), "cell_hit", "fastln", cell_id="9.0|bf16|pair_c128|N<=800|eager|fwd")
    C.set_context(size=1200)
    C.record("ln", dict(KEY, shape="pair_c256"), "cell_hit", "fastln", cell_id="9.0|bf16|pair_c256|N<=1200|eager|fwd")
    C.set_context(size=None)
    rows = C.table()
    assert rows[0]["context"] == {"engine": "kitword", "mode": "fast", "card": "H100", "size": 800}
    assert rows[1]["context"]["size"] == 1200
    assert "size" not in C.context() and C.context()["engine"] == "kitword"
    d = json.loads(C.dump("json"))
    assert d["context"] == {"engine": "kitword", "mode": "fast", "card": "H100"} and d["counts"]["cell_hit"] == 2
    tsv = C.dump("tsv").splitlines()
    assert tsv[0].split("\t")[:8] == list(C.KEY_FIELDS) and len(tsv) == 3 and "\tkitword\t" in tsv[1]
    with pytest.raises(ValueError):
        C.dump("xml")


# --------------------------------------------------------------------------------------------------------------------------------- tokens
def test_token_grammar_is_pinned():
    C.record("trimul", KEY, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")
    C.record("triattn", dict(cc="9.0", stack="torch2.13-cp311", dtype="bf16", shape="D32H4", bucket="N<=800", form="fwd", word="fast"), "named_fallback", "k2b",
             cell_id="9.0|bf16|D32|H4|N<=800|fwd", refused="triattn_native@v10:no prebuilt for this stack")
    C.record("apb", dict(cc="8.0", dtype="fp32", shape="dit_h16d48", bucket="N<=800", form="eager.S1", word="exact"), "stock", "sdpa:auto", cell_id="8.0|fp32|dit_h16d48|S1|N<=800|eager|fwd")
    C.record("ln", dict(cc="9.0", dtype="bf16w", shape="pair_c128", bucket="N<=400", form="fwd.eager", word="exactln:widen"), "opt_in", "exactln:widen")
    C.record("transition", dict(cc="9.0", dtype="bf16", shape="pair_c128_n4", bucket="N<=800", form="fwd.eager.swiglu", word="fast"), "cell_hit", "v2:fast",
             cell_id="9.0|bf16|pair_c128_n4|N<=800|eager|fwd")
    C.record("pallas", dict(cc="9.0", stack="jax0.7", dtype="bf16", shape="trimul:af2_pair_c128_ch128_outgoing", bucket="N=384", form="fwd", word="fast"), "inherited", "xla",
             cell_id=None, note="family:none(jax line)")
    assert C.tokens() == [
        "UNCOVERED_CELL:trimul|cc9.0|H100:2.13.0+cu130/3.7.1/cueq0.11.1|bf16|C128H128|N=3000|out.fwd word=fast served=v4 nearest=9.0|bf16|C128|H128|N<=2048|out|fwd note=beyond_measured",
        "NAMED_FALLBACK:triattn|cc9.0|torch2.13-cp311|bf16|D32H4|N<=800|fwd word=fast refused=triattn_native@v10:no_prebuilt_for_this_stack served=k2b",
        "STOCK_CELL:apb|cc8.0|-|fp32|dit_h16d48|N<=800|eager.S1 word=exact served=sdpa:auto cell=8.0|fp32|dit_h16d48|S1|N<=800|eager|fwd",
        "OPT_IN:ln|cc9.0|-|bf16w|pair_c128|N<=400|fwd.eager word=exactln:widen served=exactln:widen cell=none",
        "CELL_HIT:transition|cc9.0|-|bf16|pair_c128_n4|N<=800|fwd.eager.swiglu word=fast served=v2:fast cell=9.0|bf16|pair_c128_n4|N<=800|eager|fwd",
        "UNCOVERED_CELL:pallas|cc9.0|jax0.7|bf16|trimul:af2_pair_c128_ch128_outgoing|N=384|fwd word=fast served=xla nearest=none note=family:none(jax_line)",
    ]
    for t in C.tokens():
        m = C.TOKEN_RE.match(t)
        assert m, t
        assert m.group(1) in ("UNCOVERED_CELL", "NAMED_FALLBACK", "CELL_HIT", "STOCK_CELL", "OPT_IN")
    assert C.alerts() == [C.tokens()[0], C.tokens()[1], C.tokens()[5]]
    assert C.TOKEN_WORDS == {"inherited": "UNCOVERED_CELL", "named_fallback": "NAMED_FALLBACK", "cell_hit": "CELL_HIT", "stock": "STOCK_CELL", "opt_in": "OPT_IN"}
    # the dump's key grammar is the token's: the seven segments + word rebuild the token head byte for byte
    for e in C.table():
        head = "%s:%s|cc%s|%s|%s|%s|%s|%s word=%s" % (C.TOKEN_WORDS[e["outcome"]], e["provider"], e["cc"], e["stack"], e["dtype"], e["shape"], e["bucket"], e["form"], e["word"])
        assert e["token"].startswith(head)


def test_clean_and_bucket_words():
    assert C.clean(None) == "-" and C.clean("  ") == "-" and C.clean("a b|c") == "a_b/c" and C.clean("a b|c", keep_bar=True) == "a_b|c"
    assert C.bucket_word(700, "9.0|bf16|C128|H128|N<=800|out|fwd", True) == "N<=800"
    assert C.bucket_word(700, "9.0|bf16|C128|H128|N<=800|out|fwd+of3_module", True) == "N<=800"
    assert C.bucket_word(3000, "9.0|bf16|C128|H128|N<=2048|out|fwd", False) == "N=3000"
    assert C.bucket_word(None, None, True) == "N=-"


# ------------------------------------------------------------------------------------------------------------------ env dump + exit lines
SCRIPT = r"""
import sys
sys.path.insert(0, %(core)r)
from opt_core import cell_census as C
C.set_context(engine="kitword", mode="fast", card="H100", size=800)
k = dict(cc="9.0", stack="s", dtype="bf16", shape="C128H128", bucket="N=3000", form="out.fwd", word="fast")
C.record("trimul", k, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")
C.record("trimul", k, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")   # same key again: nothing printed
C.record("trimul", dict(k, bucket="N<=800"), "cell_hit", "v4", cell_id="9.0|bf16|C128|H128|N<=800|out|fwd")
C.record("apb", dict(k, shape="dit_h16d48"), "named_fallback", "sdpa:auto", cell_id="x", refused="dit_exact:no_prebuilt")
print("body-done", file=sys.stderr)
"""


def _run(tmp_path, env_extra):
    env = dict(os.environ)
    env.pop(C.ENV_FILE, None); env.pop(C.ENV_PRINT, None)
    env.update(env_extra)
    r = subprocess.run([sys.executable, "-c", SCRIPT % {"core": CORE_DIR}], capture_output=True, text=True, env=env, cwd=str(tmp_path), timeout=120)
    assert r.returncode == 0, r.stderr
    return r.stderr.splitlines()


def test_env_dump_file_and_print_once_per_key(tmp_path):
    dump = tmp_path / "cells.json"
    err = _run(tmp_path, {C.ENV_FILE: str(dump), C.ENV_PRINT: "1"})
    body = err[: err.index("body-done")]
    assert body == ["UNCOVERED_CELL:trimul|cc9.0|s|bf16|C128H128|N=3000|out.fwd word=fast served=v4 nearest=9.0|bf16|C128|H128|N<=2048|out|fwd note=beyond_measured",
                    "NAMED_FALLBACK:apb|cc9.0|s|bf16|dit_h16d48|N=3000|out.fwd word=fast refused=dit_exact:no_prebuilt served=sdpa:auto"]   # alerts only, once each, at decision time
    tail = err[err.index("body-done") + 1:]
    assert tail[0].startswith("[opt_core] CELLS pid=") and " keys=3 cell_hit=1 inherited=1 named_fallback=1 stock=0 opt_in=0 alerts=2 dump=%s" % dump in tail[0]
    assert tail[0].endswith(" context=engine:kitword,mode:fast,card:H100,size:800")
    assert tail[1:] == body                                                                   # the exit summary repeats every alert token
    d = json.loads(dump.read_text())
    assert d["schema"] == C.SCHEMA and d["counts"] == {"cell_hit": 1, "inherited": 1, "named_fallback": 1, "stock": 0, "opt_in": 0}
    assert d["alerts"] == body and len(d["entries"]) == 3 and d["entries"][0]["n"] == 2 and d["context"]["engine"] == "kitword"


def test_env_unset_prints_the_exit_lines_only_and_print2_prints_every_key(tmp_path):
    err = _run(tmp_path, {})
    assert err[0] == "body-done" and err[1].startswith("[opt_core] CELLS pid=") and " dump=none " in err[1] and len(err) == 4
    err2 = _run(tmp_path, {C.ENV_PRINT: "2"})
    assert [l.split(":")[0] for l in err2[: err2.index("body-done")]] == ["UNCOVERED_CELL", "CELL_HIT", "NAMED_FALLBACK"]


def test_print0_is_silent_and_a_directory_receives_a_per_pid_file(tmp_path):
    d = tmp_path / "cells_dir"
    d.mkdir()
    err = _run(tmp_path, {C.ENV_FILE: str(d) + "/", C.ENV_PRINT: "0"})
    assert err == ["body-done"]
    files = list(d.iterdir())
    assert len(files) == 1 and files[0].name.startswith("cell_census.") and files[0].name.endswith(".json")
    assert json.loads(files[0].read_text())["counts"]["inherited"] == 1


def test_summarize_merges_dumps_into_the_coverage_table(tmp_path):
    C.set_context(engine="kitA", mode="fast", card="H100", size=800)
    C.record("trimul", KEY, "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")
    C.record("trimul", dict(KEY, bucket="N<=800"), "cell_hit", "v4", cell_id="c1")
    C.record("trimul", dict(KEY, shape="C256H256"), "inherited", "cueq", cell_id=None, note="family:none(no_cell)")
    C.record("apb", dict(KEY, shape="dit"), "inherited", "sdpa:auto", cell_id="c2", note="guard:samples(S7)")
    a = tmp_path / "a.json"; a.write_text(C.dump())
    C.reset()
    C.record("ln", dict(KEY, shape="pair_c128"), "stock", "aten", cell_id="c3")                 # no context: the fill-ins name the axes
    b = tmp_path / "b.json"; b.write_text(C.dump())
    rows = C.summarize([str(a), str(b)], engine="kitB", overrides={"mode": "exact", "card": "A100", "size": 400})
    by = {(r["engine"], r["provider"]): r for r in rows}
    ta = by[("kitA", "trimul")]
    assert (ta["mode"], ta["card"], ta["size"]) == ("fast", "H100", "800")                        # the entry's own context wins over fill-ins
    assert (ta["cell_hit"], ta["inherited"], ta["inherited_size"], ta["inherited_family"], ta["verdict"]) == (1, 2, 1, 1, "inherited")
    assert by[("kitA", "apb")]["inherited_guard"] == 1
    lb = by[("kitB", "ln")]
    assert (lb["mode"], lb["card"], lb["size"], lb["stock"], lb["verdict"]) == ("exact", "A100", "400", 1, "stock")
    assert len(ta["alerts"]) == 2 and all(C.TOKEN_RE.match(t) for t in ta["alerts"])
    # the CLI
    env = dict(os.environ, PYTHONPATH=CORE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""), **{C.ENV_PRINT: "0"})
    r = subprocess.run([sys.executable, "-m", "opt_core.cell_census", "--summarize", str(a), str(b), "--engine", "kitB", "--fmt", "tsv"], capture_output=True, text=True,
                       cwd=CORE_DIR, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0].split("\t")[:6] == ["engine", "mode", "card", "size", "provider", "verdict"] and len(lines) == 4
    r = subprocess.run([sys.executable, "-m", "opt_core.cell_census", "--tokens", str(a), str(b)], capture_output=True, text=True, cwd=CORE_DIR, timeout=120, env=env)
    assert r.returncode == 0 and r.stdout.splitlines() == ta["alerts"] + by[("kitA", "apb")]["alerts"]     # every alert token of the dumps, in order


def test_true_gaps_lists_the_no_cell_keys_and_sets_sibling_stack_hits_aside(tmp_path):
    C.set_context(engine="kitA", mode="fast", card="A100", size=800)
    k8 = dict(KEY, cc="8.0", stack="A100:2.7.1+cu128/3.3.1/cueq0.10.0", bucket="N<=800")
    C.record("trimul", k8, "inherited", "v4", cell_id="8.0|bf16|C128|H128|N<=800|out|fwd", note="stack(A100:2.13.0+cu130/3.7.1/cueq0.11.1)")   # a LEGACY dump line (cores 0.5.72-0.5.80): sibling stack
    C.record("trimul", dict(k8, bucket="N=3000"), "inherited", "v4", cell_id="8.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")           # size gap
    C.record("trimul", dict(k8, shape="C128H128", form="in.fwd"), "cell_hit", "v4", cell_id="8.0|bf16|C128|H128|N<=800|in|fwd",
             note=C.sibling_note("A100:2.13.0+cu130/3.7.1/cueq0.11.1"))                                                                            # a current sibling-stack hit
    C.record("triattn", dict(k8, shape="D32H7", form="fwd"), "inherited", "k2b", cell_id="8.0|bf16|D32|H4|N<=800|fwd", note="guard:heads(heads_7_not_measured)")   # guard gap
    C.record("pallas", dict(k8, stack="jax0.7", shape="trimul:f", form="fwd"), "inherited", "xla", cell_id=None, note="family:none(jax_line)")             # family gap
    C.record("trimul", dict(k8, word="exact"), "named_fallback", "cueq", cell_id="8.0|bf16|C128|H128|N<=800|out|fwd", refused="tmk3_exact:exact_vouch_not_recorded_on:A100:2.7.1")  # vouch owed
    C.record("triattn", dict(k8, shape="D32H4", form="fwd"), "named_fallback", "k2b", cell_id="8.0|bf16|D32|H4|N<=800|fwd", refused="triattn_native:no_prebuilt")   # structural: NOT a gap
    a = tmp_path / "a.json"; a.write_text(C.dump())
    C.reset()
    C.set_context(engine="kitB", mode="big", card="A100", size=1200)
    C.record("trimul", dict(k8, bucket="N=3000", word="big"), "inherited", "v4", cell_id="8.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")   # another word: another key
    C.record("trimul", dict(k8, bucket="N=3000"), "inherited", "v4", cell_id="8.0|bf16|C128|H128|N<=2048|out|fwd", note="beyond_measured")               # the same key from another engine
    b = tmp_path / "b.json"; b.write_text(C.dump())
    gaps = C.true_gaps([str(a), str(b)])
    kinds = [g["kind"] for g in gaps]
    assert kinds == ["size", "size", "guard", "family", "exact_vouch"], kinds                       # the legacy sibling line and the current sibling hit are NOT gaps; structural fallbacks neither
    g0 = [g for g in gaps if g["kind"] == "size" and g["word"] == "fast"][0]
    assert g0["engines"] == ["kitA", "kitB"] and g0["sizes"] == ["800", "1200"] and g0["n"] == 2 and len(g0["dumps"]) == 2 and g0["nearest"].endswith("N<=2048|out|fwd")
    assert [g for g in gaps if g["kind"] == "exact_vouch"][0]["refused"].startswith("tmk3_exact:exact_vouch_not_recorded_on")
    # the summariser sets the sibling hits aside too (legacy line reclassified on read)
    rows = {(r["engine"], r["provider"]): r for r in C.summarize([str(a)])}
    ta = rows[("kitA", "trimul")]
    assert (ta["cell_hit"], ta["sibling_stack"], ta["inherited"], ta["inherited_size"], ta["named_fallback"]) == (2, 2, 1, 1, 1), ta
    assert rows[("kitA", "triattn")]["inherited_guard"] == 1 and rows[("kitA", "pallas")]["inherited_family"] == 1
    d = json.loads(a.read_text())
    leg = [e for e in d["entries"] if str(e.get("note", "")).startswith("stack(")][0]
    n = C.normalize_entry(leg)
    assert n["outcome"] == "cell_hit" and n["note"] == "inherited_stack(measured_on=A100:2.13.0+cu130/3.7.1/cueq0.11.1)" and n["token"].startswith("CELL_HIT:trimul|cc8.0|")
    assert C.TOKEN_RE.match(n["token"]) and " cell=8.0|bf16|C128|H128|N<=800|out|fwd note=inherited_stack(" in n["token"]
    # the CLI
    env = dict(os.environ, PYTHONPATH=CORE_DIR + os.pathsep + os.environ.get("PYTHONPATH", ""), **{C.ENV_PRINT: "0"})
    r = subprocess.run([sys.executable, "-m", "opt_core.cell_census", "--true-gaps", str(a), str(b)], capture_output=True, text=True, cwd=CORE_DIR, timeout=120, env=env)
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert lines[0] == "# true no-cell keys: 5 (size=2, guard=1, family=1, exact_vouch=1)" and lines[1].split("\t")[:3] == ["kind", "provider", "cc"] and len(lines) == 7
    r = subprocess.run([sys.executable, "-m", "opt_core.cell_census", "--true-gaps", str(a), "--fmt", "json"], capture_output=True, text=True, cwd=CORE_DIR, timeout=120, env=env)
    assert r.returncode == 0 and [g["kind"] for g in json.loads(r.stdout)] == ["size", "guard", "family", "exact_vouch"]


def test_a_transcript_of_token_lines_reads_back_as_a_dump(tmp_path):
    C.set_context(engine="kitL", mode="exact", card="H100", size=400)
    k = dict(KEY, bucket="N<=400", word="exact")
    C.record("trimul", k, "named_fallback", "cueq", cell_id="9.0|bf16|C128|H128|N<=400|out|fwd", refused="tmk3_exact:exact_vouch_not_recorded_on:X:1")
    C.record("trimul", dict(k, word="fast"), "inherited", "v4", cell_id="9.0|bf16|C128|H128|N<=400|out|fwd", note="stack(H100:2.12.0+cu130/3.7.0/cueq0.10.0)")   # legacy sibling line
    C.record("transition", dict(k, shape="c64_h128_pair", cc="-"), "inherited", "caller:torch_swiglu", cell_id=None, note="family:none(no_cell:9.0/bf16/c64)")
    C.record("apb", dict(k, shape="dit_h16d48", form="eager.S1", bucket="N=90000"), "inherited", "sdpa:auto", cell_id="9.0|bf16|dit_h16d48|S1|N<=2048|eager|fwd", note="beyond_measured")
    C.record("ln", dict(k, shape="pair_c128", form="fwd.eager"), "cell_hit", "fastln", cell_id="9.0|bf16|pair_c128|N<=400|eager|fwd")
    lines = ["    12.5 " + l for l in C.exit_lines()] + ["  13.0 [rank1] " + t for t in C.alerts()]      # a launcher's timestamp column + a second rank repeating the alerts
    log = tmp_path / "run.log"; log.write_text("noise line\n" + "\n".join(lines) + "\nmore noise UNCOVERED_CELL without grammar\n")
    doc = C.parse_log(log.read_text())
    assert doc["source"] == "log" and doc["context"] == {"engine": "kitL", "mode": "exact", "card": "H100", "size": "400"} and len(doc["entries"]) == 4   # the CELL_HIT is not an exit line
    by = {e["provider"] + e["word"]: e for e in doc["entries"]}
    assert by["trimulexact"]["outcome"] == "named_fallback" and by["trimulexact"]["refused"] == "tmk3_exact:exact_vouch_not_recorded_on:X:1" and by["trimulexact"]["n"] == 2
    assert by["trimulfast"]["cell"] == "9.0|bf16|C128|H128|N<=400|out|fwd" and by["transitionexact"]["cell"] is None and by["apbexact"]["bucket"] == "N=90000"
    for e in doc["entries"]:
        assert C.TOKEN_RE.match(e["token"]) and e["context"]["engine"] == "kitL"
    gaps = C.true_gaps([str(log)])
    assert [(g["kind"], g["provider"]) for g in gaps] == [("size", "apb"), ("family", "transition"), ("exact_vouch", "trimul")]   # the legacy sibling line is a hit, not a gap
    rows = {r["provider"]: r for r in C.summarize([str(log)])}
    assert rows["trimul"]["cell_hit"] == 1 and rows["trimul"]["sibling_stack"] == 1 and rows["trimul"]["named_fallback"] == 1 and rows["apb"]["inherited_size"] == 1


def test_lazy_export():
    import opt_core
    assert json.loads(opt_core.cell_census_dump())["schema"] == C.SCHEMA
    assert opt_core.cell_census_dump("tsv").startswith("provider\t")


# --------------------------------------------------------------------------------------------------------------------------------- nearest
CANDS = [dict(id="9.0|bf16|C128|H128|N<=400|out|fwd", cc="9.0", dtype="bf16", shape="C128H128", form="out.fwd", cls="tol", size=400),
         dict(id="9.0|bf16|C128|H128|N<=800|out|fwd", cc="9.0", dtype="bf16", shape="C128H128", form="out.fwd", cls="bitwise", size=800),
         dict(id="9.0|bf16|C128|H128|N<=2048|out|fwd", cc="9.0", dtype="bf16", shape="C128H128", form="out.fwd", cls="tol", size=2048),
         dict(id="8.0|bf16|C128|H128|N<=4096|out|fwd", cc="8.0", dtype="bf16", shape="C128H128", form="out.fwd", cls="tol", size=4096)]
Q = dict(cc="9.0", dtype="bf16", shape="C128H128", form="out.fwd", cls="tol", n=700)


def test_nearest_covering_bucket_is_a_hit_and_the_size_neighbour_is_served():
    assert tuple(C.nearest(Q, CANDS)) == ("9.0|bf16|C128|H128|N<=800|out|fwd", "covered", "bucket:N<=800")
    assert tuple(C.nearest(dict(Q, n=100), CANDS)) == ("9.0|bf16|C128|H128|N<=400|out|fwd", "covered", "bucket:N<=400")
    near = C.nearest(dict(Q, n=3000), CANDS)                                     # above every cc 9.0 bucket: the largest cc 9.0 cell, NOT the cc 8.0 N<=4096 one
    assert tuple(near) == ("9.0|bf16|C128|H128|N<=2048|out|fwd", "nearest_size", "nearest:N<=2048")
    assert (near.cell_id, near.relation, near.why) == tuple(near)


def test_nearest_guard_cross_card():
    assert tuple(C.nearest(dict(Q, cc="8.0", n=300), CANDS)) == ("8.0|bf16|C128|H128|N<=4096|out|fwd", "covered", "bucket:N<=4096")
    assert tuple(C.nearest(dict(Q, cc="10.0"), CANDS)) == (None, None, "cc:10.0")               # an H100 / A100 launch cell is never served on another card


STACK_A, STACK_B = "H100:2.13.0+cu130/3.7.1/cueq0.11.1", "A100:2.10.0+cu128/3.6.0/cueq0.10.0"
CANDS_V = [dict(c, vouched_on=[STACK_A]) if c["cls"] == "bitwise" else c for c in CANDS]      # the exact cell's vouch was recorded on STACK_A only


def test_nearest_guard_exact_never_inherits_a_tolerance_cell():
    ex = C.nearest(dict(Q, cls="exact", n=3000, stack=STACK_A), CANDS_V)
    assert tuple(ex) == ("9.0|bf16|C128|H128|N<=800|out|fwd", "nearest_size", "nearest:N<=800")   # the only exact-class (bitwise) cell of the family: a SIZE neighbour on the vouched stack
    assert tuple(C.nearest(dict(Q, cls="exact", n=3000, stack=STACK_A), [c for c in CANDS_V if c["cls"] != "bitwise"])) == (None, None, "cls:exact")
    assert C.nearest(dict(Q, cls="bitwise:cueq", n=100, stack=STACK_A), CANDS_V).cell_id == "9.0|bf16|C128|H128|N<=800|out|fwd"
    assert C.nearest(dict(Q, cls="tol(5e-3)", n=100), CANDS).cell_id == "9.0|bf16|C128|H128|N<=400|out|fwd"    # a tolerance request may take any class, on any stack


def test_nearest_guard_exact_vouch_is_stack_specific():
    assert tuple(C.nearest(dict(Q, cls="exact", n=3000, stack=STACK_B), CANDS_V)) == (None, None, "vouch:" + STACK_B)   # vouched on another stack: refused by name, the floor serves
    assert tuple(C.nearest(dict(Q, cls="exact", n=3000), CANDS_V)) == (None, None, "vouch:stack_unknown")            # an exact query naming no stack cannot inherit
    assert tuple(C.nearest(dict(Q, cls="exact", n=3000, stack=STACK_A), CANDS)) == (None, None, "vouch:" + STACK_A)     # a cell with no vouch record is not inherited under exact
    one = [dict(CANDS[1], stack=STACK_B)]                                                                             # a candidate's one `stack` word stands for its vouch list
    assert C.nearest(dict(Q, cls="exact", n=100, stack=STACK_B), one).relation == "covered"
    assert C.nearest(dict(Q, cls="tol", n=100, stack=STACK_B), CANDS_V).relation == "covered"                          # fast / big words unaffected


def test_nearest_guard_unserved_shape_form_dtype_are_refusals_not_inheritance():
    assert tuple(C.nearest(dict(Q, shape="C384H384"), CANDS)) == (None, None, "shape:C384H384")
    assert tuple(C.nearest(dict(Q, form="out.fwd+of3_module"), CANDS)) == (None, None, "form:out.fwd+of3_module")
    assert tuple(C.nearest(dict(Q, dtype="fp32"), CANDS)) == (None, None, "dtype:fp32")
    assert tuple(C.nearest(dict(Q, n="x"), CANDS)) == (None, None, "n:unparsable")
    assert tuple(C.nearest(Q, [])) == (None, None, "family:empty")
    assert tuple(C.nearest(Q, [dict(CANDS[0], size=None)])) == (None, None, "size:none_measured")


def _trimul_candidates(cc, prec, C_, D_, dirw, pas):
    out = []
    for key in trimul.table()["cells"]:
        p = key.split("|")
        if len(p) == 7 and "+" not in key:
            out.append(dict(id=key, cc=p[0], dtype=p[1], shape=p[2] + p[3], form="%s.%s" % (p[5], p[6]), size=int(p[4][3:]), cls=None))
    return out, dict(cc=cc, dtype=prec, shape="C%dH%d" % (C_, D_), form="%s.%s" % (dirw, pas), cls=None)


@pytest.mark.parametrize("cc,prec,C_,D_,dirw", [("9.0", "bf16", 128, 128, "out"), ("9.0", "bf16", 256, 256, "in"), ("8.0", "bf16", 128, 128, "in"),
                                                  ("9.0", "tf32", 128, 128, "out"), ("8.0", "fp32", 256, 256, "out")])
def test_trimul_cell_key_is_the_nearest_rule(cc, prec, C_, D_, dirw):
    """The trimul face's own bucket rule (smallest measured size >= n, else the largest) IS nearest() within the family for every size."""
    cands, q = _trimul_candidates(cc, prec, C_, D_, dirw, "fwd")
    fam = [c for c in cands if (c["cc"], c["dtype"], c["shape"], c["form"]) == (q["cc"], q["dtype"], q["shape"], q["form"])]
    if not fam:
        pytest.skip("no per-direction family here")
    dtype, tf32 = ("fp32", True) if prec == "tf32" else (prec, None)
    for n in (1, 200, 399, 400, 401, 800, 1201, 2048, 2049, 5000, 100000):
        key, size_measured, note = trimul.cell_key(cc, dtype, C_, D_, n, {"out": "outgoing", "in": "incoming"}[dirw], tf32=tf32)
        near = C.nearest(dict(q, n=n), cands)
        assert key == near.cell_id, (n, key, tuple(near))
        assert ("beyond_measured" in note) == (near.relation == "nearest_size"), (n, note, tuple(near))
        assert size_measured == (near.relation == "covered" and near.why == "bucket:N<=%d" % n)     # the face's word: n IS a measured size


# ------------------------------------------------------------------------------------------------------------------------- face hooks
def _last(provider):
    rows = [e for e in C.table() if e["provider"] == provider]
    assert rows, provider
    return rows[-1]


def test_trimul_hook_outcomes(monkeypatch):
    s = trimul.select("9.0", "bf16", 128, 128, 800, "outgoing", word="fast")
    e = _last("trimul")
    assert e["outcome"] in ("cell_hit", "stock") and e["served"] == s.row and e["cell"] == s.cell and e["bucket"] == "N<=800" and e["form"] == "out.fwd"
    assert (e["outcome"] == "stock") == (s.row in trimul.STOCK_ROWS)
    s = trimul.select("8.0", "bf16", 128, 128, 100000, "outgoing", word="fast")               # above every bucket: the largest cell's row, inherited (the 8.0 column: from
    e = _last("trimul")                                                                          # CORE R1 the 9.0 column's largest bucket names native, whose envelope ends at N 4096 -> named fallback there)
    assert e["cell"] == s.cell and e["bucket"] == "N=100000" and ((e["outcome"] == "inherited" and e["note"] == "beyond_measured" and e["token"].startswith("UNCOVERED_CELL:trimul|cc8.0|"))
                                                                     or (e["outcome"] == "named_fallback" and e["token"].startswith("NAMED_FALLBACK:trimul|cc8.0|")))   # the largest cell's word refusing at that N steps aside by name
    trimul.select("8.0", "fp16", 128, 128, 384, "incoming", word="fast")                       # no fp16 family: the named stock row, nearest=none
    e = _last("trimul")
    assert e["outcome"] == "inherited" and e["cell"] is None and e["note"].startswith("family:none(") and " nearest=none " in e["token"] + " "
    trimul.select("9.0", "bf16", 128, 128, 800, "outgoing", word="v4")                         # a row word: opt-in
    assert _last("trimul")["outcome"] == "opt_in" and _last("trimul")["word"] == "v4"
    with pytest.raises(trimul.Refusal):
        trimul.select("9.0", "bf16", 384, 384, 800, "outgoing", word="v4")                     # a width the row does not serve: refused BY NAME, never inherited
    e = _last("trimul")
    assert e["outcome"] == "named_fallback" and e["refused"].startswith("v4:") and e["served"].startswith("caller:") and e["note"] == "raised"
    # a sibling stack of the cc (no column of its own): the reference column's cell IS the cell -> CELL_HIT measured_on; above every bucket it stays a size
    # gap -- unless the cell's winner refuses by name on that stack (a stack-gated payload row), when the step-aside is the fact (NAMED_FALLBACK)
    XS = "X:2.0.0+cu000/0.0.0/cueq0.0.0"
    for cc_ in ("8.0", "9.0"):
        ss = trimul.select(cc_, "bf16", 128, 128, 400, "outgoing", word="fast", stack=XS)
        e = _last("trimul")
        assert ss.stack and ss.stack != XS, (cc_, ss.stack)
        if "skip " in (ss.reason or ""):
            assert e["outcome"] == "named_fallback" and e["served"] == ss.row, (cc_, e)
        else:
            assert e["outcome"] in ("cell_hit", "stock") and e["note"] == "inherited_stack(measured_on=%s)" % ss.stack, (cc_, e)
            assert e["token"].split(":")[0] in ("CELL_HIT", "STOCK_CELL") and " nearest=" not in e["token"]
        sb = trimul.select(cc_, "bf16", 128, 128, 100000, "outgoing", word="fast", stack=XS)
        e = _last("trimul")
        if "skip " in (sb.reason or ""):
            assert e["outcome"] == "named_fallback" and e["served"] == sb.row, (cc_, e)
        else:
            assert e["outcome"] == "inherited" and e["note"] == "beyond_measured;inherited_stack(measured_on=%s)" % sb.stack, (cc_, e)
    # EXACT is stack-specific: an exact-class row serves under the exact word on a vouched stack only; elsewhere refused BY NAME, the stock op serves
    vo = (trimul.table()["cells"].get("9.0|bf16|C128|H128|N<=400|out|fwd") or {}).get("vouched_on") or {}
    xrow = [r for r in vo if r not in trimul.STOCK_ROWS and vo[r]]
    if xrow:
        sv = trimul.select("9.0", "bf16", 128, 128, 400, "outgoing", word="exact", stack=vo[xrow[0]][0])
        assert sv.row not in trimul.STOCK_ROWS and _last("trimul")["outcome"] == "cell_hit"
        su = trimul.select("9.0", "bf16", 128, 128, 400, "outgoing", word="exact", stack="X:2.0.0+cu000/0.0.0/cueq0.0.0")
        e = _last("trimul")
        assert su.row in trimul.STOCK_ROWS and e["outcome"] == "named_fallback" and ":exact_vouch_not_recorded_on:X:2.0.0+cu000/0.0.0/cueq0.0.0" in e["refused"]
        assert e["served"] == su.row and not e["refused"].startswith("caller")
    # a tier word whose winner refuses by name at selection time: the next measured row serves = named_fallback (monkeypatched admission)
    win = trimul.select("9.0", "bf16", 128, 128, 800, "outgoing", word="fast").row
    real = trimul.admits

    def picky(row, *a, **k):
        if row == win:
            raise trimul.Refusal("test_refusal", row, "cueq")
        return real(row, *a, **k)
    monkeypatch.setattr(trimul, "admits", picky)
    s2 = trimul.select("9.0", "bf16", 128, 128, 800, "outgoing", word="fast", stack="X:test")
    e = _last("trimul")
    assert s2.row != win and e["outcome"] == "named_fallback" and e["refused"] == "%s:test_refusal" % win and e["served"] == s2.row and e["stack"] == "X:test"
    # a form cell on an unmeasured stack: inherited (stack); the exclude re-selection of a step-aside is not recorded by select
    n0 = len(C.table())
    trimul.select("9.0", "bf16", 128, 128, 800, "outgoing", word="fast", exclude=("v4",))
    assert len(C.table()) == n0


def test_triattn_hook_outcomes():
    s = triattn.select("9.0", "bf16", 32, 4, 768, word="fast")
    e = _last("triattn")
    assert e["outcome"] in ("cell_hit", "stock", "named_fallback") and e["served"] == s.row and e["shape"] == "D32H4" and e["bucket"] == "N<=800"
    s = triattn.select("9.0", "bf16", 32, 4, 100000, word="fast")
    e = _last("triattn")
    assert e["outcome"] == "inherited" and e["bucket"] == "N=100000" and e["note"].startswith("beyond_measured(") and e["cell"] == s.cell
    with pytest.raises(triattn.Refusal):
        triattn.select("9.0", "bf16", 32, 7, 300, word="fast")                                 # a head count no cell measured: a tier word refuses by name (the stock op serves)
    e = _last("triattn")
    assert e["outcome"] == "named_fallback" and e["refused"] == "fast:no_cell:heads=7" and e["served"] == "caller:cueq" and e["shape"] == "D32H7"
    triattn.select("9.0", "bf16", 32, 7, 300, word="k2b")                                      # the row word at that head count: opt-in, flagged unmeasured
    e = _last("triattn")
    assert e["outcome"] == "opt_in" and e["note"].startswith("heads_7_not_measured")   # the census cleans words
    triattn.select("9.0", "bf16", 32, 4, 768, word="fast", stack="no-such-stack")               # the prebuilt rows pass over by name on this stack: named_fallback
    e = _last("triattn")
    assert e["outcome"] == "named_fallback" and ":" in e["refused"] and e["stack"] == "no-such-stack" and not e["served"].startswith("caller:")
    triattn.select("9.0", "fp32", 32, 4, 768, word="exact")                                     # the exact tier for fp32: the stock op by the table
    e = _last("triattn")
    assert e["outcome"] == "stock" and e["served"] in triattn.STOCK_ROWS
    triattn.select("9.0", "bf16", 16, 4, 300, word="k2b")
    assert _last("triattn")["outcome"] == "opt_in"
    with pytest.raises(triattn.Refusal):
        triattn.select("9.0", "bf16", 32, 4, 768, word="triattn_native", stack="no-such-stack")
    e = _last("triattn")
    assert e["outcome"] == "named_fallback" and e["served"].startswith("caller:") and e["refused"].startswith("triattn_native:")
    xs = "9.0|torch0.0.0|cueq0.0.0"                                                              # the exact word on a running stack without a vouch: the stock op serves --
    sx = triattn.select("9.0", "bf16", 32, 4, 768, word="exact", exact_stack=xs)                # by name (named_fallback + alert) where the cell's exact winner is a vouched
    e = _last("triattn")                                                                         # kernel row passed over on this stack, plainly (stock) where the winner is the stock op
    winner = triattn.cells()[sx.cell]["exact"]
    assert sx.row in triattn.STOCK_ROWS and e["stack"] == C.clean(xs)
    if winner in triattn.STOCK_ROWS:
        assert e["outcome"] == "stock"
    else:
        assert e["outcome"] == "named_fallback" and e["refused"].startswith(winner + ":exact_vouch_not_recorded") and e["served"] == sx.row, e
    s16 = triattn.select("9.0", "bf16", 16, 4, 768, word="exact", exact_stack=xs)               # a cell no kernel row is vouched in (head_dim 16): the stock op, plainly
    e = _last("triattn")
    assert s16.row in triattn.STOCK_ROWS and triattn.cells()[s16.cell]["exact"] in triattn.STOCK_ROWS and e["outcome"] == "stock", e
    xv = triattn.table()["rows"]["triattn_exact"]["vouched_on"][0].rsplit("+", 1)[0]           # ... and on a vouched stack the vouched row is the cell hit
    sv = triattn.select("9.0", "bf16", 32, 4, 768, word="exact", exact_stack=xv)
    e = _last("triattn")
    assert sv.row == "triattn_exact" and e["outcome"] == "cell_hit" and e["served"] == "triattn_exact", e
    triattn.select("9.0", "bf16", 72, 4, 300, word="fast")                                     # a head_dim no row serves: refused by name, the stock row serves
    e = _last("triattn")
    assert e["outcome"] == "named_fallback" and e["refused"].startswith("k2b:head_dim") and e["served"] in triattn.STOCK_ROWS


def test_apb_and_ln_hook_outcomes(monkeypatch):
    s = apb.select("9.0", "bf16", "dit_h16d48", 800, word="fast")
    e = _last("apb")
    assert e["outcome"] in ("cell_hit", "stock") and (e["outcome"] == "stock") == (s.row in apb.STOCK_ROWS) and e["form"] == "eager.S1"
    apb.select("9.0", "bf16", "dit_h16d48", 8000, word="fast")                                  # beyond the top bucket, inside every row's own bounds
    assert _last("apb")["outcome"] == "inherited" and _last("apb")["note"] == "beyond_measured"
    s7 = apb.select("9.0", "bf16", "dit_h16d48", 800, word="fast", samples=7)                   # a sample count no cell measured: the nearest rule is SIZE-only ->
    assert _last("apb")["outcome"] == "named_fallback" and _last("apb")["refused"].startswith("cells:samples_unmeasured:S7")   # the boundary's stock row BY NAME
    assert s7.row in apb.STOCK_ROWS and _last("apb")["served"] == apb.arm_word(s7.row, s7.variant)
    sib = apb.select("9.0", "bf16", "dit_h16d48", 800, word="fast", stack="X:test")               # a stack with no column: the same-cc reference column's cell IS the cell (a hit, measured_on named)
    assert not sib.stack_measured and _last("apb")["outcome"] in ("cell_hit", "stock") and _last("apb")["note"] == "inherited_stack(measured_on=%s)" % sib.stack
    assert " nearest=" not in _last("apb")["token"] and _last("apb")["token"].split(":")[0] in ("CELL_HIT", "STOCK_CELL")
    apb.select("9.0", "bf16", "dit_h16d48", 8000, word="fast", stack="X:test")                     # a size neighbour on a sibling stack: still a TRUE gap (size), the sibling named after it
    assert _last("apb")["outcome"] == "inherited" and _last("apb")["note"].startswith("beyond_measured;inherited_stack(measured_on=")
    # EXACT is stack-specific: the cell's exact winner (dit_exact, fp32) serves on a vouched stack only; elsewhere the floor BY NAME, never inherited
    xkey = [k for k in apb.table()["cells"] if k.startswith("9.0|fp32|dit_h16d48|S1|") and (apb.table()["cells"][k].get("vouched_on") or {}).get("dit_exact")]
    if xkey:
        von = apb.table()["cells"][xkey[0]]["vouched_on"]["dit_exact"][0]
        n_ = int(xkey[0].split("|")[4][3:])
        sv = apb.select("9.0", "fp32", "dit_h16d48", n_, word="exact", stack=von, head_dim=48)
        assert sv.row == "dit_exact" and _last("apb")["outcome"] == "cell_hit" and _last("apb")["refused"] is None
        su = apb.select("9.0", "fp32", "dit_h16d48", n_, word="exact", stack="X:unvouched", head_dim=48)
        e = _last("apb")
        assert su.row in apb.STOCK_ROWS and e["outcome"] == "named_fallback" and e["refused"] == "dit_exact:exact_vouch_not_recorded_on:X:unvouched"
        assert e["served"] == apb.arm_word(su.row, su.variant) and e["token"].startswith("NAMED_FALLBACK:apb|cc9.0|X:unvouched|fp32|")
        apb.select("9.0", "fp32", "dit_h16d48", n_, word="exact", head_dim=48)                        # no stack named: the vouch cannot be checked -> the floor by name
        assert _last("apb")["outcome"] == "named_fallback" and _last("apb")["refused"] == "dit_exact:exact_vouch_not_recorded_on:unnamed_stack"
    apb.select("9.0", "bf16", None, 800, word="fast", heads=3, head_dim=11)                # no cell family for the geometry: the boundary's stock row BY NAME
    assert _last("apb")["outcome"] == "named_fallback" and _last("apb")["cell"] is None and _last("apb")["refused"].startswith("cells:no_family")
    apb.select("9.0", "bf16", "dit_h16d48", 800, word="sdpa")
    assert _last("apb")["outcome"] == "opt_in"
    with pytest.raises(apb.Refusal):
        apb.select("9.0", "fp32", "dit_h16d48", 800, word="dit_exact", abi="no-such-abi", head_dim=48)
    assert _last("apb")["outcome"] == "named_fallback" and _last("apb")["served"].startswith("caller:")
    win = apb.select("9.0", "bf16", "bias_c128h16", 800, word="fast")
    real = apb.admits

    def picky(row, *a, **k):
        if row == win.row:
            raise apb.Refusal("test_refusal", row, "sdpa")
        return real(row, *a, **k)
    monkeypatch.setattr(apb, "admits", picky)
    s2 = apb.select("9.0", "bf16", "bias_c128h16", 800, word="fast", stack="Y:test")
    e = _last("apb")
    assert e["outcome"] == "named_fallback" and e["refused"] == "%s:test_refusal" % apb.arm_word(win.row, win.variant).split(":")[0] or e["refused"].startswith(win.row)
    assert e["served"] == apb.arm_word(s2.row, s2.variant)
    monkeypatch.setattr(apb, "admits", real)
    # ln
    s = ln.select("9.0", "bf16", "pair_c128", 800, word="fast")
    assert _last("ln")["outcome"] in ("cell_hit", "stock") and _last("ln")["served"] == ln.arm_word(s.row, s.variant) and _last("ln")["form"] == "fwd.eager"
    ln.select("9.0", "fp32", "pair_c128", 100000, word="fast")
    assert _last("ln")["outcome"] == "inherited" and _last("ln")["note"] == "beyond_measured"
    ln.select("9.0", "fp32", "pair_c128", 100000, word="exact")                                  # exact with no stack named: the exact arm is held back BY NAME (its vouch cannot be checked)
    assert _last("ln")["outcome"] == "named_fallback" and _last("ln")["refused"].endswith(":exact_vouch_not_recorded_on:unnamed_stack")
    s = ln.select("9.0", "bf16", "pair_c128", 800, word="exact")
    assert (_last("ln")["outcome"] == "stock") == (s.row in ln.STOCK_ROWS)
    sib = ln.select("9.0", "bf16", "pair_c128", 800, word="fast", stack="X:test")                     # sibling stack of the cc: a hit, measured_on named
    assert not sib.stack_measured and _last("ln")["outcome"] in ("cell_hit", "stock") and _last("ln")["note"] == "inherited_stack(measured_on=%s)" % sib.stack
    lkey = [k for k in ln.table()["cells"] if k.startswith("9.0|fp32|pair_c128|") and "|eager|fwd" in k and (ln.table()["cells"][k].get("vouched_on") or {}).get("exactln")]
    if lkey:
        von = ln.table()["cells"][lkey[0]]["vouched_on"]["exactln"][0]
        n_ = int(lkey[0].split("|")[3][3:])
        sv = ln.select("9.0", "fp32", "pair_c128", n_, word="exact", stack=von)
        assert sv.row == "exactln" and _last("ln")["outcome"] == "cell_hit"
        su = ln.select("9.0", "fp32", "pair_c128", n_, word="exact", stack="X:unvouched")
        assert su.row in ln.STOCK_ROWS and _last("ln")["outcome"] == "named_fallback" and _last("ln")["refused"] == "exactln:exact_vouch_not_recorded_on:X:unvouched"
    ln.select("9.0", "fp32", None, 800, word="fast", C=100)                                 # no family lists the width: the statement BY NAME
    assert _last("ln")["outcome"] == "named_fallback" and _last("ln")["cell"] is None and _last("ln")["refused"].startswith("cells:no_family")
    ln.select("9.0", "fp32", "pair_c128", 800, word="exactln")
    assert _last("ln")["outcome"] == "opt_in"
    # a timing form no cell measured (graph asked, only eager cells for that family): the statement BY NAME, never the eager neighbour's winner
    fams = {tuple(k.split("|")[i] for i in (0, 1, 2, 5)): k for k in ln.table()["cells"] if k.startswith("9.0|") and k.split("|")[4] == "eager"}   # ln keys: cc|form|cell|N<=n|timing|pass
    graph = {tuple(k.split("|")[i] for i in (0, 1, 2, 5)) for k in ln.table()["cells"] if k.split("|")[4] == "graph"}
    only_eager = sorted(v for f, v in fams.items() if f not in graph and f[1] == "fp32" and f[3] == "fwd")
    if only_eager:
        cc_, dt_, cell_, _n, _t, pas_ = only_eager[0].split("|")
        g = ln.select(cc_, dt_, cell_, int(_n[3:]), word="fast", capture=True)
        assert g.row in ln.STOCK_ROWS and _last("ln")["outcome"] == "named_fallback" and _last("ln")["refused"].startswith("cells:timing_unmeasured")
    with pytest.raises(ln.Refusal):
        ln.select("9.0", "fp16", "pair_c128", 800, word="exactln")
    assert _last("ln")["outcome"] == "named_fallback" and _last("ln")["refused"].startswith("exactln:dtype")


def test_transition_hook_outcomes():
    s = transition.select("fast", c=128, hidden=512, n_tokens=800, cc="9.0", stack="A:none")     # a stack with no column: the reference column's cell IS the cell (a hit, measured_on named)
    e = _last("transition")
    assert e["shape"] == "pair_c128_n4" and e["form"] == "fwd.eager.swiglu" and e["bucket"] == "N<=800"
    assert not s.stack_measured and e["outcome"] == ("stock" if s.row in transition.STOCK_ROWS else "cell_hit") and e["note"] == "inherited_stack(measured_on=%s)" % s.stack
    with pytest.raises(transition.Refusal) as ri:                                                   # EXACT is stack-specific: refused by name on an unvouched stack, the module floor serves
        transition.select("exact", c=128, hidden=512, n_tokens=800, cc="9.0", stack="A:none")
    e = _last("transition")
    assert ri.value.kind.startswith("exact_vouch_not_recorded_on") and e["outcome"] == "named_fallback" and e["refused"].startswith("exact:exact_vouch_not_recorded_on")
    assert e["served"] == "caller:%s" % ri.value.fallback
    cell = transition.table()["cells"][s.cell_key]
    st = sorted(cell["stacks"])[0]
    s = transition.select("fast", c=128, hidden=512, n_tokens=800, cc="9.0", stack=st)          # a measured stack column: a hit (or the stock arm)
    assert _last("transition")["outcome"] in ("cell_hit", "stock") and _last("transition")["stack"] == st
    transition.select("fast", c=128, hidden=512, n_tokens=100000, cc="9.0", stack=st)
    assert _last("transition")["outcome"] == "inherited" and _last("transition")["note"] == "beyond_measured" and _last("transition")["bucket"] == "N=100000"
    with pytest.raises(transition.Refusal):
        transition.select("fast", c=100, hidden=400, n_tokens=800, cc="9.0")                     # no family: the tier word is refused by name, nearest=none
    e = _last("transition")
    assert e["outcome"] == "inherited" and e["cell"] is None and e["served"] == "caller:torch_swiglu" and e["note"].startswith("family:none(")
    transition.select("v2", c=128, hidden=512, n_tokens=800, cc="9.0", stack=st)
    assert _last("transition")["outcome"] == "opt_in"
    with pytest.raises(transition.Refusal):
        transition.select("torch_swiglu", c=128, hidden=512, cc="9.0", capture=True) if "torch_swiglu" in transition.CAPTURE_UNSAFE_ROWS else \
            transition.select("compile", c=128, hidden=512, cc="9.0", capture=True)
    assert _last("transition")["outcome"] == "named_fallback" and _last("transition")["served"].startswith("caller:")


def test_pallas_and_triattn_xla_hook_outcomes():
    s = pallas.select("0.10.2", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="fast")
    e = _last("pallas")
    assert e["outcome"] in ("cell_hit", "stock") and e["stack"] == "jax0.10" and e["shape"] == "trimul:af2_pair_c128_ch128_outgoing" and e["served"] == s.arm
    pallas.select("0.10.2", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 100000, word="fast")
    assert _last("pallas")["outcome"] == "inherited" and _last("pallas")["note"] == "beyond_measured"
    pallas.select("0.7.0", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="fast")       # an unlisted jax line: no cell, the stock statement
    e = _last("pallas")
    assert e["outcome"] == "inherited" and e["cell"] is None and e["served"] == "xla" and e["stack"] == "jax0.7"
    pallas.select("0.10.2", "9.0", "f32", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="exact")
    assert _last("pallas")["outcome"] == "stock"
    pallas.select("0.10.2", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="cd_trimul")
    assert _last("pallas")["outcome"] == "opt_in"
    with pytest.raises(pallas.Refusal):
        pallas.select("0.10.2", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="no_such_row")
    assert _last("pallas")["outcome"] == "named_fallback" and _last("pallas")["served"].startswith("caller:")
    sel = pallas.select("0.10.2", "9.0", "bf16", "trimul", "af2_pair_c128_ch128_outgoing", 384, word="fast")
    if len(sel.candidates) > 1:                                                                          # the serving walk's step-aside re-records the key by name
        pallas.census_stepaside(sel, sel.candidates[1], sel.row, "launch_fails", [sel.candidates[0]])
        e = [x for x in C.table() if x["provider"] == "pallas" and x["word"] == "fast" and x["bucket"] == "N<=400" and x["dtype"] == "bf16"][0]   # revised in place (keeps its position)
        assert e["outcome"] == "named_fallback" and e["served"] == sel.candidates[1] and e["refused"] == "%s:launch_fails" % sel.row and e["revised"] >= 1

    class DT:
        def __init__(self, n):
            self.name = n
    row, cell = triattn_xla.select("9.0", DT("bfloat16"), 32, 384, H=4)
    e = _last("triattn_xla")
    assert e["outcome"] in ("cell_hit", "named_fallback") and e["served"] == row and e["bucket"] == "S=384" and e["cell"] == "by_cc:9.0"
    triattn_xla.select("8.6", DT("bfloat16"), 32, 384, H=4)
    assert _last("triattn_xla")["outcome"] == "cell_hit" and _last("triattn_xla")["cell"] == "by_cc:8.6"
    triattn_xla.select("9.0", DT("bfloat16"), 32, 384, H=4, impl="k2b_aot")
    assert _last("triattn_xla")["outcome"] == "opt_in"
    with pytest.raises(triattn_xla.Refused):
        triattn_xla.select("9.0", DT("bfloat16"), 72, 384, H=4)
    e = _last("triattn_xla")
    assert e["outcome"] == "named_fallback" and e["served"] == "caller:xla" and "head_dim" in e["refused"]


# ------------------------------------------------------------------------------------------------------------------- selection identity
def _grid_trimul():
    out = []
    for cc in ("9.0", "8.0"):
        for prec, kw in (("bf16", {}), ("fp32", {}), ("tf32", dict(tf32=True)), ("fp16", {})):
            dt = "fp32" if prec == "tf32" else prec
            for C_, D_ in ((128, 128), (256, 256), (64, 128), (384, 384), (128, 32)):
                for n in (100, 400, 800, 1201, 3000):
                    for d in ("outgoing", "incoming"):
                        for word in ("fast", "exact", "big", "v4", "cueq", "esm_v5_fwd", "exact+of3_module"):
                            for extra in ({}, dict(stack="H100:2.13.0+cu130/3.7.1/cueq0.11.1", abi="none"), dict(backward=True)):
                                out.append(((cc, dt, C_, D_, n, d), dict(kw, word=word, **extra)))
    return out


def _grid_triattn():
    return [((cc, dt, D, H, n), dict(word=w, **ex)) for cc in ("9.0", "8.0") for dt in ("bf16", "fp32", "fp16") for D in (16, 32, 64, 72) for H in (2, 4, 7)
            for n in (100, 768, 5000) for w in ("fast", "exact", "k2b", "cueq", "triattn_native", "tf32") for ex in ({}, dict(stack="nostack"), dict(direction="fwdbwd"))]


def _grid_apb():
    return [((cc, dt, cell, n), dict(word=w, **ex)) for cc in ("9.0", "8.0") for dt in ("bf16", "fp32") for cell in ("dit_h16d48", "bias_c128h16", "atom_h4d32w32x128", None)
            for n in (400, 800, 5000) for w in ("fast", "exact", "big", "sdpa", "fpf_apb", "dit_exact") for ex in ({}, dict(samples=5), dict(capture=True), dict(stack="X", abi="none", head_dim=48))]


def _grid_ln():
    return [((cc, dt, cell, n), dict(word=w, **ex)) for cc in ("9.0", "8.0") for dt in ("bf16", "fp32", "bf16w") for cell in ("pair_c128", "pair_c256", "single_c384", None)
            for n in (400, 800, 5000) for w in ("fast", "exact", "faithful", "exactln", "aten", "fastln") for ex in ({}, dict(capture=True), dict(stack="X", C=100))]


def _grid_transition():
    return [((w,), dict(c=c, hidden=h, n_tokens=n, cc=cc, **ex)) for w in ("fast", "exact", "big", "faithful", "v2", "v1", "torch_swiglu", "af3_fused")
            for c, h in ((128, 512), (128, 256), (256, 1024), (64, 256), (100, 400)) for n in (None, 400, 800, 5000) for cc in ("9.0", "8.0")
            for ex in ({}, dict(stack="A100:torch2.11.0+cu128/3.6.0"), dict(capture=True), dict(family="esmpair"), dict(form="liger"))]


def _grid_pallas():
    return [((line, cc, dt, op, fam, n), dict(word=w)) for line in ("0.10.2", "0.6.0", "0.7.0") for cc in ("9.0", "8.0") for dt in ("bf16", "f32")
            for op, fam in (("trimul", "af2_pair_c128_ch128_outgoing"), ("triattn", "af2_pair_c128_h4_d32_starting"), ("ln", "atom_c128"), ("attn", "tri_h4_d32"))
            for n in (384, 800, 100000) for w in ("fast", "exact", "big", "cd_trimul", "xla", "no_such_row")]


def _run_grid(face, grid, exc):
    out = []
    for args, kw in grid:
        try:
            r = face.select(*args, **kw)
            out.append(repr(tuple(r)) if isinstance(r, tuple) else repr(r.__dict__) if hasattr(r, "__dict__") else repr([getattr(r, s) for s in r.__slots__]))
        except exc as e:
            out.append("refusal:%s|%s|%s" % (getattr(e, "kind", ""), getattr(e, "row", ""), getattr(e, "fallback", "")))
        except (ValueError, TypeError) as e:                       # argument errors are the face's, census or not
            out.append("error:%s:%s" % (type(e).__name__, e))
    return out


FACES = [("trimul", trimul, _grid_trimul, trimul.Refusal), ("triattn", triattn, _grid_triattn, triattn.Refusal), ("apb", apb, _grid_apb, apb.Refusal),
         ("ln", ln, _grid_ln, ln.Refusal), ("transition", transition, _grid_transition, transition.Refusal), ("pallas", pallas, _grid_pallas, pallas.Refusal)]


@pytest.mark.parametrize("name,face,gridf,exc", FACES, ids=[f[0] for f in FACES])
def test_selection_identity_with_the_census_recording_raising_or_absent(monkeypatch, name, face, gridf, exc):
    grid = gridf()
    triattn.select_cache_clear()
    base = _run_grid(face, grid, exc)                              # census recording
    assert len(C.table()) > 0

    def boom(*a, **k):
        raise RuntimeError("census must never break a selection")
    monkeypatch.setattr(C, "record", boom)
    triattn.select_cache_clear()
    assert _run_grid(face, grid, exc) == base                      # census raising inside: identical decisions
    monkeypatch.setattr(face, "_census", lambda *a, **k: None)
    triattn.select_cache_clear()
    assert _run_grid(face, grid, exc) == base                      # census absent: identical decisions
    assert len(set(base)) > 3, name                                # the grid exercised more than one decision


def test_triattn_xla_selection_identity(monkeypatch):
    class DT:
        def __init__(self, n):
            self.name = n
    grid = [((cc, DT(dt), D, S), dict(H=H, impl=impl)) for cc in ("9.0", "8.0", "8.6", "7.5") for dt in ("bfloat16", "float32") for D in (32, 72) for S in (64, 384, 3000)
            for H in (4, 2) for impl in ("auto", "k2b_aot", "triattn_native")]
    base = _run_grid(triattn_xla, grid, triattn_xla.Refused)
    monkeypatch.setattr(C, "record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _run_grid(triattn_xla, grid, triattn_xla.Refused) == base
    monkeypatch.setattr(triattn_xla, "_census", lambda *a, **k: None)
    assert _run_grid(triattn_xla, grid, triattn_xla.Refused) == base
