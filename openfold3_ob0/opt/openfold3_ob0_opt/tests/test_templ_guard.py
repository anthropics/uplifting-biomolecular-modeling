"""The template guard on synthetic stock modules (templ_guard.py under this kit's policy consume(upstream_0.5.0)): a query
that declares template inputs is accepted and parsed (``TEMPLATES PARSED …: templates parsed; consumed at inference (upstream 0.5.0
behaviour)``), ``TEMPLATE INPUT DROPPED`` names a declared chain the preprocessor left untemplated, and for the sampler's consuming call (a
template cache directory — 0.5.0's own dataset passes it) the stock function still drops SILENTLY a template whose aligned residues are
missing from the deposited model (the 8IO9 shape: 4/4 templates dropped): the guard names every drop, prints the per-chain census
``TEMPLATES chain=<id> real=<k>/<N> dropped=<d>`` and the all-dummy event, and returns the stock result; a sampler call WITHOUT a cache
directory (the ignore policy's shape, never 0.5.0's dataset) is the stock call untouched and named. Nothing refuses and no template-policy
switch exists (module, modes table, CLI). Counts close."""
import ast
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from openfold3_ob0_opt import templ_guard
from openfold3_ob0_opt.tests import _stubs

HERE = os.path.dirname(os.path.abspath(__file__))
STOCK = os.path.normpath(os.path.join(HERE, "..", "..", "..", "stock", "src", "openfold3", "core", "data", "primitives", "structure", "template.py"))
HOME = os.path.normpath(os.path.join(HERE, "..", "..", ".."))                        # the release tree (env.kit_dirs reads its hook directories)
COUNTER_KEYS = ("installed", "declared", "alive", "input_dropped", "ignored", "sampled", "real", "dropped", "all_dummy")


class FakeAtomArray:
    """Attribute access + boolean/integer indexing over equal-length numpy columns (the AtomArray surface the stock function uses)."""
    COLS = ("chain_id", "res_id", "res_name", "atom_name", "molecule_type_id", "occupancy", "token_position", "token_id")

    def __init__(self, **cols):
        self._cols = {k: np.asarray(v) for k, v in cols.items()}

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError(k)
        return self._cols[k]

    def __getitem__(self, idx):
        return FakeAtomArray(**{k: v[idx] for k, v in self._cols.items()})

    def __len__(self):
        return len(next(iter(self._cols.values())))


def chain(res_ids, chain_id="A", res_name="ALA", atom_name="CA", token0=0):
    n = len(res_ids)
    return FakeAtomArray(chain_id=[chain_id] * n, res_id=list(res_ids), res_name=[res_name] * n, atom_name=[atom_name] * n, molecule_type_id=[0] * n,
                         occupancy=[1.0] * n, token_position=list(range(token0, token0 + n)), token_id=list(range(token0, token0 + n)))


def stock_map_function():
    """map_token_pos_to_template_residues compiled from the stock source with numpy stand-ins for its module globals."""
    src = open(_stubs.stock_src("openfold3", "core", "data", "primitives", "structure", "template.py"), encoding="utf-8").read()   # = STOCK; skips by name while stock/src is absent
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "map_token_pos_to_template_residues")
    fn.decorator_list = []                                               # @log_runtime_memory: the memory logger is not the subject
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "AtomArray": FakeAtomArray, "TemplateCacheEntry": object,
          "struc": types.SimpleNamespace(get_residue_starts=lambda a: np.flatnonzero(np.r_[True, a.res_id[1:] != a.res_id[:-1]]) if len(a) else np.zeros(0, int)),
          "get_token_starts": lambda a: np.flatnonzero(np.r_[True, a.token_id[1:] != a.token_id[:-1]]) if len(a) else np.zeros(0, int),
          "MoleculeType": lambda ids: int(np.asarray(ids).ravel()[0]), "MOLECULE_TYPE_TO_RESIDUES_3": {0: ["ALA", "GLY", "SER"]},
          "TemplateSlice": lambda **kw: types.SimpleNamespace(**kw)}
    exec(compile(mod, STOCK, "exec"), ns)
    return ns["map_token_pos_to_template_residues"]


class Entry:
    def __init__(self, idx_map, name):
        self.idx_map = np.asarray(idx_map, dtype=int); self.template_pdb_chain_id = name; self.index = 0; self.release_date = "2020-01-01"


class CacheEntry:
    """TemplateCacheEntry's constructor surface (structure/template.py: index, release_date, idx_map)."""
    def __init__(self, index, release_date, idx_map):
        self.index, self.release_date, self.idx_map = index, release_date, idx_map


def stock_sample_function():
    """sample_templates compiled from the stock source (structure/template.py:126-253) with numpy / pathlib and the cache-entry stand-in."""
    src = open(_stubs.stock_src("openfold3", "core", "data", "primitives", "structure", "template.py"), encoding="utf-8").read()   # = STOCK; skips by name while stock/src is absent
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "sample_templates")
    fn.decorator_list = []
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {"np": np, "Path": Path, "Any": object, "TemplateCacheEntry": CacheEntry, "logger": types.SimpleNamespace(warning=lambda *a, **k: None)}
    exec(compile(mod, STOCK, "exec"), ns)
    return ns["sample_templates"]


def write_cache_entry(path, template_ids, n=10):
    """A template cache entry .npz as the TemplatePreprocessor writes it: one pickled dict per template id (index, release_date, idx_map)."""
    np.savez(str(path), **{tid: np.array({"index": i, "release_date": "2020-01-01", "idx_map": np.asarray([[q, 100 + q] for q in range(1, n + 1)])}, dtype=object)
                          for i, tid in enumerate(template_ids)})
    return Path(str(path) if str(path).endswith(".npz") else str(path) + ".npz")


def structure_array_dir(root, template_ids, fmt="npz"):
    """template_structure_array_directory with `<pdb id>/<template id>.<fmt>` present for the given ids (stock filters ids by these files)."""
    for tid in template_ids:
        d = os.path.join(root, tid.split("_")[0])
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, f"{tid}.{fmt}"), "wb").close()
    return Path(root)


@pytest.fixture()
def stock_modules(monkeypatch):
    """Two fake modules under the stock names: the struct module carries the STOCK map function + an align loop shaped like stock's
    (map looked up as a module global); the pipeline module loops chains calling align as ITS module global (stock's shape)."""
    struct = types.ModuleType(templ_guard.STRUCT_MODULE)
    struct.map_token_pos_to_template_residues = stock_map_function()
    struct.sample_templates = stock_sample_function()

    def align_template_to_query(sampled_template_data, template_structures_directory, template_structure_array_directory, template_file_format, ccd, atom_array_query_chain):
        slices = []
        for name, (entry, tmpl_arr) in sampled_template_data.items():
            struct.map_token_pos_to_template_residues(slices, entry, atom_array_query_chain, tmpl_arr)
        return slices
    struct.align_template_to_query = align_template_to_query
    pipe = types.ModuleType(templ_guard.PIPELINE_MODULE)
    pipe.align_template_to_query = align_template_to_query
    pipe.sample_templates = struct.sample_templates                       # imported as a module global in stock's pipeline module

    def process_template_structures_of3(chains, assembly_data=None, **sample_kw):
        """Stock's loop shape: per chain, sample (when assembly data is given) then align, both looked up as THIS module's globals."""
        out = {}
        for cid, (q, sampled) in chains.items():
            if assembly_data is not None:
                got = pipe.sample_templates(assembly_data=assembly_data, template_cache_directory=None, chain_id=cid, **sample_kw)
                sampled = {tid: (Entry(e.idx_map, tid), FULL) for tid, e in got.items()}
            out[cid] = pipe.align_template_to_query(sampled_template_data=sampled, template_structures_directory=None, template_structure_array_directory=None,
                                                    template_file_format="cif", ccd=None, atom_array_query_chain=q)
        return out
    pipe.process_template_structures_of3 = process_template_structures_of3
    monkeypatch.setitem(sys.modules, templ_guard.STRUCT_MODULE, struct)
    monkeypatch.setitem(sys.modules, templ_guard.PIPELINE_MODULE, pipe)
    for k in COUNTER_KEYS:
        monkeypatch.setitem(templ_guard.CENSUS, k, 0 if k != "installed" else False)
    monkeypatch.setitem(templ_guard.CENSUS, "events", [])
    monkeypatch.setattr(templ_guard, "DECLARED", {})
    templ_guard.install()
    yield struct, pipe
    if templ_guard.FINDER in sys.meta_path:
        sys.meta_path.remove(templ_guard.FINDER)


QUERY = chain(range(1, 11))                                              # 10 residues, one token each
IDX = [[q, 100 + q] for q in range(1, 11)]                               # SEQRES-numbered alignment: query 1..10 -> template 101..110
FULL = chain(range(101, 111), chain_id="T")                              # every aligned residue resolved
PARTIAL = chain(range(101, 107), chain_id="T")                           # 107..110 unresolved in the deposited model (the 8IO9 shape)


def test_stock_drops_silently_and_the_guard_names_it(stock_modules, capsys):
    struct, _ = stock_modules
    slices = []
    struct.map_token_pos_to_template_residues(slices, Entry(IDX, "8io9_B"), QUERY, PARTIAL)
    assert slices == []                                                  # stock: dropped, nothing appended
    err = capsys.readouterr().err
    assert "[openfold3_ob0-opt] TEMPLATE DROPPED chain=A template=8io9_B reason=unresolved_aligned_residues=4/10" in err, err
    assert templ_guard.CENSUS["dropped"] == 1 and templ_guard.CENSUS["events"][0]["reason"] == "unresolved_aligned_residues=4/10"
    struct.map_token_pos_to_template_residues(slices, Entry(IDX, "1abc_A"), QUERY, FULL)
    assert len(slices) == 1 and list(slices[0].template_residue_repeats) == [1] * 10          # a resolvable template is kept, no event
    assert "DROPPED" not in capsys.readouterr().err


def test_census_per_chain_and_all_dummy_is_an_event_not_a_refusal(stock_modules, capsys):
    _, pipe = stock_modules
    four_bad = {f"8io9_{c}": (Entry(IDX, f"8io9_{c}"), PARTIAL) for c in "BCDE"}         # 4/4 slots drop (8IO9) for a consuming caller
    out = pipe.process_template_structures_of3({"A": (QUERY, four_bad)})
    assert out == {"A": []}                                                                # the stock result is returned: nothing raises
    err = capsys.readouterr().err
    assert "[openfold3_ob0-opt] TEMPLATES chain=A real=0/4 dropped=4" in err and "[openfold3_ob0-opt] TEMPLATE SLOTS ALL DUMMY (A:0/4; dropped: A/8io9_B:unresolved_aligned_residues=4/10" in err, err
    assert "refused" not in err and templ_guard.CENSUS["all_dummy"] == 1 and templ_guard.CENSUS["sampled"] == 4 and templ_guard.CENSUS["real"] == 0


def test_partial_census_is_printed(stock_modules, capsys):
    _, pipe = stock_modules
    three_of_four = {"8io9_B": (Entry(IDX, "8io9_B"), PARTIAL), "1abc_A": (Entry(IDX, "1abc_A"), FULL), "1abc_B": (Entry(IDX, "1abc_B"), FULL), "1abc_C": (Entry(IDX, "1abc_C"), FULL)}
    out = pipe.process_template_structures_of3({"A": (QUERY, three_of_four), "B": (chain(range(1, 11), chain_id="B"), {"7vcu_X": (Entry(IDX, "7vcu_X"), PARTIAL)})})
    assert len(out["A"]) == 3 and out["B"] == []                          # chain B alone is all-dummy but the QUERY kept real slots: no refusal
    err = capsys.readouterr().err
    assert "TEMPLATES chain=A real=3/4 dropped=1" in err and "TEMPLATES chain=B real=0/1 dropped=1" in err and "refused" not in err, err
    c = templ_guard.counters()
    assert (c["templ_guard"], c["templ_sampled"], c["templ_real"], c["templ_dropped"], c["templ_all_dummy"]) == ("on", 5, 3, 2, 0), c
    assert c["templ_policy"] == "consume(upstream_0.5.0)", c


def test_untemplated_query_passes_through_silently(stock_modules, capsys):
    _, pipe = stock_modules
    out = pipe.process_template_structures_of3({"A": (QUERY, {}), "B": (chain(range(1, 6), chain_id="B"), {})})
    assert out == {"A": [], "B": []} and capsys.readouterr().err == "" and templ_guard.counters()["templ_sampled"] == 0


def test_install_is_idempotent_and_patches_late_imports(monkeypatch):
    monkeypatch.setitem(templ_guard.CENSUS, "installed", False)          # restored after the test: other tests read an uninstalled guard
    monkeypatch.delitem(sys.modules, templ_guard.STRUCT_MODULE, raising=False)
    monkeypatch.delitem(sys.modules, templ_guard.PIPELINE_MODULE, raising=False)
    templ_guard.install(); templ_guard.install()
    assert sys.meta_path.count(templ_guard.FINDER) == 1
    struct = types.ModuleType(templ_guard.STRUCT_MODULE)
    struct.map_token_pos_to_template_residues = lambda *a: None
    struct.align_template_to_query = lambda *a, **k: []
    struct.sample_templates = lambda *a, **k: {}
    templ_guard.patch_struct_module(struct)                              # what the finder does after the module body runs
    assert struct.map_token_pos_to_template_residues._of3opt_guard and struct.align_template_to_query._of3opt_guard and struct.sample_templates._of3opt_guard
    templ_guard.patch_struct_module(struct)                              # twice: no double wrap
    assert not getattr(struct.map_token_pos_to_template_residues.__wrapped__, "_of3opt_guard", False)
    sys.meta_path.remove(templ_guard.FINDER)


def test_finder_is_not_a_kit_hook():
    from openfold3_ob0_opt import env
    assert env.kit_hooks_installed(home=HOME, meta_path=[templ_guard.FINDER]) == []     # the ENV-CLEAN / one-route census counts kit hooks by the file defining the finder (a hook directory); the guard's is the package's own module


# --------------------------------------------------------------------------------------------- the sampler without a cache directory (the ignore policy's shape): stock untouched, named
class FakeChain:
    def __init__(self, chain_ids, aln, ids=(), moltype="PROTEIN", sequence="A" * 10):
        self.chain_ids, self.template_alignment_file_path, self.template_entry_chain_ids = list(chain_ids), aln, list(ids)
        self.molecule_type, self.sequence = moltype, sequence


def fake_preprocessor(queries, outcome):
    """A TemplatePreprocessor stand-in with stock's two inference methods' EFFECTS: parse = collect the inputs; update = per chain the cache
    entry path (None when preprocessing failed) and template_entry_chain_ids = ALL ids of the cache entry (`outcome[(query, idx)] = (ok, ids)`)."""
    mod = types.ModuleType(templ_guard.PREPROC_MODULE)

    class TemplatePreprocessor:
        moltypes = ("PROTEIN",)

        def __init__(self):
            self.input_set = types.SimpleNamespace(queries=queries)
            self.inputs = None

        def _parse_inference_query_set(self):
            self.inputs = [(qn, i) for qn, q in self.input_set.queries.items() for i, ch in enumerate(q.chains) if ch.template_alignment_file_path is not None]

        def _update_inference_query_set(self):
            for qn, q in self.input_set.queries.items():
                for i, ch in enumerate(q.chains):
                    ok, ids = outcome.get((qn, i), (False, []))
                    ch.template_alignment_file_path = Path(f"/cache/{qn}_{i}.npz") if ok else None
                    ch.template_entry_chain_ids = list(ids)
    mod.TemplatePreprocessor = TemplatePreprocessor
    templ_guard.patch_preproc_module(mod)
    return mod.TemplatePreprocessor


def test_a_templated_chain_is_not_consumed_exactly_as_stock_and_named(stock_modules, tmp_path, capsys):
    struct, _ = stock_modules
    ids = ["1abc_A", "2xyz_B"]
    npz = write_cache_entry(tmp_path / "entry", ids)
    arr = structure_array_dir(str(tmp_path / "arrays"), ids)                                             # a real, consumable cache entry + structure arrays on disk
    assembly = {"A": {"template_ids": ids, "cache_entry_file_path": npz}, "B": {"template_ids": [], "cache_entry_file_path": None}}
    kw = dict(n_templates=4, take_top_k=True, template_structure_array_directory=arr, template_file_format="npz")
    stock = struct.sample_templates.__wrapped__
    assert stock(assembly_data=assembly, template_cache_directory=None, chain_id="A", **kw) == {}          # STOCK predict: the preprocessed entry is never read
    assert struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="A", **kw) == {}   # the guarded call: the same — not consumed
    err = capsys.readouterr().err
    assert "[openfold3_ob0-opt] TEMPLATES chain=A ids=2 sampled=0: templates parsed; not consumed at inference (upstream 0.5.0 behaviour)" in err, err
    assert struct.sample_templates(assembly_data=assembly, template_cache_directory=None, chain_id="B", **kw) == {} and capsys.readouterr().err == ""   # untemplated chain: untouched, silent
    c = templ_guard.counters()
    assert c["templ_ignored"] == 1 and c["templ_sampled"] == 0 and c["templ_policy"] == "consume(upstream_0.5.0)", c
    got = struct.sample_templates(assembly_data=assembly, template_cache_directory=tmp_path, chain_id="A", **kw)   # a consuming CALLER (its own cache directory): stock's branch, untouched, uncounted
    assert sorted(got) == ids and templ_guard.counters()["templ_ignored"] == 1 and capsys.readouterr().err == ""


def test_a_templated_query_is_accepted_and_parsed_and_the_one_line_is_printed(stock_modules, capsys):
    aln = Path("/aln/q")
    queries = {"q1": types.SimpleNamespace(chains=[FakeChain(["A", "B"], aln, ids=["x2", "x1", "zz"]), FakeChain(["C"], aln), FakeChain(["L"], None, moltype="LIGAND")]),
               "q2": types.SimpleNamespace(chains=[FakeChain(["A"], aln)])}
    TP = fake_preprocessor(queries, {("q1", 0): (True, ["x1", "x2", "x3"]), ("q1", 1): (False, []), ("q2", 0): (True, ["y1"])})
    pre = TP()
    pre._parse_inference_query_set()                                                                       # accepted: stock's parse ran (no refusal by name)
    assert pre.inputs == [("q1", 0), ("q1", 1), ("q2", 0)]
    err = capsys.readouterr().err
    assert err.count("\n") == 1 and "[openfold3_ob0-opt] TEMPLATES PARSED queries=2 chains=3: templates parsed; consumed at inference (upstream 0.5.0 behaviour)" in err, err
    pre._update_inference_query_set()
    err = capsys.readouterr().err
    assert "TEMPLATE INPUT DROPPED query=q1 chain=C reason=preprocess_failed" in err and "refused" not in err, err
    assert queries["q1"].chains[0].template_entry_chain_ids == ["x1", "x2", "x3"] and queries["q2"].chains[0].template_entry_chain_ids == ["y1"]   # stock's id lists stand (no id policy)
    c = templ_guard.counters()
    assert c["templ_parsed"] == 3 and c["templ_alive"] == 2 and c["templ_input_dropped"] == 1 and c["templ_parsed"] == c["templ_alive"] + c["templ_input_dropped"], c
    # a query whose declared chains ALL fail preprocessing: named per chain, never refused (stock predicts it untemplated too)
    monkeypatch_declared = dict(templ_guard.DECLARED)
    dead = {"q3": types.SimpleNamespace(chains=[FakeChain(["A"], aln), FakeChain(["B"], aln, ids=["nope"])])}
    TP3 = fake_preprocessor(dead, {("q3", 0): (False, []), ("q3", 1): (False, [])})
    pre3 = TP3(); pre3._parse_inference_query_set(); pre3._update_inference_query_set()
    err = capsys.readouterr().err
    assert "TEMPLATE INPUT DROPPED query=q3 chain=A reason=preprocess_failed" in err and "TEMPLATE INPUT DROPPED query=q3 chain=B reason=preprocess_failed" in err and "refused" not in err, err
    assert set(templ_guard.DECLARED) == set(monkeypatch_declared) | {("q3", 0), ("q3", 1)}
    untemplated = fake_preprocessor({"q0": types.SimpleNamespace(chains=[FakeChain(["A"], None)])}, {})()
    untemplated._parse_inference_query_set()                                                               # nothing declared: untouched, silent
    assert untemplated.inputs == [] and capsys.readouterr().err == ""


def test_featurizer_returns_the_stock_result_for_templated_chains_that_consumed_nothing(stock_modules, tmp_path, capsys):
    _, pipe = stock_modules
    ids = ["1abc_A"]
    npz = write_cache_entry(tmp_path / "entry", ids)
    arr = structure_array_dir(str(tmp_path / "arrays"), ids)
    assembly = {"A": {"template_ids": ids, "cache_entry_file_path": npz}}
    kw = dict(n_templates=4, take_top_k=True, template_structure_array_directory=arr, template_file_format="npz")
    out = pipe.process_template_structures_of3({"A": (QUERY, {})}, assembly_data=assembly, **kw)          # stock's loop: sample (None directory) then align
    assert out == {"A": []}                                                                                # no template slot: the dummy slots stand, as stock
    err = capsys.readouterr().err
    assert "TEMPLATES chain=A ids=1 sampled=0: templates parsed; not consumed at inference (upstream 0.5.0 behaviour)" in err and "refused" not in err and "ALL DUMMY" not in err, err
    c = templ_guard.counters()
    assert (c["templ_ignored"], c["templ_sampled"], c["templ_real"], c["templ_all_dummy"]) == (1, 0, 0, 0), c


def test_input_dropped_names_the_cache_entry_the_engine_looked_for(stock_modules, tmp_path, capsys, monkeypatch):
    """A declared chain left without a cache entry is named with the entry the ENGINE keys — 0.4.1: `<sha256(sequence)>.npz`
    (get_sequence_hash), 0.5.0: `seq-<sha>.aln-<sha>.npz` (build_template_cache_key) — and, under a 0.5.0-shaped module, a directory that holds
    the sequence's entry under openfold3's 0.4.1 key says so by name (of3_keyed_cache_entry) instead of running the chain untemplated silently."""
    import hashlib
    aln = tmp_path / "hits.m8"; aln.write_text("q\tt\n")
    seq_sha = hashlib.sha256(b"MKV").hexdigest()

    def engine(build: bool):
        queries = {"q": types.SimpleNamespace(chains=[FakeChain(["A"], aln, sequence="MKV")])}
        TP = fake_preprocessor(queries, {("q", 0): (False, [])})
        mod = types.ModuleType(templ_guard.PREPROC_MODULE)
        mod.get_sequence_hash = lambda s: hashlib.sha256(s.encode()).hexdigest()
        if build:                                                                                          # 0.5.0's key statement, shape only: seq-<sha>.aln-<sha of the file's bytes>
            mod.build_template_cache_key = lambda *, sequence, aln_path=None, cif_paths=None, cif_chain_ids=None: (
                f"seq-{mod.get_sequence_hash(sequence)}.aln-{hashlib.sha256(Path(aln_path).read_bytes()).hexdigest()}")
        monkeypatch.setitem(sys.modules, templ_guard.PREPROC_MODULE, mod)
        pre = TP(); pre.cache_directory = tmp_path
        pre._parse_inference_query_set(); capsys.readouterr()
        pre._update_inference_query_set()
        return capsys.readouterr().err

    err = engine(build=False)
    assert f"TEMPLATE INPUT DROPPED query=q chain=A reason=cache_key_miss:{seq_sha}.npz" in err, err
    err = engine(build=True)
    aln_sha = hashlib.sha256(aln.read_bytes()).hexdigest()
    assert f"TEMPLATE INPUT DROPPED query=q chain=A reason=cache_key_miss:seq-{seq_sha}.aln-{aln_sha}.npz" in err, err
    (tmp_path / f"{seq_sha}.npz").write_bytes(b"")                                                        # the same sequence's entry under OpenFold3 0.4.1's key sits in the directory
    err = engine(build=True)
    assert f"TEMPLATE INPUT DROPPED query=q chain=A reason=of3_keyed_cache_entry:{seq_sha}.npz" in err, err
    err = engine(build=False)                                                                              # a 0.4.1 engine READS that entry's key: stock found it absent only because the stub says so — the word stays a plain miss
    assert f"reason=cache_key_miss:{seq_sha}.npz" in err, err


def test_no_line_consumes_or_refuses_templates_and_no_switch_reaches_a_policy():
    """The pin: templates behave as upstream 0.5.0 on every line — no Line carries a template policy, the modes table has no policy words,
    the guard has no policy variable / refusal class / raise statement, install() takes no policy, and the CLI has no template-policy flag."""
    import inspect
    from openfold3_ob0_opt import cli, modes
    assert templ_guard.policy() == ("consume", "upstream_0.5.0") and templ_guard.counters()["templ_policy"] == "consume(upstream_0.5.0)"
    for key, ln in modes.LINES.items():
        assert not hasattr(ln, "templates"), key                                                           # no per-line template policy: one behaviour, stock's
    for name in ("TEMPLATE_POLICIES", "template_policy", "TP_TEMPLATED_ENV"):
        assert not hasattr(modes, name), name
    for name in ("ENV_POLICY", "ENV_IDS", "ENV_ALLOW_UNTEMPLATED", "POLICIES", "SENTINEL_CACHE_DIR", "REFUSAL", "REFUSAL_BY_LINE", "REFUSAL_NOT_CONSUMED",
                 "TemplatesRefusedByLine", "TemplatesDeclaredNotConsumed", "TemplateSlotsAllDummy", "ids_mode", "allow_untemplated"):
        assert not hasattr(templ_guard, name), name
    assert list(inspect.signature(templ_guard.install).parameters) == []
    tree = ast.parse(inspect.getsource(templ_guard))
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Raise)]                                     # events only: nothing in the guard raises into the run
    assert "os.environ" not in inspect.getsource(templ_guard)                                             # no environment switch reaches the guard
    ap = cli.build_parser()
    base = ["pred", "--mode", "exact", "--query-json", "q.json", "--output-dir", "o"]
    assert ap.parse_args(base + ["--use-templates", "true"]).use_templates == "true"                      # OpenFold3's own switch stays (stock parses templates under it)
    for gone in (["--templates", "consume"], ["--templates", "refuse"], ["--template-ids", "declared"], ["--allow-untemplated"]):
        with pytest.raises(SystemExit):
            ap.parse_args(base + gone)                                                                     # no template-policy flag exists
    assert not hasattr(cli, "apply_template_flags")
