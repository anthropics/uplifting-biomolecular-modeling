"""`hostfeat` (cells/of3_hostfeat.py + cells/hostfeat.py): the a3m deletion-matrix / letter-matrix statement and the MSA letter -> index table,
proven equal — dtype, shape, every element — to the ENGINE'S OWN TEXT (sliced from stock/src: parse_a3m + parse_fasta + _msa_list_to_np run
verbatim on a stub MsaArray; core/data/resources/residues.py loaded as it is) on random alignments and the edge cases (leading / trailing
insertions, gaps, unknown letters, one row, ragged rows and non-ASCII text -> the engine's statement); the patch mechanics through a real import
of a file-backed package carrying the engine's texts (digest accepted -> patched + registry re-pointed; unknown text -> that part refused by
name; parts word; census words); the kit wiring (every kit line but off exports the switch, registry row, probe, .pth declaration, LEVERS_OFF)."""
import ast
import importlib
import importlib.util
import os
import random
import string
import sys
import textwrap
import types

import numpy as np
import pytest

from openfold3_opt.tests import _stubs

HOME = _stubs.tree_home()
SRC = os.path.join(HOME, "stock", "src", "openfold3")
MSA_IO = os.path.join(SRC, "core", "data", "io", "sequence", "msa.py")
FASTA = os.path.join(SRC, "core", "data", "io", "sequence", "fasta.py")
RESIDUES = os.path.join(SRC, "core", "data", "resources", "residues.py")


def _live():
    g = globals()
    g["_autoload"], g["modes"], g["registry"], g["stack"] = (importlib.import_module("openfold3_opt." + n) for n in ("_autoload", "modes", "registry", "stack"))
    g["core"] = importlib.import_module("openfold3_opt.cells.of3_hostfeat")
    g["cell"] = importlib.import_module("openfold3_opt.cells.hostfeat")


_live()


@pytest.fixture(autouse=True)
def live_modules():
    _live()
    core.uninstall()
    yield
    core.uninstall()


def _fn_text(path, name):
    src = open(path, encoding="utf-8").read(); lines = src.splitlines(keepends=True)
    for n in ast.walk(ast.parse(src)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return "".join(lines[(n.decorator_list[0].lineno if n.decorator_list else n.lineno) - 1:n.end_lineno])
    raise AssertionError(name)


ENGINE_MSA_TEXT = '''
import string
from collections.abc import Sequence
import numpy as np

class MsaArray:
    """stub: records what the engine's tail hands it"""
    def __init__(self, msa, deletion_matrix, metadata):
        self.msa, self.deletion_matrix, self.metadata, self.truncated = msa, deletion_matrix, metadata, None
    def truncate(self, row_slice, inplace=False):
        self.truncated = (row_slice, inplace)
        return None if inplace else self

''' + "\n".join(textwrap.dedent(_fn_text(p, n)) for p, n in ((FASTA, "parse_fasta"), (MSA_IO, "_msa_list_to_np"), (MSA_IO, "parse_a3m"))) + '''

def parse_stockholm(s, max_seq_count=None):
    raise NotImplementedError

MSA_PARSER_REGISTRY = {".a3m": parse_a3m, ".sto": parse_stockholm}
'''


def _engine_msa_module(name="nfh_engine_msa"):
    m = types.ModuleType(name); m.__file__ = "<engine text>"
    exec(compile(ENGINE_MSA_TEXT, MSA_IO, "exec"), m.__dict__)
    return m


def _residues_module(name="nfh_engine_residues"):
    spec = importlib.util.spec_from_file_location(name, RESIDUES)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _random_a3m(rng, n_rows, L, p_ins=0.15, p_gap=0.1, alphabet=string.ascii_uppercase + "-", trailing=True):
    rows, meta = [], []
    for i in range(n_rows):
        out = []
        for j in range(L):
            while rng.random() < p_ins:                       # insertions (lowercase) before this column, possibly several
                out.append(rng.choice(string.ascii_lowercase))
            out.append("-" if rng.random() < p_gap else rng.choice(alphabet))
        if trailing:
            while rng.random() < p_ins:
                out.append(rng.choice(string.ascii_lowercase))
        rows.append("".join(out)); meta.append(f"seq{i} some description|{i}")
    text = "".join(f">{m}\n{r}\n" for m, r in zip(meta, rows))
    return text, rows, meta


def _same(a, b):
    assert isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
    assert a.dtype == b.dtype and a.shape == b.shape and a.flags["C_CONTIGUOUS"] == b.flags["C_CONTIGUOUS"], (a.dtype, b.dtype, a.shape, b.shape)
    assert np.array_equal(a, b)


def test_a3m_arrays_are_the_engines_deletion_matrix_and_letters_exactly():
    E = _engine_msa_module()
    rng = random.Random(260911)
    cases = [_random_a3m(rng, n, L, p, g) for n, L, p, g in ((1, 1, 0.0, 0.0), (1, 40, 0.3, 0.2), (7, 25, 0.2, 0.1), (60, 120, 0.1, 0.3), (300, 33, 0.5, 0.5), (12, 300, 0.05, 0.0))]
    cases.append((">q\nMKV\n>h1\naaMK-v\n>h2\n-K-\n", None, None))           # leading insertions, trailing insertion, gaps
    cases.append((">q\nAX9*\n>h\nA.9*bb\n", None, None))                      # unknown letters, digits, punctuation: not lowercase -> kept, like the engine
    cases.append((">only\nACDEFGHIKLMNPQRSTVWY\n", None, None))               # one row
    cases.append((">q desc\r\nAB\r\n\r\n#comment\n>h\nabAB\n", None, None))   # CRLF, blank and comment lines (parse_fasta's rules)
    for text, _, _ in cases:
        ref = E.parse_a3m(text)
        seqs, meta = E.parse_fasta(text)
        got = core.a3m_arrays(seqs)
        assert got is not None, text[:60]
        _same(got[0], ref.deletion_matrix); _same(got[1], ref.msa)
        assert got[0].dtype == np.array([0]).dtype and got[1].dtype == np.dtype("<U1")


def test_inputs_outside_the_byte_identity_go_to_the_engines_statement():
    assert core.a3m_arrays([]) is None
    assert core.a3m_arrays(["ABC", "AB"]) is None                              # ragged aligned rows: the engine's np.array / row assignment decide (raise)
    assert core.a3m_arrays(["abc"]) is None                                    # no aligned column at all
    assert core.a3m_arrays(["AB\u00e9C".replace("\u00e9", "\u00e9")]) is None  # non-ASCII: str.islower / translate semantics differ from the byte view
    E = _engine_msa_module()
    f = core.make_parse_a3m(E, E.parse_a3m, "v041")
    r = f(">q\nA\u00c9B\n")                                                    # falls back to the engine's function: its arrays, its tail
    _same(r.msa, E.parse_a3m(">q\nA\u00c9B\n").msa)
    with pytest.raises(ValueError):
        f(">q\nABC\n>h\nAB\n")                                                 # ragged: the engine raises (np.array of ragged lists) — so does the port, from the engine's text
    with pytest.raises(ValueError):
        E.parse_a3m(">q\nABC\n>h\nAB\n")


def test_the_restated_parse_a3m_hands_the_engines_tail_the_same_arrays_and_truncates_as_the_text_does():
    E = _engine_msa_module()
    rng = random.Random(7)
    text, rows, meta = _random_a3m(rng, 40, 50, 0.2, 0.2)
    for variant in ("v041",):
        f = core.make_parse_a3m(E, E.parse_a3m, variant)
        got, ref = f(text, 10), E.parse_a3m(text, 10)
        _same(got.msa, ref.msa); _same(got.deletion_matrix, ref.deletion_matrix)
        assert list(got.metadata) == list(ref.metadata) == meta and got.truncated == ref.truncated == (10, False)   # 0.4.1's tail: truncate(max) as written
        assert f(text).truncated is None
    E5 = _engine_msa_module("nfh_engine_msa_050")
    E5.MsaArray.from_parsed = classmethod(lambda cls, msa, deletion_matrix, metadata: cls(msa, deletion_matrix, metadata))
    f5 = core.make_parse_a3m(E5, E5.parse_a3m, "v050")
    assert f5(text, 10).truncated == (10, True)                                # 0.5.0's tail: from_parsed + truncate(inplace=True)


def test_the_index_table_is_the_engines_mapping_for_every_molecule_type_and_every_byte():
    R = _residues_module()
    orig = R.map_str_array_to_idx_array
    f = core.make_map_str_array_to_idx_array(orig)
    rng = np.random.default_rng(5)
    letters = np.array(list(string.ascii_uppercase + string.ascii_lowercase + string.digits + "-.*? "), dtype="<U1")
    def same_or_same_error(arr, mt):
        try:
            ref = orig(arr, mt)
        except Exception as e:                                                 # 0.4.1 raises for RNA / DNA (its unknown-letter lookup): the port raises the same, from the engine's text
            with pytest.raises(type(e)):
                f(arr, mt)
            return False
        _same(f(arr, mt), ref)
        return True
    served = set()
    for mt in R.MoleculeType:
        for shape in ((1, 1), (3, 17), (64, 129), (5, 0)):
            arr = letters[rng.integers(0, len(letters), size=shape)]
            if same_or_same_error(arr, mt):
                served.add(mt)
        every = np.array([chr(i) for i in range(1, 256)], dtype="<U1").reshape(1, 255)          # every non-NUL byte value ('<U1' cannot hold NUL: it reads back as '')
        same_or_same_error(every, mt)
        col = np.ascontiguousarray(letters[rng.integers(0, len(letters), size=(9, 4))].T)      # a transposed-then-copied matrix; and a non-contiguous view
        same_or_same_error(col, mt); same_or_same_error(col[:, ::2], mt)
    assert R.MoleculeType.PROTEIN in served
    wide = np.array([["A", "\u03a9"]], dtype="<U1")                                              # a code point >= 256: the engine's statement
    same_or_same_error(wide, R.MoleculeType.PROTEIN)
    u3 = np.array([["ALA", "GLY"]])                                                              # not '<U1': the engine's statement (whatever it does with it — 0.5.0 raises)
    same_or_same_error(u3, R.MoleculeType.PROTEIN)


@pytest.fixture()
def engine_pkg(tmp_path, monkeypatch):
    """A file-backed package `nfh_of3.{msa,residues}` carrying the engine's texts, imported through the real path finder (so the cell's finder
    and digest gate run as they run in the predicting process)."""
    d = tmp_path / "nfh_of3"; d.mkdir()
    (d / "__init__.py").write_text("")
    (d / "msa.py").write_text(ENGINE_MSA_TEXT)
    (d / "residues.py").write_text(open(RESIDUES, encoding="utf-8").read())
    (d / "user.py").write_text("from nfh_of3.residues import map_str_array_to_idx_array\nfrom nfh_of3.msa import parse_a3m\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    before = set(sys.modules)
    saved = {k: getattr(core, k) for k in core.CONFIGURABLE}
    core.configure(ENV="NFH_HOSTFEAT", ENV_PARTS="NFH_HOSTFEAT_PARTS", PREFIX="[nfh/hostfeat]", PARTS=("a3m", "msaidx"), M_MSA_IO="nfh_of3.msa", M_RESIDUES="nfh_of3.residues",
                   REBIND={"a3m": ("nfh_of3.user",), "msaidx": ("nfh_of3.user",)}, DIGESTS=dict(cell._core.DIGESTS) if False else
                   {"parse_a3m": dict(saved["DIGESTS"]["parse_a3m"]), "map_str_array_to_idx_array": dict(saved["DIGESTS"]["map_str_array_to_idx_array"])})
    yield types.SimpleNamespace(tmp=tmp_path)
    core.uninstall()
    core.configure(**saved)
    for k in set(sys.modules) - before:
        if k.startswith("nfh_of3"):
            del sys.modules[k]


def test_an_accepted_text_is_patched_at_import_the_registry_and_by_name_importers_re_pointed_and_the_features_equal(engine_pkg):
    st = core.install({"NFH_HOSTFEAT": "1"})
    assert st["state"] == "on" and core.FINDER in sys.meta_path
    user = importlib.import_module("nfh_of3.user")                              # imports both targets through the finder
    M, R = sys.modules["nfh_of3.msa"], sys.modules["nfh_of3.residues"]
    assert getattr(M.parse_a3m, core.MARK, False) and M.MSA_PARSER_REGISTRY[".a3m"] is M.parse_a3m and getattr(R.map_str_array_to_idx_array, core.MARK, False)
    assert user.parse_a3m is M.parse_a3m and user.map_str_array_to_idx_array is R.map_str_array_to_idx_array      # imported after the patch: the patched names
    assert st["patched"]["a3m"]["variant"] == "v041" and st["patched"]["a3m"]["registry"] == 1
    rng = random.Random(3)
    text, rows, meta = _random_a3m(rng, 30, 40, 0.2, 0.2)
    got, ref = M.parse_a3m(text), M.parse_a3m.__wrapped__(text)
    _same(got.msa, ref.msa); _same(got.deletion_matrix, ref.deletion_matrix)
    idx_got, idx_ref = R.map_str_array_to_idx_array(got.msa, R.MoleculeType.PROTEIN), R.map_str_array_to_idx_array.__wrapped__(ref.msa, R.MoleculeType.PROTEIN)
    _same(idx_got, idx_ref)
    line = core.census_line()
    for word in ("LEVER name=hostfeat", "state=on", "parts=a3m+msaidx", "a3m_calls=1", "a3m_rows=30", "a3m_fallback=0", "msaidx_calls=1", f"msaidx_cells={30 * 40}", "msaidx_fallback=0", "aside=none"):
        assert word in line, (word, line)
    core.uninstall()
    assert not getattr(M.parse_a3m, core.MARK, False) and M.MSA_PARSER_REGISTRY[".a3m"] is M.parse_a3m and user.parse_a3m is M.parse_a3m and not getattr(R.map_str_array_to_idx_array, core.MARK, False)


def test_already_imported_targets_are_patched_at_install_and_the_parts_word_switches_each_part_alone(engine_pkg):
    importlib.import_module("nfh_of3.user")                                     # the engine imported before activation (late arming)
    st = core.install({"NFH_HOSTFEAT": "1", "NFH_HOSTFEAT_PARTS": "msaidx"})
    M, R = sys.modules["nfh_of3.msa"], sys.modules["nfh_of3.residues"]
    assert st["parts"] == ("msaidx",) and not getattr(M.parse_a3m, core.MARK, False) and getattr(R.map_str_array_to_idx_array, core.MARK, False)
    assert sys.modules["nfh_of3.user"].map_str_array_to_idx_array is R.map_str_array_to_idx_array and st["patched"]["msaidx"]["rebound"] == 1   # the by-name importer re-pointed
    assert "parts=msaidx" in core.census_line() and core.FINDER not in sys.meta_path
    core.uninstall()
    with pytest.raises(ValueError):
        core.parts({"NFH_HOSTFEAT_PARTS": "a3m,templ"})
    st = core.install({"NFH_HOSTFEAT": "1", "NFH_HOSTFEAT_PARTS": "bogus"})    # a word the cell does not know: steps aside by name
    assert st["state"] == "refused" and st["reason"].startswith("bad_word:") and "state=refused" in core.census_line()


def test_a_text_the_port_does_not_restate_refuses_that_part_by_name_the_other_part_serving(engine_pkg):
    core.configure(DIGESTS={"parse_a3m": {"0" * 16: "v041"}, "map_str_array_to_idx_array": dict(core.DIGESTS["map_str_array_to_idx_array"])})
    st = core.install({"NFH_HOSTFEAT": "1"})
    importlib.import_module("nfh_of3.user")
    M, R = sys.modules["nfh_of3.msa"], sys.modules["nfh_of3.residues"]
    assert not getattr(M.parse_a3m, core.MARK, False) and getattr(R.map_str_array_to_idx_array, core.MARK, False)
    assert st["state"] == "on" and st["refused_parts"]["a3m"].startswith("digest:parse_a3m=") and "aside=a3m:digest:parse_a3m=" in core.census_line() and "parts=msaidx" in core.census_line()


def test_the_kits_digest_pins_are_the_engines_texts_in_the_tree_and_every_kit_line_but_off_carries_the_lever():
    import hashlib
    for fn, path, name in (("parse_a3m", MSA_IO, "parse_a3m"), ("map_str_array_to_idx_array", RESIDUES, "map_str_array_to_idx_array")):
        d = hashlib.sha256(textwrap.dedent(_fn_text(path, name)).encode("utf-8")).hexdigest()[:16]
        assert d in cell._core.DIGESTS[fn], (fn, d, cell._core.DIGESTS[fn])
    for key, ln in modes.LINES.items():
        carries = key[0] != "off"
        assert (ln.env.get(cell.ENV) == "1") is carries and ("hostfeat" in ln.levers) is carries and cell.ENV_PARTS not in ln.env, key
    lv = registry.LEVERS["hostfeat"]
    assert (lv.kit, lv.cls, lv.tier, lv.env_keys, lv.target) == ("cells", registry.FORWARD, registry.EXACT, modes.HOSTFEAT_ENVS, registry.T_MSA_IO)
    assert set(lv.modes) == {"exact", "fast", "big/resident", "big/tp"} and lv.marker == "[openfold3-opt/hostfeat] installed" and lv.source == "opt/openfold3_opt/cells/hostfeat.py"
    assert registry.tested_sm("hostfeat") == ("sm90",) and "hostfeat" in registry.ARCH_NOTES
    assert modes.HOSTFEAT_ENVS == (cell.ENV, cell.ENV_PARTS) and modes.HOSTFEAT_ENVS in modes.CELL_FAMILIES and modes.HOSTFEAT_ENVS in modes.ACTIVATION_FAMILIES
    assert "hostfeat" in modes.ACTIVATION_CELLS and "hostfeat" in stack._PROBES and modes.LEVER_SWITCHES["hostfeat"] == modes.HOSTFEAT_ENVS
    assert cell.ENV in _autoload.DECLARED and cell.ENV_PARTS in _autoload.DECLARED
    assert cell._core.M_MSA_IO == "openfold3.core.data.io.sequence.msa" and cell._core.M_RESIDUES == "openfold3.core.data.resources.residues" and cell._core.PARTS == ("a3m", "msaidx")
    r = modes.resolve("fast", HOME, environ={modes.ENV_LEVERS_OFF: "hostfeat"}, n_tokens=400)
    assert "hostfeat" not in r.levers and cell.ENV not in r.exports and "fastjson" in r.levers
