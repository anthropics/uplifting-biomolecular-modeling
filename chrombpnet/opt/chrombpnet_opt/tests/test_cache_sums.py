"""The shipped Triton caches are held file by file to their SHA256SUMS before apply() installs anything of them
(chrombpnet_k1.forward.shipped_cache_sums_refusal / hold_shipped_cache, under opt/kit/torch). Every shipped cache dir of this tree passes as it
is, silently, and a copy of it installs as before (every non-record file copied); one altered byte in one .cubin of a copy is refused by that
file's name with one stderr line and hold_shipped_cache gives the cold path (no dir, the note apply() prints); an extra, unlisted file in an
entry and a cache dir without SHA256SUMS are refused likewise; the record files at the top (JIT_IDENTITY.json, SHA256SUMS, DONE, SERVES) are
read, never installed, so an unlisted DONE does not count. No GPU: the loader module is imported by path from opt/kit/torch (torch and triton
must be importable, as for the rest of the K1 route)."""
import contextlib, hashlib, io, json, os, shutil, sys
import pytest
KIT_TORCH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "kit", "torch")   # opt/kit/torch: the K1 route's packages live there
pytest.importorskip("torch"); pytest.importorskip("triton"); pytest.importorskip("pandas")
if not os.path.isdir(os.path.join(KIT_TORCH, "chrombpnet_k1")):
    pytest.skip("opt/kit/torch is not in this tree", allow_module_level=True)
if KIT_TORCH not in sys.path:
    sys.path.insert(0, KIT_TORCH)
import chrombpnet_k1.forward as F  # noqa: E402

LIST = os.path.join(KIT_TORCH, "triton_cache_of_record.json")
DIRS = [d for d in sorted({os.path.join(KIT_TORCH, e["dir"]) for e in json.load(open(LIST)).get("entries", [])} if os.path.isfile(LIST) else ()) if os.path.isdir(d)]


def _non_record_files(d):
    out = []
    for base, _, fs in os.walk(d):
        for f in fs:
            rel = os.path.relpath(os.path.join(base, f), d)
            if not (os.path.dirname(rel) == "" and f in F.CACHE_RECORD_FILES):
                out.append(rel)
    return sorted(out)


@pytest.fixture
def cache_copy(tmp_path):
    """A copy of the first shipped cache dir under its own name -> (copy path, its name, its cubins relative to it)."""
    if not DIRS:
        pytest.skip("no shipped Triton cache dir in this tree")
    src = DIRS[0]; name = os.path.basename(src); c = str(tmp_path / name); shutil.copytree(src, c)
    cubins = [r for r in _non_record_files(src) if r.endswith(".cubin")]
    assert cubins, "the shipped cache carries no .cubin"
    return c, name, cubins


def test_the_shipped_caches_pass_silently_and_install_whole(tmp_path):
    if not DIRS:
        pytest.skip("no shipped Triton cache dir in this tree")
    for d in DIRS:
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            assert F.shipped_cache_sums_refusal(d) is None
        assert err.getvalue() == ""
        assert F.hold_shipped_cache(d) == (d, None)
        dst = str(tmp_path / ("dst_" + os.path.basename(d)))
        assert F._install_entries(d, dst) == len(_non_record_files(d)) > 0


def test_one_altered_cubin_refuses_the_whole_cache_by_name_and_gives_the_cold_path(cache_copy):
    c, name, cubins = cache_copy
    victim = cubins[0]; vp = os.path.join(c, victim)
    b = bytearray(open(vp, "rb").read()); b[min(64, len(b) - 1)] ^= 1
    with open(vp, "wb") as fh:
        fh.write(bytes(b))
    want = [ln.split()[0] for ln in open(os.path.join(c, "SHA256SUMS")) if ln.split() and ln.split()[1].lstrip("./") == victim][0]
    reason = "sha256 %s != SHA256SUMS %s" % (hashlib.sha256(bytes(b)).hexdigest()[:16], want[:16])
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        why = F.shipped_cache_sums_refusal(c)
    assert why == "%s/%s: %s" % (name, victim, reason)
    assert err.getvalue() == "[chrombpnet-opt] SHA256SUMS: refused %s/%s: %s\n" % (name, victim, reason)
    with contextlib.redirect_stderr(io.StringIO()):
        held = F.hold_shipped_cache(c)
    assert held[0] is None
    assert held[1].startswith("shipped cache REFUSED by its SHA256SUMS (%s/%s: %s) -> nothing of it installed; cold JIT" % (name, victim, reason))


def test_apply_holds_the_cache_before_its_install_step():
    src = open(F.__file__, encoding="utf-8").read()
    i = src.index("def apply("); j = src.index("hold_shipped_cache(cache_dir)", i); k = src.index("_install_entries(cache_dir", i)
    assert i < j < k


def test_an_unlisted_file_in_an_entry_is_refused_by_name(cache_copy):
    c, name, cubins = cache_copy
    extra = os.path.join(os.path.dirname(cubins[0]), "extra.cubin")
    with open(os.path.join(c, extra), "wb") as fh:
        fh.write(b"\0" * 16)
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        why = F.shipped_cache_sums_refusal(c)
    assert why == "%s/%s: not listed in its SHA256SUMS" % (name, extra)
    assert "refused %s/%s: not listed" % (name, extra) in err.getvalue()


def test_a_cache_without_sha256sums_is_refused(cache_copy):
    c, name, _cubins = cache_copy
    os.remove(os.path.join(c, "SHA256SUMS"))
    with contextlib.redirect_stderr(io.StringIO()):
        why = F.shipped_cache_sums_refusal(c)
        held = F.hold_shipped_cache(c)
    assert why is not None and why.startswith("%s/SHA256SUMS: absent" % name)
    assert held[0] is None


def test_top_level_record_files_are_not_held_but_the_same_name_inside_an_entry_is(cache_copy):
    c, name, cubins = cache_copy
    with open(os.path.join(c, "DONE"), "w") as fh:
        fh.write("x\n")
    with contextlib.redirect_stderr(io.StringIO()):
        assert F.shipped_cache_sums_refusal(c) is None
    entry = os.path.dirname(cubins[0])
    with open(os.path.join(c, entry, "DONE"), "w") as fh:
        fh.write("x\n")
    with contextlib.redirect_stderr(io.StringIO()):
        why = F.shipped_cache_sums_refusal(c)
    assert why == "%s/%s/DONE: not listed in its SHA256SUMS" % (name, entry)
