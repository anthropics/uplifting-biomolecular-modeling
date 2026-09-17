"""The triangle-attention package generates its per-ROWS kernel modules at first use.  On a writable install they go under the module's
own _gen/ directory (unchanged); under a read-only tree (container image file systems, shared read-only checkouts) they go to the kit's
JIT root, else the temp dir — byte-identical source.  A file already at that name is compared with the generated text and never imported:
the module runs the text generated in this process; a directory that group or others can write, or that belongs to another user, is refused
by name and skipped.  CPU only: the helpers are exercised from the module source (no triton import needed), except the two-process
generation test, which imports the modules (torch + triton, no device)."""
import ast, hashlib, inspect, os, stat, subprocess, sys, tempfile, types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(os.path.dirname(HERE), "opt_core", "kernels", "triattn", "triattn_native", "pkg", "v11", "triattn_pkg")
FILES = {"k13": os.path.join(PKG, "dispatch", "kernels", "k13.py"), "k10": os.path.join(PKG, "triton", "triattn", "k10.py"),
         "k11": os.path.join(PKG, "triton", "triattn", "k11.py"), "k12": os.path.join(PKG, "triton", "triattn", "k12.py")}
REL = {"k13": os.path.join("triattn_pkg", "dispatch", "kernels", "_gen")}
for _k in ("k10", "k11", "k12"):
    REL[_k] = os.path.join("triattn_pkg", "triton", "triattn", "_gen")


TAIL = ["_gen_dirs", "_gen_file", "_gen_why_not", "_gen_found", "_GEN_SAID", "_gen_say", "_gen_exec"]    # the module's last statements, in order


def _namespace(name, gen_dir):
    """The generated-source helpers of module `name` (its tail, from _gen_dirs on), executed from source with _GEN_DIR bound to `gen_dir`."""
    src = open(FILES[name], encoding="utf-8").read(); tree = ast.parse(src)
    first = [i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "_gen_dirs"]
    assert len(first) == 1, name
    tail = tree.body[first[0]:]
    assert [getattr(n, "name", None) or n.targets[0].id for n in tail] == TAIL, name
    ns = {"os": os, "sys": sys, "_GEN_DIR": gen_dir}
    exec(compile(ast.Module(body=tail, type_ignores=[]), FILES[name], "exec"), ns)
    return ns


def _helpers(name, gen_dir):
    ns = _namespace(name, gen_dir)
    return ns["_gen_dirs"], ns["_gen_file"]


def _ro(path):
    os.chmod(path, stat.S_IRUSR | stat.S_IXUSR)
    try:
        open(os.path.join(path, ".probe"), "w").close()
    except OSError:
        return
    os.unlink(os.path.join(path, ".probe")); os.chmod(path, stat.S_IRWXU)
    pytest.skip("cannot make a directory read-only for this user")


@pytest.mark.parametrize("name", sorted(FILES))
def test_kernel_for_writes_through_gen_file_only(name):
    src = open(FILES[name], encoding="utf-8").read()
    assert "os.makedirs(_GEN_DIR" not in src and src.count("path = _gen_file(") == 1 and src.count("def _gen_file(") == 1 and "spec_from_file_location(name, path)" in src
    assert src.count("_gen_exec(mod, src, path)") == 1 and "exec_module(" not in src and "os.path.exists(path)" not in src   # the loader (and a .pyc beside the file) is never asked; no file is served because it exists


@pytest.mark.parametrize("name", sorted(FILES))
def test_writable_install_keeps_the_module_gen_dir(name, tmp_path, monkeypatch):
    gen = tmp_path / "tree" / "_gen"
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit")); monkeypatch.setenv("MODEL_OPT_STACK_KEY", "torch0-cu0-sm90")
    _gen_dirs, _gen_file = _helpers(name, str(gen))
    p = _gen_file("k_rows3_abc.py", "SRC = 3\n")
    assert p == str(gen / "k_rows3_abc.py") and open(p).read() == "SRC = 3\n"
    assert not (tmp_path / "jit").exists()                                   # the JIT root is not touched when the tree takes the write
    st = os.stat(p)
    assert _gen_file("k_rows3_abc.py", "SRC = 3\n") == p and (os.stat(p).st_ino, os.stat(p).st_mtime_ns) == (st.st_ino, st.st_mtime_ns)   # identical bytes: kept, not rewritten
    assert stat.S_IMODE(os.stat(str(gen)).st_mode) & 0o022 == 0 and not [f for f in os.listdir(str(gen)) if ".tmp" in f]


@pytest.mark.parametrize("name", sorted(FILES))
def test_read_only_tree_generates_under_the_jit_root_byte_identically(name, tmp_path, monkeypatch):
    tree = tmp_path / "tree"; tree.mkdir(); _ro(tree)                        # _gen cannot even be created
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit")); monkeypatch.setenv("MODEL_OPT_STACK_KEY", "torch0-cu0-sm90")
    try:
        _gen_dirs, _gen_file = _helpers(name, str(tree / "_gen"))
        p = _gen_file("k_rows2_def.py", "def _fwd():\n    return 2\n")
        assert p == str(tmp_path / "jit" / "torch0-cu0-sm90" / "opt_core_gen" / REL[name] / "k_rows2_def.py"), p
        assert open(p).read() == "def _fwd():\n    return 2\n"
        import importlib.util
        spec = importlib.util.spec_from_file_location("gen_probe_" + name, p); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        assert mod._fwd() == 2
    finally:
        os.chmod(tree, stat.S_IRWXU)


@pytest.mark.parametrize("name", sorted(FILES))
def test_no_key_and_unwritable_root_fall_to_the_temp_dir(name, tmp_path, monkeypatch):
    tree = tmp_path / "tree"; tree.mkdir(); _ro(tree)
    rootp = tmp_path / "ro_root"; rootp.mkdir(); _ro(rootp)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(rootp)); monkeypatch.delenv("MODEL_OPT_STACK_KEY", raising=False)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "tmp")); (tmp_path / "tmp").mkdir(); monkeypatch.setattr(tempfile, "tempdir", None)
    try:
        _gen_dirs, _gen_file = _helpers(name, str(tree / "_gen"))
        dirs = _gen_dirs()
        assert dirs[0] == str(tree / "_gen") and dirs[1] == str(rootp / "opt_core_gen" / REL[name]) and dirs[2].startswith(str(tmp_path / "tmp"))
        p = _gen_file("k_rows4_x.py", "X = 4\n")
        assert p == os.path.join(str(tmp_path / "tmp"), "opt_core_gen-uid%d" % os.getuid(), REL[name], "k_rows4_x.py") and open(p).read() == "X = 4\n"
    finally:
        os.chmod(tree, stat.S_IRWXU); os.chmod(rootp, stat.S_IRWXU); monkeypatch.setattr(tempfile, "tempdir", None)


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_read_only_gen_dir_that_already_carries_the_file_serves_it(name, tmp_path, monkeypatch):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True); (gen / "k_rows1_pre.py").write_text("PRE = 1\n"); _ro(gen)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    try:
        _gen_dirs, _gen_file = _helpers(name, str(gen))
        assert _gen_file("k_rows1_pre.py", "PRE = 1\n") == str(gen / "k_rows1_pre.py")
        assert not (tmp_path / "jit").exists()
    finally:
        os.chmod(gen, stat.S_IRWXU)


# ------------------------------------------------------------------------------------ an existing file is compared, never served as found
def _lines(capsys):
    return [l for l in capsys.readouterr().err.splitlines() if l.startswith("[opt_core] GENERATED-SOURCE refused ")]


@pytest.mark.parametrize("name", sorted(FILES))
def test_other_bytes_at_the_file_name_are_replaced_and_named_once(name, tmp_path, monkeypatch, capsys):
    gen = tmp_path / "tree" / "_gen"
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    _gen_dirs, _gen_file = _helpers(name, str(gen))
    p = _gen_file("k_rows3_abc.py", "SRC = 3\n")
    assert _lines(capsys) == []                                              # the plain path says nothing
    for planted in (b"SRC = 3\nimport os\n", b"SRC = ", b"", b"SRC = 3\n" + b"#" * 100000):   # appended, truncated, empty, long
        open(p, "wb").write(planted)
        assert _gen_file("k_rows3_abc.py", "SRC = 3\n") == p and open(p, "rb").read() == b"SRC = 3\n"
    got = _lines(capsys)
    assert len(got) == 1 and f"file {p}:" in got[0] and "replaced by the generated source" in got[0], got   # one line per file per process
    assert not (tmp_path / "jit").exists() and not [f for f in os.listdir(str(gen)) if ".tmp" in f]


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_read_only_gen_dir_with_other_bytes_is_refused_by_name_and_skipped(name, tmp_path, monkeypatch, capsys):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True); (gen / "k_rows1_pre.py").write_text("PRE = 1\nimport os\n"); _ro(gen)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    try:
        _gen_dirs, _gen_file = _helpers(name, str(gen))
        p = _gen_file("k_rows1_pre.py", "PRE = 1\n")
        assert p == str(tmp_path / "jit" / "opt_core_gen" / REL[name] / "k_rows1_pre.py") and open(p).read() == "PRE = 1\n"   # as if the file were absent: the next location
        assert (gen / "k_rows1_pre.py").read_text() == "PRE = 1\nimport os\n"
        got = _lines(capsys)
        assert len(got) == 1 and f"file {gen / 'k_rows1_pre.py'}:" in got[0] and "cannot be replaced" in got[0] and "delete that file" in got[0], got
    finally:
        os.chmod(gen, stat.S_IRWXU)


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_link_at_the_file_name_or_at_the_temporary_name_is_never_followed(name, tmp_path, monkeypatch, capsys):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True, mode=0o755); os.chmod(gen, 0o755)
    victim = tmp_path / "victim.txt"; victim.write_text("SRC = 3\n")           # even a link to identical bytes is not a generated file
    os.symlink(str(victim), str(gen / "k_rows3_abc.py")); os.symlink(str(victim), str(gen / ("k_rows3_abc.py.tmp%d" % os.getpid())))
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    _gen_dirs, _gen_file = _helpers(name, str(gen))
    p = _gen_file("k_rows3_abc.py", "SRC = 33\n")
    assert p == str(gen / "k_rows3_abc.py") and not os.path.islink(p) and open(p).read() == "SRC = 33\n" and victim.read_text() == "SRC = 3\n"
    assert sorted(os.listdir(str(gen))) == ["k_rows3_abc.py"] and len(_lines(capsys)) == 1


# ------------------------------------------------------------------------------------ the directory rule
@pytest.mark.parametrize("mode", [0o775, 0o757, 0o777, 0o1777])
@pytest.mark.parametrize("name", sorted(FILES))
def test_a_directory_group_or_others_can_write_is_refused_by_name(name, mode, tmp_path, monkeypatch, capsys):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True); os.chmod(gen, mode); (gen / "k_rows2_x.py").write_text("X = 2\n")
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    _gen_dirs, _gen_file = _helpers(name, str(gen))
    p = _gen_file("k_rows2_x.py", "X = 2\n")                                  # identical bytes inside do not redeem the directory
    assert p == str(tmp_path / "jit" / "opt_core_gen" / REL[name] / "k_rows2_x.py") and open(p).read() == "X = 2\n"
    assert _gen_file("k_rows2_y.py", "Y = 2\n") == str(tmp_path / "jit" / "opt_core_gen" / REL[name] / "k_rows2_y.py")
    got = _lines(capsys)
    assert len(got) == 1 and f"directory {gen}:" in got[0] and "group or others can write it" in got[0] and "chmod go-w" in got[0] and ("mode %04o" % mode) in got[0], got
    os.chmod(gen, 0o755)                                                     # the named fix: the directory serves again
    assert _gen_file("k_rows2_x.py", "X = 2\n") == str(gen / "k_rows2_x.py")


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_directory_of_another_user_is_refused_by_name_and_uid_0_accepts_any_owner(name, tmp_path, monkeypatch, capsys):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True); os.chmod(gen, 0o755)
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    ns = _namespace(name, str(gen))
    me = os.getuid() or 1000; real_stat = os.stat                            # a run as uid 0 plays an ordinary user here
    foreign = types.SimpleNamespace(st_mode=real_stat(gen).st_mode, st_uid=me + 1)
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: foreign if str(p) == str(gen) else real_stat(p, *a, **k))
    monkeypatch.setattr(os, "getuid", lambda: me)
    why = ns["_gen_why_not"](str(gen))
    assert f"uid {me + 1}" in why and f"uid {me}" in why
    p = ns["_gen_file"]("k_rows2_x.py", "X = 2\n")
    assert p == str(tmp_path / "jit" / "opt_core_gen" / REL[name] / "k_rows2_x.py") and not (gen / "k_rows2_x.py").exists()
    got = _lines(capsys)
    assert len(got) == 1 and f"directory {gen}:" in got[0] and "it belongs to uid" in got[0], got
    monkeypatch.setattr(os, "getuid", lambda: 0)                             # uid 0: a container's bound directories belong to the host user
    assert ns["_gen_why_not"](str(gen)) is None and ns["_gen_file"]("k_rows2_x.py", "X = 2\n") == str(gen / "k_rows2_x.py")


@pytest.mark.parametrize("name", sorted(FILES))
def test_a_root_owned_directory_nobody_else_can_write_is_accepted(name, tmp_path, monkeypatch):
    gen = tmp_path / "tree" / "_gen"; gen.mkdir(parents=True); os.chmod(gen, 0o755)
    ns = _namespace(name, str(gen)); st = os.stat(gen)
    fake = types.SimpleNamespace(st_mode=st.st_mode, st_uid=0)               # an image's own tree: owner root, mode 0755
    real_stat = os.stat
    monkeypatch.setattr(os, "stat", lambda p, *a, **k: fake if str(p) == str(gen) else real_stat(p, *a, **k))
    monkeypatch.setattr(os, "getuid", lambda: 4242)
    assert ns["_gen_why_not"](str(gen)) is None


@pytest.mark.parametrize("name", sorted(FILES))
def test_directories_and_files_made_here_pass_the_rule_under_umask_002(name, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    prev = os.umask(0o002)
    try:
        ns = _namespace(name, str(tmp_path / "tree" / "_gen"))
        p = ns["_gen_file"]("k_rows2_x.py", "X = 2\n")
    finally:
        os.umask(prev)
    assert p == str(tmp_path / "tree" / "_gen" / "k_rows2_x.py") and _lines(capsys) == []
    assert stat.S_IMODE(os.stat(os.path.dirname(p)).st_mode) == 0o755 and stat.S_IMODE(os.stat(p).st_mode) == 0o644


# ------------------------------------------------------------------------------------ what runs is the generated text
@pytest.mark.parametrize("name", sorted(FILES))
def test_the_module_runs_the_generated_text_not_the_bytes_on_disk(name, tmp_path, monkeypatch):
    import importlib.util
    from importlib._bootstrap_external import _code_to_timestamp_pyc
    monkeypatch.setenv("MODEL_OPT_JIT_ROOT", str(tmp_path / "jit"))
    ns = _namespace(name, str(tmp_path / "tree" / "_gen"))
    src = "import os\nRAN = 'generated'\n\n\ndef _fwd():\n    return 7\n"
    p = ns["_gen_file"]("k_rows7_abc.py", src)
    mark = tmp_path / "planted_ran"
    planted = f"open({str(mark)!r}, 'w').write('x')\nRAN = 'planted'\n"
    st = os.stat(p); pyc = importlib.util.cache_from_source(p); os.makedirs(os.path.dirname(pyc), exist_ok=True)
    open(pyc, "wb").write(_code_to_timestamp_pyc(compile(planted, p, "exec"), int(st.st_mtime), st.st_size))   # a .pyc the loader would accept for this file
    open(p, "w").write(planted)                                              # and the file swapped after the comparison
    modname = "gen_exec_probe_" + name
    spec = importlib.util.spec_from_file_location(modname, p); mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, modname, mod)
    ns["_gen_exec"](mod, src, p)
    assert mod.RAN == "generated" and mod._fwd() == 7 and not mark.exists() and mod.__file__ == p and mod._fwd.__code__.co_filename == p
    assert inspect.getsource(mod._fwd) == "def _fwd():\n    return 7\n" and inspect.getsourcelines(mod._fwd)[1] == 5   # the Triton JIT's reader sees the generated text and its line numbers


_AT_ONCE = r"""
import ast, os, sys, time
mod_file, gen_dir, start = sys.argv[1], sys.argv[2], float(sys.argv[3])
tree = ast.parse(open(mod_file, encoding="utf-8").read())
first = [i for i, n in enumerate(tree.body) if isinstance(n, ast.FunctionDef) and n.name == "_gen_dirs"][0]
ns = {"os": os, "sys": sys, "_GEN_DIR": gen_dir}
exec(compile(ast.Module(body=tree.body[first:], type_ignores=[]), mod_file, "exec"), ns)
src = "X = 1\n" * 20000
time.sleep(max(0.0, start - time.time()))
for i in range(40):
    print(ns["_gen_file"]("k_rows%d_same.py" % (i % 4), src))
"""


@pytest.mark.parametrize("name", ["k10", "k13"])
def test_processes_generating_the_same_file_at_once_stay_silent(name, tmp_path):
    import time
    gen = tmp_path / "tree" / "_gen"
    env = dict(os.environ, MODEL_OPT_JIT_ROOT=str(tmp_path / "jit"))
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, "-c", _AT_ONCE, FILES[name], str(gen), repr(start)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(8)]
    outs = [p.communicate(timeout=300) + (p.returncode,) for p in procs]
    assert all(rc == 0 and err == "" for out, err, rc in outs), [(err[-300:], rc) for out, err, rc in outs]   # no refusal line: a file another rank wrote meanwhile is the generated source
    assert {l for out, err, rc in outs for l in out.splitlines()} == {str(gen / ("k_rows%d_same.py" % i)) for i in range(4)}
    assert sorted(os.listdir(str(gen))) == ["k_rows%d_same.py" % i for i in range(4)] and not (tmp_path / "jit").exists()
    assert all((gen / f).read_text() == "X = 1\n" * 20000 for f in os.listdir(str(gen)))


def _need_torch_and_triton():
    """Skip when the renderers' imports are missing -- asked of the finder, not by importing: the modules are rendered in child processes,
    and this process must stay free of torch and triton for the tests that check sys.modules."""
    import importlib.util
    for name in ("torch", "triton"):
        if importlib.util.find_spec(name) is None:
            pytest.skip(f"{name} is not installed")


_RENDER = r"""
import hashlib, importlib, importlib.util, os, sys
root, pkg = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
P = "opt_core.kernels.triattn.triattn_native.pkg.v11.triattn_pkg.triton.triattn"
k10, k11, k12 = (importlib.import_module(P + "." + m) for m in ("k10", "k11", "k12"))
spec = importlib.util.spec_from_file_location("k13_render_probe", os.path.join(pkg, "dispatch", "kernels", "k13.py")); k13 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k13)
out = []
for rows in (1, 2, 3, 4, 8):
    out += [("k10", rows, pipe, ws, k10._render(rows, pipe, ws)) for pipe in (False, True) for ws in (False, True)]
for rows in (2, 3, 4, 8):
    out += [(m.__name__.rsplit(".", 1)[-1], rows, fp16, 0, m._render(rows, fp16)) for m in (k11, k12) for fp16 in (False, True)] + [("k13", rows, 0, 0, k13._render(rows))]
for k, rows, a, b, src in out:
    assert src.isascii() and src.endswith("\n") and "\r" not in src
    print(k, rows, int(a), int(b), len(src), hashlib.sha1(src.encode()).hexdigest()[:10], hashlib.sha256(src.encode()).hexdigest())
"""


def test_generation_is_byte_identical_across_processes():
    _need_torch_and_triton()
    root = os.path.dirname(HERE)
    outs = []
    for seed in ("0", "1", "4242"):                                          # three interpreters, three string-hash seeds
        env = dict(os.environ, PYTHONHASHSEED=seed, PYTHONDONTWRITEBYTECODE="1")
        r = subprocess.run([sys.executable, "-c", _RENDER, root, PKG], env=env, capture_output=True, text=True, timeout=600)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2] and len(outs[0].splitlines()) == 40, [len(o.splitlines()) for o in outs]
    tags = [l.split()[5] for l in outs[0].splitlines()]
    assert len(set(tags)) == len(tags)                                       # one file name per generated text: the name's tag is the text's sha1


# sha256 of the generated text per (module, ROWS, flag, flag): the text names the file (its sha1 tag) and keys the compiled-kernel cache, so a
# tree or cache warmed earlier keeps serving only while these bytes stay as they are
GENERATED_SHA256 = {
    ('k10', 2, 0, 0): "b0e8107227db4d0747ff006c34e93f2e190dd4cb75e567075e4f9aaa16771a05",
    ('k10', 2, 0, 1): "3ad5d3498d8ce9a2d5e5c2b256902aa12d2504c9ff246ad88b3a3b9d3d3335a1",
    ('k10', 2, 1, 0): "50d1359cab7de288a4a3a7d9fe7abbe57c502155740c6a9f1e3f89b3554af66c",
    ('k10', 2, 1, 1): "cdcd2f41f3e6b3569433370f5d119a8583974ab3db89a39d27e279b094c5d7c4",
    ('k10', 3, 0, 0): "dbb581964eb0f86a8de0a38644c365a12a0c850551afab8e9f917d531dc70df4",
    ('k10', 3, 0, 1): "567a82a965ee976adf13cb7c344ae12d355e6bf6a96f8bd0c7b3e4b16a1fa6a0",
    ('k10', 3, 1, 0): "cf631d901dd96ca8c68b0129d60ce1360f2ac39ab222c0b3c784332cec0c71e3",
    ('k10', 3, 1, 1): "f5f040fe5cd9ea87d4c5b3fa85f18848a62f7b83f6eccc5e6c3a32beb808c1ac",
    ('k10', 4, 0, 0): "ca0cb4e6addaacb4857f9ab55d3c03aa77b854230c7df16f01f89db27028420c",
    ('k10', 4, 0, 1): "402db1b0c5da2c324620da19247c816dc55f16aa29ab90d4b0462bfbc23bf9b8",
    ('k10', 4, 1, 0): "77a3661ea129e5a5baa106a9f7ee774d32ed4ed6ad0e057ae07d0abf85f98181",
    ('k10', 4, 1, 1): "91e15de75d9306f2ac9cfc934ceda1b775e0a9450d21fd2f65ce7a108048ccc7",
    ('k10', 8, 0, 0): "63d4293f0ce8ecf568b120e850d0cafeda6dfc0d892d05d65b5f020c33cc90c6",
    ('k10', 8, 0, 1): "eba3f147a0eb5e3354cc6040c3271a556f6da35234a056f560d8fe425ea832a1",
    ('k10', 8, 1, 0): "f2e043c2fa5893766ee810d3f4fb7b046c028f0f35bdc4a49ef71879ab4516ad",
    ('k10', 8, 1, 1): "bc8236285d516edf484665ef1baf9cb8b570f05968784aa0d2eb3baab5aec3bf",
    ('k11', 2, 0, 0): "872f31cceb79e10310b1db7659502226d279448a82a5195ef13d8492ad0eff9b",
    ('k11', 2, 1, 0): "740313ce7848792988f4d789c9838ba396be5130e4fa095190ff9347bbb418d1",
    ('k11', 3, 0, 0): "8bb28ca8ebd9cbe849fc673879ebf984ba4c0e27ce3293809ea59630df2c97a2",
    ('k11', 3, 1, 0): "707a527798db66fe73f7a41faadfb3fabf16ec3b287ac929c76d9814c4c2b35e",
    ('k11', 4, 0, 0): "b0048d8b6d5c59654b36a70cd5196b15982ad8edaf10c6b0e2fc81b03c169b2d",
    ('k11', 4, 1, 0): "0595d2f313592a59beb31405f23f65063a24721e8a2b7d706b488977d889462f",
    ('k11', 8, 0, 0): "56b544389928c51796f1d4f161590a69b4ad8b8b8976f8dde1f922151b10144b",
    ('k11', 8, 1, 0): "1dbed147d785b3c50ea121db8b89056808b3858b00bf4bf7d28e59a52c1ec0a6",
    ('k12', 2, 0, 0): "7a5d54cf5dd2fc9a8a26f9ab88758f17f2781c2c07ef6eedd9e884aa32819fb7",
    ('k12', 2, 1, 0): "b36031308f291c14e73bce2352feb413043f5887061b55acdbfa3045689b847b",
    ('k12', 3, 0, 0): "0ee1c4efab9fc1e54dbfb9108ed04ac0422df3f9958c86d04e28d61e31116ccd",
    ('k12', 3, 1, 0): "9cc664f46e58d07b1d34286a87b37e312a93f92dff0bcf04ff3fc3da99c2fd5a",
    ('k12', 4, 0, 0): "790f3860fc368cb83464bd9238d74145bd07d6232ce66a76ab46c98ac9f37457",
    ('k12', 4, 1, 0): "4bbf7fc934d3b64157193643cf926b76c4b81e28f6d549679f31b48186852c16",
    ('k12', 8, 0, 0): "0e62727d47ad6eedfd86c9cf2b6930e5847f53befcc1adab283b56f2f04291ae",
    ('k12', 8, 1, 0): "ded7bd81b3863361025745e8a61aec689068c421717b9744de51051f202aa1e7",
    ('k13', 2, 0, 0): "07cde5ecaba667f931f8ced89db4a50fee75a364805c78c32be95ae98a35e77b",
    ('k13', 3, 0, 0): "b959eca5879f33daa48e703324a9a7d6a4d9083ffde079e053e4e090dface0de",
    ('k13', 4, 0, 0): "a350b7981333f87484fb2e66b561fa5a322bc54e9610794d9a8944cf9a10ce77",
    ('k13', 8, 0, 0): "c334ad011d84f88b641428c953c9b1fda513d264c20e1f745de9bfd6f00e928a",
}


def test_the_generated_text_is_the_text_warm_trees_and_caches_hold():
    _need_torch_and_triton()
    r = subprocess.run([sys.executable, "-c", _RENDER, os.path.dirname(HERE), PKG], env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"), capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    got = {(t[0], int(t[1]), int(t[2]), int(t[3])): t[6] for t in (l.split() for l in r.stdout.splitlines())}
    assert {k: got.get(k) for k in GENERATED_SHA256} == GENERATED_SHA256
