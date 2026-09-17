"""The pre-filled cache tree an environment may carry (<prefix of AF3_TORCH_PY>/share/af3_torch_cache/<key>/{triton,inductor,jax}) and
the seed a pred / warm takes from it into a cold cache root: the key's form, the copy, the states of the ONE CACHE line
(seeded | kept | none:<reason>), and that nothing else of the root is touched. CPU only: the compute capability is passed in or stubbed."""
import json
import os
import re
import tempfile

import pytest

from af3_torch_opt import cli, stack

CACHE_RE = re.compile(r"^\[af3-torch-opt\] CACHE cache_seed=(seeded:\S+ files=\d+ mb=[\d.]+ seconds=[\d.]+ groups_rewritten=\d+ children_ok=\d+/\d+( groups_dropped=\d+:\S+)? from=\S+|kept files=\d+|none:\S+) root=\S+$")


def _tree(root, files):
    for rel, data in files.items():
        p = os.path.join(root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(data)


def _venv(tmp_path, monkeypatch):
    """A venv-shaped interpreter path (<prefix>/bin/python) named by AF3_TORCH_PY; returns (prefix, seed home)."""
    prefix = tmp_path / "torch_venv"; (prefix / "bin").mkdir(parents=True); (prefix / "bin" / "python").write_text("#!/bin/sh\n")
    monkeypatch.setenv("AF3_TORCH_PY", str(prefix / "bin" / "python"))
    return str(prefix), os.path.join(str(prefix), "share", "af3_torch_cache")


def test_key_is_the_house_form_from_the_pins():
    P = stack.pins(); torch_v = str(P["check_packages"]["torch_python"]["torch"]).split("+")[0]; cuda = str(P["cuda"]).replace(".", "")
    assert stack.cache_seed_key("90") == stack.cache_seed_key("9.0") == f"torch{torch_v}-cu{cuda}-sm90"
    assert re.match(r"^torch\d+\.\d+\.\d+-cu\d+-sm80$", stack.cache_seed_key("8.0"))


def test_seed_home_is_under_the_interpreters_prefix_not_its_realpath(tmp_path, monkeypatch):
    prefix, home_ = _venv(tmp_path, monkeypatch)
    real = tmp_path / "base" / "bin"; real.mkdir(parents=True); (real / "python3.12").write_text("#!/bin/sh\n")
    os.remove(os.path.join(prefix, "bin", "python")); os.symlink(str(real / "python3.12"), os.path.join(prefix, "bin", "python"))   # a venv's python: a symlink into the base installation
    assert stack.cache_seed_home() == home_ == os.path.join(prefix, "share", "af3_torch_cache")
    monkeypatch.delenv("AF3_TORCH_PY")
    assert stack.cache_seed_home() is None


def test_a_cold_root_is_seeded_leaf_for_leaf_and_the_line_says_so(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    key = stack.cache_seed_key("90")
    _tree(os.path.join(home_, key), {"triton/ab12/kernel.cubin": b"\x00" * 1000, "triton/ab12/__grp__kernel.json": b"{}", "inductor/fx/cd34/entry": b"x" * 500})
    _tree(os.path.join(home_, stack.cache_seed_key("80")), {"triton/zz99/other.cubin": b"\x01"})   # another card's tree: never copied
    root = str(tmp_path / "cache")
    _tree(root, {"weights_digests.json": b"{}"})                                                  # a root holding only the digest memo is still cold
    rec = stack.seed_cache_root(root, cc="90")
    assert rec["state"] == "seeded" and rec["key"] == key and rec["files"] == 3 and rec["mb"] == round(1502 / 1e6, 1) and rec["src"] == os.path.join(home_, key)
    assert sorted(os.path.relpath(os.path.join(dp, f), root) for dp, _, fs in os.walk(root) for f in fs) == \
        ["inductor/fx/cd34/entry", "triton/ab12/__grp__kernel.json", "triton/ab12/kernel.cubin", "weights_digests.json"]
    with open(os.path.join(root, "triton/ab12/kernel.cubin"), "rb") as f: assert f.read() == b"\x00" * 1000
    ln = stack.cache_seed_line(rec)
    assert CACHE_RE.match(ln) and ln.startswith(f"[af3-torch-opt] CACHE cache_seed=seeded:{key} files=3 mb=0.0 seconds=") and ln.endswith(f" from={os.path.join(home_, key)} root={root}")
    again = stack.seed_cache_root(root, cc="90")                                                 # the root now holds kernels: kept, nothing copied twice
    assert again["state"] == "kept" and again["files"] == 3 and stack.cache_seed_line(again) == f"[af3-torch-opt] CACHE cache_seed=kept files=3 root={root}"


def test_a_root_holding_kernels_is_kept_as_it_is(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    _tree(os.path.join(home_, stack.cache_seed_key("90")), {"triton/ab12/kernel.cubin": b"seed"})
    root = str(tmp_path / "cache"); _tree(root, {"inductor/own/entry": b"mine"})
    rec = stack.seed_cache_root(root, cc="90")
    assert rec["state"] == "kept" and rec["files"] == 1 and not os.path.exists(os.path.join(root, "triton"))


def test_none_by_name_no_tree_for_this_card_no_seed_dir_no_capability_unwritable_root(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    root = str(tmp_path / "cache")
    assert stack.seed_cache_root(root, cc="90")["reason"] == "no_seed_dir"                      # the environment carries no tree at all
    _tree(os.path.join(home_, stack.cache_seed_key("80")), {"triton/x/y": b"1"})
    rec = stack.seed_cache_root(root, cc="90")
    assert (rec["state"], rec["reason"]) == ("none", f"key_absent:{stack.cache_seed_key('90')}") and not os.path.exists(root)
    assert stack.cache_seed_line(rec) == f"[af3-torch-opt] CACHE cache_seed=none:key_absent:{stack.cache_seed_key('90')} root={root}"
    monkeypatch.setattr(stack, "device_cc", lambda environ=None: None)
    assert stack.seed_cache_root(root)["reason"] == "cc_unknown"                               # no nvidia-smi and no AF3_TORCH_GPU class word
    blocker = tmp_path / "file"; blocker.write_text("x")                                         # a root that cannot be made (its parent is a file): named, never raised
    rec = stack.seed_cache_root(str(blocker / "cache"), cc="80")
    assert rec["state"] == "none" and rec["reason"].startswith("unwritable:")
    monkeypatch.delenv("AF3_TORCH_PY")
    assert stack.seed_cache_root(root, cc="90")["reason"] == "no_seed_dir"


def test_device_cc_falls_back_to_the_gpu_class_word_without_nvidia_smi(monkeypatch):
    import shutil
    monkeypatch.setattr(shutil, "which", lambda name: None)
    monkeypatch.setenv("AF3_TORCH_GPU", "A100"); assert stack.device_cc() == "80"
    monkeypatch.setenv("AF3_TORCH_GPU", "H100"); assert stack.device_cc() == "90"
    monkeypatch.setenv("AF3_TORCH_GPU", "a100_80gb"); assert stack.device_cc() == "80"
    monkeypatch.delenv("AF3_TORCH_GPU"); assert stack.device_cc() is None


def test_pred_and_warm_print_one_cache_line_before_any_model_process(box, capsys, monkeypatch):
    """On the stub box (conftest): warm seeds the cold root from the environment's tree BEFORE its first census, so the WARM line's
    cache_before already counts the seeded files; each pred of warm prints its own CACHE line (kept) ahead of its COMMAND lines."""
    monkeypatch.setattr(stack, "device_cc", lambda environ=None: "90")
    home_ = stack.cache_seed_home(); assert home_ == str(box["tmp"] / "share" / "af3_torch_cache")   # the stub interpreter's prefix is the box's tmp dir
    key = stack.cache_seed_key("90")
    _tree(os.path.join(home_, key), {"triton/ab12/kernel.cubin": b"\x00" * 10, "inductor/fx/cd34/entry": b"x"})
    (box["tmp"] / "t").mkdir(); monkeypatch.setattr(tempfile, "tempdir", str(box["tmp"] / "t"))
    assert cli.main(["warm", "--mode", "off"]) == 0
    err = capsys.readouterr().err
    cache = [ln for ln in err.splitlines() if ln.startswith("[af3-torch-opt] CACHE ")]
    assert len(cache) == 2 and cache[0].startswith(f"[af3-torch-opt] CACHE cache_seed=seeded:{key} files=2 ") and cache[1] == f"[af3-torch-opt] CACHE cache_seed=kept files=2 root={stack.cache_root()}", err
    assert err.index(cache[0]) < err.index("[af3-torch-opt] ACTIVE ") < err.index(cache[1]) < err.index("[af3-torch-opt] COMMAND ") < err.index("[af3-torch-opt] WARM ")
    warm = [ln for ln in err.splitlines() if ln.startswith("[af3-torch-opt] WARM ")][0]
    assert re.search(r" cache_before=(\d+)files/", warm) and int(re.search(r" cache_before=(\d+)files/", warm).group(1)) >= 2   # the seeded files are counted before the pred ran
    assert os.path.isfile(os.path.join(stack.cache_root(), "triton", "ab12", "kernel.cubin"))


# ---- portability: Triton's group manifests name the copies after the seed ----------------------------------------------------------------
BAKE_ROOT = "/tmp/bake/root_warm"          # the root the tree was filled under on the machine that compiled it (any prefix: the seed maps by tail, not by this string)


def _group(members):
    """A Triton FileCacheManager group manifest as put_group writes it: {"child_paths": {member: <absolute path under the compiling root>}}."""
    return json.dumps({"child_paths": members}).encode()


def _baked_tree(home_, key):
    """A source tree shaped like a real one: triton/<key dir>/{members, __grp__ manifest} (two kernels), one Inductor-side Triton kernel with its
    manifest, one manifest whose member sits one directory below it (mapped by tail), and loose files with no manifest (autotune results)."""
    t = os.path.join(home_, key)
    files = {}
    for kdir, name in (("triton/AB12", "_glu_kernel"), ("triton/CD34", "triton_poi_fused_0")):
        for ext in ("cubin", "json", "ttir", "ptx"):
            files[f"{kdir}/{name}.{ext}"] = f"{name}.{ext}".encode()
        files[f"{kdir}/__grp__{name}.json"] = _group({f"{name}.{ext}": f"{BAKE_ROOT}/{kdir}/{name}.{ext}" for ext in ("cubin", "json", "ttir", "ptx")})
    files["inductor/triton/0/EF56/triton_red_1.cubin"] = b"r"; files["inductor/triton/0/EF56/triton_red_1.json"] = b"{}"
    files["inductor/triton/0/EF56/__grp__triton_red_1.json"] = _group({"triton_red_1.cubin": f"{BAKE_ROOT}/inductor/triton/0/EF56/triton_red_1.cubin", "triton_red_1.json": f"{BAKE_ROOT}/inductor/triton/0/EF56/triton_red_1.json"})
    files["triton/GH78/sub/deep.cubin"] = b"d"                                                    # member NOT beside its manifest: resolved by the recorded path's tail after `triton/`
    files["triton/GH78/__grp__deep.json"] = _group({"deep.cubin": f"/some/other/prefix/triton/GH78/sub/deep.cubin"})
    files["triton/IJ90/_glu_kernel.autotune.json"] = b"{}"; files["inductor/fxgraph/aa/key"] = b"pickled"   # no manifest: copied as they are
    _tree(t, files)
    return t


def _replica_get_group(manifest_path):
    """triton 3.7.1 runtime/cache.py FileCacheManager.get_group, the part that matters here: a member is returned only if its recorded path exists."""
    with open(manifest_path) as f:
        cps = json.load(f).get("child_paths")
    return {c: q for c, q in cps.items() if os.path.exists(q)}


def test_seeded_group_manifests_name_the_copies_and_triton_reads_full_groups(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    key = stack.cache_seed_key("90"); _baked_tree(home_, key)
    root = str(tmp_path / "elsewhere" / "cache")                                                  # a DIFFERENT root than the one the tree was filled under
    rec = stack.seed_cache_root(root, cc="90")
    assert rec["state"] == "seeded" and rec["files"] == 17
    assert (rec["groups_rewritten"], rec["children_ok"], rec["children"], rec["groups_dropped"], rec["first_dropped"]) == (4, 11, 11, 0, None)
    manifests = [os.path.join(dp, f) for dp, _, fs in os.walk(root) for f in fs if f.startswith("__grp__")]
    assert len(manifests) == 4
    for m in manifests:                                                                           # (a) every member names a file under THIS root, and it exists
        cps = json.load(open(m))["child_paths"]
        assert cps and all(q.startswith(root + os.sep) and os.path.isfile(q) for q in cps.values()), (m, cps)
        assert _replica_get_group(m) == cps                                                       # (b') triton 3.7.1's existence filter keeps every member
        assert not os.path.exists(m + ".seedtmp")
    deep = json.load(open(os.path.join(root, "triton/GH78/__grp__deep.json")))["child_paths"]
    assert deep == {"deep.cubin": os.path.join(root, "triton/GH78/sub/deep.cubin")}              # mapped by tail, whatever the recorded prefix was
    ln = stack.cache_seed_line(rec)
    assert CACHE_RE.match(ln) and " groups_rewritten=4 children_ok=11/11 from=" in ln and "groups_dropped" not in ln, ln
    try:                                                                                          # (b) Triton's own reader, when this interpreter has it (the kit's package venv does not: (b') stands in)
        from triton.runtime.cache import FileCacheManager
    except Exception:
        return
    monkeypatch.setenv("TRITON_CACHE_DIR", os.path.join(root, "triton"))
    grp = FileCacheManager("AB12").get_group("_glu_kernel.json")
    assert grp == {f"_glu_kernel.{e}": os.path.join(root, "triton/AB12", f"_glu_kernel.{e}") for e in ("cubin", "json", "ttir", "ptx")}, grp
    monkeypatch.setenv("TRITON_CACHE_DIR", os.path.join(root, "inductor/triton/0"))
    assert FileCacheManager("EF56").get_group("triton_red_1.json") == {f"triton_red_1.{e}": os.path.join(root, "inductor/triton/0/EF56", f"triton_red_1.{e}") for e in ("cubin", "json")}


def test_a_manifest_with_a_lost_member_is_named_once_and_never_fatal(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    key = stack.cache_seed_key("80"); t = _baked_tree(home_, key)
    os.remove(os.path.join(t, "triton/CD34/triton_poi_fused_0.ptx"))                              # one member of one group missing from the tree itself
    _tree(t, {"triton/ZZ00/__grp__corrupt.json": b"{not json"})                                   # and one unreadable manifest
    root = str(tmp_path / "r80")
    rec = stack.seed_cache_root(root, cc="80")
    assert rec["state"] == "seeded"
    assert (rec["groups_rewritten"], rec["children_ok"], rec["children"], rec["groups_dropped"]) == (3, 10, 11, 2)
    assert rec["first_dropped"] in ("triton/CD34/__grp__triton_poi_fused_0.json", "triton/ZZ00/__grp__corrupt.json")
    kept = json.load(open(os.path.join(root, "triton/CD34/__grp__triton_poi_fused_0.json")))["child_paths"]
    assert kept["triton_poi_fused_0.ptx"] == f"{BAKE_ROOT}/triton/CD34/triton_poi_fused_0.ptx"   # the lost member keeps its recorded path (Triton: a miss → that kernel recompiles)
    assert kept["triton_poi_fused_0.cubin"] == os.path.join(root, "triton/CD34/triton_poi_fused_0.cubin")   # its present siblings still name the copies
    ln = stack.cache_seed_line(rec)
    assert CACHE_RE.match(ln) and f" groups_rewritten=3 children_ok=10/11 groups_dropped=2:{rec['first_dropped']} from=" in ln, ln


def test_reroot_by_tail_takes_the_last_cache_segment_whatever_the_prefix():
    assert stack._reroot_by_tail("/tmp/bake/root_warm/triton/AB/x.cubin", "/r") == "/r/triton/AB/x.cubin"
    assert stack._reroot_by_tail("/work/jit/k/af3_torch/inductor/triton/0/EF/y.json", "/r") == "/r/triton/0/EF/y.json"   # the LAST segment splits (a manifest under inductor/triton/… is still found beside itself first)
    assert stack._reroot_by_tail("/no/segment/here/x.cubin", "/r") is None


# ---- the byte-gate stamp directory travels with the caches ---------------------------------------------------------------------------------
def test_model_processes_get_the_verdict_dir_under_the_cache_root_unless_the_caller_set_it(tmp_path, monkeypatch):
    monkeypatch.setenv("AF3_TORCH_CACHE_ROOT", str(tmp_path / "root"))
    assert stack.cache_dirs()["OPT_CORE_VERDICT_DIR"] == str(tmp_path / "root" / "verdict")
    monkeypatch.delenv("OPT_CORE_VERDICT_DIR", raising=False)
    env = stack.model_process_env(jax=True)                                                       # the JAX venv's env needs no kernel exports: the cache leaves alone
    assert env["OPT_CORE_VERDICT_DIR"] == str(tmp_path / "root" / "verdict") and env["TRITON_CACHE_DIR"] == str(tmp_path / "root" / "triton")
    for theirs in ("0", str(tmp_path / "mine")):                                                  # the caller's value wins: "0" (no stamp) or a directory of theirs
        monkeypatch.setenv("OPT_CORE_VERDICT_DIR", theirs)
        assert stack.model_process_env(jax=True)["OPT_CORE_VERDICT_DIR"] == theirs
        assert stack.model_process_env({"OPT_CORE_VERDICT_DIR": theirs, "PATH": "/bin"}, jax=True)["OPT_CORE_VERDICT_DIR"] == theirs


def test_the_seed_carries_the_verdict_leaf(tmp_path, monkeypatch):
    _prefix, home_ = _venv(tmp_path, monkeypatch)
    key = stack.cache_seed_key("90")
    _tree(os.path.join(home_, key), {"triton/ab12/kernel.cubin": b"\x00", "verdict/trimul_native/gate-0123456789abcdef01234567.json": b'{"schema": 1}'})
    root = str(tmp_path / "root")
    rec = stack.seed_cache_root(root, cc="90")
    assert rec["state"] == "seeded" and rec["files"] == 2
    assert os.path.isfile(os.path.join(root, "verdict", "trimul_native", "gate-0123456789abcdef01234567.json"))
    assert stack.seed_cache_root(root, cc="90")["state"] == "kept"                                # a verdict alone never makes a root `kept`; a kernel file does
