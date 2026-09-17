"""boltz2_opt.jitcache: a copied Triton cache tree is re-rooted in place (the group files' absolute child paths), idempotently. CPU only."""
import json
import os
import subprocess
import sys

from .. import jitcache as JC


def _tree(tmp_path, old_root="/old/warm/root", key="torch2.12.0-cu130-sm90", h="abc123def", names=("_fused_k.cubin", "_fused_k.json", "_fused_k.ptx")):
    root = tmp_path / "kit" / ".jit"; d = root / key / "triton" / h; d.mkdir(parents=True)
    for n in names: (d / n).write_bytes(b"\x00" + n.encode())
    grp = d / f"{JC.GROUP_PREFIX}_fused_k.json"
    grp.write_text(json.dumps({JC.CHILD_PATHS: {n: f"{old_root}/{key}/triton/{h}/{n}" for n in names}}))
    return root, d, grp


def test_relocate_reroots_the_group_files_child_paths_beside_them_and_is_idempotent(tmp_path):
    root, d, grp = _tree(tmp_path)
    census = JC.relocate(str(root))
    assert census == {"root": str(root), "groups": 1, "rewritten": 1, "already": 0, "unresolved": 0, "invalid": 0}
    paths = json.load(open(grp))[JC.CHILD_PATHS]
    assert paths == {n: str(d / n) for n in ("_fused_k.cubin", "_fused_k.json", "_fused_k.ptx")} and all(os.path.exists(p) for p in paths.values()), \
        "each child path now names <new root>/<stack key>/triton/<hash>/<file>, and the file is there"
    assert all(p.startswith(str(root) + "/torch2.12.0-cu130-sm90/triton/abc123def/") for p in paths.values()), "the <stack key>/triton/<hash>/ layout is kept"
    again = JC.relocate(str(root))
    assert again["rewritten"] == 0 and again["already"] == 1 and json.load(open(grp))[JC.CHILD_PATHS] == paths, "a second relocate is a no-op"
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]


def test_a_tree_compiled_in_place_an_unresolved_child_and_a_foreign_json_are_left_alone_and_counted(tmp_path):
    root, d, grp = _tree(tmp_path)
    inplace = d.parent / "fff000"; inplace.mkdir(); (inplace / "k2.cubin").write_bytes(b"k2")
    (inplace / f"{JC.GROUP_PREFIX}k2.json").write_text(json.dumps({JC.CHILD_PATHS: {"k2.cubin": str(inplace / "k2.cubin")}}))   # compiled here: already right
    gone = d.parent / "9e9e9e"; gone.mkdir()
    (gone / f"{JC.GROUP_PREFIX}k3.json").write_text(json.dumps({JC.CHILD_PATHS: {"k3.cubin": "/old/warm/root/x/triton/9e9e9e/k3.cubin"}}))   # its child was not shipped
    (d.parent / "notes").mkdir(); (d.parent / "notes" / f"{JC.GROUP_PREFIX}readme.json").write_text("not json at all")
    before_gone = (gone / f"{JC.GROUP_PREFIX}k3.json").read_text()
    census = JC.relocate(str(root))
    assert (census["groups"], census["rewritten"], census["already"], census["unresolved"], census["invalid"]) == (4, 1, 1, 1, 1), census
    assert (gone / f"{JC.GROUP_PREFIX}k3.json").read_text() == before_gone and (d.parent / "notes" / f"{JC.GROUP_PREFIX}readme.json").read_text() == "not json at all"
    assert JC.relocate(str(tmp_path / "no" / "such"))["groups"] == 0, "a missing root is an empty census, not an error (the image's layer may be empty)"


def test_the_command_line_prints_one_census_line(tmp_path, capsys):
    root, d, grp = _tree(tmp_path)
    assert JC.main(["relocate", "--root", str(root)]) == 0
    out = capsys.readouterr().out.strip()
    assert out == f"{JC.PREFIX} relocate root={root} groups=1 rewritten=1 already=0 unresolved=0 invalid=0", out
    assert JC.main([]) == 2 and JC.main(["relocate"]) == 2
    r = subprocess.run([sys.executable, "-s", "-m", "boltz2_opt.jitcache", "relocate", f"--root={root}"], capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.pathsep.join([os.path.dirname(os.path.dirname(JC.__file__))] + [p for p in sys.path if p])})
    assert r.returncode == 0 and "already=1" in r.stdout, r.stderr


def test_the_image_re_roots_the_shipped_cache_right_after_unpacking_it():
    """The image ships pre-filled compile caches under /opt/jit_cache (an optional build-context tar, unpacked by one COPY + RUN pair), and
    run.sh seeds them on first use: in place when writable, else copied once to a writable root with every Triton group file's absolute
    child paths re-rooted there (the shell form of :func:`relocate` for that prefix) before MODEL_OPT_JIT_ROOT is exported."""
    kit = os.path.dirname(os.path.dirname(os.path.dirname(JC.__file__)))
    text = open(os.path.join(kit, "environment", "Dockerfile")).read()
    i = text.index("COPY boltz2/environment/Dockerfile _jitcache/boltz2-*-jit.ta[r] /tmp/jit/"); j = text.index("rm -rf /tmp/jit", i)
    layer = text[i:j]
    assert "mkdir -p /opt/jit_cache" in layer and 'tar -xf "$t" -C /opt/jit_cache' in layer and "chmod -R a+rX,u+w /opt/jit_cache" in layer, \
        "the optional tar is unpacked under /opt/jit_cache and made readable in the same RUN"
    run_sh = open(os.path.join(kit, "run.sh")).read()
    k = run_sh.index('source "$HERE/configs/$CFG.env"'); b = run_sh.index('I="${MODEL_OPT_JIT_IMAGE:-/opt/jit_cache}"', k)
    seed = run_sh[b:run_sh.index('export MODEL_OPT_JIT_ROOT="$J"', b)]
    assert 'cp -a "$I/." "$J/"' in seed and f"""find "$J" -name '{JC.GROUP_PREFIX}*.json' -exec sed -i "s#$I/#$J/#g" {{}} +""" in seed and 'touch "$J/.seeded"' in seed, \
        "after the config is sourced: a read-only image cache is copied once, its group files re-rooted, then the root is exported"
    assert run_sh.index("export TRITON_CACHE_DIR=$MODEL_OPT_JIT_ROOT/$MODEL_OPT_STACK_KEY/triton", b) > b, "the Triton cache is keyed under the seeded root"
