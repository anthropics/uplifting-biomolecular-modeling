"""The JIT cache switch (configs/h100.env EF2INV_JIT_CACHE; modes.jit_cache_*; launch.arm_env / det_cache): shared = the keyed dirs
under the operator's optional shared root MODEL_OPT_JIT_ROOT (a speed lever and a cross-box state channel; unset = nothing exported), per-box = fresh box-local
dirs (the det-arm form every --det >= 1 arm gets by itself), frozen:<sha> named but refused. No torch, no GPU."""
import os
import subprocess
import shutil

import pytest

from .. import launch as LA, modes as MD
from ._paths import ROOT, PINS

PREFIXES = PINS["stock_environment"]["must_be_absent_prefixes"]
ROOT_SHARED = "/mnt/team-jitcache"                                                     # an example shared root (MODEL_OPT_JIT_ROOT): any directory every box mounts
SHARED = {"TRITON_CACHE_DIR": f"{ROOT_SHARED}/torch2.11.0-cu128-sm90/triton", "TORCHINDUCTOR_CACHE_DIR": f"{ROOT_SHARED}/torch2.11.0-cu128-sm90/inductor"}
WITH_ROOT = dict(SHARED, MODEL_OPT_JIT_ROOT=ROOT_SHARED)


def test_forms_and_dirs(tmp_path):
    assert MD.JIT_CACHE_FORMS == ("shared", "per-box") and MD.jit_cache_form(None) == "shared" and MD.jit_cache_form(" per-box ") == "per-box"
    for bad in ("frozen:" + "0" * 64, "local", ""):
        if bad == "":
            assert MD.jit_cache_form(bad) == "shared"; continue
        with pytest.raises(ValueError, match="not shipped" if bad.startswith("frozen") else "shipped forms"):
            MD.jit_cache_form(bad)
    assert MD.jit_cache_dirs("shared", key="torch2.11.0-cu128-sm90", shared_root=ROOT_SHARED) == SHARED == MD.jit_cache_dirs("shared", key="torch2.11.0-cu128-sm90", shared_root=ROOT_SHARED + "/")
    with pytest.raises(ValueError, match="key required"):
        MD.jit_cache_dirs("shared", shared_root=ROOT_SHARED)
    with pytest.raises(ValueError, match=MD.JIT_SHARED_ROOT_VAR + " is not set"):       # no default shared root: the variable named, never guessed
        MD.jit_cache_dirs("shared", key="torch2.11.0-cu128-sm90")
    d = MD.jit_cache_dirs("per-box", local_root=str(tmp_path))
    assert d == {"TRITON_CACHE_DIR": f"{tmp_path}/triton", "TORCHINDUCTOR_CACHE_DIR": f"{tmp_path}/inductor"}
    fresh = MD.jit_cache_dirs("per-box")
    assert not fresh["TRITON_CACHE_DIR"].startswith(ROOT_SHARED + "/") and os.path.isdir(os.path.dirname(fresh["TRITON_CACHE_DIR"]))
    shutil.rmtree(os.path.dirname(fresh["TRITON_CACHE_DIR"]))
    assert MD.jit_cache_state(WITH_ROOT)["form"] == "shared" and MD.jit_cache_state(WITH_ROOT)["shared_root"] == ROOT_SHARED
    assert MD.jit_cache_state(SHARED)["form"] == "per-box"                              # dirs with no shared root named: nothing marks them shared
    assert MD.jit_cache_state(dict(d, MODEL_OPT_JIT_ROOT=ROOT_SHARED))["form"] == "per-box" and MD.jit_cache_state({})["form"] == "unset"


def test_det_arms_get_a_per_box_cache(tmp_path):
    base = dict(WITH_ROOT, HF_HOME="/data/hf", PATH="/usr/bin", EF2INV_JIT_LOCAL_ROOT=str(tmp_path))
    env0 = LA.arm_env(MD.MODES["exact"], PREFIXES, base=base, det=0)
    assert {k: env0[k] for k in SHARED} == SHARED                                    # det 0: the deployment's dirs pass through
    env1 = LA.arm_env(MD.MODES["exact"], PREFIXES, base=base, det=1)
    assert env1["TRITON_CACHE_DIR"] == f"{tmp_path}/triton" and env1["TORCHINDUCTOR_CACHE_DIR"] == f"{tmp_path}/inductor"   # det 1: per-box by itself
    assert not [k for k in env1 if k.startswith("EF2INV_")] and not [k for k in env0 if k.startswith("EF2INV_")]   # the switch, the shared root and the local root never reach the arm
    assert LA.det_cache(base, 1) == {"form": "per-box", "explicit": False} and LA.det_cache(base, 0)["form"] == "shared"
    shared = dict(base, EF2INV_JIT_CACHE="shared")                                    # the operator's explicit choice is honoured and reported
    assert LA.det_cache(shared, 1) == {"form": "shared", "explicit": True}
    assert {k: LA.arm_env(MD.MODES["exact"], PREFIXES, base=shared, det=1)[k] for k in SHARED} == SHARED
    with pytest.raises(ValueError, match="not shipped"):
        LA.det_cache(dict(base, EF2INV_JIT_CACHE="frozen:" + "ab" * 32), 1)


@pytest.mark.skipif(not shutil.which("bash"), reason="bash")
def test_a100_env(tmp_path):
    """configs/a100.env = h100.env's parameters with the card's values: the shared JIT dir is keyed …-sm80 (EF2INV_JIT_KEY, derived from the
    running stack on the card; explicit here), the card NOTE compares the A100 80GB name + MiB,
    the fast-environment requirement is inherited unchanged; with no shared root nothing is exported and no key is derived; a shared root with
    a key that cannot be computed (no card on this host) is refused by name, never defaulted to another card's key."""
    def source(env, echo='"$TRITON_CACHE_DIR|$TORCHINDUCTOR_CACHE_DIR|$EF2INV_GPU|$EF2INV_GPU_MIB|$EF2INV_REQUIRE_FAST_ENV"'):
        r = subprocess.run(["bash", "-c", f'set -a; source configs/a100.env && echo {echo}'], cwd=ROOT, env=dict(dict(PATH=os.environ["PATH"], HOME=str(tmp_path)), **env), capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    rc, out, err = source({"MODEL_OPT_JIT_ROOT": ROOT_SHARED, "EF2INV_JIT_KEY": "torch2.11.0-cu128-sm80"})
    assert (rc, err) == (0, "") and out == f"{ROOT_SHARED}/torch2.11.0-cu128-sm80/triton|{ROOT_SHARED}/torch2.11.0-cu128-sm80/inductor|A100|81920|40960|1"
    rc, out, err = source({})                                                             # no shared root: nothing exported, the key is not needed and not derived
    assert (rc, err) == (0, "") and out == "||A100|81920|40960|1"
    rc, out, err = source({"MODEL_OPT_TARGET_GPU": "NVIDIA A100 80GB PCIe"})                # a launcher's full card name is the expected name
    assert rc == 0 and out.split("|")[2] == "NVIDIA A100 80GB PCIe"
    rc, out, err = source({"MODEL_OPT_JIT_ROOT": ROOT_SHARED, "PATH": "/usr/bin:/bin"}, echo='"$TRITON_CACHE_DIR"')   # a shared root, no card / no package on this PATH's python: refused by name
    assert rc == 3 and "configs/a100.env: the shared JIT-cache key of the running stack could not be computed" in err and "EF2INV_JIT_KEY=torch2.11.0-cu128-sm80" in err
    rc, out, err = source({"MODEL_OPT_JIT_ROOT": ROOT_SHARED, "EF2INV_JIT_CACHE": "per-box", "EF2INV_JIT_LOCAL_ROOT": str(tmp_path / "jit")}, echo='"$TRITON_CACHE_DIR"')
    assert rc == 0 and out == f"{tmp_path}/jit/triton"                                     # the per-box form needs no key


def test_h100_env_switch(tmp_path):
    def source(extra_env):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **extra_env}
        r = subprocess.run(["bash", "-c", 'set -a; source configs/h100.env && echo "$TRITON_CACHE_DIR|$TORCHINDUCTOR_CACHE_DIR"'], cwd=ROOT, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr
    assert source({"MODEL_OPT_JIT_ROOT": ROOT_SHARED}) == (0, "|".join(SHARED.values()), "")
    assert source({}) == (0, "|", "")                                                    # no shared root, no switch: no cache dir exported (the tools' own defaults; cache=unset)
    rc, out, err = source({"EF2INV_JIT_CACHE": "shared"})                                # the shared form asked for by name with no shared root: refused by the variable's name
    assert rc == 2 and out == "" and "MODEL_OPT_JIT_ROOT is not set — see README.md §Variables" in err and "EF2INV_JIT_CACHE=per-box" in err
    rc, out, _ = source({"EF2INV_JIT_CACHE": "per-box", "EF2INV_JIT_LOCAL_ROOT": str(tmp_path), "TRITON_CACHE_DIR": SHARED["TRITON_CACHE_DIR"]})
    assert rc == 0 and out == f"{tmp_path}/triton|{tmp_path}/inductor"                # per-box is forced: a pre-set shared dir is not inherited
    rc, out, err = source({"EF2INV_JIT_CACHE": "frozen:" + "0" * 64})
    assert rc != 0 and "not shipped" in err


def test_per_box_default_root_is_a_private_fresh_directory(tmp_path):
    """EF2INV_JIT_CACHE=per-box without EF2INV_JIT_LOCAL_ROOT: the root is made by `mktemp -d` under ${TMPDIR:-/tmp} — mode 0700, a fresh name per
    sourcing (not the shell's pid), never a pre-existing path taken over with `mkdir -p`; EF2INV_JIT_LOCAL_ROOT keeps naming the root outright."""
    import stat
    def source(extra_env):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "TMPDIR": str(tmp_path), **extra_env}
        r = subprocess.run(["bash", "-c", 'set -a; source configs/h100.env && echo "$TRITON_CACHE_DIR|$TORCHINDUCTOR_CACHE_DIR|$$"'], cwd=ROOT, env=env, capture_output=True, text=True)
        return r.returncode, r.stdout.strip(), r.stderr
    rc, out, err = source({"EF2INV_JIT_CACHE": "per-box"})
    assert rc == 0 and err == "", (rc, out, err)
    triton, inductor, pid = out.split("|")
    root = os.path.dirname(triton)
    assert root == os.path.dirname(inductor) and triton.endswith("/triton") and inductor.endswith("/inductor")
    assert os.path.dirname(root) == str(tmp_path) and os.path.basename(root).startswith("ef2inv-jitcache-"), root      # under ${TMPDIR:-/tmp}
    assert not os.path.basename(root).endswith("-" + pid), root                                                          # not the pid-named path
    assert os.path.isdir(root) and stat.S_IMODE(os.stat(root).st_mode) == 0o700, oct(os.stat(root).st_mode)
    assert os.path.dirname(source({"EF2INV_JIT_CACHE": "per-box"})[1].split("|")[0]) != root                              # a fresh directory each time
    rc, out, _ = source({"EF2INV_JIT_CACHE": "per-box", "EF2INV_JIT_LOCAL_ROOT": str(tmp_path / "named")})
    assert rc == 0 and out.split("|")[:2] == [f"{tmp_path}/named/triton", f"{tmp_path}/named/inductor"] and os.path.isdir(tmp_path / "named")
    text = open(os.path.join(ROOT, "configs", "h100.env")).read()
    assert "ef2inv-jitcache-$$" not in text and 'mktemp -d "${TMPDIR:-/tmp}/ef2inv-jitcache-XXXXXXXX"' in text


def test_check_report_forms():
    """cli.jit_cache_report — what `check` prints as cache=<form>, words as notes (the run proceeds on the dirs as given): a shared dir not
    keyed by the running stack, a switch that disagrees with the dirs; per-box dirs never subject to the key rule; frozen = the one refusal (usage)."""
    from .. import cli
    key = "torch2.11.0-cu128-sm90"
    pb = []; rep = cli.jit_cache_report(dict(WITH_ROOT), key, pb)
    assert rep["form"] == "shared" and rep["shared_root"] == ROOT_SHARED and pb == []
    pb = []; rep = cli.jit_cache_report({"MODEL_OPT_JIT_ROOT": ROOT_SHARED, "TRITON_CACHE_DIR": f"{ROOT_SHARED}/torch2.13.0-cu130-sm90/triton"}, key, pb)
    assert pb == [] and len(rep["notes"]) == 1 and "not keyed by the running stack" in rep["notes"][0]   # worded, never refused
    pb = []; rep = cli.jit_cache_report({"EF2INV_JIT_CACHE": "per-box", "TRITON_CACHE_DIR": "/tmp/ef2inv-jitcache-858/triton", "TORCHINDUCTOR_CACHE_DIR": "/tmp/ef2inv-jitcache-858/inductor"}, key, pb)
    assert rep["form"] == "per-box" and pb == []                                        # the e2e box's per-box check: no key rule on private dirs
    pb = []; rep = cli.jit_cache_report(dict(WITH_ROOT, EF2INV_JIT_CACHE="per-box"), key, pb)
    assert pb == [] and len(rep["notes"]) == 1 and "imply shared" in rep["notes"][0]
    pb = []; rep = cli.jit_cache_report(dict(SHARED, EF2INV_JIT_CACHE="shared"), key, pb)      # the switch says shared but no shared root marks the dirs: worded
    assert pb == [] and len(rep["notes"]) == 1 and "imply per-box" in rep["notes"][0]
    pb = []; cli.jit_cache_report(dict(WITH_ROOT, EF2INV_JIT_CACHE="frozen:" + "0" * 64), key, pb)
    assert len(pb) == 1 and "not shipped" in pb[0]
    pb = []; assert cli.jit_cache_report({}, None, pb)["form"] == "unset" and pb == []


# ---------------------------------------------------------------- the launch: the base read before torch, the pair completed, the ARM's dirs judged (launch_base / arm_jit_view / cli.launch_context)
KEY = "torch2.11.0-cu128-sm90"


def test_launch_base_completes_the_jit_pair_by_the_shared_root_rule():
    T, I = SHARED["TRITON_CACHE_DIR"], SHARED["TORCHINDUCTOR_CACHE_DIR"]
    base, notes = LA.launch_base({"TRITON_CACHE_DIR": T, "MODEL_OPT_JIT_ROOT": ROOT_SHARED, "HOME": "/root"})      # a box naming the root and only the Triton cache
    assert base["TORCHINDUCTOR_CACHE_DIR"] == I and base["TRITON_CACHE_DIR"] == T and base["HOME"] == "/root"
    assert len(notes) == 1 and notes[0][0] == "TORCHINDUCTOR_CACHE_DIR unset" and I in notes[0][1]
    base, notes = LA.launch_base({"TRITON_CACHE_DIR": T, "MODEL_OPT_JIT_ROOT": ROOT_SHARED + "/", "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_root"})   # torch's own default, inherited: = unset
    assert base["TORCHINDUCTOR_CACHE_DIR"] == I and "torch's default" in notes[0][0]
    base, notes = LA.launch_base({"TORCHINDUCTOR_CACHE_DIR": I, "MODEL_OPT_JIT_ROOT": ROOT_SHARED})                    # the other way round
    assert base["TRITON_CACHE_DIR"] == T and notes[0][0] == "TRITON_CACHE_DIR unset"
    for untouched in (dict(WITH_ROOT), {}, {"TRITON_CACHE_DIR": T}, {"MODEL_OPT_JIT_ROOT": ROOT_SHARED}, {"TRITON_CACHE_DIR": "/tmp/ef2inv-jitcache-1/triton", "MODEL_OPT_JIT_ROOT": ROOT_SHARED},
                      {"TRITON_CACHE_DIR": ROOT_SHARED, "MODEL_OPT_JIT_ROOT": ROOT_SHARED}, {"TRITON_CACHE_DIR": T, "TORCHINDUCTOR_CACHE_DIR": "/elsewhere/inductor", "MODEL_OPT_JIT_ROOT": ROOT_SHARED}):
        base, notes = LA.launch_base(dict(untouched))          # no root named, nothing or both set, per-box, the root itself, a foreign sibling: returned as read
        assert base == untouched and notes == [], untouched
    assert LA.TORCH_DEFAULT_INDUCTOR_RE.match("/tmp/torchinductor_root") and not LA.TORCH_DEFAULT_INDUCTOR_RE.match("/tmp/torchinductor_root/x") and not LA.TORCH_DEFAULT_INDUCTOR_RE.match(I)


def test_the_arms_jit_dirs_are_judged_not_the_launching_process():
    """A launching process that imports torch with TORCHINDUCTOR_CACHE_DIR unset carries torch's /tmp default in os.environ; judged raw beside a
    keyed Triton dir under the shared root that reads `shared` + unkeyed. The launch judges the ARM's pair instead (per-box at det >= 1, the
    completed keyed pair at det 0), read from the base before torch."""
    from .. import cli
    T, I = SHARED["TRITON_CACHE_DIR"], SHARED["TORCHINDUCTOR_CACHE_DIR"]
    problems = []
    rep = cli.jit_cache_report({"TRITON_CACHE_DIR": T, "TORCHINDUCTOR_CACHE_DIR": "/tmp/torchinductor_root", "MODEL_OPT_JIT_ROOT": ROOT_SHARED}, KEY, problems)
    assert problems == [] and rep["notes"] == [f"TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor_root is not keyed by the running stack ({KEY})"]
    mode = MD.MODES["off"]
    base, _ = LA.launch_base({"TRITON_CACHE_DIR": T, "MODEL_OPT_JIT_ROOT": ROOT_SHARED})
    for det, want_form in ((1, "per-box"), (0, "shared")):
        env = LA.arm_env(mode, PREFIXES, base=base, det=det)
        view = LA.arm_jit_view(base, env); problems = []
        rep = cli.jit_cache_report(view, KEY, problems)
        assert problems == [] and rep["notes"] == [] and rep["form"] == want_form, (det, rep, problems)
        if det == 0:
            assert env["TRITON_CACHE_DIR"] == T and env["TORCHINDUCTOR_CACHE_DIR"] == I                                # the production arm runs the keyed pair
        else:
            assert not env["TRITON_CACHE_DIR"].startswith(ROOT_SHARED) and os.path.dirname(env["TRITON_CACHE_DIR"]) == os.path.dirname(env["TORCHINDUCTOR_CACHE_DIR"])   # one fresh per-box root
    b2 = dict(base, EF2INV_JIT_CACHE="shared"); env = LA.arm_env(mode, PREFIXES, base=b2, det=1); problems = []   # the operator's explicit shared on a det arm: the completed pair passes through
    assert cli.jit_cache_report(LA.arm_jit_view(b2, env), KEY, problems)["form"] == "shared" and problems == [] and env["TORCHINDUCTOR_CACHE_DIR"] == I
    b3 = dict(base, EF2INV_JIT_CACHE="per-box"); env = LA.arm_env(mode, PREFIXES, base=b3, det=0); problems = []  # a switch that disagrees with the dirs is still worded through the view
    rep = cli.jit_cache_report(LA.arm_jit_view(b3, env), KEY, problems)
    assert problems == [] and rep["notes"] and "imply shared" in rep["notes"][0]


def test_design_resolves_the_jit_pair_through_launch_context(monkeypatch, tmp_path, capsys):
    """`design` reads the base before its launch facts import torch, completes the pair (one NOTE), builds the arm env from that base, judges the
    arm's dirs (judged=arm on the JIT line) — per-box at --det 1, the completed keyed pair at --det 0."""
    from .. import cli
    from .test_model_switches import _design_rig
    ns, card, runs, writes = _design_rig(monkeypatch, tmp_path); card("NVIDIA H100 80GB HBM3", 81559)
    monkeypatch.setenv("TRITON_CACHE_DIR", SHARED["TRITON_CACHE_DIR"]); monkeypatch.setenv("MODEL_OPT_JIT_ROOT", ROOT_SHARED)
    monkeypatch.delenv("TORCHINDUCTOR_CACHE_DIR", raising=False); monkeypatch.delenv("EF2INV_JIT_CACHE", raising=False); monkeypatch.delenv("EF2INV_JIT_LOCAL_ROOT", raising=False)
    for det in (1, 0):
        assert cli.cmd_design(ns(det=det)) == 0
        argv, env = runs[-1]
        assert argv[argv.index("--det") + 1] == str(det)
        if det >= 1:
            assert not env["TRITON_CACHE_DIR"].startswith(ROOT_SHARED) and env["TORCHINDUCTOR_CACHE_DIR"] and os.path.dirname(env["TRITON_CACHE_DIR"]) == os.path.dirname(env["TORCHINDUCTOR_CACHE_DIR"])
        else:
            assert {v: env[v] for v in MD.JIT_DIR_VARS} == SHARED
        err = capsys.readouterr().err
        assert err.count("NOTE TORCHINDUCTOR_CACHE_DIR unset") == 1 and f"JIT det={det} cache={'per-box' if det else 'shared'} judged=arm form={'per-box' if det else 'shared'} key={KEY}" in err, err
    src = open(cli.__file__, encoding="utf-8").read()
    body = src[src.index("def cmd_design("):]; body = body[:body.index("\ndef ", 10)]
    assert "launch_context(a, a.det)" in body and "L.arm_env(" not in body and "det_cache(" not in body     # the design verb resolves through launch_context, no second copy
