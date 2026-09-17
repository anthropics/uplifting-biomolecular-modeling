"""The tp line's lazy template featurizer reaches the process that featurizes. OpenFold3 0.5.0 runs the DataLoader in forkserver workers
(``core/data/framework/data_module.py``): fresh interpreters that import the dataset modules and never the runner. The line's import hook
(``tp_rowpair/hook/sitecustomize.py``, first on a rank's PYTHONPATH) must therefore rebind ``featurize_template_structures_of3`` in every process
of the rank that imports the featurizer module — this test starts a real forkserver child under a rank's environment and asks it which function
the featurizer module and the inference dataset module hold. CPU only; needs the openfold3 wheel (the dataset modules are imported for real)."""
import json
import os
import socket
import subprocess
import sys
import textwrap

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)                                              # …/opt/openfold3_ob0_opt
OPT = os.path.dirname(PKG)                                               # …/opt
HOOK_DIR = os.path.join(PKG, "tp_rowpair", "hook")

PROBE_MAIN = textwrap.dedent('''
    import json, multiprocessing as mp, os, sys

    def which(tag):
        """Import the featurizer module and the inference dataset module the way a DataLoader worker does; report what they hold."""
        from openfold3.core.data.pipelines.featurization import template as T
        import openfold3.core.data.framework.single_datasets.inference as inf
        rec = {"tag": tag, "pid": os.getpid(), "template_module": T.featurize_template_structures_of3.__name__,
               "inference_module": inf.featurize_template_structures_of3.__name__,
               "runner_imported": "openfold3.projects.of3_all_atom.runner" in sys.modules}
        if os.environ.get("OF3TP_RANK"):                                   # the data seams have their own record; the model-seam record stays empty until the runner import installs them
            from openfold3_ob0_opt.tp_rowpair import data, model
            rec.update(data_patches=data.PATCHES.names(), model_patches=model.PATCHES.names())
        return rec

    def child(q):
        q.put(which("forkserver-child"))

    if __name__ == "__main__":
        ctx = mp.get_context("forkserver")
        q = ctx.Queue()
        p = ctx.Process(target=child, args=(q,))
        p.start()
        got = q.get(timeout=900)
        p.join(timeout=120)
        print("PROBE " + json.dumps({"child": got, "parent": which("rank-main"), "child_exitcode": p.exitcode}))
''')


def _unix_sockets_available() -> bool:
    """What multiprocessing's forkserver needs: a fresh listening AF_UNIX socket (a sandbox may allow socketpair() and still refuse this)."""
    import tempfile
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    except OSError:
        return False
    try:
        with tempfile.TemporaryDirectory() as d:
            s.bind(os.path.join(d, "probe.sock"))
            s.listen(1)
        return True
    except OSError:
        return False
    finally:
        s.close()


def _run_probe(tmp_path, env_more: dict) -> dict:
    pytest.importorskip("openfold3", reason="needs the openfold3 wheel (the dataset modules are imported for real)")
    if not _unix_sockets_available():
        pytest.skip("forkserver needs AF_UNIX sockets, which this environment refuses")
    main = tmp_path / "probe_main.py"
    main.write_text(PROBE_MAIN)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([HOOK_DIR, OPT] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])   # the hook dir FIRST, as tp.rank_env puts it
    env.update(env_more)
    r = subprocess.run([sys.executable, str(main)], env=env, capture_output=True, text=True, timeout=1500, cwd=str(tmp_path))
    line = [l for l in r.stdout.splitlines() if l.startswith("PROBE ")]
    assert r.returncode == 0 and line, (r.returncode, r.stdout[-2000:], r.stderr[-4000:])
    out = json.loads(line[-1][len("PROBE "):])
    out["stderr"] = r.stderr
    return out


def test_forkserver_worker_featurizes_with_the_lazy_template_path(tmp_path):
    """Under a rank's environment (OF3TP_WORLD=2, OF3TP_RANK=0, the hook dir first on PYTHONPATH) a forkserver child that
    imports the dataset modules — and never the runner — holds the LAZY featurizer in both the featurizer module and the inference dataset module;
    so does the rank's main process. The hook's per-process lines name where the data patches went."""
    out = _run_probe(tmp_path, {"OF3TP_WORLD": "2", "OF3TP_RANK": "0"})
    for who in ("child", "parent"):
        rec = out[who]
        assert rec["template_module"] == "featurize_template_structures_lazy", (who, rec, out["stderr"][-2000:])
        assert rec["inference_module"] == "featurize_template_structures_lazy", (who, rec, out["stderr"][-2000:])
        assert rec["runner_imported"] is False, (who, rec)                 # the point: no runner import was needed
        assert rec["data_patches"] == ["openfold3.core.data.pipelines.featurization.template.featurize_template_structures_of3"], (who, rec)
        assert rec["model_patches"] == [], (who, rec)                        # or the rank's model install (model.install: `if PATCHES.names(): return`) would find itself 'done' with no seam bound
    assert out["child"]["pid"] != out["parent"]["pid"] and out["child_exitcode"] == 0
    assert out["stderr"].count("data patches installed in pid") >= 2, out["stderr"][-3000:]   # one line per featurizing process (child + main)


def test_no_rank_environment_installs_nothing_in_workers(tmp_path):
    """The n_gpu=1 rule holds in workers too: without OF3TP_WORLD/OF3TP_RANK the hook arms nothing and the stock featurizer stays."""
    out = _run_probe(tmp_path, {"OF3TP_WORLD": "1", "OF3TP_RANK": ""})
    for who in ("child", "parent"):
        assert out[who]["template_module"] == "featurize_template_structures_of3", (who, out[who])
    assert "data patches installed" not in out["stderr"]
