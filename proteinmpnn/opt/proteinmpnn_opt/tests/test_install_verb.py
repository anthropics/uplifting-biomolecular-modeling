"""`run.sh install [--weights DIR]` and proteinmpnn_opt.weights: the verb's step order on a stub interpreter (pip step or its named
skip → the core pin gate → the weights fetch when asked → the pin check of what is installed), its usage errors, and the weights module end to end
without network — a kept checkout digested against PINS.json, wrong bytes refused by name and left in place, the clone's preconditions (an empty
directory, git on PATH). CPU only."""
import contextlib
import copy
import hashlib
import io
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # proteinmpnn/
RUN_SH = os.path.join(KIT, "run.sh")
PINS = json.load(open(os.path.join(KIT, "stock", "PINS.json"), encoding="utf-8"))


def _exe(path, text):
    with open(path, "w") as fh:
        fh.write(text)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


class RunShInstall(unittest.TestCase):
    """run.sh install on a stub `python` that records its argv: the verb's own logic, no interpreter work."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mpnn_install_")
        self.log = os.path.join(self.tmp, "calls.log")
        self.bin = os.path.join(self.tmp, "bin"); os.mkdir(self.bin)

    def stub(self, body):
        _exe(os.path.join(self.bin, "python"), "#!/bin/bash\necho \"$*\" >> %s\n%s\n" % (self.log, body))

    def run_sh(self, *args, **env):
        e = dict(os.environ, PATH=self.bin + os.pathsep + os.environ.get("PATH", ""), PROTEINMPNN_VARIANT="", **env)
        r = subprocess.run(["bash", RUN_SH, "install", *args], capture_output=True, text=True, env=e)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r, calls

    def test_fresh_environment_with_weights(self):
        self.stub('case "$*" in *find_spec*) exit 1 ;; esac; exit 0')          # not yet installed from this tree: the pip step runs
        r, calls = self.run_sh("--weights", self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr)
        steps = [c.split()[:3] for c in calls]
        self.assertIn("find_spec", calls[0])                                                      # 1 the installed-from-this-tree probe
        self.assertEqual(calls[1].split()[:3], ["-m", "pip", "install"])                          # 2 pip install -e core -e opt
        self.assertIn("-e %s" % os.path.join(KIT, "..", "common", "opt_core"), calls[1]); self.assertIn("-e %s" % os.path.join(KIT, "opt"), calls[1])
        self.assertLess(calls[1].index("opt_core"), calls[1].index(os.path.join(KIT, "opt")))     # the core first
        self.assertIn("core_gate", calls[2])                                                      # 3 the core pin gate
        self.assertEqual(calls[3], "-m proteinmpnn_opt.weights %s" % self.tmp)                     # 4 the weights fetch + digest check
        self.assertEqual(calls[4], "-I %s --checks package" % os.path.join(KIT, "stock", "check_pins.py"))   # 5 the pin check of what is installed
        self.assertEqual(len(calls), 5)

    def test_installed_tree_skips_pip_and_variant_selects_the_check(self):
        self.stub("exit 0")                                                     # already installed from this tree
        r, calls = self.run_sh("--variant", "vanilla", "--weights=" + self.tmp)          # --variant: accepted and inert, one checkout serves both weight sets
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("the pip step is skipped", r.stdout)
        self.assertFalse(any("pip install" in c for c in calls))
        self.assertEqual(calls[-2], "-m proteinmpnn_opt.weights %s" % self.tmp)
        self.assertTrue(calls[-1].endswith("check_pins.py --checks package"), calls[-1])

    def test_bare_install_ends_at_the_pin_check_and_relays_its_refusal(self):
        self.stub('case "$*" in *check_pins.py*) echo "check_pins: MPNN_DIR is not set" >&2; exit 3 ;; esac; exit 0')
        r, calls = self.run_sh()
        self.assertEqual(r.returncode, 3)
        self.assertFalse(any("proteinmpnn_opt.weights" in c for c in calls))    # no --weights: nothing fetched
        self.assertIn("refused by the pin check", r.stderr); self.assertIn("MPNN_DIR", r.stderr)

    def test_usage_errors(self):
        self.stub("exit 0")
        for args in (["--bogus"], ["--weights"], ["--weights", "--variant", "x"], ["--variant", "nope", "--weights", self.tmp], ["extra"]):
            r, _ = self.run_sh(*args)
            self.assertEqual(r.returncode, 2, (args, r.stderr))


class WeightsModule(unittest.TestCase):
    """proteinmpnn_opt.weights.main on temporary directories, PINS.json digests swapped for the test files' where a match is wanted."""

    @classmethod
    def setUpClass(cls):
        sys.path[:0] = [os.path.join(KIT, "opt"), os.path.join(KIT, "..", "common", "opt_core")]
        os.environ.setdefault("MODEL_OPT", KIT)
        from proteinmpnn_opt import weights
        cls.W = weights

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mpnn_weights_")
        self.saved_path = os.environ.get("PATH", "")

    def tearDown(self):
        os.environ["PATH"] = self.saved_path

    def main(self, *argv, pins=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = self.W.main(list(argv), pins=pins)
        return rc, out.getvalue(), err.getvalue()

    @staticmethod
    def digest(data):
        return hashlib.sha256(data).hexdigest()

    def fake_checkout(self, d):
        os.makedirs(d, exist_ok=True); open(os.path.join(d, "protein_mpnn_run.py"), "w").close()
        data = {}
        for key, ent in PINS["weights"].items():
            p = os.path.join(d, ent["path"]); os.makedirs(os.path.dirname(p), exist_ok=True)
            data[key] = ("test weights %s" % key).encode(); open(p, "wb").write(data[key])
        return data

    def test_kept_checkout_digested_ok(self):
        d = os.path.join(self.tmp, "ProteinMPNN"); data = self.fake_checkout(d)
        pins = copy.deepcopy(PINS)
        for key in pins["weights"]: pins["weights"][key]["sha256"] = self.digest(data[key])
        rc, out, err = self.main(d, pins=pins)
        self.assertEqual(rc, 0, err)
        self.assertIn("weights: kept: %s already holds protein_mpnn_run.py (not a git checkout" % d, out)
        self.assertIn("WEIGHTS OK: 2/2 pinned files match stock/PINS.json under %s — export MPNN_DIR=%s" % (d, d), out)

    def test_wrong_bytes_refused_by_name_and_left_in_place(self):
        d = os.path.join(self.tmp, "ProteinMPNN"); self.fake_checkout(d)
        rc, out, err = self.main(d, "--variant", "vanilla", pins=copy.deepcopy(PINS))     # the real digests: the test bytes are not them
        self.assertEqual(rc, 1)
        for ent in PINS["weights"].values():
            self.assertIn("WEIGHTS REFUSED: %s: sha256 " % ent["path"], err)
            self.assertTrue(os.path.isfile(os.path.join(d, ent["path"])))
        self.assertIn("left in place", err)

    def test_clone_preconditions(self):
        d = os.path.join(self.tmp, "busy"); os.makedirs(d); open(os.path.join(d, "x"), "w").close()
        rc, out, err = self.main(d, pins=copy.deepcopy(PINS))
        self.assertEqual(rc, 1); self.assertIn("WEIGHTS FAILED: %s is not empty and holds no protein_mpnn_run.py" % d, err)
        empty_bin = os.path.join(self.tmp, "nobin"); os.makedirs(empty_bin); os.environ["PATH"] = empty_bin
        rc, out, err = self.main(os.path.join(self.tmp, "fresh"), pins=copy.deepcopy(PINS))
        self.assertEqual(rc, 1); self.assertIn("WEIGHTS FAILED: git is not on PATH", err); self.assertIn(PINS["upstream"]["repo"], err)





if __name__ == "__main__":
    unittest.main()
