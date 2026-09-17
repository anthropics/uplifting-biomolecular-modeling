"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the weights step's digest gate (chrombpnet_opt.weights on a temporary weights root; no network), and the environment recipe's agreement
with stock/PINS.json (environment/Dockerfile + the two locks name the pinned base, versions, side root and stock archive). CPU only."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
ENV_DIR = os.path.join(TREE, "environment")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m chrombpnet_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="chrombpnet_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({"STUB_RC_%s" % k.upper(): str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, ["-m pip install -e %s/opt" % TREE, "-I %s/stock/check_pins.py" % TREE], calls)   # the kit alone: it depends on no opt_core

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/weights"], ["install", "--weights=/data/weights"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(len(calls), 3, calls)
            self.assertEqual(calls[2], "-m chrombpnet_opt.weights /data/weights", calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--det"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "fast"], ["install", "--config", "h100"], ["install", "--det"], ["--mode", "off", "install"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, ["-I %s/stock/check_pins.py" % TREE, "-m chrombpnet_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_install_is_in_the_usage_text_and_the_other_verbs_keep_their_gate(self):
        rc, out, calls = self.run_sh([])
        self.assertEqual(rc, 2); self.assertIn("run.sh install [--weights DIR]", out)
        rc, out, calls = self.run_sh(["check"], pins=3)                                     # the pre-existing verbs still run stock/check_pins.py first (no -I) and refuse on it
        self.assertEqual(rc, 3, out); self.assertEqual(calls, ["%s/stock/check_pins.py" % TREE], calls)


class WeightsDigestGate(unittest.TestCase):
    """chrombpnet_opt.weights.check / main: every pinned file under the root hashed with stock/check_pins.py's sha256_file against stock/PINS.json
    weights — a missing file or a digest off the pin fails by name; nothing is fetched, nothing deleted."""

    def setUp(self):
        from chrombpnet_opt import weights
        self.W = weights
        self.root = tempfile.mkdtemp(prefix="chrombpnet_weights_")
        self.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        self.files = self.W.pinned_files(self.pins)

    def plant(self, rel, data):
        p = os.path.join(self.root, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(data)
        return p

    def test_pinned_files_are_stock_mds_three(self):
        rels = [rel for rel, _, _ in self.files]
        self.assertEqual(rels, ["GM12878_ATAC/fold_0/bias_scaled.h5", "GM12878_ATAC/fold_0/chrombpnet_recompiled.h5", "GM12878_ATAC/fold_0/nobias.h5"])
        for rel, sha, size in self.files:
            self.assertRegex(sha, r"^[0-9a-f]{64}$"); self.assertGreater(size, 0)
            self.assertIn(os.path.basename(rel), self.W.ORIGIN)                            # every pinned file has a where-it-comes-from line

    def test_missing_and_mismatch_are_named_and_left_in_place(self):
        p = self.plant("GM12878_ATAC/fold_0/nobias.h5", b"not the model")
        out = io.StringIO()
        r = self.W.check(self.root, stream=out)
        text = out.getvalue()
        self.assertEqual(r["GM12878_ATAC/fold_0/nobias.h5"]["status"], "MISMATCH")
        self.assertEqual(r["GM12878_ATAC/fold_0/bias_scaled.h5"]["status"], "MISSING")
        self.assertIn("MISMATCH  " + p, text); self.assertIn(hashlib.sha256(b"not the model").hexdigest(), text)
        self.assertIn("ENCFF142IOR", text)                                                # the MISSING line says where the file comes from
        self.assertTrue(os.path.isfile(p))                                                 # left in place
        self.assertEqual(self.W.main([self.root]), 1)

    def test_the_gate_uses_the_pin_gates_own_digest_routine(self):
        """A file whose sha256 equals the pin passes: the digest is check_pins.sha256_file's (planted through a pin table pointing at small bytes)."""
        data = b"tiny stand-in weights"
        tree = tempfile.mkdtemp(prefix="chrombpnet_tree_"); os.makedirs(os.path.join(tree, "stock"))
        pins = {"weights": {"root": "test", "GM12878_ATAC/fold_0": {n: {"sha256": hashlib.sha256(data + n.encode()).hexdigest(), "size_bytes": len(data) + len(n)}
                                                                   for n in ("chrombpnet_recompiled.h5", "bias_scaled.h5", "nobias.h5")}}}
        with open(os.path.join(tree, "stock", "PINS.json"), "w") as f: json.dump(pins, f)
        with open(os.path.join(TREE, "stock", "check_pins.py"), "rb") as src, open(os.path.join(tree, "stock", "check_pins.py"), "wb") as dst: dst.write(src.read())
        for n in ("chrombpnet_recompiled.h5", "bias_scaled.h5", "nobias.h5"): self.plant("GM12878_ATAC/fold_0/" + n, data + n.encode())
        out = io.StringIO()
        r = self.W.check(self.root, tree=tree, stream=out)
        self.assertEqual({v["status"] for v in r.values()}, {"ok"}, out.getvalue())
        self.assertEqual(out.getvalue().count("ok        "), 3)

    def test_usage_and_not_a_directory(self):
        self.assertEqual(self.W.main([]), 2)
        self.assertEqual(self.W.main(["--weights"]), 2)
        self.assertEqual(self.W.main([os.path.join(self.root, "absent")]), 1)


class EnvironmentRecipeMatchesPins(unittest.TestCase):
    """environment/ is one definition of the pinned stack: the Dockerfile's base is PINS docker_base, its stock archive is PINS upstream.archive, its side
    root is pins_s1_only.root; requirements.lock carries every PINS pin at its version and PINS python's minor; requirements-opt-torch.lock carries torch /
    triton at pins_s1_only's versions; the image's one install step is `run.sh install`; apptainer.def builds FROM the Dockerfile's image."""

    def setUp(self):
        self.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        self.dockerfile = open(os.path.join(ENV_DIR, "Dockerfile"), encoding="utf-8").read()
        self.lock = open(os.path.join(ENV_DIR, "requirements.lock"), encoding="utf-8").read()
        self.torch_lock = open(os.path.join(ENV_DIR, "requirements-opt-torch.lock"), encoding="utf-8").read()

    @staticmethod
    def lock_versions(text):
        return {m.group(1).lower().replace("_", "-"): m.group(2) for m in re.finditer(r"^([A-Za-z0-9_.\-]+)==(\S+)$", text, re.M)}

    def test_dockerfile_base_archive_root_and_install_step(self):
        base = self.pins["docker_base"].split()[0]
        self.assertRegex(self.dockerfile, r"(?m)^FROM %s$" % re.escape(base))
        archive = os.path.basename(self.pins["upstream"]["chrombpnet"]["archive"])
        self.assertTrue(os.path.isfile(os.path.join(TREE, "stock", archive)), archive)
        self.assertIn("COPY chrombpnet/stock/%s " % archive, self.dockerfile)
        self.assertIn("--target %s " % self.pins["pins_s1_only"]["root"], self.dockerfile)
        self.assertRegex(self.dockerfile, r"(?m)^RUN bash run.sh install$")
        self.assertIn("COPY chrombpnet/environment/requirements.lock ", self.dockerfile)
        self.assertIn("COPY chrombpnet/environment/requirements-opt-torch.lock ", self.dockerfile)
        for var in ("CHROMBPNET_OPT_WEIGHTS=", "CHROMBPNET_OPT_DATA=", "OMP_NUM_THREADS", "MODAL_"):      # no weights/data root and no platform plumbing baked in
            self.assertNotRegex(self.dockerfile, r"(?m)^ENV[^\n]*%s|^\s+%s" % (re.escape(var), re.escape(var)), var)

    def test_lock_carries_every_pin(self):
        have = self.lock_versions(self.lock)
        norm = lambda v: re.sub(r"(\.0)+$", "", v.split("+")[0])
        for name, want in self.pins["pins"].items():
            key = name.lower().replace("_", "-")
            if key == "chrombpnet": continue                                                # stock is the archive in stock/, not a lock line
            self.assertIn(key, have, name); self.assertEqual(norm(have[key]), norm(want), name)
        self.assertNotIn("chrombpnet", have)                                              # `@ file:///` lines are not portable: stock installs from stock/
        self.assertNotRegex(self.lock, r"(?m)^(torch|triton)==")                          # the K1 stack lives in the side root's lock, not here
        self.assertIn("Python %s" % self.pins["python"], self.lock.splitlines()[0] + self.lock.splitlines()[1])

    def test_side_root_lock_carries_torch_and_triton_at_the_s1_pins(self):
        have = self.lock_versions(self.torch_lock); s1 = self.pins["pins_s1_only"]
        self.assertEqual(have.get("triton"), s1["triton"])
        m = re.search(r"(?m)^torch @ https://download\.pytorch\.org/whl/cu124/torch-([0-9.]+)%2Bcu124-cp38-cp38-linux_x86_64\.whl#sha256=[0-9a-f]{64}$", self.torch_lock)
        self.assertTrue(m, "torch is pinned by its file on download.pytorch.org with a sha256"); self.assertEqual(m.group(1) + "+cu124", s1["torch"])
        for lock in (self.lock, self.torch_lock):                                          # no index lines: every line is one file (PyPI's, or a direct URL)
            self.assertNotRegex(lock, r"(?m)^--(extra-)?index-url|^--find-links|^-f ")

    def test_apptainer_def_builds_from_the_dockerfiles_image(self):
        text = open(os.path.join(ENV_DIR, "apptainer.def"), encoding="utf-8").read()
        self.assertRegex(text, r"(?m)^Bootstrap: docker-daemon$"); self.assertRegex(text, r"(?m)^From: chrombpnet-kit:dev$")
        self.assertIn("exec bash /kit/chrombpnet/run.sh \"$@\"", text); self.assertIn("PYTHONNOUSERSITE=1", text)


if __name__ == "__main__":
    unittest.main()
