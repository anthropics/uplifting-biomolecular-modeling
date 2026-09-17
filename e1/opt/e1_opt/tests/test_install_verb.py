"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's fetch-then-digest gate (e1_opt.weights.fetch with injected downloaders over the kit's own pins module, its digests
pointed at the test's payload; no network, no torch, no upstream). CPU only."""
import copy
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m e1_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="e1_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        lines = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        self.probes = [c for c in lines if c.startswith("-I -c import os,sys")]           # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")]

    def test_install_runs_pip_then_the_software_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py --checks package,stack"], calls)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/hf_home"], ["install", "--weights=/data/hf_home"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m e1_opt.weights /data/hf_home"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--multi"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--variant", "600m"]):
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
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py --checks package,stack", "-m e1_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)


class WeightsFetchAndDigestGate(unittest.TestCase):
    """e1_opt.weights.fetch over the kit's own pins module (its digests pointed at the test payload): the injected downloaders write what upstream's
    would, then every pinned file is checked with the routes' own comparisons — a mismatch, a missing file or a fetch error fails by name."""

    KERNEL_BYTES = b"# layer_norm.py stand-in\n"

    def setUp(self):
        from e1_opt import stack, weights
        self.W, self.stack = weights, stack
        self.pins = stack.load_pins()
        self.saved = (copy.deepcopy(self.pins.WEIGHTS), copy.deepcopy(self.pins.KERNEL), dict(os.environ))
        self.dir = tempfile.mkdtemp(prefix="e1_weights_")
        self.payload = {size: f"{size}-weights".encode() * 3 for size in self.pins.WEIGHTS}
        for size, blob in self.payload.items():                          # the pins now name the test payload (repo / rev / file layout unchanged)
            self.pins.WEIGHTS[size]["sha256"] = hashlib.sha256(blob).hexdigest(); self.pins.WEIGHTS[size]["bytes"] = len(blob)
        self.pins.KERNEL["layer_norm_py_sha256"] = hashlib.sha256(self.KERNEL_BYTES).hexdigest()
        self.calls = []

    def tearDown(self):
        weights, kernel, env = self.saved
        self.pins.WEIGHTS.clear(); self.pins.WEIGHTS.update(weights); self.pins.KERNEL.clear(); self.pins.KERNEL.update(kernel)
        os.environ.clear(); os.environ.update(env)

    # the downloaders' stand-ins write the layout upstream's write: <cache_dir>/models--<org>--<name>/snapshots/<rev>/<files>
    def download(self, repo_id, revision, cache_dir, **_):
        self.calls.append(("download", repo_id, revision, cache_dir))
        size = [s for s, w in self.pins.WEIGHTS.items() if w["repo"] == repo_id][0]
        snap = os.path.join(cache_dir, "models--" + repo_id.replace("/", "--"), "snapshots", revision); os.makedirs(snap, exist_ok=True)
        for name, blob in (("config.json", b"{}"), ("model.safetensors", self.payload[size])):
            if not os.path.exists(os.path.join(snap, name)):
                with open(os.path.join(snap, name), "wb") as f: f.write(blob)
        return snap

    def install(self, repo_id, revision, **_):
        root = os.environ["KERNELS_CACHE"] if os.environ.get("KERNELS_CACHE") else os.environ["HF_HUB_CACHE"]   # the kernels package's rule: its own root, else the hub cache
        self.calls.append(("install", repo_id, revision, root))
        d = os.path.join(root, "models--" + repo_id.replace("/", "--"), "snapshots", revision, os.path.dirname(self.W.KERNEL_FILE)); os.makedirs(d, exist_ok=True)
        with open(os.path.join(root, "models--" + repo_id.replace("/", "--"), "snapshots", revision, self.W.KERNEL_FILE), "wb") as f: f.write(self.KERNEL_BYTES)
        return "triton_layer_norm", d

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(kw.pop("directory", self.dir), pins=self.pins, download=kw.pop("download", self.download), install=kw.pop("install", self.install), out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_checked_and_the_ref_pinned(self):
        os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["KERNELS_CACHE"] = "/elsewhere"
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 4/4", out); self.assertIn(f"export HF_HOME={self.dir}", out); self.assertIn("HF_HUB_OFFLINE lifted", out)
        hub = os.path.join(self.dir, "hub")
        self.assertEqual([c[0] for c in self.calls], ["download", "download", "download", "install"])
        self.assertTrue(all(c[3] == hub for c in self.calls), self.calls)                     # everything under DIR/hub — KERNELS_CACHE was unset for the step
        self.assertEqual((os.environ.get("HF_HOME"), os.environ.get("KERNELS_CACHE")), (self.dir, None))
        for size in self.pins.WEIGHTS:
            self.assertTrue(os.path.isfile(os.path.join(self.pins.weights_snapshot(size, self.dir), "model.safetensors")))
        ref = os.path.join(hub, "models--kernels-community--triton-layer-norm", "refs", "main")
        self.assertEqual(open(ref).read(), self.pins.KERNEL["rev"])

    def test_present_files_are_kept_and_checked_without_a_fetch(self):
        self.fetch()
        self.calls.clear()
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.calls, []); self.assertEqual(out.count(": present"), 4, out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        snap = self.pins.weights_snapshot("300m", self.dir); os.makedirs(snap)
        p = os.path.join(snap, "model.safetensors")
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        last = out.strip().splitlines()[-1]
        self.assertIn("REFUSED: 1 of 4", last); self.assertIn("300m model.safetensors", last)
        self.assertEqual(open(p, "rb").read(), b"other bytes")                                 # left in place
        self.assertNotIn(("download", "Profluent-Bio/E1-300m"), [c[:2] for c in self.calls])   # present = not fetched again

    def test_a_kernel_file_off_the_pin_is_refused(self):
        self.pins.KERNEL["layer_norm_py_sha256"] = hashlib.sha256(b"another kernel").hexdigest()
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("hub kernel", out.strip().splitlines()[-1])

    def test_a_fetch_error_is_relayed_by_name(self):
        def broken(**_): raise ConnectionError("network unreachable")
        rc, out = self.fetch(download=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching Profluent-Bio/E1-150m@", out); self.assertIn("ConnectionError: network unreachable", out)
        self.assertFalse(any(c[0] == "install" for c in self.calls))                           # the sequence stopped

    def test_a_ref_naming_another_revision_fails_and_is_left_alone(self):
        ref = os.path.join(self.dir, "hub", "models--kernels-community--triton-layer-norm", "refs", "main"); os.makedirs(os.path.dirname(ref))
        with open(ref, "w") as f: f.write("0" * 40)
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("names revision " + "0" * 40, out)
        self.assertEqual(open(ref).read(), "0" * 40)

    def test_the_kernels_cache_form_stages_the_kernel_only(self):
        root = os.path.join(self.dir, "kernels_cache")
        rc, out = self.fetch(directory=root, kernels_cache=True)
        self.assertEqual(rc, 0, out)
        self.assertIn("KERNEL OK: 1/1", out); self.assertIn(f"export KERNELS_CACHE={root}", out)
        self.assertEqual([c[0] for c in self.calls], ["install"]); self.assertEqual(self.calls[0][3], root)
        self.assertEqual(os.environ.get("KERNELS_CACHE"), root)
        self.assertTrue(os.path.isfile(os.path.join(root, "models--kernels-community--triton-layer-norm", "snapshots", self.pins.KERNEL["rev"], self.W.KERNEL_FILE)))
        self.assertEqual(open(os.path.join(root, "models--kernels-community--triton-layer-norm", "refs", "main")).read(), self.pins.KERNEL["rev"])

    def test_without_an_installer_the_kernels_cache_form_stages_the_tree_copy(self):
        """No injected installer: the package directory shipped under stock/hub_kernel is copied into the cache layout (no network), refs/main is pinned,
        and the routes' kernel gate accepts the staged layer_norm.py against the real pin."""
        self.pins.KERNEL["layer_norm_py_sha256"] = self.saved[1]["layer_norm_py_sha256"]        # the real pin: the tree copy must hash to it
        root = os.path.join(self.dir, "kernels_cache")
        rc, out = self.fetch(directory=root, kernels_cache=True, install=None)
        self.assertEqual(rc, 0, out)
        self.assertIn("staging the tree's copy", out); self.assertIn("KERNEL OK: 1/1", out); self.assertEqual(self.calls, [])
        pkg = os.path.join(root, "models--kernels-community--triton-layer-norm", "snapshots", self.pins.KERNEL["rev"], os.path.dirname(self.W.KERNEL_FILE))
        shipped = self.W.vendored_kernel_dir()
        for rel in ("__init__.py", "_ops.py", "layer_norm.py", "layers.py", os.path.join("utils", "__init__.py"), os.path.join("utils", "library.py"), os.path.join("utils", "torch.py")):
            self.assertEqual(open(os.path.join(pkg, rel), "rb").read(), open(os.path.join(shipped, rel), "rb").read(), rel)
        self.assertEqual(open(os.path.join(root, "models--kernels-community--triton-layer-norm", "refs", "main")).read(), self.pins.KERNEL["rev"])

    def test_the_tree_copy_of_the_kernel_hashes_to_the_pins(self):
        """stock/hub_kernel/triton_layer_norm carries exactly the files stock/PINS.json hub_kernel.files_sha256 lists, byte for byte, and its layer_norm.py is the pinned digest."""
        import json
        P = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["hub_kernel"]
        shipped = self.W.vendored_kernel_dir()
        found = sorted(os.path.relpath(os.path.join(d, f), shipped).replace(os.sep, "/") for d, _, fs in os.walk(shipped) for f in fs if "__pycache__" not in d)
        self.assertEqual(found, sorted(P["files_sha256"]))
        for rel, digest in P["files_sha256"].items():
            self.assertEqual(hashlib.sha256(open(os.path.join(shipped, rel), "rb").read()).hexdigest(), digest, rel)
        self.assertEqual(P["files_sha256"]["layer_norm.py"], P["layer_norm_py_sha256"]); self.assertEqual(P["layer_norm_py_sha256"], self.saved[1]["layer_norm_py_sha256"])

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2); self.assertEqual(self.W.main(["--kernels-cache"]), 2); self.assertEqual(self.W.main(["a", "b"]), 2)

    def test_pins_json_names_what_the_step_fetches(self):
        """stock/PINS.json (what README / STOCK.md cite) agrees with the pins module the step reads: the three checkpoints (repo, rev) and the kernel (repo, rev, file)."""
        import json
        P = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        weights, kernel, _ = self.saved
        for size, w in weights.items():
            self.assertEqual((P["weights"][size]["repo"], P["weights"][size]["rev"], P["weights"][size]["sha256"]), (w["repo"], w["rev"], w["sha256"]), size)
        self.assertEqual((P["hub_kernel"]["repo"], P["hub_kernel"]["rev"]), (kernel["repo"], kernel["rev"]))
        self.assertEqual(P["hub_kernel"]["file"], self.W.KERNEL_FILE.replace(os.sep, "/"))
        self.assertEqual(P["hub_kernel"]["layer_norm_py_sha256"], kernel["layer_norm_py_sha256"])


if __name__ == "__main__":
    unittest.main()
