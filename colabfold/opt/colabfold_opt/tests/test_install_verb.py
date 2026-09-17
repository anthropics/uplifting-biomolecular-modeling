"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the weights step's digest gate (colabfold_opt.weights.fetch with an injected upstream route over the package's own comparator,
stack.weights_check, on small files the test pins), and the environment recipe's agreement with stock/PINS.json. CPU only, no network."""
import copy
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest

from colabfold_opt import stack, weights

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))
TREE = os.path.dirname(OPT)                                             # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m colabfold_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="colabfold_install_verb_")
        bindir = os.path.join(self.tmp, "bin"); os.makedirs(bindir)
        stub = os.path.join(bindir, "python")
        with open(stub, "w") as fh:
            fh.write(STUB)
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
        self.log = os.path.join(self.tmp, "calls.log")
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("COLABFOLD_OPT", "AF_PALLAS_ATTN", "MODEL_OPT", "STUB_"))}
        self.env["PATH"] = bindir + os.pathsep + self.env.get("PATH", ""); self.env["STUB_LOG"] = self.log

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_sh(self, *args, **stub_rc):
        env = dict(self.env, **{k: str(v) for k, v in stub_rc.items()})
        if os.path.exists(self.log): os.remove(self.log)                  # one call sequence per invocation
        r = subprocess.run(["bash", RUN_SH, *args], capture_output=True, text=True, env=env, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r, calls

    def kinds(self, calls):
        out = []
        for c in calls:
            if "-I -c import os,sys" in c: out.append("probe")
            elif "-m pip install" in c: out.append("pip")
            elif "check_pins.py" in c: out.append("pins")
            elif "-m colabfold_opt.weights" in c: out.append("weights")
            else: out.append("other:" + c)
        return out

    def test_install_is_core_then_kit_editable_then_the_software_pins(self):
        r, calls = self.run_sh("install")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.kinds(calls), ["probe", "pip", "pins"])
        pip = calls[1]
        self.assertRegex(pip, r"-m pip install -e \S+/\.\./common/opt_core -e \S+/opt$")             # the shared core first, then the kit, both editable, one pip call
        self.assertTrue(pip.split(" -e ")[1].startswith(TREE) and pip.split(" -e ")[2] == os.path.join(TREE, "opt"), pip)
        self.assertEqual(calls[2], f"-I {os.path.join(TREE, 'stock', 'check_pins.py')} --checks packages,files")   # the software pins only: no parameters exist yet at install

    def test_already_installed_from_this_tree_skips_pip(self):
        r, calls = self.run_sh("install", STUB_RC_PROBE=0)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(self.kinds(calls), ["probe", "pins"])
        self.assertIn("installed from this tree already", r.stdout)

    def test_weights_step_runs_after_the_pins_with_the_directory(self):
        r, calls = self.run_sh("install", "--weights", "/some/dir")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertEqual(self.kinds(calls), ["probe", "pip", "pins", "weights"])
        self.assertTrue(calls[3].endswith("-m colabfold_opt.weights /some/dir"), calls[3])
        r, calls = self.run_sh("install", "--weights=/eq/dir")
        self.assertEqual(r.returncode, 0, r.stderr); self.assertTrue(calls[-1].endswith("-m colabfold_opt.weights /eq/dir"), calls[-1])

    def test_failures_exit_by_class_and_stop_the_sequence(self):
        r, calls = self.run_sh("install", STUB_RC_PIP=1)
        self.assertEqual(r.returncode, 1); self.assertEqual(self.kinds(calls), ["probe", "pip"]); self.assertIn("the install failed", r.stderr)
        r, calls = self.run_sh("install", "--weights", "/d", STUB_RC_PINS=3)
        self.assertEqual(r.returncode, 3); self.assertEqual(self.kinds(calls), ["probe", "pip", "pins"]); self.assertIn("refused by the pin check", r.stderr)
        r, calls = self.run_sh("install", "--weights", "/d", STUB_RC_WEIGHTS=1)
        self.assertEqual(r.returncode, 1); self.assertEqual(self.kinds(calls), ["probe", "pip", "pins", "weights"])

    def test_usage_refusals_call_nothing(self):
        for args, words in ((("install", "--config", "h100"), "install takes no --config / --mode"),
                            (("install", "--mode", "fast"), "install takes no --config / --mode"),
                            (("install", "--bogus"), "install takes no argument '--bogus'"),
                            (("install", "--weights"), "install --weights takes a directory"),
                            (("install", "--weights", "--other"), "install --weights takes a directory"),
                            (("install", "extra"), "install takes no argument 'extra'")):
            r, calls = self.run_sh(*args)
            self.assertEqual(r.returncode, 2, (args, r.stderr)); self.assertIn(words, r.stderr); self.assertEqual(calls, [], args)

    def test_usage_text_names_the_verb(self):
        r, _ = self.run_sh()
        self.assertEqual(r.returncode, 2); self.assertIn("run.sh install [--weights DIR]", r.stderr)


class WeightsDigestGate(unittest.TestCase):
    """weights.fetch over the package's own comparator (stack.weights_check → stock/check_pins.py weights_report, digests through the memo):
    small files stand in for the 373 MB parameters — the pin this process reads (stack.pins()) is pointed at their sizes and digests — and an
    injected route stands in for upstream's downloader (it writes what the test says and touches the marker, as upstream does)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="colabfold_weights_gate_"); self.dir = os.path.join(self.tmp, "af2")
        self.real = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        self.saved = stack._CACHE.get("pins"); self.fake = copy.deepcopy(self.real)
        self.bodies = {}
        for i, rel in enumerate(sorted(self.real["weights"]["files"])):
            body = (f"pinned parameters {i}\n" * 64).encode(); self.bodies[rel] = body
            self.fake["weights"]["files"][rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        stack._CACHE["pins"] = self.fake; stack._CACHE.pop("weights_sha256", None)
        self.marker_rel = self.real["weights"]["marker"]["file"]
        self.saved_root = os.environ.get("COLABFOLD_OPT_JIT_ROOT"); os.environ["COLABFOLD_OPT_JIT_ROOT"] = os.path.join(self.tmp, "jit")   # the digest memo lands in the test's own cache root
        self.fetched = []

    def tearDown(self):
        if self.saved is None: stack._CACHE.pop("pins", None)
        else: stack._CACHE["pins"] = self.saved
        stack._CACHE.pop("weights_sha256", None); shutil.rmtree(self.tmp, ignore_errors=True)
        if self.saved_root is None: os.environ.pop("COLABFOLD_OPT_JIT_ROOT", None)
        else: os.environ["COLABFOLD_OPT_JIT_ROOT"] = self.saved_root

    def route(self, bodies=None, marker=True, exc=None):
        """upstream's downloader as a stand-in: writes `bodies` (default: the pinned bytes) under DIR and touches the marker."""
        def _route(model_type):
            self.fetched.append(model_type)
            if exc: raise exc
            for rel, body in (self.bodies if bodies is None else bodies).items():
                p = os.path.join(self.dir, rel); os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as fh: fh.write(body)
            if marker:
                m = os.path.join(self.dir, self.marker_rel); os.makedirs(os.path.dirname(m), exist_ok=True); open(m, "w").close()
        return _route

    def fetch(self, route):
        out = io.StringIO()
        rc = weights.fetch(self.dir, route=route, out=out)
        return rc, out.getvalue()

    def test_empty_dir_fetches_through_upstream_then_every_file_is_pinned(self):
        rc, text = self.fetch(self.route())
        self.assertEqual(rc, 0, text); self.assertEqual(self.fetched, [self.real["weights"]["model_type"]])          # upstream called once, with PINS' model type
        self.assertEqual(text.count("(pinned)"), 5); self.assertIn("WEIGHTS OK: 5/5", text); self.assertIn(f"export COLABFOLD_OPT_DATA_DIR={self.dir}", text)
        self.fetched.clear()
        rc, text = self.fetch(self.route(exc=AssertionError("must not be called")))                                    # idempotent: the marker is present, nothing is fetched, files re-checked
        self.assertEqual(rc, 0, text); self.assertEqual(self.fetched, []); self.assertIn("nothing is fetched", text); self.assertIn("WEIGHTS OK: 5/5", text)

    def test_a_file_off_its_pin_is_refused_by_name_and_left_in_place(self):
        bad_rel = sorted(self.bodies)[2]; bodies = dict(self.bodies); bodies[bad_rel] = b"another checkpoint under the stock name\n"
        rc, text = self.fetch(self.route(bodies=bodies))
        self.assertEqual(rc, 1, text); self.assertIn(f"{bad_rel}: sha256 ", text); self.assertIn("is not the pin's", text)
        self.assertRegex(text, r"REFUSED: 1 of 5 parameter files .* " + re.escape(bad_rel)); self.assertNotIn("WEIGHTS OK", text)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, bad_rel)))                                                # never deleted
        self.assertEqual(text.count("(pinned)"), 4)

    def test_marker_present_but_a_file_missing_is_refused_by_name_without_a_fetch(self):
        bodies = dict(self.bodies); missing = sorted(bodies)[0]; del bodies[missing]
        self.route(bodies=bodies)("prime")                                                                               # a populated dir short one file, marker present
        self.fetched.clear()
        rc, text = self.fetch(self.route(exc=AssertionError("must not be called")))
        self.assertEqual(rc, 1, text); self.assertEqual(self.fetched, []); self.assertIn(f"{missing}: MISSING", text); self.assertIn("remove", text)

    def test_upstream_failure_and_unfinished_transfer_are_relayed(self):
        rc, text = self.fetch(self.route(exc=RuntimeError("Error downloading files")))
        self.assertEqual(rc, 1); self.assertIn("FAILED fetching the alphafold2_multimer_v3 parameters: RuntimeError: Error downloading files", text)
        rc, text = self.fetch(self.route(marker=False))
        self.assertEqual(rc, 1); self.assertIn("marker", text); self.assertIn("is absent", text)

    def test_upstream_downloader_is_the_one_pins_cites(self):
        """PINS weights.source names colabfold's own archive and marker; upstream's download.py (this tree's stock/src copy of the pinned wheel)
        fetches that archive for PINS' model type and touches that marker — the weights step restates neither."""
        src = open(os.path.join(TREE, "stock", "src", "colabfold", "download.py"), encoding="utf-8").read()
        w = self.real["weights"]
        block = src.split(f'model_type == "{w["model_type"]}"', 1)[1].split("elif", 1)[0]                              # upstream's branch for PINS' model type
        self.assertIn(f'"{os.path.basename(w["marker"]["file"])}"', block); self.assertIn("success_marker", block)
        archive = re.search(r"colabfold's own fetch URL is (\S+\.tar)", w["source"]).group(1)
        self.assertIn(f'url = "https://storage.googleapis.com/alphafold/{archive}"', block)
        self.assertEqual(os.path.dirname(w["marker"]["file"]), "params"); self.assertIn('params_dir = data_dir.joinpath("params")', src)   # upstream writes under <DIR>/params, where PINS' relative paths point
        self.assertNotRegex(open(weights.__file__, encoding="utf-8").read(), r"https?://")                                # no URL restated in the kit's step


class EnvironmentRecipe(unittest.TestCase):
    """environment/ builds the pinned stack: the Dockerfile's CUDA base tag is PINS `image.cuda_base_image`'s, its interpreter PINS `image.python`,
    it installs environment/requirements.lock, the two stock wheels PINS names and the kit through `run.sh install`; the lock's entries for the
    packages PINS checks are lock lines; apptainer.def converts the Docker image."""

    def test_dockerfile_builds_the_pinned_stack(self):
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read()
        base = re.search(r"(?m)^FROM (\S+)$", df).group(1)
        self.assertEqual(base, f"nvidia/cuda:{pins['stack']['cuda_base_image']}-runtime-ubuntu22.04")                  # PINS stack.cuda_base_image (12.4.1) → that release's runtime flavour on Ubuntu 22.04
        self.assertIn(f"cpython-{pins['image']['python_version']}+", df); self.assertEqual(pins["stack"]["python"], pins["image"]["python_version"])
        self.assertIn("COPY colabfold/environment/requirements.lock ", df)
        self.assertRegex(df, r"(?m)^COPY colabfold/stock/\*\.whl /tmp/stock/$"); self.assertIn("pip install --no-cache-dir --no-deps /tmp/stock/*.whl", df)   # the stock wheels: whatever stock/ holds …
        self.assertEqual(sorted(f for f in os.listdir(os.path.join(TREE, "stock")) if f.endswith(".whl")),
                         sorted(os.path.basename(pins["upstream"][k]["wheel"]["file"]) for k in ("colabfold", "alphafold_colabfold")))   # … which is exactly the two wheels PINS names
        self.assertRegex(df, r"(?m)^RUN bash run\.sh install$")
        env = dict(re.findall(r"(?m)^(?:ENV | +)([A-Z_]+)=(\S+?)(?: \\)?$", df))
        for k in ("JAX_PLATFORMS", "JAX_COMPILATION_CACHE_DIR", "XLA_PYTHON_CLIENT_MEM_FRACTION", "TF_FORCE_UNIFIED_MEMORY", "PYTHONHASHSEED", "CFLAGS"):
            self.assertIn(k, env, k)                                                                                       # the tested stack's process environment, every name of it
        self.assertEqual(env, pins["image"]["env"])                                                                        # and stock/PINS.json image.env restates exactly this block — one list
        for k in env:
            self.assertNotRegex(k, r"NUM_THREADS|^PYTHONPATH$|COLABFOLD_OPT_DATA_DIR|JIT_ROOT", k)                            # no cpu-count-tracking pools, payload paths or weights roots baked in
        self.assertRegex(open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read(), r"(?m)^From: colabfold-kit:")

    def test_lock_restates_the_pinned_packages(self):
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        lock = {}
        for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#"):
                name, _, version = line.partition("=="); lock[name.lower().replace("_", "-")] = version
        self.assertEqual(pins["freeze"], {"file": "environment/requirements.lock", "packages": len(lock)})                # the pin check reads this very file: the stack's one package list
        for name in pins["check_packages"]:
            self.assertIn(name.lower().replace("_", "-"), lock, name)                                                      # every package the pin check holds is a lock line
        self.assertEqual(lock["colabfold"], pins["upstream"]["colabfold"]["version"]); self.assertEqual(lock["alphafold-colabfold"], pins["upstream"]["alphafold_colabfold"]["version"])
        self.assertEqual([f for f in os.listdir(os.path.join(TREE, "stock")) if "freeze" in f], [])                       # no second list of the stack in the kit
        self.assertEqual([f for f in os.listdir(os.path.join(TREE, "environment")) if f.startswith("requirements")], ["requirements.lock"])   # one stack, one lock



if __name__ == "__main__":
    unittest.main()
