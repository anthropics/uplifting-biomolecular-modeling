"""stock/PINS.json against the bytes it pins: the two wheels present on disk at their pinned sizes, the three pinned files read
directly out of the wheels and compared to this tree's own stock/src/ copies, every check_packages entry in the freeze, the freeze
count (environment/requirements.lock), the parameters block (five files + the marker), and stock/check_pins.py on this box (no colabfold installed: the packages
and files not met)."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import unittest
import zipfile

from colabfold_opt import stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TREE = os.path.dirname(OPT)
STOCK = os.path.join(TREE, "stock")


def pins():
    return json.load(open(os.path.join(STOCK, "PINS.json")))


def wheel_of(rel):
    p = pins()
    up = p["upstream"]["colabfold" if rel.startswith("colabfold/") else "alphafold_colabfold"]
    return os.path.join(TREE, up["wheel"]["file"])


class TestPins(unittest.TestCase):
    def test_stock_src_matches_the_wheels(self):
        p = pins()
        py_rels = ("colabfold/batch.py", "colabfold/input.py", "colabfold/download.py", "colabfold/utils.py", "colabfold/alphafold/models.py",
                   "alphafold/model/config.py", "alphafold/model/modules.py")                       # the cited modules, the wheels' bytes
        license_rels = ("alphafold/LICENSE", "colabfold/LICENSE")                                    # carried beside them, from each wheel's dist-info
        on_disk = sorted(os.path.relpath(os.path.join(dp, f), os.path.join(STOCK, "src")) for dp, _, fn in os.walk(os.path.join(STOCK, "src")) for f in fn)
        self.assertEqual(on_disk, sorted(py_rels + license_rels))
        for rel in py_rels:
            with zipfile.ZipFile(wheel_of(rel)) as z:
                self.assertEqual(open(os.path.join(STOCK, "src", rel), "rb").read(), z.read(rel), rel)   # byte-identical, read directly — no stored digest
        for rel in license_rels:
            dist = "alphafold_colabfold-2.3.13" if rel.startswith("alphafold/") else "colabfold-1.6.1"
            with zipfile.ZipFile(wheel_of(rel)) as z:
                self.assertEqual(open(os.path.join(STOCK, "src", rel), "rb").read(), z.read(f"{dist}.dist-info/LICENSE"), rel)
        self.assertEqual(sorted(p["stock_files"]), ["alphafold/model/modules.py", "colabfold/batch.py", "colabfold/input.py"])
        for key in ("colabfold", "alphafold_colabfold"):
            w = p["upstream"][key]["wheel"]
            self.assertEqual(os.path.getsize(os.path.join(TREE, w["file"])), w["bytes"])
        self.assertEqual(p["freeze"]["file"], "environment/requirements.lock")                                          # the stack's one package list, relative to the kit root
        lines = [ln for ln in open(os.path.join(TREE, p["freeze"]["file"])).read().splitlines() if ln.strip() and not ln.startswith("#")]
        self.assertEqual(len(lines), p["freeze"]["packages"]); self.assertEqual(len(lines), 77)
        self.assertEqual(p["upstream"]["colabfold"]["version"], "1.6.1"); self.assertEqual(p["upstream"]["alphafold_colabfold"]["version"], "2.3.13")
        self.assertIn("colabfold==1.6.1", lines); self.assertIn("alphafold-colabfold==2.3.13", lines); self.assertIn("jax==0.5.3", lines)

    def test_entry_point_and_ranges(self):
        p = pins()
        with zipfile.ZipFile(wheel_of("colabfold/batch.py")) as z:
            ep = z.read([n for n in z.namelist() if n.endswith("entry_points.txt")][0]).decode()
            md = z.read([n for n in z.namelist() if n.endswith("METADATA")][0]).decode()
        self.assertIn("colabfold_batch=colabfold.batch:main", ep.replace(" ", ""))
        self.assertEqual(p["upstream"]["colabfold"]["cli"]["entry_point"], "colabfold.batch:main")
        self.assertIn("Requires-Python: >=3.10", md); self.assertEqual(p["upstream"]["colabfold"]["requires_python"], ">=3.10")
        self.assertIn("Requires-Dist: alphafold-colabfold (==2.3.13)", md)
        freeze = dict(ln.split("==", 1) for ln in open(os.path.join(TREE, p["freeze"]["file"])).read().splitlines() if "==" in ln and not ln.startswith("#"))
        norm = {k.lower().replace("_", "-"): v for k, v in freeze.items()}
        for name in p["check_packages"]:
            self.assertIn(name.lower().replace("_", "-"), norm, name)
        self.assertEqual(norm["nvidia-cuda-runtime-cu12"], "12.9.79"); self.assertEqual(p["stack"]["cuda_runtime"], "12.9 (nvidia-cuda-runtime-cu12 12.9.79)")

    def test_weights_block(self):
        w = pins()["weights"]
        self.assertEqual(sorted(w["files"]), [f"params/params_model_{i}_multimer_v3.npz" for i in range(1, 6)])
        for spec in w["files"].values():
            self.assertRegex(spec["sha256"], r"^[0-9a-f]{64}$"); self.assertEqual(spec["bytes"], 373043148)
        self.assertEqual(w["marker"]["file"], "params/download_complexes_multimer_v3_finished.txt"); self.assertEqual(w["marker"]["bytes"], 0)
        self.assertEqual((w["env"], w["flag"], w["model_type"]), (stack.ENV_DATA, "--data", "alphafold2_multimer_v3"))
        self.assertNotIn("root", w)                                                                                                # the root comes from the environment (--data / COLABFOLD_OPT_DATA_DIR), never a baked path
        with zipfile.ZipFile(wheel_of("colabfold/batch.py")) as z:
            dl = z.read("colabfold/download.py").decode().splitlines()
        self.assertEqual(dl[38].strip(), 'params_dir = data_dir.joinpath("params")')                                               # :39
        self.assertIn('"download_complexes_multimer_v3_finished.txt"', dl[42])                                                     # :43

    def test_image_block(self):
        img = pins()["image"]
        self.assertNotIn("id", img); self.assertIn("tested_on", img); self.assertEqual(img["python_version"], "3.11.5")   # the image is a documentation line, never compared
        self.assertEqual(img["env"]["JAX_PLATFORMS"], "cuda"); self.assertEqual(img["env"]["XLA_PYTHON_CLIENT_MEM_FRACTION"], "0.95")
        self.assertEqual(img["env"]["TF_FORCE_UNIFIED_MEMORY"], "0")
        with open(os.path.join(TREE, "environment", "Dockerfile")) as f:                                                            # image.env restates the Dockerfile's one ENV block, nothing else
            m = re.search(r"^ENV ((?:[A-Z_][A-Z0-9_]*=\S+(?:\s*\\\n\s*)?)+)", f.read(), re.M)
        self.assertIsNotNone(m); self.assertEqual(img["env"], dict(re.findall(r"([A-Z_][A-Z0-9_]*)=(\S+)", m.group(1))))
        with zipfile.ZipFile(wheel_of("colabfold/batch.py")) as z:
            b = z.read("colabfold/batch.py").decode().splitlines()
        self.assertEqual(b[3].strip(), 'ENV = {"TF_FORCE_UNIFIED_MEMORY":"1", "XLA_PYTHON_CLIENT_MEM_FRACTION":"4.0"}')                # :4
        self.assertEqual(b[5].strip(), "if k not in os.environ: os.environ[k] = v")                                                  # :6


class TestCheckPinsScript(unittest.TestCase):
    def test_script_on_this_box(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("COLABFOLD_OPT")}
        r = subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py"), "--json"], capture_output=True, text=True, env=env)
        rep = json.loads(r.stdout)
        self.assertEqual(rep["weights"]["ok"], None)                                                       # no data directory named: unchecked
        self.assertIn(r.returncode, (0, 3))
        if rep["packages"]["ok"]:
            self.assertEqual(r.returncode, 0 if rep["files"]["ok"] else 3)
        else:
            self.assertEqual(r.returncode, 3)
        r = subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py"), "--data", "/nonexistent"], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 3); self.assertIn("weights=NOT MET", r.stdout); self.assertIn("marker missing", r.stdout)
        self.assertTrue(r.stdout.startswith("[colabfold-opt] PINS "))

    def test_package_uses_the_script(self):
        cp = stack.check_pins_module()
        self.assertTrue(cp.__file__.endswith(os.path.join("stock", "check_pins.py")))
        self.assertTrue(callable(cp.files_report) and callable(cp.weights_report) and callable(cp.packages_report))
        spec = importlib.util.find_spec("colabfold_opt.stack")
        src = open(spec.origin, encoding="utf-8").read()
        self.assertIn("cp.files_report(p)", src); self.assertIn("cp.weights_report(pins(), data_dir, digest=True, digest_fn=functools.partial(weights_digest, refresh=refresh))", src)


class TestWeightsGate(unittest.TestCase):
    """The parameters gate is PRESENCE; the bytes are a reported state: the pinned digests print `(pinned)`, any other checkpoint under the
    stock file names prints `NOT PINNED — …` and RUNS, a missing file or marker is refused by name. Small files stand in for the
    373 MB parameters: the pin this process reads (`stack.pins()`) is pointed at their sizes and digests for the test."""

    def setUp(self):
        import copy, tempfile
        self.tmp = tempfile.mkdtemp(); self.data = os.path.join(self.tmp, "af2"); os.makedirs(os.path.join(self.data, "params"))
        real = pins(); self.saved = stack._CACHE.get("pins"); fake = copy.deepcopy(real)
        open(os.path.join(self.data, real["weights"]["marker"]["file"]), "w").close()
        for i, rel in enumerate(sorted(real["weights"]["files"])):
            body = (f"pinned parameters {i}\n" * 64).encode()
            with open(os.path.join(self.data, rel), "wb") as f:
                f.write(body)
            fake["weights"]["files"][rel] = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
        stack._CACHE["pins"] = fake; stack._CACHE.pop("weights_sha256", None)
        self.rels = sorted(real["weights"]["files"])
        self.saved_root = os.environ.get("COLABFOLD_OPT_JIT_ROOT"); os.environ["COLABFOLD_OPT_JIT_ROOT"] = os.path.join(self.tmp, "jit")   # the digest memo lands in the test's own cache root

    def tearDown(self):
        import shutil
        if self.saved is None:
            stack._CACHE.pop("pins", None)
        else:
            stack._CACHE["pins"] = self.saved
        stack._CACHE.pop("weights_sha256", None); shutil.rmtree(self.tmp, ignore_errors=True)
        if self.saved_root is None:
            os.environ.pop("COLABFOLD_OPT_JIT_ROOT", None)
        else:
            os.environ["COLABFOLD_OPT_JIT_ROOT"] = self.saved_root

    def test_pinned_digests_print_pinned(self):
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, []); self.assertIs(det["ok"], True); self.assertIs(det["pinned"], True)
        lines = stack.weights_lines(det)
        self.assertEqual(len(lines), 5)
        for rel, line in zip(self.rels, lines):
            d = det["files"][rel]
            self.assertEqual(d["state"], "pinned"); self.assertEqual(d["sha256"], stack.pins()["weights"]["files"][rel]["sha256"])
            self.assertEqual(line, f"[colabfold-opt] weights={os.path.basename(rel)} sha256={d['sha256'][:12]} (pinned)")

    def test_another_checkpoint_is_not_pinned_and_runs(self):
        same_size, other_size = self.rels[1], self.rels[3]
        body = open(os.path.join(self.data, same_size), "rb").read()
        open(os.path.join(self.data, same_size), "wb").write(body[::-1])                       # the same size, other bytes: only the digest tells
        open(os.path.join(self.data, other_size), "wb").write(b"a user checkpoint of another size")
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, [], "another checkpoint is never a refusal"); self.assertIs(det["ok"], True); self.assertIs(det["pinned"], False)
        states = {rel: det["files"][rel]["state"] for rel in self.rels}
        self.assertEqual(states, {r: ("not_pinned" if r in (same_size, other_size) else "pinned") for r in self.rels})
        lines = stack.weights_lines(det)
        unc = [l for l in lines if "NOT PINNED" in l]
        self.assertEqual(len(lines), 5); self.assertEqual(len(unc), 2)
        d = det["files"][same_size]
        self.assertEqual(unc[0], f"[colabfold-opt] weights={os.path.basename(same_size)} sha256={d['sha256'][:12]} NOT PINNED — the kit's measurements apply to the pinned weights only")
        self.assertNotEqual(d["sha256"], d["pinned_sha256"]); self.assertTrue(d["size_ok"]); self.assertFalse(det["files"][other_size]["size_ok"])
        # the launcher's gate (`pred`, every mode incl. off): the lines, no refusal, the launch proceeds; the verdict rides in the gate report → the manifest
        import io, contextlib
        from colabfold_opt import cli
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc, rep = cli.pred_gates("off", self.data)
        self.assertIsNone(rc); self.assertIsNone(rep["would_refuse"]); self.assertIs(rep["weights"]["pinned"], False)
        self.assertEqual([l for l in err.getvalue().splitlines() if "NOT PINNED" in l], unc); self.assertNotIn("NOT ACTIVE", err.getvalue())
        # the script run.sh runs first: present-but-not-pinned parameters are reported, never NOT MET (sizes without --digest, digests with it)
        cp = stack.check_pins_module()
        rep = cp.weights_report(stack.pins(), self.data)
        self.assertIs(rep["ok"], True); self.assertIs(rep["pinned"], False); self.assertEqual(rep["files"][other_size]["state"], "not_pinned"); self.assertEqual(rep["files"][same_size]["state"], "size_ok")
        rep = cp.weights_report(stack.pins(), self.data, digest=True)
        self.assertIs(rep["ok"], True); self.assertEqual(rep["files"][same_size]["state"], "not_pinned")
        self.assertEqual(len(cp.weights_lines(rep, quiet=True)), 2); self.assertEqual(len(cp.weights_lines(rep)), 5)

    def test_missing_parameters_are_refused_by_name(self):
        os.remove(os.path.join(self.data, self.rels[4]))
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(reasons, [f"parameters under {self.data}: files missing: {self.rels[4]}"]); self.assertIs(det["ok"], False); self.assertIsNone(det["pinned"])
        self.assertEqual(det["files"][self.rels[4]], {"state": "missing"}); self.assertEqual(len(stack.weights_lines(det)), 4)   # the present four still carry their verdict
        os.remove(os.path.join(self.data, stack.pins()["weights"]["marker"]["file"]))
        reasons, det = stack.weights_check(self.data)
        self.assertEqual(len(reasons), 1); self.assertIn("marker " + os.path.join(self.data, stack.pins()["weights"]["marker"]["file"]) + " missing (colabfold would fetch from the network)", reasons[0])
        self.assertIn("files missing: " + self.rels[4], reasons[0])
        import io, contextlib
        from colabfold_opt import cli, report
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc, rep = cli.pred_gates("off", self.data)
        self.assertEqual(rc, report.EXIT_NOT_ACTIVE); self.assertIn("[colabfold-opt] NOT ACTIVE: parameters under " + self.data, err.getvalue())
        reasons, det = stack.weights_check(None)
        self.assertEqual(reasons, ["no data directory (--data or $COLABFOLD_OPT_DATA_DIR): the parameters were not checked"]); self.assertIsNone(det["ok"])


if __name__ == "__main__":
    unittest.main()
