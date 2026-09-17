"""The carried files and the pins: every carried kit/add-on directory is in the tree with its launch files (byte identity is the git
commit's, never re-hashed here); the touched_modules stock/src carries are present; the vendored wheel is present at its pinned size;
stock/pip_freeze.txt carries the pinned stack's deviation; the stock caller's seed line is the kit's own; the .pth is the build
backend's generated guard for this package; the build backend is the tree's one copy; the configs export no lever switch."""
import os
import re
import unittest

from boltzgen_opt import codes, modes, report, stack, stock_design

HOME = stack.opt_home()
TREE = stack.tree_home()


class TestTreeAndPins(unittest.TestCase):
    def test_carried_kit_dirs_are_present_with_their_launch_files(self):
        """The five carried kit directories (modes.KITS) are in the tree; the two vendor kits' launch files (stack.KIT_LAUNCH_FILES: the runner;
        the partner's sitecustomize / in-process pipeline / hook / graph sampler) are present — a presence check by name."""
        self.assertEqual(len(modes.KITS), 5)
        for rel in modes.KITS:
            self.assertTrue(os.path.isdir(stack.kit_dir(rel)), rel)
            self.assertTrue(os.path.isdir(stack.kit_src(rel)), rel)
            self.assertEqual(stack.kit_missing(rel), [], rel)
        self.assertEqual(set(stack.KIT_LAUNCH_FILES), {modes.KIT_XATTEMPT, modes.KIT_PARTNER})
        self.assertEqual(stack.KIT_LAUNCH_FILES[modes.KIT_XATTEMPT], (modes.RUNNER,))
        self.assertEqual(stack.KIT_LAUNCH_FILES[modes.KIT_PARTNER], (modes.SITECUSTOMIZE, os.path.join("src", "bg_inproc.py"), os.path.join("src", "bg_hook.py"), os.path.join("src", "bg_graph_patch.py")))
        self.assertFalse(os.path.exists(os.path.join(HOME, "serving")))              # no worker kit, no packed form
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, "forward"))), ["fast_inference", "fast_levers", "size_levers", "xattempt_addon"])
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, "host"))), ["writer_levers"])

    def test_carried_kit_dirs_hold_sources_a_readme_and_the_example_input(self):
        """Each vendor kit directory is its `src/` and a README; the partner's also carries the example design spec the kit README's
        commands and `warm` use (`tests/specs/`: the spec, its target PDB and that file's attribution NOTICE); the house lever directories are
        `src/` alone."""
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, modes.KIT_XATTEMPT))), ["README.md", "src"])
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, modes.KIT_PARTNER))), ["README.md", "src", "tests"])
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, modes.KIT_PARTNER, "tests"))), ["specs"])
        self.assertEqual(sorted(os.listdir(os.path.join(HOME, modes.KIT_PARTNER, "tests", "specs"))), ["NOTICE", "PDL1_IgV_Q9NZQ7_18-132.pdb", "pdl1_ref.yaml"])
        for rel in (modes.KIT_SIZE_LEVERS, modes.KIT_FAST_LEVERS, modes.KIT_HOST_LEVERS):
            self.assertEqual(sorted(os.listdir(os.path.join(HOME, rel))), ["src"], rel)

    def test_stock_wheel_is_present_at_the_pinned_size(self):
        """Our vendored wheel is present at the pinned size (its bytes are the git commit's, never re-hashed here)."""
        st = os.path.join(TREE, "stock")
        pins = stack.pins()
        wheel_path = os.path.join(st, os.path.basename(pins["wheel"]["file"]))
        self.assertTrue(os.path.isfile(wheel_path), wheel_path)
        self.assertEqual(pins["wheel"]["bytes"], os.path.getsize(wheel_path))
        self.assertIn(pins["boltzgen_version"], os.path.basename(wheel_path))

    def test_touched_modules_are_present_in_stock_src(self):
        """touched_modules (stock/PINS.json) names every upstream module our levers patch or read; each is present (the bytes are the
        git commit's, never re-hashed here)."""
        st = os.path.join(TREE, "stock")
        pins = stack.pins()
        for rel in pins["touched_modules"]:
            self.assertTrue(os.path.isfile(os.path.join(st, "src", rel)), rel)
        self.assertTrue({"boltzgen/model/models/boltz.py", "boltzgen/task/predict/predict.py", "boltzgen/model/modules/diffusion.py",
                         "boltzgen/model/modules/encoders.py", "boltzgen/model/modules/transformers.py", "boltzgen/task/predict/writer.py"} <= set(pins["touched_modules"]))

    def test_weights_pins_name_six_files_with_digests(self):
        pins = stack.pins()
        self.assertEqual(len(pins["weights"]["files"]), 6)
        for row in pins["weights"]["files"]:
            self.assertRegex(row["sha256"], r"^[0-9a-f]{64}$", row["file"])

    def test_environment_lock_is_the_pinned_stack(self):
        """environment/requirements.lock is the stack's one package list (the image and README route C install it): its boltzgen, triton, numpy,
        numba, pytorch-lightning and cuEquivariance CUDA-13 pair are stock/PINS.json stack_of_record's (torch: test_lock_carries_the_pinned_deviation)."""
        pins = stack.pins(); sor = pins["stack_of_record"]
        lock = {}
        for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                k, v = line.split("==", 1); lock[k.lower().replace("_", "-")] = v
        self.assertNotIn("torch", lock); self.assertTrue(sor["torch"].endswith("+cu" + sor["cuda"].replace(".", "")))   # torch: the URL line (test_lock_carries_the_pinned_deviation)
        self.assertEqual(lock["boltzgen"], pins["boltzgen_version"])
        for k in ("numpy", "numba"):
            self.assertEqual(lock[k], sor[k], k)
        urls = [l.strip() for l in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8") if " @ " in l and not l.startswith("#")]
        self.assertEqual(sorted(u.split(" @ ")[0] for u in urls), ["torch", "triton"])          # the PyTorch index's two wheels, by URL + sha256
        tri = [u for u in urls if u.startswith("triton @ ")][0]
        self.assertRegex(tri, r"^triton @ https://download\.pytorch\.org/whl/triton-%s-cp311-cp311-\S+x86_64\.whl#sha256=[0-9a-f]{64}$" % re.escape(sor["triton"]))
        self.assertEqual(lock["pytorch-lightning"], sor["pytorch_lightning"])
        for k in ("cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu13", "cuequivariance-ops-cu13"):
            self.assertEqual(lock[k], sor["cuequivariance"].split(" ", 1)[0], k)          # `0.11.1 (…)`: the version, then the words naming the CUDA-13 pair
        self.assertEqual(set(sor["freeze_deviation"]["added"]) - set(l.strip() for l in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8")), set())
        self.assertEqual(sor["python"], "3.11")

    def test_environment_dockerfile_builds_the_pinned_stack(self):
        """environment/Dockerfile's base image, interpreter and install steps are the pinned stack's: the CUDA base tag carries PINS stack_of_record
        `cuda`, the python-build-standalone tarball PINS `python`, the lock and the stock wheel are the files it installs, `run.sh install` is its
        kit step; environment/apptainer.def converts that image."""
        pins = stack.pins(); sor = pins["stack_of_record"]
        df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read()
        m = re.search(r"^FROM (\S+)$", df, re.M); self.assertIsNotNone(m)
        self.assertTrue(m.group(1).startswith(f"nvidia/cuda:{sor['cuda']}.") and "-base-ubuntu22.04" in m.group(1), m.group(1))
        self.assertRegex(df, r"cpython-%s\.\d+\+" % re.escape(sor["python"]))
        self.assertIn("COPY boltzgen/environment/requirements.lock ", df); self.assertIn(f"COPY boltzgen/{pins['wheel']['file']} ", df)
        self.assertNotIn("index-url", df)                                     # the index choice lives in the lock, per line, not in the build
        self.assertIn("pip install --no-cache-dir --no-deps -r /tmp/stack.txt", df)
        self.assertIn("RUN bash run.sh install", df)
        self.assertNotRegex(df, r"(?m)^ENV .*(BOLTZGEN_CACHE|MODEL_OPT_JIT_ROOT|NUM_THREADS)")      # no weights / cache path and no thread-pool size is baked
        self.assertRegex(open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read(), r"(?m)^From: boltzgen-kit:")

    def test_lock_carries_the_pinned_deviation(self):
        """environment/requirements.lock (stock/PINS.json stack_of_record.freeze) = the kits' 103-line freeze with stack_of_record.freeze_deviation
        applied — the CUDA-12 cuEquivariance ops pair and its two CUDA-12 library wheels out, the CUDA-13 pair of the same version in — plus the
        interpreter's pip, setuptools and wheel; torch is the one line named by URL (the PyTorch index's CUDA 13.0 wheel, sha256-pinned)."""
        sor = stack.pins()["stack_of_record"]
        self.assertEqual(sor["freeze"], "environment/requirements.lock")
        lines = [l.strip() for l in open(os.path.join(TREE, sor["freeze"]), encoding="utf-8") if l.strip() and not l.startswith("#")]
        self.assertFalse([l for l in lines if l.startswith("-")], "no pip option lines: the index choice is per line, by URL")
        dev = sor["freeze_deviation"]
        removed, added = list(dev["removed"]), list(dev["added"])
        tooling = [l for l in lines if l.split("==")[0] in ("pip", "setuptools", "wheel")]
        self.assertEqual(len(tooling), 3)
        self.assertEqual(len(lines) - len(tooling), 103 - len(removed) + len(added))
        self.assertTrue(set(added) <= set(lines) and not set(removed) & set(lines))
        self.assertTrue(all("-cu12" not in l for l in lines))                  # one CUDA major in the pinned stack: torch's 13
        torch = [l for l in lines if re.match(r"torch( @ |==)", l)]
        self.assertEqual(len(torch), 1)
        m = re.match(r"torch @ https://download\.pytorch\.org/whl/cu(\d+)/torch-([0-9.]+)%2Bcu\1-cp311-cp311-manylinux_2_28_x86_64\.whl#sha256=[0-9a-f]{64}$", torch[0])
        self.assertIsNotNone(m, torch[0])
        self.assertEqual(f"{m.group(2)}+cu{m.group(1)}", sor["torch"]); self.assertEqual(m.group(1), sor["cuda"].replace(".", ""))

    def test_seed_line_is_the_kits_own(self):
        """The stock caller's SEED_LINE is the partner runner's own seeding statements (bg_inproc.py l.35-36, per step with `step_seed`):
        one recipe on every arm."""
        self.assertEqual(stock_design.SEED_LINE, "pl.seed_everything(seed, workers=True); random.seed(seed); np.random.seed(seed % (2**32)); torch.manual_seed(seed)")
        inproc = open(os.path.join(HOME, modes.KIT_PARTNER, "src", "bg_inproc.py"), encoding="utf-8").read()
        for stmt in stock_design.SEED_LINE.split("; "):
            self.assertIn(stmt.replace("(seed", "(step_seed"), inproc, stmt)         # the same statements, the runner's per-step name in the argument slot
        self.assertIn("pl.seed_everything(step_seed, workers=True)", inproc)
        self.assertIn("random.seed(step_seed); np.random.seed(step_seed % (2**32)); torch.manual_seed(step_seed)", inproc)

    def test_pth_and_backend(self):
        pth = open(os.path.join(HOME, "boltzgen_opt_autoload.pth"), encoding="utf-8").read().splitlines()
        self.assertEqual(pth[0], f"# opt_core autoload guard: package=boltzgen_opt env={modes.ENV} tag={report.TAG} exit={codes.EXIT_NOT_ACTIVE}")   # the generated header; the whole text == pth_text(): test_core_adoption
        self.assertEqual(len(pth), 2)
        self.assertIn("import boltzgen_opt._autoload", pth[1])
        backend = os.path.join(HOME, "_build_backend.py")
        self.assertTrue(os.path.isfile(backend))
        pyproject = open(os.path.join(HOME, "pyproject.toml"), encoding="utf-8").read()
        self.assertIn('classifiers = ["Private :: Do Not Upload"]', pyproject)
        self.assertIn("dependencies = []", pyproject)
        self.assertIn('boltzgen-opt = "boltzgen_opt.__main__:main"', pyproject)

    def test_no_lever_switches_in_configs(self):
        names = sorted(os.listdir(os.path.join(TREE, "configs")))
        self.assertEqual(names, ["a100.env", "h100.env", "h200.env"])                                            # one file per card; a100.env sets its GPU word and sources h100.env
        for name in names:
            cfg = open(os.path.join(TREE, "configs", name), encoding="utf-8").read()
            exports = "\n".join(ln for ln in cfg.splitlines() if not ln.lstrip().startswith("#"))   # the lines that set something (comments describe the routes)
            for bad in ("BG_GRAPH", "XA_FAST_INIT", "XA_HOIST", "SZ_TD_CHUNK", "FL_COND_DEDUP", "HL_ASYNC_WRITER", "BOLTZGEN_OPT=", "PYTHONPATH", "BOLTZGEN_OPT_FORCE", "PACK_K"):
                self.assertNotIn(bad, exports, f"{name}: {bad}")
            self.assertNotRegex(exports, r"(?m)^export (BG_|XA_|SZ_|FL_|HL_)", f"{name}: a lever switch exported by the config")
            self.assertNotIn("PACK_K", cfg, name); self.assertNotIn("CUDA_MPS", cfg, name)                    # no packing: the config carries no MPS setting

    def test_python_floor_is_the_upstream_wheels(self):
        pyproject = open(os.path.join(HOME, "pyproject.toml"), encoding="utf-8").read()
        m = re.search(r'requires-python = "([^"]+)"', pyproject)
        self.assertEqual(m.group(1), stack.pins()["wheel_requires"]["python"])


if __name__ == "__main__":
    unittest.main()
