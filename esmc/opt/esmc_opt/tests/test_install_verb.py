"""run.sh `install`: the install step outside users run — the kit package installed editable into the interpreter on PATH (esmc_opt imports
no other package of the bundle: opt/pyproject.toml), then the pin check, then the extension build; `--weights DIR --variant V` adds the weights fetch (upstream's downloader, then the digest check against stock/PINS.json). CPU
tests: run.sh runs against a stub `python` that records its arguments, and the digest gate of esmc_opt.weights runs on small files."""
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.normpath(os.path.join(HERE, "..", "..", ".."))            # esmc/
RUN_SH = os.path.join(KIT, "run.sh")
BUILD = "-m esmc_opt.kits.residual_ln.build"   # the extension build step (kernels.cu compiled once into the per-user cache; an already-built capability is left alone)

STUB = textwrap.dedent(r'''
    #!/bin/sh
    # stand-in interpreter: records each call, answers run.sh's install-step probes
    printf '%s\n' "$*" >> "$STUB_LOG"
    case "$*" in
      "-I -c import os,sys"*) exit ${STUB_PROBE_RC:-1} ;;
      "-m pip install"*) exit ${STUB_PIP_RC:-0} ;;
      *check_pins.py*) exit ${STUB_PINS_RC:-0} ;;
      "-m esmc_opt.kits.residual_ln.build"*) exit ${STUB_BUILD_RC:-0} ;;
      "-m esmc_opt.weights"*) exit ${STUB_WEIGHTS_RC:-0} ;;
    esac
    exit 0
''').lstrip()


class InstallVerbArguments(unittest.TestCase):
    """run.sh install parses its own arguments (usage errors exit 2 before anything runs) and runs pip → pin check → extension build → weights in that order."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); d = self.tmp.name
        self.bin = os.path.join(d, "bin"); os.makedirs(self.bin)
        stub = os.path.join(self.bin, "python")
        with open(stub, "w") as f: f.write(STUB)
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR)
        self.log = os.path.join(d, "calls.log")

    def tearDown(self):
        self.tmp.cleanup()

    def run_sh(self, *args, **rc):
        env = {"PATH": self.bin + os.pathsep + "/usr/bin" + os.pathsep + "/bin", "STUB_LOG": self.log, "HOME": self.tmp.name}
        env.update({f"STUB_{k.upper()}_RC": str(v) for k, v in rc.items()})
        p = subprocess.run(["bash", RUN_SH, *args], env=env, capture_output=True, text=True)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return p.returncode, p.stdout + p.stderr, [c for c in calls if not c.startswith("-I -c ")]   # the already-installed probe is not a step

    def test_install_runs_pip_then_pin_check_then_the_extension_build(self):
        rc, out, calls = self.run_sh("install")
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {KIT}/opt", f"-I {KIT}/stock/check_pins.py", BUILD])

    def test_weights_dir_and_variant_add_the_weights_step_last(self):
        for args in (("install", "--weights", "/data/esmc", "--variant", "6b"), ("install", "--weights=/data/esmc", "--variant=6b"),
                     ("--variant", "6b", "install", "--weights", "/data/esmc")):          # --variant is run.sh's global flag: any position
            os.path.exists(self.log) and os.remove(self.log)
            rc, out, calls = self.run_sh(*args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[-1], "-m esmc_opt.weights /data/esmc 6b", args); self.assertEqual(len(calls), 4, calls); self.assertEqual(calls[2], BUILD)

    def test_usage_errors_exit_2_and_call_nothing(self):
        for args in (("install", "--weights"), ("install", "--weights="), ("install", "--bogus"), ("install", "extra"),
                     ("install", "--mode", "exact"), ("install", "--variant", "6b"), ("install", "--weights", "/d"),
                     ("install", "--weights", "/d", "--variant", "7b"), ("install", "--weights", "--variant", "6b")):
            os.path.exists(self.log) and os.remove(self.log)
            rc, out, calls = self.run_sh(*args)
            self.assertEqual(rc, 2, (args, out)); self.assertEqual(calls, [], args); self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh("install", pip=1)
        self.assertEqual(rc, 1); self.assertEqual(len(calls), 1); self.assertIn("the install failed", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh("install", "--weights", "/d", "--variant", "6b", pins=3)
        self.assertEqual(rc, 3); self.assertEqual(len(calls), 2, calls); self.assertIn("refused by the pin check", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh("install", "--weights", "/d", "--variant", "6b", build=1)
        self.assertEqual(rc, 1); self.assertEqual(len(calls), 3, calls); self.assertIn("extension could not be built", out)   # the weights step does not run
        os.remove(self.log)
        rc, out, calls = self.run_sh("install", "--weights", "/d", "--variant", "6b", weights=1)
        self.assertEqual(rc, 1); self.assertEqual(len(calls), 4, calls)

    def test_an_already_installed_tree_skips_pip_by_name(self):
        """In the container image the kit is installed from /kit already: install says so and runs the pin check only (a read-only image
        cannot re-run pip); --weights still follows."""
        rc, out, calls = self.run_sh("install", probe=0)
        self.assertEqual(rc, 0, out); self.assertEqual(calls, [f"-I {KIT}/stock/check_pins.py", BUILD]); self.assertIn("installed from this tree already", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh("install", "--weights", "/d", "--variant", "300m", probe=0)
        self.assertEqual(rc, 0, out); self.assertEqual(calls, [f"-I {KIT}/stock/check_pins.py", BUILD, "-m esmc_opt.weights /d 300m"])


class WeightsDigestGate(unittest.TestCase):
    """esmc_opt.weights.fetch: files present are only checked; a missing file goes through the (injected) upstream route; a digest mismatch or a
    moved default revision is named and fails with the files left in place; the summary line counts n/n."""

    def setUp(self):
        sys.path.insert(0, os.path.join(KIT, "opt"))
        import esmc_opt.weights as W
        self.W = W
        self.tmp = tempfile.TemporaryDirectory(); self.root = self.tmp.name
        import hashlib
        self.payload = {"300m": {"model.safetensors": b"small-weights"}, "6b": {f"model-0000{i}-of-00006.safetensors": b"shard %d" % i for i in range(1, 7)}}
        self.repos = {"300m": "org/M-300M", "6b": "org/M-6B"}; self.commits = {"300m": "a" * 40, "6b": "b" * 40}
        self.pins = {"variants": {v: {"hf_repo": r} for v, r in self.repos.items()},
                     "weights": {self.repos[v]: {"snapshot_commit": self.commits[v], "weights_files": sorted(files),
                                                 "total_weights_bytes": sum(len(b) for b in files.values()),
                                                 "files": {f: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for f, b in files.items()}}
                                 for v, files in self.payload.items()}}
        import importlib.util
        spec = importlib.util.spec_from_file_location("esmc_check_pins_t", os.path.join(KIT, "stock", "check_pins.py"))
        self.cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.cp)     # the real digest routine (stock/check_pins.py weights)

    def tearDown(self):
        self.tmp.cleanup(); sys.path.remove(os.path.join(KIT, "opt"))

    def snap(self, v):
        return self.W.snapshot_path(self.root, self.repos[v], self.commits[v])

    def write(self, v, name, data=None):
        p = os.path.join(self.snap(v), name); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(self.payload[v][name] if data is None else data)

    def route_writing(self, v, calls, data=None):
        def route(repo):
            calls.append(repo)
            for name in self.payload[v]: os.path.exists(os.path.join(self.snap(v), name)) or self.write(v, name, data)
            return self.snap(v)
        return route

    def run_fetch(self, v, route):
        out = io.StringIO(); rc = self.W.fetch(self.root, v, pins=self.pins, route=route, checker=self.cp.weights, out=out)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        calls = []
        rc, out = self.run_fetch("6b", self.route_writing("6b", calls))
        self.assertEqual(rc, 0, out); self.assertEqual(calls, [self.repos["6b"]])
        self.assertRegex(out, r"WEIGHTS OK: 6/6 weight files of org/M-6B @ bbbbbbbbbbbb under .* — export HF_HOME=" + re.escape(os.path.join(self.root, "hf")))
        self.assertIn("0/6 weight files present", out)

    def test_present_files_are_checked_not_fetched(self):
        for name in self.payload["300m"]: self.write("300m", name)
        rc, out = self.run_fetch("300m", route=lambda repo: self.fail("no fetch expected"))
        self.assertEqual(rc, 0, out); self.assertIn("1/1 weight files present", out); self.assertIn("WEIGHTS OK: 1/1", out)

    def test_mismatch_is_named_and_fails_with_the_file_left_in_place(self):
        rc, out = self.run_fetch("300m", self.route_writing("300m", [], data=b"corrupt"))
        self.assertEqual(rc, 1, out)
        self.assertRegex(out, r"org/M-300M/model\.safetensors: sha256 [0-9a-f]{16}… != pinned [0-9a-f]{16}…")
        self.assertIn("REFUSED: 1 finding(s) on the 1 weight files", out); self.assertNotIn("WEIGHTS OK", out)
        self.assertTrue(os.path.exists(os.path.join(self.snap("300m"), "model.safetensors")))     # never deleted

    def test_a_failed_transfer_names_the_file_and_fails(self):
        def route(repo): raise OSError("connection reset")
        rc, out = self.run_fetch("300m", route)
        self.assertEqual(rc, 1); self.assertIn("FAILED fetching org/M-300M: OSError: connection reset", out); self.assertNotIn("WEIGHTS OK", out)

    def test_a_moved_default_revision_is_named_and_fails(self):
        other = os.path.join(os.path.dirname(self.snap("300m")), "c" * 40)
        def route(repo): os.makedirs(other, exist_ok=True); return other
        rc, out = self.run_fetch("300m", route)
        self.assertEqual(rc, 1, out); self.assertIn("default revision resolved to " + "c" * 40 + ", not the pinned snapshot " + "a" * 40, out)
        self.assertNotIn("WEIGHTS OK", out)

    def test_an_unknown_variant_is_a_usage_error(self):
        rc, out = self.run_fetch("7b", route=lambda repo: self.fail("no fetch expected"))
        self.assertEqual(rc, 2); self.assertIn("'7b' is not a variant (300m|6b: stock/PINS.json variants)", out)

    def test_the_layout_is_check_pins_weights_root(self):
        """install --weights DIR and check_pins.py --weights DIR read one layout: DIR/hf = HF_HOME, snapshots under DIR/hf/hub/models--…/snapshots/<commit>/."""
        self.assertEqual(self.W.hf_home_of(self.root), os.path.join(self.root, "hf"))
        self.assertEqual(self.snap("6b"), os.path.join(self.root, "hf", "hub", "models--org--M-6B", "snapshots", "b" * 40))
        pins = json.load(open(os.path.join(KIT, "stock", "PINS.json"), encoding="utf-8"))
        self.assertIn("<root>/hf/hub/", pins["weights_layout"]["manifest"]); self.assertTrue(pins["weights_layout"]["form"].startswith("HF cache: hf/hub/models--"))

    def test_pins_repos_are_upstreams_constants(self):
        """The repos the downloader is pointed at (stock/PINS.json variants) are upstream's own constants, read from the stock tree as text."""
        pins = json.load(open(os.path.join(KIT, "stock", "PINS.json"), encoding="utf-8"))
        cfg = open(os.path.join(KIT, "stock", "src", "esm", "esm", "models", "esmc", "config.py"), encoding="utf-8").read()
        names = open(os.path.join(KIT, "stock", "src", "esm", "esm", "utils", "constants", "models.py"), encoding="utf-8").read()
        for v, V in pins["variants"].items():
            self.assertRegex(cfg, rf'(?m)^ESMC_{v.upper()}_HF_REPO = "{re.escape(V["hf_repo"])}"', v)
            self.assertRegex(names, rf'(?m)^ESMC_{v.upper()} = "{re.escape(V["model_name"])}"', v)
            self.assertIn(V["hf_repo"], pins["weights"])

    def test_upstream_route_sets_hf_home_and_calls_upstreams_resolver(self):
        """upstream_route binds HF_HOME first, refuses a cache that resolves elsewhere, and reaches upstream's resolve_model_dir with the repo id
        only (upstream's defaults) — checked on stand-in modules, so no URL, revision or file pattern is restated in the kit."""
        rec = []
        hub = types.ModuleType("esm.models.hub"); hub.resolve_model_dir = lambda repo_id, **kw: rec.append((repo_id, kw)) or f"/snap/{repo_id}"
        hf = types.ModuleType("huggingface_hub"); const = types.ModuleType("huggingface_hub.constants"); hf.constants = const
        hf_home = os.path.join(self.root, "hf"); const.HF_HUB_CACHE = os.path.join(hf_home, "hub")
        mods = {"huggingface_hub": hf, "huggingface_hub.constants": const, "esm": types.ModuleType("esm"), "esm.models": types.ModuleType("esm.models"), "esm.models.hub": hub}
        saved = {k: sys.modules.get(k) for k in mods}; saved_env = os.environ.get("HF_HOME")
        sys.modules.update(mods)
        try:
            route = self.W.upstream_route(hf_home)
            self.assertEqual(os.environ["HF_HOME"], hf_home)
            self.assertEqual(route("org/M-6B"), "/snap/org/M-6B"); self.assertEqual(rec, [("org/M-6B", {})])
            const.HF_HUB_CACHE = os.path.join(self.root, "elsewhere")
            with self.assertRaisesRegex(RuntimeError, "resolves its cache to .*elsewhere"): self.W.upstream_route(hf_home)
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
            if saved_env is None: os.environ.pop("HF_HOME", None)
            else: os.environ["HF_HOME"] = saved_env


class EnvironmentIsThePinnedStack(unittest.TestCase):
    """environment/ is the one definition of the tested stack: the lock restates stock/PINS.json's pins (torch sans its local tag, the upstream
    commits as pip's VCS form, the two accelerator wheels by digest), the Dockerfile builds that Python / CUDA and installs the lock and the kit,
    apptainer.def converts the same image."""

    def setUp(self):
        self.pins = json.load(open(os.path.join(KIT, "stock", "PINS.json"), encoding="utf-8"))
        self.lock = [l.strip() for l in open(os.path.join(KIT, "environment", "requirements.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]

    def pin(self, name):
        """The lock's one line for distribution `name` (names compared in their normalised form)."""
        hits = [l for l in self.lock if re.split(r"[ =@]", l, 1)[0].lower().replace("-", "_") == name]
        self.assertEqual(len(hits), 1, (name, hits)); return hits[0]

    def version(self, name):
        line = self.pin(name); self.assertIn("==", line, name); return line.split("==", 1)[1]

    def test_lock_restates_the_pins(self):
        P, A, U = self.pins["pins"], self.pins["stacks"]["accel"], self.pins["upstream"]
        self.assertEqual(self.version("torch"), P["torch"]); self.assertEqual(P["torch_runtime"], f"{P['torch']}+cu{P['cuda'].replace('.', '')}")
        for k in ("triton", "huggingface_hub", "safetensors", "tokenizers", "einops", "nvidia_cudnn_cu13", "nvidia_cuda_runtime", "nvidia_cuda_nvrtc"):
            self.assertEqual(self.version(k), P[k], k)
        for k in ("nvidia_cublas", "numpy", "transformer_engine"):
            self.assertEqual(self.version(k), A[k], k)
        for name in ("esm", "transformers"):                                        # pip's VCS pin form of the upstream commits
            self.assertEqual(self.pin(name), f"{name} @ git+{U[name]['repo']}.git@{U[name]['commit']}", name)
        for name, key in (("flash_attn", "flash_attn_wheel"), ("transformer_engine_torch", "transformer_engine_torch_wheel")):   # the accelerator wheels by file name + digest
            line = self.pin(name); w = A[key]
            self.assertRegex(line, rf"^{name} @ https://\S+/{re.escape(w['file'])}#sha256={w['sha256']}$", name)
        self.assertEqual([l for l in self.lock if "xformers" in l.lower()], []); self.assertEqual([l for l in self.lock if l.startswith(("pip==", "pip @", "esmc_opt", "esmc-opt"))], [])
        self.assertEqual([l for l in self.lock if "file://" in l], [])                # every line installs anywhere

    def test_dockerfile_builds_the_pinned_stack(self):
        P = self.pins["pins"]
        df = open(os.path.join(KIT, "environment", "Dockerfile"), encoding="utf-8").read()
        m = re.search(r"^FROM (\S+)$", df, re.M); self.assertIsNotNone(m)
        self.assertTrue(m.group(1).startswith(f"nvidia/cuda:{P['cuda']}.") and "-runtime-ubuntu24.04" in m.group(1), m.group(1))
        self.assertIn(f"cpython-{self.pins['python']}+", df)
        self.assertIn("COPY esmc/environment/requirements.lock ", df); self.assertIn("COPY esmc /kit/esmc", df); self.assertIn("RUN bash run.sh install", df)
        self.assertNotRegex(df, r"(?m)^COPY (?!esmc[ /])")                          # the image carries this kit's tree only
        self.assertRegex(open(os.path.join(KIT, "environment", "apptainer.def"), encoding="utf-8").read(), r"(?m)^Bootstrap: docker-daemon\nFrom: esmc-kit:dev$")


if __name__ == "__main__":
    unittest.main()
