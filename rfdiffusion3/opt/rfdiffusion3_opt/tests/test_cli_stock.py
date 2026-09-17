"""The command line and the stock caller, end to end on fake interpreters with a stub `rfd3.cli:app` (no torch):
`design --mode off` proves the pristine interpreter and runs upstream's entry point with the spelled line; `design --mode exact`
activates, runs the same entry point in-process with the delta exported, writes the manifest and the exit tally; `check` exit codes;
`warm` without a checkpoint is refused by name; the run parameters pass through to the literal line; upstream's own switches pass
through the stock arm and are reported on its STOCK line; the removed flags are usage errors."""
import gzip
import json
import os
import tempfile
import unittest

from .. import cli, manifest, modes, registry, settings, stock_design
from . import _fixtures as fx


def _design(d, base, cif: bytes, meta: dict, mtime: int):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, base + ".cif.gz"), "wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=mtime) as fh:
            fh.write(cif)
    json.dump(meta, open(os.path.join(d, base + ".json"), "w"))


class TestRunParameters(unittest.TestCase):
    def test_the_line_restates_no_stock_default(self):
        """Upstream ships its own diffusion_batch_size and compile_model=False (rfd3/configs/inference_engine/rfdiffusion3.yaml in the stock
        archive); the package restates neither: the composed line adds ckpt_path= and nothing else, and settings carries no batch default."""
        import tarfile
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            y = tf.extractfile(fx.ARCHIVE_PREFIX + "models/rfd3/configs/inference_engine/rfdiffusion3.yaml").read().decode()
        vals = {}
        for line in y.splitlines():
            t = line.split("#")[0].strip()
            if ":" in t and not t.endswith(":"):
                k, v = t.split(":", 1)
                vals[k.strip()] = v.strip()
        self.assertTrue(vals["diffusion_batch_size"].isdigit(), vals["diffusion_batch_size"])   # upstream's own number, whatever it is: no token = that
        self.assertEqual(vals["compile_model"].lower(), "false")
        self.assertEqual(settings.line(["inputs=/s.json", "out_dir=/o", "seed=1", "ckpt_path=/c.ckpt"]), ["rfd3", "design", "inputs=/s.json", "out_dir=/o", "seed=1", "ckpt_path=/c.ckpt"])   # the line is the caller's tokens, verbatim, in order
        self.assertEqual(settings.line(["inputs=A11-20,B5", "+x=1"]), ["rfd3", "design", "inputs=A11-20,B5", "+x=1"])                    # any value upstream's key takes, any hydra form: never parsed here
        for gone in ("compose", "as_args"):
            self.assertFalse(hasattr(settings, gone), gone)                                     # no kit composition of upstream's settings
        for gone in ("STOCK_DEFAULT_BATCH", "batch_of", "PRESETS"):
            self.assertFalse(hasattr(settings, gone), gone)


class TestDesignRoutes(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.make_site(os.path.join(self.td.name, "p"), "patched")
        self.pristine = fx.make_site(os.path.join(self.td.name, "s"), "pristine")
        fx.stub_cli(self.patched)
        fx.stub_cli(self.pristine)
        self.spec = fx.fake_spec(os.path.join(self.td.name, "public.json"))
        self.ckpt = os.path.join(self.td.name, "fake.ckpt")
        open(self.ckpt, "wb").write(b"not a checkpoint")
        self.stockpy = fx.make_venv(self.td.name, self.pristine)               # the pristine interpreter of the stock arm (isolated mode needs a real one)

    def tearDown(self):
        self.td.cleanup()

    def _design(self, site, mode, extra=(), env=None, out="out", python=None):
        out_dir = os.path.join(self.td.name, out)
        extra = list(extra); tail = extra[extra.index("--") + 1:] if "--" in extra else []; head = extra[:extra.index("--")] if "--" in extra else extra
        args = ["design", "--mode", mode, "--ckpt", self.ckpt] + head + [f"inputs={self.spec}", f"out_dir={out_dir}", "diffusion_batch_size=2"] + tail   # the line: upstream's own tokens, verbatim (the kit's flags may stand anywhere before a `--`)
        r = fx.run_cli(args, site, env=env, bin_dir=self.bin, python=python)
        return r, out_dir

    def test_off_route_on_the_pristine_interpreter(self):
        """No RFDIFFUSION3_STOCK_PYTHON: the stock arm runs in the calling interpreter (here the pristine venv itself) and proves it."""
        r, out = self._design(self.pristine, "off", env={"RFD3_HOIST": "1"}, python=self.stockpy)    # a lever in the caller's environment is stripped, never seen
        self.assertEqual(r.returncode, 0, r.stderr)
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertEqual(seen["argv"][:1], ["design"])
        self.assertEqual(seen["argv"][1:], [f"inputs={self.spec}", f"out_dir={out}", "diffusion_batch_size=2", f"ckpt_path={self.ckpt}"])   # the caller's tokens in order, then the one token the kit adds (--ckpt -> ckpt_path=) — nothing else
        self.assertIsNone(seen["RFD3_HOIST"])
        self.assertIsNone(seen["RFDIFFUSION3_OPT"])
        self.assertNotIn("rfdiffusion3_opt.stack", seen["modules"])            # nothing of the package's activation core in the stock process
        self.assertEqual(os.path.realpath(seen["python"]), os.path.realpath(self.stockpy))
        self.assertFalse(os.path.exists(os.path.join(out, "stock_env_proof.json")))               # the proof is the manifest's stock_env_proof, not a file of its own
        proof = manifest.read(out)["stock_env_proof"]
        self.assertTrue(proof["ok"], proof)
        self.assertTrue(proof["pins"]["pristine"])
        self.assertEqual(proof["after"]["exit_code"], 0)
        self.assertEqual((proof["det"], proof["upstream_env"], len(proof["forbidden_names_checked"])), ({"level": 0}, {}, 11))   # the kit's 10 must-be-absent names; upstream's own switches are not among them
        self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt stock\] STOCK rc-foundry=0\.2\.1\.dev13\+g4010e3e2e tree=pristine python=\S+ forbidden_absent=11 upstream_env=none det=0$")   # stock_design.stock_line, whole
        self.assertIn("design mode=off det=0 stock interpreter=", r.stderr)
        self.assertIn(f"--det 0 -- design inputs={self.spec} out_dir={out} diffusion_batch_size=2 ckpt_path={self.ckpt}", r.stderr)   # cli.stock_command: the level travels to the child before `--`
        self.assertNotIn("] PINS ", r.stderr)                                          # the fake distribution sits at the pin: nothing to report
        man = manifest.read(out)
        self.assertEqual(man["mode"], "off")
        self.assertEqual(man["interpreter"]["label"], "pristine")
        self.assertEqual(man["settings"]["values"], {"diffusion_batch_size": "2"})
        self.assertEqual(man["spec"], {"value": self.spec, "is_file": True, "path": os.path.abspath(self.spec), "bytes": os.path.getsize(self.spec)})   # manifest.spec_info: the value, and the file's facts when it is one
        cifs = [o for o in man["outputs"] if o["file"].endswith(".cif.gz")]
        self.assertEqual(sorted(cifs[0]), ["bytes", "file", "sha256"])                 # manifest.output_listing: name, size and digest per design
        self.assertEqual(cifs[0]["sha256"], registry.sha256_of(os.path.join(out, cifs[0]["file"])))
        self.assertEqual(man["checkpoint"]["sha256"], registry.sha256_of(self.ckpt))
        self.assertEqual(man["checkpoint"]["sha256_source"], "hashed")
        self.assertFalse(man["checkpoint"]["is_pinned_checkpoint"])
        self.assertEqual(len([o for o in man["outputs"] if o["file"].endswith(".cif.gz")]), 2)
        self.assertEqual(man["stock_env_proof"]["ok"], True)
        self.assertEqual({k: man[k] for k in manifest.LEVER_KEYS}, {"levers_applied": [], "levers_fallback": [], "levers_unavailable": [], "partial": False,
                                                                    "partial_reason": None, "allow_partial": False})
        r, out2 = self._design(self.pristine, "off", out="out2", python=self.stockpy)          # the same checkpoint again: the digest comes from the cache
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(manifest.read(out2)["checkpoint"]["sha256_source"], "cache")
        self.assertEqual(manifest.read(out2)["checkpoint"]["sha256"], man["checkpoint"]["sha256"])

    def test_removed_flags_are_usage_errors(self):
        """`design --attn`, `warm --timeout`, `warm --log` and `--version` are not options (nor is any numerics opt-in but `--det`): argparse
        refuses them (exit 2) before anything runs. The design parser's options are exactly the documented ones."""
        for extra in (["--attn", "sparse"], ["--tf32", "1"], ["--compile"], ["--seed", "7"], ["--no-ckpt-sha"], ["--batch-size", "8"]):   # no kit name for an upstream setting either: compile_model=, seed=, diffusion_batch_size= are upstream's own tokens
            for site, mode, py in ((self.pristine, "off", self.stockpy), (self.patched, "exact", None)):
                r, out = self._design(site, mode, extra=extra, out="gone", python=py)
                self.assertEqual(r.returncode, cli.EXIT_USAGE, r.stderr)
                self.assertIn(f"error: unrecognized arguments: {extra[0]}", r.stderr)
                self.assertFalse(os.path.isdir(out))                                    # nothing ran
        self.assertEqual(cli.DESIGN_FLAGS, ("--mode", "--ckpt", "--det", "--allow-partial"))          # the kit's whole design surface (--config is run.sh's: configs/<cfg>.env)
        for extra in (["--timeout", "5"], ["--log", os.path.join(self.td.name, "w.log")]):
            r = fx.run_cli(["warm", "--mode", "exact", "--ckpt", self.ckpt] + extra, self.patched, bin_dir=self.bin)
            self.assertEqual(r.returncode, cli.EXIT_USAGE, r.stderr); self.assertIn(f"unrecognized arguments: {extra[0]}", r.stderr)
        r = fx.run_cli(["--version"], self.patched)
        self.assertEqual(r.returncode, cli.EXIT_USAGE, r.stderr)
        import argparse
        subs = next(a for a in cli.build_parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
        opts = lambda verb: sorted(o for a in subs[verb]._actions for o in a.option_strings if o.startswith("--"))
        self.assertEqual(opts("design"), ["--allow-partial", "--ckpt", "--det", "--help", "--mode"])   # (--config is run.sh's: configs/<cfg>.env)
        self.assertEqual(opts("warm"), ["--allow-partial", "--ckpt", "--help", "--json", "--mode"])
        for mod, gone in ((stock_design, ("GUARDS", "DEFAULT_GUARD", "GUARD_VARIABLES")), (modes, ("ATTN_LEVELS", "DEFAULT_ATTN", "ATTN_ENV", "ATTN_UNDER_KIT_MODE", "REFUSED_OVERRIDES", "refused_overrides", "readme_check"))):
            for name in gone:
                self.assertFalse(hasattr(mod, name), name)
        self.assertEqual(set(modes.ROUTES), {"stock", "default", "exact", "fast"})   # no sparse stock route: upstream's switch is upstream's, passed through (below)

    def test_upstreams_own_switches_pass_through_the_stock_arm(self):
        """`--mode off` with upstream's own switches in the caller's environment (RFD3_LOW_MEMORY_MODE, RFD3_DENSE_SDPA_ATTENTION): the strip
        rule removes the KIT's names only, the child sees upstream's switches as set, its proof passes and records them (`upstream_env`), the
        STOCK line names them; torch's variables likewise reach the child unchanged. The KERNELS census on the stock routes reports, never
        refuses: the run's exit code is upstream's own (0)."""
        env = {"RFD3_HOIST": "1", "RFD3_LOW_MEMORY_MODE": "1", "RFD3_DENSE_SDPA_ATTENTION": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "FOUNDRY_DET_SCATTER": "1"}
        for extra, route in (([], "default"), (["compile_model=true"], "stock")):                     # upstream's own compile switch names the compiled-stock route
            r, out = self._design(self.pristine, "off", extra=extra, env=env, python=self.stockpy, out=f"ups_{route}")
            self.assertEqual(r.returncode, 0, r.stderr)
            seen = json.load(open(os.path.join(out, "cli_seen.json")))                    # upstream ran ...
            self.assertEqual((seen["RFD3_HOIST"], seen["RFD3_DENSE_SDPA_ATTENTION"], seen["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]), (None, "0", "1"))   # ... the kit lever stripped, upstream's switch and torch's variable as the caller set them
            self.assertEqual("compile_model=true" in seen["argv"], bool(extra))
            proof = manifest.read(out)["stock_env_proof"]
            self.assertTrue(proof["ok"], proof["reasons"])
            self.assertEqual((proof["forbidden_present"], proof["upstream_env"]), ([], {"RFD3_DENSE_SDPA_ATTENTION": "0", "RFD3_LOW_MEMORY_MODE": "1"}))
            self.assertNotIn("RFD3_LOW_MEMORY_MODE", proof["forbidden_names_checked"]); self.assertIn("FOUNDRY_DET_SCATTER", proof["forbidden_names_checked"])   # the det names stay must-be-absent: an inherited copy is stripped, the child sets its own at --det 1
            self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt stock\] STOCK rc-foundry=\S+ tree=pristine python=\S+ forbidden_absent=11 upstream_env=RFD3_DENSE_SDPA_ATTENTION=0,RFD3_LOW_MEMORY_MODE=1 det=0$")
            self.assertNotIn("NOT ACTIVE", r.stderr); self.assertNotIn("NOT STOCK", r.stderr)
            klines = [l for l in r.stderr.splitlines() if "KERNELS route=" in l]
            self.assertEqual(len(klines), 1, r.stderr); self.assertIn(f"KERNELS route={route} mode=off ", klines[0])
            man = manifest.read(out)
            self.assertEqual((man["mode"], man["env_delta"], man["exit_code"], man["kernels"]["refused"]), ("off", {}, 0, None))
        self.assertEqual(cli.stock_env({"PATH": "/bin", "RFD3_DENSE_SDPA_ATTENTION": "0", "RFD3_HOIST": "1", "RFDIFFUSION3_OPT_DET": "1", "CUBLAS_WORKSPACE_CONFIG": ":16:8", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}),
                         {"PATH": "/bin", "RFD3_DENSE_SDPA_ATTENTION": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1", "MODEL_OPT": registry.tree_home()})   # stack.strip_env: the package switches and the kit's names go, the rest is the caller's
        self.assertEqual(fx.stack.strip_env({"RFD3_LOW_MEMORY_MODE": "1", "RFD3_HOIST": "1", "RFDIFFUSION3_OPT": "exact", "HOME": "/h"}), {"RFD3_LOW_MEMORY_MODE": "1", "HOME": "/h"})
        p = stock_design.env_proof(environ={"PATH": "/usr/bin", "RFD3_DENSE_SDPA_ATTENTION": "0", "RFD3_LOW_MEMORY_MODE": "1"})   # in-process (no rfd3 here: not ok for THAT reason, never for the switches)
        self.assertEqual((p["forbidden_present"], p["upstream_env"]), ([], {"RFD3_DENSE_SDPA_ATTENTION": "0", "RFD3_LOW_MEMORY_MODE": "1"}))
        self.assertEqual([x for x in p["reasons"] if "environment" in x], [])
        p = stock_design.env_proof(environ={"PATH": "/usr/bin", "FOUNDRY_DET_SCATTER": "1", "RFD3_TOKEN_SDPA": "1"})
        self.assertEqual(p["forbidden_present"], ["FOUNDRY_DET_SCATTER", "RFD3_TOKEN_SDPA"]); self.assertIn("forbidden environment names present: ['FOUNDRY_DET_SCATTER', 'RFD3_TOKEN_SDPA']", "; ".join(p["reasons"]))   # an inherited det name is forbidden like a lever name: the child sets its own at --det 1

    def test_the_checkpoint_token(self):
        """`--ckpt <file>` (or RFD3_CKPT) becomes upstream's `ckpt_path=<file>` appended after the caller's tokens (both arms); a line that
        names `ckpt_path=` itself keeps its token and gains none; the flag and the token naming different files is a usage error before
        anything runs; no checkpoint anywhere is a usage error naming the three ways to give one."""
        fx.fake_torch(self.patched)
        for site, mode, py in ((self.pristine, "off", self.stockpy), (self.patched, "exact", None)):
            r, out = self._design(site, mode, out=f"ck_{mode}", python=py, env={"RFD3_FAKE_IMPORT_HOIST": "1"})
            self.assertEqual(r.returncode, 0, r.stderr)
            argv = json.load(open(os.path.join(out, "cli_seen.json")))["argv"]
            self.assertEqual([t for t in argv if t.startswith("ckpt_path=")], [f"ckpt_path={self.ckpt}"]); self.assertEqual(argv[-1], f"ckpt_path={self.ckpt}")
            r, out = self._design(site, mode, extra=[f"ckpt_path={self.ckpt}"], out=f"ck_tok_{mode}", python=py, env={"RFD3_FAKE_IMPORT_HOIST": "1"})   # the line's own token (and --ckpt agreeing)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual([t for t in json.load(open(os.path.join(out, "cli_seen.json")))["argv"] if t.startswith("ckpt_path=")], [f"ckpt_path={self.ckpt}"])   # once: nothing appended
            r, out = self._design(site, mode, extra=["ckpt_path=/elsewhere.ckpt"], out=f"ck_clash_{mode}", python=py)
            self.assertEqual(r.returncode, cli.EXIT_USAGE, r.stderr)
            self.assertIn(f"design: --ckpt {self.ckpt} and ckpt_path=/elsewhere.ckpt on the line disagree", r.stderr)
            self.assertFalse(os.path.isdir(out))                                                # nothing ran
        r = fx.run_cli(["design", "--mode", "off", f"inputs={self.spec}", f"out_dir={os.path.join(self.td.name, 'nock')}"], self.pristine, bin_dir=self.bin, python=self.stockpy)
        self.assertEqual(r.returncode, cli.EXIT_USAGE, r.stderr); self.assertIn("design: no checkpoint named — --ckpt <file>, RFD3_CKPT, or upstream's ckpt_path=<file> on the line", r.stderr)

    def test_the_earlier_flag_spellings_read_as_upstreams_keys(self):
        """`--spec X` / `--out_dir Y` (the grammar design lines were sealed in before the pass-through one) reach upstream as `inputs=X` /
        `out_dir=Y`, in place, on both arms — nothing printed, nothing else accepted (cli._SEALED_SPELLINGS)."""
        fx.fake_torch(self.patched)
        for site, mode, py in ((self.pristine, "off", self.stockpy), (self.patched, "exact", None)):
            out = os.path.join(self.td.name, f"sealed_{mode}")
            r = fx.run_cli(["design", "--mode", mode, "--spec", self.spec, "--out_dir", out, "--ckpt", self.ckpt, "--", "diffusion_batch_size=2"], site, bin_dir=self.bin, python=py, env={"RFD3_FAKE_IMPORT_HOIST": "1"})
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual(json.load(open(os.path.join(out, "cli_seen.json")))["argv"][1:], [f"inputs={self.spec}", f"out_dir={out}", "diffusion_batch_size=2", f"ckpt_path={self.ckpt}"])
        self.assertEqual(cli._SEALED_SPELLINGS, {"--spec": "inputs", "--out_dir": "out_dir"})

    def test_off_route_refuses_a_patched_interpreter(self):
        patchedpy = fx.make_venv(os.path.join(self.td.name, "pv"), self.patched)
        r, out = self._design(self.patched, "off", python=patchedpy)           # no stock interpreter named: this (patched) one is proven and refused
        self.assertEqual(r.returncode, 3)
        self.assertIn("NOT STOCK", r.stderr)
        self.assertIn("rfd3 tree is not pristine", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))    # nothing ran
        proof = manifest.read(out)["stock_env_proof"]
        self.assertFalse(proof["ok"])

    def test_off_route_uses_the_named_stock_interpreter(self):
        """From the patched interpreter, `--mode off` reaches the pristine one; the caller's PYTHONPATH (the patched site here) is not inherited (-I)."""
        r, out = self._design(self.patched, "off", env={"RFDIFFUSION3_STOCK_PYTHON": self.stockpy})
        self.assertEqual(r.returncode, 0, r.stderr)
        proof = manifest.read(out)["stock_env_proof"]
        self.assertTrue(proof["ok"], proof)
        self.assertEqual(os.path.realpath(proof["python"]), os.path.realpath(self.stockpy))
        self.assertIn(f"stock interpreter={self.stockpy}", r.stderr)

    def test_exact_route_in_process(self):
        """The kit's module (the real one, under a stub torch) is imported while the stub CLI runs, as the model build does: the APPLIED
        line is the evidence of application the exit rule needs (test_exit_rule: without it a completed run is partial, exit 3)."""
        fx.fake_torch(self.patched)
        r, out = self._design(self.patched, "exact", extra=["--", "inference_sampler.step_scale=3.0", "seed=5", "dump_trajectories=True"], env={"RFD3_FAKE_IMPORT_HOIST": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertEqual(seen["RFD3_HOIST"], "1")                                  # the delta was in the environment when the CLI ran
        self.assertIn("[rfdiffusion3-opt] APPLIED rfd3.model.hoist RFD3_HOIST=True", r.stderr)
        self.assertIn("diffusion_batch_size=2", seen["argv"])
        self.assertIn("inference_sampler.step_scale=3.0", seen["argv"])
        self.assertIn("seed=5", seen["argv"])
        self.assertEqual(seen["argv"][-2:], ["dump_trajectories=True", f"ckpt_path={self.ckpt}"])   # the caller's last token, then the kit's one
        self.assertIn("ACTIVE mode=exact env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1 interpreter=patched(10/10 patched)", r.stderr)
        self.assertRegex(r.stderr, r" applied=at-import attn_pin=1 alloc_conf=\S+ det=0 trigger=design")   # report.activation_line field order
        self.assertIn(f"design mode=exact det=0 command=rfd3 design inputs={self.spec} out_dir={out} diffusion_batch_size=2 inference_sampler.step_scale=3.0 seed=5 dump_trajectories=True ckpt_path={self.ckpt}", r.stderr)
        self.assertIn("[rfdiffusion3-opt] EXIT pid=", r.stderr)
        man = manifest.read(out)
        self.assertEqual(man["mode"], "exact")
        self.assertTrue(man["active"])
        self.assertEqual(man["env_delta"], {"RFD3_HOIST": "1", "RFD3_FZT": "1", "RFD3_CUDAGRAPH": "1", "RFD3_INIT_CHUNK": "1"})
        self.assertEqual(man["settings"]["values"], {"diffusion_batch_size": "2", "inference_sampler.step_scale": "3.0", "seed": "5", "dump_trajectories": "True"})
        self.assertEqual(man["interpreter"]["states"]["rfd3/model/hoist.py"], "patched")
        self.assertEqual((man["spec"]["value"], man["spec"]["is_file"]), (self.spec, True)); self.assertNotIn("example_ids", man["spec"])
        self.assertTrue(all("sha256" in o for o in man["outputs"]), man["outputs"])
        self.assertEqual({k: man[k] for k in manifest.LEVER_KEYS}, {"levers_applied": ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"], "levers_fallback": [], "levers_unavailable": [], "partial": False,
                                                                    "partial_reason": None, "allow_partial": False})
        self.assertEqual(man["activation_report"]["levers_planned"], ["RFD3_CUDAGRAPH", "RFD3_FZT", "RFD3_HOIST", "RFD3_INIT_CHUNK"])
        self.assertEqual((man["activation_report"]["det"], man["activation_report"].get("declined")), ({"level": 0}, None))
        self.assertEqual(man["exit_code"], 0)

    def test_kit_modes_attempt_any_batch(self):
        """A kit mode caps no diffusion batch: `design` under exact and fast at a batch far above what the GPU serves at the design length
        (500 residues, B=512) activates and hands upstream's entry point the line as given — the pass's outcome (torch's out-of-memory
        error, the KERNELS gate) is the run's own, as under `off`."""
        fx.fake_torch(self.patched)
        spec = os.path.join(self.td.name, "L500.json")
        json.dump({"L500_000": {"dialect": 2, "length": "500"}}, open(spec, "w"))
        for mode in ("exact", "fast"):
            out = os.path.join(self.td.name, f"big_{mode}")
            r = fx.run_cli(["design", "--mode", mode, "--ckpt", self.ckpt, f"inputs={spec}", f"out_dir={out}", "diffusion_batch_size=512"], self.patched, bin_dir=self.bin, env={"RFD3_FAKE_IMPORT_HOIST": "1"})
            self.assertIn(f"[rfdiffusion3-opt] ACTIVE mode={mode} ", r.stderr, r.stderr)          # activated: nothing stood between the line and the mode's gates
            seen = json.load(open(os.path.join(out, "cli_seen.json")))                              # upstream's entry point ran ...
            self.assertIn("diffusion_batch_size=512", seen["argv"])                                # ... with the batch as given

    def test_exact_route_refused_on_pristine(self):
        r, out = self._design(self.pristine, "exact")
        self.assertEqual(r.returncode, 3)
        self.assertIn("NOT ACTIVE: the kit is not installed in this interpreter", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))

    def test_mode_argument_precedence_and_usage(self):
        fx.fake_torch(self.patched)
        r, out = self._design(self.patched, "exact", env={"RFDIFFUSION3_OPT": "off", "RFD3_FAKE_IMPORT_HOIST": "1"})   # --mode wins over the variable (the core's mode_argument)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("ACTIVE mode=exact env=RFD3_HOIST=1", r.stderr)
        self.assertEqual(manifest.read(out)["mode"], "exact")
        r, _ = self._design(self.patched, "exact_lowmem")                              # a lever subset is not a mode: argparse's choices (modes.MODES)
        self.assertEqual(r.returncode, 2)
        self.assertIn("invalid choice: 'exact_lowmem'", r.stderr)
        r = fx.run_cli(["design", "--mode", "exact", f"inputs={self.spec}", f"out_dir={os.path.join(self.td.name, 'o')}"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 2)
        self.assertIn("--ckpt", r.stderr)
        r = fx.run_cli(["design", "--mode", "exact", "--det", "2", "--ckpt", self.ckpt, f"inputs={self.spec}", f"out_dir={os.path.join(self.td.name, 'o')}"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 2); self.assertIn("--det: invalid choice: 2", r.stderr)   # det.LEVELS
        r = fx.run_cli([], self.patched)
        self.assertEqual(r.returncode, 2)
        r = fx.run_cli(["--version"], self.patched)                                    # not an option
        self.assertEqual(r.returncode, 2)

    def test_inputs_is_upstreams_value_verbatim(self):
        """`inputs=` is upstream's key: a value that is not a file (upstream's comma list, a glob, anything its key takes) is not looked
        at by the package — it reaches the line verbatim on both arms, and the run's outcome is upstream's own (the stub CLI here can
        only open files: exit 1, its error, not a usage error); a relative path stays as written (upstream resolves it, as under `rfd3 design`)."""
        fx.fake_torch(self.patched)
        for site, mode, py in ((self.patched, "exact", None), (self.pristine, "off", self.stockpy)):
            out = os.path.join(self.td.name, f"verbatim_{mode}")
            r = fx.run_cli(["design", "--mode", mode, "--ckpt", self.ckpt, "inputs=A11-20,/0,B5-9", f"out_dir={out}"], site, bin_dir=self.bin, python=py, env={"RFD3_FAKE_IMPORT_HOIST": "1"})
            self.assertNotEqual(r.returncode, cli.EXIT_USAGE, r.stderr); self.assertNotIn("not found", r.stderr.split("Traceback")[0])
            self.assertIn(f"-- design inputs=A11-20,/0,B5-9 out_dir={out} ckpt_path={self.ckpt}" if mode == "off" else f"command=rfd3 design inputs=A11-20,/0,B5-9 out_dir={out} ckpt_path={self.ckpt}", r.stderr)
            self.assertEqual(r.returncode, cli.EXIT_FAIL, r.stderr)                    # the stub's own FileNotFoundError: the run failed, exit 1
        rel = os.path.relpath(self.spec)
        r = fx.run_cli(["design", "--mode", "off", "--ckpt", self.ckpt, f"inputs={rel}", f"out_dir={os.path.join(self.td.name, 'abs')}", "--", "diffusion_batch_size=1"], self.pristine, bin_dir=self.bin, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"inputs={rel} ", r.stderr)                                      # as written: the kit rewrites no upstream value

    def test_check_exit_codes(self):
        r = fx.run_cli(["check"], self.patched, bin_dir=self.bin)                        # --mode omitted, RFDIFFUSION3_OPT unset: the default (modes.DEFAULT_MODE) with its own delta
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"DRY-RUN mode={modes.DEFAULT_MODE} env={modes.describe_env(modes.resolve(None).env)} interpreter=patched(10/10 patched)", r.stderr)
        r = fx.run_cli(["check", "--mode", "exact"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("DRY-RUN mode=exact env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1 interpreter=patched(10/10 patched)", r.stderr)
        self.assertIn("applied=dry-run", r.stderr)
        r = fx.run_cli(["check", "--mode", "exact", "--json"], self.pristine, bin_dir=self.bin)
        self.assertEqual(r.returncode, 3)
        rep = json.loads(r.stdout)
        self.assertFalse(rep["active"])
        self.assertTrue(any("the kit is not installed" in f for f in rep["findings"]))
        self.assertIn("would_refuse=", r.stderr)
        r = fx.run_cli(["check", "--mode", "off"], self.pristine, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("stock interpreter = this one: tree pristine (8/8 stock, 2 kit files absent); pins ok", r.stderr)
        r = fx.run_cli(["check", "--mode", "off"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 3)                                          # off in a patched interpreter: the stock arm has no pristine tree here
        r = fx.run_cli(["check", "--mode", "off"], self.patched, env={"RFDIFFUSION3_STOCK_PYTHON": self.stockpy}, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(f"stock interpreter {self.stockpy}: pristine", r.stderr); self.assertNotIn("NOT pristine", r.stderr)
        unpinned = fx.unpin_distribution(fx.make_site(os.path.join(self.td.name, "u"), "pristine"))   # a pristine tree whose distribution is off the pin: the TREE is the gate, the pin a report
        r = fx.run_cli(["check", "--mode", "off"], unpinned, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("tree pristine (8/8 stock, 2 kit files absent); pins (reported): rc-foundry 0.2.1.dev13+g4010e3e2e: installed from a non-git, non-archive source", r.stderr)
        self.assertEqual(len([l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] PINS ")]), 1, r.stderr)
        r = fx.run_cli(["check", "--mode", "exact"], fx.unpin_distribution(fx.make_site(os.path.join(self.td.name, "up"), "patched")), bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr); self.assertIn("] PINS rc-foundry ", r.stderr); self.assertNotIn("would_refuse", r.stderr)   # a kit mode likewise: reported, the dry run passes
        r = fx.run_cli(["check", "--mode", "fast"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)                                # fast gates like exact: a dry run of its delta
        self.assertIn("DRY-RUN mode=fast env=RFD3_HOIST=1,RFD3_FZT=1,RFD3_CUDAGRAPH=1,RFD3_INIT_CHUNK=1,RFD3_COMPILE=1,RFD3_TOKEN_SDPA=1,RFD3_GATHER_ATTN=1", r.stderr)
        r = fx.run_cli(["check", "--mode", "exact_lowmem"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 2)                                          # a documented candidate: refused with its reason, usage
        r = fx.run_cli(["check"], self.patched, env={"RFDIFFUSION3_OPT": "exact"}, bin_dir=self.bin)
        self.assertEqual(r.returncode, 0, r.stderr)                                # the environment names the mode

    def test_warm_child_reports_after_initialize(self):
        """The kit's module is imported while the engine builds the model: the APPLIED line must be read after initialize()."""
        from .. import warm
        self.assertLess(warm.CHILD.index("engine.initialize()"), warm.CHILD.index("stack.report_applied()"))
        self.assertLess(warm.CHILD.index("rfdiffusion3_opt.enable(mode"), warm.CHILD.index("from rfd3.engine import"))   # the delta before the import

    def test_warm_without_checkpoint(self):
        r = fx.run_cli(["warm", "--mode", "exact"], self.patched, bin_dir=self.bin)
        self.assertEqual(r.returncode, 2)
        self.assertIn("WARM FAIL", r.stderr)
        self.assertIn("checkpoint not found", r.stderr)

    def test_warm_not_active_on_pristine(self):
        """The warm child activates first: on a pristine interpreter it stops at the NOT ACTIVE line (exit 3), before any upstream import."""
        r = fx.run_cli(["warm", "--mode", "exact", "--ckpt", self.ckpt], self.pristine, bin_dir=self.bin)
        self.assertEqual(r.returncode, 3)
        self.assertIn("WARM FAIL", r.stderr)
        self.assertIn("error=NotActive", r.stderr)
        self.assertIn("the kit is not installed in this interpreter", r.stderr)

    def test_warm_off_is_proven_stock(self):
        """`warm --mode off` = the stock caller's proof in the pristine interpreter with every lever name stripped, then the engine
        build: PASS needs the STOCK line and `APPLIED none`. A lever in the caller's environment is stripped, never seen."""
        from .. import warm
        fx.stub_engine(self.pristine)
        seen = os.path.join(self.td.name, "warm_seen.json")
        r = fx.run_cli(["warm", "--mode", "off", "--ckpt", self.ckpt, "--json"], self.pristine,
                       env={"RFD3_HOIST": "1", "RFD3_LOW_MEMORY_MODE": "1", "RFD3_FAKE_SEEN": seen}, bin_dir=self.bin, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
        self.assertIn("WARM PASS mode=off", r.stderr)
        res = json.loads(r.stdout)
        self.assertNotIn("killed", res); self.assertTrue(os.path.isfile(res["log"]), res)   # warm keeps its own log (no --log / --timeout): the path rides in the result
        self.assertTrue(res["stock_proof"].startswith("[rfdiffusion3-opt stock] STOCK rc-foundry=0.2.1.dev13+g4010e3e2e tree=pristine"), res["stock_proof"])
        self.assertTrue(res["stock_proof"].endswith(" forbidden_absent=11 upstream_env=RFD3_LOW_MEMORY_MODE=1 det=0"), res["stock_proof"])   # upstream's own switch reached the child and is named
        self.assertTrue(res["applied"].startswith("[rfdiffusion3-opt] APPLIED none"))
        self.assertTrue(res["activation"].startswith("[rfdiffusion3-opt] NOT ACTIVE mode=off env=none interpreter=pristine("))
        self.assertIn("env_present=RFD3_LOW_MEMORY_MODE (kit names are stripped from the stock arm's child; upstream's own switches pass through)", res["activation"])   # reported: the lever name was stripped, upstream's switch was not
        text = open(res["log"]).read()
        self.assertLess(text.index("] STOCK "), text.index("] APPLIED none"))    # proof first, then the build
        d = json.load(open(seen))
        self.assertIsNone(d["RFD3_HOIST"])
        self.assertIsNone(d["RFDIFFUSION3_OPT"])
        self.assertFalse(d["hoist_imported"])
        self.assertEqual(os.path.realpath(d["python"]), os.path.realpath(self.stockpy))
        from unittest import mock
        with mock.patch.dict(os.environ, {"RFD3_HOIST": "1", "RFDIFFUSION3_OPT_DET": "1", "RFD3_LOW_MEMORY_MODE": "1", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}):
            env = warm.child_env("off")
        self.assertTrue(all(n not in env for n in list(fx.stack.lever_names()) + list(fx.stack.PACKAGE_ENV)), sorted(env))
        self.assertEqual((env["RFD3_LOW_MEMORY_MODE"], env["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]), ("1", "1"))   # upstream's switch and torch's variable pass through

    def test_warm_off_refuses_a_non_stock_interpreter(self):
        """A patched interpreter (no pristine one named) with a lever in the caller's environment: NOT STOCK, exit 3, nothing built."""
        fx.stub_engine(self.patched)
        patchedpy = fx.make_venv(os.path.join(self.td.name, "pv"), self.patched)
        seen = os.path.join(self.td.name, "warm_seen.json")
        r = fx.run_cli(["warm", "--mode", "off", "--ckpt", self.ckpt, "--json"], self.patched, env={"RFD3_HOIST": "1", "RFD3_FAKE_SEEN": seen},
                       bin_dir=self.bin, python=patchedpy)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("WARM FAIL mode=off", r.stderr)
        self.assertIn("error=NotStock", r.stderr)
        self.assertIn("rfd3 tree is not pristine", r.stderr)
        self.assertFalse(os.path.exists(seen))                                    # the engine was never built
        res = json.loads(r.stdout)
        self.assertTrue(res["stock_proof"].startswith("[rfdiffusion3-opt stock] NOT STOCK"))
        self.assertIsNone(res["applied"])

    def test_checkpoint_digest_cache(self):
        """The checkpoint's sha256 is hashed once per (path, bytes, mtime) and served from the cache after; a changed file is rehashed."""
        from unittest import mock
        with mock.patch.dict(os.environ, {manifest.CACHE_ENV: os.path.join(self.td.name, "cache")}):
            sha, src = manifest.checkpoint_sha256(self.ckpt)
            self.assertEqual((sha, src), (registry.sha256_of(self.ckpt), "hashed"))
            self.assertEqual(manifest.checkpoint_sha256(self.ckpt), (sha, "cache"))
            with open(self.ckpt, "ab") as fh:
                fh.write(b"!")
            sha2, src2 = manifest.checkpoint_sha256(self.ckpt)
            self.assertEqual((sha2 == registry.sha256_of(self.ckpt), sha2 != sha, src2), (True, True, "hashed"))
            self.assertTrue(os.path.isfile(manifest.digest_cache_path()))


class TestStockProof(unittest.TestCase):
    def test_env_proof_names_every_failure(self):
        with tempfile.TemporaryDirectory() as td:
            site = fx.make_site(td, "pristine")
            code = '''
import json, os, sys
os.environ["MODEL_OPT"] = os.environ["MODEL_OPT"]
from rfdiffusion3_opt import stock_design
p = stock_design.env_proof(dict(os.environ, RFD3_CUDAGRAPH="1", RFD3_LOW_MEMORY_MODE="1"))
q = stock_design.env_proof(dict(os.environ, RFD3_LOW_MEMORY_MODE="1"))
print(json.dumps({"ok": p["ok"], "present": p["forbidden_present"], "reasons": p["reasons"], "n": len(p["forbidden_names_checked"]), "ups": p["upstream_env"], "notes": p["pins"]["notes"],
                  "q_ok": q["ok"], "q_line": stock_design.stock_line(q), "q_pins": stock_design.pins_note_lines(q)}))
'''
            r = fx.run_py(code, site)
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertFalse(d["ok"])
            self.assertEqual(d["present"], ["RFD3_CUDAGRAPH"])
            self.assertEqual(d["reasons"], ["forbidden environment names present: ['RFD3_CUDAGRAPH']"])
            self.assertEqual((d["n"], d["ups"], d["notes"]), (11, {"RFD3_LOW_MEMORY_MODE": "1"}, []))   # the kit's 10 must-be-absent names; upstream's switch recorded, not forbidden; at the pin: no note
            self.assertTrue(d["q_ok"]); self.assertEqual(d["q_pins"], [])
            self.assertRegex(d["q_line"], r"^\[rfdiffusion3-opt stock\] STOCK rc-foundry=0\.2\.1\.dev13\+g4010e3e2e tree=pristine python=\S+ forbidden_absent=11 upstream_env=RFD3_LOW_MEMORY_MODE=1 det=0$")
            sitedir = os.path.join(td, "sitedir"); os.makedirs(sitedir)                # the package's .pth line, processed as site.py processes it
            with open(os.path.join(sitedir, "rfdiffusion3_opt_autoload.pth"), "w") as f:
                f.write(open(os.path.join(fx.PKG_PARENT, "rfdiffusion3_opt_autoload.pth")).read())
            r = fx.run_py(f"import site; site.addsitedir({sitedir!r})\nimport json,os\nfrom rfdiffusion3_opt import stock_design\np=stock_design.env_proof()\nprint(json.dumps({{'ok': p['ok'], 'r': p['reasons']}}))", site, env={"RFDIFFUSION3_OPT": "exact"})
            d = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertFalse(d["ok"])
            self.assertTrue(any("autoload finder is armed" in x for x in d["r"]), d)

    def test_the_tree_is_the_gate_the_pin_a_note(self):
        """env_proof in a pristine interpreter whose distribution is OFF the pin (an index install): ok, the finding in `pins.notes` and one
        PINS line from pins_note_lines; in a PATCHED interpreter (at the pin): NOT ok, the reason names the tree's files, no note."""
        with tempfile.TemporaryDirectory() as td:
            probe = ("import json, os\nfrom rfdiffusion3_opt import stock_design\np = stock_design.env_proof()\n"
                     "print(json.dumps({'ok': p['ok'], 'reasons': p['reasons'], 'notes': p['pins']['notes'], 'lines': stock_design.pins_note_lines(p), 'pristine': p['pins']['pristine']}))")
            d = json.loads(fx.run_py(probe, fx.unpin_distribution(fx.make_site(os.path.join(td, "u"), "pristine"))).stdout.strip().splitlines()[-1])
            self.assertEqual((d["ok"], d["reasons"], d["pristine"]), (True, [], True))
            self.assertEqual(len(d["notes"]), 1); self.assertIn("rc-foundry 0.2.1.dev13+g4010e3e2e: installed from a non-git, non-archive source; want version", d["notes"][0])
            self.assertEqual(d["lines"], [f"[rfdiffusion3-opt stock] PINS {d['notes'][0]}"])
            d = json.loads(fx.run_py(probe, fx.make_site(os.path.join(td, "p"), "patched")).stdout.strip().splitlines()[-1])
            self.assertEqual((d["ok"], d["pristine"]), (False, False))
            self.assertEqual(len(d["reasons"]), 1); self.assertTrue(d["reasons"][0].startswith("rfd3 tree is not pristine: rfd3/model/"), d["reasons"])
            self.assertTrue(all(n.startswith("rfd3 tree is not pristine: ") for n in d["notes"]), d["notes"])   # the pin checker's own tree finding rides as a note too; the distribution is at its pin: no other
            r = fx.run_py("import sys\nfrom rfdiffusion3_opt import stock_design\nsys.exit(stock_design.main(['--det', '0', '--', 'design', 'out_dir=/nonexistent/o']))", fx.make_site(os.path.join(td, "p2"), "patched"))
            self.assertEqual(r.returncode, stock_design.EXIT_NOT_STOCK, r.stderr)      # 3: the stock child in a patched interpreter stops at NOT STOCK, before any upstream import
            self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt stock\] NOT STOCK: rfd3 tree is not pristine: ")

    def test_usage(self):
        self.assertEqual(stock_design.main(["--", "check"]), 2)
        self.assertEqual(stock_design.main([]), 2)



if __name__ == "__main__":
    unittest.main()
