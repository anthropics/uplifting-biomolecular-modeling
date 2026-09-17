"""`run.sh install [--weights DIR [--fetch]]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation;
nothing is installed or fetched), the weights step's verdicts (af3_torch_opt.weights.check with the kit's real digest judge on an injected pin table:
pinned / unpinned / missing, and --fetch with a stand-in download and converter), and the environment recipe's agreement with stock/PINS.json
(environment/Dockerfile, the two locks, .gitignore). CPU only, no network."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
ENV_DIR = os.path.join(TREE, "environment")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_FETCH / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s%s\\n' "$*" "${AF3_TORCH_OPT:+ [AF3_TORCH_OPT=$AF3_TORCH_OPT]}" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"fetch_upstream.py"*) exit "${STUB_RC_FETCH:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m af3_torch_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="af3t_install_verb_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        self.stub = os.path.join(self.bin, "python")
        with open(self.stub, "w") as f: f.write(STUB)
        os.chmod(self.stub, os.stat(self.stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, args, env_extra=None, **rc):
        if os.path.exists(self.log): os.remove(self.log)                                   # one call record per run.sh invocation
        env = {"PATH": self.bin + os.pathsep + "/usr/bin:/bin", "HOME": self.tmp, "STUB_LOG": self.log}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()}); env.update(env_extra or {})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=self.tmp)
        calls = open(self.log).read().splitlines() if os.path.exists(self.log) else []
        return r.returncode, r.stdout + r.stderr, calls

    def kinds(self, calls):
        out = []
        for c in calls:
            if "-I -c import os,sys" in c: out.append("probe")
            elif "-m pip install" in c: out.append("pip")
            elif "fetch_upstream.py" in c: out.append("fetch")
            elif "check_pins.py" in c: out.append("pins")
            elif "-m af3_torch_opt.weights" in c: out.append("weights")
            else: out.append("other:" + c)
        return out

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.kinds(calls), ["probe", "pip", "fetch", "pins"])
        pip = calls[1]
        self.assertIn("-m pip install -e " + os.path.join(TREE, "..", "common", "opt_core") + " -e " + os.path.join(TREE, "opt"), pip)   # the shared core first, then the kit, both editable
        self.assertEqual(calls[2], "-I " + os.path.join(TREE, "stock", "fetch_upstream.py"))                                              # xfold's pinned archive made present (fetched when the tree lacks it)
        self.assertTrue(calls[3].startswith("-I " + os.path.join(TREE, "stock", "check_pins.py")), calls[3])                              # the pin check, isolated mode, no --digest (software pins)
        self.assertNotIn("--digest", calls[3])

    def test_weights_step_follows_the_pin_check(self):
        d = os.path.join(self.tmp, "params")
        rc, out, calls = self.run_sh(["install", "--weights", d])
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.kinds(calls), ["probe", "pip", "fetch", "pins", "weights"])
        self.assertTrue(calls[4].endswith("-m af3_torch_opt.weights " + d), calls[4])           # no --fetch unless asked
        rc, out, calls = self.run_sh(["install", "--weights=" + d])
        self.assertEqual((rc, self.kinds(calls)[-1]), (0, "weights"), out)
        rc, out, calls = self.run_sh(["install", "--weights", d, "--fetch"])
        self.assertEqual((rc, self.kinds(calls)), (0, ["probe", "pip", "fetch", "pins", "weights"]), out)
        self.assertTrue(calls[4].endswith("-m af3_torch_opt.weights " + d + " --fetch"), calls[4])   # --fetch reaches the weights step, nothing else
        rc, out, calls = self.run_sh(["install", "--fetch", "--weights", d])
        self.assertTrue(rc == 0 and calls[4].endswith("-m af3_torch_opt.weights " + d + " --fetch"), (out, calls))   # either order

    def test_the_mode_variable_is_kept_from_the_probes(self):
        rc, out, calls = self.run_sh(["install", "--weights", self.tmp], env_extra={"AF3_TORCH_OPT": "fast"})
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in calls if ("check_pins.py" in c or "af3_torch_opt.weights" in c) and "[AF3_TORCH_OPT=" in c], calls)   # env -u AF3_TORCH_OPT on both

    def test_an_install_from_this_tree_skips_pip(self):
        rc, out, calls = self.run_sh(["install"], probe=0)
        self.assertEqual((rc, self.kinds(calls)), (0, ["probe", "fetch", "pins"]), out)
        self.assertIn("the pip step is skipped", out)

    def test_exit_codes_follow_the_failing_step(self):
        self.assertEqual(self.run_sh(["install"], pip=1)[0], 1)
        rc, out, calls = self.run_sh(["install", "--weights", self.tmp], fetch=1)
        self.assertEqual((rc, self.kinds(calls)), (1, ["probe", "pip", "fetch"]), out)         # the archive step failed by name: nothing after it runs
        self.assertIn("stock/fetch_upstream.py", out)
        rc, out, calls = self.run_sh(["install", "--weights", self.tmp], pins=3)
        self.assertEqual((rc, self.kinds(calls)), (3, ["probe", "pip", "fetch", "pins"]), out)  # refused by the pin check: the weights step does not run
        self.assertEqual(self.run_sh(["install", "--weights", self.tmp], weights=1)[0], 1)

    def test_usage_errors(self):
        for args in (["install", "--weights"], ["install", "--weights", "--bogus"], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--config", "h100"], ["install", "--mode", "fast"], ["install", "--fetch"], ["install", "--weights", "--fetch"]):
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out)); self.assertEqual(calls, [], (args, calls))   # refused before any interpreter call
        rc, out, _ = self.run_sh([])
        self.assertEqual(rc, 2); self.assertIn("run.sh install [--weights DIR [--fetch]]", out) # the usage text lists the verb


class WeightsStep(unittest.TestCase):
    """af3_torch_opt.weights.check on a directory, with the kit's real digest judge (stock/check_pins.py digest_report) and an injected pin table
    whose checkpoint is a small file of known digest."""
    def setUp(self):
        from af3_torch_opt import stack, weights
        self.stack, self.weights = stack, weights
        self.tmp = tempfile.mkdtemp(prefix="af3t_weights_")
        self.blob = b"of3 port test checkpoint\n" * 64
        conv = b'{"layout": "of3-preview2"}\n'
        self.pins = json.loads(json.dumps(stack.pins()))
        self.pins["variants"] = {"p2": {"name": "OpenFold3-preview2", "layout": "of3-preview2",
                                        "converted": {"file": "of3_ported_weights.bin.zst", "sha256": hashlib.sha256(self.blob).hexdigest(), "bytes": len(self.blob)},
                                        "source": "the test's own bytes",
                                        "conventions": {"file": "of3_conventions.json", "sha256": hashlib.sha256(conv).hexdigest(), "bytes": len(conv)}}}
        self.conv = conv
        self.saved = os.environ.get("AF3_TORCH_PARAMS_DIR")

    def tearDown(self):
        if self.saved is None: os.environ.pop("AF3_TORCH_PARAMS_DIR", None)
        else: os.environ["AF3_TORCH_PARAMS_DIR"] = self.saved

    def check(self, d):
        out = io.StringIO(); rc = self.weights.check(d, pins=self.pins, out=out); return rc, out.getvalue()

    def test_pinned(self):
        d = os.path.join(self.tmp, "pinned"); os.makedirs(d)
        open(os.path.join(d, "of3_ported_weights.bin.zst"), "wb").write(self.blob); open(os.path.join(d, "of3_conventions.json"), "wb").write(self.conv)
        rc, text = self.check(d)
        self.assertEqual(rc, 0, text); self.assertIn("WEIGHTS OK", text); self.assertIn("export AF3_TORCH_PARAMS_DIR=" + d, text); self.assertIn("has the pinned digest", text)
        self.assertEqual(os.environ.get("AF3_TORCH_PARAMS_DIR"), d)                             # the judge read the directory through the deployment variable

    def test_unpinned_runs(self):
        d = os.path.join(self.tmp, "other"); os.makedirs(d)
        open(os.path.join(d, "my_weights.bin.zst"), "wb").write(b"another checkpoint")
        rc, text = self.check(d)
        self.assertEqual(rc, 0, text); self.assertIn("WEIGHTS UNPINNED", text); self.assertIn(self.stack.UNPINNED_WORDS, text); self.assertIn("is absent (optional", text)

    def test_missing_is_refused_with_the_file_to_obtain(self):
        d = os.path.join(self.tmp, "empty"); os.makedirs(d)
        rc, text = self.check(d)
        self.assertEqual(rc, 1, text); self.assertIn("WEIGHTS MISSING", text); self.assertIn("of3_ported_weights.bin.zst", text); self.assertIn("the test's own bytes", text)   # variants.p2.source, verbatim
        self.assertEqual(self.check(os.path.join(self.tmp, "absent"))[0], 1)

    def test_usage(self):
        self.assertEqual(self.weights.main([]), 2); self.assertEqual(self.weights.main(["--help"]), 2); self.assertEqual(self.weights.main(["a", "b"]), 2)
        self.assertEqual(self.weights.main(["--fetch"]), 2)                                     # --fetch alone names no directory

    # --fetch: a stand-in download (an in-memory opener keyed by URL) and a stand-in converter (a script under a scratch "fork checkout" that
    # writes the injected table's converted bytes), run through the real fetch_and_convert on this interpreter as AF3_TORCH_JAX_PY.
    CKPT = b"public checkpoint bytes\n" * 32

    def fetch_fixture(self, converter_body=None, ckpt_sha=None):
        repo = os.path.join(self.tmp, "fork"); os.makedirs(repo, exist_ok=True)
        script = os.path.join(repo, "convert_of3_weights.py")
        with open(script, "w") as f:
            f.write(converter_body or (
                "import os, sys\n"
                "a = sys.argv[1:]; src = a[a.index('--of3_checkpoint') + 1]; out = a[a.index('--output_dir') + 1]\n"
                "assert open(src, 'rb').read() == %r\n"
                "assert os.environ.get('JAX_PLATFORMS') == 'cpu' and not [k for k in os.environ if k.startswith('AF3_TORCH_OPT')]\n"
                "open(os.path.join(out, 'of3_ported_weights.bin.zst'), 'wb').write(%r)\n" % (self.CKPT, self.blob)))
        self.pins["variants"]["p2"]["checkpoint"] = {"file": "of3-p2-155k.pt", "url": "https://example.invalid/of3-p2-155k.pt",
                                                     "sha256": ckpt_sha or hashlib.sha256(self.CKPT).hexdigest(), "bytes": len(self.CKPT)}
        self.pins["variants"]["p2"]["converter"] = {"script": "convert_of3_weights.py"}
        env = {"AF3_TORCH_JAX_PY": os.environ.get("AF3_TORCH_JAX_PY"), "AF3_TORCH_JAX_REPO": os.environ.get("AF3_TORCH_JAX_REPO")}
        self.addCleanup(lambda: [os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v) for k, v in env.items()])
        os.environ["AF3_TORCH_JAX_PY"], os.environ["AF3_TORCH_JAX_REPO"] = sys.executable, repo
        os.environ["AF3_TORCH_OPT_PROBE"] = "1"; self.addCleanup(os.environ.pop, "AF3_TORCH_OPT_PROBE", None)   # a package switch: must not reach the converter
        urls = []

        class Opener:
            def __init__(s_, url): urls.append(url); s_.b = io.BytesIO(self.CKPT)
            def __enter__(s_): return s_.b
            def __exit__(s_, *a): return False
        return Opener, urls

    def fetch(self, d, opener):
        out = io.StringIO(); rc = self.weights.check(d, pins=self.pins, out=out, fetch=True, opener=opener); return rc, out.getvalue()

    def test_fetch_downloads_converts_and_judges(self):
        opener, urls = self.fetch_fixture()
        d = os.path.join(self.tmp, "fetched")
        rc, text = self.fetch(d, opener)
        self.assertEqual(rc, 0, text)
        self.assertEqual(urls, ["https://example.invalid/of3-p2-155k.pt"])
        self.assertRegex(text, r"(?s)FETCH downloaded .*CONVERT running .*convert_of3_weights\.py --of3_checkpoint .*/checkpoint/of3-p2-155k\.pt --output_dir .*CONVERT done .*WEIGHTS OK")
        self.assertEqual(open(os.path.join(d, "checkpoint", "of3-p2-155k.pt"), "rb").read(), self.CKPT)                 # kept for the user, nothing deleted
        rc, text = self.fetch(d, opener)                                                                                  # a second run: the checkpoint is judged present, nothing fetched
        self.assertEqual((rc, urls), (0, ["https://example.invalid/of3-p2-155k.pt"]), text); self.assertIn("WEIGHTS OK", text); self.assertNotIn("FETCH", text)

    def test_fetch_keeps_a_checkpoint_already_downloaded(self):
        opener, urls = self.fetch_fixture()
        d = os.path.join(self.tmp, "predownloaded"); os.makedirs(os.path.join(d, "checkpoint"))
        open(os.path.join(d, "checkpoint", "of3-p2-155k.pt"), "wb").write(self.CKPT)
        rc, text = self.fetch(d, opener)
        self.assertEqual((rc, urls), (0, []), text); self.assertIn("FETCH kept", text); self.assertIn("WEIGHTS OK", text)

    def test_fetch_refuses_a_download_with_another_digest(self):
        opener, urls = self.fetch_fixture(ckpt_sha="0" * 64)
        d = os.path.join(self.tmp, "baddigest")
        rc, text = self.fetch(d, opener)
        self.assertEqual(rc, 1, text); self.assertIn("WEIGHTS FETCH FAILED", text); self.assertIn("is not the pinned", text); self.assertNotIn("CONVERT", text)
        self.assertTrue(os.path.isfile(os.path.join(d, "checkpoint", "of3-p2-155k.pt.part")))                            # kept for inspection, not under the checkpoint's name

    def test_fetch_names_a_failed_download(self):
        opener, urls = self.fetch_fixture()

        def offline(url): raise OSError("network is unreachable")
        rc, text = self.fetch(os.path.join(self.tmp, "offline"), offline)
        self.assertEqual(rc, 1, text); self.assertIn("WEIGHTS FETCH FAILED url=https://example.invalid/of3-p2-155k.pt", text)

    def test_fetch_names_a_failed_conversion(self):
        opener, urls = self.fetch_fixture(converter_body="import sys\nprint('converter transcript line')\nsys.exit(7)\n")
        rc, text = self.fetch(os.path.join(self.tmp, "badconv"), opener)
        self.assertEqual(rc, 1, text); self.assertIn("WEIGHTS CONVERT FAILED rc=7", text); self.assertNotIn("WEIGHTS OK", text)

    def test_fetch_needs_the_jax_interpreter_and_the_fork_checkout(self):
        opener, urls = self.fetch_fixture()
        os.environ["AF3_TORCH_JAX_REPO"] = os.path.join(self.tmp, "no_such_checkout")
        rc, text = self.fetch(os.path.join(self.tmp, "norepo"), opener)
        self.assertEqual((rc, urls), (1, []), text); self.assertIn("WEIGHTS CONVERT FAILED: convert_of3_weights.py not found under AF3_TORCH_JAX_REPO", text)
        os.environ.pop("AF3_TORCH_JAX_PY")
        rc, text = self.fetch(os.path.join(self.tmp, "nopy"), opener)
        self.assertEqual((rc, urls), (1, []), text); self.assertIn("AF3_TORCH_JAX_PY (not set) is not an executable interpreter", text)

    def test_without_fetch_nothing_is_downloaded(self):
        opener, urls = self.fetch_fixture()
        d = os.path.join(self.tmp, "plain"); os.makedirs(d)
        rc, text = self.check(d)
        self.assertEqual((rc, urls), (1, []), text); self.assertIn("WEIGHTS MISSING", text); self.assertIn("nothing was downloaded", text); self.assertIn("--fetch", text)


class UpstreamArchiveStep(unittest.TestCase):
    """stock/fetch_upstream.py — the install step that lays xfold's pinned archive under stock/ — on a scratch copy of the script beside a synthetic
    PINS.json and synthetic archives: present (judged, nothing fetched), absent (fetched through an injected opener, judged, placed atomically),
    --archive FILE (an offline copy), and the refusals (wrong root, no CLI member, not a tar, a failed fetch). No network."""
    COMMIT = "0123456789abcdef0123456789abcdef01234567"

    def setUp(self):
        import importlib.util, shutil
        self.tmp = tempfile.mkdtemp(prefix="af3t_fetch_upstream_")
        self.stock = os.path.join(self.tmp, "stock"); os.makedirs(self.stock)
        shutil.copy(os.path.join(TREE, "stock", "fetch_upstream.py"), self.stock)
        self.good = self.targz({f"xfold-{self.COMMIT}/run_alphafold.py": b"print('cli')\n", f"xfold-{self.COMMIT}/xfold/__init__.py": b""})
        json.dump({"upstream": {"name": "xfold", "repo": "https://example.invalid/u/xfold", "commit": self.COMMIT,
                                "archive": {"file": "xfold-0123456.tar.gz", "bytes": len(self.good)}}}, open(os.path.join(self.stock, "PINS.json"), "w"))
        spec = importlib.util.spec_from_file_location("fetch_upstream_under_test", os.path.join(self.stock, "fetch_upstream.py"))
        self.fu = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.fu)
        self.dest = os.path.join(self.stock, "xfold-0123456.tar.gz")

    def listing(self):
        """The scratch stock dir's entries, the interpreter's own bytecode cache aside (present when bytecode writing is on)."""
        return sorted(n for n in os.listdir(self.stock) if n != "__pycache__")

    def targz(self, members):
        import tarfile
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, data in members.items():
                ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
        return buf.getvalue()

    def run_step(self, argv=(), payload=None, fail=None):
        urls = []

        class Opener:
            def __init__(s_, url):
                urls.append(url)
                if fail: raise fail
                s_.b = io.BytesIO(payload)
            def __enter__(s_): return s_.b
            def __exit__(s_, *a): return False
        out = io.StringIO(); rc = self.fu.main(list(argv), opener=Opener, out=out)
        return rc, out.getvalue().strip(), urls

    def test_present_is_judged_not_fetched(self):
        open(self.dest, "wb").write(self.good)
        rc, text, urls = self.run_step(payload=b"unused")
        self.assertEqual((rc, urls), (0, []), text)
        self.assertRegex(text, r"^\[af3-torch-opt install\] UPSTREAM ARCHIVE present file=stock/xfold-0123456\.tar\.gz members=2 root=xfold-%s/ %d bytes = stock/PINS\.json upstream\.archive\.bytes$" % (self.COMMIT, len(self.good)))

    def test_absent_is_fetched_from_the_pinned_commit_and_placed(self):
        rc, text, urls = self.run_step(payload=self.good)
        self.assertEqual(rc, 0, text)
        self.assertEqual(urls, [f"https://example.invalid/u/xfold/archive/{self.COMMIT}.tar.gz"])          # <repo>/archive/<commit>.tar.gz, nothing else
        self.assertIn("UPSTREAM ARCHIVE fetched url=https://example.invalid/u/xfold/archive/", text)
        self.assertEqual(open(self.dest, "rb").read(), self.good)
        self.assertEqual(self.listing(), ["PINS.json", "fetch_upstream.py", "xfold-0123456.tar.gz"])                # no .part / scratch left behind

    def test_another_framing_of_the_same_tree_is_reported_not_refused(self):
        regz = self.targz({f"xfold-{self.COMMIT}/run_alphafold.py": b"print('cli')\n", f"xfold-{self.COMMIT}/xfold/__init__.py": b"", f"xfold-{self.COMMIT}/pad": b"x" * 300})
        rc, text, _ = self.run_step(payload=regz)
        self.assertEqual(rc, 0, text); self.assertIn(f"stock/PINS.json upstream.archive.bytes says {len(self.good)}", text)

    def test_archive_flag_places_an_offline_copy(self):
        src = os.path.join(self.tmp, "downloaded.tgz"); open(src, "wb").write(self.good)
        rc, text, urls = self.run_step(["--archive", src])
        self.assertEqual((rc, urls), (0, []), text); self.assertIn("UPSTREAM ARCHIVE placed from=", text)
        self.assertEqual(open(self.dest, "rb").read(), self.good)
        os.remove(self.dest)
        self.assertEqual(self.run_step(["--archive", os.path.join(self.tmp, "nope.tgz")])[0], 1)      # a missing --archive file is a refusal, not a fetch

    def test_refusals_place_nothing(self):
        cases = {"wrong root": self.targz({"other-dir/run_alphafold.py": b"x"}),
                 "no cli": self.targz({f"xfold-{self.COMMIT}/README.md": b"x"}),
                 "not a tar": b"\x1f\x8b not really"}
        for why, payload in cases.items():
            rc, text, _ = self.run_step(payload=payload)
            self.assertEqual(rc, 1, (why, text)); self.assertIn("UPSTREAM ARCHIVE REFUSED", text, why); self.assertIn("nothing was placed under stock/", text, why)
            self.assertFalse(os.path.exists(self.dest), why)
            self.assertEqual(self.listing(), ["PINS.json", "fetch_upstream.py"], why)
        open(self.dest, "wb").write(b"truncated"); rc, text, _ = self.run_step()
        self.assertEqual(rc, 1, text); self.assertIn("UPSTREAM ARCHIVE REFUSED file=stock/xfold-0123456.tar.gz", text)   # a bad file already in place is named, not overwritten

    def test_a_failed_fetch_names_the_url_and_the_offline_route(self):
        import urllib.error
        rc, text, urls = self.run_step(fail=urllib.error.URLError("name resolution failed"))
        self.assertEqual(rc, 1, text)
        self.assertIn(f"UPSTREAM ARCHIVE FETCH FAILED url=https://example.invalid/u/xfold/archive/{self.COMMIT}.tar.gz", text); self.assertIn("--archive FILE", text)
        self.assertFalse(os.path.exists(self.dest))

    def test_the_tree_names_the_same_route(self):
        """The kit's real PINS: the URL the step fetches is GitHub's archive of upstream.commit, the words STOCK.md and check_pins.py print."""
        real = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        spec = __import__("importlib.util").util.spec_from_file_location("fetch_upstream_real", os.path.join(TREE, "stock", "fetch_upstream.py"))
        fu = __import__("importlib.util").util.module_from_spec(spec); spec.loader.exec_module(fu)
        self.assertEqual(fu.source_url(real), f"{real['upstream']['repo']}/archive/{real['upstream']['commit']}.tar.gz")
        self.assertTrue(real["upstream"]["archive"]["recipe"].startswith(fu.source_url(real)))
        self.assertIn("stock/fetch_upstream.py", real["upstream"]["archive"]["install"])


class EnvironmentRecipe(unittest.TestCase):
    """environment/ (Dockerfile, the two locks, apptainer.def) and .gitignore agree with stock/PINS.json: the fork the build fetches, patches and compiles
    (or takes prebuilt) is the one the pins name, the locks pin the packages check_pins.py checks at the same versions, the interpreters the image's variables name are the
    ones the build makes."""
    @classmethod
    def setUpClass(cls):
        cls.dockerfile = open(os.path.join(ENV_DIR, "Dockerfile"), encoding="utf-8").read()
        cls.lock_torch = open(os.path.join(ENV_DIR, "requirements-torch.lock"), encoding="utf-8").read()
        cls.lock_jax = open(os.path.join(ENV_DIR, "requirements-jax.lock"), encoding="utf-8").read()
        cls.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))

    @staticmethod
    def pinned(lock):
        return {m.group(1).lower(): m.group(2) for m in (re.match(r"^([A-Za-z0-9_.\-]+)==(\S+)$", l) for l in lock.splitlines()) if m}

    def test_environment_members(self):
        self.assertEqual(sorted(os.listdir(ENV_DIR)), ["Dockerfile", "apptainer.def", "requirements-jax.lock", "requirements-torch.lock"])

    def test_the_two_routes_to_the_fork(self):
        ref = self.pins["reference"]; art = ref["artefacts"]
        self.assertEqual(art["dir"], "stock/wheels")
        self.assertRegex(self.dockerfile, r"(?m)^ARG WHEELS_FROM=build$")                                # from source by default …
        self.assertRegex(self.dockerfile, r"(?m)^ARG BUILD_JOBS=0$"); self.assertIn("nproc --all", self.dockerfile)
        self.assertIn(f"git clone --quiet {ref['repo']} /app/alphafold", self.dockerfile)                # … the fork's tree at the pinned commit …
        self.assertIn(f"git checkout --quiet {ref['commit']}", self.dockerfile)
        self.assertIn(f"COPY af3_torch/stock/{ref['patch']['file']} ", self.dockerfile.replace("af3_torch/stock/PINS.json ", ""))   # … with the one patch the pins name
        self.assertTrue(os.path.isfile(os.path.join(TREE, "stock", ref["patch"]["file"])))
        self.assertIn(f"patch -p1 < /tmp/stock/{os.path.basename(ref['patch']['file'])}", self.dockerfile)
        self.assertIn("COPY af3_torch/stock/wheels/ /tmp/wheels/", self.dockerfile)                       # the prebuilt route reads the directory the pins name …
        self.assertIn('["reference"]["artefacts"]["files"]', self.dockerfile)                            # … and checks its files against the pins
        names = {f["file"] for f in art["files"]}
        self.assertEqual(names, {"alphafold3_open-3.1.4-cp312-cp312-linux_x86_64.whl", "alphafold3-build_data.tar.gz"})
        for f in art["files"]:
            self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$"); self.assertGreater(f["bytes"], 0); self.assertIn("/tmp/wheels/" + f["file"], self.dockerfile)
        ignore = open(os.path.join(TREE, ".gitignore")).read().split()
        self.assertIn("stock/wheels/*", ignore); self.assertIn("!stock/wheels/.keep", ignore)              # the artefacts stay out of git, the directory stays in …
        self.assertTrue(os.path.isfile(os.path.join(TREE, "stock", "wheels", ".keep")))
        tracked = subprocess.run(["git", "ls-files", "stock/wheels"], cwd=TREE, capture_output=True, text=True)
        if tracked.returncode == 0: self.assertEqual(tracked.stdout.split(), ["stock/wheels/.keep"])        # … and nothing else there is tracked (checked when the tree is a git checkout)

    def test_the_locks_pin_what_the_pin_check_checks(self):
        t, j = self.pinned(self.lock_torch), self.pinned(self.lock_jax)
        for name, version in self.pins["check_packages"]["torch_python"].items(): self.assertEqual(t.get(name), version, name)
        for name, version in self.pins["check_packages"]["jax_python"].items(): self.assertEqual(j.get(name), version, name)
        self.assertEqual(t["numpy"], j["numpy"])                                                     # the stock venv composer overlays the two: one numpy (STOCK.md)
        wheel = [f["file"] for f in self.pins["reference"]["artefacts"]["files"] if f["file"].endswith(".whl")]
        self.assertEqual(len(wheel), 1); name, version = wheel[0].split("-")[:2]; dist = name.replace("_", "-")
        self.assertIsNone(j.get(dist))                                                                # the wheel's distribution is on no package index: the lock carries NO installable line for it (`pip install -r` of the lock as it is asks no index for that name) …
        installable = [l for l in self.lock_jax.splitlines() if l.strip() and not l.startswith("#")]
        self.assertFalse([l for l in installable if re.match(rf"(?i)^{re.escape(dist)}\b|^{re.escape(name)}\b", l)], installable)
        self.assertRegex(self.lock_jax, rf"(?m)^# {re.escape(dist)}=={re.escape(version)} ")            # … its entry is a comment at the wheel's version, so the list still states the whole environment …
        self.assertIn(f"grep -v -E '^(#|{dist}==)'", self.dockerfile)                                  # … and the build's filter drops it either way before the -r pass; the wheel file is installed after it
        self.assertNotIn("@ file:", self.lock_torch + self.lock_jax)                                  # no machine-local paths in a lock

    def test_the_image_variables_name_what_the_build_makes(self):
        instructions = re.sub(r"\\\n", " ", "\n".join(l for l in self.dockerfile.splitlines() if not l.lstrip().startswith("#"))).splitlines()
        env = dict(kv for ins in instructions if ins.startswith("ENV ") for kv in re.findall(r"\b(AF3_TORCH_[A-Z_]+)=(\S+)", ins))
        self.assertEqual(set(env), {"AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO"})          # the weights directory (AF3_TORCH_PARAMS_DIR) is never baked in
        for var, venv in (("AF3_TORCH_PY", "/torch_venv"), ("AF3_TORCH_JAX_PY", "/alphafold3_venv")):
            self.assertEqual(env[var], venv + "/bin/python"); self.assertRegex(self.dockerfile, r"uv venv [^\n]*" + re.escape(venv) + r"\b")
        self.assertEqual(env["AF3_TORCH_JAX_REPO"], "/app/alphafold"); self.assertIn(" /app/alphafold", self.dockerfile)
        self.assertIn("bash run.sh install", self.dockerfile)                                          # the image installs the kit by the kit's own step
        self.assertRegex(open(os.path.join(ENV_DIR, "apptainer.def")).read(), r"(?m)^From: af3_torch-kit:dev$")


if __name__ == "__main__":
    unittest.main()
