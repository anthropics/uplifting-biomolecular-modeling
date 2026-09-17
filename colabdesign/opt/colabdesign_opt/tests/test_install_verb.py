"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the fetch step's digest gate (stock/check_pins.py `fetch` on a copy of the pins pointed at a local file:// mirror: placed, kept,
refused by name, failed by name; no network, no upstream), the weights step's archive fetch and digest gate (colabdesign_opt.weights.fetch with an
injected archive, opener and pin table), and the environment recipe's agreement with stock/PINS.json (environment/: the conda lock, the pip
lock, the Dockerfile, the Apptainer definition). CPU only."""
import hashlib
import importlib.util
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tarfile
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
ENV_DIR = os.path.join(TREE, "environment")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_FETCH / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py --fetch"*) exit "${STUB_RC_FETCH:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m colabdesign_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="colabdesign_install_verb_")
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

    def test_install_runs_pip_then_the_fetch_step_then_the_software_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(len(calls), 3, calls)
        self.assertEqual(calls[0], f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt")
        self.assertEqual(calls[1], f"-I {TREE}/stock/check_pins.py --fetch")            # BindCraft's DSSP executable: fetched from the pinned commit, sha256-checked, before the pin check
        self.assertEqual(calls[2], f"-I {TREE}/stock/check_pins.py --no-gpu")            # the installed software, not the box: no GPU at image build or in route C's shell

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/af2"], ["install", "--weights=/data/af2"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[3:], ["-m colabdesign_opt.weights /data/af2"], calls)
            self.assertEqual(len(calls), 4, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--json"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "fast"], ["install", "--config", "h100"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], fetch=1)                # the DSSP executable not in place as pinned: the install stops, named
        self.assertEqual((rc, len(calls)), (1, 2), (out, calls)); self.assertIn("DSSP executable is not in place as pinned", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 3), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 4), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py --fetch", f"-I {TREE}/stock/check_pins.py --no-gpu", "-m colabdesign_opt.weights /w"], calls)   # fetch step (keeps the file in place), pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)


CHECK_PINS = os.path.join(TREE, "stock", "check_pins.py")
FETCHED_KEY = "src/bindcraft/functions/dssp"                              # stock/PINS.json upstream.bindcraft.fetched.files: the one install-fetched file


def _check_pins():
    spec = importlib.util.spec_from_file_location("check_pins_under_test", CHECK_PINS)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class FetchStepGate(unittest.TestCase):
    """stock/check_pins.py `fetch` (run.sh install's `check_pins.py --fetch`) on a scratch kit directory whose PINS.json is the tree's own with the
    fetched file's URL re-pointed at a local file:// mirror: an absent file is fetched, sha256-checked, placed with mode 755 and kept on the next
    run; a mirror serving other bytes is refused by name and nothing is placed; a present file off its pin is refused by name and left in place;
    an unreachable source fails by name; a pinned file that lost its executable bit gets it back. `check_fetched` reports the same states."""

    PAYLOAD = b"\x7fELF stand-in for BindCraft's DSSP executable\n" * 64

    def setUp(self):
        self.cp = _check_pins()
        self.tmp = tempfile.mkdtemp(prefix="colabdesign_fetch_step_")
        self.home = os.path.join(self.tmp, "kit"); os.makedirs(os.path.join(self.home, "stock"))
        self.mirror = os.path.join(self.tmp, "mirror"); os.makedirs(self.mirror)
        self.served = os.path.join(self.mirror, "dssp")
        with open(self.served, "wb") as fh: fh.write(self.PAYLOAD)
        self.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        rec = self.pins["upstream"]["bindcraft"]["fetched"]["files"][FETCHED_KEY]
        self.upstream_url = rec["url"]
        rec.update(url="file://" + self.served, sha256=hashlib.sha256(self.PAYLOAD).hexdigest(), bytes=len(self.PAYLOAD))
        self.rec = rec
        self.dest = os.path.join(self.home, "stock", FETCHED_KEY)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def fetch(self, **kw):
        out = io.StringIO(); rc = self.cp.fetch(self.pins, home=self.home, out=out, **kw); return rc, out.getvalue()

    def test_the_tree_pins_one_fetched_file_at_the_upstream_commit(self):
        files = self.cp.fetched_files(json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8")))
        self.assertEqual([k for k, _ in files], [FETCHED_KEY])
        commit = self.pins["upstream"]["bindcraft"]["commit"]
        self.assertEqual(self.upstream_url, f"https://raw.githubusercontent.com/martinpacesa/BindCraft/{commit}/functions/dssp")

    def test_absent_is_fetched_checked_placed_executable_then_kept(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn("placed", out); self.assertIn("FETCH OK: 1/1", out)
        self.assertEqual(open(self.dest, "rb").read(), self.PAYLOAD)
        self.assertEqual(stat.S_IMODE(os.stat(self.dest).st_mode), 0o755); self.assertTrue(os.access(self.dest, os.X_OK))
        self.assertFalse(os.path.exists(self.dest + ".part"))
        os.remove(self.served)                                                          # the second run touches no source: the file is in place and pinned
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn("present", out); self.assertNotIn("fetching", out)
        self.assertEqual(self.cp.check_fetched(self.pins, home=self.home)[1][FETCHED_KEY]["status"], "pinned")

    def test_other_bytes_served_are_refused_by_name_and_nothing_is_placed(self):
        with open(self.served, "ab") as fh: fh.write(b"tampered upstream")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED stock/" + FETCHED_KEY, out); self.assertIn("not the pinned", out); self.assertIn("FETCH REFUSED: 1/1", out)
        self.assertFalse(os.path.exists(self.dest)); self.assertFalse(os.path.exists(self.dest + ".part"))
        self.assertEqual(self.cp.check_fetched(self.pins, home=self.home)[1][FETCHED_KEY]["status"], "absent")

    def test_a_present_file_off_its_pin_is_refused_by_name_and_left_in_place(self):
        os.makedirs(os.path.dirname(self.dest)); open(self.dest, "wb").write(b"someone else's dssp"); os.chmod(self.dest, 0o755)
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED stock/" + FETCHED_KEY, out); self.assertIn("left in place", out); self.assertNotIn("fetching", out)
        self.assertEqual(open(self.dest, "rb").read(), b"someone else's dssp")                  # never overwritten, never re-fetched
        d = self.cp.check_fetched(self.pins, home=self.home)[1][FETCHED_KEY]
        self.assertEqual(d["status"], "not_pinned"); self.assertTrue(self.cp.fetched_lines({FETCHED_KEY: d})[1])   # ONE note line for the pin check's report

    def test_an_unreachable_source_fails_by_name_and_places_nothing(self):
        def refuse(url): raise OSError(f"unreachable: {url}")
        rc, out = self.fetch(opener=refuse)
        self.assertEqual(rc, 1, out); self.assertIn("FAILED  stock/" + FETCHED_KEY, out); self.assertIn("By hand: fetch file://", out)
        self.assertFalse(os.path.exists(self.dest)); self.assertFalse(os.path.exists(self.dest + ".part"))
        self.rec["url"] = "file://" + os.path.join(self.mirror, "no_such_file")                   # urllib's own error for a missing source: the same named failure
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("FAILED", out); self.assertFalse(os.path.exists(self.dest))

    def test_a_pinned_file_without_its_executable_bit_gets_it_back(self):
        os.makedirs(os.path.dirname(self.dest)); open(self.dest, "wb").write(self.PAYLOAD); os.chmod(self.dest, 0o644)
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertEqual(stat.S_IMODE(os.stat(self.dest).st_mode), 0o755)

    def test_the_pin_check_reports_never_refuses(self):
        bad, detail = self.cp.check_fetched(self.pins, home=self.home)                        # absent: reported, not a finding
        self.assertEqual((bad, detail[FETCHED_KEY]["status"]), ([], "absent"))
        ok, notes = self.cp.fetched_lines(detail)
        self.assertEqual(ok, []); self.assertEqual(len(notes), 1); self.assertIn("absent", notes[0]); self.assertIn("run.sh install", notes[0])
        self.fetch()
        ok, notes = self.cp.fetched_lines(self.cp.check_fetched(self.pins, home=self.home)[1])
        self.assertEqual(notes, []); self.assertRegex(ok[0], r"^fetched file stock/src/bindcraft/functions/dssp: sha256=[0-9a-f]{12} \(pinned\)$")

    def test_run_sh_names_the_step_and_gitignore_keeps_the_file_out_of_the_tree(self):
        sh = open(RUN_SH, encoding="utf-8").read()
        self.assertIn('"$HERE/stock/check_pins.py" --fetch ||', sh); self.assertLess(sh.index("check_pins.py\" --fetch"), sh.index("check_pins.py\" --no-gpu ||"))
        ignored = open(os.path.join(TREE, ".gitignore"), encoding="utf-8").read().split()
        self.assertIn("stock/" + FETCHED_KEY, ignored); self.assertIn("stock/" + FETCHED_KEY + ".part", ignored)


def _tar_bytes(members):
    """An in-memory tar archive: {member path: bytes}."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, data in members.items():
            info = tarfile.TarInfo(path); info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class WeightsArchiveGate(unittest.TestCase):
    """colabdesign_opt.weights.fetch: the missing files streamed out of the archive, then every file's sha256 against the pin — a mismatch, or a
    member the archive lacks, fails by name; present files are kept and checked, never re-fetched."""

    SOURCE = "https://upstream.example/alphafold_params.tar"

    def setUp(self):
        from colabdesign_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="colabdesign_weights_")
        self.payload = {f"params_model_{k}_multimer_v3.npz": f"multimer-{k}".encode() * k for k in (1, 2, 3)}
        self.files = {n: {"sha256": hashlib.sha256(b).hexdigest(), "bytes": len(b)} for n, b in self.payload.items()}
        extra = {"./params_model_1.npz": b"monomer weights: not wanted", "./LICENSE": b"cc-by"}
        self.archive = _tar_bytes({**extra, **{f"./{n}": b for n, b in self.payload.items()}})
        self.opened = []

    def opener(self, url):
        self.opened.append(url)
        return io.BytesIO(self.archive)

    def matcher(self, root):
        """stock/check_pins.py check_weights' contract over the test's pin table: (bad, {name: "missing" | {sha256, bytes, pinned, pinned_sha256, pinned_bytes}})."""
        bad, detail = [], {}
        for n, w in self.files.items():
            p = os.path.join(root, "params", n)
            if not os.path.isfile(p):
                detail[n] = "missing"; bad.append(f"weights: {p} missing"); continue
            got = hashlib.sha256(open(p, "rb").read()).hexdigest(); size = os.path.getsize(p)
            detail[n] = {"sha256": got, "bytes": size, "pinned": got == w["sha256"] and size == w["bytes"], "pinned_sha256": w["sha256"], "pinned_bytes": w["bytes"]}
        return bad, detail

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), source=kw.pop("source", self.SOURCE), opener=kw.pop("opener", self.opener),
                          matcher=kw.pop("matcher", self.matcher), out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 3/3", out); self.assertIn(f"export COLABDESIGN_PARAMS_DIR={self.dir}", out.splitlines()[-1])
        self.assertEqual(self.opened, [self.SOURCE])                                        # one stream for every missing file
        self.assertEqual(sorted(os.listdir(os.path.join(self.dir, "params"))), sorted(self.payload))   # the pinned members only: no monomer file, no LICENSE, no .part
        for n, b in self.payload.items():
            self.assertEqual(open(os.path.join(self.dir, "params", n), "rb").read(), b)

    def test_present_files_are_kept_and_checked_without_a_fetch(self):
        os.makedirs(os.path.join(self.dir, "params"))
        for n, b in self.payload.items():
            with open(os.path.join(self.dir, "params", n), "wb") as f: f.write(b)
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(self.opened, [])                                                   # nothing missing: the archive is never opened
        self.assertEqual(out.count("present  params/"), 3, out)

    def test_only_the_missing_members_are_written(self):
        os.makedirs(os.path.join(self.dir, "params"))
        keep = "params_model_2_multimer_v3.npz"
        with open(os.path.join(self.dir, "params", keep), "wb") as f: f.write(self.payload[keep])
        mtime = os.stat(os.path.join(self.dir, "params", keep)).st_mtime_ns
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertEqual(os.stat(os.path.join(self.dir, "params", keep)).st_mtime_ns, mtime)   # the present file was not rewritten
        self.assertEqual(out.count("extracting "), 2, out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "params", "params_model_1_multimer_v3.npz"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1/3", out.splitlines()[-1]); self.assertIn("params_model_1_multimer_v3.npz", out.splitlines()[-1])
        self.assertEqual(open(p, "rb").read(), b"other bytes")                              # left in place for inspection

    def test_a_member_the_archive_lacks_fails_by_name(self):
        files = dict(self.files); files["params_model_9_multimer_v3.npz"] = {"sha256": "0" * 64, "bytes": 1}
        rc, out = self.fetch(files=files)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED: 1/4", out.splitlines()[-1]); self.assertIn("params_model_9_multimer_v3.npz", out)
        self.assertFalse([f for f in os.listdir(os.path.join(self.dir, "params")) if f.endswith(".part")])

    def test_the_defaults_are_the_pins(self):
        """Without injected arguments the file table and the archive URL are stock/PINS.json `weights` (read through cli.check_pins_module) — asserted
        here without opening the URL: every pinned file present and correct needs no fetch, and a wrong byte is refused against PINS' digests."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["weights"]
        os.makedirs(os.path.join(self.dir, "params"))
        for n in pins["files"]:
            with open(os.path.join(self.dir, "params", n), "wb") as f: f.write(b"x")
        out = io.StringIO()
        def refuse_open(url): raise AssertionError(f"opened {url}: every pinned file is present")
        rc = self.W.fetch(self.dir, opener=refuse_open, out=out)
        self.assertEqual(rc, 1, out.getvalue())
        self.assertIn(f"REFUSED: {len(pins['files'])}/{len(pins['files'])}", out.getvalue().splitlines()[-1])
        self.assertTrue(pins["source"].startswith("https://storage.googleapis.com/alphafold/") and pins["source"].endswith(".tar"), pins["source"])


class EnvironmentRecipe(unittest.TestCase):
    """environment/ is ONE environment definition and it is the pinned stack's: the conda lock carries PINS `python` and every PINS `pins` entry that
    is a conda package at the pinned version (jax-cuda12-plugin / -pjrt travel inside conda-forge's CUDA build of jaxlib; colabdesign is the pip
    layer's), the pip lock carries PINS `upstream.colabdesign.install` verbatim, the Dockerfile builds on the micromamba base, installs both locks
    as they stand, and runs `run.sh install`; the Apptainer definition converts that image."""

    def setUp(self):
        self.pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        self.conda = [l.strip() for l in open(os.path.join(ENV_DIR, "conda-linux-64.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        self.req = [l.strip() for l in open(os.path.join(ENV_DIR, "requirements.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]
        self.dockerfile = open(os.path.join(ENV_DIR, "Dockerfile"), encoding="utf-8").read()

    def conda_pkg(self, name):
        """(version, build) of the conda lock's package `name` (None when absent)."""
        for line in self.conda:
            m = re.match(r"^https://conda\.anaconda\.org/conda-forge/(?:linux-64|noarch)/(.+)-([^-]+)-([^-]+)\.(?:conda|tar\.bz2)#[0-9a-f]{32}$", line)
            if m and m.group(1) == name:
                return m.group(2), m.group(3)
        return None

    def test_the_conda_lock_is_explicit_and_carries_the_pinned_stack(self):
        self.assertEqual(self.conda[0], "@EXPLICIT")
        self.assertTrue(all(re.match(r"^https://conda\.anaconda\.org/conda-forge/(linux-64|noarch)/\S+#[0-9a-f]{32}$", l) for l in self.conda[1:]), "every entry = a conda-forge URL with its md5")
        self.assertEqual(self.conda_pkg("python")[0], self.pins["python"])
        for name, version in self.pins["pins"].items():
            if name == "colabdesign" or name.startswith("jax-cuda12-"):
                continue
            self.assertEqual((name, (self.conda_pkg(name) or (None,))[0]), (name, version))
        self.assertIn("cuda", self.conda_pkg("jaxlib")[1])                                   # conda-forge's CUDA build: the jax-cuda12-plugin / -pjrt entries PINS names are inside it
        self.assertEqual(self.conda_pkg("cuda-version")[0].split(".")[0], "12")

    def test_the_pip_lock_is_the_stock_pin_and_freesasa(self):
        up = self.pins["upstream"]["colabdesign"]
        self.assertIn(up["install"], self.req)                                                # the freeze's VCS line, verbatim (pip's record of the pinned commit)
        self.assertEqual(sorted(l.split("==")[0].split(" @ ")[0] for l in self.req), ["colabdesign", "freesasa"])

    def test_the_dockerfile_builds_the_pinned_stack(self):
        df = self.dockerfile
        self.assertRegex(df, r"(?m)^FROM mambaorg/micromamba:2\.1\.1$")
        self.assertRegex(df, r"(?m)^USER root$")
        self.assertIn("COPY colabdesign/environment/conda-linux-64.lock ", df); self.assertIn("micromamba install -y -n base -f /tmp/conda-linux-64.lock", df)
        self.assertIn("COPY colabdesign/environment/requirements.lock ", df); self.assertIn("pip install --no-cache-dir --no-deps -r /tmp/requirements.lock", df)   # the lock as it stands: the VCS line clones the pin
        self.assertRegex(df, r"apt-get install [^\n]*\bgit\b")                                # pip's VCS install needs git
        self.assertRegex(df, r"(?m)^RUN bash run.sh install$")
        self.assertRegex(df, r"(?m)^\s*LD_LIBRARY_PATH=/opt/conda/lib\b"); self.assertRegex(df, r"(?m)^\s*PYTHONHASHSEED=0\b")
        self.assertNotRegex("\n".join(l for l in df.splitlines() if not l.startswith("#")), r"COLABDESIGN_PARAMS_DIR=")   # the weights root is named at run time, never baked (an ENV line would set it; the header's docker-run example is a comment)
        self.assertRegex(open(os.path.join(ENV_DIR, "apptainer.def"), encoding="utf-8").read(), r"(?m)^From: colabdesign-kit:dev$")

    def test_environment_holds_the_definition_only(self):
        self.assertEqual(sorted(os.listdir(ENV_DIR)), ["Dockerfile", "apptainer.def", "conda-linux-64.lock", "requirements.lock"])


if __name__ == "__main__":
    unittest.main()
