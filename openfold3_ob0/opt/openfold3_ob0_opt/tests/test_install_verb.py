"""`run.sh install [--weights DIR] [--ccd FILE] [--wheel FILE] [--src TARBALL]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation;
nothing is installed), and the two fetched-input steps of openfold3_ob0_opt.weights with injected upstream calls and the real pin table
(stock/PINS.json): the CCD placement gate — upstream's fetch, or a file fetched beforehand (`--ccd FILE`: judged against the pin, then copied,
no fetch) — and the checkpoint digest gate. No network, no upstream import, CPU only."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest

from openfold3_ob0_opt.tests import _stubs

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG (newlines folded); exit codes per call kind by STUB_RC_PROBE / _PIP / _UPSTREAM / _PINS / _CCD / _WEIGHTS
printf '%s\\n' "$(echo $*)" >> "$STUB_LOG"
case "$*" in
  *find_spec*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"install_upstream.py"*) exit "${STUB_RC_UPSTREAM:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m openfold3_ob0_opt.weights --ccd"*) exit "${STUB_RC_CCD:-0}" ;;
  *"-m openfold3_ob0_opt.weights "*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="ob0_install_verb_")
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
        self.probes = [c for c in lines if "find_spec" in c]                              # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if "find_spec" not in c]

    def test_install_runs_pip_then_upstream_then_the_pin_check_then_the_ccd(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/install_upstream.py", f"-I {TREE}/stock/check_pins.py", "-m openfold3_ob0_opt.weights --ccd"], calls)
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/openfold3"], ["install", "--weights=/data/openfold3"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[4:], ["-m openfold3_ob0_opt.weights /data/openfold3"], calls)
            self.assertEqual(len(calls), 5, calls)

    def test_ccd_file_is_handed_to_the_ccd_step(self):
        """`--ccd FILE` / `--ccd=FILE`: the CCD step receives the file (openfold3_ob0_opt.weights --ccd FILE: judged against the pin, then copied — no fetch);
        the sequence and the --weights step are otherwise unchanged."""
        for args, tail in ((["install", "--ccd", "/x/components.bcif"], []), (["install", "--ccd=/x/components.bcif"], []),
                           (["install", "--weights", "/w", "--ccd", "/x/components.bcif"], ["-m openfold3_ob0_opt.weights /w"]),
                           (["install", "--ccd=/x/components.bcif", "--weights=/w"], ["-m openfold3_ob0_opt.weights /w"])):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, (args, out))
            self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/install_upstream.py", f"-I {TREE}/stock/check_pins.py", "-m openfold3_ob0_opt.weights --ccd /x/components.bcif"] + tail, (args, calls))
        rc, out, calls = self.run_sh(["install", "--ccd", "/x/wrong.bcif"], ccd=1)          # the step's refusal (a digest off the pin) stops the sequence with its code
        self.assertEqual((rc, calls[-1]), (1, "-m openfold3_ob0_opt.weights --ccd /x/wrong.bcif"), (out, calls))

    def test_wheel_and_src_copies_are_handed_to_the_upstream_step(self):
        """`--wheel FILE` / `--src TARBALL` (copies fetched beforehand, for a machine without network access): stock/install_upstream.py receives them —
        `--wheel` before `--src` whatever the order given — and the rest of the sequence is unchanged."""
        for args, up in ((["install", "--wheel", "/x/of3.whl"], "--wheel /x/of3.whl"), (["install", "--wheel=/x/of3.whl"], "--wheel /x/of3.whl"),
                         (["install", "--src", "/x/src.tar.gz"], "--src /x/src.tar.gz"), (["install", "--src=/x/src.tar.gz", "--wheel", "/x/of3.whl"], "--wheel /x/of3.whl --src /x/src.tar.gz"),
                         (["install", "--ccd", "/x/components.bcif", "--src", "/x/src.tar.gz", "--wheel=/x/of3.whl"], "--wheel /x/of3.whl --src /x/src.tar.gz")):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, (args, out))
            self.assertEqual(calls[:3], [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/install_upstream.py" + " " + up, f"-I {TREE}/stock/check_pins.py"], (args, calls))
        rc, out, calls = self.run_sh(["install", "--wheel", "/x/wrong.whl"], upstream=1)   # the step's refusal (a digest off the pin) stops the sequence with its code
        self.assertEqual((rc, calls[-1]), (1, f"-I {TREE}/stock/install_upstream.py" + " --wheel /x/wrong.whl"), (out, calls))

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--det"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--n_gpu", "2"],
                     ["install", "--ccd"], ["install", "--ccd="], ["install", "--ccd", "--weights", "/w"],
                     ["install", "--wheel"], ["install", "--wheel="], ["install", "--src"], ["install", "--src="], ["install", "--src", "--wheel", "/x.whl"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls)); self.assertEqual(self.probes, [], (args, self.probes))
            self.assertIn("run.sh", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        for knob, code, ncalls in (("pip", 1, 1), ("upstream", 1, 2), ("pins", 3, 3), ("ccd", 1, 4), ("weights", 1, 5)):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(["install", "--weights", "/w"], **{knob: code if knob != "pins" else 3})
            self.assertEqual((rc, len(calls)), (code, ncalls), (knob, out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check, the CCD
        (present at its pin: only hashed) and --weights — a read-only image cannot re-run pip; the upstream step runs and finds the wheel, the install and
        stock/src present. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/install_upstream.py", f"-I {TREE}/stock/check_pins.py", "-m openfold3_ob0_opt.weights --ccd", "-m openfold3_ob0_opt.weights /w"], calls)
        self.assertEqual(len(self.probes), 1, self.probes)


def _pins():
    import json
    return json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))


class CcdPlacementGate(unittest.TestCase):
    """openfold3_ob0_opt.weights.ccd: a file at the pin is only hashed; anything else runs upstream's placement once, then the digest decides."""

    def setUp(self):
        from openfold3_ob0_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="ob0_ccd_")
        self.path = os.path.join(self.dir, "components.bcif")
        self.pin = _pins()["ccd"]
        self.placed = []

    def hasher(self, p):                                      # the pin's digest for the stand-in 'pinned' bytes, the real sha256 otherwise
        b = open(p, "rb").read()
        return self.pin["sha256"] if b == b"P" * 7 else hashlib.sha256(b).hexdigest()

    def located(self, writes=b""):
        def place():
            self.placed.append(self.path)
            with open(self.path, "wb") as f: f.write(writes)
        return lambda: (self.path, place)

    def run_ccd(self, **kw):
        out = io.StringIO()
        size = os.path.getsize
        self.W.os.path.getsize = lambda p: self.pin["bytes"] if open(p, "rb").read() == b"P" * 7 else size(p)   # the stand-in's size reads as the pin's
        try:
            rc = self.W.ccd(home=TREE, located=kw.pop("located", self.located(b"P" * 7)), hasher=self.hasher, out=out, source=kw.pop("source", None))
        finally:
            self.W.os.path.getsize = size
        return rc, out.getvalue()

    def test_absent_is_placed_then_checked(self):
        rc, out = self.run_ccd()
        self.assertEqual(rc, 0, out); self.assertIn("CCD OK", out); self.assertIn("placed by upstream's setup", out)
        self.assertEqual(self.placed, [self.path])

    def test_present_at_the_pin_is_only_hashed(self):
        with open(self.path, "wb") as f: f.write(b"P" * 7)
        rc, out = self.run_ccd()
        self.assertEqual(rc, 0, out); self.assertIn("present, nothing fetched", out)
        self.assertEqual(self.placed, [])

    def test_biotites_subset_is_replaced(self):
        with open(self.path, "wb") as f: f.write(b"subset")
        rc, out = self.run_ccd()
        self.assertEqual(rc, 0, out); self.assertEqual(self.placed, [self.path])

    def test_a_served_dictionary_off_the_pin_is_refused_and_left_in_place(self):
        rc, out = self.run_ccd(located=self.located(b"republished"))
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out); self.assertTrue(os.path.isfile(self.path))

    def test_a_placement_error_is_relayed(self):
        def broken():
            def place(): raise PermissionError("read-only file system")
            return self.path, place
        rc, out = self.run_ccd(located=broken)
        self.assertEqual(rc, 1, out); self.assertIn("PermissionError: read-only file system", out)

    # ---- --ccd FILE: a dictionary fetched beforehand (a machine without network access)
    def source(self, data):
        p = os.path.join(self.dir, "fetched", "components.bcif"); os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as f: f.write(data)
        return p

    def test_a_ccd_file_at_the_pin_is_placed_without_the_fetch(self):
        src = self.source(b"P" * 7)
        rc, out = self.run_ccd(source=src)
        self.assertEqual(rc, 0, out); self.assertEqual(self.placed, [])                       # upstream's placement call bound, never made
        self.assertIn(f"CCD placed from --ccd {src} sha256={self.pin['sha256']} (no fetch)", out); self.assertIn("CCD OK", out); self.assertIn(f"placed from --ccd {src}", out.splitlines()[-1])
        self.assertEqual(open(self.path, "rb").read(), b"P" * 7)                                # the file's bytes at biotite's CCD path
        self.assertEqual(sorted(os.listdir(self.dir)), ["components.bcif", "fetched"])          # no temporary left beside it

    def test_a_ccd_file_off_the_pin_is_refused_by_name_and_nothing_is_placed(self):
        src = self.source(b"another dictionary")
        rc, out = self.run_ccd(source=src)
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out); self.assertIn(f"--ccd {src}", out); self.assertIn("nothing placed", out)
        self.assertFalse(os.path.exists(self.path)); self.assertEqual(self.placed, [])          # biotite's path untouched, no fetch either

    def test_a_ccd_file_that_is_not_a_file_is_refused(self):
        rc, out = self.run_ccd(source=os.path.join(self.dir, "absent.bcif"))
        self.assertEqual(rc, 1, out); self.assertIn("is not a file", out); self.assertFalse(os.path.exists(self.path)); self.assertEqual(self.placed, [])

    def test_a_ccd_file_when_the_dictionary_is_present_writes_nothing(self):
        with open(self.path, "wb") as f: f.write(b"P" * 7)
        before = os.stat(self.path).st_mtime_ns
        rc, out = self.run_ccd(source=self.source(b"P" * 7))
        self.assertEqual(rc, 0, out); self.assertIn("nothing written", out); self.assertEqual(os.stat(self.path).st_mtime_ns, before); self.assertEqual(self.placed, [])
        rc, out = self.run_ccd(source=self.source(b"another dictionary"))                     # the file is judged first even then: a wrong file is a refusal, not a pass
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out)

    def test_a_ccd_file_replaces_biotites_subset(self):
        with open(self.path, "wb") as f: f.write(b"subset")
        rc, out = self.run_ccd(source=self.source(b"P" * 7))
        self.assertEqual(rc, 0, out); self.assertEqual(open(self.path, "rb").read(), b"P" * 7); self.assertEqual(self.placed, [])

    def test_main_routes_the_ccd_forms(self):
        seen = []
        real = self.W.ccd
        self.W.ccd = lambda **kw: seen.append(kw.get("source")) or 0
        try:
            codes = [self.W.main(a) for a in (["--ccd"], ["--ccd", "/f/components.bcif"], ["--ccd=/f/components.bcif"], ["--ccd", "--weights"], ["--ccd="], ["--ccd", "a", "b"])]
        finally:
            self.W.ccd = real
        self.assertEqual((codes, seen), ([0, 0, 0, 2, 2, 2], [None, "/f/components.bcif", "/f/components.bcif"]))

    def test_upstream_ccd_binds_setup_openfolds_own_call(self):
        """upstream_ccd() = (biotite.setup_ccd.OUTPUT_CCD, setup_biotite_ccd(ccd_path=that, force_download=False)) — on stand-in modules."""
        import sys, types
        from pathlib import Path
        rec = []
        bs = types.ModuleType("biotite.setup_ccd"); bs.OUTPUT_CCD = Path(self.dir) / "structure" / "info" / "components.bcif"
        so = types.ModuleType("openfold3.setup_openfold"); so.setup_biotite_ccd = lambda *, ccd_path, force_download: rec.append((str(ccd_path), force_download))
        mods = {"biotite": types.ModuleType("biotite"), "biotite.setup_ccd": bs, "openfold3": types.ModuleType("openfold3"), "openfold3.setup_openfold": so}
        mods["biotite"].setup_ccd = bs; mods["openfold3"].setup_openfold = so
        saved = {k: sys.modules.get(k) for k in mods}; sys.modules.update(mods)
        try:
            path, place = self.W.upstream_ccd(); place()
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        self.assertEqual((path, rec), (str(bs.OUTPUT_CCD), [(str(bs.OUTPUT_CCD), False)]))


class WeightsDigestGate(unittest.TestCase):
    """openfold3_ob0_opt.weights.fetch: fetch through upstream's route when absent, then the routes' digest comparison — a mismatch, a file without
    a route, or a download error fails by name and deletes nothing."""

    def setUp(self):
        from openfold3_ob0_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="ob0_weights_")
        self.pin = _pins()["weights"]
        self.path = os.path.join(self.dir, self.pin["file"])
        self.calls = []

    def route(self, local):
        def call(d):
            self.calls.append((local, d))
            p = os.path.join(d, local)
            if not os.path.exists(p):                                   # upstream's rule: fetched only when absent
                with open(p, "wb") as f: f.write(b"ckpt")
        return call

    def gate(self, p):
        """stack.weights_gate's record over the test's bytes: b'ckpt' is 'the pinned checkpoint', anything else is not."""
        b = open(p, "rb").read() if os.path.isfile(p) else None
        pinned = b == b"ckpt"
        return {"path": p, "exists": b is not None, "bytes": len(b or b""), "sha256": self.pin["sha256"] if pinned else hashlib.sha256(b or b"").hexdigest(),
                "hashed": True, "pinned_file": self.pin["file"], "pinned_sha256": self.pin["sha256"], "is_pinned": pinned}

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, home=TREE, route=kw.pop("route", self.route), gate=kw.pop("gate", self.gate), out=out)
        return rc, out.getvalue()

    def test_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn("WEIGHTS OK: 1/1", out); self.assertIn(f"OPENFOLD3_OB0_CKPT={self.path}", out)
        self.assertEqual(self.calls, [(self.pin["file"], self.dir)]); self.assertTrue(os.path.isfile(self.path))

    def test_a_present_file_is_kept_and_checked(self):
        with open(self.path, "wb") as f: f.write(b"ckpt")
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out); self.assertIn("present", out); self.assertEqual(self.calls, [])

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        with open(self.path, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out); self.assertIn("REFUSED", out.splitlines()[-1]); self.assertIn(self.pin["file"], out.splitlines()[-1])
        self.assertTrue(os.path.isfile(self.path))

    def test_a_file_without_an_upstream_route_is_refused(self):
        rc, out = self.fetch(route=lambda local: None)
        self.assertEqual(rc, 1, out); self.assertIn("has no entry with this file name", out); self.assertEqual(os.listdir(self.dir), [])

    def test_a_fetch_error_is_relayed(self):
        def broken(local):
            def call(d): raise ConnectionError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out); self.assertIn("ConnectionError: network unreachable", out)
        self.assertEqual(os.listdir(self.dir), [])                       # nothing written, nothing deleted

    def test_the_pin_is_upstreams_registry_default(self):
        """stock/PINS.json weights.file is the file upstream's checkpoint registry serves for its DEFAULT_CHECKPOINT_NAME, and the pin's url names
        upstream's bucket + key for it — read from the pinned source as text (no torch import)."""
        import re
        src = open(_stubs.stock_src("openfold3", "entry_points", "parameters.py"), encoding="utf-8").read()
        default = re.search(r'DEFAULT_CHECKPOINT_NAME = "([^"]+)"', src).group(1)
        bucket = re.search(r'OPENFOLD_BUCKET = "([^"]+)"', src).group(1)
        entry = re.search(r'"' + re.escape(default) + r'":\s*\w+\(\s*(?:[^)]*?)file_name="([^"]+)"', src, re.S)
        self.assertIsNotNone(entry, default); self.assertEqual(entry.group(1), self.pin["file"])
        self.assertIn(f"https://{bucket}.s3.amazonaws.com/openfold3-parameters/{self.pin['file']}", self.pin["url"])

    def test_upstream_route_selects_by_file_name(self):
        """upstream_route(local) binds download_model_parameters(dir, <the registry name whose file_name is local>, force_download=False,
        skip_confirmation=True); a file no registry entry serves has no route — on a stand-in parameters module."""
        import sys, types
        rec = []
        P = types.ModuleType("openfold3.entry_points.parameters")
        P.OPENFOLD_MODEL_CHECKPOINT_REGISTRY = {"legacy": types.SimpleNamespace(file_name="old.pt"), "openbind": types.SimpleNamespace(file_name=self.pin["file"])}
        P.download_model_parameters = lambda d, name, force_download, skip_confirmation: rec.append((str(d), name, force_download, skip_confirmation))
        ep = types.ModuleType("openfold3.entry_points"); ep.parameters = P
        mods = {"openfold3": types.ModuleType("openfold3"), "openfold3.entry_points": ep, "openfold3.entry_points.parameters": P}
        saved = {k: sys.modules.get(k) for k in mods}; sys.modules.update(mods)
        try:
            self.W.upstream_route(self.pin["file"])(self.dir)
            self.assertIsNone(self.W.upstream_route("elsewhere.pt"))
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        self.assertEqual(rec, [(self.dir, "openbind", False, True)])


if __name__ == "__main__":
    unittest.main()
