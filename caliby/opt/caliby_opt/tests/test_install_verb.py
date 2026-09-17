"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's digest gate (caliby_opt.weights.fetch with an injected route and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import tarfile
import tempfile
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")
PINS = os.path.join(TREE, "stock", "PINS.json")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m caliby_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="caliby_install_verb_")
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

    def test_install_runs_pip_then_the_pin_check(self):
        rc, out, calls = self.run_sh(["install"])
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls)
        self.assertEqual(len(self.probes), 1, self.probes)

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/model_params"], ["install", "--weights=/data/model_params"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m caliby_opt.weights /data/model_params"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--bogus"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "fast"], ["install", "--config", "h100"], ["--config", "h100", "install"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertEqual(self.probes, [], (args, self.probes))

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls)); self.assertIn("run.sh: the install failed", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls)); self.assertIn("refused by the pin check", out)
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m caliby_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)


class WeightsDigestGate(unittest.TestCase):
    """caliby_opt.weights.fetch: fetch through the route, then every pinned file's sha256 against the pin — a mismatch or a file without a route fails by name."""

    def setUp(self):
        from caliby_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="caliby_weights_")
        self.payload = {"caliby/soluble_caliby.ckpt": b"ckpt-bytes", "caliby/caliby.ckpt": b"other-ckpt", "protpardelle-1c/weights/cc95_epoch3490.pth": b"pp-weights",
                        "protpardelle-1c/configs/cc95.yaml": b"pp-config"}
        self.files = {k: {"sha256": hashlib.sha256(v).hexdigest(), "size_bytes": len(v), **({"ckpt": os.path.basename(k)[:-5]} if k.startswith("caliby/") else {})}
                      for k, v in self.payload.items()}
        self.calls = []

    def write(self, rel):
        p = os.path.join(self.dir, rel)
        if not os.path.exists(p):                                       # upstream's rule: fetched only when absent
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f: f.write(self.payload[rel])

    def route(self, rel, info):
        def call():
            self.calls.append(rel)
            if rel.startswith("protpardelle-1c/"):                       # a directory fetch brings every file of the directory
                for r in self.payload:
                    if r.startswith("protpardelle-1c/"): self.write(r)
            else:
                self.write(rel)
        return call

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), route=kw.pop("route", self.route), companions=kw.pop("companions", []), out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 4/4", out); self.assertIn(f"export MODEL_PARAMS_DIR={self.dir}", out)
        self.assertEqual(self.calls, ["caliby/soluble_caliby.ckpt", "caliby/caliby.ckpt", "protpardelle-1c/weights/cc95_epoch3490.pth"])   # the second Protpardelle file arrived with the directory
        for rel in self.files: self.assertTrue(os.path.isfile(os.path.join(self.dir, rel)))

    def test_a_present_file_is_kept_and_checked(self):
        self.write("caliby/caliby.ckpt")
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("caliby/caliby.ckpt: present", out); self.assertNotIn("caliby/caliby.ckpt", self.calls)

    def test_companion_directories_are_fetched_when_absent_and_named(self):
        got = []
        def mpnn():
            got.append("proteinmpnn"); os.makedirs(os.path.join(self.dir, "proteinmpnn"))
        rc, out = self.fetch(companions=[("proteinmpnn/", mpnn)])
        self.assertEqual(rc, 0, out); self.assertEqual(got, ["proteinmpnn"]); self.assertIn("proteinmpnn/: fetching", out)
        rc, out = self.fetch(companions=[("proteinmpnn/", mpnn)])
        self.assertEqual(rc, 0, out); self.assertEqual(got, ["proteinmpnn"]); self.assertIn("proteinmpnn/: present", out)   # kept, not fetched again

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "caliby", "soluble_caliby.ckpt"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1 of 4", out); self.assertIn("caliby/soluble_caliby.ckpt", out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p)); self.assertEqual(open(p, "rb").read(), b"other bytes")

    def test_a_file_without_an_upstream_route_is_refused(self):
        rc, out = self.fetch(route=lambda rel, info: None if rel.startswith("protpardelle-1c/") else self.route(rel, info))
        self.assertEqual(rc, 1, out)
        self.assertIn("no download entry point for protpardelle-1c/weights/cc95_epoch3490.pth", out)

    def test_a_fetch_error_is_relayed(self):
        def broken(rel, info):
            def call(): raise ConnectionError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching caliby/soluble_caliby.ckpt: ConnectionError: network unreachable", out)
        self.assertEqual(os.listdir(self.dir), [])                       # nothing written, nothing deleted

    def test_a_fetch_that_leaves_no_file_is_refused(self):
        rc, out = self.fetch(route=lambda rel, info: (lambda: None))
        self.assertEqual(rc, 1, out); self.assertIn("caliby/soluble_caliby.ckpt is not at", out)

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2); self.assertEqual(self.W.main(["--weights"]), 2); self.assertEqual(self.W.main(["a", "b"]), 2)

    def test_pins_are_upstreams_constants(self):
        """stock/PINS.json weights: hf_repo is upstream's HF_REPO_ID (caliby/weights.py in the pinned archive, and download_model_params.sh in stock/src);
        every pinned checkpoint's `ckpt` name resolves through upstream's MODEL_REGISTRY to that file; the Protpardelle-1c files sit under the
        directory generate_ensembles ensures — read from the pinned sources as text (no upstream import)."""
        pins = json.load(open(PINS, encoding="utf-8"))
        arc = os.path.join(TREE, pins["upstream"]["caliby"]["archive"])
        with tarfile.open(arc) as t:
            member = next(n for n in t.getnames() if n.endswith("/caliby/weights.py"))
            weights_py = t.extractfile(member).read().decode("utf-8")
            api_py = t.extractfile(member.replace("/caliby/weights.py", "/caliby/api.py")).read().decode("utf-8")
        self.assertEqual(re.search(r'HF_REPO_ID = "([^"]+)"', weights_py).group(1), pins["weights"]["hf_repo"])
        registry = dict(re.findall(r'"([A-Za-z0-9_]+)":\s*"([^"]+\.ckpt)"', weights_py.split("MODEL_REGISTRY", 1)[1].split("}", 1)[0]))
        dl = open(os.path.join(TREE, "stock", "src", "caliby", "download_model_params.sh"), encoding="utf-8").read()
        self.assertIn(f"snapshot_download('{pins['weights']['hf_repo']}'", dl)
        for rel, info in pins["weights"]["files"].items():
            if rel.startswith(self.W.CKPT_DIR):
                self.assertEqual(registry.get(info["ckpt"]), rel, rel)             # resolve_ckpt_path(info['ckpt']) fetches exactly this file
            else:
                self.assertTrue(rel.startswith(self.W.PROTPARDELLE_DIR), rel)
        for d in (self.W.PROTPARDELLE_DIR.rstrip("/"),) + self.W.COMPANION_DIRS:
            self.assertIn(f'ensure_dir(f"{{model_params_path}}/{d}")', api_py, d)   # generate_ensembles' own directory fetches

    def test_the_lock_is_the_pinned_stack(self):
        """environment/requirements.lock (the one list of the stack: PINS.json "pinned_stack"."freeze" names it) carries PINS.json's own pins —
        the three upstream `@ git+https` lines at their commits and the interpreter's torch / triton."""
        pins = json.load(open(PINS, encoding="utf-8"))
        self.assertEqual(pins["pinned_stack"]["freeze"], "environment/requirements.lock")
        lock = [l.rstrip("\n") for l in open(os.path.join(TREE, pins["pinned_stack"]["freeze"]), encoding="utf-8") if l.strip() and not l.startswith("#")]
        for name, pin in pins["upstream"].items():                      # `name @ git+<repo>[.git]@<commit>`
            self.assertTrue(any(l.startswith(f"{name} @ git+{pin['repo']}") and l.endswith(f"@{pin['commit']}") for l in lock), name)
        self.assertIn(f"torch=={pins['pinned_stack']['torch'].split('+')[0]}", lock)
        self.assertIn(f"triton=={pins['pinned_stack']['triton']}", lock)
        self.assertEqual(len(lock), len(set(lock)))                      # no line twice

    def test_upstream_route_table(self):
        """upstream_route's selection rule by PINS path, on a stand-in caliby.weights: caliby/<x>.ckpt with a registry name → resolve_ckpt_path(name),
        protpardelle-1c/* → ensure_dir(DIR/protpardelle-1c), the companion proteinmpnn/ → ensure_dir(DIR/proteinmpnn); anything else has no route.
        MODEL_PARAMS_DIR is bound to DIR and HF_HUB_OFFLINE lifted."""
        rec = []
        up = types.ModuleType("caliby.weights")
        up.MODEL_REGISTRY = {"soluble_caliby": "caliby/soluble_caliby.ckpt", "caliby": "caliby/caliby.ckpt"}
        up.resolve_ckpt_path = lambda name: rec.append(("ckpt", name))
        up.ensure_dir = lambda path: rec.append(("dir", path))
        pkg = types.ModuleType("caliby"); pkg.__path__ = []; pkg.weights = up
        saved = {k: sys.modules.get(k) for k in ("caliby", "caliby.weights")}
        saved_env = {k: os.environ.get(k) for k in ("MODEL_PARAMS_DIR", "HF_HUB_OFFLINE")}
        sys.modules.update({"caliby": pkg, "caliby.weights": up}); os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            out = io.StringIO()
            route, companions = self.W.upstream_route(self.dir, out)
            self.assertEqual(os.environ["MODEL_PARAMS_DIR"], self.dir); self.assertNotIn("HF_HUB_OFFLINE", os.environ); self.assertIn("HF_HUB_OFFLINE lifted", out.getvalue())
            route("caliby/soluble_caliby.ckpt", {"ckpt": "soluble_caliby"})(); route("protpardelle-1c/configs/cc95.yaml", {})()
            self.assertIsNone(route("caliby/soluble_caliby_v1.ckpt", {"ckpt": "soluble_caliby_v1"}))   # a name upstream's registry does not carry: no entry point
            self.assertIsNone(route("elsewhere/file.bin", {}))
            self.assertEqual([n for n, _ in companions], ["proteinmpnn/"]); companions[0][1]()
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
            for k, v in saved_env.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        self.assertEqual(rec, [("ckpt", "soluble_caliby"), ("dir", os.path.join(self.dir, "protpardelle-1c")), ("dir", os.path.join(self.dir, "proteinmpnn"))])


if __name__ == "__main__":
    unittest.main()
