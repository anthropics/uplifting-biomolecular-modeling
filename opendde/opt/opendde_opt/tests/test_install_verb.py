"""`run.sh install [--weights DIR]` — the install verb's arguments and the two commands it runs, and the weights step's digest gate.

The verb (README.md 'Install') installs the shared core and this kit editable into the python on PATH, runs the pin check, and with
`--weights DIR` hands DIR to `python -m opendde_opt.weights`. These tests put a recording stand-in first on PATH as `python`, so they assert
on the exact argv run.sh produces without installing anything; the weights step is exercised with an injected route and matcher (no
network, no upstream import) to show that a digest mismatch is refused by name and the file is left in place; its file list and upstream
route are checked against upstream's own release records in stock/src/.
"""
import io
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # opendde/
RUN_SH = os.path.join(TREE, "run.sh")
UPSTREAM = os.path.join(TREE, "stock", "src", "opendde", "config")     # upstream's dependency_url.py + model_manifest.json, as released

STUB = textwrap.dedent(r'''
    #!/bin/sh
    # recording stand-in for `python`: one line per call (args joined by |) in $STUB_LOG; exit code per call kind from the environment
    printf '%s\n' "$*" | tr ' ' '|' >> "$STUB_LOG"
    case "$*" in
      *"import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
      *"pip install"*) exit "${STUB_RC_PIP:-0}" ;;
      *check_pins.py*) exit "${STUB_RC_PINS:-0}" ;;
      *opendde_opt.weights*) exit "${STUB_RC_WEIGHTS:-0}" ;;
    esac
    exit 0
''').lstrip()


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="odde_install_")
        self.bin = os.path.join(self.tmp, "bin"); os.makedirs(self.bin)
        stub = os.path.join(self.bin, "python")
        with open(stub, "w") as fh:
            fh.write(STUB)
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        self.log = os.path.join(self.tmp, "calls.log")

    def run_sh(self, *args, **rc):
        env = {k: v for k, v in os.environ.items() if not k.startswith(("OPENDDE_OPT", "MODEL_OPT"))}
        env.update(PATH=self.bin + os.pathsep + env.get("PATH", ""), STUB_LOG=self.log)
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        open(self.log, "w").close()
        p = subprocess.run(["bash", RUN_SH, *args], env=env, capture_output=True, text=True, timeout=60)
        calls = [line.split("|") for line in open(self.log).read().splitlines() if line]
        return p.returncode, [c for c in calls if not (len(c) > 1 and c[0] == "-I" and c[1] == "-c")], p.stdout + p.stderr   # the probe call is plumbing; assert on the commands

    def test_install_runs_pip_then_the_pin_check(self):
        rc, calls, out = self.run_sh("install")
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [
            ["-m", "pip", "install", "-e", f"{TREE}/../common/opt_core", "-e", f"{TREE}/opt"],
            ["-I", f"{TREE}/stock/check_pins.py"],
        ])

    def test_a_tree_already_installed_skips_pip_by_name(self):
        rc, calls, out = self.run_sh("install", probe=0)
        self.assertEqual(rc, 0, out)
        self.assertEqual(calls, [["-I", f"{TREE}/stock/check_pins.py"]])
        self.assertIn("installed from this tree already", out)

    def test_weights_dir_is_handed_to_the_weights_step_after_the_pin_check(self):
        for args in (["install", "--weights", "/w"], ["install", "--weights=/w"]):
            rc, calls, out = self.run_sh(*args)
            self.assertEqual(rc, 0, out)
            self.assertEqual([c[:2] for c in calls], [["-m", "pip"], ["-I", f"{TREE}/stock/check_pins.py"], ["-m", "opendde_opt.weights"]], args)
            self.assertEqual(calls[2], ["-m", "opendde_opt.weights", "/w"])

    def test_pip_failure_stops_before_the_pin_check(self):
        rc, calls, out = self.run_sh("install", "--weights", "/w", pip=1)
        self.assertEqual(rc, 1)
        self.assertEqual(len(calls), 1, calls)
        self.assertIn("the install failed", out)

    def test_pin_refusal_is_exit_3_and_skips_the_weights(self):
        rc, calls, out = self.run_sh("install", "--weights", "/w", pins=3)
        self.assertEqual(rc, 3)
        self.assertEqual([c[:2] for c in calls], [["-m", "pip"], ["-I", f"{TREE}/stock/check_pins.py"]])
        self.assertIn("refused by the pin check", out)

    def test_weights_failure_is_the_verbs_exit_code(self):
        rc, calls, out = self.run_sh("install", "--weights", "/w", weights=1)
        self.assertEqual(rc, 1)
        self.assertEqual(len(calls), 3)

    def test_usage_errors_exit_2_and_run_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                     ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--weights", "--mode"]):
            rc, calls, out = self.run_sh(*args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], args)


class WeightsDigestGate(unittest.TestCase):
    """opendde_opt.weights.fetch with an injected route + matcher: fetches only what is absent, refuses a mismatch by name, deletes nothing."""

    def setUp(self):
        sys.path.insert(0, os.path.join(TREE, "opt"))
        import opendde_opt.weights as W
        self.W = W
        self.dir = tempfile.mkdtemp(prefix="odde_weights_")
        self.files = [{"local": "checkpoint/opendde.pt", "sha256": "aa" * 32, "bytes": 3}, {"local": "common/components.cif", "sha256": "bb" * 32, "bytes": 3}]

    def route(self, record):
        def r(local, target):
            def call():
                record.append((local, target))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, "wb") as fh:
                    fh.write(b"new")
            return call
        return r

    def test_fetches_absent_files_then_passes_when_digests_match(self):
        rec = []; out = io.StringIO()
        rc = self.W.fetch(self.dir, files=self.files, route=self.route(rec), matcher=lambda d: {"status": "pinned", "files": 2, "unknown": [], "seconds": 0.0}, out=out)
        self.assertEqual(rc, 0, out.getvalue())
        self.assertEqual(rec, [(f["local"], os.path.join(self.dir, f["local"])) for f in self.files])
        self.assertIn("WEIGHTS OK: 2/2", out.getvalue())
        self.assertIn(f"OPENDDE_ROOT_DIR={self.dir}", out.getvalue())

    def test_present_files_are_not_refetched(self):
        os.makedirs(os.path.join(self.dir, "checkpoint"))
        with open(os.path.join(self.dir, "checkpoint/opendde.pt"), "wb") as fh:
            fh.write(b"old")
        rec = []; out = io.StringIO()
        rc = self.W.fetch(self.dir, files=self.files, route=self.route(rec), matcher=lambda d: {"status": "pinned", "files": 2, "unknown": []}, out=out)
        self.assertEqual(rc, 0)
        self.assertEqual(rec, [("common/components.cif", os.path.join(self.dir, "common/components.cif"))])
        self.assertEqual(open(os.path.join(self.dir, "checkpoint/opendde.pt"), "rb").read(), b"old")

    def test_digest_mismatch_is_refused_by_name_and_left_in_place(self):
        rec = []; out = io.StringIO()
        bad = lambda d: {"status": "unknown", "files": 2, "unknown": [{"local": "checkpoint/opendde.pt", "sha256": "cc" * 32}]}
        rc = self.W.fetch(self.dir, files=self.files, route=self.route(rec), matcher=bad, out=out)
        self.assertEqual(rc, 1)
        text = out.getvalue()
        self.assertIn("REFUSED: 1 of 2", text); self.assertIn("checkpoint/opendde.pt", text); self.assertNotIn("WEIGHTS OK", text)
        self.assertTrue(os.path.isfile(os.path.join(self.dir, "checkpoint/opendde.pt")))          # left in place, never deleted

    def test_upstream_error_is_relayed_and_nothing_else_runs(self):
        def failing(local, target):
            def call():
                raise OSError("network unreachable")
            return call
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=self.files, route=failing, matcher=lambda d: self.fail("matcher must not run"), out=out)
        self.assertEqual(rc, 1)
        self.assertIn("FAILED fetching checkpoint/opendde.pt: OSError: network unreachable", out.getvalue())

    def test_a_file_upstream_cannot_fetch_is_named(self):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=self.files, route=lambda local, target: None, matcher=lambda d: self.fail("matcher must not run"), out=out)
        self.assertEqual(rc, 1)
        self.assertIn("no download entry point for checkpoint/opendde.pt", out.getvalue())

    def test_usage(self):
        self.assertEqual(self.W.main([]), 2)
        self.assertEqual(self.W.main(["--help"]), 2)
        self.assertEqual(self.W.main(["a", "b"]), 2)

    def test_the_file_list_is_the_pin_files_in_upstreams_layout(self):
        files = self.W.pinned_files(TREE)
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        self.assertEqual([f["local"] for f in files], ["checkpoint/opendde.pt"] + [f"common/{n}" for n in pins["common"]["files"]])
        self.assertEqual(len(files), 5)
        for f in files:
            self.assertRegex(f["sha256"], r"^[0-9a-f]{64}$"); self.assertGreater(f["bytes"], 0)

    def test_check_hashes_every_file_against_its_pin(self):
        """The real comparator on small files: the checkpoint through manifest.weights (memo dir redirected), the common/ files hashed directly;
        a wrong digest and an absent file are each named."""
        import hashlib
        root = tempfile.mkdtemp(prefix="odde_root_"); memo = tempfile.mkdtemp(prefix="odde_memo_")
        files = []
        for local, data in (("checkpoint/opendde.pt", b"ckpt"), ("common/components.cif", b"cif"), ("common/release_date_cache.json", b"{}")):
            p = os.path.join(root, local); os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as fh:
                fh.write(data)
            files.append({"local": local, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
        saved = os.environ.get(self.W_manifest().DIGEST_DIR_ENV); os.environ[self.W_manifest().DIGEST_DIR_ENV] = memo
        try:
            wid = self.W.check(root, files, TREE)
            self.assertEqual((wid["status"], wid["files"], wid["unknown"]), ("pinned", 3, []), wid)
            with open(os.path.join(root, "common/components.cif"), "wb") as fh:
                fh.write(b"other")
            os.remove(os.path.join(root, "common/release_date_cache.json"))
            wid = self.W.check(root, files, TREE)
            self.assertEqual(wid["status"], "unknown")
            self.assertEqual({u["local"]: u["sha256"] for u in wid["unknown"]},
                             {"common/components.cif": hashlib.sha256(b"other").hexdigest(), "common/release_date_cache.json": "absent"})
        finally:
            if saved is None: os.environ.pop(self.W_manifest().DIGEST_DIR_ENV, None)
            else: os.environ[self.W_manifest().DIGEST_DIR_ENV] = saved

    def W_manifest(self):
        import opendde_opt.manifest as M
        return M

    def test_pins_are_upstreams_release_records(self):
        """stock/PINS.json restates upstream's own records: the checkpoint entry is the default model's default checkpoint in the packaged
        model_manifest.json (file, bytes, sha256, source revision), and every common/ file is one of dependency_url.py's managed assets (its URL
        ends in that file name; bytes and sha256 equal) — so the weights step's match-by-name reaches every pinned file and upstream's own
        validation agrees with the pin."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        man = json.load(open(os.path.join(UPSTREAM, "model_manifest.json")))
        model = [m for m in man["models"] if m["name"] == man["default_model"]][0]
        ck = [c for c in model["checkpoints"] if c["filename"] == model["default_checkpoint"]][0]
        self.assertEqual(model["name"], pins["cli_defaults"]["model_name"])
        self.assertEqual((os.path.basename(pins["checkpoint"]["file"]), int(pins["checkpoint"]["bytes"]), pins["checkpoint"]["sha256"]),
                         (ck["filename"], int(ck["size_bytes"]), ck["sha256"]))
        self.assertEqual(pins["checkpoint"]["hf_commit"], man["source"]["revision"])
        self.assertTrue(pins["checkpoint"]["hf"].startswith(man["source"]["repository"] + "/") and pins["checkpoint"]["hf"].endswith("/" + ck["filename"]))
        src = open(os.path.join(UPSTREAM, "dependency_url.py")).read()
        assets = {m.group(1): (int(m.group(2).replace("_", "")), m.group(3))
                  for m in re.finditer(r'"(\w+)":\s*ManagedAsset\(\s*size=([\d_]+),\s*sha256=\(?\s*((?:"[0-9a-f]+"\s*)+)\)?', src)}
        assets = {k: (size, "".join(re.findall(r'"([0-9a-f]+)"', sha))) for k, (size, sha) in assets.items()}
        by_name = {m.group(2): m.group(1) for m in re.finditer(r'"(\w+)":\s*common_url\("([^"]+)"\)', src)}
        common = pins["common"]
        self.assertEqual(sorted(common["files"]), sorted(common["managed_assets"]))
        for name in common["files"]:
            self.assertIn(name, by_name, name)                                                     # upstream has a URL entry ending in this name
            self.assertEqual(assets[by_name[name]], (int(common["managed_assets"][name]["bytes"]), common["managed_assets"][name]["sha256"]), name)
        self.assertIn(f'CHECKPOINT_FILES = {{\n    model["name"]: model["default_checkpoint"]', src)   # the checkpoint's URL entry is keyed by the model name and ends in its file name

    def test_upstream_route_matches_pinned_files_to_upstreams_assets_by_name(self):
        """upstream_route binds upstream's per-asset downloader: the checkpoint to the model-name asset, each common/ file to the asset whose URL
        ends in its name, nothing else — shown with recording stand-ins for the two upstream modules."""
        import types
        rec = []
        du = types.ModuleType("opendde.config.dependency_url")
        du.URL = {"opendde_v1": "https://hub.example/r/abc/opendde.pt", "ccd_components_file": "https://deps.example/common/components.cif",
                  "release_dates_path": "https://deps.example/common/release_date_cache.json", "not_managed": "https://deps.example/common/obsolete_to_successor.json"}
        du.MANAGED_ASSETS = {"opendde_v1": object(), "ccd_components_file": object(), "release_dates_path": object()}
        dl = types.ModuleType("opendde.utils.download"); dl._ensure_managed_asset = lambda key, target: rec.append((key, target))
        mods = {"opendde": types.ModuleType("opendde"), "opendde.config": types.ModuleType("opendde.config"), "opendde.config.dependency_url": du,
                "opendde.utils": types.ModuleType("opendde.utils"), "opendde.utils.download": dl}
        mods["opendde.config"].dependency_url = du; mods["opendde.utils"].download = dl
        mods["opendde"].config = mods["opendde.config"]; mods["opendde"].utils = mods["opendde.utils"]
        saved = {k: sys.modules.get(k) for k in mods}
        sys.modules.update(mods)
        try:
            route = self.W.upstream_route()
            route("checkpoint/opendde.pt", "/r/checkpoint/opendde.pt")(); route("common/components.cif", "/r/common/components.cif")()
            route("common/release_date_cache.json", "/r/common/release_date_cache.json")()
            self.assertIsNone(route("common/obsolete_to_successor.json", "/r/common/obsolete_to_successor.json"))   # a URL entry that is not a managed asset is not a route
            self.assertIsNone(route("elsewhere/file.bin", "/r/elsewhere/file.bin"))
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
        self.assertEqual(rec, [("opendde_v1", "/r/checkpoint/opendde.pt"), ("ccd_components_file", "/r/common/components.cif"), ("release_dates_path", "/r/common/release_date_cache.json")])


if __name__ == "__main__":
    unittest.main()
