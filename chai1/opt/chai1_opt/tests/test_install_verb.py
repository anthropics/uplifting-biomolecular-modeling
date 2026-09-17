"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), and the weights step's digest gate (chai1_opt.weights.fetch with an injected route and pin table; no network, no upstream). CPU only."""
import hashlib
import io
import os
import stat
import subprocess
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/
RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s%s\\n' "$*" "${CHAI1_OPT_STRICT_STACK:+ [CHAI1_OPT_STRICT_STACK=$CHAI1_OPT_STRICT_STACK]}" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m chai1_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


class InstallVerbArguments(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="chai1_install_verb_")
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
        self.assertEqual(len(calls), 2, calls)
        self.assertEqual(calls[0], f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt")
        self.assertEqual(calls[1], f"-I {TREE}/stock/check_pins.py")

    def test_weights_dir_adds_the_weights_step_last(self):
        for args in (["install", "--weights", "/data/chai_downloads"], ["install", "--weights=/data/chai_downloads"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 0, out)
            self.assertEqual(calls[2:], ["-m chai1_opt.weights /data/chai_downloads"], calls)
            self.assertEqual(len(calls), 3, calls)

    def test_usage_errors_call_nothing(self):
        for args in (["install", "--weights"], ["install", "--weights", "--strict-stack"], ["install", "--weights="], ["install", "--bogus"],
                     ["install", "extra"], ["install", "--mode", "exact"], ["install", "--config", "h100"]):
            if os.path.exists(self.log): os.remove(self.log)
            rc, out, calls = self.run_sh(args)
            self.assertEqual(rc, 2, (args, out))
            self.assertEqual(calls, [], (args, calls))
            self.assertIn("run.sh:", out)

    def test_a_failed_step_stops_the_sequence_with_its_code(self):
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pip=1)
        self.assertEqual((rc, len(calls)), (1, 1), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], pins=3)
        self.assertEqual((rc, len(calls)), (3, 2), (out, calls))
        os.remove(self.log)
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], weights=1)
        self.assertEqual((rc, len(calls)), (1, 3), (out, calls))

    def test_strict_stack_reaches_the_pin_check_through_the_environment(self):
        rc, out, calls = self.run_sh(["install", "--strict-stack"])
        self.assertEqual((rc, len(calls)), (0, 2), (out, calls))   # the switch is the exported variable stock/check_pins.py reads, not an argument of the verb
        self.assertTrue(all(c.endswith(" [CHAI1_OPT_STRICT_STACK=1]") for c in calls), calls)

    def test_a_tree_already_installed_skips_pip_by_name(self):
        """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
        --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
        rc, out, calls = self.run_sh(["install", "--weights", "/w"], probe=0)
        self.assertEqual(rc, 0, out); self.assertIn("installed from this tree already", out)
        self.assertEqual(calls, [f"-I {TREE}/stock/check_pins.py", "-m chai1_opt.weights /w"], calls)   # pin check, weights — no pip
        self.assertEqual(len(self.probes), 1, self.probes)


class WeightsDigestGate(unittest.TestCase):
    """chai1_opt.weights.fetch: fetch through the route, then every file's sha256 against the pin — a mismatch or a file without a route fails by name."""

    def setUp(self):
        from chai1_opt import weights
        self.W = weights
        self.dir = tempfile.mkdtemp(prefix="chai1_weights_")
        self.payload = {"models_v2/trunk.pt": b"trunk-bytes", "conformers_v1.apkl": b"conformers", "esm/model.pt": b"esm-bytes"}
        self.files = [{"local": k, "url": f"https://upstream.example/{k}", "sha256": hashlib.sha256(v).hexdigest(), "bytes": len(v)} for k, v in self.payload.items()]
        self.calls = []

    def route(self, local):
        def call():
            self.calls.append(local)
            p = os.path.join(self.dir, local)
            if not os.path.exists(p):                                   # upstream's rule: fetched only when absent
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "wb") as f: f.write(self.payload[local])
        return call

    def matcher(self, d):
        """stack.weights_match's contract over the test's pin table: {status pinned|unknown, files, unknown [{local, sha256}], seconds}."""
        unknown = []
        for f in self.files:
            p = os.path.join(d, f["local"]); digest = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else "absent"
            if digest != f["sha256"]: unknown.append({"local": f["local"], "sha256": digest})
        return {"status": "unknown" if unknown else "pinned", "files": len(self.files), "unknown": unknown, "seconds": 0.0}

    def fetch(self, **kw):
        out = io.StringIO()
        rc = self.W.fetch(self.dir, files=kw.pop("files", self.files), route=kw.pop("route", self.route), matcher=kw.pop("matcher", self.matcher), out=out, **kw)
        return rc, out.getvalue()

    def test_all_fetched_and_pinned(self):
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("WEIGHTS OK: 3/3", out)
        self.assertEqual(self.calls, [f["local"] for f in self.files])
        for f in self.files: self.assertTrue(os.path.isfile(os.path.join(self.dir, f["local"])))

    def test_a_present_file_is_kept_and_checked(self):
        p = os.path.join(self.dir, "conformers_v1.apkl"); os.makedirs(self.dir, exist_ok=True)
        with open(p, "wb") as f: f.write(b"conformers")
        rc, out = self.fetch()
        self.assertEqual(rc, 0, out)
        self.assertIn("conformers_v1.apkl: present", out)

    def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(self):
        p = os.path.join(self.dir, "models_v2", "trunk.pt"); os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f: f.write(b"other bytes")
        rc, out = self.fetch()
        self.assertEqual(rc, 1, out)
        self.assertIn("REFUSED: 1 of 3", out); self.assertIn("models_v2/trunk.pt", out.splitlines()[-1])
        self.assertTrue(os.path.isfile(p))

    def test_a_file_without_an_upstream_route_is_refused(self):
        rc, out = self.fetch(route=lambda local: None if local.startswith("esm/") else self.route(local))
        self.assertEqual(rc, 1, out)
        self.assertIn("no download entry point for esm/model.pt", out)

    def test_a_fetch_error_is_relayed(self):
        def broken(local):
            def call(): raise ConnectionError("network unreachable")
            return call
        rc, out = self.fetch(route=broken)
        self.assertEqual(rc, 1, out)
        self.assertIn("FAILED fetching models_v2/trunk.pt: ConnectionError: network unreachable", out)
        self.assertEqual(os.listdir(self.dir), [])                       # nothing written, nothing deleted

    def test_pins_urls_are_upstreams_constants(self):
        """stock/PINS.json weights.files[].url (what the fetching line could cite, what STOCK.md lists) are the URLs upstream's entry points fetch:
        paths.COMPONENT_URL per models_v2 component, paths.cached_conformers, esm.ESM_URL — read from the pinned source as text (no torch import)."""
        import json, re
        src = os.path.join(TREE, "stock", "src", "chai_lab")
        paths_py = open(os.path.join(src, "utils", "paths.py"), encoding="utf-8").read(); esm_py = open(os.path.join(src, "data", "dataset", "embeddings", "esm.py"), encoding="utf-8").read()
        component = re.search(r'COMPONENT_URL = \(?\s*"([^"]+)"', paths_py).group(1); conformers = re.search(r'cached_conformers = Downloadable\(\s*url="([^"]+)"', paths_py).group(1)
        esm_url = re.search(r'ESM_URL = "([^"]+)"', esm_py).group(1)
        for w in json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))["weights"]["files"]:
            local = w["local"]
            expect = conformers if local == "conformers_v1.apkl" else esm_url if local.startswith("esm/") else component.format(comp_key=local.split("/", 1)[1])
            self.assertEqual(w["url"], expect, local)
            if local.startswith("esm/"): self.assertEqual(os.path.basename(local), os.path.basename(esm_url))

    def test_upstream_route_table(self):
        """upstream_route's selection rule by PINS path, on a stand-in paths module: models_v2/*.pt → chai1_component, the conformer cache →
        cached_conformers, esm/* → download_if_not_exists with upstream's ESM URL; anything else has no route."""
        import sys, types
        rec = []
        paths = types.ModuleType("chai_lab.utils.paths")
        from pathlib import Path
        paths.downloads_path = Path(self.dir)
        paths.chai1_component = lambda key: rec.append(("component", key))
        paths.cached_conformers = types.SimpleNamespace(get_path=lambda: rec.append(("conformers",)))
        paths.download_if_not_exists = lambda url, path: rec.append(("download", url, str(path)))
        esm = types.ModuleType("chai_lab.data.dataset.embeddings.esm"); esm.ESM_URL = "https://upstream.example/esm2/model_fp16.pt"
        mods = {"chai_lab": types.ModuleType("chai_lab"), "chai_lab.utils": types.ModuleType("chai_lab.utils"), "chai_lab.utils.paths": paths,
                "chai_lab.data": types.ModuleType("chai_lab.data"), "chai_lab.data.dataset": types.ModuleType("chai_lab.data.dataset"),
                "chai_lab.data.dataset.embeddings": types.ModuleType("chai_lab.data.dataset.embeddings"), "chai_lab.data.dataset.embeddings.esm": esm}
        mods["chai_lab.utils"].paths = paths
        saved = {k: sys.modules.get(k) for k in mods}; saved_env = os.environ.get("CHAI_DOWNLOADS_DIR")
        sys.modules.update(mods)
        try:
            route = self.W.upstream_route(self.dir)
            self.assertEqual(os.environ["CHAI_DOWNLOADS_DIR"], self.dir)
            route("models_v2/trunk.pt")(); route("conformers_v1.apkl")(); route("esm/model_fp16.pt")()
            self.assertIsNone(route("elsewhere/file.bin"))
        finally:
            for k, v in saved.items():
                if v is None: sys.modules.pop(k, None)
                else: sys.modules[k] = v
            if saved_env is None: os.environ.pop("CHAI_DOWNLOADS_DIR", None)
            else: os.environ["CHAI_DOWNLOADS_DIR"] = saved_env
        self.assertEqual(rec, [("component", "trunk.pt"), ("conformers",), ("download", esm.ESM_URL, os.path.join(self.dir, "esm", "model_fp16.pt"))])


if __name__ == "__main__":
    unittest.main()
